"""The voice agent: a WebSocket per conversation, full duplex. The page (or any client) sends microphone audio as int16
mono at 16 kHz and gets back JSON events and int16 PCM of the reply. Speaking while the agent talks interrupts it.

    python -m voice_agent.server --profile profiles/mac.json

Protocol on /ws:
  client -> server: binary int16 16 kHz mono frames, any size; or JSON {"type": "text", "text": "..."} to type a turn,
                    {"type": "reset"} to forget the conversation.
  server -> client: JSON events {"type": "ready"|"speech_start"|"speech_end"|"transcript"|"reply"|"audio_start"|
                    "audio_end"|"interrupt"|"metrics"|"error", ...}; binary int16 PCM of the reply at audio_start's rate.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from pathlib import Path

import numpy as np
import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .engines import SIDES, SAMPLE_RATE, Brain, Clock, Said, sides, Recognizer, Vad, Voice, cancel, jpeg_to_png, sentences, unrepeated, wants_eyes
from .memory import PAST, Memory, with_context
from . import backstage, d1, lookup, recall, replies

HERE = Path(__file__).resolve().parent
LOGGED = {"transcript", "reply", "deep", "task", "confirm", "error", "metrics", "emotion", "repeat", "routed", "degenerate"}


HANDS = re.compile(r"\b(fingers?|thumbs?|hands?|holding up|gestures?|peace sign|counting|how many am i)\b")
# finger counts come from the hand tracker alone: the camera model miscounts them, and with no hand in view it rambles
FOLLOW = re.compile(r"\b(now|again|still|sure|really|this|that|these|those|it|them|how about|what about|look|check|"
                    r"right|wrong|correct|count|which|what color|what colour)\b")
FINGERS = re.compile(r"\b(fingers?|thumbs? up|peace sign|counting|how many am i|holding up|which (hand|arm)|(hand|arm) am i)\b")


EYES_SYSTEM = ("You are {name}, a friendly voice assistant looking through the user's camera. The person in the picture is "
               "the user you are talking to, never you: call them \"you\" (\"you're wearing\", \"you have\"), never \"I\". "
               "Your words are spoken aloud: answer in one to three short sentences, plain words, no lists, no brackets, "
               "no markdown.")


def tool_name(step: str) -> str:
    """A back agent's step as only its tool's name ("terminal", "web_search"): the page never shows the command, path
    or address the tool was given, so a shared screen or recording can't leak them."""

    name = re.split(r"[:(\s]", step.strip(), maxsplit=1)[0]
    return name[:40] or "tool"


def step_text(kind: str, value: str) -> str:
    """What a task card shows for a step: the tool's name, or for a permission "allowed · terminal"."""

    if kind == "permission" and ": " in value:
        option, title = value.split(": ", 1)
        return f"{'allowed' if option.startswith('allow') else 'denied'} · {tool_name(title)}"
    return tool_name(value)


def hands_note(hands: list) -> str:
    """The page's hand tracking as plain facts for the brain (landmarks are exact; a vision model miscounts fingers)."""

    if not hands:
        return "[Hand tracking from the user's camera (exact): I can't see your hands right now.]"
    parts = []
    for h in hands:
        up = h.get("up") or []
        fingers = ", ".join(up) if up else "none (a closed fist)"
        parts.append(f"your {h.get('side', 'unknown')} hand has {len(up)} finger{'s' if len(up) != 1 else ''} up: {fingers}")
    return ("[Hand tracking from the user's camera (exact, trust it over the picture), written as you should say it to "
            "them: " + "; ".join(parts) + ".]")


def stage_direction(names) -> re.Pattern:
    """A reply's leading stage direction, "(Boo says)" or "(Hermes reports):", for the given speaker names."""

    who = "|".join(re.escape(n) for n in names if n)
    return re.compile(rf"^\s*\(\s*(?:{who})\s+(?:says|said|asks|reports|replies|answers)\s*\)\s*[:,-]?\s*", re.I)


async def undirected(pieces, names):
    """The reply without a leading stage direction such as "(Boo says)": a brain that sees a labelled line in the
    history can start its own reply the same way, and the label must not be spoken or stored. The start of the reply
    waits until it is clear (an open "(" waits for its ")", at most 48 characters); the rest goes straight through."""

    pattern, head, cut = stage_direction(names), "", False  # cut: a direction was dropped, the reply starts next
    async with aclosing(pieces) as source:
        async for piece in source:
            if head is None:
                if cut:
                    piece = piece.lstrip()
                    cut = not piece
                if piece:
                    yield piece
                continue
            head += piece
            start = head.lstrip()
            if not start or (start.startswith("(") and ")" not in start and len(start) < 48):
                continue
            rest = pattern.sub("", head, count=1)
            cut, head = rest != head and not rest.strip(), None
            if rest.strip():
                yield rest
    if head:
        yield pattern.sub("", head, count=1)


def log(event) -> None:
    """One line per turn event in the server's output (agent.log): what was heard, said, handed over and how fast."""

    if isinstance(event, dict) and event.get("type") in LOGGED and not (event["type"] == "task" and event.get("state") == "step"):
        print(time.strftime("%H:%M:%S"), json.dumps({k: (v[:200] if isinstance(v, str) else v) for k, v in event.items()}), flush=True)


ENGINE_LABELS = {"full": "Full (Drama 3)", "lite": "Lite (Kokoro)"}
TIME_ASKED = re.compile(r"\b(time|clock|o'clock|hour|minutes?|late|early|tonight|morning|afternoon|evening|noon|midnight|now)\b", re.I)
MOODS = {"neutral", "happy", "excited", "curious", "thinking", "surprised", "concerned", "sad", "playful"}


class Engines:
    def __init__(self, profile: dict) -> None:
        self.profile = profile
        self.asr = Recognizer(profile["asr"])
        self.brain = Brain(profile["brain"])
        # the voice engine: Drama 3 ("full") by default; a profile can name Kokoro-82M there instead, or add it as
        # voice_lite, and then the page offers Full and Lite and each conversation picks one
        self.voices = {"full": Voice(profile["voice"])}
        if profile.get("voice_lite"):
            self.voices["lite"] = Voice(profile["voice_lite"])
        self.voice = self.voices["full"]
        # a brain that can see (Bolt with its vision tower, on CUDA): turns about the camera go to it with the picture
        self.eyes = Brain({**profile["brain"], **profile["eyes"]}) if profile.get("eyes") else None
        # Eyes of Kai: captions every camera frame into a timestamped log the brain can look back through
        self.memory = Memory(profile["memory"]) if profile.get("memory") else None
        # memory search by meaning (optional): captions and conversation lines embedded by an /v1/embeddings server
        self.recall = recall.Recall(profile["memory_search"]) if profile.get("memory_search") else None
        if self.memory:
            self.memory.recall = self.recall
        # d1-2B (optional): routes a turn the patterns below cannot place, and reads a tool-permission reply they miss
        self.d1 = d1.D1(profile["d1"]) if profile.get("d1", {}).get("url") else None
        # the back agents a turn is handed to when the front brain tags it [deep]: a direct model (Redsnow 5.3) or an
        # ACP agent with tools and skills (Hermes, Claude Code, Codex); the page picks which
        self.agents = backstage.load(profile)
        # live facts (president, weather, news) looked up here and answered by the front brain
        self.lookup = httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2)) if profile.get("lookup", True) else None
        self.voice_ids: dict[str, set[str]] = {}  # each voice engine's named voices, read when a page connects
        self.vad_path = profile.get("vad", {}).get("model", str(HERE.parent / "silero_vad.onnx"))
        self.asr_pool = ThreadPoolExecutor(1, thread_name_prefix="asr")  # one recognizer, one thread (MLX, CUDA)


class Conversation:
    """One WebSocket: listens all the time, answers, and stops answering when the person speaks over it."""

    def __init__(self, ws: WebSocket, engines: Engines) -> None:
        self.ws, self.e = ws, engines
        turn = engines.profile.get("turn", {})
        self.start_p, self.end_p = turn.get("speech_threshold", 0.5), turn.get("silence_threshold", 0.35)
        self.end_windows = max(1, int(turn.get("end_silence_ms", 400) / 32))
        # after this much silence the turn starts quietly (recognition, brain, first audio) and is let out at
        # end_silence_ms, or dropped if the person goes on talking; 0 turns it off
        self.eager_windows = int(turn.get("eager_silence_ms", 0) / 32)
        self.start_windows = max(1, int(turn.get("start_speech_ms", 64) / 32))
        self.barge_windows = max(1, int(turn.get("barge_in_ms", 192) / 32))
        self.preroll = int(turn.get("preroll_ms", 256) / 32)
        self.vad = Vad(engines.vad_path)
        self.pending = np.zeros(0, dtype=np.float32)
        self.recent: list[np.ndarray] = []  # windows before speech starts (pre-roll)
        self.speech: list[np.ndarray] = []
        self.in_speech, self.voiced, self.silent = False, 0, 0
        self.reply: asyncio.Task | None = None
        self.speaking = False
        self.history = [{"role": "system", "content": engines.profile["brain"].get("system", "You are a helpful voice assistant.")}]
        self.send_lock = asyncio.Lock()
        self.warming: asyncio.Event | None = None  # set = stop warming
        self.audio_at = 0.0  # when the microphone last sent audio
        self.closed = False  # the page has gone
        self.held: list | None = None  # a quiet turn's events, kept until it is let out
        self.frame: str | None = None  # the camera's latest picture (base64 JPEG), when the page sends one
        self.frame_at = 0.0
        self.visual: tuple | None = None  # the last camera turn: (kind: eyes, hands or sides, time, its question)
        self.hands: list = []  # the page's hand tracking: [{side, up: [finger names], count}], exact where a model guesses
        self.hands_at = 0.0
        self.heard = {"yes": True}
        self.agent = engines.profile.get("agent") or next(iter(engines.agents), None)
        self.tasks: set[asyncio.Task] = set()  # back agents at work after their turn ended
        self.asking: dict | None = None  # a back agent's tool waiting for the person's "yes, go ahead"
        self.engine = "full"  # which voice engine speaks (the page can switch to "lite" when the profile has both)
        self.voice_name = engines.profile["voice"].get("voice")  # the page can pick another
        self.name = engines.profile.get("assistant_name", "Boo")  # what the assistant is called; the page can rename it

    @property
    def voice(self) -> Voice:
        return self.e.voices[self.engine]

    async def send(self, event: dict | bytes) -> None:
        if isinstance(event, dict) and event.get("type") in ("task", "confirm"):
            log(event)
        async with self.send_lock:
            if isinstance(event, bytes):
                await self.ws.send_bytes(event)
            else:
                await self.ws.send_text(json.dumps(event))

    async def emit(self, event: dict | bytes) -> None:
        """What a turn says: held while the turn is still quiet, sent otherwise."""

        if self.held is not None:
            self.held.append(event)
            return
        if isinstance(event, dict) and event.get("type") == "audio_start":
            self.speaking = True
        log(event)
        await self.send(event)

    async def let_out(self) -> None:
        self.heard["yes"] = True
        while self.held:  # the turn may add more while these go out; it sends directly once the list is gone
            event = self.held.pop(0)
            if isinstance(event, dict) and event.get("type") == "audio_start":
                self.speaking = True
            await self.send(event)
        self.held = None
        if self.reply is None or self.reply.done():
            # the quiet turn finished before the person's silence was confirmed (a fast short reply): its finally
            # already cleared speaking, so the audio_start just let out must not leave it set, or the session goes deaf
            self.speaking = False

    # ----- listening -----

    async def audio(self, data: bytes) -> None:
        self.audio_at = time.monotonic()
        self.pending = np.concatenate([self.pending, np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768])
        while len(self.pending) >= Vad.WINDOW:
            window, self.pending = self.pending[: Vad.WINDOW], self.pending[Vad.WINDOW :]
            await self.window(window)

    async def window(self, window: np.ndarray) -> None:
        p = self.vad(window)
        if not self.in_speech:
            self.recent = (self.recent + [window])[-max(self.preroll, self.start_windows) :]
            self.voiced = self.voiced + 1 if p >= self.start_p else 0
            if self.speaking and self.voiced >= self.barge_windows:
                await self.interrupt("barge-in")
            if self.voiced >= self.start_windows and not self.speaking:
                self.in_speech, self.silent, self.speech = True, 0, list(self.recent)
                await self.send({"type": "speech_start"})
                self.start_warming()
            return
        self.speech.append(window)
        self.silent = self.silent + 1 if p < self.end_p else 0
        if self.held is not None and self.silent == 0:  # still talking: drop the quiet turn
            await self.interrupt(None)
        if self.eager_windows and self.silent == self.eager_windows and self.held is None:
            self.stop_warming()
            await self.start_turn(audio=self.utterance(), quiet=True)
        if self.silent >= self.end_windows:
            self.in_speech, self.voiced = False, 0
            audio = self.utterance()
            self.speech, self.recent = [], []
            self.stop_warming()
            await self.send({"type": "speech_end", "seconds": round(len(audio) / SAMPLE_RATE, 2)})
            if self.held is not None:
                await self.let_out()
            else:
                await self.start_turn(audio=audio)

    def utterance(self) -> np.ndarray:
        return np.concatenate(self.speech[: len(self.speech) - self.silent + 3])  # keep ~100 ms of the tail

    def stop_warming(self) -> None:
        if self.warming:
            self.warming.set()

    def start_warming(self) -> None:
        """While the person speaks, keep both engines from going idle: a server that has been quiet for a second or
        more is slow to start (on an M5 Ultra, Bolt's first token 355 ms instead of 105, Drama 3's first audio 470 ms
        instead of 360). Bolt gets a one-token request with the conversation so far every keep_warm_s; Drama 3 speaks
        a short throwaway line again and again. Both stop when the person stops, letting the request in flight finish:
        cancelling a Drama 3 stream midway slows Bolt's next first token to ~750 ms."""

        warm = self.e.profile.get("turn", {}).get("keep_warm", {"brain": True, "voice": True})
        if not warm or (self.warming and not self.warming.is_set()):
            return
        period = self.e.profile.get("turn", {}).get("keep_warm_s", 0.7)
        stop = self.warming = asyncio.Event()
        began = time.monotonic()

        def going() -> bool:
            # only while this page is connected, its microphone still sends, and for at most a minute: a page that
            # closed or went quiet mid-speech never ends the utterance, and its warming must not run on for hours
            # (it held one of Drama 3's two speech slots for good, so replies got 503s)
            now = time.monotonic()
            if stop.is_set() or self.closed or now - self.audio_at > 2 or now - began > 60:
                stop.set()
                return False
            return True

        async def brain():
            while going():
                # the system text as the reply's request has it (date, name): the prompt cache then holds all of it
                await self.e.brain.warm(self.messages() + [{"role": "user", "content": "…"}])
                try:
                    await asyncio.wait_for(stop.wait(), period)
                except asyncio.TimeoutError:
                    pass

        speaker, name = self.voice, self.voice_name

        async def voice():
            while going():
                # one throwaway line at a time across every page, and none while anything is being said
                if not await speaker.warm(name):
                    try:
                        await asyncio.wait_for(stop.wait(), 0.1)
                    except asyncio.TimeoutError:
                        pass

        # an engine that answers at once (Kokoro: "Mm." in ~20 ms) needs no warming and would only spin
        for job, on in ((brain, warm.get("brain")), (voice, warm.get("voice") and speaker.keep_warm)):
            if on:
                asyncio.create_task(job())

    # ----- answering -----

    async def start_turn(self, audio: np.ndarray | None = None, text: str | None = None, quiet: bool = False) -> None:
        self.stop_warming()
        await self.interrupt("new turn")
        self.held = [] if quiet else None
        self.reply = asyncio.create_task(self.turn(audio, text))

    async def interrupt(self, why: str | None) -> None:
        """Stops the turn under way; why=None for a quiet turn nobody heard."""

        quiet = self.held is not None
        self.held = None
        if self.reply and not self.reply.done():
            await cancel(self.reply)
            self.speaking = False
            if not quiet and why:
                await self.send({"type": "interrupt", "why": why})

    def messages(self, system: str | None = None, last: str | None = None) -> list[dict]:
        """The conversation for a brain: its system prompt carries the local date, and a newest user message that asks
        about the time carries the clock time (a model only knows its training date); ``last`` replaces the newest user
        message (the words with what the camera saw around them)."""

        # the system prompt carries the date only and the clock time goes with the newest user message when it asks
        # about the time: a server's prompt cache keeps a conversation only up to the first token that changed, and a
        # clock in the system prompt changed every minute (on a DGX Spark, Qwen3.8-27B's first token after a new
        # minute ~1.0 s instead of ~0.13 s, and longer as the conversation grows)
        now = datetime.datetime.now().astimezone()
        when = f"Today is {now:%A %-d %B %Y}."
        rest = self.history[1:]
        if rest and rest[-1]["role"] == "user":
            words = last if last else rest[-1]["content"]
            if isinstance(words, str) and TIME_ASKED.search(words):
                words = f"{words} (It is now {now:%-I:%M %p} {now.tzname()}.)"
            rest = rest[:-1] + [{"role": "user", "content": words}]
        # the name goes last, after the instructions, so a rename beats what the history says ("I am Boo")
        named = (f"Your name is {self.name}. If you were called something else earlier in this conversation, that has "
                 f"changed: you are {self.name} now, and you always say so when asked.") if self.name else ""
        return [{"role": "system", "content": f"{system or self.history[0]['content']} {when} {named}".strip()}] + rest

    def start_task(self, agent, text: str, last: str | None) -> None:
        # only the user's own words (with what the camera or hand tracker measured, when the turn used them): the front
        # brain's lines are guesses and stalls that the agent would copy; its own session remembers earlier tasks
        prompt = (f"{last or text}\n\n[Do it now with your tools and reply with the answer itself, never that you "
                  f"will check: nothing after this reply reaches the user.]")
        job = asyncio.create_task(self.backstage(agent, prompt))
        self.tasks.add(job)
        job.add_done_callback(self.tasks.discard)

    async def backstage(self, agent, prompt: str) -> None:
        """A back agent's task: its steps go to the page as they happen, its answer is spoken once nobody is talking."""

        task = f"t{id(prompt) % 100000}"
        await self.send({"type": "task", "id": task, "agent": agent.label, "state": "started"})

        def progress(kind: str, value: str) -> None:
            asyncio.create_task(self.send({"type": "task", "id": task, "agent": agent.label, "state": "step",
                                           "kind": kind, "text": step_text(kind, value)}))

        async def confirm(title: str, kind: str) -> str:
            """Asks the person, out loud and on the page, whether the agent may use a tool; 90 s without an answer
            is a no."""

            ask = self.asking = {"id": f"c{id(title) % 100000}", "future": asyncio.get_running_loop().create_future()}
            words = f"{agent.label} wants to {backstage.plain_tool(title, kind)}"
            await self.send({"type": "confirm", "id": ask["id"], "task": task, "agent": agent.label, "text": words,
                             "detail": tool_name(title)})
            await self.when_quiet(lambda: self.say(f"{words}. Should I go ahead?", agent.label, "curious", "asks"))
            try:
                choice = await asyncio.wait_for(ask["future"], 90)
            except asyncio.TimeoutError:
                choice = "deny"
            finally:
                if self.asking is ask:
                    self.asking = None
            await self.send({"type": "confirm", "id": ask["id"], "task": task, "answer": choice})
            return choice

        started = time.monotonic()
        try:
            answer = await (agent.run(prompt, progress, confirm) if agent.kind == "acp" else agent.run(prompt, progress))
        except Exception as error:
            await self.send({"type": "task", "id": task, "agent": agent.label, "state": "error", "text": str(error)[:300]})
            answer = f"Sorry, {agent.label} ran into a problem: {str(error)[:120]}"
        else:
            await self.send({"type": "task", "id": task, "agent": agent.label, "state": "done", "text": answer,
                             "seconds": round(time.monotonic() - started, 1)})
        if answer:
            await self.when_quiet(lambda: self.say(answer, agent.label))

    async def when_quiet(self, speech) -> None:
        """Starts ``speech()`` as the next turn once nobody is talking (never over the person or another reply)."""

        while self.in_speech or self.speaking or self.held is not None or (self.reply and not self.reply.done()):
            await asyncio.sleep(0.25)
        self.reply = asyncio.create_task(speech())

    async def say(self, text: str, source: str, mood: str = "happy", verb: str = "reports", remember: bool = True) -> None:
        """Speaks a back agent's words (or the assistant's own fixed line) as their own turn, interruptible like any
        other. A back agent's words go in the history as "(Hermes reports) ..."; the assistant's own lines go in
        plain, never as "(Boo says) ...", which the brain copied to the start of its next replies; and a line that
        is no part of the conversation (the sample after a voice switch) is not kept at all (``remember=False``)."""

        spoken: list[str] = []
        finished = False
        self.heard = {"yes": True}
        speaker = self.voice  # a switch of engine mid-reply applies to the next reply (one sample rate per reply)
        try:
            await self.emit({"type": "emotion", "emotion": mood})
            started = False
            async def pieces():
                yield text
            async for sentence in sentences(pieces(), **self.e.profile["voice"].get("chunks", {})):
                await self.emit({"type": "reply", "text": sentence, "source": source})
                async with aclosing(speaker.stream(sentence, self.voice_name)) as audio:
                    async for pcm in audio:
                        if not started:
                            started = True
                            await self.emit({"type": "audio_start", "sample_rate": speaker.sample_rate})
                        await self.emit(pcm)
                spoken.append(sentence)
            finished = True
            await self.emit({"type": "audio_end"})
        finally:
            self.speaking = False
            if spoken and remember:
                label = "" if source == self.name else f"({source} {verb}) "
                said = " ".join(spoken) if finished else replies.cut_off(" ".join(spoken))
                self.history.append({"role": "assistant", "content": label + said})
                if self.e.recall:
                    self.e.recall.add(f"{source} said", " ".join(spoken))

    async def turn(self, audio: np.ndarray | None, text: str | None) -> None:
        clock = Clock()
        speaker = self.voice  # a switch of engine mid-reply applies to the next reply (one sample rate per reply)
        spoken: list[str] = []
        finished = False  # the whole reply was spoken (not stopped by barge-in, a new turn or an engine error)
        written: list[str] = []  # the front brain's reply as far as it was written, read ahead of the voice
        asked = False
        tag = [""]  # the mood tag, kept in the history so the brain keeps tagging
        jobs: list[asyncio.Task] = []  # the back brain, stopped with the turn
        heard = self.heard = {"yes": self.held is None}  # let_out flips it for a quiet turn
        try:
            if audio is not None:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(self.e.asr_pool, self.e.asr, audio)
                clock.mark("asr_ms")
            await self.emit({"type": "transcript", "text": text})
            if not text:
                return
            ask = self.asking
            choice = backstage.answer(text) if ask and not ask["future"].done() else None
            if ask and not choice and not ask["future"].done() and self.e.d1:  # words the patterns miss: d1 reads them
                choice = await self.e.d1.permission(text)
                clock.mark("d1_ms")
            if ask and not ask["future"].done() and choice:
                while not heard["yes"]:  # a quiet turn answers only once it is let out (the person really stopped)
                    await asyncio.sleep(0.05)
                if not ask["future"].done():
                    ask["future"].set_result(choice)
                said = {"allow_once": "Okay, going ahead.", "allow_session": "Okay, I won't ask again for this task.",
                        "deny": "Okay, I told it not to."}[choice]
                await self.say(said, self.name, "happy" if choice != "deny" else "neutral", "says")
                return
            self.history.append({"role": "user", "content": text})
            asked = True


            last = None  # with the camera on: what it sees now and, for questions about the past, what it saw then
            low = text.lower()
            camera_live = bool(self.frame) and time.monotonic() - self.frame_at < 6
            # a short follow-up to a camera turn ("how about now?", "are you sure?", "look again") looks again at the
            # live frame; without this the front brain answers from the conversation, about a picture seconds old
            recent = self.visual if self.visual and time.monotonic() - self.visual[1] < 45 else None
            follow = bool(camera_live and recent and len(text.split()) <= 12 and FOLLOW.search(low)
                          and not backstage.LIVE.search(low))
            hands_q = bool(HANDS.search(low)) or (follow and recent[0] == "hands")
            fingers_q = bool(FINGERS.search(low)) or (follow and recent[0] == "hands")
            eyes_q = wants_eyes(text) or (follow and recent[0] in ("eyes", "sides"))
            agent = self.e.agents.get(self.agent)
            routed = None  # where d1 sent a turn the patterns could not place
            if self.e.d1 and not (hands_q or eyes_q or SIDES.search(text) or backstage.LIVE.search(low) or follow
                                  or (self.e.memory and PAST.search(text)) or (self.e.recall and recall.asks(text)) or (agent and backstage.asked_for(text, agent))):
                offered = {"camera", "side", "fingers"} if camera_live and self.e.eyes else set()
                offered |= {"memory"} if self.e.memory or self.e.recall else set()
                offered |= {"fact"} if self.e.lookup else set()
                offered |= {"task"} if agent else set()
                routed = await self.e.d1.route(text, offered | {"chat"})
                clock.mark("d1_ms")
                clock.marks["d1_call_ms"] = round(self.e.d1.last_ms, 1) if self.e.d1.last_ms is not None else -1  # -1: gave up
                if routed:
                    await self.emit({"type": "routed", "route": routed, "ms": self.e.d1.last_ms})
                hands_q = fingers_q = hands_q or routed == "fingers"
                eyes_q = eyes_q or routed == "camera"
            if self.e.memory or self.e.recall:  # the log answers about the past even with the camera off now
                # the caption log's time window and the search by meaning side by side; the search waits at most
                # its timeout_ms, and either may come back empty
                async def nothing():
                    return None
                earlier, hits = await asyncio.gather(
                    self.e.memory.past(text, force=routed == "memory") if self.e.memory else nothing(),
                    self.e.recall.search(text) if self.e.recall and (recall.asks(text) or routed == "memory") else nothing())
                recalled = recall.lines(hits) if hits else None
                seen = self.e.memory.now() if self.e.memory and camera_live and (earlier or recalled or eyes_q) else None
                if seen or earlier or recalled:
                    last = with_context(text, seen, earlier, recalled=recalled)
                    clock.mark("remembered_ms" if earlier or recalled else "seen_ms")
                    if earlier or recalled:
                        await self.emit({"type": "remembering"})

            if camera_live and hands_q:  # fingers: say exactly what the hand tracker measured, now
                asked = text if not (follow and recent[0] == "hands") else \
                    f"{recent[2]} (Asked again now: \"{text}\"; answer only from the hand tracking above.)"
                last = f"{hands_note(self.hands)}\n{asked if not last else last}"
                clock.mark("hands_ms")
                self.visual = ("hands", time.monotonic(), text if not follow else recent[2])

            brain, picture = self.e.brain, None
            if self.e.eyes and camera_live and (SIDES.search(text) or routed == "side") and not fingers_q:
                # left or right: the picture side from the person's own (mirrored) view, said as theirs, word for word
                await self.emit({"type": "looking"})
                mirrored = await asyncio.to_thread(jpeg_to_png, self.frame, 640, True)
                where = await sides(self.e.eyes, mirrored, text)
                clock.mark("looked_ms")
                if where:
                    brain = Said(where)
                    self.visual = ("sides", time.monotonic(), text)
            if isinstance(brain, Said):
                pass
            elif self.e.eyes and camera_live and eyes_q and not (last and "[Earlier" in last) and not fingers_q:
                if self.e.memory:
                    self.e.memory.hold()  # the camera model answers the question first
                brain, picture = self.e.eyes, await asyncio.to_thread(jpeg_to_png, self.frame)
                # it has the picture itself: the memory's last caption may be of an earlier frame, so only the hand facts
                asked = text
                if follow and not wants_eyes(text):  # "how about now?": the earlier camera question, asked of this frame
                    asked = f"{recent[2]} (Look at this new picture; the user now says: \"{text}\")"
                last = f"{hands_note(self.hands)}\n{asked}" if hands_q else (asked if asked != text else None)
                self.visual = ("eyes", time.monotonic(), recent[2] if follow and not wants_eyes(text) else text)
                await self.emit({"type": "looking"})
                clock.mark("looked_ms")

            deep: list[asyncio.Queue] = []  # a direct back agent's reply, when the front brain hands the turn over
            handed = [False]

            async def think_deep(queue: asyncio.Queue) -> None:
                try:
                    async for piece in agent.stream(self.messages(agent.system, last)):
                        await queue.put(piece)
                finally:
                    await queue.put(None)

            async def hand_over() -> None:
                handed[0] = True
                if agent.kind == "direct":  # fast: its answer follows the front brain's line in this turn
                    deep.append(asyncio.Queue())
                    jobs.append(asyncio.create_task(think_deep(deep[0])))
                else:  # tools take a while: it works in the background and reports when done
                    self.start_task(agent, text, last)
                await self.emit({"type": "deep", "agent": agent.name, "label": agent.label, "kind": agent.kind})
                clock.mark("deep_ms")

            # the camera model gets its own short instruction: Boo's mood tags are for the front brain, and a camera
            # model copies their brackets around a whole answer
            note = EYES_SYSTEM.format(name=self.name) if picture is not None else None
            looked_up = None
            if self.e.lookup and (backstage.LIVE.search(text.lower()) or routed == "fact") \
                    and not (agent and backstage.asked_for(text, agent, live=False)):
                # a live fact: search now and let the front brain answer in this turn (seconds, not a back agent's half minute)
                await self.emit({"type": "emotion", "emotion": "thinking"})
                looked_up = await lookup.look_up(text, self.e.lookup)
                clock.mark("lookup_ms")
                if looked_up:
                    last, brain, picture = looked_up, self.e.brain, None
            if looked_up:
                pass
            elif agent and (backstage.asked_for(text, agent) or routed == "task"):  # "ask Hermes ...", live facts, d1's tasks: at once
                await hand_over()
                if agent.kind == "acp":  # a fixed short line: the front brain would guess the answer from memory
                    await self.emit({"type": "emotion", "emotion": "thinking"})
                    brain = backstage.Ack()

            spoken_any = [False, False]
            said_text = [""]

            async def answer():
                """The brain's reply as it streams; the front brain's is guarded (replies.guarded): a reply that is
                only tags, or two words that only begin an earlier reply, is asked for once more with a nudge, and
                never spoken as silence."""

                asked_of = self.messages(note, last)
                if brain is not self.e.brain or picture is not None:
                    async for piece in brain.stream(asked_of, picture):
                        yield piece
                    return
                earlier = [m["content"] for m in self.history if m["role"] == "assistant" and isinstance(m["content"], str)]
                nudged = [{**asked_of[0], "content": f"{asked_of[0]['content']} {replies.RETRY_NOTE}"}] + asked_of[1:]

                def first() -> None:
                    clock.mark("first_token_ms")  # the brain's own first token, not the end of the hold

                def retried(text: str) -> None:
                    log({"type": "degenerate", "text": text})
                    clock.mark("retry_ms")

                async for piece in replies.guarded(brain.stream(asked_of), earlier, lambda: brain.stream(nudged),
                                                   lambda: handed[0], first, retried, written):
                    yield piece

            async def pieces():
                # tags such as "[happy]" and "[deep]" go to the face and the back agent, never to the voice; a model
                # may put one mid-reply and a token may hold half of one, so text after an open "[" waits for its "]"
                carry = ""
                async for piece in answer():
                    clock.mark("first_token_ms")
                    carry += piece
                    if "[" in carry[carry.rfind("]") + 1:] and len(carry) - carry.rfind("[") < 24:
                        continue
                    for name in " ".join(re.findall(r"\[(\w+(?: \w+)*)\]", carry)).split():  # "[deep happy]" holds two
                        name = name.lower()
                        if name == "deep":
                            if agent and not handed[0] and picture is None:  # the camera model's answer is the answer
                                await hand_over()
                        elif name in MOODS:
                            tag[0] = f"[{name}] "
                            await self.emit({"type": "emotion", "emotion": name})
                    piece, carry = re.sub(r"\s*\[\w+(?: \w+)*\]", "", carry), ""
                    said_text[0] += piece
                    if piece.strip() or (piece and spoken_any[0]):
                        spoken_any[0] = True
                        yield piece.lstrip() if not spoken_any[1] else piece
                        spoken_any[1] = True
                if carry.strip():
                    said_text[0] += re.sub(r"\s*\[\w+(?: \w+)*\]", "", carry)
                    yield re.sub(r"\s*\[\w+(?: \w+)*\]", "", carry)
                if agent and not handed[0] and picture is None and backstage.promised(said_text[0]):  # "I'll check that" with no tag
                    await hand_over()
                if deep:  # the back brain's answer follows the front brain's short line
                    yield "\n"
                    while (piece := await deep[0].get()) is not None:
                        clock.mark("deep_first_token_ms")
                        yield piece


            if self.e.profile["voice"].get("after_brain"):
                # one GPU for both (a DGX Spark): the voice starts once the brain's short reply is written, instead
                # of sharing the GPU with it
                whole = [piece async for piece in pieces()]
                clock.mark("brain_done_ms")

                async def pieces():
                    for piece in whole:
                        yield piece

            started = False
            repeat: list[str] = []  # the line the brain began to say a second time, where its reply was ended
            names = [self.name] + [a.label for a in self.e.agents.values()]
            async for sentence in sentences(unrepeated(undirected(pieces(), names), repeat),
                                            **self.e.profile["voice"].get("chunks", {})):
                clock.mark("first_sentence_ms")
                await self.emit({"type": "reply", "text": sentence})
                # closed at once when the turn is stopped, so the engine's speech slot is let go now, not at GC
                async with aclosing(speaker.stream(sentence, self.voice_name)) as audio:
                    async for pcm in audio:
                        if not started:
                            started = True
                            clock.mark("first_audio_ms")
                            await self.emit({"type": "audio_start", "sample_rate": speaker.sample_rate})
                        await self.emit(pcm)
                spoken.append(sentence)
            clock.mark("done_ms")
            finished = True
            if repeat:
                log({"type": "repeat", "text": repeat[0]})
            await self.emit({"type": "audio_end"})
            await self.emit({"type": "metrics", **clock.marks})
        except asyncio.CancelledError:
            raise
        except Exception as error:  # an engine down: say so, keep listening
            await self.emit({"type": "error", "error": f"{type(error).__name__}: {error}"})
        finally:
            self.speaking = False
            for job in jobs:
                job.cancel()
            if asked and self.history[-1]["role"] == "user":
                if heard["yes"] and spoken:  # what was actually said goes in, even when cut off
                    # a cut reply is stored whole with the cut marked (replies.cut_reply): stored bare, its fragment
                    # reads as a finished reply and a brain that copies its earlier turns ends its next replies the
                    # same way ("I am on" -> "I" -> nothing)
                    said = " ".join(spoken) if finished else replies.cut_reply(" ".join(spoken), "".join(written))
                    self.history.append({"role": "assistant", "content": tag[0] + said})
                    if self.e.recall:
                        self.e.recall.add("you said", self.history[-2]["content"])
                        self.e.recall.add(f"{self.name} said", " ".join(spoken))
                else:
                    # a quiet turn nobody heard, or a reply stopped before its first words: as if it never happened (a
                    # placeholder such as "(interrupted)" in the history is copied by the brain and spoken as a reply)
                    self.history.pop()


def build(profile: dict) -> FastAPI:
    app = FastAPI(title="keys voice agent")
    engines = Engines(profile)

    @app.get("/")
    def page():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/voices/{name}.wav")
    def voice_sample(name: str):
        for spec in (profile["voice"], profile.get("voice_lite") or {}):
            folder = spec.get("samples")
            path = Path(folder).expanduser() / f"{name}.wav" if folder and re.fullmatch(r"[\w-]+", name) else None
            if path and path.is_file():
                return FileResponse(path)
        return JSONResponse({"error": "no sample"}, status_code=404)

    @app.get("/health")
    def health():
        return JSONResponse({"ok": True, "profile": profile.get("name"), "asr": profile["asr"]["model"],
                             "brain": profile["brain"]["model"], "voice": profile["voice"]["model"],
                             "voice_lite": (profile.get("voice_lite") or {}).get("model"),
                             "eyes": (profile.get("eyes") or {}).get("model"),
                             "memory_search": (profile.get("memory_search") or {}).get("model"),
                             "d1": "d1-2B (Liquid AI)" if profile.get("d1", {}).get("url") else None})

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        talk = Conversation(socket, engines)
        await talk.send({"type": "ready", "input_rate": SAMPLE_RATE})

        async def offer_voices() -> None:
            """The voice picker's list, for the engine this conversation speaks with."""
            voices = await talk.voice.voices()
            engines.voice_ids[talk.engine] = {v["id"] for v in voices}
            if talk.voice_name and talk.voice_name not in engines.voice_ids[talk.engine]:  # not on this engine: its default
                talk.voice_name = None
            if voices:
                await talk.send({"type": "voices", "engine": talk.engine, "current": talk.voice_name, "list": voices})

        await offer_voices()
        if len(engines.voices) > 1:  # both engines configured: offer the switch, listing only the ones that answer
            up = [name for name, v in engines.voices.items() if await v.voices()]
            if len(up) > 1:
                await talk.send({"type": "engines", "current": talk.engine,
                                 "list": [{"id": n, "label": ENGINE_LABELS.get(n, n)} for n in up]})
        await talk.send({"type": "agents", "current": talk.agent,
                         "list": [{"id": a.name, "label": a.label, "kind": a.kind} for a in engines.agents.values()]})
        try:
            while True:
                message = await socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await talk.audio(message["bytes"])
                elif message.get("text"):
                    event = json.loads(message["text"])
                    if event.get("type") == "text":
                        await talk.start_turn(text=event["text"])
                    elif event.get("type") == "reset":
                        await talk.interrupt("reset")
                        talk.history = talk.history[:1]
                    elif event.get("type") == "confirm" and talk.asking and event.get("id") == talk.asking["id"]:
                        if not talk.asking["future"].done() and event.get("answer") in ("allow_once", "allow_session", "deny"):
                            talk.asking["future"].set_result(event["answer"])
                    elif event.get("type") == "engine" and event.get("name") in engines.voices and event["name"] != talk.engine:
                        talk.engine = event["name"]
                        spec = engines.profile["voice_lite" if talk.engine == "lite" else "voice"]
                        talk.voice_name = spec.get("voice")
                        await offer_voices()
                        if not event.get("quiet"):
                            asyncio.create_task(talk.when_quiet(lambda: talk.say("Hi, this is how I sound now.", talk.name, "happy", "says", remember=False)))
                    elif event.get("type") == "voice" and event.get("name") in engines.voice_ids.get(talk.engine, ()):
                        changed = event["name"] != talk.voice_name
                        talk.voice_name = event["name"]
                        if changed and not event.get("quiet"):
                            asyncio.create_task(talk.when_quiet(lambda: talk.say("Hi, this is how I sound now.", talk.name, "happy", "says", remember=False)))
                    elif event.get("type") == "agent" and event.get("name") in engines.agents:
                        talk.agent = event["name"]
                    elif event.get("type") == "stop":
                        await talk.interrupt("stop")
                    elif event.get("type") == "frame":
                        talk.frame, talk.frame_at = event.get("jpeg"), time.monotonic()
                        if engines.memory and talk.frame:
                            engines.memory.look(talk.frame)
                    elif event.get("type") == "scan" and engines.memory:  # how often to caption the camera, from the page
                        engines.memory.every(event.get("seconds", 3))
                    elif event.get("type") == "keep" and engines.memory:  # how many days the camera memory keeps
                        asyncio.create_task(engines.memory.keep(event.get("days", 2)))
                        if engines.recall:  # the search index keeps no longer than the captions
                            engines.recall.keep(event.get("days", 2))
                    elif event.get("type") == "name":  # the page renamed the assistant
                        name = re.sub(r"[^\w '-]", "", str(event.get("name", ""))).strip()[:24]
                        old, talk.name = talk.name, name or engines.profile.get("assistant_name", "Boo")
                        if old and old != talk.name:  # its earlier replies say the old name, and a model repeats them
                            for m in talk.history[1:]:
                                if m["role"] == "assistant" and isinstance(m["content"], str):
                                    m["content"] = re.sub(rf"\b{re.escape(old)}\b", talk.name, m["content"])
                    elif event.get("type") == "hands":
                        talk.hands, talk.hands_at = event.get("hands") or [], time.monotonic()
                    elif event.get("type") == "presence":
                        talk.present = bool(event.get("face"))
        except WebSocketDisconnect:
            pass
        finally:
            talk.closed = True
            talk.stop_warming()
            await talk.interrupt("closed")
            for job in list(talk.tasks):
                job.cancel()

    return app


# the lite voice: Kokoro-82M behind voice_agent.kokoro_server on its default port; a profile's voice_lite overrides it
KOKORO = {"url": "http://127.0.0.1:8197/v1", "model": "kokoro-82m", "sample_rate": 24000, "chunks": {"first_words": 5},
          "voice": "bf_emma", "keep_warm": False}


def choose_voice(profile: dict, engine: str | None) -> dict:
    """Applies the voice switch: "full" speaks with the profile's voice (Drama 3), "lite" with Kokoro-82M instead,
    "both" keeps Drama 3 and adds Kokoro, and the page offers Full and Lite. From --voice, else the profile's
    voice_engine, else full (or both when the profile already has a voice_lite)."""

    engine = engine or profile.get("voice_engine") or ("both" if profile.get("voice_lite") else "full")
    if engine not in ("full", "lite", "both"):
        raise SystemExit(f"voice engine must be full, lite or both, not {engine!r}")
    lite = {**KOKORO, **(profile.get("voice_lite") or {})}
    profile = dict(profile)
    if engine == "full":
        profile.pop("voice_lite", None)
    elif engine == "lite":
        profile["voice"] = lite
        profile.pop("voice_lite", None)
    else:
        profile["voice_lite"] = lite
    profile["voice_engine"] = engine
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--host", default=os.environ.get("VOICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VOICE_PORT", "8095")))
    parser.add_argument("--voice", choices=["full", "lite", "both"], default=os.environ.get("VOICE_ENGINE") or None,
                        help="full: Drama 3 (default); lite: Kokoro-82M; both: Drama 3 with a Full/Lite switch in the page")
    args = parser.parse_args()
    profile = choose_voice(json.loads(Path(args.profile).read_text()), args.voice)
    uvicorn.run(build(profile), host=args.host, port=args.port, log_level="warning", ws_max_size=2**24)


if __name__ == "__main__":
    main()
