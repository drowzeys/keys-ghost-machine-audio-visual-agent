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

from .engines import SENTENCE_END, SIDES, SAMPLE_RATE, Brain, Clock, Said, sides, Recognizer, Vad, Voice, cancel, jpeg_to_png, sentences, unrepeated, wants_eyes
from .memory import PAST, Memory, with_context
from . import backstage, d1, lookup, pocket, recall, replies

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
        # Pocket (optional): one speech-to-speech model (LFM3-Omni-2B) hears each spoken turn and answers it out loud
        # in place of brain and voice; the recognizer still writes the transcript beside it. A Pocket profile may
        # leave out the brain (typed turns and camera questions need one)
        self.pocket = pocket.Pocket(profile["pocket"]) if profile.get("pocket", {}).get("url") else None
        self.brain = Brain(profile["brain"]) if profile.get("brain") or not self.pocket else None
        # the voice engine: Drama 3 ("full") by default; a profile can name Kokoro-82M there instead, or add it as
        # voice_lite, and then the page offers Full and Lite and each conversation picks one
        self.voices = {"full": Voice(profile["voice"])}
        if profile.get("voice_lite"):
            self.voices["lite"] = Voice(profile["voice_lite"])
        self.voice = self.voices["full"]
        # a brain that can see (Bolt with its vision tower, on CUDA): turns about the camera go to it with the picture
        self.eyes = Brain({**profile.get("brain", {}), **profile["eyes"]}) if profile.get("eyes") else None
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
        front = engines.profile.get("brain") or engines.profile.get("pocket") or {}
        self.history = [{"role": "system", "content": front.get("system", "You are a helpful voice assistant.")}]
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
        if self.e.pocket:
            # Pocket answers spoken turns itself and its server takes one request at a time: nothing to warm, except
            # in hybrid the brain, which answers most turns (its first token 0.2 s warm, 1-5 s on a cold prompt)
            if self.e.pocket.mode != "hybrid" or self.e.brain is None or not warm:
                return
            warm = {"brain": warm.get("brain", True), "voice": False}
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

    def start_task(self, agent, text: str, last: str | None, messages: list[dict] | None = None) -> None:
        # only the user's own words (with what the camera or hand tracker measured, when the turn used them): the front
        # brain's lines are guesses and stalls that the agent would copy; its own session remembers earlier tasks
        prompt = (f"{last or text}\n\n[Do it now with your tools and reply with the answer itself, never that you "
                  f"will check: nothing after this reply reaches the user.]")
        job = asyncio.create_task(self.backstage(agent, prompt, messages))
        self.tasks.add(job)
        job.add_done_callback(self.tasks.discard)

    async def backstage(self, agent, prompt: str, messages: list[dict] | None = None) -> None:
        """A back agent's task: its steps go to the page as they happen, its answer is spoken once nobody is talking.
        A direct agent (a chat model) answers ``messages``, the conversation (Pocket hands it over this way)."""

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
            if agent.kind == "acp":
                answer = await agent.run(prompt, progress, confirm)
            else:
                answer = backstage.speakable("".join([p async for p in agent.stream(messages or [{"role": "user", "content": prompt}])]))
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
            playback = [None, 0.0]  # when the page began playing, the seconds sent
            async for sentence in sentences(pieces(), **self.e.profile["voice"].get("chunks", {})):
                await self.emit({"type": "reply", "text": sentence, "source": source})
                async with aclosing(speaker.stream(sentence, self.voice_name)) as audio:
                    async for pcm in audio:
                        if not started:
                            started = True
                            await self.emit({"type": "audio_start", "sample_rate": speaker.sample_rate})
                            playback[0] = time.monotonic()
                        playback[1] += len(pcm) / 2 / speaker.sample_rate
                        await self.emit(pcm)
                spoken.append(sentence)
            if playback[0] is not None and self.e.pocket:  # Kokoro is done long before the page: wait while it plays
                await asyncio.sleep(max(0.0, playback[0] + playback[1] - time.monotonic()))
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

    def camera_live(self) -> bool:
        return bool(self.frame) and time.monotonic() - self.frame_at < 6

    def pocket_waits(self) -> bool:
        """Whether a Pocket turn needs its words before it goes anywhere: a back agent's tool is waiting for a yes, or
        the camera is on and the eyes (with a brain) could be the ones to answer (not when Omni sees for itself)."""

        asking = bool(self.asking and not self.asking["future"].done())
        if self.pocket_sees():  # Omni sees; with vision_mirror "sides" the words pick the frame's side
            return asking or (self.e.pocket.vision_mirror == "sides" and self.camera_live())
        return asking or bool(self.e.eyes and self.e.brain and self.camera_live())

    def camera_question(self, text: str) -> bool:
        """A turn for the eyes rather than Pocket (an Omni that is not shown the camera cannot see): the camera is on
        and the words ask about it."""

        return bool(not self.pocket_sees() and self.e.eyes and self.e.brain and self.camera_live()
                    and (wants_eyes(text) or SIDES.search(text) or HANDS.search(text.lower())))

    def pocket_sees(self) -> bool:
        """Whether Pocket's Omni is shown the camera (the profile's pocket.vision)."""

        return bool(self.e.pocket and self.e.pocket.vision)

    async def pocket_frame(self, words: asyncio.Future | None = None, view: bool | None = None) -> tuple[str | None, bool]:
        """The camera's latest frame for a Pocket turn that sees (pocket.vision, the camera on): a PNG as wide as the
        profile says, the camera's own view or, with vision_mirror, the person's (mirrored) one; with "sides", the
        mirrored one for a left-or-right question (the turn's words, in by then). Also whether it is such a question:
        that one goes without the conversation, as sides() asks the eyes (earlier answers about other pictures pull a
        small model's left or right after them)."""

        omni = self.e.pocket
        if not (self.pocket_sees() and self.camera_live()) or view is False:
            return None, False
        mirror, side = omni.vision_mirror, False
        if mirror == "sides":
            ready = words is not None and words.done() and not words.cancelled() and words.exception() is None
            said = words.result() if ready else ""
            mirror = side = bool(said and SIDES.search(said))
        try:
            return await asyncio.to_thread(jpeg_to_png, self.frame, omni.vision_width, mirror), side
        except Exception:  # a frame that does not decode: the turn goes on without it
            return None, False

    async def pocket_turn(self, audio: np.ndarray, words: asyncio.Future, clock: Clock, heard: dict,
                          told: bool = False, gate: asyncio.Future | None = None, view: bool = False) -> str | None:
        """A spoken turn answered by Pocket (LFM3-Omni-2B): its streamed speech is the voice and its streamed text the
        reply. Mood tags are read and never shown; [deep] in the reply, words that ask for the back agent, or a reply
        that only promises to do it, hand the turn over (the agent starts once the turn was heard, and its answer is
        spoken by the profile's voice). A cut reply goes in the history as replies.py stores one, cut where the speech
        had got to.

        Nothing goes out until ``gate`` (hybrid: the words chose Omni) is True and, with a brain to fall back on,
        the reply's first sentence (or six words) is written and repeats none of the last replies (pocket.repeats). A
        reply that does, or says nothing (replies.degenerate), is dropped, and this returns "fallback" (or
        "fallback-told", the transcript already shown): the brain answers the turn instead. A gate that turns False
        drops everything; the caller then stops this turn. In hybrid, the camera's frame goes only with ``view`` (the
        words asked about the camera, pocket.vision on); a speculative start goes without it."""

        omni, agent = self.e.pocket, self.e.agents.get(self.agent)
        system = self.messages(omni.system)[0]["content"]
        handed, tag = [False], [""]
        written = ""  # the reply's text so far, tags left out
        line = ""  # the sentence being written, for the page's reply events
        first_audio, sent_s, got_audio = None, 0.0, False
        finished = False
        job = [False]  # the back agent's task was started
        shown = [told]  # the transcript is on the page
        # held back until the gate opens and the reply is judged; judged at once when there is no brain to fall back on
        earlier = [m["content"] for m in self.history[1:] if m["role"] == "assistant" and isinstance(m["content"], str)][-6:]
        judged = [True]  # set once the frame is known: a camera answer may rightly repeat an earlier one
        held: list = []
        sent = [False]  # something of this reply went out

        def open_() -> bool:
            return gate is None or (gate.done() and not gate.cancelled() and gate.result() is True)

        def released() -> bool:
            return open_() and judged[0]

        async def send(event) -> None:
            nonlocal first_audio
            sent[0] = True
            if isinstance(event, bytes) and first_audio is None:
                first_audio = time.monotonic()  # the page starts playing now (held audio goes out all at once)
                clock.mark("first_audio_ms")
            await self.emit(event)

        async def out(event) -> None:
            """Sends ``event`` (None: only what was held), or holds it while the turn is not Omni's yet."""

            if gate is not None and gate.done() and not open_():
                return  # the brain has the turn
            if not released():
                if event is not None:
                    held.append(event)
                return
            while held:
                await send(held.pop(0))
            if event is not None:
                await send(event)

        async def hand_over() -> None:
            if agent and not handed[0]:
                handed[0] = True
                await out({"type": "deep", "agent": agent.name, "label": agent.label, "kind": agent.kind})
                clock.mark("deep_ms")

        async def listen() -> None:
            text = await asyncio.shield(words)  # stopping the turn stops this, never the recognizer's words
            clock.mark("asr_ms")
            if not told:
                await self.emit({"type": "transcript", "text": text})
                shown[0] = True
            if text and agent and backstage.asked_for(text, agent):  # "ask Hermes...", a live fact: to the back agent
                await hand_over()

        def judge(done: bool) -> bool:
            """Whether the reply so far may be spoken (it is judged once); False: the brain answers instead."""

            judged[0] = True
            if replies.words(written) and not pocket.repeats(written, earlier) \
                    and not (done and replies.degenerate(tag[0] + written, earlier)):
                return True
            log({"type": "degenerate", "text": written or "(nothing)", "by": omni.model})
            clock.marks["omni_dropped"] = 1
            return False

        listening = asyncio.create_task(listen())
        if gate is not None:  # what was held goes out the moment the words choose Omni, not with Omni's next event
            gate.add_done_callback(lambda _: self.later(out(None)))
        try:
            frame, side = await self.pocket_frame(words, None if gate is None else bool(view))
            if frame:
                clock.mark("frame_ms")
                clock.marks["camera"] = 1
            judged[0] = not (omni.guard and self.e.brain is not None and earlier and not frame)
            async with aclosing(omni.stream(audio, system, [] if side else self.history[1:], frame)) as events:
                carry = ""
                async for kind, value in events:
                    if kind == "text":
                        clock.mark("first_text_ms")
                        carry += value
                        if "[" in carry[carry.rfind("]") + 1:] and len(carry) - carry.rfind("[") < 24:
                            continue  # half a tag: wait for its "]"
                        for name in " ".join(re.findall(r"\[(\w+(?: \w+)*)\]", carry)).split():
                            name = name.lower()
                            if name == "deep":
                                await hand_over()
                            elif name in MOODS:
                                tag[0] = f"[{name}] "
                                await out({"type": "emotion", "emotion": name})
                        piece, carry = pocket.TAGS.sub("", carry), ""
                        if not written:
                            piece = piece.lstrip()
                        written += piece
                        line += piece
                        while end := SENTENCE_END.search(line):
                            said, line = line[: end.end()].strip(), line[end.end():]
                            if said:
                                await out({"type": "reply", "text": said})
                        if not judged[0] and (len(replies.words(written)) >= 6 or SENTENCE_END.search(written)):
                            if not judge(False):
                                return "fallback-told" if shown[0] else "fallback"
                            await out(None)  # lets out what was held (when the gate is open)
                    elif kind == "audio":
                        if not got_audio:
                            got_audio = True
                            if not tag[0]:
                                await out({"type": "emotion", "emotion": pocket.mood_of(written)})
                            await out({"type": "audio_start", "sample_rate": omni.sample_rate})
                        sent_s += len(value) / 2 / omni.sample_rate
                        await out(value)
                    elif kind == "done":
                        for key in ("first_text_ms", "first_audio_ms", "total_ms", "speech_s", "image_tokens"):
                            if key in value:
                                clock.marks[f"omni_{key}"] = value[key]
                if carry.strip():
                    rest = pocket.TAGS.sub("", carry)
                    written, line = written + rest, line + rest
            if line.strip():
                await out({"type": "reply", "text": line.strip()})
            if not judged[0] and not judge(True):
                return "fallback-told" if shown[0] else "fallback"
            if gate is not None:
                await gate
                if not open_():
                    return None
            await out(None)
            clock.mark("done_ms")
            try:
                await listening  # the transcript (long done by now) may still hand the turn over
            except Exception:
                pass
            if agent and not handed[0] and gate is None \
                    and (backstage.promised(written.replace("\u2019", "'")) or pocket.unable(written)):
                # "On it, ..." with no tag, or "I can't create a clip": the back agent's work (not in hybrid, where
                # Omni has only small talk and the camera: "I'll keep at it, see you!" to a goodbye is no task)
                await hand_over()
            if handed[0]:
                self.pocket_job(words, heard, agent)
                job[0] = True
            if first_audio is not None:
                # Omni writes a reply several times faster than it is spoken: the turn lasts while the page plays it,
                # so talking over it still stops it (barge-in), and a cut is stored where the speech had got to
                await asyncio.sleep(max(0.0, first_audio + sent_s - time.monotonic()))
            finished = True
            await out({"type": "audio_end"})
            await out({"type": "metrics", "pocket": omni.model, "route": "omni", **clock.marks})
        except asyncio.CancelledError:
            raise
        except Exception as error:  # Omni down: the brain answers, or (no brain) say so and keep listening
            if self.e.brain is not None and not sent[0]:
                log({"type": "error", "error": f"Pocket: {type(error).__name__}: {error}", "by": omni.model})
                return "fallback-told" if shown[0] else "fallback"
            await self.emit({"type": "error", "error": f"{type(error).__name__}: {error}"})
        finally:
            self.speaking = False
            listening.cancel()
            written = written.strip()
            mine = released() and (first_audio is not None or finished)  # what went out was Omni's answer
            if not mine:
                said = ""
            elif finished:
                said = written
            elif first_audio is not None:  # cut: as much of the text as its speech had got through
                played = min(time.monotonic() - first_audio, sent_s)
                spoken = pocket.heard_part(written, played)
                said = replies.cut_reply(spoken, written) if spoken else ""
            else:
                said = ""
            if mine and handed[0] and not job[0] and heard["yes"]:  # handed over, then cut off: the task still goes ahead
                self.pocket_job(words, heard, agent)
            if (heard["yes"] or finished) and said:
                self.later(self.pocket_remember(words, heard, tag[0] + said))
        return None

    def looks_at_camera(self, text: str) -> bool:
        """Whether a turn asks about the camera while it is on: the eyes' words, left or right, hands, or the words
        they miss ("Where am I?", "What does my sign say?"), or a short follow-up to a camera turn."""

        if not self.camera_live() or not text:
            return False
        low = text.lower()
        recent = self.visual if self.visual and time.monotonic() - self.visual[1] < 45 else None
        follow = bool(recent and len(text.split()) <= 12 and FOLLOW.search(low) and not backstage.LIVE.search(low))
        return bool(wants_eyes(text) or SIDES.search(text) or HANDS.search(low) or pocket.looks(text) or follow)

    async def hybrid_turn(self, audio: np.ndarray, words: asyncio.Future, clock: Clock, heard: dict) -> str | None:
        """A spoken turn in Pocket's hybrid mode: Omni starts at once (``speculate``), its reply held, while the
        recognizer writes the words; small talk lets Omni's reply out, anything else stops it and goes to the brain.
        A camera question goes to Omni with the frame when it sees (pocket.vision; finger counts stay with the hand
        tracker), else to the brain's eyes. Returns None when Omni answered, else the words for the brain (the
        transcript already shown)."""

        omni = self.e.pocket
        gate = asyncio.get_running_loop().create_future()
        trying = asyncio.create_task(self.pocket_turn(audio, words, clock, heard, told=True, gate=gate)) \
            if omni.speculate else None
        try:
            try:
                text = await asyncio.shield(words)
            except Exception:
                text = ""
            clock.mark("asr_ms")
            await self.emit({"type": "transcript", "text": text})
            last = next((m["content"] for m in reversed(self.history[1:]) if m["role"] == "assistant"
                         and isinstance(m["content"], str)), None)
            asking = bool(self.asking and not self.asking["future"].done())
            camera = not asking and self.looks_at_camera(text)
            if not text or asking:
                route = "brain"
            elif camera:
                sees = self.pocket_sees() and not FINGERS.search(text.lower()) \
                    and not (SIDES.search(text) and self.e.pocket.vision_sides == "eyes" and self.e.eyes)
                route = "omni-eyes" if sees else "brain"
            else:
                route = "omni" if pocket.small_talk(text, last) else "brain"
            if route == "omni" and trying is not None and trying.done():  # Omni gave up already (down, or a repeat)
                route, clock.marks["fallback"] = "brain", 1
            clock.marks["route"] = route
            if camera:
                clock.marks["camera"] = 1
            log({"type": "routed", "route": route, "by": "pocket", "ms": clock.marks.get("asr_ms")})
            if route == "omni-eyes":  # a new request with the frame (the speculative one went without it)
                gate.set_result(False)
                await cancel(trying)
                self.visual = ("sides" if SIDES.search(text) else "eyes", time.monotonic(), text)  # for "how about now?"
                seen = asyncio.get_running_loop().create_future()
                seen.set_result(True)
                trying = asyncio.create_task(self.pocket_turn(audio, words, clock, heard, told=True, gate=seen, view=True))
                if not (await trying or "").startswith("fallback"):
                    return None
                clock.marks["route"], clock.marks["fallback"] = "brain", 1
                return text
            if route == "omni":
                if trying is None:
                    trying = asyncio.create_task(self.pocket_turn(audio, words, clock, heard, told=True, gate=gate))
                gate.set_result(True)
                if not (await trying or "").startswith("fallback"):
                    return None
                clock.marks["route"], clock.marks["fallback"] = "brain", 1
                return text
            gate.set_result(False)
            await cancel(trying)
            for key in [k for k in clock.marks if k.startswith("omni_") or k == "first_text_ms"]:
                del clock.marks[key]  # the speculative Omni's, not this turn's
            return text
        finally:
            if not gate.done():
                gate.set_result(False)
            if trying is not None and not trying.done():
                await cancel(trying)

    def later(self, work) -> None:
        self.tasks.add(task := asyncio.create_task(work))
        task.add_done_callback(self.tasks.discard)

    async def pocket_words(self, words: asyncio.Future, heard: dict) -> str | None:
        """A Pocket turn's words once the recognizer has them (a few ms after the reply began, so this almost never
        waits) and the turn was heard; None when it was not: a quiet turn the person talked on through (a newer turn
        took its place) leaves no trace."""

        try:
            text = await asyncio.shield(words)
        except Exception:
            return None
        for _ in range(6000):
            if heard["yes"]:
                return text or None
            if self.heard is not heard or self.closed:
                return None
            await asyncio.sleep(0.02)
        return None

    async def pocket_remember(self, words: asyncio.Future, heard: dict, said: str) -> None:
        """A heard Pocket turn into the history: the person's words and what was said back."""

        if not (text := await self.pocket_words(words, heard)):
            return
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": said}]
        if self.e.recall:
            self.e.recall.add("you said", text)
            self.e.recall.add(f"{self.name} said", replies.TAGS.sub("", said).strip())

    def pocket_job(self, words: asyncio.Future, heard: dict, agent) -> None:
        """Starts a Pocket turn's hand-over once its words are in: the back agent gets the person's own words (a
        direct one the conversation up to them), never Pocket's "On it"."""

        async def start() -> None:
            if not (text := await self.pocket_words(words, heard)):
                return
            conversation = None
            if agent.kind == "direct":  # an ACP agent keeps its own session and gets only the words (no .system)
                conversation = self.messages(agent.system)
                while len(conversation) > 1 and conversation[-1]["role"] == "assistant":
                    conversation.pop()
                if conversation[-1] != {"role": "user", "content": text}:
                    conversation.append({"role": "user", "content": text})
            self.start_task(agent, text, None, conversation)
        self.later(start())

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
        spoken_turn = None  # Pocket: the spoken turn, when its words were needed before it could go to Omni
        playback: list = [None, 0.0]  # when the page began playing the reply, and the seconds of speech sent
        ends: list[float] = []  # where each spoken sentence ends in the reply's speech (seconds)
        try:
            told = False  # the transcript is on the page already
            if audio is not None and self.e.pocket and self.e.pocket.mode == "hybrid" and self.e.brain is not None:
                # hybrid: Omni answers small talk, the brain the rest; the words (about 0.1 s) decide
                loop = asyncio.get_running_loop()
                words = asyncio.ensure_future(loop.run_in_executor(self.e.asr_pool, self.e.asr, audio))
                text = await self.hybrid_turn(audio, words, clock, heard)
                if text is None:
                    return
                told = True
            elif audio is not None and self.e.pocket:
                # the recognizer runs beside Omni and is not waited for, unless its words decide the turn: a tool
                # permission being asked, or a camera question for the eyes (only with the camera on)
                loop = asyncio.get_running_loop()
                words = asyncio.ensure_future(loop.run_in_executor(self.e.asr_pool, self.e.asr, audio))
                if not self.pocket_waits():
                    gave_up = await self.pocket_turn(audio, words, clock, heard)
                    if not gave_up:
                        return
                    text, told = await words, gave_up == "fallback-told"  # a repeat or nothing: the brain answers
                    clock.marks["route"], clock.marks["fallback"] = "brain", 1
                else:
                    text, spoken_turn = await words, (audio, words)
                clock.mark("asr_ms")
            elif audio is not None:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(self.e.asr_pool, self.e.asr, audio)
                clock.mark("asr_ms")
            if not told:
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
            if spoken_turn and not self.camera_question(text):
                return await self.pocket_turn(*spoken_turn, clock, heard, told=True)
            if self.e.brain is None:  # a Pocket profile without a brain: only spoken turns get an answer
                await self.emit({"type": "error", "error": "This profile answers spoken turns only (Pocket, no text brain)."})
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
            eyes_q = wants_eyes(text) or (follow and recent[0] in ("eyes", "sides")) \
                or bool(self.e.pocket and pocket.looks(text))  # "Where am I?": never the brain alone
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
                            playback[0] = time.monotonic()
                        playback[1] += len(pcm) / 2 / speaker.sample_rate
                        await self.emit(pcm)
                spoken.append(sentence)
                ends.append(playback[1])
            clock.mark("done_ms")
            if playback[0] is not None and self.e.pocket:
                # a fast voice (Kokoro: a reply in a fraction of its length) is done long before the page has played
                # it: the turn lasts while the page plays, so talking over it still stops it (barge-in), as Omni's do
                await asyncio.sleep(max(0.0, playback[0] + playback[1] - time.monotonic()))
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
                if not finished and spoken and playback[0] is not None and self.e.pocket:
                    # sent is not heard: only what the page had played when the person cut in
                    played = time.monotonic() - playback[0]
                    kept = [x for x, end in zip(spoken, ends) if end <= played]
                    if len(kept) < len(spoken):
                        start = ends[len(kept) - 1] if kept else 0.0
                        part = pocket.heard_part(spoken[len(kept)], played - start)
                        kept += [part] if part else []
                    spoken = kept
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
                             "brain": (profile.get("brain") or {}).get("model")
                             or (profile.get("pocket") or {}).get("model", "lfm3-omni-2b") + " (Pocket)",
                             "voice": profile["voice"]["model"],
                             "pocket": (profile.get("pocket") or {}).get("model") if profile.get("pocket", {}).get("url") else None,
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
