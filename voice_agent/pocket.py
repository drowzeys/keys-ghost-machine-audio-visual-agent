"""The Pocket tier: one small model hears the person and answers out loud, in place of recognizer -> brain -> voice.

LFM3-Omni-2B (Liquid AI) on TensorFold takes the spoken turn itself and streams back its reply as text and 24 kHz
speech together (``lfm3-omni-generate --serve``: POST /v1/audio/chat, server-sent events). The agent keeps what costs
little around it: voice activity and barge-in, the conversation (sent as text turns, ``history``), the transcript
(the profile's recognizer runs beside it, never waited for), and hand-overs to a back agent.

    "pocket": {"url": "http://127.0.0.1:8094", "model": "lfm3-omni-2b", "system": "...", "history_turns": 12,
               "extra": {"audio_temperature": 0.4, "audio_repetition_penalty": 1.2}}

``"mode": "hybrid"`` (with a brain in the profile) keeps Omni for small talk only. Omni starts on every spoken turn at
once, its reply held back, while the recognizer writes the words (about 0.1 s, about as long as Omni takes to its
first audio); the words then decide: a greeting, thanks, an acknowledgement or a short how-are-you (``small_talk``)
lets Omni's reply out, anything else (a question, a studio task, a fact, a follow-up, the camera) hangs Omni up and
goes to the brain, spoken by the profile's voice, as in a Boo without Pocket. ``"speculate": false`` starts Omni only
once the words chose it. ``"mode": "omni"`` (the default) answers every spoken turn with Omni.

In both modes Omni's reply is held until its first sentence (or six words) is written and dropped for the brain when
it repeats one of the last few replies or says nothing (``repeats``, replies.degenerate); and Omni sees fewer turns
of the conversation than the brain (``history_turns``, 4 in hybrid): a 2B model copies its own earlier lines.

With ``"vision": true`` (a server built with the vision tower) Omni also sees: while the camera is on, each spoken turn
goes with the camera's latest frame (``images``, a PNG at most ``vision_width`` wide), and the system text gets
``vision_note``, so Omni answers camera questions itself, without waiting for the transcript. The frame is the camera's
own view (``vision_mirror`` false, the default: text in it reads the right way round), the person's own mirrored view
(true: "my left hand" is then on the picture's left, as the page shows it), or, with ``"sides"``, the camera's view
except for a left-or-right question, which gets the mirrored one and goes without the conversation (as Boo's sides()
asks the eyes); ``"sides"`` waits for the transcript (about 0.1 s) while the camera is on, to know which. Without
``vision`` (the default), a camera question waits for the transcript and goes to the profile's eyes and brain. In
hybrid, a camera question (the camera on, the words about it) goes to Omni with the frame, except finger counts (the
hand tracker) and, with ``"vision_sides": "eyes"`` (the default), left-or-right questions (the eyes' sides()).
"""

from __future__ import annotations

import base64
import io
import json
import re
import wave
from typing import AsyncIterator

import httpx
import numpy as np

from .engines import SAMPLE_RATE
from . import replies

SYSTEM = ("Respond with interleaved text and audio. You are a friendly voice assistant. You are speaking out loud: "
          "answer in one to three short spoken sentences, plain words, no lists, no markdown, no emoji.")
CHARS_PER_S = 14.0  # how fast speech gets through a reply's text (as engines.unspoken): where a cut reply was cut
TAGS = re.compile(r"\s*\[\w+(?: \w+)*\]")
VISION_NOTE = ("You can see the person through their camera: the picture with their words is what the camera shows right "
               "now, the person facing it. Use it when they ask about what you see, what they wear, hold or show, or "
               "where they are; otherwise just answer them.")


def wav_bytes(audio: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    """float32 mono as a 16-bit PCM WAV file."""

    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())
    return out.getvalue()


def text_turns(history: list[dict], turns: int) -> list[dict]:
    """The conversation as the Omni server takes it: the last ``turns`` user and assistant lines as plain text, without
    mood tags (a model that speaks its text would say them; the cut marker stays, as replies.py keeps it), and never
    with the server's special-token syntax."""

    out = []
    for m in history:
        if m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
            continue
        content = m["content"]
        if m["role"] == "assistant":
            content = TAGS.sub(lambda t: t.group(0) if t.group(0).strip() == "[interrupted]" else "", content)
        content = content.replace("<|", "<").strip()
        if content:
            out.append({"role": m["role"], "content": content})
    return out[-turns:] if turns > 0 else []


def heard_part(written: str, seconds: float, per_second: float = CHARS_PER_S) -> str:
    """The start of a reply's text that ``seconds`` of its speech got through, ended at a word: the text runs ahead of
    the speech, so on barge-in this, not everything written, is what the person heard."""

    if seconds <= 0:
        return ""
    at = int(seconds * per_second)
    if at >= len(written):
        return written
    space = written.rfind(" ", 0, at + 1)
    return written[: space if space > 0 else at].rstrip()


class Pocket:
    def __init__(self, spec: dict) -> None:
        self.url = spec["url"].rstrip("/")
        if self.url.endswith("/v1"):
            self.url = self.url[:-3]
        self.model = spec.get("model", "lfm3-omni-2b")
        self.system = spec.get("system", SYSTEM)
        self.mode = spec.get("mode", "omni")  # "omni": every spoken turn; "hybrid": small talk only, the rest to the brain
        if self.mode not in ("omni", "hybrid"):
            raise ValueError(f"pocket.mode is \"omni\" or \"hybrid\", not {self.mode!r}")
        self.speculate = bool(spec.get("speculate", True))  # hybrid: Omni starts before the words are in
        self.guard = bool(spec.get("guard", True))  # a reply that repeats an earlier one goes to the brain instead
        self.turns = int(spec.get("history_turns", 4 if self.mode == "hybrid" else 12))
        self.extra = dict(spec.get("extra", {}))  # audio_temperature, audio_repetition_penalty, max_new, seed
        self.vision = bool(spec.get("vision", False))  # the camera's frame goes with each spoken turn
        self.vision_note = spec.get("vision_note", VISION_NOTE)
        self.vision_width = int(spec.get("vision_width", 640))
        # hybrid: who answers a left-or-right question while Omni sees: "eyes" (the profile's eyes, Boo's sides())
        # or "omni" (Omni with the mirrored frame). Omni's which-hand answers flip with small changes to the prompt
        # or the audio (3 to 5 of 8 in a Pocket Boo, against 6 of 8 for Cosmos3)
        self.vision_sides = spec.get("vision_sides", "eyes")
        mirror = spec.get("vision_mirror", False)
        self.vision_mirror = mirror if mirror == "sides" else bool(mirror)  # True, False or "sides"
        self.sample_rate = 24_000  # what the server speaks (pcm16 mono); the page plays the rate audio_start names
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=3))

    async def stream(self, audio: np.ndarray, system: str, history: list[dict],
                     image_png: str | None = None) -> AsyncIterator[tuple[str, object]]:
        """The reply to one spoken turn as it comes: ("text", delta), ("audio", int16 PCM bytes at sample_rate) and at
        the end ("done", the server's timings). ``image_png`` (base64) is what the camera shows, seen with the words.
        Closing it hangs up, and the server stops that reply."""

        body = {"audio": base64.b64encode(wav_bytes(audio)).decode(), "system": system,
                "history": text_turns(history, self.turns), "stream": True, **self.extra}
        if image_png:
            body["images"] = [image_png]
            if self.vision_note:
                body["system"] = f"{system} {self.vision_note}"
        async with self.client.stream("POST", self.url + "/v1/audio/chat", json=body) as reply:
            if reply.status_code != 200:
                detail = (await reply.aread()).decode(errors="replace")[:200]
                raise RuntimeError(f"Pocket ({self.model}) answered {reply.status_code}: {detail}")
            async for line in reply.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                event = json.loads(data)
                kind = event.get("type")
                if kind == "text" and event.get("delta"):
                    yield "text", event["delta"]
                elif kind == "audio" and event.get("data"):
                    yield "audio", base64.b64decode(event["data"])
                elif kind == "done":
                    yield "done", {k: v for k, v in event.items() if k != "type"}



# a small speech model asked to do something often says it can't ("I can't create a video clip directly, but ...",
# "I'm not able to adjust the music levels in this mode") instead of handing over: that is the back agent's work.
# Its own reply about itself, so "I" first; "I can't wait", "I can't believe" and the like are no refusal
CANNOT_DO = re.compile(r"\bi(?:\s+|')(?:can'?t|cannot|can not|am unable to|m unable to|am not able to|m not able to|"
                       r"don'?t have the ability to|do not have the ability to)\s+(?:directly\s+|actually\s+|really\s+)?"
                       r"(?!wait\b|believe\b|help but\b|tell\b|hear\b|understand\b|stop\b)[a-z]+")


def unable(text: str) -> bool:
    """Whether a reply says it cannot make or do the thing asked (a hand-over the reply itself missed)."""

    return bool(CANNOT_DO.search(text.lower().replace("\u2019", "'")))


def mood_of(text: str) -> str:
    """A face for a reply that came without a mood tag: a question looks curious, an exclamation happy."""

    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    if first.endswith("?"):
        return "curious"
    if first.endswith("!"):
        return "happy"
    return "neutral"


# ----- hybrid: which turns Omni answers -----

_THANKS = r"(?:thanks|thank you|thank u|thx|cheers|ta|many thanks|much appreciated|(?:i )?appreciate (?:it|that))"
_NICE = (r"(?:so |really |very |pretty |super |just |quite )?(?:great|cool|nice|awesome|perfect|funny|amazing|good|fine|ok|"
         r"okay|fair|interesting|lovely|wonderful|brilliant|hilarious|neat|sweet|fun|true|right|helpful|useful|excellent|"
         r"fantastic|beautiful|incredible|epic|exciting|clever|smart|wild|crazy|impressive)")
_SMALL = [
    # greetings and goodbyes
    r"(?:hi|hello|hey|hiya|heya|howdy|yo|greetings|good (?:morning|afternoon|evening|night|day)|morning|evening)",
    r"(?:bye|goodbye|bye bye|see (?:you|ya)(?: (?:later|soon|tomorrow|then))?|talk (?:to you )?(?:later|soon)|night night|"
    r"catch you later|take care|have a (?:good|great|nice) (?:one|day|night|evening|weekend))",
    # thanks
    _THANKS + r"(?: (?:so much|a lot|a bunch|again|anyway|very much|for (?:that|this|everything|the help|your help)))*",
    # how are you
    r"(?:how are (?:you|things|ya)(?: (?:doing|today|going|tonight))*|how(?:'s| is) it going|how(?:'s| is) (?:your|the) "
    r"(?:day|night|evening)(?: going)?|what'?s up|what is up|sup|how have you been|how do you do|how(?:'s| is) life|"
    r"(?:are )?you (?:ok|okay|alright|all right|good|there|awake|doing (?:ok|okay|well|good)))",
    r"(?:(?:i'?m|i am) (?:good|fine|great|ok|okay|alright|all right|well|tired|back|here|doing (?:well|good|great|fine|ok|okay))|"
    r"not (?:bad|much)|pretty good|all good|same here|me too|you too)",
    # reactions: "That was really helpful", "Sounds awesome", "Love it"
    rf"(?:(?:that|this|it)(?:'s| is| was| sounds| looks| seems)?|sounds|looks|very|how|so|what a)(?: an?)? {_NICE}(?: (?:idea|one|answer|job|work))?(?: to me)?",
    rf"(?:(?:i )?love (?:it|that|this)|(?:i )?like (?:it|that)|(?:good|great|nice) (?:job|one|stuff|work)|well done|nice one|{_NICE})",
    # acknowledgements
    r"(?:ok|okay|k|alright|all right|right|sure|wow|whoa|oh|ah|aha|ha|haha|hahaha|lol|hmm|mm|hm|uh huh|yes|yeah|yep|"
    r"yup|no|nope|nah|got it|gotcha|i see|makes sense|fair enough|no worries|no problem|all set|that'?s (?:all|it)|nothing|"
    r"not really|exactly|indeed|of course|definitely|absolutely|totally|nice to meet you|good to (?:see|hear) you|"
    r"you'?re (?:welcome|funny|the best|great|awesome|sweet)|my pleasure)",
]
FILLER = r"(?:um|uh|er|erm|well|so|and|just|oh|ah|please|then|again|too|buddy|friend|mate|man|dude|there|boo|poo|bo|bu)"
SMALL_TALK = re.compile(rf"^(?:{FILLER} )*(?:{'|'.join(_SMALL)})(?: (?:{FILLER}|{'|'.join(_SMALL)}))*$")
# the words of a bare answer to a question the assistant just asked ("Want me to render it?" -> "Yes"): it carries
# the earlier turn's meaning, so the brain answers it
CONFIRM = re.compile(r"\b(?:yes|yeah|yep|yup|sure|no|nope|nah|ok|okay|alright|all right|please|do it|go ahead|of course|"
                     r"definitely|absolutely|fine|right|exactly|true|not really)\b")


def plain(text: str) -> str:
    """Words as the small-talk patterns read them: lower case, no punctuation but apostrophes, single spaces."""

    text = text.lower().replace("\u2019", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text).split())


def small_talk(text: str, last_reply: str | None = None, max_words: int = 12) -> bool:
    """Whether a turn is small talk Omni can answer: every word of it is a greeting, thanks, an acknowledgement, a
    how-are-you or a goodbye ("Hey Boo, thanks!", "Okay, cool.", "Good morning, how are you?"). One word more (a
    question, a request, a name, a number) sends it to the brain, and so does a bare yes or no right after the
    assistant asked something (``last_reply`` ends with a question)."""

    words = plain(text)
    if not words or len(words.split()) > max_words or not SMALL_TALK.match(words):
        return False
    if last_reply and CONFIRM.search(words):
        asked = replies.TAGS.sub(" ", last_reply).strip()
        if asked.endswith("?"):
            return False
    return True


def repeats(reply: str, earlier: list[str]) -> bool:
    """Whether a reply (of four words or more) says again what one of ``earlier`` said: it opens with the same five
    words, or shares four in five of its words with it. A 2B model's habit: with a few turns of history it answers
    a new question with an old answer."""

    said = replies.words(reply)
    if len(said) < 4:
        return False
    for line in earlier:
        before = replies.words(line)
        if len(before) < 4:
            continue
        n = min(5, len(said))
        if before[:n] == said[:n]:
            return True
        mine, theirs = set(said), set(before)
        if len(mine & theirs) / len(mine | theirs) >= 0.8:
            return True
    return False


# camera questions the eyes' word list misses ("Where am I right now?", "What does my sign say?"): with the camera on,
# Pocket sends these to a model that sees, never to the brain alone, which would answer them from the conversation
LOOKS = re.compile(r"\b(?:where am i|where are we|what room|what(?:'s| is| does) (?:this|that|my|the) [\w ]{0,20}(?:say|read|says)|"
                   r"read (?:this|that|it|my|the)|my (?:sign|shirt|hat|hair|glasses|face|outfit|room|desk|background|top|"
                   r"jacket|hoodie|sweater|mug|cup|book|phone|paper|note|card|drawing|t shirt|tshirt)\b|"
                   r"behind me|around me|in my hands?|am i (?:wearing|holding|smiling|in)|do i look|can you see|"
                   r"what(?:'s| is) (?:this|that|in my)|what colou?r)")


def looks(text: str) -> bool:
    """Whether a turn asks about what the camera shows, in words the eyes' list (engines.wants_eyes) does not have."""

    return bool(LOOKS.search(plain(text)))
