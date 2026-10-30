"""The Pocket tier: one small model hears the person and answers out loud, in place of recognizer -> brain -> voice.

LFM3-Omni-2B (Liquid AI) on TensorFold takes the spoken turn itself and streams back its reply as text and 24 kHz
speech together (``lfm3-omni-generate --serve``: POST /v1/audio/chat, server-sent events). The agent keeps what costs
little around it: voice activity and barge-in, the conversation (sent as text turns, ``history``), the transcript
(the profile's recognizer runs beside it, never waited for), and hand-overs to a back agent.

    "pocket": {"url": "http://127.0.0.1:8094", "model": "lfm3-omni-2b", "system": "...", "history_turns": 12,
               "extra": {"audio_temperature": 0.4, "audio_repetition_penalty": 1.2}}
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

SYSTEM = ("Respond with interleaved text and audio. You are a friendly voice assistant. You are speaking out loud: "
          "answer in one to three short spoken sentences, plain words, no lists, no markdown, no emoji.")
CHARS_PER_S = 14.0  # how fast speech gets through a reply's text (as engines.unspoken): where a cut reply was cut
TAGS = re.compile(r"\s*\[\w+(?: \w+)*\]")


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
        self.turns = int(spec.get("history_turns", 12))
        self.extra = dict(spec.get("extra", {}))  # audio_temperature, audio_repetition_penalty, max_new, seed
        self.sample_rate = 24_000  # what the server speaks (pcm16 mono); the page plays the rate audio_start names
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=3))

    async def stream(self, audio: np.ndarray, system: str, history: list[dict]) -> AsyncIterator[tuple[str, object]]:
        """The reply to one spoken turn as it comes: ("text", delta), ("audio", int16 PCM bytes at sample_rate) and at
        the end ("done", the server's timings). Closing it hangs up, and the server stops that reply."""

        body = {"audio": base64.b64encode(wav_bytes(audio)).decode(), "system": system,
                "history": text_turns(history, self.turns), "stream": True, **self.extra}
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
