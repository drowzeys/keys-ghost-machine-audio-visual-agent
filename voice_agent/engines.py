"""The three engines a turn goes through: speech recognition (Parakeet or Nemotron ASR), the brain (Nemotron 3.5 Bolt
on TensorFold, OpenAI-compatible chat) and the voice (Fish Audio Drama 3 on TensorFold, /v1/audio/speech streamed as
PCM). Each is chosen by the profile (profiles/*.json), so the same agent runs on a Mac and on a DGX Spark."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import AsyncIterator, Callable

import httpx
import numpy as np

SAMPLE_RATE = 16_000  # what the microphone sends and the recognizers take


# ---------- voice activity: Silero VAD v5 through ONNX Runtime (no PyTorch) ----------

class Vad:
    """Speech probability for each 512-sample (32 ms) window at 16 kHz."""

    WINDOW = 512

    def __init__(self, path: str) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate([self.context, window.reshape(1, -1)], axis=1)
        out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        self.context = x[:, -64:]
        return float(out[0][0])


# ---------- speech recognition ----------

class Recognizer:
    """Text of one utterance (float32 mono at 16 kHz)."""

    def __init__(self, spec: dict) -> None:
        self.kind, self.model_id = spec["kind"], spec["model"]
        if self.kind == "parakeet-mlx":
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel

            self.model = from_pretrained(self.model_id)
            self._mel = get_logmel
        elif self.kind == "mlx-audio":
            from mlx_audio.stt import load

            self.model = load(self.model_id)
        elif self.kind == "nemo":
            import nemo.collections.asr as nemo_asr

            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_id).eval().cuda()
        else:
            raise ValueError(f"unknown recognizer kind {self.kind}")
        self.language = spec.get("language")
        self(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))  # warm up: the first call compiles kernels

    def __call__(self, audio: np.ndarray) -> str:
        if self.kind == "parakeet-mlx":
            import mlx.core as mx

            mel = self._mel(mx.array(audio), self.model.preprocessor_config)
            return self.model.generate(mel)[0].text.strip()
        if self.kind == "mlx-audio":
            import mlx.core as mx

            kwargs = {"language": self.language} if self.language else {}
            return self.model.generate(mx.array(audio), **kwargs).text.strip()
        import torch

        with torch.inference_mode():
            out = self.model.transcribe([audio], batch_size=1, verbose=False)
        first = out[0][0] if isinstance(out, tuple) else out[0]
        return (first.text if hasattr(first, "text") else str(first)).strip()


# ---------- the brain: chat on TensorFold (OpenAI-compatible) ----------

# the camera faces the person: without this a model answers "what am I wearing?" about itself ("I'm wearing...")
CAMERA = ("[This picture is from the user's camera, looking at them: it shows the person you are talking to and their "
          "surroundings, not you. Speak about them as \"you\" (\"you're wearing...\"), and be specific: name logos, characters, "
          "printed graphics, text and colours you can see.]\nThe user asks: ")


class Brain:
    def __init__(self, spec: dict) -> None:
        self.url = spec["url"].rstrip("/") + "/chat/completions"
        self.model = spec["model"]
        self.extra = spec.get("extra", {})
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5))

    async def warm(self, messages: list[dict]) -> None:
        """A one-token request while the person is still speaking: wakes an idle GPU (a Metal server that has gone
        quiet takes ~250 ms more for its first token) and puts the conversation so far in the prompt cache."""

        try:
            await self.client.post(self.url, json={"model": self.model, "messages": messages, **self.extra, "max_tokens": 1})
        except httpx.HTTPError:
            pass

    async def stream(self, messages: list[dict], image_png: str | None = None) -> AsyncIterator[str]:
        """The reply, piece by piece; with image_png (base64), the picture goes with the last user message."""

        if image_png:
            # only the instructions and this question: a small camera model copies its earlier answers out of the
            # history instead of looking at the new picture
            last = messages[-1]
            messages = [m for m in messages[:1] if m["role"] == "system"] + [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_png}},
                {"type": "text", "text": CAMERA + last["content"]}]}]
        body = {"model": self.model, "messages": messages, "stream": True, **self.extra}
        async with self.client.stream("POST", self.url, json=body) as reply:
            reply.raise_for_status()
            async for line in reply.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                delta = json.loads(line[5:])["choices"][0]["delta"].get("content")
                if delta:
                    yield delta


# left and right: a model trained on first-person (robot) cameras knows which side of a picture a thing is on, but turns a
# face-forward selfie into the person's sides by habit ("your left wrist" either way). So it is shown the mirrored frame
# (the person's own view, where the picture's left is their left), asked only for the picture side, and code says whose.
SIDES = re.compile(r"\b(my|your|the|on the|to the|to my) (left|right)\b|\bwhich (side|wrist|shoulder|ear|eye|leg|foot|cheek)\b", re.I)
WHERE = ("Someone asks about this picture: \"{q}\"\nFor each thing the question is about, say only where it is in the "
         "picture: on the left of the picture, on the right of the picture, or in the middle. Do not say whose left or "
         "right. One short sentence.")
PICTURE_SIDE = re.compile(r"(?:the\s+)?(left|right)(?:\s+(?:side|half|part))?\s+of\s+the\s+(?:picture|image|frame)", re.I)


async def sides(brain: "Brain", mirrored_png: str, question: str) -> str | None:
    """Where the asked-about things are, as the person's own left and right; None when the model gave no picture side."""

    body = {"model": brain.model, "max_tokens": 60, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + mirrored_png}},
                {"type": "text", "text": WHERE.format(q=question)}]}]}
    try:
        reply = await brain.client.post(brain.url, json=body)
        text = reply.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None
    if not PICTURE_SIDE.search(text):
        return None
    return PICTURE_SIDE.sub(lambda m: f"your {m.group(1).lower()} side", text)


class Said:
    """A reply already worked out in code, spoken as it is."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def stream(self, messages: list[dict], image_png: str | None = None) -> AsyncIterator[str]:
        yield self.text


LOOKING = ("see", "look", "camera", "finger", "hand", "holding up", "gesture", "holding", "wearing", "show", "read this", "what is this", "what's this",
           "in front of", "describe", "color", "colour", "how do i look", "picture", "screen")


def wants_eyes(text: str, words: tuple[str, ...] = LOOKING) -> bool:
    """Whether a turn asks about what the camera shows."""

    low = f" {text.lower()} "
    return any(f" {w}" in low for w in words)


def jpeg_to_png(jpeg_b64: str, width: int = 640, mirror: bool = False) -> str:
    """The camera's JPEG as PNG (the CUDA vision tower reads PNG without an extra decoder), at most ``width`` wide;
    ``mirror`` flips it to the person's own selfie view."""

    import base64
    import io

    from PIL import Image, ImageOps

    image = Image.open(io.BytesIO(base64.b64decode(jpeg_b64))).convert("RGB")
    if mirror:
        image = ImageOps.mirror(image)
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)))
    out = io.BytesIO()
    image.save(out, "PNG", optimize=False, compress_level=1)
    return base64.b64encode(out.getvalue()).decode()


# ---------- the voice: Drama 3 on TensorFold ----------

class Voice:
    def __init__(self, spec: dict) -> None:
        self.url = spec["url"].rstrip("/") + "/audio/speech"
        self.model = spec["model"]
        self.sample_rate = int(spec.get("sample_rate", 44_100))
        self.samples = spec.get("samples")  # the voices' files and .wav samples
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5))

    async def voices(self) -> list[dict]:
        """The engine's named voices ([{id, description}]); empty when it only has its built-in voice. An engine that
        lists only names (the CUDA build) gets each description from its voice file in the samples folder."""

        try:
            reply = await self.client.get(self.url.rsplit("/audio/", 1)[0] + "/voices", timeout=3)
            listing = reply.json() if reply.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return []
        if listing.get("details"):
            return listing["details"]
        return [{"id": v["name"], "description": self.described(v["name"])} for v in listing.get("data", []) if v.get("name")]

    def described(self, name: str) -> str:
        import os

        try:
            with open(os.path.join(os.path.expanduser(self.samples or ""), f"{name}.json")) as f:
                return json.load(f).get("description", "")
        except (OSError, ValueError):
            return ""

    async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        """int16 mono PCM at ``sample_rate``, chunk by chunk as the engine decodes it, in ``voice`` (a named reference
        voice keeps one speaker across requests; the built-in voice picks a new one for every sentence).

        Drama 3 sometimes stalls after a sentence: the rest is silence or a flat hum under -30 dBFS until the frame cap
        (TensorFold 1.1.4's silence guard ends such a stream instead, but never redraws it). The same text stalls every
        time, since its seed comes from the prompt, so a stalled take's quiet tail is dropped and the text it most likely
        left unsaid is asked for again, at most twice."""

        for attempt in range(3):
            stalled: list[float] = []
            async for pcm in self.take(text, voice, stalled):
                yield pcm
            if not stalled:
                return
            text = unspoken(text, stalled[0], attempt)
            if not text:
                return

    async def take(self, text: str, voice: str | None, stalled: list[float]) -> AsyncIterator[bytes]:
        """One request. Quiet after the first second of speech is held back until speech resumes; STALL_S of it ends
        the take, and the seconds spoken before it go into ``stalled``."""

        body = {"model": self.model, "input": text, "response_format": "pcm", "stream": True}
        if voice:
            body["voice"] = voice
        async with self.client.stream("POST", self.url, json=body) as reply:
            reply.raise_for_status()
            rate = reply.headers.get("x-sample-rate")
            if rate:
                self.sample_rate = int(rate)
            carry = b""
            # greedy decoding can loop to the frame limit (seen on CUDA: 47 s for one sentence); a spoken sentence needs
            # about 0.08 s a character, so audio far past that is cut off
            limit = int(self.sample_rate * (2.0 + 0.15 * len(text))) * 2
            sent = 0
            window = self.sample_rate // 20 * 2  # 50 ms, in bytes
            seconds = heard = 0.0  # audio so far; where the last speech ended
            held = b""
            async for chunk in reply.aiter_bytes():
                chunk = carry + chunk
                cut = len(chunk) - len(chunk) % 2  # whole samples only
                carry = chunk[cut:]
                out = b""
                for i in range(0, cut, window):
                    piece = chunk[i : min(i + window, cut)]
                    seconds += len(piece) / 2 / self.sample_rate
                    if loud(piece):
                        heard = seconds
                    elif heard >= 1.0:
                        held += piece
                        continue
                    out, held = out + held + piece, b""
                if out:
                    sent += len(out)
                    yield out
                if heard >= 1.0 and seconds - heard >= STALL_S:
                    stalled.append(heard)
                    return
                if sent >= limit:
                    return
            if held:
                yield held


STALL_S = 3.0  # quiet that means a stalled take; Drama 3's own pauses run up to about 2.4 s
QUIET = 32768 * 10 ** (-30 / 20)  # -30 dBFS: under it is silence or the stall's hum (about -36 dBFS)


def loud(pcm: bytes) -> bool:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return bool(x.size) and float(np.sqrt(np.mean(x * x))) >= QUIET


def unspoken(text: str, heard: float, attempt: int, per_second: float = 14.0) -> str:
    """What a take that stalled after ``heard`` seconds of speech most likely left unsaid: the text after the sentence
    end nearest to where that much speech gets to (Drama 3 speaks about 12 to 18 characters a second). Nothing when
    that is the end; the whole text again, with spaces added for a new seed, when that is the start."""

    ends = [0] + [m.end() for m in re.finditer(r"[.!?;:]+[\"')]*\s+", text)] + [len(text)]
    at = min(ends, key=lambda i: abs(i - heard * per_second))
    if at == 0:
        return text + " " * (attempt + 1)
    return text[at:].strip()


def sentences(pieces: AsyncIterator[str], first_min: int = 12, min_chars: int = 40, first_words: int = 0) -> AsyncIterator[str]:
    """Text pieces regrouped into what the voice speaks one at a time: the first at its first clause break (so speech
    starts early), or after first_words words when no break has come by then; the rest at sentence ends."""

    async def gen():
        text, said = "", 0
        async for piece in pieces:
            text += piece
            while True:
                limit = first_min if said == 0 else min_chars
                stops = [i for i, ch in enumerate(text) if ch in ".!?;:\n" or (said == 0 and ch == ",")]
                cut = next((i for i in stops if i + 1 >= limit and (i + 1 == len(text) or text[i + 1] in " \n\"')")), None)
                if cut is None and said == 0 and first_words and len(text.split()) > first_words:
                    cut = len(" ".join(text.split()[:first_words])) + len(text) - len(text.lstrip()) - 1
                if cut is None:
                    break
                out, text = text[: cut + 1].strip(), text[cut + 1 :]
                if out:
                    said += 1
                    yield out
        if text.strip():
            yield text.strip()

    return gen()


class Clock:
    """Marks within one turn, in ms from its start (the end of the user's speech)."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, round((time.perf_counter() - self.start) * 1000, 1))


async def cancel(task: asyncio.Task | None) -> None:
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


Callback = Callable[[dict], None]
