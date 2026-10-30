"""The three engines a turn goes through: speech recognition (Parakeet, or Nemotron 3.5 Transcribe on TensorFold with
Parakeet behind it), the brain (Nemotron 3.5 Bolt
on TensorFold, OpenAI-compatible chat) and the voice (Fish Audio Drama 3 on TensorFold, /v1/audio/speech streamed as
PCM). Each is chosen by the profile (profiles/*.json), so the same agent runs on a Mac and on a DGX Spark."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import struct
import sys
import time
from contextlib import aclosing
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

def _guard_nemo_nvrtc_log() -> None:
    """NeMo compiles its CUDA-graph decoder kernel with NVRTC and reads the compile log into b" " * size. An empty log is
    one byte, and b" " * 1 is Python's shared b" " object, so NVRTC writes a NUL into it: from then on every b" " in the
    process is b"\\x00" (HTTP status lines, websockets' request parser, and any .pyc written afterwards). Give NVRTC a
    buffer of its own for such a log; the log is not read."""
    try:
        from cuda.bindings import nvrtc
    except ImportError:
        return
    get_log = nvrtc.nvrtcGetProgramLog
    if getattr(get_log, "_own_buffer", False):
        return

    def own_buffer(prog, log):
        if isinstance(log, bytes) and len(log) <= 1:
            log = bytes(len(log))  # a new object, never a shared one
        return get_log(prog, log)

    own_buffer._own_buffer = True
    nvrtc.nvrtcGetProgramLog = own_buffer


# the model card's prompt for readable text: without a transcription instruction this size of speech LLM answers a spoken
# request ("Give me a logline for...") instead of writing it down
TRANSCRIBE_PROMPT = ("Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the "
                     "remaining spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only.")


def wav_bytes(audio: np.ndarray) -> bytes:
    """A 16-bit PCM WAV file (mono, 16 kHz) of float32 samples."""
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
    return b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm


class SpeechServer:
    """Text of one utterance from an OpenAI-compatible chat server that takes audio as input_audio parts: Nemotron 3.5
    Transcribe on TensorFold. None when it is down, refuses or misses its deadline (timeout_ms), so the caller falls
    back; after a refused connection, or two misses in a row, it is skipped for retry_s seconds."""

    def __init__(self, spec: dict, transport: httpx.BaseTransport | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.url = spec["url"].rstrip("/") + "/chat/completions"
        self.model = spec.get("model", "transcribe")
        self.timeout = spec.get("timeout_ms", 1500) / 1000
        self.prompt = spec.get("prompt", TRANSCRIBE_PROMPT)
        self.retry_s = spec.get("retry_s", 30)
        self.max_tokens = spec.get("max_tokens", 256)
        self.client = httpx.Client(transport=transport, timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 1.0)))
        self.clock = clock
        self.down_until = 0.0
        self.misses = 0
        self.last_error: str | None = None

    def _down(self, why: str) -> None:
        self.last_error = why
        self.down_until = self.clock() + self.retry_s

    def __call__(self, audio: np.ndarray) -> str | None:
        if self.clock() < self.down_until:
            return None
        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav_bytes(audio)).decode(), "format": "wav"}}]}]}
        try:
            reply = self.client.post(self.url, json=body)
        except httpx.TimeoutException:
            self.misses += 1
            self.last_error = "timeout"
            if self.misses >= 2:
                self._down("timeout twice")
            return None
        except httpx.HTTPError as e:  # refused, reset: the server is down or restarting
            self._down(type(e).__name__)
            return None
        if reply.status_code != 200:
            self.last_error = f"HTTP {reply.status_code}"
            if reply.status_code >= 500:
                self._down(self.last_error)
            return None
        try:
            text = reply.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            self.last_error = "bad reply"
            return None
        self.misses = 0
        self.last_error = None
        return (text or "").strip()


def cap_mlx_cache(mb) -> None:
    """Cap MLX's buffer cache on Apple GPUs (Metal) at ``mb`` megabytes; ``None`` or a negative value leaves it alone.

    MLX keeps the buffers of finished arrays for reuse. After a long utterance Parakeet's cache holds about 1.5 GB that
    no later call needs (measured on an M5 Ultra: a 20 s utterance, 3.15 GB footprint, 2.23 GB after clearing it). A cap
    keeps the reuse for the usual short turns and gives the rest back. Metal only: never on MLX's CUDA backend."""

    if mb is None or float(mb) < 0 or sys.platform != "darwin":
        return
    import mlx.core as mx

    if not mx.metal.is_available():
        return
    setter = getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit  # mx.metal.* before MLX 0.24
    setter(int(float(mb) * 1024 * 1024))


class Recognizer:
    """Text of one utterance (float32 mono at 16 kHz).

    In process by default. With ``"server"`` in the spec (a speech server such as Nemotron 3.5 Transcribe on
    TensorFold, ``SpeechServer``), the utterance goes there first; when it is down, refuses or misses its deadline, the
    Parakeet recognizer below answers. With ``"url"`` in the spec (``voice_agent.asr_server``), that Parakeet is a
    recognizer shared by the agents on one machine, so they keep one copy of the model; when that server does not
    answer, the utterance is recognized here (the model loads on the first such turn) and the server is tried again
    after ``retry_s`` seconds. ``"fallback": false`` turns the local copy off (the turn then fails). ``where`` names
    who gave the last text: "server", "shared" or "local"."""

    def __init__(self, spec: dict, transport: httpx.BaseTransport | None = None) -> None:
        self.server = SpeechServer(spec["server"], transport) if (spec.get("server") or {}).get("url") else None
        self.kind, self.model_id = spec["kind"], spec["model"]
        self.language = spec.get("language")
        self.cache_mb = spec.get("mlx_cache_mb", 256)
        self.url = (spec.get("url") or "").rstrip("/")
        self.fallback = spec.get("fallback", True)
        self.retry_s = float(spec.get("retry_s", 30))
        self.model = None
        self.failed_at = 0.0  # when the shared recognizer last failed (0: it has not)
        self.where = "local"  # who recognized the last utterance: "server", "shared" or "local"
        shared = False
        if self.url:
            self.remote = httpx.Client(timeout=httpx.Timeout(float(spec.get("timeout_s", 10)), connect=1.0), transport=transport)
            try:
                self.remote.get(self.url + "/health").raise_for_status()
                shared = True
            except httpx.HTTPError as error:
                if not self.fallback:
                    raise RuntimeError(f"the shared recognizer at {self.url} does not answer: {error}") from error
                print(f"asr: the shared recognizer at {self.url} does not answer ({error}); recognizing in process", flush=True)
                self.failed_at = time.monotonic()
        if not shared:
            self.load()
        if self.server:  # warm up the speech server too; a slow first answer is not held against it
            self.server(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
            self.server.misses = 0

    def load(self) -> None:
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

            _guard_nemo_nvrtc_log()
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_id).eval().cuda()
        else:
            raise ValueError(f"unknown recognizer kind {self.kind}")
        if self.kind in ("parakeet-mlx", "mlx-audio"):
            cap_mlx_cache(self.cache_mb)
        self.local(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))  # warm up: the first call compiles kernels

    def __call__(self, audio: np.ndarray) -> str:
        if self.server:
            text = self.server(audio)
            if text is not None:
                self.where = "server"
                return text
        if self.url and (not self.failed_at or time.monotonic() - self.failed_at >= self.retry_s):
            try:
                answer = self.remote.post(self.url + "/v1/transcribe", content=np.ascontiguousarray(audio, dtype="<f4").tobytes(),
                                          headers={"Content-Type": "application/octet-stream"},
                                          params={"language": self.language} if self.language else None)
                answer.raise_for_status()
                self.failed_at, self.where = 0.0, "shared"
                return str(answer.json()["text"]).strip()
            except (httpx.HTTPError, ValueError, KeyError) as error:
                if not self.fallback:
                    raise
                print(f"asr: the shared recognizer at {self.url} failed ({error}); recognizing in process", flush=True)
                self.failed_at = time.monotonic()
        if self.model is None:
            self.load()
        self.where = "local"
        return self.local(audio)

    def local(self, audio: np.ndarray) -> str:
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
        self.extra = dict(spec.get("extra", {}))
        # "thinking": false sends enable_thinking false with every request: models whose chat template thinks unless
        # told not to (Cosmos3, LFM3-VL) otherwise spend seconds reasoning before a camera answer
        if spec.get("thinking") is False:
            self.extra["chat_template_kwargs"] = {**self.extra.get("chat_template_kwargs", {}), "enable_thinking": False}
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
# small camera models often answer with the side alone ("Left.", "On the right side.") or say "on the left" in a sentence
# without "of the picture"; both are still the picture side. "the left hand", "the left of you" and the like are not.
BARE_SIDE = re.compile(r"^\W*(?:(?:it'?s|it\s+is)\s+)?(?:(?:on|to|at)\s+)?(?:the\s+)?(left|right)(?:\s+(?:side|half|part))?\W*$", re.I)
LOOSE_SIDE = re.compile(r"\b(?:on|to|at)\s+the\s+(left|right)(?:\s+(?:side|half|part))?\b"
                        r"(?!\s+(?:side|half|part|hand|arm|wrist|shoulder|ear|eye|leg|foot|cheek|of)\b)", re.I)


def picture_sides(text: str) -> str | None:
    """The model's picture sides said as the person's own ("your left side"); None when it named no picture side."""

    if PICTURE_SIDE.search(text):
        return PICTURE_SIDE.sub(lambda m: f"your {m.group(1).lower()} side", text)
    bare = BARE_SIDE.match(text)
    if bare:
        return f"On your {bare.group(1).lower()} side."
    if LOOSE_SIDE.search(text):
        return LOOSE_SIDE.sub(lambda m: f"on your {m.group(1).lower()} side", text)
    return None


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
    return picture_sides(text)


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
        # warm it while the person speaks (Drama 3 on a Mac is slow to start when idle); Kokoro answers at once and sets
        # false, and so does the DGX Spark profile: its CUDA Drama 3 runs one synthesis at a time and starts fast from
        # idle, so a throwaway line still being spoken when the person stops held the reply back (503s, ~0.4 s more)
        self.keep_warm = bool(spec.get("keep_warm", True))
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5))
        # Drama 3 takes two speech streams at once (TensorFold --parallel, default 2) and answers 503 "speech output
        # capacity is full" to a third, and a stream the agent let go holds its slot until the server notices (~50 ms).
        # So speech counts its streams (every conversation shares this Voice), warming speaks only when no speech is
        # under way and only one line at a time, and a refused request is asked again for up to retry_s.
        self.speaking = 0
        self.warming = False
        self.retry_s = float(spec.get("retry_s", 6.0))

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

        self.speaking += 1
        try:
            for attempt in range(3):
                stalled: list[float] = []
                async for pcm in self.take(text, voice, stalled):
                    yield pcm
                if not stalled:
                    return
                text = unspoken(text, stalled[0], attempt)
                if not text:
                    return
        finally:
            self.speaking -= 1

    async def warm(self, voice: str | None = None) -> bool:
        """Speaks a throwaway "Mm." to keep the engine from going idle, unless speech or another warm line is under
        way (both keep it warm already, and a third stream would be refused). False when it did not speak."""

        if self.speaking or self.warming:
            return False
        self.warming = True
        try:
            async for _ in self.take("Mm.", voice, [], retry=False):
                pass
            return True
        except Exception:
            return False
        finally:
            self.warming = False

    async def opened(self, body: dict, retry: bool) -> httpx.Response:
        """The streamed reply to ``body``; a 503 (all speech slots busy, or one still being let go) is asked again
        every 50 ms for up to retry_s, so a spoken line waits a moment instead of being lost."""

        deadline = time.monotonic() + (self.retry_s if retry else 0)
        while True:
            reply = await self.client.send(self.client.build_request("POST", self.url, json=body), stream=True)
            if reply.status_code != 503 or time.monotonic() >= deadline:
                return reply
            await reply.aclose()
            await asyncio.sleep(0.05)

    async def take(self, text: str, voice: str | None, stalled: list[float], retry: bool = True) -> AsyncIterator[bytes]:
        """One request. Quiet after the first second of speech is held back until speech resumes; STALL_S of it ends
        the take, and the seconds spoken before it go into ``stalled``."""

        body = {"model": self.model, "input": text, "response_format": "pcm", "stream": True}
        if voice:
            body["voice"] = voice
        reply = await self.opened(body, retry)
        try:
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
        finally:
            await reply.aclose()


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


SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")


def said_as(sentence: str) -> str:
    """A sentence as words only, for spotting one said again ("I'm on it." and "I'm on it!" are the same)."""

    return " ".join(re.sub(r"[^\w\s']", " ", sentence.lower()).split())


async def unrepeated(pieces: AsyncIterator[str], stopped: list | None = None) -> AsyncIterator[str]:
    """The reply up to where it starts saying itself again. In a long conversation a small model can fall into a copy
    loop ("That's Boo. I'm on it. That's Boo. I'm on it. ..." until max_tokens), so a sentence already said in this
    reply ends it there, unsaid, and the model's stream is closed. The first sentence goes through piece by piece (it
    cannot repeat anything, and speech starts on it); later ones go out once whole. ``stopped`` gets the repeat."""

    seen: set[str] = set()
    first, head, rest = True, "", ""  # head: the first sentence so far; rest: a later sentence not yet checked
    async with aclosing(pieces) as source:
        async for piece in source:
            if first:
                head += piece
                end = SENTENCE_END.search(head)
                if not end:
                    yield piece
                    continue
                first = False
                cut = len(piece) - (len(head) - end.end())  # where in this piece the first sentence ends
                if cut > 0:
                    yield piece[:cut]
                seen.add(said_as(head[: end.end()]))
                piece = piece[max(cut, 0):]
            rest += piece
            while end := SENTENCE_END.search(rest):
                sentence, rest = rest[: end.end()], rest[end.end():]
                words = said_as(sentence)
                if len(words) >= 6 and words in seen:
                    if stopped is not None:
                        stopped.append(sentence.strip())
                    return
                seen.add(words)
                yield sentence
        if rest:
            words = said_as(rest)
            if len(words) >= 6 and words in seen:
                if stopped is not None:
                    stopped.append(rest.strip())
                return
            yield rest


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
