# Ported from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman), the AutoPilot
# branch, app/cosmos.py at 6cc6e67c5e138499ea2c17183eda261b710d1217 (data_url, chat, TooMany, look). Apache-2.0:
# LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Changed: one Eyes class for any OpenAI-compatible vision server, chosen by the profile (Gemma 4 12B by default,
# Cosmos3-Edge selectable); its lifecycle is the app's engine registry (loaded on the first look, held and unloaded by
# the memory governor while a render runs) instead of cosmos.py's own instance; images only (the Spark's vLLM build
# crashes on audio input); thinking off and a low temperature for yes/no; the MLX build's leaked <image|> / <audio|>
# tokens are filtered from every answer.
"""AutoPilot's eyes: the vision model every take, keyframe and join is looked at with.

    "autopilot": {"eyes": {"engine": "gemma", "url": "http://127.0.0.1:8311/v1", "model": "gemma-4-12b",
                           "registry": "gemma", "max_images": 9}}

``engine`` picks the defaults: ``gemma`` (Google's Gemma 4 12B, Apache-2.0: vLLM on a DGX Spark, mlx-vlm on a Mac) or
``cosmos`` (NVIDIA Cosmos3-Edge on TensorFold, the fallback; it takes at most 4 pictures unless started with
--vision-max-images). ``registry`` names the app's engine that serves it: each look goes through ``Registry`` so the
engine is loaded on demand and counted in flight, and on a Spark the governor's render schedule does not keep it, so it
is never resident beside a render (it loads again between renders).

Why Gemma: on the brain evaluation's 20 clips (person swaps, cuts, wardrobe colour changes; 52 yes/no questions)
Gemma 4 12B answered 98 % (Mac) and 96 % (Spark vLLM) right with no false alarms; Cosmos3-Edge answered 65 %: it
caught none of 8 person swaps and none of 8 cuts (it says "same person, no cut" almost always).
"""

from __future__ import annotations

import base64
import contextlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

KINDS = {
    "gemma": {"label": "Gemma 4 12B", "model": "gemma-4-12b", "max_images": 9, "temperature": 0.0},
    "cosmos": {"label": "Cosmos3-Edge", "model": "cosmos3", "max_images": 4, "temperature": 0.2},
    # Ghost 3.8 27B on TensorFold 1.2.4 --vision: the brain itself, one picture a request (a three-picture request
    # crashed the GPU on the DGX Spark, 2026-11-08: Xid 119); no log-probabilities, so gma/ap/qc.py GhostLooker asks for ratings
    "ghost": {"label": "Ghost 3.8 27B (vision)", "model": "ghost-3.8-27b", "max_images": 1, "temperature": 0.0},
}
# tokens an MLX build of Gemma 4 leaves in its text (harmless for yes/no, but never part of an answer)
LEAKED = re.compile(r"<\|?(?:image|audio|video|image_soft_token|audio_soft_token|end_of_image|end_of_audio|boi|eoi|boa|eoa)\|>")
TIMEOUT = 240


class TooMany(RuntimeError):
    """The server refused the images (more than it takes, or too many rows): ask again with one frame sheet."""


def data_url(picture: Path) -> str:
    kind = "png" if Path(picture).suffix.lower() == ".png" else "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(Path(picture).read_bytes()).decode()


def clean(text: str) -> str:
    return LEAKED.sub("", text or "").strip()


class Eyes:
    def __init__(self, spec: dict | None, registry=None) -> None:
        spec = dict(spec or {})
        self.engine = spec.get("engine", "gemma")
        base = KINDS.get(self.engine, KINDS["gemma"])
        self.spec = {**base, **spec}
        self.url = str(self.spec.get("url") or "").rstrip("/")
        if self.url and not self.url.endswith("/v1"):
            self.url += "/v1"
        self.model = self.spec["model"]
        self.label = self.spec.get("label") or base["label"]
        self.max_images = int(self.spec.get("max_images") or base["max_images"])
        self.registry = registry
        self.name = self.spec.get("registry")  # the registry engine serving it, if this app runs it
        self.requests = 0
        self.seconds = 0.0
        self.loads: list[dict] = []  # when this app loaded the engine for a look, and how long it took

    def describe(self) -> dict:
        return {"engine": self.engine, "label": self.label, "model": self.model, "url": self.url, "registry": self.name,
                "max_images": self.max_images, "requests": self.requests, "seconds": round(self.seconds, 1), "loads": self.loads[-6:]}

    @contextlib.contextmanager
    def request(self):
        """The server's address for one request: the registry engine loaded first when it is not (it waits while a
        render holds the memory), and counted in flight so no release cuts the answer."""

        one = self.registry.get(self.name) if self.registry is not None and self.name else None
        if one is None or not hasattr(one, "acquire"):
            yield self.url
            return
        began, was = time.time(), one.loaded()
        base = one.acquire()
        if not was:
            self.loads.append({"at": began, "seconds": round(time.time() - began, 1)})
        try:
            yield (base.rstrip("/") + "/v1") if base else self.url
        finally:
            one.release_request()

    def warm(self) -> None:
        """Loads the registry engine now when it is not loaded (no question asked); nothing for a server this app does
        not run. AutoPilot calls it while the ears listen, so the eyes' cold load overlaps other work."""

        with self.request():
            pass

    def post(self, endpoint: str, body: dict) -> str:
        call = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                      {"Content-Type": "application/json"})
        with urllib.request.urlopen(call, timeout=TIMEOUT) as reply:
            return json.loads(reply.read())["choices"][0]["message"].get("content") or ""

    def odds(self, system: str, frames: list, text: str, answers: tuple[str, str]) -> float:
        """P(answers[0]) against answers[1] for one-token answers, from the server's top log-probabilities (vLLM's
        ``logprobs``): the self-check's yes/no and A/B questions (gma/ap/qc.py)."""

        content = []
        for caption, picture in frames:
            content += [{"type": "text", "text": caption}, {"type": "image_url", "image_url": {"url": data_url(picture)}}]
        content.append({"type": "text", "text": text})
        body = {"model": self.model, "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        began = time.time()
        try:
            with self.request() as endpoint:
                call = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                              {"Content-Type": "application/json"})
                with urllib.request.urlopen(call, timeout=TIMEOUT) as reply:
                    top = json.loads(reply.read())["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        finally:
            self.requests += 1
            self.seconds += time.time() - began
        import math
        got = [sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == a.lower()) for a in answers]
        return got[0] / (got[0] + got[1]) if sum(got) > 0 else 0.5

    def chat(self, endpoint: str, system: str, frames: list, text: str, max_tokens: int = 1500) -> str:
        """One request: the frames in time order, each after its caption ("0.00 s"), then the question; thinking off
        (a server that refuses the switch is asked without it)."""

        content = []
        for caption, picture in frames:
            content.append({"type": "text", "text": caption})
            content.append({"type": "image_url", "image_url": {"url": data_url(picture)}})
        content.append({"type": "text", "text": text})
        plain = {"model": self.model, "temperature": float(self.spec.get("temperature", 0.0)), "max_tokens": max_tokens,
                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        began = time.time()
        try:
            try:
                return clean(self.post(endpoint, {**plain, "chat_template_kwargs": {"enable_thinking": False}}))
            except urllib.error.HTTPError as error:
                code, said = error.code, error.read().decode(errors="replace")[:300]
            if code in (400, 422) and "image" not in said.lower():
                try:
                    return clean(self.post(endpoint, plain))
                except urllib.error.HTTPError as error:
                    code, said = error.code, error.read().decode(errors="replace")[:300]
            raise TooMany(said) if code in (400, 413, 422) and "image" in said.lower() else RuntimeError(f"{code}: {said}")
        finally:
            self.requests += 1
            self.seconds += time.time() - began

    def look(self, system: str, frames: list, sheet: Path | None, text: str) -> tuple[str, str]:
        """(answer, who): every frame in one request, or, when the eyes take fewer pictures, the one time-ordered
        sheet of the same frames."""

        with self.request() as endpoint:
            try:
                if len(frames) > self.max_images:
                    raise TooMany(f"{len(frames)} images, {self.max_images} at most")
                return self.chat(endpoint, system, frames, text), self.label
            except TooMany:
                if sheet is None:
                    raise
                note = ("The picture is a contact sheet of the frames, left to right then top to bottom, in time order: "
                        + "; ".join(caption for caption, _ in frames) + ".\n\n")
                return self.chat(endpoint, system, [("Frame sheet", sheet)], note + text), self.label + " (frame sheet)"


def make(profile: dict, registry=None) -> Eyes | None:
    """The eyes a profile names (its "autopilot.eyes", else its "eyes" section as Cosmos3-Edge), or None."""

    ap = profile.get("autopilot") or {}
    spec = ap.get("eyes")
    if isinstance(spec, str):  # a name among the profile's choices: {"eyes": "gemma", "eyes_choices": {"gemma": {...}, ...}}
        spec = (ap.get("eyes_choices") or {}).get(spec)
    if spec is None and profile.get("eyes"):
        spec = {"engine": "cosmos", **{k: v for k, v in profile["eyes"].items() if k in ("url", "model")}}
    return Eyes(spec, registry) if spec else None
