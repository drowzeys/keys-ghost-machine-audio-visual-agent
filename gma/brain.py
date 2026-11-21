"""The brain behind one client interface, so the model can be swapped by the profile alone (a brain evaluation of
Nemotron 3.5 Bolt, Gemma 4 12B and Qwen3.8-27B runs beside this work: each is an OpenAI-compatible server, a URL and a
model name), and a split set-up can be said: a main brain for the words, a small multimodal model as its eyes (camera
turns, Ghost Machine's "eyes" section), and optionally ears (an audio-in model; used from phase 4).

Every client has the interface the conversation uses:

    async stream(messages, image_png=None) -> async iterator of text pieces
    async warm(messages)                    # a one-token request while the person still speaks
    async complete(messages, **settings) -> str   # one whole reply (the director lane)
    url, model, client, extra               # sides() and the video planner read these

A new kind of client registers itself in KINDS ("openai" is the default and the only one phase 1 needs).
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass

from gm_core.engine import Refusal
from voice_agent.engines import Brain

from .registry import Registry, engine_for_url

KINDS: dict = {"openai": Brain}


def make_client(spec: dict, registry: Registry | None = None, engine: str | None = None):
    """A brain client for one profile section. ``engine`` (or the section's own "engine", or the registry engine on
    the section's loopback port) is the registry engine whose lifecycle each request goes through."""

    kind = spec.get("kind", "openai")
    if kind not in KINDS:
        raise ValueError(f"unknown brain kind {kind!r} (known: {', '.join(sorted(KINDS))})")
    name = spec.get("engine") or engine or (engine_for_url(registry, spec.get("url")) if registry else None)
    use = (lambda: registry.use(name, wait=False)) if registry is not None and name else None
    return KINDS[kind](spec, use=use)


# messages of the conversation the fallback brain reads (the newest): the person's question alone. With the whole history
# its first audio went from ~0.7 s to ~2 s over 50 turns (prefilled beside the DiT pass); with the last six messages
# Cosmos3-Edge repeated the reply before ("Okay, I'll keep the film going.") to three questions in a row
FALLBACK_TURNS = 1


class HeldFallback:
    """The main brain, and a small one for the voice turns while the main one is held for a render (on a DGX Spark the
    governor releases Bolt for the DiT pass of a live render: with Kokoro and Parakeet resident, Bolt beside a 10 s
    864x480 pass would leave the machine under its 20 GB floor). A turn whose main brain is held is answered by the
    fallback (``brain_fallback`` in the profile: Cosmos3-Edge, already resident) instead of a "paused" line; the Director
    lane (``complete``) always waits for the main brain."""

    def __init__(self, main, fallback, away=None) -> None:
        self.main, self.fallback = main, fallback
        self.away = away or (lambda: False)  # True: the main brain is out for a render (held, or released by one)
        self.answered_by_fallback = 0

    def __getattr__(self, name):  # url, model, client, extra, use: the main brain's
        return getattr(self.main, name)

    async def warm(self, messages: list[dict]) -> None:
        if self.away():
            return await self.fallback.warm(messages)
        try:
            await self.main.warm(messages)
        except Refusal:
            await self.fallback.warm(messages)

    async def complete(self, messages: list[dict], **settings) -> str:
        return await self.main.complete(messages, **settings)

    async def stream(self, messages: list[dict], image_png: str | None = None):
        said = False
        if not self.away():  # never a 40 s load of the main brain in the middle of a render for one voice turn
            try:
                async with aclosing(self.main.stream(messages, image_png)) as stream:  # its request ends with the turn
                    async for piece in stream:
                        said = True
                        yield piece
                return
            except Refusal:  # held for the render (Held), or no room to load it now
                if said:
                    raise
        self.answered_by_fallback += 1
        system = [m for m in messages[:1] if m.get("role") == "system"]
        async with aclosing(self.fallback.stream(system + messages[len(system):][-FALLBACK_TURNS:], image_png)) as stream:
            async for piece in stream:
                yield piece


def director_spec(profile: dict) -> dict | None:
    """AutoPilot's Director brain, when it is not the main brain: ``autopilot.director`` names one of
    ``autopilot.director_choices`` (or is a brain section itself). None, "brain", a missing choice, or a choice on the main
    brain's own URL and model: the main brain directs (one engine, its lanes and its hold for a render).

        "autopilot": {"director": "qwen", "director_choices": {
            "qwen": {"engine": "qwen", "url": "http://127.0.0.1:8093/v1", "model": "qwen38-a"},
            "bolt": {"engine": "bolt", "url": "http://127.0.0.1:8091/v1", "model": "nvidia-nemotron-3.5-bolt"}}}

    The brain-quality bake-off picked Qwen3.8-27B (MLX 4-bit + DFlash2) as the Director: 13 of 13 plans valid, 12 of 13
    on the target length (Bolt 7), 8 of 8 films finished on the first try (Bolt 5), blinded plan preference 10-2."""

    ap = profile.get("autopilot") or {}
    spec = ap.get("director")
    if isinstance(spec, str):
        spec = None if spec == "brain" else (ap.get("director_choices") or {}).get(spec)
    if not spec:
        return None
    brain = profile.get("brain") or {}
    if spec.get("url") == brain.get("url") and spec.get("model", brain.get("model")) == brain.get("model"):
        return None
    return spec


@dataclass
class BrainSet:
    """The brain and what it may lean on: ``main`` answers every turn, ``eyes`` the turns about the camera (None: the
    main brain has none), ``ears`` an audio-in model (phase 4; a profile may name it now), ``director`` AutoPilot's
    Director when a profile gives it a brain of its own (None: the main brain directs; see ``director_spec``)."""

    main: object
    eyes: object | None = None
    ears: object | None = None
    director: object | None = None

    @property
    def split(self) -> bool:
        return self.eyes is not None or self.ears is not None

    def describe(self) -> dict:
        out = {"main": getattr(self.main, "model", None), "eyes": getattr(self.eyes, "model", None),
               "ears": getattr(self.ears, "model", None), "split": self.split}
        if self.director is not None:
            out["director"] = getattr(self.director, "model", None)
        return out


def make_brains(profile: dict, registry: Registry | None = None) -> BrainSet:
    brain = profile["brain"]
    main = make_client(brain, registry)
    if profile.get("brain_fallback"):  # voice turns while the main brain is held for a render (HeldFallback)
        name = brain.get("engine") or (engine_for_url(registry, brain.get("url")) if registry else None)

        def away() -> bool:
            one = registry.get(name) if registry is not None and name else None
            if one is None or one.loaded():
                return False
            return bool(one.hold) or getattr(registry, "rendering", lambda: False)()
        main = HeldFallback(main, make_client({**{k: v for k, v in brain.items() if k != "engine"}, **profile["brain_fallback"]}, registry), away)
    # the eyes inherit the brain's settings (Ghost Machine's {**brain, **eyes}), with their own URL and model
    eyes = make_client({**{k: v for k, v in brain.items() if k != "engine"}, **profile["eyes"]}, registry) if profile.get("eyes") else None
    ears = make_client({**{k: v for k, v in brain.items() if k != "engine"}, **profile["ears"]}, registry) if profile.get("ears") else None
    # the Director inherits the brain's settings too (each Director question overrides the length and the thinking)
    spec = director_spec(profile)
    director = make_client({**{k: v for k, v in brain.items() if k not in ("engine", "system")}, **spec}, registry) if spec else None
    return BrainSet(main, eyes, ears, director)
