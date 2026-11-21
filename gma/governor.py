"""The unified memory governor: one machine (a DGX Spark or a Mac), every engine on it, and the phases it moves
between.

- **talking**: the voice loop's engines load on demand; a load is admitted only when the machine keeps its floor after
  it (``admit``, called by every engine before it starts).
- **rendering**: a render (its memory request comes from the renderer, gma/renderer.py) is admitted by a schedule that
  says which engines stay resident beside it and which are evicted. ``plan`` works it out without acting; ``enter``
  evicts and grants, or refuses; ``leave`` lets the evicted engines load again.

The Spark schedules of the plan:
  "live"  a 480p clip from the FP4 clip worker (gma/clips.py: FastH3 V2 on FastVideo's FP4 sparse recipe, its whole
          process ~65-70 GB at its peak): Kokoro, Parakeet and Cosmos3-Edge stay resident beside it, and the worker
          itself between takes; the brain is released for the clip (brain_during_dit) and loads again after it.
  "sharp" "go sharper": 1376x768, ~10 s (243 frames); the brain is evicted too.
  "keyframe" a keyframe's picture model (~30 GB, the Studio's ComfyUI): AutoPilot's judge stays resident, the brain
          where it fits; the clip worker is released (it loads again for the clip).

A clip's need is the worker's measured peak less what the worker already holds (gma/studio.py memory_request); a
keyframe's and the film's sound's are the profile's stated figures. When a voice engine cannot load beside an idle clip
worker, the worker gives way (``admit``). The available memory is MemAvailable on a Spark and vm_stat on a Mac
(gm_core/memstat.py).

Every refusal is a ``Refusal``: the whole reason for the panel and the log, and one short line for the voice.
"""

from __future__ import annotations

import asyncio
import time

from gm_core.engine import Engine, Held, Refusal
from gm_core.memstat import MemReader

from . import memguard
from .registry import Registry

RENDERER = "renderer"  # the role of the render server (the headless Studio): never evicted, it is the render
MAC_PEAK_GB = {"video": 33.0, "keyframe": 40.0}  # an M5 Ultra's measured peaks (the Studio's app/bolt.py MAC_PEAK_GB)


class StatedBudget:
    """A render's need without the renderer to ask (tests, and a governor with no renderer): the profile's
    {"need_gb", "error_gb"}, or the FP4 clip worker's peaks measured on a DGX Spark (gma/clips.py NEED_MB).
    ``preset: "fh8-480"`` is 864x480 at the render's frames (124 unless said)."""

    name = "budget"
    label = "renderer memory estimate (stated)"

    def __init__(self, platform: str, spec: dict | None = None) -> None:
        self.platform, self.spec = platform, spec or {}

    def memory_request(self, render: dict) -> dict:
        if render.get("need_gb"):
            return {"need_mb": int(float(render["need_gb"]) * 1024), "error_mb": int(float(render.get("error_gb", 0)) * 1024),
                    "route": "stated", "rows": None, "source": "profile"}
        if self.platform == "mac":
            kind = render.get("kind", "video")
            return {"need_mb": int(MAC_PEAK_GB.get(kind, MAC_PEAK_GB["video"]) * 1024), "error_mb": 4096, "route": kind, "rows": None,
                    "source": "Studio MAC_PEAK_GB (M5 Ultra)"}
        from . import clips
        width, height = int(render.get("width") or 864), int(render.get("height") or 480)
        frames = int(render.get("frames") or 124)
        return {"need_mb": clips.need_mb(width, height, frames), "error_mb": clips.ERROR_MB, "route": "fp4", "rows": None,
                "frames": frames, "source": "the FP4 clip worker's peak, measured (gma/clips.py)"}

    def floor_mb(self) -> int | None:
        return None

    def free_cache(self) -> dict:
        return {"freed": False, "why": "no renderer"}


RendererStub = StatedBudget
DIT_RELEASE_WAIT_S = 10.0  # the most a brain released for the DiT pass waits for its answers in flight

# the Spark profile's schedules, by engine name (profiles/spark.example.json uses these names)
SPARK_SCHEDULES = {
    "live": {"render": {"preset": "fh8-480"}, "keep": ["bolt", "kokoro", "asr", "cosmos", "fp4"],
             "label": "480p live + Kokoro",
             "fallbacks": [{"step": "asr_to_cpu", "frees_gb": 3.0}, {"step": "brain_during_dit"}]},
    "sharp": {"render": {"width": 1376, "height": 768, "frames": 243}, "keep": ["kokoro", "asr", "fp4"],
              "label": "sharp 1376x768, 10 s (brain evicted)"},
    # a keyframe (FLUX 3 Klein or Qwen-Image): ~30 GB measured on a Spark (84.4 -> 54.9 GB available). AutoPilot's judge
    # (d1-2B, ~5.5 GB) stays; Gemma 4 12B is no longer in the default schedule (d1-2B is the only judge since
    # 2026-10-29). Bolt stays where it fits (the opening keyframe, drawn while the Director plans) and goes where it does
    # not: the phase 4 smoke had 49 GB for a 31 +- 4 GB picture with Bolt and Gemma resident, under the floor. Bolt is
    # loaded again for the next Director while the judge checks the clip
    "keyframe": {"render": {"need_gb": 31, "error_gb": 4}, "keep": ["bolt", "kokoro", "asr", "cosmos", "d1"],
                 "fallbacks": [{"step": "evict", "engine": "bolt"}],
                 "label": "keyframe (the judge stays; the brain too where it fits)"},
}

SAID = {
    "no_room": "There isn't enough memory to render that right now.",
    "no_room_sharp": "I can't go sharper right now. There isn't enough memory.",
    "adopted": "Part of me runs outside my control, so I can't free the memory for that.",
    "load": "I don't have the memory to start my {role} right now.",
    "paused": "My {role} is paused while I render. Give me a moment.",
    "busy": "I'm already rendering. Let me finish this one first.",
}


class Governor:
    def __init__(self, spec: dict | None, registry: Registry, reader: MemReader | None = None,
                 renderer=None, platform: str | None = None) -> None:
        self.spec = spec or {}
        self.registry = registry
        self.reader = reader or registry.reader
        self.platform = platform or self.reader.platform
        self.renderer = renderer or StatedBudget(self.platform, self.spec.get("renderer"))
        self.schedules = self.spec.get("schedules") or (SPARK_SCHEDULES if self.platform != "mac" else {})
        self.default = self.spec.get("default_schedule") or next(iter(self.schedules), None)
        self.phase = "talking"
        self.schedule: str | None = None
        self.grant: dict | None = None
        self.evicted: list[str] = []
        self.refusals: list[dict] = []
        self.lock = asyncio.Lock()
        self.hooks: dict = {}  # "asr_to_cpu": f(to_cpu: bool), set by the app when its recognizer can move
        self.asr_on_cpu = False
        self.dit_evict: list[str] = []
        self.cache_freed: dict | None = None
        self.in_dit = False
        self.guard = None  # the live memory guard (gma/memguard.py), set by the supervisor
        self.cooling: dict | None = None  # the thermal gate's wait now
        self.thermal_waits: list[dict] = []
        self.released: list[dict] = []  # idle models the admission released for another's load

    # -- the machine --

    def total_mb(self) -> int:
        return self.reader.total_mb()

    def floor_mb(self) -> int:
        if self.spec.get("floor_gb") is not None:
            return int(float(self.spec["floor_gb"]) * 1024)
        if self.platform == "mac":
            return max(16 * 1024, self.total_mb() // 10)
        said = getattr(self.renderer, "floor_mb", lambda: None)()
        if said:
            return int(said)  # the Studio's own floor, from its estimates
        return max(8000, self.total_mb() // 10)  # its rule (spark/memory_guard.py floor_mb): a tenth, never under 8 GB

    def thermal_gate(self, what: str) -> dict:
        """The thermal gate (gma/memguard.py thermal_gate) before a render or a big load: it waits for the GPU to cool,
        says so on the run's status line (hooks["say"]), logs every wait, and refuses when it stays too hot."""

        log = self.guard.log if self.guard is not None else None
        say = self.hooks.get("say")
        self.cooling = {"what": what, "since": time.time()}
        try:
            record = memguard.thermal_gate(self.spec, what, log=log, say=say)
        except memguard.TooHot as error:
            self.cooling = None
            raise self.refuse(str(error), "The machine is too hot to start that; it needs to cool down first.") from None
        self.cooling = None
        if record["waited_s"]:
            self.thermal_waits.append({"at": time.time(), **record})
        return record

    def refuse(self, detail: str, said: str) -> Refusal:
        self.refusals = (self.refusals + [{"at": time.time(), "detail": detail, "said": said}])[-10:]
        return Refusal(detail, said)

    # -- talking: a load --

    def kept(self, engine) -> bool:
        if self.phase != "rendering" or not self.schedule or engine.role == RENDERER:
            return True  # the renderer's own server is the render: never evicted by it
        keep = self.schedules[self.schedule].get("keep", [])
        return engine.name in keep or engine.role in keep

    def admit(self, engine: Engine) -> None:
        """Before an engine loads (called by the engine, its mutex held): refused while a render that evicts it runs,
        or when the load would leave the machine under its floor."""

        role = engine.role
        if engine.hold or not self.kept(engine):
            raise Held(f"{engine.label} may not load while rendering ({self.schedule})", SAID["paused"].format(role=role))
        if engine.need_gb <= 0:
            return
        if engine.need_gb >= float(memguard.thermal_options(self.spec)["big_load_gb"]) and self.reader.platform != "mac":
            self.thermal_gate(f"loading {engine.label}")
        free = self.reader.available_mb()
        if free < 0:
            return  # unknown: the kill-guard still stands under it
        need = int(engine.need_gb * 1024)
        floor = self.floor_mb()
        if free - need < floor and self.phase == "talking":
            # a renderer that kept its models after a render holds the room (on a Spark Bolt could not reload beside
            # ComfyUI's ~50 GB cache, status -6): free it, then look again
            self.cache_freed = self.renderer.free_cache()
            if self.cache_freed.get("freed"):
                self.reader.settle()
                free = self.reader.available_mb()
        if free - need < floor and self.phase == "talking" and engine.role != "clips":
            # the clip worker idle between takes gives way to a voice engine (it loads again for the next clip)
            for one in self.registry.by_role("clips"):
                if one.loaded() and not one.in_flight and not one.adopted and one.release(f"{engine.label} needs the room", 0):
                    one.unhold("it gave way")
                    self.reader.settle()
                    free = self.reader.available_mb()
        if free - need < floor:
            # the memory guard's admission: this app's other idle models go first, biggest first (never one answering)
            for one in memguard.idle_engines(self.registry, {engine.name}):
                if free - need >= floor:
                    break
                if one.release(f"{engine.label} needs the room", 0):
                    one.unhold("it gave way; it loads again when admitted")
                    self.released.append({"at": time.time(), "engine": one.name, "for": engine.name})
                    self.reader.settle()
                    free = self.reader.available_mb()
        wait_s = float(memguard.options(self.spec)["admit_wait_s"])
        ends = time.time() + wait_s
        while free - need < floor and time.time() < ends:
            # what is left is not this app's (another server stepping aside, a cache being reclaimed): wait for it
            time.sleep(2.0)
            free = self.reader.available_mb()
        if free - need < floor:
            users = memguard.named(memguard.biggest(4))
            raise self.refuse(f"{engine.label} needs about {need / 1024:.0f} GB to load; this machine has {free / 1024:.0f} GB "
                              f"available and keeps {floor / 1024:.0f} GB free as its floor; the biggest users: {users}",
                              SAID["load"].format(role=role))

    # -- rendering --

    def plan(self, name: str | None = None, frames: int | None = None) -> dict:
        """What a render of schedule ``name`` would take and give back, and whether it fits; nothing is changed.

        The fit is explicit: ``budget`` holds the machine's available memory now, what the evicted engines give back,
        the render's need and error, the floor and what is left. When what is left is under the floor, the schedule's
        ``fallbacks`` are tried in order until it is not (``asr_to_cpu``: the in-process recognizer moves to the CPU;
        ``evict``: one more engine is released; ``brain_during_dit``: the brain is released for the DiT pass, from its
        start until the render ends and the renderer's cache is freed)."""

        name = name or self.default
        if name not in self.schedules:
            raise self.refuse(f"no render schedule {name!r} in this profile", "I don't know that render setting.")
        schedule = self.schedules[name]
        render = dict(schedule.get("render", {}))
        if frames:
            render["frames"] = int(frames)  # the clip's own length (a 10 s AutoPilot piece is 243 frames)
        request = self.renderer.memory_request(render)
        keep = set(schedule.get("keep", []))
        evict, blockers, giveback = [], [], 0
        renderer_mb = 0
        for one in self.registry.engines.values():
            if one.role == RENDERER:
                if one.spec.get("counts_in_render", True):
                    renderer_mb += one.held_mb()  # what the renderer holds now is part of its own need, not extra
                continue
            if one.name in keep or one.role in keep or not one.loaded():
                continue
            if getattr(one, "mode", "") == "inprocess":
                continue
            if one.adopted:
                blockers.append(one.name)  # running outside this app: its memory stays taken
                continue
            evict.append(one.name)
            giveback += one.held_mb()
        available = self.reader.available_mb()
        floor = self.floor_mb()
        out = {"schedule": name, "label": schedule.get("label", name), "request": request, "available_mb": available,
               "giveback_mb": giveback, "floor_mb": floor, "evict": evict, "adopted_blockers": blockers,
               "keep": sorted(keep), "platform": self.platform, "fallbacks": [], "dit_evict": []}
        if available < 0:
            out.update(fits=False, reason="this machine's available memory cannot be read", left_mb=None)
            return out
        # what the renderer's own server holds (the Studio's ComfyUI between jobs) is part of its need, not extra
        renderer_mb += int(getattr(self.renderer, "held_mb", lambda: 0)() or 0)
        out["renderer_held_mb"] = renderer_mb
        room = available + giveback + renderer_mb
        left = room - request["need_mb"] - request["error_mb"]
        reason = (f"this render needs about {request['need_mb'] / 1024:.0f} GB at its peak (give or take "
                  f"{request['error_mb'] / 1024:g}); this machine has {room / 1024:.0f} GB for it and keeps "
                  f"{floor / 1024:.0f} GB free as its floor")
        budget = {"available_mb": available, "giveback_mb": giveback, "need_mb": request["need_mb"],
                  "error_mb": request["error_mb"], "floor_mb": floor, "left_before_fallbacks_mb": left}
        for step in schedule.get("fallbacks", []) if left < floor else []:
            frees = self.fallback_frees(step, evict, blockers)
            if frees is None:
                continue
            out["fallbacks"].append({**step, "frees_mb": frees})
            left += frees
            if step["step"] == "evict":
                evict.append(step["engine"])
            elif step["step"] == "brain_during_dit":
                out["dit_evict"].append(self.brain_name(step))
            if left >= floor:
                break
        budget["left_mb"] = left
        fits = left >= floor
        out.update(fits=fits, left_mb=left, budget=budget, reason="" if fits else reason)
        if not fits and blockers:
            out["reason"] += (f". Running outside this app and not evicted: {', '.join(blockers)}")
        if not fits:
            out["reason"] += ". The biggest memory users: " + memguard.named(memguard.biggest(4))
        return out

    def brain_name(self, step: dict) -> str | None:
        if step.get("engine"):
            return step["engine"]
        found = [e.name for e in self.registry.by_role("brain")]
        return found[0] if found else None

    def fallback_frees(self, step: dict, evict: list, blockers: list) -> int | None:
        """MB a fallback step gives back now, or None when it does not apply here."""

        kind = step.get("step")
        if kind == "asr_to_cpu":
            asr = next((e for e in self.registry.by_role("asr") if e.loaded()), None)
            if asr is None or "asr_to_cpu" not in self.hooks or self.asr_on_cpu:
                return None
            return int(float(step.get("frees_gb", 0)) * 1024)
        name = step.get("engine") if kind == "evict" else self.brain_name(step) if kind == "brain_during_dit" else None
        one = self.registry.get(name)
        if one is None or not one.loaded() or name in evict or getattr(one, "mode", "") == "inprocess":
            return None
        if one.adopted:
            if name not in blockers:
                blockers.append(name)
            return None
        return one.held_mb()

    async def enter(self, name: str | None = None, frames: int | None = None) -> dict:
        """Rendering with schedule ``name``: the engines it evicts are released (after their answers in flight) and
        held, its fallbacks applied; a Refusal when it does not fit or a render runs already."""

        async with self.lock:
            if self.phase == "rendering":
                raise self.refuse(f"a render with schedule {self.schedule} runs already", SAID["busy"])
            found = await self.settled(name, frames)
            if not found["fits"]:
                said = SAID["no_room_sharp"] if found["schedule"] == "sharp" else SAID["no_room"]
                if found["adopted_blockers"] and found["available_mb"] >= 0:
                    said = SAID["adopted"]
                raise self.refuse(found["reason"], said)
            self.phase, self.schedule, self.evicted = "rendering", found["schedule"], []
            self.dit_evict = [n for n in found["dit_evict"] if n]
            why = f"a render ({found['label']})"
            try:
                for name_ in found["evict"]:
                    if await asyncio.to_thread(self.registry[name_].release, why):
                        self.evicted.append(name_)
                if any(s["step"] == "asr_to_cpu" for s in found["fallbacks"]):
                    try:
                        await asyncio.to_thread(self.hooks["asr_to_cpu"], True)
                        self.asr_on_cpu = True
                    except Exception as error:  # noqa: BLE001 - the recognizer stays; the brain goes for the DiT instead
                        # measured on the DGX Spark (2026-11-07): Parakeet's move raised a CUDA illegal-address error mid-film
                        found["asr_move_failed"] = str(error)[:200]
                        brain = self.brain_name({})
                        if brain and brain not in self.dit_evict and self.registry.get(brain) is not None \
                                and self.registry[brain].loaded() and not self.registry[brain].adopted:
                            self.dit_evict.append(brain)
                # engines that are down now must not load during the render either
                for one in self.registry.engines.values():
                    if isinstance(one, Engine) and not self.kept(one) and not one.adopted:
                        one.hold = one.hold or why
                await asyncio.to_thread(self.reader.settle)
            except BaseException:
                # never left in the rendering phase by a step that failed: every hold is let go again
                for one in self.registry.engines.values():
                    one.unhold("the render's start failed")
                self.phase, self.schedule, self.grant, self.evicted, self.dit_evict = "talking", None, None, [], []
                raise
            self.grant = {**found, "evicted": list(self.evicted), "available_after_mb": self.reader.available_mb()}
            return self.grant

    async def settled(self, name: str | None, frames: int | None) -> dict:
        """The plan, asked again while it needs a fallback (or does not fit) and the machine's available memory is
        still rising: what a render before this one gave back arrives over seconds. Measured on the DGX Spark (2026-11-19): a
        clip planned 1 s after its keyframe's server stopped read 65 GB of the 90 GB that were there 2 s later, and the
        brain was released for the whole rehearsal for nothing."""

        found = self.plan(name, frames)
        step, ends = float(self.spec.get("settle_step_s", 1.5)), time.monotonic() + float(self.spec.get("settle_s", 15))
        still = 0
        while (found["fallbacks"] or not found["fits"]) and found["available_mb"] >= 0 and step > 0 and time.monotonic() < ends:
            await asyncio.sleep(step)
            again = self.plan(name, frames)
            still = 0 if again["available_mb"] - found["available_mb"] >= 512 else still + 1
            found = again
            if still >= 2:  # two readings in a row with nothing more coming back
                break
        return found

    def rendering(self) -> bool:
        return self.phase == "rendering"

    async def dit(self, starting: bool) -> None:
        """The renderer's word at the start (and the end) of its DiT pass: a brain_during_dit fallback releases the
        brain at the start; it stays held until the render ends and the renderer's cache is freed (leave), since the
        decoders that follow the pass still need the room."""

        self.in_dit = bool(starting) and self.phase == "rendering"
        if starting and self.phase == "rendering":
            for name in self.dit_evict:
                # the render already holds its memory: the brain's answers in flight get a few seconds, not ten minutes
                # (on the phase 4 Spark latency test a request left "in flight" kept Bolt beside the DiT pass, the
                # machine sat at 15 GB, and the engines' own guards killed ComfyUI and Kokoro)
                if await asyncio.to_thread(self.registry[name].release, f"the DiT pass ({self.schedule})", DIT_RELEASE_WAIT_S):
                    self.evicted.append(name)

    async def leave(self, why: str = "rendering ended") -> None:
        """Rendering ends: the renderer's model cache is freed first (a brain could not reload beside ComfyUI's ~50 GB
        cache on a Spark), the recognizer goes back to the GPU, then the evicted engines may load again."""

        async with self.lock:
            if self.phase == "rendering":
                self.cache_freed = await asyncio.to_thread(self.renderer.free_cache)
                if self.asr_on_cpu:
                    await asyncio.to_thread(self.hooks["asr_to_cpu"], False)
                    self.asr_on_cpu = False
            for one in self.registry.engines.values():
                one.unhold(why)
            self.phase, self.schedule, self.grant, self.evicted, self.dit_evict = "talking", None, None, [], []
            self.in_dit = False

    def status(self) -> dict:
        return {"phase": self.phase, "schedule": self.schedule, "default_schedule": self.default,
                "asr_on_cpu": self.asr_on_cpu, "dit_evict": self.dit_evict, "in_dit": self.in_dit, "cache_freed": self.cache_freed,
                "schedules": {k: v.get("label", k) for k, v in self.schedules.items()}, "platform": self.platform,
                "available_mb": self.reader.available_mb(), "total_mb": self.total_mb(), "floor_mb": self.floor_mb(),
                "grant": self.grant, "renderer": self.renderer.label, "refusals": self.refusals[-3:], "released": self.released[-5:],
                "memguard": self.guard.status() if getattr(self, "guard", None) is not None else None,
                "cooling": self.cooling, "thermal_waits": {"count": len(self.thermal_waits), "seconds": round(sum(w["waited_s"] for w in self.thermal_waits), 1),
                                                           "last": self.thermal_waits[-3:]},
                # the render stopped for voice turns (gma/studio.py hold_for_voice): how often, and for how long
                "voice_holds": len(getattr(self.renderer, "voice_pauses", None) or []),
                "voice_hold_s_p50": sorted(getattr(self.renderer, "voice_pauses", None) or [0])[len(getattr(self.renderer, "voice_pauses", None) or [0]) // 2]}
