"""The supervisor: one app process that holds the engine registry, the memory governor and the brain lanes, and the
HTTP routes that show and steer them.

    GET  /api/engines                     every engine: state, adopted or started here, memory, in flight
    POST /api/engines/{name}/load         load (or adopt) it now
    POST /api/engines/{name}/unload       unload it (refused while it answers; an adopted server is only let go)
    GET  /api/governor                    phase, schedule, the machine's memory, recent refusals
    GET  /api/governor/plan?schedule=s    what a render of that schedule would take and evict (nothing changes)
    POST /api/governor/phase              {"phase": "rendering", "schedule": "live"|"sharp"} or {"phase": "talking"}
    GET  /api/lanes                       the brain lanes
    GET  /api/credits                     what this app is built from (the page's credits)
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gm_core.engine import Engine, Refusal
from gm_core.memstat import MemReader

from .credits import CREDITS
from .governor import Governor
from .lanes import Lanes
from .registry import Registry
from .studio import HeadlessRenderer, studio_engine


class Supervisor:
    def __init__(self, profile: dict, reader: MemReader | None = None, busy=None) -> None:
        self.profile = profile
        self.reader = reader or MemReader(profile.get("platform"))
        sup = profile.get("supervisor") or {}
        studio = profile.get("studio")
        if studio and studio.get("mode", "managed") != "external":
            # the headless Imagination Studio: a child engine of this app (or adopted), the render side of everything
            engines = dict(profile.get("engines") or {})
            engines.setdefault(studio.get("engine", "studio"), studio_engine(studio, self.reader.platform))
            profile = {**profile, "engines": engines}
            self.profile = profile
        self.registry = Registry(profile, self.reader, monitor_s=float(sup.get("monitor_s", 5)), log_dir=sup.get("home"))
        gov = profile.get("governor") or {}
        self.renderer = None
        if (profile.get("video") or {}).get("renderer") in ("studio-headless", "local"):
            # the headless Studio's queue: the governor's memory view and the video panel's render host in one
            spec = {**(studio or {}), **(gov.get("renderer") or {}), **((profile.get("video") or {}).get("local") or {})}
            spec.setdefault("engine", (studio or {}).get("engine", "studio"))
            self.renderer = HeadlessRenderer(self.reader.platform, spec, registry=self.registry, reader=self.reader)
        self.governor = Governor(gov, self.registry, self.reader, renderer=self.renderer)
        if self.renderer is not None:
            self.renderer.governor = self.governor
        self.registry.set_admit(self.governor.admit)
        self.registry.rendering = self.governor.rendering  # a brain fallback answers while a render has the brain out
        self.lanes = Lanes(busy=busy)
        self.adopted: dict = {}
        # the live memory guard (gma/memguard.py): on where the profile's governor has "memguard" (the Spark profiles do)
        from pathlib import Path
        from . import memguard
        guard_spec = gov.get("memguard") or {}
        self.guard = None
        if guard_spec and guard_spec.get("live", True) and self.reader.platform != "mac":
            home = Path(sup.get("home") or "~/.gma").expanduser() / "logs"
            self.guard = memguard.LiveGuard(self.registry, self.reader, gov, home, stop_step=self.stop_step)
        self.governor.guard = self.guard

    def start(self) -> dict:
        """At the app's start: running servers adopted (never started twice), the monitor ticking."""

        self.adopted = self.registry.adopt_running()
        self.registry.start_monitor()
        if self.guard is not None:
            self.guard.start()
        studio = self.profile.get("studio") or {}
        one = self.registry.get(studio.get("engine", "studio")) if studio.get("start_at_launch", True) else None
        if isinstance(one, Engine) and not one.loaded():
            # the headless Studio starts with the app (about 2 s), so the first film's keyframe does not wait for it
            def warm() -> None:
                try:
                    one.acquire()
                    one.release_request()
                except Exception:  # noqa: BLE001 - a render starts it again, and says why when it cannot
                    pass
            import threading
            threading.Thread(target=warm, name="studio-start", daemon=True).start()
        return self.adopted

    def stop_step(self, why: str) -> str:
        """The memory guard's last step: the step at work is stopped cleanly. The clip worker (the biggest thing a
        step loads) is ended, so its clip fails with this reason and the run stops on it instead of the kernel's
        driver running out of memory."""

        ended = []
        for one in self.registry.by_role("clips"):
            if isinstance(one, Engine) and one.loaded() and not one.adopted:
                one.stop(why)
                ended.append(one.name)
        return f"ended {', '.join(ended)}" if ended else "no step to stop"

    def shutdown(self) -> None:
        if self.guard is not None:
            self.guard.stop()
        self.registry.shutdown()

    def routes(self, app: FastAPI) -> None:
        def refused(error: Refusal, code: int = 409):
            return JSONResponse({"refused": True, "error": str(error), "said": error.said}, status_code=code)

        @app.get("/api/engines")
        def engines():
            return JSONResponse(self.registry.status())

        @app.post("/api/engines/{name}/load")
        async def load(name: str):
            one = self.registry.get(name)
            if one is None:
                return JSONResponse({"error": f"no engine {name}"}, status_code=404)
            if not isinstance(one, Engine):
                return JSONResponse(one.status())
            try:
                await asyncio.to_thread(lambda: one.release_request() if one.acquire(wait=False) else None)
            except Refusal as error:
                return refused(error)
            except Exception as error:  # noqa: BLE001
                return JSONResponse({"error": str(error)[:400]}, status_code=500)
            return JSONResponse(one.status())

        @app.post("/api/engines/{name}/unload")
        async def unload(name: str):
            one = self.registry.get(name)
            if not isinstance(one, Engine):
                return JSONResponse({"error": f"no engine {name} to unload"}, status_code=404)
            if one.in_flight:
                return JSONResponse({"error": f"{one.label} is answering now"}, status_code=409)
            await asyncio.to_thread(one.stop, "unloaded from the page")
            return JSONResponse(one.status())

        @app.get("/api/governor")
        def governor():
            return JSONResponse(self.governor.status())

        @app.get("/api/governor/plan")
        def plan(schedule: str | None = None):
            try:
                return JSONResponse(self.governor.plan(schedule))
            except Refusal as error:
                return refused(error, 400)

        @app.post("/api/governor/phase")
        async def phase(request: Request):
            try:
                body = await request.json()
            except ValueError:
                body = {}
            body = body if isinstance(body, dict) else {}
            if body.get("phase") == "talking":
                await self.governor.leave("asked from the page")
                return JSONResponse(self.governor.status())
            if body.get("phase") != "rendering":
                return JSONResponse({"error": "phase is rendering or talking"}, status_code=400)
            try:
                return JSONResponse(await self.governor.enter(body.get("schedule")))
            except Refusal as error:
                return refused(error)

        @app.get("/api/lanes")
        def lanes():
            return JSONResponse(self.lanes.status())

        @app.get("/api/credits")
        def credits():
            return JSONResponse(CREDITS)
