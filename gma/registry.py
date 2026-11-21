"""The engine registry: every child engine of the app by name, and the roles they fill.

A profile's "engines" section names them (see profiles/*.example.json):

    "engines": {
      "bolt":   {"role": "brain",  "mode": "managed", "port": 8091, "argv": [...], "need_gb": 43, "resident_gb": 31},
      "kokoro": {"role": "voice",  "mode": "adopt",   "port": 8197, "resident_gb": 2},
      "asr":    {"role": "asr",    "mode": "inprocess", "resident_gb": 2.5}
    }

Without one, the engines are read from the brain, voice, voice_lite and eyes sections as servers to adopt (Ghost
Machine's own behaviour: everything already runs at the URLs the profile names).

``use(name)`` is the voice path's way in: an async context that counts a request in flight on the engine (no release
cuts it), loading the engine first in a worker thread when it is not up; when it is up this is a lock and two adds.
A monitor thread ticks every engine every ``monitor_s`` seconds: idle unloads, ended servers, adopted servers that went
away.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from urllib.parse import urlparse

from gm_core.engine import Engine, InProcess
from gm_core.memstat import MemReader


def derived_engines(profile: dict) -> dict:
    """Engines to adopt, from the URLs of a profile without an "engines" section."""

    out: dict = {}
    for section, role in (("brain", "brain"), ("voice", "voice"), ("voice_lite", "voice"), ("eyes", "vision")):
        spec = profile.get(section)
        if not spec or not spec.get("url"):
            continue
        url = urlparse(spec["url"])
        if url.hostname not in ("127.0.0.1", "localhost", "::1") or not url.port:
            continue  # not a child of this machine: talked to, never supervised
        name = f"{role}@{url.port}"
        if any(e.get("port") == url.port for e in out.values()):
            continue
        out[name] = {"role": role, "mode": "adopt", "port": url.port, "label": spec.get("model", name)}
    asr = profile.get("asr")
    if asr:
        out["asr"] = {"role": "asr", "mode": "inprocess", "label": asr.get("model", "asr"), "resident_gb": 2.5}
    return out


def engine_for_url(registry: "Registry", url: str | None) -> str | None:
    """The registry's engine serving ``url`` (by loopback port), if any."""

    if not url:
        return None
    parsed = urlparse(url)
    for name, one in registry.engines.items():
        if isinstance(one, Engine) and parsed.port and one.fixed_port == parsed.port and parsed.hostname in (
                "127.0.0.1", "localhost", "::1"):
            return name
    return None


class Registry:
    def __init__(self, profile: dict, reader: MemReader | None = None, admit=None, monitor_s: float = 5.0,
                 log_dir: str | None = None) -> None:
        self.reader = reader or MemReader(profile.get("platform"))
        specs = profile.get("engines") or derived_engines(profile)
        self.engines: dict = {}
        for name, spec in specs.items():
            if spec.get("mode") == "inprocess":
                self.engines[name] = InProcess(name, spec)
            else:
                self.engines[name] = Engine(name, spec, self.reader, admit=admit, log_dir=log_dir)
        self.monitor_s = monitor_s
        self._monitor: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lookup --

    def __getitem__(self, name: str):
        return self.engines[name]

    def get(self, name: str | None):
        return self.engines.get(name) if name else None

    def by_role(self, role: str) -> list:
        return [e for e in self.engines.values() if e.role == role]

    def set_admit(self, admit) -> None:
        for one in self.engines.values():
            if isinstance(one, Engine):
                one.admit = admit

    # -- the voice path --

    @contextlib.asynccontextmanager
    async def use(self, name: str | None, wait: bool = True):
        """One request on engine ``name`` (None, or an in-process engine: nothing to count)."""

        one = self.engines.get(name) if name else None
        if not isinstance(one, Engine):
            yield None
            return
        if not one.try_fast():
            # a load can take a minute: a turn cancelled meanwhile (the person spoke again) must not leave the request
            # counted, or no release ever finds the engine idle (on the phase 4 Spark latency test two turns cut
            # while Bolt loaded kept it "in flight", resident beside the DiT pass)
            loading = asyncio.ensure_future(asyncio.to_thread(one.acquire, lambda: False, wait))
            try:
                await asyncio.shield(loading)
            except asyncio.CancelledError:
                loading.add_done_callback(lambda done: done.cancelled() or done.exception() is not None or one.release_request())
                raise
        try:
            yield one.base()
        finally:
            one.release_request()

    # -- lifecycle --

    def adopt_running(self) -> dict:
        """At start-up: every engine whose server already answers on its port is adopted (never a duplicate)."""

        found = {}
        for name, one in self.engines.items():
            if isinstance(one, Engine):
                with one.mutex:
                    found[name] = one.loaded() or one.adopt()
        return found

    def tick(self) -> None:
        for one in list(self.engines.values()):
            try:
                one.tick()
            except Exception:  # noqa: BLE001 - one engine's trouble never stops the others' ticks
                pass

    def start_monitor(self) -> None:
        if self._monitor is None:
            self._stop.clear()

            def run() -> None:
                while not self._stop.wait(self.monitor_s):
                    self.tick()

            self._monitor = threading.Thread(target=run, name="engine-monitor", daemon=True)
            self._monitor.start()

    def shutdown(self) -> None:
        """At the app's end: the servers started here end (adopted ones keep running)."""

        self._stop.set()
        for one in self.engines.values():
            if isinstance(one, Engine) and one.process is not None:
                one.stop("the app is stopping")

    def status(self) -> dict:
        return {"engines": [one.status() for one in self.engines.values()], "at": time.time()}
