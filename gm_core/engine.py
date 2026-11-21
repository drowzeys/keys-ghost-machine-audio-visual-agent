# Ported from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman), app/bolt.py at
# 912374e1b6111fdab5d55947cc695c9c303f83b7 (Engine, Held, Unloaded, own_environment, free_port, idle_seconds,
# group_rss_gb, kill_group). Apache-2.0: LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and
# LICENSES/Imagination-Studio-NOTICE.
# Changed: one Engine for any child server (brain, voice, vision), its command line, port, health path and memory
# figures taken from the profile instead of Bolt's backends; an "adopt" mode, and adoption of a server already
# answering on the port, instead of starting a duplicate (an adopted server is never stopped or unloaded here); the
# idle unload and the dead-process check are one tick() the registry calls, instead of a thread an engine; the
# memory check before a load is the governor's (admit), which refuses with a short spoken line (Refusal); logs under
# the app's own folder. The leash (gm_core/leash.py) and the kill-guard are the Studio's. An optional "stop_argv" (a
# server in a container: ``docker rm -f <name>``) runs before a start and after a stop, so no container outlives it;
# it runs with the engine's own environment (the headless Studio's ComfyUI port), and "stop_wait_s" sets how long a
# stop waits for the server to end by itself (the Studio stops its ComfyUI first).
"""A child engine of the supervisor: an HTTP server on a loopback port, run under the leash and the kill-guard,
loaded on demand, unloaded when idle or when a render needs its memory; or a server that already runs, adopted.

States: ``unloaded`` -> ``loading`` -> ``loaded`` -> ``unloaded``; ``adopted`` while a server this app did not start
answers on the port; ``held`` while a render has the memory (no request may load it).

- a request (``request``) loads it first when it is not loaded, and is counted in flight, so no release cuts it;
- ``release(why)`` holds it (no new request starts), waits for the requests in flight, then unloads it;
- ``tick()`` (every few seconds, from the registry): a server that ended by itself is noted, and one idle for
  ``idle_s`` with nothing in flight is unloaded;
- the kill-guard reads the available memory and kills the engine's process group under ``guard_gb``;
- it dies with the app: the leash ends it when the app's pipe closes.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .memstat import MemReader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
HEALTH_TIMEOUT_S = 900  # the most a load may take
RELEASE_WAIT_S = 600  # the most a release waits for an answer in flight
LOG_BYTES = 50 * 1024 ** 2


class Refusal(RuntimeError):
    """Why something cannot happen now. ``str(error)`` is the whole reason (the panel, the log); ``said`` is the short
    line the voice speaks."""

    def __init__(self, detail: str, said: str | None = None) -> None:
        super().__init__(detail)
        self.said = said or (detail if len(detail) <= 90 else "I can't do that right now. The panel has the details.")


class Held(Refusal):
    """The engine is unloaded while a render has its memory: a request that may not wait is refused with this."""


class Unloaded(RuntimeError):
    """A load stopped because a render took the memory meanwhile (the request waits for it, then loads)."""


def own_environment() -> dict:
    """The app's environment for the engine, without the app's own address (a server that reads PORT or HOST would
    take the app's)."""

    return {k: v for k, v in os.environ.items() if k not in ("PORT", "HOST")}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def idle_seconds(text) -> float:
    """Seconds, or ``never`` / 0 for an engine kept until the app ends."""

    text = str(text if text is not None else "").strip().lower()
    if text in ("never", "off", "0", "inf"):
        return 0.0
    try:
        return max(0.0, float(text)) if text else 600.0
    except ValueError:
        return 600.0


def group_rss_gb(group: int) -> float | None:
    """The resident memory of a process group (the leash and its server), GB; None when ps cannot say."""

    try:
        said = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    total = 0
    for line in said.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == str(group) and parts[1].isdigit():
            total += int(parts[1])
    return total / 1024 ** 2 if total else None


def kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def expand(argv: list, port: int) -> list[str]:
    """The profile's command line: ``{port}``, ``{python}`` (this interpreter) and ``{repo}`` filled in, ``~`` expanded."""

    out = []
    for part in argv:
        part = str(part).replace("{port}", str(port)).replace("{python}", sys.executable).replace("{repo}", str(REPO))
        out.append(os.path.expanduser(part) if part.startswith("~") else part)
    return out


class Engine:
    """One child server, from its profile entry:

        {"role": "brain", "mode": "managed" | "adopt", "port": 8091, "health": "/v1/models",
         "argv": ["~/engines/bolt/bin/tensorfold-native", "serve", "...", "--port", "{port}"], "env": {...},
         "need_gb": 43, "resident_gb": 31, "idle_s": 600, "guard_gb": 8, "label": "...", "load_s": 40}

    ``managed`` starts it on demand, unless a server already answers on its port, which is adopted instead.
    ``adopt`` never starts anything: the engine is the server on that port, or it is down."""

    def __init__(self, name: str, spec: dict, reader: MemReader | None = None, admit=None,
                 log_dir: str | os.PathLike | None = None) -> None:
        self.name = name
        self.spec = spec
        self.role = spec.get("role", name)
        self.label = spec.get("label", name)
        self.mode = spec.get("mode", "managed")
        if self.mode not in ("managed", "adopt"):
            raise ValueError(f"engine {name}: mode must be managed or adopt, not {self.mode!r}")
        self.host = spec.get("host", "127.0.0.1")
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"engine {name}: child engines live on loopback only, not {self.host}")
        self.fixed_port = int(spec.get("port") or 0)
        self.health_path = spec.get("health", "/v1/models")  # TensorFold and the Kokoro server both answer it
        self.need_gb = float(spec.get("need_gb", 0))
        self.resident_gb = float(spec.get("resident_gb", 0))
        self.load_s = float(spec.get("load_s", 30))
        self.idle_s = idle_seconds(spec.get("idle_s"))
        self.guard_gb = float(spec.get("guard_gb", 8))
        self.reader = reader or MemReader()
        self.admit = admit  # admit(engine): raises Refusal when it may not load now (the governor)
        self.log_dir = Path(log_dir or os.environ.get("GMA_HOME", "~/.gma")).expanduser() / "logs"
        self.mutex = threading.RLock()
        self.changed = threading.Condition(self.mutex)
        self.process: subprocess.Popen | None = None
        self.port = self.fixed_port
        self.phase = "unloaded"  # unloaded, loading, loaded, adopted
        self.adopted = False  # the server on the port was not started here: never stopped here
        self.in_flight = 0
        self.turns = 0
        self.hold = ""
        self.loaded_at = self.last_used = 0.0
        self.before_gb = -1.0
        self.held_gb: float | None = None
        self.load_s_measured: float | None = None
        self.killed = ""
        self.error = ""
        self.log: list[str] = []
        self.events: list[dict] = []
        self.checked_at = 0.0

    # -- what it is --

    def present(self) -> str:
        """What is missing to start it ("" when nothing): files only, no process."""

        if self.mode == "adopt":
            return ""
        argv = self.spec.get("argv")
        if not argv:
            return f"{self.label} has no command line in the profile (\"argv\"), so it can only be adopted."
        first = expand(argv[:1], self.port or 0)[0]
        if os.sep in first and not os.access(first, os.X_OK):
            return f"No {self.label} engine at {first}."
        if os.sep not in first and not shutil.which(first):
            return f"No {first} on PATH for {self.label}."
        return ""

    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def note(self, what: str, **more) -> None:
        self.events = (self.events + [{"at": time.time(), "what": what, **more}])[-20:]

    def loaded(self) -> bool:
        if self.phase == "adopted":
            return True
        return self.phase == "loaded" and self.process is not None and self.process.poll() is None

    def state(self) -> str:
        if self.phase in ("loading", "adopted"):
            return self.phase
        if self.loaded():
            return "loaded"
        return "held" if self.hold else "unloaded"

    def held_mb(self) -> int:
        """What it holds now (MB): measured at its load, else resident_gb; 0 when not loaded. An adopted server counts
        its resident_gb (it is in the machine's used memory either way)."""

        if not self.loaded():
            return 0
        return int(1024 * (self.held_gb if self.held_gb is not None else self.resident_gb))

    def status(self) -> dict:
        loaded = self.loaded()
        return {"name": self.name, "role": self.role, "label": self.label, "mode": self.mode, "state": self.state(),
                "adopted": self.adopted, "loaded": loaded, "base": self.base() if self.port else None,
                "port": self.port or None, "memory_gb": round(self.held_mb() / 1024, 1) if loaded else None,
                "resident_gb": self.resident_gb, "need_gb": self.need_gb, "load_s": self.load_s_measured or self.load_s,
                "idle_unload_s": self.idle_s or None,
                "idle_for_s": round(time.time() - self.last_used, 1) if loaded and self.last_used else None,
                "in_flight": self.in_flight, "turns": self.turns, "held_for": self.hold or None,
                "error": self.error or None, "killed": self.killed or None, "events": self.events[-6:]}

    # -- the server --

    def probe(self, timeout: float = 2.0) -> bool:
        """Whether a server answers its health path on the port."""

        if not self.port:
            return False
        try:
            with urllib.request.urlopen(self.base() + self.health_path, timeout=timeout) as answer:
                ok = answer.status == 200
        except Exception:  # noqa: BLE001 - not up (refused, 503 while loading)
            ok = False
        self.checked_at = time.time()
        return ok

    def adopt(self) -> bool:
        """Takes the server already answering on the port as this engine. Called with the mutex held."""

        if not self.fixed_port or not self.probe():
            return False
        if not self.adopted:
            self.note("adopted", port=self.port)
        self.phase, self.adopted, self.error = "adopted", True, ""
        self.loaded_at = self.loaded_at or time.time()
        self.last_used = self.last_used or time.time()
        self.changed.notify_all()
        return True

    def read(self, process: subprocess.Popen) -> None:
        """The engine's own output: its last 200 lines here, all of it in <GMA_HOME>/logs/<name>.log."""

        handle = path = None
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{self.name}.log"
            if path.exists():
                path.replace(self.log_dir / f"{self.name}.log.1")
            handle = path.open("a", buffering=1)
            handle.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.name} engine started\n")
        except OSError:
            handle = None
        for line in process.stdout:
            self.log = (self.log + [line.rstrip("\n")])[-200:]
            if handle is not None:
                try:
                    handle.write(time.strftime("%H:%M:%S ") + line if line.endswith("\n") else line + "\n")
                    if handle.tell() > LOG_BYTES:
                        handle.close()
                        path.replace(self.log_dir / f"{self.name}.log.1")
                        handle = path.open("a", buffering=1)
                except OSError:
                    handle = None

    def guard(self, process: subprocess.Popen) -> None:
        """The available memory four times a second (once on a Mac: vm_stat is a process); under the floor the
        engine's whole process group is killed."""

        mac = self.reader.platform == "mac"
        while process.poll() is None:
            free = self.reader.available_gb()
            if 0 <= free < self.guard_gb:
                self.killed = (f"available memory {free:.1f} GB < {self.guard_gb:g} GB: the memory guard killed "
                               f"{self.label} (pid {process.pid})")
                kill_group(process)
                return
            time.sleep(1.0 if mac else 0.25)

    def tick(self, now: float | None = None) -> None:
        """From the registry every few seconds: a server that ended by itself is noted; one idle for idle_s with
        nothing in flight is unloaded; an adopted server that stopped answering is let go (the next request looks
        again, and a managed engine then starts its own)."""

        now = now or time.time()
        with self.mutex:
            if self.phase == "adopted":
                if not self.probe():
                    self.phase = "unloaded"
                    self.note("the adopted server stopped answering")
                    self.changed.notify_all()
                return
            if self.process is not None and self.process.poll() is not None and self.phase == "loaded":
                self.stop("the engine ended (" + (self.killed or f"status {self.process.returncode}") + ")")
                return
            if (self.loaded() and self.idle_s and not self.in_flight and not self.turns
                    and now - self.last_used >= self.idle_s):
                self.stop(f"idle for {self.idle_s:g} s")

    def start(self, stopped=lambda: False) -> None:
        """Loaded (or adopted) and answering on its port, or an error saying why not. Called with the mutex held."""

        if self.loaded():
            return
        if self.process is not None:
            self.stop("the engine had ended (" + (self.killed or f"status {self.process.returncode}") + ")")
        if self.adopt():
            return
        if self.mode == "adopt":
            self.error = f"{self.label} is not answering on port {self.port} (adopt mode starts nothing)"
            raise Refusal(self.error, f"My {self.role} isn't running right now.")
        why = self.present()
        if why:
            self.error = why
            raise Refusal(why, f"My {self.role} isn't set up on this machine.")
        if self.admit is not None:
            self.admit(self)  # raises Refusal when the memory is not there
        self.run_stop_argv()  # a container left by a crash goes before a new one starts
        self.port = self.fixed_port or free_port()
        self.phase, self.log, self.killed, self.error, self.held_gb = "loading", [], "", "", None
        self.changed.notify_all()
        began, self.before_gb = time.time(), self.reader.available_gb()
        env = own_environment()
        env.update({k: os.path.expanduser(str(v)) for k, v in (self.spec.get("env") or {}).items()})
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-I", str(HERE / "leash.py"), "--", *expand(self.spec["argv"], self.port)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
                cwd=expand([self.spec.get("cwd") or str(REPO)], self.port)[0], start_new_session=True)
            threading.Thread(target=self.read, args=(self.process,), daemon=True).start()
            threading.Thread(target=self.guard, args=(self.process,), daemon=True).start()
            process = self.process
            timeout = float(self.spec.get("load_timeout_s", HEALTH_TIMEOUT_S))
            while not self.probe():
                if self.process is not process or self.phase != "loading":
                    raise Unloaded(self.hold or "unloaded while loading")
                if process.poll() is not None:
                    tail = [line for line in self.log if line.strip()][-3:]
                    raise RuntimeError(f"{self.label} stopped while loading: " + (self.killed or f"status {process.returncode}")
                                       + (" (" + " / ".join(tail)[-400:] + ")" if tail else ""))
                if stopped():
                    raise InterruptedError("cancelled while loading")
                if time.time() - began > timeout:
                    raise RuntimeError(f"{self.label} did not answer in {timeout:g} s")
                self.changed.wait(0.25)
            self.phase = "loaded"
            self.loaded_at = self.last_used = time.time()
            self.load_s_measured = round(self.loaded_at - began, 1)
            after = self.reader.available_gb()
            fell = self.before_gb - after if self.before_gb >= 0 and after >= 0 else None
            rss = group_rss_gb(process.pid)
            if fell is not None or rss is not None:
                self.held_gb = max(0.0, fell or 0.0, rss or 0.0)
                if self.resident_gb and self.held_gb < 0.5 * self.resident_gb:
                    # another engine loaded at the same time skews the drop, and a GPU server's RSS is not its memory
                    # (phase 4 Spark smoke: Bolt loaded beside Gemma measured ~0 GB, so evicting it "freed" nothing)
                    self.held_gb = self.resident_gb
            self.note("loaded", seconds=self.load_s_measured, port=self.port,
                      held_gb=round(self.held_gb, 1) if self.held_gb is not None else None)
            self.changed.notify_all()
        except Unloaded:
            raise
        except BaseException as error:
            self.error = str(error)[:400]
            self.stop("it failed to load")
            raise

    def stop(self, why: str) -> None:
        """A server started here is ended (its leash's pipe closed, then the group killed after 25 s). An adopted
        server is only let go: it is not this app's to end."""

        with self.mutex:
            process, self.process = self.process, None
            was = self.phase
            self.phase = "unloaded"
            if self.adopted and process is None:
                self.adopted = False
                if was == "adopted":
                    self.note("let go (adopted, still running)", why=why)
                self.changed.notify_all()
                return
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=float(self.spec.get("stop_wait_s", 25)))
                    except subprocess.TimeoutExpired:
                        kill_group(process)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                self.run_stop_argv()
                if was != "unloaded":
                    self.note("unloaded", why=why, available_gb=round(self.reader.available_gb(), 1))
            if not self.fixed_port:
                self.port = 0
            self.changed.notify_all()

    def run_stop_argv(self) -> None:
        """The profile's "stop_argv" (a containerised server: ``docker rm -f <name>``), when it has one."""

        argv = self.spec.get("stop_argv")
        if not argv:
            return
        env = own_environment()
        env.update({k: os.path.expanduser(str(v)) for k, v in (self.spec.get("env") or {}).items()})
        try:
            subprocess.run(expand(argv, self.port or 0), capture_output=True, timeout=120, env=env)
        except (OSError, subprocess.SubprocessError):
            pass

    # -- requests and releases --

    def acquire(self, stopped=lambda: False, wait: bool = True) -> str:
        """Loads it if need be and counts one request in flight; returns the base address. ``release_request`` ends it."""

        with self.mutex:
            while True:
                if self.hold:
                    if not wait:
                        raise Held(f"{self.label} is unloaded while {self.hold} has the memory",
                                   f"My {self.role} is paused while I render. Give me a moment.")
                elif self.loaded():
                    break
                elif self.phase != "loading":
                    try:
                        self.start(stopped)
                    except Unloaded:
                        continue
                    break
                if stopped():
                    raise InterruptedError("cancelled while waiting for the memory")
                self.changed.wait(1.0)
            self.in_flight += 1
            self.last_used = time.time()
            return self.base()

    def release_request(self) -> None:
        with self.mutex:
            self.in_flight -= 1
            self.last_used = time.time()
            self.changed.notify_all()

    def try_fast(self) -> bool:
        """Counts a request in flight without waiting when the engine is up and not held (the voice path)."""

        with self.mutex:
            if self.hold or not self.loaded():
                return False
            self.in_flight += 1
            self.last_used = time.time()
            return True

    @contextlib.contextmanager
    def request(self, stopped=lambda: False, wait: bool = True):
        base = self.acquire(stopped, wait)
        try:
            yield base
        finally:
            self.release_request()

    @contextlib.contextmanager
    def turn(self):
        """No idle unload while it is open."""

        with self.mutex:
            self.turns += 1
        try:
            yield
        finally:
            with self.mutex:
                self.turns -= 1
                self.last_used = time.time()

    def release(self, why: str, wait_s: float = RELEASE_WAIT_S) -> bool:
        """Held for ``why`` (no new request starts), then unloaded once the requests in flight are answered (at most
        ``wait_s``). True when a server of this app's was unloaded. An adopted server is held but never ended."""

        with self.mutex:
            self.hold = why
            self.changed.notify_all()
            if self.adopted:
                self.note("held (adopted: it keeps running)", why=why)
                return False
            if not self.loaded() and self.phase != "loading":
                return False
            ends = time.time() + wait_s
            while self.in_flight and time.time() < ends:
                self.changed.wait(0.5)
            if self.in_flight:
                self.note("an answer was cut", why=f"{why} waited {wait_s:g} s")
            self.stop(why)
            return True

    def unhold(self, why: str = "") -> None:
        with self.mutex:
            if self.hold:
                self.note("free to load again", why=why or "rendering ended")
            self.hold = ""
            self.changed.notify_all()


class InProcess:
    """An engine that lives in the app's own process (Parakeet ASR, as Ghost Machine runs it): counted by the
    governor at ``resident_gb`` while loaded; nothing to start, stop or leash."""

    mode = "inprocess"
    adopted = False
    in_flight = 0
    hold = ""

    def __init__(self, name: str, spec: dict, factory=None) -> None:
        self.name, self.spec = name, spec
        self.role = spec.get("role", name)
        self.label = spec.get("label", name)
        self.resident_gb = float(spec.get("resident_gb", 0))
        self.need_gb = float(spec.get("need_gb", self.resident_gb))
        self.factory = factory
        self.value = None
        self.phase = "unloaded"

    def load(self):
        if self.value is None and self.factory is not None:
            self.value = self.factory()
        self.phase = "loaded"
        return self.value

    def set(self, value) -> None:
        self.value, self.phase = value, "loaded"

    def loaded(self) -> bool:
        return self.phase == "loaded"

    def held_mb(self) -> int:
        return int(1024 * self.resident_gb) if self.loaded() else 0

    def tick(self, now: float | None = None) -> None:
        return None

    def release(self, why: str, wait_s: float = 0) -> bool:
        return False

    def unhold(self, why: str = "") -> None:
        return None

    def stop(self, why: str) -> None:
        return None

    def status(self) -> dict:
        return {"name": self.name, "role": self.role, "label": self.label, "mode": self.mode,
                "state": "loaded" if self.loaded() else "unloaded", "adopted": False, "loaded": self.loaded(),
                "memory_gb": self.resident_gb if self.loaded() else None, "resident_gb": self.resident_gb,
                "in_flight": 0, "held_for": None}
