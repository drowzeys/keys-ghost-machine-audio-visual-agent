"""Ghost Studio's render side: a headless Imagination Studio, driven through its own HTTP API.

The Studio (drowzeys/Imagination-Studio-by-keys-Wasserman, unmodified) runs as a child engine of this app: its
``app/server.py`` on 127.0.0.1 and a port of its own (7890), its own Studio home folder, its own ComfyUI port (8290),
sharing an existing install (PREFIX) and its models read only. It runs with ``STUDIO_PIDFILE`` set, the switch the
Studio gives an app another app starts (a worker Spark's): no self-updater, no background captioner, and its launcher is
bypassed. Nobody opens its page: Ghost Studio's own panel is the only UI.

- ``studio_engine(profile)``: the registry entry that starts it (or adopts a Studio already answering on the port).
  Role ``renderer``: the governor never evicts it; it is the render.
- ``StudioClient``: the routes used, synchronously (the governor and AutoPilot's thread call it as they are):
  ``/api/state``, ``/api/jobs`` (scout, edit, soundtrack...), ``/api/jobs/{id}``,
  ``/api/jobs/{id}/cancel``, ``/api/voice/takes`` (Fish Audio Drama 3), ``/api/music/takes`` (MiniMax Music 3),
  ``/api/music/soundtrack`` (a song laid under a clip, ducked under its dialogue), ``/api/upload``, ``/files/...``.
- ``HeadlessRenderer``: the video panel's and AutoPilot's ``Renderer`` (voice_agent/video.py). A keyframe is a
  ``scout`` job of the Studio (an ``edit`` job beside reference pictures: the place, the cast); a clip is never the
  Studio's: it is made by the FP4 clip worker (gma/clips.py, gma/fp4_worker.py: FastH3 V2 on FastVideo's FP4
  sparse-attention recipe), with no other route. Each runs inside one governor session (gma/governor.py: the
  schedule's evictions), and the session ends with the Studio's ComfyUI models freed (its ``/free``; ``after_render:
  stop`` stops that server), so an evicted brain can load again.

What the Studio's API does not give, and how it is made up here (each is said in the README's "Studio API gaps"):

- freeing the keyframe's models after a job: ComfyUI's ``POST /free`` on the Studio's ComfyUI port (``after_render:
  stop`` runs the Studio's own ``spark/comfy.sh stop`` for that port); the Studio keeps ComfyUI warm by design.

A voice turn first: the clip worker's process is stopped (SIGSTOP) from the end of the person's speech to the reply's
first audio, 4 s at most.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from gm_core import h3_prompt, lengths

from . import clips
from gm_core.engine import Engine, Refusal
from voice_agent.video import Progress, Refused, Renderer

DEFAULT_PORT = 7890
DEFAULT_COMFY_PORT = 8290
SAID_FAILED = "The render failed. The panel has the details."
NOT_SET_UP = "Video isn't set up on this machine."
PRESET_OF_SIZE = {(864, 480): "small", (1376, 768): "sharp", (1344, 768): "native", (1280, 720): "p720"}


class StudioError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status


def expand(path) -> Path:
    return Path(os.path.expanduser(str(path)))


def studio_engine(spec: dict, platform: str = "spark") -> dict:
    """The registry's engine entry for the headless Studio, from the profile's ``studio`` section:

        {"src": "~/ghost-studio/studio-src", "home": "~/ghost-studio/home", "prefix": "~/.local/opt/tensorfold-studio",
         "port": 7890, "comfy_port": 8290, "mode": "managed" | "adopt", "env": {...}}"""

    src, home = expand(spec.get("src", "~/ghost-studio/studio-src")), expand(spec.get("home", "~/ghost-studio/home"))
    prefix = expand(spec.get("prefix", "~/.local/opt/tensorfold-studio"))
    port, comfy = int(spec.get("port", DEFAULT_PORT)), int(spec.get("comfy_port", DEFAULT_COMFY_PORT))
    comfy_dir = home / ".comfy"  # ComfyUI's own folders under the Studio home: nothing is written into the install's
    env = {"PREFIX": str(prefix), "STUDIO_HOME": str(home), "HOST": "127.0.0.1", "PORT": str(port),
           # the switch the Studio gives an app started by another app: no updater, no captioner, no launcher
           "STUDIO_PIDFILE": str(home / "app.pid"),
           "SPARK_COMFY_PORT": str(comfy),
           "SPARK_COMFY_ARGS": " ".join(["--use-ck-attention", f"--input-directory {comfy_dir / 'input'}",
                                         f"--output-directory {comfy_dir / 'output'}", f"--temp-directory {comfy_dir / 'temp'}",
                                         f"--user-directory {comfy_dir / 'user'}"]),
           "LEASH_GRACE_S": "60",  # its shutdown stops its ComfyUI first (spark/comfy.sh stop)
           **{k: str(v) for k, v in (spec.get("env") or {}).items()}}
    if platform == "mac":
        env.pop("SPARK_COMFY_ARGS")
    for sub in ("input", "output", "temp", "user"):
        (comfy_dir / sub).mkdir(parents=True, exist_ok=True)
    python = expand(spec.get("python") or prefix / "venv" / "bin" / "python")
    out = {"role": "renderer", "label": spec.get("label", "Imagination Studio (headless)"), "mode": spec.get("mode", "managed"),
           "port": port, "health": "/api/jobs", "argv": [str(python), str(src / "app" / "server.py")], "env": env,
           "cwd": str(src), "need_gb": float(spec.get("need_gb", 0)), "resident_gb": float(spec.get("resident_gb", 2.0)),
           "load_s": 5, "load_timeout_s": 120, "idle_s": "never", "guard_gb": float(spec.get("guard_gb", 12)),
           "stop_wait_s": 70,
           # the Studio app's own memory is not the render's: its estimate counts only what its ComfyUI holds
           "counts_in_render": False}
    if platform != "mac":
        # whatever ended the Studio, its ComfyUI (its own session, setsid) goes too, before a start and after a stop
        out["stop_argv"] = ["bash", str(src / "spark" / "comfy.sh"), "stop"]
    return out


class StudioClient:
    """The Studio's HTTP API, synchronous (urllib): small JSON calls with a timeout each."""

    def __init__(self, url: str, timeout: float = 30.0, opener=None) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self._home: Path | None = None
        self._state: dict | None = None

    def call(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with self.opener(request, timeout=timeout or self.timeout) as answer:
                raw = answer.read()
        except urllib.error.HTTPError as error:
            try:
                said = json.loads(error.read() or b"{}")
                said = said.get("error") or said.get("detail") or error.reason
            except ValueError:
                said = error.reason
            raise StudioError(error.code, str(said)) from None
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise StudioError(0, f"the Studio is not answering ({error})") from None
        return json.loads(raw) if raw else {}

    # -- what it is --

    def up(self) -> bool:
        try:
            self.call("GET", "/api/jobs", timeout=3)
            return True
        except StudioError:
            return False

    def state(self, fresh: bool = False) -> dict:
        if self._state is None or fresh:
            self._state = self.call("GET", "/api/state", timeout=60)
            self._home = Path(self._state.get("home") or "")
        return self._state

    def home(self) -> Path:
        if self._home is None:
            self.state()
        return self._home

    def image_models(self) -> list[str]:
        return [m.get("id") if isinstance(m, dict) else str(m) for m in self.state().get("image_models") or []]

    def inside(self, path: str | Path) -> str | None:
        """``path`` as the Studio names it (relative to its home), or None when it is outside the home."""

        try:
            return Path(path).resolve().relative_to(self.home().resolve()).as_posix()
        except ValueError:
            return None

    def bring(self, path: str | Path) -> str:
        """A file of this app's as a path in the Studio's home: as it is when it is there already, else copied into
        ``ghost-inputs/`` (the same machine) or sent to ``/api/upload``."""

        found = self.inside(path)
        if found is not None:
            return found
        source = Path(path)
        home = self.home()
        if home.is_dir():
            target = home / "ghost-inputs" / f"{uuid.uuid4().hex[:10]}{source.suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return target.relative_to(home).as_posix()
        return self.upload(source)

    def upload(self, source: Path) -> str:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{source.name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + source.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(self.url + "/api/upload", data=body, method="POST",
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with self.opener(request, timeout=300) as answer:
            return json.loads(answer.read())["path"]

    def file_bytes(self, path: str) -> bytes | None:
        local = self.home() / path if not Path(path).is_absolute() else Path(path)
        if local.is_file():
            return local.read_bytes()
        try:
            with self.opener(self.url + "/files/" + urllib.parse.quote(self.inside(path) or path), timeout=120) as answer:
                return answer.read()
        except (urllib.error.URLError, OSError):
            return None

    def fetch(self, path: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        local = self.home() / path if not Path(path).is_absolute() else Path(path)
        if local.is_file():
            if local.resolve() == dest.resolve():
                return
            shutil.copyfile(local, part)
        else:
            data = self.file_bytes(path)
            if data is None:
                raise StudioError(404, f"the Studio has no file {path}")
            part.write_bytes(data)
        part.replace(dest)

    # -- the queue --

    def submit(self, kind: str, params: dict, title: str = "") -> dict:
        return self.call("POST", "/api/jobs", {"kind": kind, "title": title[:120] or f"Ghost Studio · {kind}", "params": params})

    def job(self, job_id: str) -> dict:
        return self.call("GET", f"/api/jobs/{urllib.parse.quote(job_id)}")

    def cancel(self, job_id: str) -> dict | None:
        try:
            return self.call("POST", f"/api/jobs/{urllib.parse.quote(job_id)}/cancel", {})
        except StudioError:
            return None

    def wait(self, job_id: str, stop: threading.Event | None = None, every: float = 1.0, limit_s: float = 3600) -> dict:
        began = time.time()
        while True:
            job = self.job(job_id)
            if job.get("status") in ("done", "failed", "cancelled"):
                return job
            if stop is not None and stop.is_set():
                self.cancel(job_id)
                return {**job, "status": "cancelled"}
            if time.time() - began > limit_s:
                self.cancel(job_id)
                raise StudioError(0, f"the Studio's job {job_id} took longer than {limit_s:.0f} s")
            time.sleep(every)

    # -- sound --

    def voice_take(self, text: str, **more) -> dict:
        """A Fish Audio Drama 3 take (POST /api/voice/takes): {text, seed?, temperature?, ref_audio?, ref_text?, consent?}."""
        return self.call("POST", "/api/voice/takes", {"text": text, **more})

    def music_take(self, caption: str, seconds: float, lyrics: str = "[Instrumental]", **more) -> dict:
        """A MiniMax Music 3 song (POST /api/music/takes): the style in ``caption``, 10 to 300 s."""
        return self.call("POST", "/api/music/takes", {"caption": caption, "lyrics": lyrics, "seconds": seconds, **more})

    def soundtrack(self, take: str, video: str, music_db: float | None = None) -> dict:
        """A song laid under a clip of the Studio's (POST /api/music/soundtrack), ducked under its dialogue."""
        return self.call("POST", "/api/music/soundtrack", {"take": take, "video": video,
                                                           **({"music_db": music_db} if music_db is not None else {})})


def comfy_pid(port: int) -> int | None:
    """The Studio's ComfyUI server process on ``port`` (spark/comfy.sh: ``main.py --listen 127.0.0.1 --port N``)."""

    try:
        said = subprocess.run(["pgrep", "-f", f"main\\.py --listen 127\\.0\\.0\\.1 --port {int(port)}( |$)"],
                              capture_output=True, text=True, timeout=5).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    pids = [int(p) for p in said if p.isdigit() and int(p) != os.getpid()]
    return pids[0] if pids else None


def gpu_mb(pid: int) -> int:
    """What a process holds on the GPU (MB, nvidia-smi's per-process figure; 0 when it cannot say)."""

    try:
        said = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    for line in said.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            return int(parts[1])
    return 0


class HeadlessRenderer(Renderer):
    """Keyframes as jobs of the headless Studio and clips from the FP4 clip worker, one at a time, each inside a
    governor session."""

    name = "studio-headless"
    governed = True  # the governor decides whether a clip fits (VideoService skips its own host check)

    def __init__(self, platform: str, spec: dict | None = None, registry=None, governor=None, reader=None,
                 client: StudioClient | None = None, worker: clips.ClipWorker | None = None) -> None:
        self.platform = platform
        self.spec = spec = dict(spec or {})
        self.registry, self.governor = registry, governor
        self.reader = reader or getattr(registry, "reader", None)
        self.url = spec.get("url") or f"http://127.0.0.1:{int(spec.get('port', DEFAULT_PORT))}"
        self.client = client or StudioClient(self.url)
        self.worker = worker  # the FP4 clip worker's API; made from its engine's port when not given (tests give one)
        self.comfy_port = int(spec.get("comfy_port", DEFAULT_COMFY_PORT))
        self.comfy_url = spec.get("comfy_url") or f"http://127.0.0.1:{self.comfy_port}"
        self.label = "headless Imagination Studio (" + self.url + ") + FP4 clip worker"
        self.width, self.height = int(spec.get("width", 864)), int(spec.get("height", 480))
        self.root = expand(spec.get("dir", "~/ghost-studio/home/ghost/renders")).resolve()
        self.previews = False
        self.preview_kind = "stages"  # the panel shows the keyframe with the stage and the forward count
        self.jobs: dict[str, dict] = {}
        self.session: dict | None = None
        self.expiry: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.last: dict = {}
        self.voice_held: dict | None = None
        self.voice_pauses: list[float] = []
        self.voice_token, self.voice_pid = 0, None
        self.worker_pid: int | None = None

    # ----- the engines -----

    def engine(self) -> Engine | None:
        one = self.registry.get(self.spec.get("engine", "studio")) if self.registry is not None else None
        return one if isinstance(one, Engine) else None

    def clip_engine(self) -> Engine | None:
        one = self.registry.get(self.spec.get("clips_engine", clips.ENGINE)) if self.registry is not None else None
        return one if isinstance(one, Engine) else None

    def missing(self, kind: str = "keyframe") -> str:
        if kind == "video":
            one = self.clip_engine()
            if one is None:
                return "" if self.worker is not None else (
                    f"no FP4 clip worker in the profile (engine \"{self.spec.get('clips_engine', clips.ENGINE)}\")")
            return "" if one.loaded() else one.present()
        one = self.engine()
        if one is None:
            return "" if self.client.up() else f"no Imagination Studio answering at {self.url}"
        if one.loaded():
            return ""
        return one.present()

    async def ready(self) -> None:
        """The Studio up (started by its engine, or adopted) and counted in use while a keyframe runs."""

        one = self.engine()
        if one is None:
            if not await asyncio.to_thread(self.client.up):
                raise Refused(f"No Imagination Studio answers at {self.url}.", NOT_SET_UP)
            return
        await asyncio.to_thread(one.acquire)
        one.release_request()

    # ----- the governor's view (gma/governor.py) -----

    def memory_request(self, render: dict) -> dict:
        """What a render asks of the machine at its peak, MB: the profile's stated figure (a keyframe, the film's
        sound), or a clip's measured peak (gma/clips.py NEED_MB) less what the clip worker, kept by the schedule,
        holds already."""

        if render.get("need_gb"):
            return {"need_mb": int(float(render["need_gb"]) * 1024), "error_mb": int(float(render.get("error_gb", 0)) * 1024),
                    "route": render.get("route", "stated"), "rows": None, "source": "profile"}
        frames = int(render.get("frames") or 124)
        peak = clips.need_mb(int(render.get("width") or self.width), int(render.get("height") or self.height), frames)
        one = self.clip_engine()
        held = one.held_mb() if one is not None else 0
        return {"need_mb": max(0, peak - held), "error_mb": clips.ERROR_MB, "route": "fp4", "rows": None, "frames": frames,
                "peak_mb": peak, "worker_held_mb": held, "source": "FP4 clip worker, measured (gma/clips.py)"}

    def floor_mb(self) -> int | None:
        return None

    def held_mb(self) -> int:
        """What the Studio's ComfyUI holds now (MB): a keyframe's models, given back when the session ends (free_cache),
        so the governor counts them in the room of the render that follows. Measured on the DGX Spark: 25.5 GB held after a
        FLUX 3 Klein edit and its /free."""

        if self.platform == "mac":
            return 0
        pid = comfy_pid(self.comfy_port)
        return gpu_mb(pid) if pid else 0

    def free_cache(self) -> dict:
        """The Studio's ComfyUI models given back after a render (its ``POST /free``; never while it is busy); with
        ``after_render: stop`` its server is stopped too (the Studio's own spark/comfy.sh, for its port)."""

        if self.platform == "mac":
            return {"freed": False, "why": "the Studio's Mac engines end with their job"}
        try:
            queue = self.comfy_post("/queue")
        except (OSError, ValueError):
            return {"freed": False, "why": "the Studio's ComfyUI is not running"}
        if queue.get("queue_running") or queue.get("queue_pending"):
            return {"freed": False, "why": "the renderer is busy"}
        try:
            self.comfy_post("/free", {"unload_models": True, "free_memory": True})
        except (OSError, ValueError) as error:
            return {"freed": False, "why": f"/free failed: {error}"}
        if self.spec.get("after_render") == "stop":
            one = self.engine()
            src = Path((one.spec.get("cwd") if one else None) or self.spec.get("src") or ".")
            env = {**os.environ, **{k: str(v) for k, v in ((one.spec.get("env") if one else None) or {}).items()},
                   "SPARK_COMFY_PORT": str(self.comfy_port)}
            subprocess.run(["bash", str(src / "spark" / "comfy.sh"), "stop"], env=env, capture_output=True, timeout=90)
            return {"freed": True, "why": "freed the Studio's ComfyUI models and stopped its server"}
        return {"freed": True, "why": "freed the Studio's ComfyUI models (POST /free)"}

    def comfy_post(self, path: str, data: dict | None = None) -> dict:
        request = urllib.request.Request(self.comfy_url + path, data=None if data is None else json.dumps(data).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as answer:
            raw = answer.read()
        return json.loads(raw) if raw else {}

    async def room(self, clip: dict) -> Refused | None:
        why = await asyncio.to_thread(self.missing, "video")
        if why:
            return Refused(f"No clip renderer here: {why.rstrip('.')}.", NOT_SET_UP)
        if any(j["state"] in ("queued", "rendering") for j in self.jobs.values()):  # a job ending is not one at work
            return Refused("This machine is already rendering a clip.", "I'm already rendering. Let me finish this one first.")
        if self.governor is None:
            return None
        # the memory a keyframe's /free and an evicted judge give back arrives over seconds: a plan that does not fit
        # is asked again (measured on the DGX Spark: a piece refused 0.6 GB short 2 s after its keyframe, fitting 10 s later)
        deadline = time.monotonic() + float(self.spec.get("room_wait_s", 60))
        while True:
            try:
                found = await asyncio.to_thread(self.governor.plan, self.schedule_for("video", clip), self.frames(clip))
            except Refusal as error:
                return Refused(str(error), error.said)
            if found["fits"] or time.monotonic() >= deadline:
                break
            if self.reader is not None and hasattr(self.reader, "settle"):
                await asyncio.to_thread(self.reader.settle)
            await asyncio.sleep(2.0)
        if not found["fits"]:
            said = "There isn't enough memory to render that right now."
            if found.get("adopted_blockers") and found.get("available_mb", -1) >= 0:
                said = "Part of me runs outside my control, so I can't free the memory for that."
            return Refused(f"No room for the clip now: {found['reason']}", said)
        return None

    def schedule_for(self, kind: str, clip: dict | None = None) -> str | None:
        schedules = getattr(self.governor, "schedules", None) or {}
        default = self.spec.get("schedule")
        if kind == "keyframe":
            asked = self.spec.get("keyframe_schedule", "keyframe")
            return asked if asked in schedules else default
        asked = (clip or {}).get("schedule")
        return asked if asked and asked in schedules else default

    async def begin(self, why: str, frames: int | None = None, schedule: str | None = None) -> None:
        gate = getattr(self.governor, "thermal_gate", None)
        if gate is not None and self.platform != "mac":
            # every clip and keyframe waits for a cool GPU, even inside a render session that is already open
            await asyncio.to_thread(gate, why)
        if self.expiry is not None:
            self.expiry.cancel()
            self.expiry = None
        if self.session is not None:
            if self.session.get("schedule") == schedule:
                return
            await self.end(f"a {why} needs the {schedule or 'default'} schedule")
        grant = await self.governor.enter(schedule, frames) if self.governor is not None else None
        self.session = {"started": time.time(), "why": why, "grant": grant, "schedule": schedule}

    async def end(self, why: str) -> None:
        if self.expiry is not None and self.expiry is not asyncio.current_task():
            self.expiry.cancel()
        self.expiry = None
        if self.session is None:
            return
        self.session = None
        if self.governor is not None:
            await self.governor.leave(why)  # frees the Studio's ComfyUI first (free_cache), then the brain may load
        else:
            await asyncio.to_thread(self.free_cache)

    def end_later(self, seconds: float) -> None:
        async def expire():
            await asyncio.sleep(seconds)
            if not any(j["kind"] == "video" and j["state"] in ("queued", "rendering") for j in self.jobs.values()):
                await self.end("the keyframe's clip never came")

        self.expiry = asyncio.create_task(expire())

    async def dit(self, starting: bool) -> None:
        if self.governor is not None:
            try:
                await self.governor.dit(starting)
            except Exception:  # noqa: BLE001 - the render goes on; the governor's trouble is its own
                pass

    # ----- a voice turn first -----

    def hold_for_voice(self, on: bool) -> bool:
        """The clip worker stopped (SIGSTOP; the GPU work already queued finishes) from the end of the person's speech
        to the reply's first audio, ``voice_hold_s`` at most, then resumed (SIGCONT). Only while a clip renders. The
        kill-guards (other processes) keep watching meanwhile."""

        if not self.spec.get("pause_for_voice", self.platform != "mac"):
            return False
        if on:
            if self.voice_held or not any(j["kind"] == "video" and j["state"] == "rendering" for j in self.jobs.values()):
                return False
            pid = self.worker_pid
            if pid is None:
                return False
            try:
                os.kill(pid, signal.SIGSTOP)
            except OSError:
                return False
            self.voice_token += 1
            self.voice_held = {"pid": pid, "at": time.monotonic(), "token": self.voice_token}
            self.voice_pid = pid
            token, limit = self.voice_token, float(self.spec.get("voice_hold_s", 4.0))

            def expire() -> None:
                if self.voice_held and self.voice_held["token"] == token:
                    self.hold_for_voice(False)
            timer = threading.Timer(limit, expire)  # a thread's timer: a busy loop never leaves the render stopped
            timer.daemon = True
            timer.start()
            return True
        held, self.voice_held = self.voice_held, None
        pid = (held or {}).get("pid") or self.voice_pid
        if pid is None:
            return False
        try:
            os.kill(pid, signal.SIGCONT)
        except OSError:
            return False
        if held:
            self.voice_pauses.append(round(time.monotonic() - held["at"], 3))
            del self.voice_pauses[:-200]
        return bool(held)

    # ----- the Renderer interface -----

    def new_job(self, kind: str, **more) -> str:
        handle = f"{kind[:1]}{uuid.uuid4().hex[:10]}"
        folder = self.root / handle
        folder.mkdir(parents=True, exist_ok=True)
        self.jobs[handle] = {"kind": kind, "state": "queued", "at": None, "of": None, "stage": "queued", "output": None,
                             "error": None, "said": None, "previews": [], "timings": {}, "folder": folder,
                             "created": time.time(), "studio_job": None, **more}
        return handle

    async def keyframe(self, prompt: str, seed: int, references: list[str] | None = None,
                       size: tuple[int, int] | None = None) -> str:
        handle = self.new_job("keyframe", prompt=prompt, seed=int(seed), references=[str(r) for r in references or []],
                              size=tuple(size) if size else None)
        self.jobs[handle]["task"] = asyncio.create_task(self.run_job(handle))
        return handle

    async def submit(self, clip: dict) -> str:
        handle = self.new_job("video", clip=dict(clip))
        self.jobs[handle]["task"] = asyncio.create_task(self.run_job(handle))
        return handle

    async def poll(self, handle: str) -> Progress:
        job = self.jobs.get(handle)
        if job is None:
            return Progress("failed", error="the render was lost when the app restarted")
        eta = None
        if job["state"] == "rendering" and job.get("of") and job.get("at") is not None and job.get("step_s"):
            eta = max(0.0, (job["of"] - job["at"]) * job["step_s"] + float(self.spec.get("decode_s", 15)))
        return Progress(job["state"], job["at"], job["of"], job["stage"], eta, job["output"], job["error"], None,
                        timings=dict(job["timings"]) or None)

    async def picture(self, ref: str) -> bytes | None:
        path = Path(ref)
        return path.read_bytes() if path.is_file() else await asyncio.to_thread(self.client.file_bytes, ref)

    async def fetch(self, output: str, dest: Path) -> None:
        try:
            if Path(output).is_file():
                if Path(output).resolve() != Path(dest).resolve():
                    Path(dest).parent.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(shutil.copyfile, output, dest)
                return
            await asyncio.to_thread(self.client.fetch, output, Path(dest))
        except (StudioError, OSError) as error:
            raise Refused(f"The render could not be fetched: {error}") from None

    async def cancel(self, handle: str) -> None:
        job = self.jobs.get(handle)
        if not job or job["state"] not in ("queued", "rendering"):
            return
        job["state"], job["stage"] = "cancelled", "cancelled"
        self.hold_for_voice(False)
        await self.cancel_studio(job)
        task = job.get("task")
        if task is not None and not task.done():
            task.cancel()

    async def cancel_studio(self, job: dict) -> None:
        if job.get("studio_job") and not job.get("cancel_sent"):
            job["cancel_sent"] = True
            await asyncio.to_thread(self.client.cancel, job["studio_job"])

    # ----- a job -----

    def frames(self, clip: dict) -> int:
        if clip.get("frames") and lengths.valid(int(clip["frames"])):
            return int(clip["frames"])
        return lengths.nearest(float(clip.get("seconds") or 5))

    def picture_model(self, edit: bool) -> str | None:
        asked = self.spec.get("edit_picture" if edit else "picture", "auto")
        if asked and asked != "auto":
            return asked
        try:
            found = self.client.image_models()
        except StudioError:
            return None
        order = ("flux3k-fast", "qwen") if edit else ("qwen", "flux3k-fast")  # FLUX 3 Klein's multi-reference for edits
        return next((m for m in order if m in found), found[0] if found else None)

    async def keyframe_params(self, job: dict) -> tuple[str, dict, str]:
        """(kind, params, title) of a keyframe's Studio job: ``scout``, or ``edit`` beside reference pictures."""

        refs = [await asyncio.to_thread(self.client.bring, r) for r in job["references"]]
        width, height = job["size"] or (self.width, self.height)
        model = await asyncio.to_thread(self.picture_model, bool(refs))
        if refs:
            params = {"image": refs[0], "images": refs[1:], "prompt": job["prompt"], "seed": job["seed"],
                      "width": int(width) // 16 * 16, "height": int(height) // 16 * 16}
            kind = "edit"
        else:
            params = {"prompt": job["prompt"], "count": 1, "seed": job["seed"]}
            preset = PRESET_OF_SIZE.get((int(width), int(height)))
            params.update({"preset": preset} if preset else {"width": int(width) // 16 * 16, "height": int(height) // 16 * 16})
            kind = "scout"
        if model:
            params["image_model"] = model
        return kind, params, "Ghost Studio · keyframe"

    async def run_job(self, handle: str) -> None:
        job = self.jobs[handle]
        began = time.monotonic()
        watch = asyncio.create_task(self.watch_memory(job))
        # the whole job, its session's end included, under the renderer's lock: a job queued the moment this one fails
        # (AutoPilot retakes at once) never enters the governor before this one has left it
        async with self.lock:
            await self.run_locked(job, began, watch)

    async def run_locked(self, job: dict, began: float, watch: asyncio.Task) -> None:
        video = job["kind"] == "video"
        try:
            if job["state"] == "cancelled":
                return
            job["state"], job["stage"] = "rendering", "waiting for the memory"
            if not video:
                await self.ready()
            await self.begin(job["kind"], self.frames(job["clip"]) if video else None,
                             self.schedule_for(job["kind"], job.get("clip")))
            if self.session is not None:
                self.session["busy"] = True
            await (self.clip_job(job) if video else self.studio_job(job))
            if job["state"] != "cancelled":
                job["state"], job["stage"] = "done", "done"
        except asyncio.CancelledError:
            job["state"], job["stage"] = "cancelled", "cancelled"
            await self.cancel_studio(job)
        except Refusal as error:
            job.update(state="failed", error=str(error), said=error.said)
        except (StudioError, clips.ClipError, Refused, RuntimeError, OSError, ValueError) as error:
            job.update(state="failed", error=str(error)[:400])
        except Exception as error:  # noqa: BLE001 - a bug: said, and the session still ends
            job.update(state="failed", error=f"{type(error).__name__}: {str(error)[:300]}")
        finally:
            watch.cancel()
            self.hold_for_voice(False)
            job["timings"]["total_s"] = round(time.monotonic() - began, 1)
            self.last = {"kind": job["kind"], "state": job["state"], **job["timings"]}
            try:
                (job["folder"] / "job.json").write_text(json.dumps(
                    {k: v for k, v in job.items() if k not in ("previews", "task", "folder")}, indent=1, default=str))
            except OSError:
                pass
            if self.session is not None:
                self.session["busy"] = False
            if job["kind"] == "keyframe" and job["state"] == "done":
                self.end_later(float(self.spec.get("keyframe_wait_s", 120)))
            else:
                await self.end(f"the {job['kind']} {job['state']}")

    async def studio_job(self, job: dict) -> None:
        """A keyframe: the Studio's scout (or edit) job, polled until its picture is made."""

        kind, params, title = await self.keyframe_params(job)
        job["studio_params"] = params
        t0 = time.monotonic()
        made = await asyncio.to_thread(self.client.submit, kind, params, title)
        job["studio_job"] = made["id"]
        if job["state"] == "cancelled":
            await self.cancel_studio(job)
            return
        poll_s = float(self.spec.get("poll_s", 0.5))
        while True:
            said = await asyncio.to_thread(self.client.job, made["id"])
            status = said.get("status")
            job["stage"] = "keyframe"
            if status in ("done", "failed", "cancelled"):
                break
            if job["state"] == "cancelled":
                await self.cancel_studio(job)
                return
            await asyncio.sleep(poll_s)
        if status != "done":
            if status == "cancelled":
                job["state"] = "cancelled"
                return
            raise RuntimeError("the Studio's job failed: " + str(said.get("error") or "no reason given")[:300])
        outputs = [o for o in said.get("outputs") or [] if isinstance(o, str)]
        found = next((o for o in outputs if o.endswith((".png", ".jpg", ".webp")) and not o.endswith("sheet.jpg")), None)
        if not found:
            raise RuntimeError("the Studio's job finished without its file")
        job["output"] = str(self.client.home() / found)
        t = job["timings"]
        t["studio_s"] = t["keyframe_s"] = round(float(said.get("elapsed") or time.monotonic() - t0), 1)
        t["picture"] = params.get("image_model")
        if "starting ComfyUI" in str(said.get("log") or ""):
            t["server_start"] = True

    async def clip_job(self, job: dict) -> None:
        """A clip: the FP4 clip worker loaded (or kept from the take before), the clip asked of it and followed; the
        brain the schedule releases for the DiT pass goes before the worker loads, since the load and the pass follow
        at once. Any failure is the take's, with the worker's reason."""

        clip, t = job["clip"], job["timings"]
        frames = self.frames(clip)
        width, height = int(clip.get("width") or self.width), int(clip.get("height") or self.height)
        first = clip.get("image_file") or clip.get("image")  # AutoPilot's keyframe file, or the panel's keyframe output
        if not first and clip.get("opens_on"):
            first = str(await asyncio.to_thread(clips.last_frame, clip["opens_on"], job["folder"] / "first.png",
                                                self.spec.get("ffmpeg") or "ffmpeg"))
        if first and not Path(first).is_file():
            first = str(self.client.home() / first)  # a panel keyframe named inside the Studio's home
        # AutoPilot's prompt is composed with its fields already; the panel's plain description gets MiniMax H3's form
        prompt = clip["prompt"] if clip.get("composed") else h3_prompt.from_idea(clip["prompt"], bool(first), frames)
        one = self.clip_engine()
        if self.spec.get("after_render") == "stop" and self.platform != "mac":
            # the picture models leave before the clip model loads: a keyframe cancelled a moment ago leaves the
            # Studio's ComfyUI busy, its stop is then refused, and both beside the brain do not fit (2026-11-19: the
            # memory guard killed the clip worker at 25 GB available)
            for _ in range(int(self.spec.get("comfy_leave_s", 90))):
                if not await asyncio.to_thread(self.held_mb):
                    break
                job["stage"] = "waiting for the picture models to leave"
                await asyncio.to_thread(self.free_cache)
                await asyncio.sleep(1.0)
        t0 = time.monotonic()
        if one is not None and not one.loaded():
            await self.dit(True)  # the load and the DiT pass need the brain's room
            job["stage"] = "loading the clip model"
            await asyncio.to_thread(one.acquire)
            t["load_s"] = round(time.monotonic() - t0, 1)
        elif one is not None:
            await asyncio.to_thread(one.acquire)
        try:
            worker = self.worker or clips.ClipWorker(one.base())
            health = await asyncio.to_thread(worker.health)
            self.worker_pid = health.get("pid")
            output = job["folder"] / "clip.mp4"
            body = {"prompt": prompt, "image": first, "last": clip.get("last_file") or None, "width": width, "height": height, "frames": frames,
                    "seed": int(clip.get("seed", 0)), "output": str(output)}
            job["clip_request"] = {**body, "prompt": prompt[:300]}
            made = await self.submit_clip(worker, body)
            in_dit, last_at = False, None
            poll_s = float(self.spec.get("poll_s", 0.5))
            while True:
                said = await asyncio.to_thread(worker.clip, made["id"])
                status, stage = said.get("status"), said.get("stage") or ""
                if stage == "denoising" and not in_dit:
                    in_dit = True
                    await self.dit(True)
                elif in_dit and stage not in ("denoising", "conditioning"):
                    in_dit = False
                    await self.dit(False)
                at = said.get("at")
                if at is not None:
                    if last_at is not None and at > last_at[0]:
                        job["step_s"] = (time.monotonic() - last_at[1]) / (at - last_at[0])
                    if last_at is None or at != last_at[0]:
                        last_at = (at, time.monotonic())
                    job["at"], job["of"] = at, said.get("of")
                job["stage"] = stage or status or ""
                if status in ("done", "failed"):
                    break
                if job["state"] == "cancelled":
                    return  # the worker finishes the clip it has; the next take waits for it (submit_clip)
                await asyncio.sleep(poll_s)
            if in_dit:
                await self.dit(False)
            if status != "done":
                raise clips.ClipError("the FP4 clip worker failed the clip: " + str(said.get("error") or "no reason given")[:300])
            if not output.is_file():
                raise clips.ClipError("the FP4 clip worker finished without its file")
            job["output"] = str(output)
            t.update({k: v for k, v in (said.get("timings") or {}).items() if not k.endswith("_stage")})
            t["clip_s"] = round(time.monotonic() - t0, 1)
            t["renderer"] = "fp4"
        finally:
            if one is not None:
                one.release_request()

    async def submit_clip(self, worker: clips.ClipWorker, body: dict) -> dict:
        """The clip asked of the worker; while it still finishes a cancelled take's clip (it says so), asked again."""

        deadline = time.monotonic() + float(self.spec.get("busy_wait_s", 300))
        while True:
            try:
                return await asyncio.to_thread(worker.submit, body)
            except clips.ClipError as error:
                if "rendering already" not in str(error) or time.monotonic() > deadline:
                    raise
            await asyncio.sleep(2.0)

    async def watch_memory(self, job: dict) -> None:
        if self.reader is None:
            return
        total = self.reader.total_mb()
        low = None
        while True:
            free = await asyncio.to_thread(self.reader.available_mb)
            if free >= 0:
                low = free if low is None else min(low, free)
                job["timings"]["min_available_gb"] = round(low / 1024, 1)
                if total:
                    job["timings"]["peak_used_gb"] = round((total - low) / 1024, 1)
            await asyncio.sleep(1.0 if self.platform == "mac" else 0.5)
