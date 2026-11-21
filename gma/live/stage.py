"""The live stage: a scene that plays as it is made, and is directed while it plays.

    stage = Stage(profile["live"], brain=..., hear=...)
    await stage.start("A woman in a red raincoat waits at a crossing at night ...")
    await stage.note("she walks toward the camera")      # typed, or the words heard from the microphone
    await stage.stop(); await stage.take()                # the scene again as one clean film

Or as on a set, with nothing filmed until the word is given:

    await stage.rehearse("A woman in a red raincoat ...")   # the table: the engine loads meanwhile, nothing is filmed
    await stage.note("Mara, you have waited an hour. How do you feel?")   # the plan grows; the character answers
    await stage.note("action")                            # the plan is filmed: its beats are given at their times
    await stage.note("cut")                               # it stops; the plan stays for the next "action"

With ``"render": "h3"`` (and the app's renderer given) the rehearsal has a picture and the take is Ghost Studio's own:
the set (gma/live/film.py) makes a reference shot of the opening picture and two loops of the character from it, so
the director talks to the person in the picture; "action" films the plan with the FastH3 clip worker, five seconds at
a time with the lines spoken in the picture, and "cut" keeps the parts already made. No stream engine is loaded.

The pictures come from the picture engine (gma/live/engine/worker.py: FastFrames as an endless stream), a child of
this service in its own container with no network, spoken to over a Unix socket. It is started when a scene is first
asked for, kept while someone watches, and stopped ``idle_s`` after the last scene. Its memory is its
own to answer for: it is not started under ``need_gb`` of free memory, and it is the first thing stopped when free
memory falls under ``floor_gb`` (above the app's own guard, so the stage goes and the app stays).

A note takes two steps, so the picture answers at once: the scene with the note tacked on goes to the engine straight
away, and the brain's rewrite (gma/live/director.py) replaces it when it is written.

The profile's "live" section (every key optional):

    {"home": "~/ff-try", "run": "~/ghost-studio/home/ghost/live", "image": "vllm-audio:local", "size": "640x480",
     "window": 12, "need_gb": 34, "floor_gb": 17, "idle_s": 300, "memory": "40g", "voices": ["af_heart", "am_michael"]}

The engine may run on another machine: with ``"host": "10.0.0.2"`` the container is started there over ssh (the app's
user must reach it without a password), its socket is forwarded here by ssh, and the memory rules are that machine's.
``home`` and the other paths are then that machine's, the engine's code is copied to ``engine`` there before each start,
and a take's files come back over the socket. The larger model wants its own numbers:

    {"host": "10.0.0.2", "model": "ff-l", "lean": true, "memory": "100g", "need_gb": 75, "floor_gb": 9,
     "load": "about five minutes"}
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shlex
import statistics
import struct
import time
from pathlib import Path

from . import director
from .blocking import Blocking, staging
from .clock import Clock
from .lips import Lips
from .film import PART_S, Set, SetError

ENGINE = Path(__file__).resolve().parent / "engine"
DEFAULTS = {"home": "~/ff-try", "run": "~/ghost-studio/home/ghost/live", "image": "vllm-audio:local", "name": "ghost-live-stage",
            "size": "640x480", "window": 12, "lead_s": 0.5, "need_gb": 34, "floor_gb": 17, "idle_s": 300, "memory": "40g",
            "start_s": 600, "fv_site": "~/ghost-studio/fv-venv/lib/python3.12/site-packages", "fv_src": "~/ghost-studio/fastvideo-src",
            "cuda": "/usr/local/cuda-13.0", "interim": True, "max_s": 600, "model": "ff-s", "sink": 3, "lean": False,
            "load": "about a minute", "host": "", "engine": "~/ghost-studio/live-engine", "host_run": "~/ghost-studio/live-run",
            "claim": "", "run_count_s": 1.0, "auto_take": False,
            "voices": ["af_heart", "am_michael", "bf_isabella", "bm_george", "af_bella", "am_fenrir"]}
FRAME_S = 0.25


def run_parts(cut_s: float) -> int:
    """The parts of a take filmed up to a cut: whole parts (the clip model's length), the number nearest the cut."""

    return max(1, round(float(cut_s) / PART_S))


def beat_key(beat: dict) -> str:
    return json.dumps([float(beat["s"]), str(beat["note"])])


class Refused(RuntimeError):
    """The stage cannot do this now; the message is for the person."""


def free_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    return 0.0


SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15"]


def launch(spec: dict) -> list[str]:
    """The picture engine's command: its container, with no network and nothing writable but its caches and run folder.
    On another machine (``host``) the same command is run there by ssh; ``there`` holds that machine's home and user."""

    host, there = spec.get("host"), spec.get("there") or {}

    def p(key):
        if not host:
            return str(Path(spec[key]).expanduser())
        return re.sub(r"^~(?=/|$)", there.get("home", "~"), str(spec[key]))
    home, run = p("home"), p("host_run" if host else "run")
    engine = p("engine") if host else str(ENGINE)
    user = f"{there['uid']}:{there['gid']}" if host else f"{os.getuid()}:{os.getgid()}"
    order = {"socket": "/run/live/stage.sock", "takes": "/run/live/takes", "size": spec["size"], "window": int(spec["window"]),
             "lead_s": float(spec["lead_s"]), "model": spec["model"], "sink": int(spec["sink"]), "lean": bool(spec["lean"]),
             "send_takes": bool(host)}
    command = ["docker", "run", "--rm", "--name", spec["name"], "--network", "none", "--read-only", "--tmpfs", "/tmp:exec,size=8g",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", user, "--pids-limit", "2048",
               "--memory", str(spec["memory"]), "--gpus", "all", "--ipc", "private",
               "-v", f"{home}/repo:/work:ro", "-v", f"{home}/checkpoints:/ckpt:ro", "-v", f"{home}/pydeps:/deps:ro",
               "-v", f"{home}/lib:/lib-h264:ro", "-v", f"{engine}:/engine:ro", "-v", f"{home}/opt/cache:/opt/ff/cache",  # the compiled pieces remember this path
               "-v", f"{home}/ficache:/tmp/.cache/flashinfer", "-v", f"{p('fv_site')}:/fv:ro", "-v", f"{p('fv_src')}:{p('fv_src')}:ro",
               "-v", f"{p('cuda')}:/usr/local/cuda:ro", "-v", f"{run}:/run/live",
               "-e", "PYTHONPATH=/engine:/fv:/deps", "-e", "HOME=/tmp", "-e", "OPENH264_LIBRARY=/lib-h264/libopenh264.so",
               "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "TORCHINDUCTOR_CACHE_DIR=/opt/ff/cache/inductor",
               "-e", "TRITON_CACHE_DIR=/opt/ff/cache/triton", "-e", "FF_KERNEL_DIR=/opt/ff/cache/kern", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "PATH=/fv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
               "-w", "/work", "--entrypoint", "python3", spec["image"], "/engine/worker.py", json.dumps(order)]
    if not host:
        return command
    remote = f"mkdir -p {shlex.quote(run)}/takes; rm -f {shlex.quote(run)}/stage.sock; "
    claim = p("claim") if spec.get("claim") else ""
    if claim:  # a line in that machine's log of who is using it, when the engine starts and when it has gone
        note = f"ghost-live: the live stage's picture engine ({spec['model']}, container {spec['name']})"
        remote += f"echo \"$(date -Is) {note} START\" >> {shlex.quote(claim)}; "
    remote += shlex.join(command)
    if claim:
        remote += f"; echo \"$(date -Is) {note} DONE\" >> {shlex.quote(claim)}"
    return [*SSH, host, remote]


class Stage:
    def __init__(self, spec: dict | None = None, brain=None, launcher=None, renderer=None) -> None:
        self.spec = {**DEFAULTS, **(spec or {})}
        self.brain = brain
        self.launcher = launcher or launch  # tests give their own engine
        self.run = Path(self.spec["run"]).expanduser()
        self.phase, self.detail = "off", ""  # off, starting, ready, live, taking, failed
        self.scene, self.cast, self.seed = "", [], 0
        self.log: list[dict] = []  # what was said and done, for the page
        self.script: list[list] = []  # [latent frame, scene] as the engine took them: the take's script
        self.frames, self.began, self.times = 0, 0.0, []
        self.takes: list[dict] = []
        try:  # the takes filmed before a restart are still in the list
            for film in sorted((self.run / "takes").glob("take-*.mp4"))[-12:]:
                if re.fullmatch(r"take-\d{8}-\d{6}", film.stem):
                    parts = len(list(film.parent.glob(f"{film.stem}-p*.mp4"))) or 1
                    self.takes.append({"name": film.stem, "state": "done", "done": parts, "frames": parts, "parts": [], "file": film.name,
                                       "seconds": round(parts * PART_S, 1), "number": None, "kept": True})
        except OSError:
            pass
        self.table: dict | None = None  # the rehearsed plan: {"scene": the opening picture, "beats": [{"s", "note"}], "seed", "text"}
        self.runner: asyncio.Task | None = None  # gives the plan's beats at their times while the scene plays
        self.shooting: Clock | None = None  # the parts of the take being filmed
        self.wrapping = False  # "cut" was called: the take ends after the part being filmed
        # the run-through: after "action" the plan plays in real time (the blocking's stick figures, the cast's lines at
        # their moments) and nothing is filmed; "cut" ends it, and then it is played again or filmed up to the cut
        self.through: dict | None = None  # {"state": preparing, playing, cut, ended; "film", "seconds", "cues", "cut_s"}
        self.through_task: asyncio.Task | None = None
        self.through_began, self.through_from = 0.0, "off"
        # what each beat does, by the scene before it and the beat: written once (when the note is given), so "action"
        # starts at once and the take films what the run-through played
        self.directed: dict[tuple[str, str], director.Direction] = {}
        self.preparing: asyncio.Task | None = None  # the beats being written ahead of "action"
        self.finding: asyncio.Task | None = None  # the cast being read from a set-up that named nobody
        self.redress = False  # the scene opens differently, said while a take was filmed: its reference shot after the take
        self.facing: str | None = None  # the character the director chose on the roster
        self.lips = Lips(self.spec["lipsync"], self.tell) if isinstance(self.spec.get("lipsync"), dict) else None
        self.set = Set(renderer, self.run, self.tell, lips=self.lips, roster=bool(self.spec.get("roster")),
                       size=tuple(self.spec["picture"]) if self.spec.get("picture") else None) \
            if renderer is not None and self.spec.get("render") == "h3" else None  # "picture": [width, height] of the still and the take
        self.filming: asyncio.Task | None = None  # the take being filmed by the set
        self.warming: asyncio.Task | None = None  # the brain being loaded for the rehearsal
        # the rehearsal's stick-figure floor plan (gma/live/blocking.py), where the profile has MotionBricks
        self.blocking = Blocking(self.spec["blocking"], self.run, self.tell, brain, free_gb=free_gb) \
            if isinstance(self.spec.get("blocking"), dict) else None
        self.extra: list[str] = []  # notes given while the set films: they go into the next part
        self.takes_n = 0
        self.watchers: set[asyncio.Queue] = set()
        self.process = self.tunnel = None
        self.host, self.host_free = self.spec.get("host") or "", None
        self.reader = self.writer = None
        self.tasks: list[asyncio.Task] = []
        self.sent: dict[int, dict] = {}  # directions sent to the engine, by id, until it says from which frame
        self.next_id, self.idle_since = 1, time.monotonic()
        self.lock, self.notes = asyncio.Lock(), asyncio.Lock()
        self.info: dict = {}

    # -- what the page sees --

    def state(self) -> dict:
        recent = self.times[-24:]
        ms = statistics.median(recent) if recent else None
        return {"phase": self.phase, "detail": self.detail, "scene": self.scene, "cast": self.cast, "seed": self.seed,
                "frames": self.frames, "seconds": round(self.frames * FRAME_S, 2), "ms": ms and round(ms, 1),
                "speed": ms and round(FRAME_S * 1e3 / ms, 2), "log": self.log[-60:], "takes": self.takes[-12:],
                "engine": self.info, "free_gb": round(self.host_free if self.host and self.host_free is not None else free_gb(), 1),
                "watchers": len(self.watchers), "table": self.table, "set": self.set.state() if self.set else None,
                "blocking": self.blocking.state() if self.blocking and self.table else None,
                "brain": "loading" if self.warming is not None and not self.warming.done() else "",
                "progress": self.progress(), "facing": self.to(), "run": self.run_state(), "wrapping": self.wrapping and self.rendering(),
                "rendering": self.rendering(), "casting": self.finding is not None and not self.finding.done(),
                "render": "h3" if self.set else "stream", "model": self.spec["model"], "load": self.spec["load"], "host": self.host}

    def run_state(self) -> dict | None:
        """The run-through for the page: its film and cues, and "t", the second it is at now (below zero while the
        count before it runs)."""

        run = self.through
        if run is None:
            return None
        return {**run, "t": round(time.monotonic() - self.through_began, 2) if run["state"] == "playing" else None}

    def progress(self) -> dict | None:
        """The steps being made now and how long each takes (gma/live/clock.py): the take's parts while one is filmed,
        else the set's (the reference shot, the blocking, the character's loops, the lips)."""

        if self.set is None:
            return None
        if self.rendering() and self.shooting is not None:
            return {"of": "take", "steps": self.shooting.state()}
        if self.table is None:
            return None
        clock = self.set.clock
        if self.blocking is not None:
            said = self.blocking.state()
            clock.seen("blocking", "running" if said["doing"] else "failed" if said["error"] else "done" if said["file"] else "waiting")
            if said["took_s"] and clock.steps.get("blocking", {}).get("state") == "done":
                clock.steps["blocking"]["took"] = float(said["took_s"])
        else:
            clock.steps.pop("blocking", None)
        if self.lips is not None:
            phase, faces = self.lips.phase, list(self.lips.faces.values())
            if self.set.roster:  # done when every close-up made so far has its lips, and all are made
                made = len(faces) >= len(self.cast) and faces and all(f in ("ready", "failed") for f in faces)
                clock.seen("lips", "failed" if phase == "failed" else "done" if made and phase == "ready" else "running" if "preparing" in faces or phase in ("starting", "preparing") else "waiting")
            else:
                clock.seen("lips", "running" if phase in ("starting", "preparing") else "done" if phase == "ready" else "failed" if phase == "failed" else "waiting")
        else:
            clock.steps.pop("lips", None)
        return {"of": "set", "steps": clock.state()}

    def rendering(self) -> bool:
        """A take is being filmed now (at the front, or behind the rehearsal)."""

        return self.filming is not None and not self.filming.done()

    def tell(self, event: dict | None = None) -> None:
        said = ("event", event or {"type": "state", **self.state()})
        for queue in list(self.watchers):
            if queue.full():
                continue
            queue.put_nowait(said)

    def record(self, kind: str, text: str, **more) -> dict:
        at = None if self.phase != "live" and kind != "stop" else round(self.frames * FRAME_S, 2)  # None: said at the table
        entry = {"kind": kind, "text": text, "at_s": at, "clock": time.time(), **more}
        self.log.append(entry)
        del self.log[:-400]
        return entry

    def watch(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=48)
        self.watchers.add(queue)
        if self.table is not None and self.phase != "filming":
            self.wake()
        return queue

    def unwatch(self, queue: asyncio.Queue) -> None:
        self.watchers.discard(queue)
        self.idle_since = time.monotonic()

    # -- the engine --

    async def _ssh(self, command: str, timeout: float = 20) -> str | None:
        """What ``command`` prints on the engine's machine; None when it could not be asked."""

        try:
            asked = await asyncio.create_subprocess_exec(*SSH, self.host, command, stdout=asyncio.subprocess.PIPE,
                                                         stderr=asyncio.subprocess.DEVNULL, stdin=asyncio.subprocess.DEVNULL)
            out, _ = await asyncio.wait_for(asked.communicate(), timeout)
        except (OSError, asyncio.TimeoutError):
            return None
        return out.decode(errors="replace") if asked.returncode == 0 else None

    async def free(self) -> float | None:
        """Free memory where the engine runs, in GB. None when the other machine did not answer."""

        if not self.host:
            return free_gb()
        said = await self._ssh("grep MemAvailable /proc/meminfo", 10)
        found = re.search(r"(\d+)", said or "")
        self.host_free = int(found.group(1)) / 1048576 if found else self.host_free
        return int(found.group(1)) / 1048576 if found else None

    async def _there(self) -> None:
        """The other machine made ready: who we are there, and this engine's code copied over."""

        said = (await self._ssh('echo "$HOME $(id -u) $(id -g)"') or "").split()
        if len(said) != 3:
            raise Refused(f"The picture engine's machine ({self.host}) does not answer.")
        self.spec["there"] = {"home": said[0], "uid": int(said[1]), "gid": int(said[2])}
        target = re.sub(r"^~(?=/|$)", said[0], str(self.spec["engine"]))
        copy = await asyncio.create_subprocess_exec(
            "rsync", "-a", "--delete", "--exclude", "__pycache__", "-e", " ".join(SSH), f"{ENGINE}/", f"{self.host}:{target}/",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, stdin=asyncio.subprocess.DEVNULL)
        if await asyncio.wait_for(copy.wait(), 120) != 0:
            raise Refused(f"The picture engine's code could not be copied to {self.host}.")

    async def ensure(self) -> None:
        """The engine up and answering (started here when it is not). Raises Refused with the reason when it cannot be."""

        async with self.lock:
            if self.writer is not None and self.phase not in ("off", "failed"):
                return
            await self._kill()  # an engine left behind by an app that died is ours to clear, and its memory ours to count
            free = await self.free()
            for _ in range(int(float(self.spec.get("settle_s", 30)) // 3)):  # an engine just stopped gives its memory back over some seconds
                if free is None or free >= float(self.spec["need_gb"]):
                    break
                await asyncio.sleep(3)
                free = await self.free()
            if free is None:
                raise Refused(f"The picture engine's machine ({self.host}) does not answer.")
            if free < float(self.spec["need_gb"]):
                raise Refused(f"There is not enough free memory for the live stage now: {free:.0f} GB free, it needs {self.spec['need_gb']}.")
            self.phase, self.detail = "starting", f"loading the picture engine ({self.spec['load']})"
            self.tell()
            (self.run / "takes").mkdir(parents=True, exist_ok=True)
            sock = self.run / "stage.sock"
            if sock.exists():
                sock.unlink()
            try:
                if self.host:
                    await self._there()
                log = open(self.run / "engine.log", "ab")
                mark = log.tell()
                self.process = await asyncio.create_subprocess_exec(*self.launcher(self.spec), stdout=log, stderr=log, stdin=asyncio.subprocess.DEVNULL)
                log.close()
                deadline, looked = time.monotonic() + float(self.spec["start_s"]), 0.0
                while not (self._up(mark) if self.host else sock.exists()):
                    if self.process.returncode is not None:
                        raise Refused("The picture engine stopped while it was loading; see the stage's engine.log.")
                    if time.monotonic() > deadline:
                        raise Refused("The picture engine did not come up in time.")
                    if time.monotonic() - looked > (10 if self.host else 0):
                        looked, free = time.monotonic(), await self.free()
                        if free is not None and free < float(self.spec["floor_gb"]):
                            raise Refused("Free memory fell too low while the picture engine was loading; it was stopped.")
                    await asyncio.sleep(0.5)
                if self.host:  # its socket, brought here by ssh
                    far = re.sub(r"^~(?=/|$)", self.spec["there"]["home"], str(self.spec["host_run"])) + "/stage.sock"
                    self.tunnel = await asyncio.create_subprocess_exec(
                        *SSH, "-N", "-o", "ExitOnForwardFailure=yes", "-o", "StreamLocalBindUnlink=yes", "-L", f"{sock}:{far}", self.host,
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, stdin=asyncio.subprocess.DEVNULL)
                    for _ in range(100):
                        if sock.exists() or self.tunnel.returncode is not None:
                            break
                        await asyncio.sleep(0.1)
                self.reader, self.writer = await asyncio.open_unix_connection(str(sock), limit=1 << 22)
                self.info = await asyncio.wait_for(self._read_header(), 30)
            except Exception as wrong:
                await self._kill()
                self.reader = self.writer = None
                self.phase, self.detail = "failed", str(wrong)
                self.tell()
                raise wrong if isinstance(wrong, Refused) else Refused(f"The picture engine could not be reached: {wrong}")
            self.phase, self.detail, self.idle_since = "ready", "", time.monotonic()
            self.tasks = [asyncio.create_task(self._listen()), asyncio.create_task(self._mind())]
            self.tell()

    def _up(self, mark: int) -> bool:
        """Has the engine said it is ready, in what it has printed since ``mark``? (Its socket is on the other machine.)"""

        with open(self.run / "engine.log", "rb") as log:
            log.seek(mark)
            return b"STAGE READY" in log.read()

    async def _read_header(self) -> dict:
        size = struct.unpack(">I", await self.reader.readexactly(4))[0]
        return json.loads(await self.reader.readexactly(size))

    async def _send(self, order: dict) -> None:
        if self.writer is None:
            raise Refused("The picture engine is not running.")
        head = json.dumps(order).encode()
        self.writer.write(struct.pack(">I", len(head)) + head)
        await self.writer.drain()

    async def _listen(self) -> None:
        """Everything the engine says: pictures to whoever watches, the rest into the state."""

        try:
            while True:
                header = await self._read_header()
                payload = await self.reader.readexactly(sum(header.get("sizes", []))) if header.get("sizes") else b""
                kind = header.get("type")
                if kind == "frame":
                    self.frames = header["f"] + 1
                    self.times.append(header["ms"])
                    del self.times[:-200]
                    head = json.dumps(header).encode()
                    packet = ("frame", struct.pack(">I", len(head)) + head + payload)
                    for queue in list(self.watchers):
                        if queue.qsize() < 40:  # a watcher that does not keep up loses pictures, the others none
                            queue.put_nowait(packet)
                    if self.frames % 8 == 0:
                        self.tell({"type": "stats", "frames": self.frames, "ms": round(statistics.median(self.times[-24:]), 1),
                                   "late_ms": header.get("late_ms", 0)})
                    if self.frames * FRAME_S >= float(self.spec["max_s"]):
                        asyncio.create_task(self.stop("the scene reached its longest length"))
                elif kind == "prompt":
                    sent = self.sent.pop(header["id"], None)
                    if sent:
                        self.script.append([header["at"], sent["scene"]])
                        sent["entry"].update(at_s=round(header["at"] * FRAME_S, 2), engine_ms=header["ms"])
                        self.tell()
                elif kind == "started":
                    self.phase = "live"
                    self.tell()
                elif kind == "stopped":
                    self.phase, self.idle_since = "ready", time.monotonic()
                    self.tell()
                elif kind == "take":
                    at = 0
                    for name, size in zip(header.get("files", []), header.get("sizes", [])):  # a take made on another machine
                        if re.fullmatch(r"take-[\w-]+\.(mp4|jpg)", name):
                            (self.run / "takes" / name).write_bytes(payload[at:at + size])
                        at += size
                    self._took(header)
                elif kind == "error":
                    self.record("error", header.get("detail", ""))
                    self.tell()
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except asyncio.CancelledError:
            raise
        if self.phase != "off":
            self.phase, self.detail = "failed", self.detail or "the picture engine stopped"
            self.writer = None
            self.tell()

    def _took(self, header: dict) -> None:
        take = next((t for t in self.takes if t["name"] == header["name"]), None)
        if take is None:
            return
        take.update({k: v for k, v in header.items() if k not in ("type", "name", "sizes", "files")})
        if header["state"] in ("done", "failed"):
            self.phase, self.idle_since = "ready", time.monotonic()
        self.tell()

    async def _mind(self) -> None:
        """Memory and idleness: the stage is the first thing to go when memory is short, and it does not stay loaded
        with nobody there."""

        try:
            turn = 0
            while self.writer is not None:
                await asyncio.sleep(2)
                turn += 1
                free = (await self.free()) if not self.host or turn % 5 == 0 else None  # the other machine is asked less often
                if free is not None and free < float(self.spec["floor_gb"]):
                    await self.shutdown(f"stopped: free memory fell under {self.spec['floor_gb']} GB", failed=True)
                    return
                idle = self.phase == "ready" and not self.watchers
                if idle and time.monotonic() - self.idle_since > float(self.spec["idle_s"]):
                    await self.shutdown("")
                    return
                if self.phase == "live" and not self.watchers and time.monotonic() - self.idle_since > 20:
                    await self.stop("nobody is watching")
        except asyncio.CancelledError:
            raise

    async def _kill(self) -> None:
        kill = ["docker", "kill", self.spec["name"]]
        try:
            killer = await asyncio.create_subprocess_exec(*([*SSH, self.host, shlex.join(kill)] if self.host else kill),
                                                          stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                                                          stdin=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(killer.wait(), 30)
        except (OSError, asyncio.TimeoutError):
            pass
        for process in (self.tunnel, self.process):  # the engine's own process last: over ssh it still has a closing line to write
            if process is not None and process.returncode is None:
                if process is self.process and self.host:
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                        continue
                    except asyncio.TimeoutError:
                        pass
                process.terminate()
        self.process = self.tunnel = None

    async def shutdown(self, why: str = "", failed: bool = False) -> None:
        current = asyncio.current_task()
        if self.set is not None:
            self.set.strike(all=True)
        if self.lips is not None:
            await self.lips.stop()
        if self.filming is not None:
            self.filming.cancel()
        for task in self.tasks:
            if task is not current:
                task.cancel()
        self.tasks = []
        if self.writer is not None:
            try:
                await self._send({"op": "quit"})
                self.writer.close()
            except (Refused, OSError):
                pass
        self.reader = self.writer = None
        await self._kill()
        self.phase, self.detail = ("failed" if failed else "off"), why
        self.tell()

    # -- directing --

    async def start(self, text: str, seed: int = 0, beats: list[dict] | None = None, opening: str | None = None) -> dict:
        """A new scene from what was typed (its cast as ``Name: look`` lines) begins to play. ``opening`` and ``beats``
        are a rehearsed plan's: the scene opens on that picture and the beats are given at their times."""

        text = (text or "").strip()
        if len(text.split()) < 3:
            raise Refused("Describe the scene in a sentence or two first.")
        if self.brain is not None:  # the brain may be unloaded: it loads while the engine and the scene start, not at the first note
            self.warming = asyncio.create_task(self._warm())
        await self.ensure()
        if self.phase == "taking":
            raise Refused("A take is being filmed; the stage is free again in a moment.")
        if self.phase == "live":
            await self.stop("a new scene")
        self.cast, self.scene, self.seed = director.cast_of(text), opening or director.opening(text), int(seed)
        self.script, self.frames, self.times, self.began = [[0, self.scene]], 0, [], time.time()
        if beats is None:
            self.log, self.table = [], None
        self.takes_n += 1
        self.record("scene", self.scene)
        await self._send({"op": "start", "prompt": self.scene, "seed": self.seed, "id": 0})
        for _ in range(200):  # the engine says "started" at once
            if self.phase == "live":
                break
            await asyncio.sleep(0.02)
        if beats:
            self.runner = asyncio.create_task(self._beats(list(beats)))
        self.tell()
        return self.state()

    # -- the rehearsal: the plan, then "action" and "cut" --

    async def rehearse(self, text: str, seed: int = 0) -> dict:
        """The table is set for a scene: nothing is filmed, the picture engine loads meanwhile, and notes now shape the
        plan (and the cast answer) until "action"."""

        text = (text or "").strip()
        if len(text.split()) < 3:
            raise Refused("Describe the scene in a sentence or two first.")
        if self.phase == "taking":
            raise Refused("A take is being filmed; the stage is free again in a moment.")
        if self.rendering():
            raise Refused("A take is still being filmed. Rehearse the new scene when it is done, or stop the take first.")
        if self.phase in ("live", "running"):
            await self.stop("a new rehearsal")
        for task in (self.preparing, self.finding):
            if task is not None and not task.done():
                task.cancel()
        self.through, self.directed, self.redress = None, {}, False
        self.cast, self.scene, self.seed = director.cast_of(text), director.opening(text), int(seed)
        self.table = {"scene": self.scene, "beats": [], "seed": int(seed), "text": text}
        self.log, self.script, self.frames, self.takes_n = [], [], 0, 0
        self.record("table", self.scene)
        if self.brain is not None:
            self.warming = asyncio.create_task(self._warm())
            if not self.cast or len(text.split()) > int(self.spec.get("story_words", 110)):
                # nobody was named on a line of their own, or a whole story was typed: the brain reads it first (who is
                # on screen, the picture it opens on, what happens after), and the set is dressed for that
                self.finding = asyncio.create_task(self._cast(text, self.table))
        if self.finding is None or self.finding.done():
            self._dress(self.table)
        self.tell()
        return self.state()

    def _dress(self, table: dict) -> None:
        if self.blocking is not None:
            self.blocking.draw(table["scene"], self.cast, table["beats"])
        if self.set is not None:
            self.set.dress(table["scene"], self.cast, table["seed"])  # the reference shot, then the character's loops
        else:
            self.loading = asyncio.create_task(self._load())

    async def _cast(self, text: str, table: dict) -> None:
        read = await director.reading(self.brain, text)
        if self.table is not table:
            return
        if read["cast"] and not self.cast:
            self.cast = read["cast"]
            self.record("cast", "The cast: " + "; ".join(f"{c['name']} ({c['look']})" for c in self.cast))
        if read["scene"]:  # a story: it opens on its first picture, and what happens after is the plan
            table["scene"], table["beats"] = read["scene"], read["beats"]
            self.scene = read["scene"]
            self.record("plan", f"Your scene as a plan: it opens on the first picture, then {len(read['beats'])} things happen. Change any of it by saying so.")
        self.directed = {}
        self._dress(table)
        self._prepare(table)
        self.tell()

    # -- what each beat does, written once --

    async def _beat(self, scene: str, beat: dict, keep: bool = True) -> director.Direction:
        """What ``beat`` does to ``scene`` and who speaks in it. Asked of the brain once for that scene and beat."""

        key = (scene, beat_key(beat))
        said = self.directed.get(key)
        if said is None:
            said = (await director.direct(self.brain, scene, beat["note"], self.cast)) if self.brain is not None else \
                director.Direction(director.interim(scene, beat["note"]), by="plain")
            if keep or said.by == "brain":  # written ahead with the brain away: asked again at "action"
                if len(self.directed) > 400:
                    self.directed.clear()
                self.directed[key] = said
        return said

    def _prepare(self, table: dict) -> None:
        """The plan moved: its beats are written now, behind the rehearsal, so "action" does not wait for them."""

        if self.preparing is not None and not self.preparing.done():
            self.preparing.cancel()
        self.preparing = asyncio.create_task(self._ahead(table, table["scene"], [dict(b) for b in table["beats"]])) if table["beats"] else None

    async def _ahead(self, table: dict, scene: str, beats: list[dict]) -> None:
        try:
            for beat in beats:
                if self.table is not table:
                    return
                scene = (await self._beat(scene, beat, keep=False)).scene
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - written at "action" instead
            pass

    def _scene_at(self, table: dict, t: float) -> str:
        """The scene as it stands ``t`` seconds in: after the beats before then, as far as they are written."""

        scene = table["scene"]
        for beat in table["beats"]:
            said = self.directed.get((scene, beat_key(beat))) if float(beat["s"]) <= t else None
            if said is None:
                break
            scene = said.scene
        return scene

    async def _load(self) -> None:
        try:
            await self.ensure()
        except Refused as wrong:
            self.record("error", str(wrong))
            self.tell()

    async def action(self, text: str = "", seed: int | None = None) -> dict:
        """ "Action": the rehearsed plan is filmed (or, with no rehearsal, the scene typed). Each "action" on the same
        plan is a new try: the next seed."""

        if self.phase in ("live", "filming", "running"):
            raise Refused("The scene is already playing; say cut first.")
        if self.table is None:
            return await self.start(text, int(seed or 0))
        table = self.table
        if self.set is not None and self.blocking is not None and self.spec.get("run_through", True):
            return self._run(table)  # played in real time first; "cut" then asks: again, or film it
        self.record("action", f"take {self.takes_n + 1}")
        self.tell()
        if self.set is not None:
            return self._roll(table)
        return await self.start(table["text"], table["seed"] + self.takes_n, beats=table["beats"], opening=table["scene"])

    def _run(self, table: dict) -> dict:
        """The run-through begins: the plan is played in real time on the page and nothing is filmed."""

        if self.through_task is not None and not self.through_task.done():
            self.through_task.cancel()
        self.through_from, self.phase = (self.phase if self.phase != "running" else self.through_from), "running"
        self.through_n = getattr(self, "through_n", 0) + 1
        self.through = {"n": self.through_n, "state": "preparing", "film": None, "seconds": 0, "cues": [], "cut_s": None, "count_s": float(self.spec.get("run_count_s", 3))}
        self.record("action", "run-through")
        self.through_task = asyncio.create_task(self._through(table, self.through))
        self.tell()
        return self.state()

    async def _through(self, table: dict, run: dict) -> None:
        try:
            drawing = self.blocking.task if self.blocking is not None else None
            if drawing is not None and not drawing.done():  # the floor plan of the last note is still being drawn
                await asyncio.wait({drawing}, timeout=float(self.spec.get("run_wait_s", 60)))
            ahead = self.preparing
            if ahead is not None and not ahead.done():  # the beats are being written already: the last of them is waited for
                await asyncio.wait({ahead}, timeout=float(self.spec.get("run_wait_s", 60)))
            scene, cues = table["scene"], []
            for beat in list(table["beats"]):  # what each beat does and who speaks in it, once: the take films the same
                said = await self._beat(scene, beat)
                scene = said.scene
                cues.append({"s": float(beat["s"]), "note": beat["note"], "who": said.who if said.line else None, "line": said.line or "",
                             "voice": self.voice_of(said.who) if said.line else None})
            if self.through is not run or self.table is not table:
                return
            shown = self.blocking.state() if self.blocking is not None else {}
            last = max([float(b["s"]) for b in table["beats"]] + [0.0])
            seconds = float(shown.get("seconds") or 0) if shown.get("file") else 0.0
            run.update(state="playing", film=shown.get("file"), seconds=seconds or min(60.0, max(10.0, last + 5)), cues=cues)
            self.through_began = time.monotonic() + run["count_s"]
            self.tell()
            await asyncio.sleep(run["count_s"] + run["seconds"])
            self._wrap(run, "ended", run["seconds"])
        except asyncio.CancelledError:
            raise
        except Exception as wrong:  # noqa: BLE001 - told on the page; the table is as it was
            if self.through is run:
                self.through, self.phase = None, self.through_from
                self.record("error", f"The run-through could not be played: {str(wrong)[:200]}")
                self.tell()

    def _wrap(self, run: dict, how: str, at: float) -> None:
        """The run-through is over ("cut" or "ended") ``at`` its second: back at the table, where it is played again or filmed."""

        if self.through is not run or run["state"] != "playing":
            return
        run.update(state=how, cut_s=round(max(0.0, min(float(at), run["seconds"])), 1))
        self.phase, self.idle_since = self.through_from, time.monotonic()
        self.record("stop", "cut" if how == "cut" else "the run-through ended")
        self.tell()
        # "auto_take": what was just played is filmed at once, behind the rehearsal, and the page plays it when it is
        # made; a cut in the first seconds is a false start, and one take is filmed at a time
        if self.spec.get("auto_take") and run["cut_s"] >= float(self.spec.get("auto_take_min_s", 2.0)) \
                and self.set is not None and not self.rendering():
            self.rolling_next = asyncio.get_running_loop().create_task(self._take_it(run))

    async def _take_it(self, run: dict) -> None:
        try:
            if self.through is run and self.phase not in ("live", "filming", "running"):
                await self.film()
        except Refused:
            pass
        self.tell()

    async def film(self) -> dict:
        """The run-through as it was played is filmed, up to where it was cut."""

        if self.table is None or self.set is None:
            raise Refused("Rehearse a scene first.")
        if self.phase in ("live", "filming", "running"):
            raise Refused("The scene is already playing; say cut first.")
        run = self.through
        until = run["cut_s"] if run is not None and run["state"] in ("cut", "ended") else None
        self.record("action", f"take {self.takes_n + 1}")
        self.through = None
        return self._roll(self.table, until=until, behind=bool(self.spec.get("behind", True)))

    def _roll(self, table: dict, until: float | None = None, behind: bool = False) -> dict:
        """The set films the plan: the take begins and this returns; its parts arrive as they are made. ``until``: the
        second the run-through was cut at; the take is as long, in whole parts, and has the beats before it."""

        if self.phase == "taking":
            raise Refused("A take is being filmed; the stage is free again in a moment.")
        if self.rendering():
            raise Refused("The take before this one is still being filmed. This one can be filmed when it is done.")
        self.takes_n += 1
        name = time.strftime("take-%Y%m%d-%H%M%S")
        # the take films the plan as it stands now: the rehearsal goes on behind it and may change the table
        table = {**table, "beats": [dict(b) for b in table["beats"] if until is None or float(b["s"]) < max(until, 0.1)], "until": until,
                 "floor": self.blocking.plan if self.blocking is not None and not self.blocking.doing else None}
        # after a run-through a draft is filmed behind the rehearsal, which goes on (notes, the cast, the next run-through); a sharp take
        # has the brain's room, so nothing can be rehearsed meanwhile and it has the stage
        behind = behind and not (self.set.quality == "sharp" and self.set.sharp)
        last = table["beats"][-1]["s"] if table["beats"] else 0
        parts = max(1, min(int(self.spec.get("max_parts", 8)), int(last // PART_S) + 1 if until is None else run_parts(until)))
        take = {"name": name, "state": "rendering", "done": 0, "frames": parts, "parts": [], "scene": table["scene"],
                "directions": len(table["beats"]), "number": self.takes_n, "behind": behind}
        self.takes.append(take)
        self.extra, self.wrapping = [], False
        if not behind:
            self.phase = "filming"
        self.shooting = Clock()
        self.shooting.usual = self.set.clock.usual  # what a clip takes is learned from the loops and the takes alike
        self.shooting.file = self.set.clock.file
        self.shooting.plan(([(f"end-{n + 1}", "face", f"Part {n + 1}: end picture") for n in range(parts)] if self.spec.get("anchors", True) else [])
                           + [(f"part-{n + 1}", "sharp" if self.set.quality == "sharp" and self.set.sharp else "clip", f"Part {n + 1}") for n in range(parts)])
        self.set.strike()  # the take has the machine: loops still being made are left
        self.filming = asyncio.create_task(self._film(table, take, table["seed"] + self.takes_n - 1))
        self.tell()
        return self.state()

    async def _film(self, table: dict, take: dict, seed: int) -> None:
        scene, beats, parts, made, cut = table["scene"], list(table["beats"]), take["frames"], [], False
        folder = self.run / "takes"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            # the plan of every part first: what is on screen after its beats, its lines, its walks and turns
            plans = []
            for part in range(parts):
                now = [b for b in beats if part * PART_S <= b["s"] < (part + 1) * PART_S or (part == parts - 1 and b["s"] >= parts * PART_S)]
                if part == 0:
                    now, self.extra = now + [{"s": -1.0, "note": note} for note in self.extra], []
                due, lines = [b["note"] for b in now], []
                for beat in now:  # each beat rewrites the picture; a beat for the cast gives the line spoken in it
                    note = beat["note"]
                    said = await self._beat(scene, beat)  # as the run-through played it: written once for this scene and beat
                    scene = said.scene
                    self.record("note", note, planned=True, scene=said.scene, by=said.by, part=part + 1)
                    if said.line:
                        lines.append({"speaker": said.who, "voice": "a natural voice", "text": said.line})
                        self.record("line", said.line, who=said.who, film=True)
                    self.tell()
                # the blocking the director saw is what is filmed: its walks and turns in this part, in words
                moves = staging(table["floor"], part * PART_S, (part + 1) * PART_S if part < parts - 1 else table.get("until") or 1e9) \
                    if table.get("floor") and self.spec.get("staging", True) else ""
                plans.append({"scene": scene, "lines": lines, "due": due, "moves": moves, "first": None, "last": None})
            # then the picture each part ends on (gma/live/film.py ENDS), so every part is filmed between two fixed
            # pictures and faces do not wear from part to part; a picture that fails leaves the parts from there chained
            if self.spec.get("anchors", True):
                opens = await self.set.ready()
                for part, plan in enumerate(plans):
                    key = f"end-{part + 1}"
                    self.shooting.begin(key)
                    self.tell()
                    try:
                        plan["first"] = opens
                        opens = plan["last"] = await self.set.ends(plan["scene"], self.cast, seed + part, opens, folder / f"{take['name']}-e{part + 1}.png",
                                                                   f"The picture part {part + 1} ends on")
                        self.shooting.end(key)
                    except SetError:
                        self.shooting.end(key, failed=True)
                        for later in range(part + 1, parts):
                            self.shooting.steps.pop(f"end-{later + 1}", None)
                        break
                    if self.wrapping:  # "cut" already: the first part is still filmed, so there is a film
                        for later in range(part + 1, parts):
                            self.shooting.steps.pop(f"end-{later + 1}", None)
                        break
            previous = None
            for part, plan in enumerate(plans):
                extra, self.extra = self.extra, []  # what the director said while the part before was filmed
                what = f"Filming part {part + 1} of {parts}"
                self.shooting.begin(f"part-{part + 1}", self.set.cold())
                self.tell()
                previous = await self.set.shoot({"scene": plan["scene"], "action": " ".join([plan["moves"], *extra]).strip(), "dialogue": plan["lines"]},
                                                seed + part, what, previous, moves=" ".join(plan["due"] + extra), first=plan["first"], last=plan["last"])
                file = folder / f"{take['name']}-p{part + 1}.mp4"
                await self.set.renderer.fetch(previous, file)
                made.append(file)
                self.shooting.end(f"part-{part + 1}")
                take.update(done=part + 1, parts=[f.name for f in made])
                self.tell()
                if self.wrapping:  # "cut" was called during this part: the take ends on it
                    cut = True
                    for later in range(part + 1, parts):
                        self.shooting.steps.pop(f"part-{later + 1}", None)
                    break
        except asyncio.CancelledError:
            cut = True
        except (SetError, Exception) as wrong:  # noqa: BLE001 - told on the page; the parts already made are kept
            take["detail"] = str(wrong)[:300]
            self.record("error", take["detail"])
        self.set.say("")
        try:
            if made:
                await self.set.join(made, folder / f"{take['name']}.mp4")
                take.update(state="done", file=f"{take['name']}.mp4", seconds=round(len(made) * PART_S, 1), cut=cut)
            else:
                take.update(state="failed", detail=take.get("detail") or "cut before the first part was filmed")
        except SetError as wrong:
            take.update(state="failed", detail=str(wrong))
        self.filming, self.idle_since = None, time.monotonic()
        if self.phase == "filming":
            self.phase = "ready" if self.writer is not None else "off"
        if self.table is not None and self.redress:  # the scene opens differently since: its reference shot now
            self.redress = False
            self.set.dress(self.table["scene"], self.cast, self.table["seed"], keep=True)
        elif self.table is not None:
            self.set.resume()  # back at the table: the character is there again
            if self.brain is not None and (self.warming is None or self.warming.done()):
                # a sharp take had the brain's room: it loads again now, and the page says so until it answers
                self.warming = asyncio.create_task(self._warm())
        self.tell()

    async def _beats(self, beats: list[dict]) -> None:
        """The plan's beats, each given as a note when the scene reaches its time (the scene's own clock, not the wall's)."""

        try:
            for beat in beats:
                while self.phase == "live" and self.frames * FRAME_S < beat["s"]:
                    await asyncio.sleep(0.05)
                if self.phase != "live":
                    return
                try:
                    await self.note(beat["note"], planned=True)
                except Refused:
                    return
        except asyncio.CancelledError:
            raise

    def _speak(self, said) -> None:
        """A character's line into the log, with that character's voice: the page speaks it."""

        if not said.line:
            return
        self.record("line", said.line, who=said.who, voice=self.voice_of(said.who))

    def to(self) -> str | None:
        """The character the director faces: the one chosen on the roster while they are in the cast, else the first."""

        names = [c["name"] for c in self.cast]
        return self.facing if self.facing in names else names[0] if names else None

    def voice_of(self, who: str | None) -> str:
        """A character's voice: by their place in the cast, a woman's or a man's as their look says."""

        voices, names = self.spec["voices"], [c["name"].lower() for c in self.cast]
        index = names.index(who.lower()) if who and who.lower() in names else len(names)
        look = f" {self.cast[index]['look'].lower()} " if index < len(self.cast) else ""
        kind = "f" if re.search(r"\b(woman|girl|she|her|lady|mother|actress|female|queen|princess|witch)\b", look) else "m" if re.search(r"\b(man|boy|he|him|his|father|gentleman|male|guy|king|prince|knight|warrior)\b", look) else ""
        fitting = [v for v in voices if len(v) > 1 and v[1] == kind] or voices
        taken = sum(1 for c in self.cast[:index] if kind and re.search(r"\b(woman|girl|she|her|lady|mother|actress|female|queen|princess|witch)\b" if kind == "f" else r"\b(man|boy|he|him|his|father|gentleman|male|guy|king|prince|knight|warrior)\b", f" {c['look'].lower()} "))
        return fitting[(taken if kind else index) % len(fitting)]

    def wake(self) -> None:
        """Somebody is at the table again: the lip-sync service its idle time stopped follows the character again."""

        lips, loop = self.lips, self.set.loops.get("idle") if self.set is not None else None
        if lips is None or lips.phase not in ("off", "failed") or self.set is None:
            return
        if self.set.roster:
            if not any(not t.done() for t in lips.casting):
                for who, name in self.set.faces.items():
                    film = self.set.file(Path(name).with_suffix(".mp4").name)
                    if film is not None:
                        lips.face(who, film)
        elif loop and (lips.task is None or lips.task.done()) and self.set.file(loop):
            lips.follow(self.set.file(loop))

    async def _plan(self, text: str, heard: bool, overheard: bool = False) -> dict:
        """A note at the table: the plan after it, and the character's answer when one was spoken to."""

        entry = self.record("note", text, heard=heard)
        self.wake()
        self.tell()
        async with self.notes:
            table = self.table
            began = time.monotonic()
            said = (await director.rehearse(self.brain, table["scene"], table["beats"], text, self.cast, overheard=overheard, to=self.to())) if self.brain is not None else \
                director.plainly(table["scene"], table["beats"], text)
            entry["brain_ms"] = round((time.monotonic() - began) * 1e3)
            if said.ignored:  # talk in the room, not a direction: it leaves the log again
                if entry in self.log:
                    self.log.remove(entry)
                self.tell()
                return {"ignored": True}
            if self.table is not table:
                raise Refused("The rehearsal was replaced.")
            reopened = said.scene != table["scene"]
            moved = reopened or said.beats != table["beats"]
            if moved:
                self.through = None  # the plan changed: the run-through shown was of the old one
            table["scene"], table["beats"] = said.scene, said.beats
            if moved and self.blocking is not None:  # the floor plan follows the plan
                self.blocking.draw(said.scene, self.cast, said.beats)
            if moved:
                self._prepare(table)  # what each beat does is written now, not at "action"
            self.scene = said.scene
            if reopened and self.set is not None and self.rendering():
                self.redress = True  # the take being filmed has the pictures: the new reference shot after it
            elif reopened and self.set is not None:  # the scene opens differently: a new reference shot, of the same people
                self.set.dress(said.scene, self.cast, table["seed"], keep=True)
            entry.update(by=said.by, ack=said.ack)
            self._speak(said)
            self.tell()
            return {"scene": said.scene, "beats": said.beats, "who": said.who, "line": said.line, "ack": said.ack, "by": said.by}

    async def _warm(self) -> None:
        try:
            await self.brain.complete([{"role": "user", "content": "Say: ready"}], max_tokens=1, timeout=300)
        except Exception:  # noqa: BLE001 - a brain that is away: notes are then taken plainly
            pass
        self.tell()

    async def _direct(self, scene: str, entry: dict) -> None:
        number, self.next_id = self.next_id, self.next_id + 1
        self.sent[number] = {"scene": scene, "entry": entry}
        await self._send({"op": "prompt", "text": scene, "id": number})

    async def note(self, text: str, heard: bool = False, planned: bool = False, overheard: bool = False) -> dict:
        """A direction, typed or heard. While the scene plays it changes the picture; at the table it changes the plan.
        The words "action" and "cut" are orders. Returns {"scene", "who", "line", "ack", "by"} (and "command" for an order)."""

        text = " ".join((text or "").split())
        if not text:
            raise Refused("I did not catch a note.")
        order = None if planned else director.command(text, heard, self.phase in ("live", "filming"))
        ran = self.through is not None and self.through["state"] in ("cut", "ended") and self.phase != "running"
        if order is None and not planned and ran:
            order = director.after_run(text)  # "render it", "again": what follows a run-through
        if order == "cut":
            if self.phase not in ("live", "filming", "running") and not self.rendering():
                raise Refused("Nothing is being filmed.")
            await self.stop("cut")
            return {"command": "cut", "ack": "Cut."}
        if order == "action":
            if self.table is None:
                raise Refused("Set the scene first, then say action.")
            await self.action()
            return {"command": "action", "ack": "Action."}
        if order == "film":
            await self.film()
            return {"command": "film", "ack": "Filming it."}
        if self.phase == "running":  # the run-through plays: what is said now is said to the cast, who answer on the spot
            return await self._aside(text, heard, overheard)
        if self.phase == "filming" and overheard:
            return {"ignored": True}  # while a take is filmed the open microphone takes orders only
        if self.phase == "filming":  # the set films in parts: the note is played in the next one
            self.extra.append(text)
            self.record("note", text, heard=heard, queued=True)
            self.tell()
            return {"ack": "In the next part.", "by": "queued"}
        if self.phase != "live":
            if self.table is None:
                raise Refused("Start or rehearse a scene first; then direct it.")
            return await self._plan(text, heard, overheard)
        entry = self.record("note", text, heard=heard, planned=planned)
        self.tell()
        async with self.notes:  # one note at a time: the second is written from the first one's scene
            if self.phase != "live":
                raise Refused("The scene has ended.")
            if self.spec.get("interim", True):
                await self._direct(director.interim(self.scene, text), {})
            began = time.monotonic()
            said = (await director.direct(self.brain, self.scene, text, self.cast)) if self.brain is not None else \
                director.Direction(director.interim(self.scene, text), by="plain", ack="Okay.")
            entry["brain_ms"] = round((time.monotonic() - began) * 1e3)
            if self.phase != "live":
                raise Refused("The scene has ended.")
            self.scene = said.scene
            await self._direct(said.scene, entry)
            entry.update(scene=said.scene, by=said.by, ack=said.ack)
            self._speak(said)
            self.tell()
            return {"scene": said.scene, "who": said.who, "line": said.line, "ack": said.ack, "by": said.by}

    async def _aside(self, text: str, heard: bool, overheard: bool) -> dict:
        """Words to the cast while the run-through plays: the character faced (or named) answers at once, and the
        answer is in the plan at that second from then on, so a retake and the take have it too."""

        run, table = self.through, self.table
        if run is None or run["state"] != "playing" or self.brain is None or not self.cast:
            if overheard:
                return {"ignored": True}
            raise Refused("The run-through is playing. Say cut, then give the note.")
        spoken = [c["line"] for c in run["cues"] if c.get("line")] + [e["text"] for e in self.log[-12:] if e["kind"] == "line"]
        if overheard and (director.chatter(text) or director.echo(text, spoken)):
            return {"ignored": True}  # the cast's own voices through the microphone, or a filler word
        low = text.lower()
        who = next((c["name"] for c in self.cast if re.search(rf"\b{re.escape(c['name'].lower())}\b", low)), None) or self.to()
        began = time.monotonic()
        line = await director.answer(self.brain, self._scene_at(table, began - self.through_began), self.cast, who, text, overheard=overheard)
        if self.through is not run or run["state"] != "playing" or self.table is not table:
            return {"ignored": True}  # it was cut meanwhile
        if line is None:
            if overheard:
                return {"ignored": True}
            raise Refused(f"{who} did not answer. Say cut, then give the note.")
        at = round(max(0.0, min(time.monotonic() - self.through_began, run["seconds"])), 1)
        self.record("note", text, heard=heard, aside=True, brain_ms=round((time.monotonic() - began) * 1e3))
        beat = director.beats_of([{"s": at, "note": f'{who} says: "{line}"'}])[0]
        table["beats"] = director.beats_of([*table["beats"], beat])
        if beat in table["beats"]:  # the line leaves the picture as it is: the beats after it are written as before
            scene = self._scene_at({**table, "beats": [b for b in table["beats"] if b != beat]}, at)
            self.directed[(scene, beat_key(beat))] = director.Direction(scene, who, line, by="brain")
            run["cues"] = sorted([*run["cues"], {"s": at, "note": beat["note"], "who": who, "line": line, "voice": self.voice_of(who), "aside": True}],
                                 key=lambda c: c["s"])
        self.record("line", line, who=who, voice=self.voice_of(who))
        self.tell()
        return {"who": who, "line": line, "ack": "", "by": "brain", "aside": True}

    async def stop(self, why: str = "") -> dict:
        if self.runner is not None and self.runner is not asyncio.current_task():
            self.runner.cancel()
        self.runner = None
        if self.phase == "running":
            run, task = self.through, self.through_task
            if task is not None and not task.done() and task is not asyncio.current_task():
                task.cancel()
            if run is not None and run["state"] == "playing":
                self._wrap(run, "cut", time.monotonic() - self.through_began)
            else:  # cut while it was being prepared: nothing was played
                self.through, self.phase = None, self.through_from
                self.record("stop", why or "stopped")
                self.tell()
            return self.state()
        if self.phase != "live" and self.rendering() and not self.wrapping and self.spec.get("cut_keeps", True):
            # "cut": the part being filmed is finished (the clip model cannot stop halfway) and the take ends there,
            # so a cut always gives a film; a second "cut" drops the part instead
            self.wrapping = True
            self.record("stop", why or "stopped")
            self.tell()
            return self.state()
        if self.phase != "live" and self.rendering():
            if not self.wrapping:
                self.record("stop", why or "stopped")
            filming = self.filming
            filming.cancel()
            await asyncio.gather(filming, return_exceptions=True)  # the parts already made are joined
            if self.takes and self.takes[-1]["state"] == "rendering":  # cut before the take's first step: nothing was made
                self.takes[-1].update(state="failed", detail="cut before the first part was filmed")
                self.filming = None
                if self.phase == "filming":
                    self.phase = "ready" if self.writer is not None else "off"
                if self.table is not None:
                    self.set.resume()
                self.tell()
            return self.state()
        if self.phase == "live":
            self.record("stop", why or "stopped")
            await self._send({"op": "stop"})
            for _ in range(100):  # the engine says "stopped" between two latent frames
                if self.phase != "live":
                    break
                await asyncio.sleep(0.05)
        return self.state()

    async def take(self) -> dict:
        """The scene just rehearsed, filmed again from its first frame as one clean file: the same seed and the same
        directions at the same moments, with nothing lost to the live stream."""

        if self.phase == "live":
            await self.stop("take")
        if self.phase != "ready" or not self.script or self.frames < 4:
            raise Refused("There is no rehearsed scene to film yet.")
        name = time.strftime("take-%Y%m%d-%H%M%S")
        take = {"name": name, "state": "rendering", "done": 0, "frames": self.frames, "scene": self.script[0][1],
                "directions": len(self.script) - 1}
        self.takes.append(take)
        self.phase = "taking"
        (self.run / "takes" / f"{name}.json").write_text(json.dumps({"seed": self.seed, "frames": self.frames, "script": self.script,
                                                                    "log": self.log, "size": self.spec["size"]}, indent=1))
        await self._send({"op": "take", "name": name, "seed": self.seed, "frames": self.frames, "script": self.script})
        self.tell()
        return take

    def take_file(self, name: str) -> Path | None:
        path = self.run / "takes" / name
        return path if path.is_file() and path.parent == self.run / "takes" else None
