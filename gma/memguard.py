"""Ghost Studio's memory guard (2026-11-08, after a big-picture film hung the DGX Spark: the kernel's NVRM driver ran out of memory
(NV_ERR_NO_MEMORY) during a clip while MemAvailable was still above the external guard's 12 GB floor; memory went too
fast for it). The same design as the Imagination Studio guard:

- **Admission** (gma/governor.py ``admit`` and ``enter``): before a load or a render its peak is compared with
  MemAvailable less the floor; idle Ghost-Studio models are released first, in ``release_order`` (biggest idle first);
  a load that still does not fit waits up to ``admit_wait_s`` for the room, then is refused with the biggest memory
  users named. A load that will not fit is never started.
- **Live guard** (``LiveGuard``, a thread in the app): every ``poll_s`` (1 s) it reads MemAvailable and the kernel's
  memory pressure (``/proc/pressure/memory``, "some avg10"). Under the floor (20 GB) or over the pressure limit (20) it
  acts: first the next idle model of this app is released; when none is left, the current step is stopped (the clip
  worker ends, and the run fails cleanly with the reason). Every action is logged (``memguard.jsonl``).
- **Foreign processes** (not started by Ghost Studio) are named in the messages, never killed, unless the profile's
  ``governor.memguard.kill_foreign`` lists a pattern of their command line (opt-in).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

DEFAULTS = {"poll_s": 1.0, "floor_gb": 20.0, "psi_limit": 20.0, "psi_mem_gb": 35.0, "cooldown_s": 4.0, "admit_wait_s": 0.0, "kill_foreign": []}
# the thermal gate and stop (2026-11-08: the FP4 worker alone holds .3's GPU at 88-89 C with growing thermal
# slowdown on back-to-back clips; two drafts froze on their third clip): governor.thermal in the profile
THERMAL = {"enabled": True, "gpu_max_c": 70.0, "zone_max_c": 85.0, "poll_s": 2.0, "timeout_s": 300.0, "hard_c": 88.0,
           "stop_gpu_c": 90.0, "stop_throttle_s": 10.0, "big_load_gb": 20.0}
HW_THERMAL = 0x8 | 0x40  # clocks_event_reasons: HW slowdown, HW thermal slowdown
LAST_STOP: dict = {}  # the live guard's last stop of a step: {"kind": "thermal"|"memory", "at": time, "why"}


def options(spec: dict | None) -> dict:
    return {**DEFAULTS, **((spec or {}).get("memguard") or {})}


def thermal_options(spec: dict | None) -> dict:
    """On where the profile's governor has "thermal" (the Spark profiles do); off otherwise (tests, a Mac)."""

    given = (spec or {}).get("thermal")
    return {**THERMAL, "enabled": given is not None, **(given or {})}


def sensors() -> dict:
    """{"gpu_c", "zone_c" (the hottest /sys/class/thermal zone), "reasons" (the clock-event bits)}; None where unread."""

    out = {"gpu_c": None, "zone_c": None, "reasons": None}
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks_event_reasons.active", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0].split(",")
        out["gpu_c"] = float(line[0])
        out["reasons"] = int(line[1].strip(), 16)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(z.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    out["zone_c"] = max(temps) if temps else None
    return out


def hottest(s: dict) -> float:
    return max([x for x in (s.get("gpu_c"), s.get("zone_c")) if x is not None] or [0.0])


class TooHot(Exception):
    """The machine stayed too hot to start the step (the gate's hard limit after its timeout)."""


def thermal_gate(spec: dict | None, what: str, read=None, sleep=None, clock=None, log=None, say=None) -> dict:
    """Before a clip, a keyframe or a big model load: wait until the GPU is at or under ``gpu_max_c`` and the hottest
    zone at or under ``zone_max_c`` (every ``poll_s``, at most ``timeout_s``). After the timeout the step starts anyway
    with a warning, unless the hottest reading is at or over ``hard_c``: then TooHot. Returns the wait's record."""

    read, sleep, clock = read or sensors, sleep or time.sleep, clock or time.time
    o = thermal_options(spec)
    first = read()
    began = clock()
    record = {"step": what, "start": first, "waited_s": 0.0, "outcome": "ok"}
    cool = lambda s: (s.get("gpu_c") is None or s["gpu_c"] <= o["gpu_max_c"]) and (s.get("zone_c") is None or s["zone_c"] <= o["zone_max_c"])
    now = first
    if not o["enabled"] or cool(first):
        record["end"] = first
        return record
    while not cool(now) and clock() - began < o["timeout_s"]:
        if say is not None:
            say(f"cooling the GPU before {what}: {now.get('gpu_c') or 0:.0f} \u00b0C -> {o['gpu_max_c']:.0f} \u00b0C"
                f" (hottest zone {now.get('zone_c') or 0:.0f} \u00b0C)")
        sleep(o["poll_s"])
        now = read()
    record.update(end=now, waited_s=round(clock() - began, 1))
    if not cool(now):
        if hottest(now) >= o["hard_c"]:
            record["outcome"] = "too hot"
            if log is not None:
                log("thermal gate: too hot to start", **record)
            raise TooHot(f"the machine stayed too hot to start {what}: GPU {now.get('gpu_c')} C, hottest zone {now.get('zone_c')} C "
                         f"after {record['waited_s']:.0f} s (the limit is {o['hard_c']:g} C)")
        record["outcome"] = "timeout: started warm"
    if log is not None:
        log("thermal gate: waited", **record)
    return record


def pressure(path: str = "/proc/pressure/memory") -> float | None:
    """The kernel's memory pressure: "some avg10" (the share of the last 10 s some task stalled on memory), or None."""

    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("some"):
                for part in line.split():
                    if part.startswith("avg10="):
                        return float(part.split("=", 1)[1])
    except (OSError, ValueError):
        return None
    return None


def ancestors(pid: int) -> list[int]:
    out = []
    for _ in range(64):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid <= 1:
            break
        out.append(pid)
    return out


def ours(pid: int, me: int | None = None) -> bool:
    """Started by this app (it or one of its children, at any depth)."""

    me = me or os.getpid()
    return pid == me or me in ancestors(pid)


def gpu_users() -> dict[int, int]:
    """pid -> MB on the GPU (nvidia-smi; empty where there is none)."""

    try:
        done = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for line in done.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = int(parts[1])
    return out


def command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def rss_mb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return 0


def biggest(top: int = 5, gpu: dict[int, int] | None = None, me: int | None = None) -> list[dict]:
    """The biggest memory users: the GPU's processes by their GPU memory, and the rest by resident size."""

    gpu = gpu_users() if gpu is None else gpu
    seen = []
    for pid, mb in gpu.items():
        seen.append({"pid": pid, "mb": mb, "where": "gpu", "ours": ours(pid, me), "what": command(pid)[:120]})
    pids = [int(p) for p in os.listdir("/proc") if p.isdigit()] if os.path.isdir("/proc") else []
    for pid in pids:
        if pid in gpu:
            continue
        mb = rss_mb(pid)
        if mb >= 1024:
            seen.append({"pid": pid, "mb": mb, "where": "ram", "ours": ours(pid, me), "what": command(pid)[:120]})
    return sorted(seen, key=lambda x: -x["mb"])[:top]


def named(users: list[dict]) -> str:
    return "; ".join(f"{u['what'].split(' ')[0].rsplit('/', 1)[-1] or u['pid']} (pid {u['pid']}, {u['mb'] / 1024:.1f} GB"
                     f"{', ' + u['where']}{', not ours' if not u['ours'] else ''})" for u in users)


def idle_engines(registry, keep: set | None = None) -> list:
    """This app's loaded models that answer nothing now, biggest first (the release order): never an adopted server,
    an in-process one, the renderer, or one the current step keeps."""

    keep = keep or set()
    out = []
    for one in registry.engines.values():
        if getattr(one, "mode", "") == "inprocess" or getattr(one, "adopted", False) or one.role == "renderer":
            continue
        if one.name in keep or not one.loaded() or getattr(one, "in_flight", 0) or getattr(one, "phase", "") == "loading":
            continue
        out.append(one)
    return sorted(out, key=lambda e: -e.held_mb())


class LiveGuard:
    """Every ``poll_s``: under the floor or over the pressure limit, the next idle model is released; with none left,
    ``stop_step`` ends the current step. ``busy()`` names the engines the step at work uses now (kept)."""

    def __init__(self, registry, reader, spec: dict | None = None, log_dir: Path | None = None, stop_step=None,
                 busy=None, psi=pressure) -> None:
        self.registry, self.reader, self.o = registry, reader, options(spec)
        self.stop_step, self.busy, self.psi = stop_step, busy or (lambda: set()), psi
        self.log_path = Path(log_dir or ".") / "memguard.jsonl"
        self.actions: list[dict] = []
        self.low_mb: int | None = None
        self.peak_psi = 0.0
        self.last_act = 0.0
        self.stopped_step = False
        self.thermal = thermal_options(spec)
        self.read_sensors = sensors
        self.hw_since: float | None = None
        self.thermal_stopped = False
        self.peak_gpu_c = 0.0
        self.hw_throttle_s = 0.0
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

    def log(self, what: str, **more) -> dict:
        entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "what": what, **more}
        self.actions = (self.actions + [entry])[-50:]
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
        return entry

    def start(self) -> None:
        self.thread = threading.Thread(target=self.loop, name="memguard", daemon=True)
        self.thread.start()
        self.log("started", floor_gb=self.o["floor_gb"], psi_limit=self.o["psi_limit"], poll_s=self.o["poll_s"])

    def stop(self) -> None:
        self._stop.set()

    def loop(self) -> None:
        while not self._stop.wait(float(self.o["poll_s"])):
            try:
                self.tick()
            except Exception as error:  # noqa: BLE001 - the guard never dies of one bad reading
                self.log("error", error=str(error)[:200])

    def thermal_tick(self, now: float) -> dict | None:
        """GPU at or over ``stop_gpu_c``, or the hardware's thermal slowdown for over ``stop_throttle_s``: the step at
        work is stopped (once a step; the render gate decides, cooled, whether it goes again)."""

        if not self.thermal["enabled"] or self.stop_step is None:
            return None
        s = self.read_sensors()
        gpu = s.get("gpu_c")
        if gpu is not None:
            self.peak_gpu_c = max(self.peak_gpu_c, gpu)
        hw = bool((s.get("reasons") or 0) & HW_THERMAL)
        if hw:
            if self.hw_since is None:
                self.hw_since = now
            self.hw_throttle_s += float(self.o["poll_s"])
        else:
            self.hw_since = None
        hot = gpu is not None and gpu >= self.thermal["stop_gpu_c"]
        held = self.hw_since is not None and now - self.hw_since > self.thermal["stop_throttle_s"]
        if not (hot or held):
            if not hw and gpu is not None and gpu < self.thermal["gpu_max_c"]:
                self.thermal_stopped = False  # cooled: a later step may be stopped again
            return None
        if self.thermal_stopped:
            return None
        self.thermal_stopped = True
        why = (f"GPU {gpu:.0f} C at or over {self.thermal['stop_gpu_c']:g}" if hot else
               f"hardware thermal slowdown for {now - self.hw_since:.0f} s")
        LAST_STOP.update(kind="thermal", at=now, why=why)
        said = self.stop_step(f"the thermal guard: {why}")
        return self.log("thermal stop: stopped the current step", why=why, said=str(said)[:200], sensors=s)

    def tick(self, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        hot = self.thermal_tick(now)
        if hot is not None:
            return hot
        free = self.reader.available_mb()
        psi = self.psi()
        if free >= 0:
            self.low_mb = free if self.low_mb is None else min(self.low_mb, free)
        if psi is not None:
            self.peak_psi = max(self.peak_psi, psi)
        low = 0 <= free < self.o["floor_gb"] * 1024
        # pressure counts only with memory getting short: a model loading from disk churns the page cache (pressure 49
        # on the DGX Spark with 43-49 GB available, 2026-11-08) and is no danger
        pressed = psi is not None and psi > self.o["psi_limit"] and 0 <= free < self.o["psi_mem_gb"] * 1024
        if not (low or pressed) or now - self.last_act < self.o["cooldown_s"]:
            return None
        self.last_act = now
        why = (f"MemAvailable {free / 1024:.1f} GB under the {self.o['floor_gb']:g} GB floor" if low else
               f"memory pressure {psi:.1f} over {self.o['psi_limit']:g}")
        for one in idle_engines(self.registry, set(self.busy())):
            if one.release(f"the memory guard: {why}", 0):
                one.unhold("released by the memory guard; it may load again when admitted")
                return self.log("released an idle model", engine=one.name, held_gb=round(one.held_mb() / 1024, 1), why=why,
                                available_gb=round(free / 1024, 1), psi=psi)
        users = biggest()
        killed = self.kill_foreign(users)
        if killed:
            return self.log("killed a foreign process (opted in)", killed=killed, why=why, users=named(users))
        if self.stop_step is not None and not self.stopped_step:
            self.stopped_step = True
            LAST_STOP.update(kind="memory", at=now, why=why)
            said = self.stop_step(f"the memory guard: {why}; the biggest users: {named(users)}")
            return self.log("stopped the current step", why=why, users=named(users), said=str(said)[:200])
        return self.log("nothing left to release", why=why, users=named(users))

    def kill_foreign(self, users: list[dict]) -> list[str]:
        out = []
        for u in users:
            if not u["ours"] and any(p and p in u["what"] for p in self.o.get("kill_foreign") or []):
                try:
                    os.kill(u["pid"], signal.SIGTERM)
                    out.append(f"{u['pid']} {u['what'][:60]}")
                except OSError:
                    pass
        return out

    def status(self) -> dict:
        return {"floor_gb": self.o["floor_gb"], "psi_limit": self.o["psi_limit"], "peak_gpu_c": self.peak_gpu_c,
                "hw_throttle_s": round(self.hw_throttle_s), "thermal_stopped": self.thermal_stopped, "low_gb": round(self.low_mb / 1024, 1) if self.low_mb else None,
                "peak_psi": self.peak_psi, "actions": self.actions[-10:]}
