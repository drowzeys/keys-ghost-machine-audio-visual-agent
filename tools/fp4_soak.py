"""The FP4 clip worker alone under sustained load (2026-11-08: two big-picture drafts froze the DGX Spark mid-denoise on their third
back-to-back clip, with MemAvailable steady at ~42 GB, pressure 0 and no NVRM/Xid line: not memory).

It starts gma/fp4_worker.py exactly as the Spark profile does, renders the given clips back to back (or with
``--gap`` seconds between them), and samples every second into one small file a minute (each line flushed and fsynced,
so the record survives a hang): MemAvailable, MemFree, memory pressure, the GPU's temperature, power, SM clock and
clock-event (throttle) reasons, the hottest ACPI zone, and the worker's RSS. It aborts (the worker killed at once) on a
GPU temperature at or over ``--max-temp`` or on any Xid / NVRM line in the kernel log.

    python tools/fp4_soak.py --jobs a.json b.json c.json --out ~/ghost-studio/out/soak1 [--gap 60]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

G = Path.home() / "ghost-studio"
ARGV = [str(G / "fv-venv/bin/python"), "-I", str(G / "repo/gma/fp4_worker.py"), "--model", str(G / "models/fastvideo/FastH3-V2-Consumer"),
        "--port", "8420", "--cache", str(G / "cache")]
ENV = {"HF_HUB_OFFLINE": "1", "FASTVIDEO_H3_ADALN_TABLE": str(G / "models/fastvideo/adaln_tables_t2v_fl2va.pt")}
URL = "http://127.0.0.1:8420"
SMI = ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.active", "--format=csv,noheader,nounits"]


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "MemFree"):
            out[k] = int(v.split()[0]) // 1024
    return out


def psi() -> float | None:
    try:
        return float(Path("/proc/pressure/memory").read_text().split()[1].split("=")[1])
    except (OSError, IndexError, ValueError):
        return None


def zone_max() -> float | None:
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(z.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def rss_mb(pid: int | None) -> int | None:
    if not pid:
        return None
    total = 0
    for p in [pid] + children(pid):
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) // 1024
        except OSError:
            pass
    return total


def children(pid: int) -> list[int]:
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if int((d / "stat").read_text().rsplit(")", 1)[1].split()[1]) == pid:
                    out.append(int(d.name))
            except (OSError, ValueError, IndexError):
                pass
    return out


class Sampler(threading.Thread):
    def __init__(self, out: Path, state: dict) -> None:
        super().__init__(daemon=True)
        self.out, self.state, self.stop = out, state, threading.Event()
        self.max = {"gpu_c": 0.0, "power_w": 0.0, "zone_c": 0.0}
        self.reasons: set[str] = set()

    def run(self) -> None:
        while not self.stop.wait(1.0):
            row = {"t": round(time.time(), 1), "clip": self.state.get("clip"), "stage": self.state.get("stage"), **meminfo(), "psi": psi(),
                   "zone_c": zone_max(), "worker_rss_mb": rss_mb(self.state.get("pid"))}
            try:
                smi = subprocess.run(SMI, capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                row.update(gpu_c=float(smi[0]), power_w=float(smi[1]), sm_mhz=float(smi[2]), reasons=smi[3].strip())
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            for k in self.max:
                if isinstance(row.get(k), (int, float)):
                    self.max[k] = max(self.max[k], row[k])
            if row.get("reasons") and row["reasons"] not in ("0x0000000000000000",):
                self.reasons.add(row["reasons"])
            self.state["last"] = row
            path = self.out / f"samples-{time.strftime('%H%M')}.jsonl"
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())


class KernelWatch(threading.Thread):
    """Any Xid or NVRM line in the kernel log since the start: the abort rule."""

    def __init__(self, state: dict) -> None:
        super().__init__(daemon=True)
        self.state = state

    def run(self) -> None:
        proc = subprocess.Popen(["journalctl", "-k", "-f", "-n", "0", "--no-pager"], stdout=subprocess.PIPE, text=True)
        for line in proc.stdout:
            if "Xid" in line or "NVRM" in line:
                self.state.setdefault("kernel", []).append(line.strip()[:300])


def get(path: str) -> dict:
    with urllib.request.urlopen(URL + path, timeout=10) as r:
        return json.loads(r.read())


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


WORKER: list[int] = []


def main() -> int:
    try:
        return run()
    finally:
        for pid in WORKER:  # never left running, whatever ended the test
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass


def run() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--jobs", nargs="+", required=True, help="job.json files of the clips (their clip_request)")
    a.add_argument("--out", required=True)
    a.add_argument("--gap", type=float, default=0.0, help="seconds between clips")
    a.add_argument("--max-temp", type=float, default=90.0)
    args = a.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    state: dict = {"stage": "loading"}
    sampler = Sampler(out, state)
    sampler.start()
    KernelWatch(state).start()
    log = (out / "worker.log").open("a")
    worker = subprocess.Popen(ARGV, env={**os.environ, **ENV}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["pid"] = worker.pid
    WORKER.append(worker.pid)
    report = {"gap_s": args.gap, "clips": [], "started": time.strftime("%H:%M:%S")}

    def abort(why: str) -> int:
        os.killpg(worker.pid, signal.SIGKILL)
        report.update(aborted=why, kernel=state.get("kernel"), max=sampler.max, reasons=sorted(sampler.reasons))
        (out / "report.json").write_text(json.dumps(report, indent=1))
        print("ABORT", why, flush=True)
        return 2

    began = time.time()
    while True:
        try:
            if get("/health").get("loaded"):
                break
        except OSError:
            pass
        if worker.poll() is not None:
            return abort("the worker exited while loading")
        time.sleep(2)
    report["load_s"] = round(time.time() - began, 1)
    print(f"worker loaded in {report['load_s']} s", flush=True)
    for i, job_path in enumerate(args.jobs, 1):
        if i > 1 and args.gap:
            state.update(clip=None, stage="gap")
            time.sleep(args.gap)
        req = dict(json.loads(Path(job_path).read_text())["clip_request"])
        req["output"] = str(out / f"clip{i}.mp4")
        state.update(clip=i, stage="submit")
        t0 = time.time()
        handle = None
        for _ in range(60):  # 409 while the worker clears the last clip (its "done" comes before it is free)
            try:
                handle = post("/v1/clips", req)["id"]
                break
            except urllib.error.HTTPError as error:
                if error.code != 409:
                    raise
                time.sleep(1)
        if handle is None:
            return abort("the worker stayed busy after the last clip")
        while True:
            time.sleep(1)
            if state.get("kernel"):
                return abort("a Xid / NVRM line in the kernel log: " + state["kernel"][-1])
            last = state.get("last") or {}
            if (last.get("gpu_c") or 0) >= args.max_temp:
                return abort(f"GPU {last['gpu_c']} C >= {args.max_temp}")
            view = get(f"/v1/clips/{handle}")
            state["stage"] = view.get("stage")
            if view["status"] in ("done", "failed"):
                break
        row = {"clip": i, "status": view["status"], "wall_s": round(time.time() - t0, 1), "timings": view.get("timings"),
               "error": view.get("error"), "max_so_far": dict(sampler.max)}
        report["clips"].append(row)
        print(json.dumps(row), flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1))
    state.update(clip=None, stage="done")
    time.sleep(3)
    os.killpg(worker.pid, signal.SIGTERM)
    report.update(max=sampler.max, reasons=sorted(sampler.reasons), kernel=state.get("kernel"), ended=time.strftime("%H:%M:%S"))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("done", json.dumps({"max": sampler.max, "reasons": sorted(sampler.reasons)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
