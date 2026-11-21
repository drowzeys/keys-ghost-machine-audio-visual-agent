"""An AutoPilot film from a typed brief, as the app wires it, without the ears and the voice: the supervisor (engine
registry, memory governor, the headless Studio's renderer), the brain on its Director lane, AutoPilot's eyes and ears, and the video
panel's service. For a smoke test of phase 3 on one machine.

    python tools/autopilot_smoke.py --profile profiles/spark.json --brief "a lighthouse keeper's evening" \
        --pieces 3 --seconds 10 --report smoke.json

    python tools/autopilot_smoke.py --profile profiles/spark.json --resume --report smoke2.json

``--resume`` starts the app again on the same folders and lets the run that was at work carry on from its checkpoint
(what a restart of the app does). It prints each change of the run, samples this machine's available memory every half
second (MemAvailable, vm_stat on a Mac) and writes a report: per piece the Director's, the keyframes', the renders' and
the checks' times and the retakes, the lowest available memory, the eyes' loads, and the film's path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gma.ap import autopilot as ap  # noqa: E402
from gma.ap import ears as ap_ears  # noqa: E402
from gma.ap import eyes as ap_eyes  # noqa: E402
from gma.ap import judge as ap_judge  # noqa: E402
from gma.ap.service import AutoPilotService  # noqa: E402
from gma.brain import make_brains  # noqa: E402
from gma.supervisor import Supervisor  # noqa: E402
from voice_agent import video  # noqa: E402


def pieces_report(run: dict, timings: dict) -> list[dict]:
    """Per piece: the Director (from the log's stage marks), keyframes, renders, checks, retakes."""

    out = []
    log = run.get("log") or []
    for seg in run.get("segments") or []:
        n = seg["index"]
        marks = [(line["at"], line["text"]) for line in log if f"Segment {n}" in line["text"] or f"Segment {n}:" in line["text"]]
        stats = seg.get("check_stats") or {}
        out.append({"index": n, "beat": seg.get("beat"), "status": seg.get("status"), "takes": len(seg.get("takes") or []),
                    "retakes": max(0, len(seg.get("takes") or []) - 1),
                    "keyframes_drawn": len(seg.get("keyframes") or []), "passed": seg.get("passed"),
                    "reasons": seg.get("reasons"), "director_rejected": len(seg.get("rejected") or []),
                    "check_requests": stats.get("requests"), "check_questions": stats.get("questions"),
                    "check_failed_answers": stats.get("failed_answers"), "check_votes": stats.get("votes"),
                    "check_seconds": stats.get("seconds"),
                    "judge": [{"take": t["take"], "retake": (t.get("judge") or {}).get("retake"), "ms": (t.get("judge") or {}).get("ms"),
                               "p": {k: q["p"] for k, q in ((t.get("judge") or {}).get("questions") or {}).items()},
                               "faults": (t.get("judge") or {}).get("faults"), "doubts": (t.get("judge") or {}).get("doubts")}
                              for t in seg.get("takes") or [] if t.get("judge")],
                    "renderer": timings.get(str(n)), "log": marks})
    return out


async def main(args) -> int:
    profile = json.loads(Path(args.profile).expanduser().read_text())
    if args.eyes:
        profile.setdefault("autopilot", {})["eyes"] = args.eyes
    if args.director:
        profile.setdefault("autopilot", {})["director"] = args.director
    if args.judge:
        profile.setdefault("autopilot", {})["judge"] = args.judge
    sup = Supervisor(profile)
    adopted = await asyncio.to_thread(sup.start)
    print("adopted:", {k: v for k, v in adopted.items() if v}, flush=True)
    brain = make_brains(profile, sup.registry)
    brain = brain.director or brain.main
    svc = video.VideoService(profile["video"], None, renderer=sup.renderer, lanes=sup.lanes)
    eyes = ap_eyes.make(profile, sup.registry)
    judge = ap_judge.make(profile, sup.registry)  # d1-2B unless --judge eyes
    ears = ap_ears.Ears((profile.get("autopilot") or {}).get("ears"), asr=profile.get("asr"))
    svc.autopilot = AutoPilotService(svc, profile.get("autopilot"), brain=brain, lanes=sup.lanes, eyes=eyes, ears=ears,
                                     loop=asyncio.get_running_loop(), judge=judge)
    print("judge:", judge.describe() if judge else "the eyes", "| eyes:", eyes.describe() if eyes and not judge else None, "| ears:", ears.ready(), flush=True)
    began = time.time()
    memory, low = [], [None]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.5):
            free = sup.reader.available_mb()
            if free >= 0:
                memory.append((round(time.time() - began, 1), free))
                low[0] = free if low[0] is None else min(low[0], free)

    threading.Thread(target=sample, daemon=True).start()
    last = [None]

    def changed():
        snap = svc.snapshot()
        a = snap.get("autopilot") or {}
        cur = snap.get("current") or {}
        line = (snap["state"], a.get("stage"), a.get("segment"), cur.get("at"), a.get("line"))
        if line != last[0]:
            last[0] = line
            print(json.dumps({"t": round(time.time() - began, 1), "state": snap["state"], "stage": a.get("stage"),
                              "piece": a.get("segment"), "step": cur.get("at"), "of": cur.get("of"), "line": (a.get("line") or "")[:160],
                              "avail_gb": round(sup.reader.available_mb() / 1024, 1)}), flush=True)

    original = svc.changed

    def spy():
        changed()
        original()

    svc.changed = spy
    if args.resume:
        await svc.resume()
        run_id = (svc.run or {}).get("id")
    else:
        snap = await svc.start(args.brief, "autopilot", args.pieces, args.seconds)
        run_id = snap["run"]
    print("run:", run_id, flush=True)
    try:
        while run_id and ap.alive(run_id):
            await asyncio.sleep(1.0)
            if time.time() - began > args.limit_s:
                print("time limit: stopping the run", flush=True)
                await svc.stop()
                break
    finally:
        stop.set()
        await asyncio.to_thread(sup.shutdown)
    run = ap.load(run_id)
    trace = memory[:: max(1, len(memory) // 600)]
    report = {"brief": args.brief, "run": run_id, "state": run.get("state"), "error": run.get("error"),
              "seconds_total": round(time.time() - began, 1), "film": run.get("film"),
              "film_path": str(svc.root / run["film"]["path"]) if run.get("film") else None,
              "folder": str(ap.run_dir(run_id)), "check_stats": run.get("check_stats"),
              "pieces": pieces_report(run, svc.autopilot.timings), "eyes": eyes.describe() if eyes and not judge else None, "judge": judge.describe() if judge else None,
              "ears": ears.ready(), "governor": sup.governor.status(),
              "engines": sup.registry.status(), "min_available_gb": round(low[0] / 1024, 1) if low[0] is not None else None,
              "memory_trace_gb": [(t, round(mb / 1024, 1)) for t, mb in trace]}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("run", "state", "error", "seconds_total", "film", "min_available_gb")}, default=str), flush=True)
    return 0 if run.get("state") == "done" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--brief", default="")
    parser.add_argument("--pieces", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--judge", default="", help="d1 (the default) or eyes (the profile's autopilot.judge_choices)")
    parser.add_argument("--eyes", default="", help="gemma or cosmos (the profile's autopilot.eyes_choices)")
    parser.add_argument("--director", default="", help="qwen, bolt or brain (the profile's autopilot.director_choices)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-s", type=float, default=4 * 3600)
    parser.add_argument("--report", default="")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
