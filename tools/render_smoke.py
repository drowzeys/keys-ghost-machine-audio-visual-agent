"""One clip from a typed request through the voice's video path, without the voice: the supervisor (engine registry,
memory governor, the headless Studio's renderer) and the video panel's service and agent, exactly as the app wires them, with the
typed words handed to the video agent as a heard turn. For a smoke test on a machine without the ears and the brain
running (the planner then takes the idea as the prompt).

    python tools/render_smoke.py --profile profiles/spark.json --text "make a video of a red kite over green hills" \
        --report smoke.json

It prints each change of the run (state, stage, step, preview count) and writes a report: the timings the renderer
measured (keyframe, load, render, decode), the lowest available memory, the previews that arrived while the clip was
denoised, and the clip's path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gma.supervisor import Supervisor  # noqa: E402
from voice_agent import video  # noqa: E402


async def main(args) -> int:
    profile = json.loads(Path(args.profile).expanduser().read_text())
    sup = Supervisor(profile)
    adopted = await asyncio.to_thread(sup.start)
    print("adopted:", {k: v for k, v in adopted.items() if v}, flush=True)
    svc = video.VideoService(profile["video"], None, renderer=sup.renderer)
    agent = video.VideoAgent("video", profile["video"], svc)
    seen: list[dict] = []
    began = time.time()
    last = [None]

    def changed():
        snap = svc.snapshot()
        cur = snap.get("current") or {}
        line = (snap["state"], cur.get("stage"), cur.get("at"), cur.get("of"), cur.get("preview"), cur.get("seq"))
        if line != last[0]:
            last[0] = line
            seen.append({"t": round(time.time() - began, 1), "state": snap["state"], "stage": cur.get("stage"),
                         "at": cur.get("at"), "of": cur.get("of"), "preview": cur.get("preview"), "seq": cur.get("seq")})
            print(json.dumps(seen[-1]), flush=True)

    original = svc.changed

    def spy():
        changed()
        original()

    svc.changed = spy
    said, detail = await agent.handle(args.text, lambda *a: None)
    print("said:", said, "|", detail, flush=True)
    if not svc.task:
        sup.shutdown()
        return 1
    try:
        await svc.task
        jobs = getattr(sup.renderer, "jobs", {})
        await asyncio.gather(*(j["task"] for j in jobs.values() if j.get("task")), return_exceptions=True)
    finally:
        await asyncio.to_thread(sup.shutdown)
    run = svc.run or {}
    clips = run.get("clips") or []
    report = {"text": args.text, "said": said, "state": run.get("state"), "message": run.get("message"),
              "seconds_total": round(time.time() - began, 1), "clip": str(svc.clip_file(run["id"], 0, "mp4")) if clips else None,
              "keyframe": clips[0].get("image") if clips else None, "prompt": clips[0].get("prompt") if clips else None,
              "keyframe_timings": clips[0].get("keyframe_timings") if clips else None,
              "timings": clips[0].get("timings") if clips else None, "governor": sup.governor.status(),
              "previews_while_denoising": sum(1 for s in seen if s["preview"] == "engine" and s["stage"] == "denoising"),
              "changes": seen}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("changes", "governor")}, indent=1, default=str), flush=True)
    return 0 if run.get("state") == "done" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--text", required=True, help="the typed request, as a person would say it")
    parser.add_argument("--report", default="")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
