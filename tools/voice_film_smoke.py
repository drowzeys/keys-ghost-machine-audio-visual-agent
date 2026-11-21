"""A hands-free film from a SYNTHETIC spoken request, through the app's microphone path (its WebSocket, 16 kHz PCM in
real time), and the voice-to-voice latency while the film renders. Every utterance is a stock Kokoro-82M voice made
with tools/synth_turns.py: never a recording of a person.

    python tools/voice_film_smoke.py --ws ws://127.0.0.1:8196/ws --http http://127.0.0.1:8196 \
        --request turns/film00.wav --steer turns/film01_make_it_rain.wav --questions turns/stock*.wav --report smoke.json

It says the request, waits for the first keyframe in the panel, says the steer once while piece 1's DiT pass runs (it
applies from piece 2), and asks questions while every DiT pass runs (and a few while nothing renders, for a baseline),
until the film is done. The report: each turn's latency (the end of the person's speech to the first reply audio), where
it fell (DiT pass, other render stages, idle) and who answered, the spoken updates heard, and the run's timeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

FRAME = 512  # 32 ms at 16 kHz


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 3)


class Ear:
    """The app's side of the socket: every event with its time; replies' audio counted."""

    def __init__(self, ws) -> None:
        self.ws, self.events, self.audio_at = ws, [], []
        self.task = asyncio.create_task(self.read())

    async def read(self) -> None:
        async for message in self.ws:
            now = time.perf_counter()
            if isinstance(message, bytes):
                self.audio_at.append(now)
                continue
            event = json.loads(message)
            event["t"] = now
            self.events.append(event)

    def since(self, t: float, kind: str | None = None) -> list[dict]:
        return [e for e in self.events if e["t"] >= t and (kind is None or e.get("type") == kind)]


async def say(ws, ear: Ear, path: str, silence_s: float = 1.6, wait_s: float = 25.0) -> dict:
    audio, rate = sf.read(path, dtype="float32")
    assert rate == 16000, path
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    began = t = time.perf_counter()
    for i in range(0, len(pcm), FRAME):
        await ws.send(pcm[i:i + FRAME].tobytes())
        t += FRAME / 16000
        await asyncio.sleep(max(0, t - time.perf_counter()))
    end = time.perf_counter()
    quiet = np.zeros(FRAME, dtype=np.int16).tobytes()
    for _ in range(int(silence_s * 16000 / FRAME)):
        await ws.send(quiet)
        t += FRAME / 16000
        await asyncio.sleep(max(0, t - time.perf_counter()))
    deadline = end + wait_s
    first = None
    while time.perf_counter() < deadline:
        later = [a for a in ear.audio_at if a > end]
        if later:
            first = later[0]
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(1.5)
    heard = [e.get("text") for e in ear.since(began, "transcript")]
    replies = [e.get("text") for e in ear.since(began, "reply")]
    metrics = next((e for e in ear.since(began, "metrics")), None)
    return {"file": Path(path).name, "spoken_s": round(len(audio) / 16000, 2), "latency_s": round(first - end, 3) if first else None,
            "transcript": heard[-1] if heard else None, "reply": " ".join(r for r in replies if r)[:300],
            "server": {k: v for k, v in (metrics or {}).items() if k not in ("type", "t")}}


async def state(client: httpx.AsyncClient) -> dict:
    try:
        video = (await client.get("/api/video/status")).json()
    except (httpx.HTTPError, ValueError):
        video = {}
    try:
        gov = (await client.get("/api/governor")).json()
    except (httpx.HTTPError, ValueError):
        gov = {}
    ap = video.get("autopilot") or {}
    return {"state": video.get("state"), "stage": ap.get("stage"), "segment": ap.get("segment"), "done": ap.get("done"),
            "in_dit": bool(gov.get("in_dit")), "phase": gov.get("phase"), "schedule": gov.get("schedule"),
            "available_gb": round((gov.get("available_mb") or 0) / 1024, 1), "line": ap.get("line"), "run": video.get("run"),
            "preview": (video.get("current") or {}).get("preview")}


async def main(args) -> int:
    report = {"turns": [], "timeline": [], "updates": []}
    questions = [str(p) for p in args.questions]
    asked = 0
    async with websockets.connect(args.ws, max_size=2 ** 24) as ws, httpx.AsyncClient(base_url=args.http, timeout=10) as client:
        ear = Ear(ws)
        while not any(e.get("type") == "ready" for e in ear.events):
            await asyncio.sleep(0.05)
        began = time.perf_counter()

        def mark(what: str, **more) -> None:
            report["timeline"].append({"t": round(time.perf_counter() - began, 1), "what": what, **more})
            print(json.dumps(report["timeline"][-1]), flush=True)

        async def turn(path: str, where: str) -> None:
            before = await state(client)
            one = await say(ws, ear, path)
            one.update(where=where, dit=before["in_dit"], stage=before["stage"], segment=before["segment"], phase=before["phase"],
                       available_gb=before["available_gb"], at=round(time.perf_counter() - began, 1))
            report["turns"].append(one)
            print(json.dumps(one), flush=True)

        for path in questions[:args.idle]:  # a baseline: nothing renders yet (the first warms the brain)
            await turn(path, "idle")
            await asyncio.sleep(1.0)
        old_run = None if args.resume else (await state(client)).get("run")  # a run left from before is not this film
        if args.clip:  # a latency test only: one typed clip, questions during its DiT pass
            said = await client.post("/api/video/start", json={"prompt": args.clip, "mode": "single", "seconds": args.clip_seconds})
            mark("clip asked", status=said.status_code)
        else:
            await turn(str(args.request), "request")
            mark("request said", reply=report["turns"][-1]["reply"])
        seen_still = False
        steered = False
        last_stage, last_dit_turn = None, 0.0
        ends = time.perf_counter() + args.max_minutes * 60
        while time.perf_counter() < ends:
            now = await state(client)
            if now.get("run") in (None, old_run):
                await asyncio.sleep(1.0)
                continue
            if "run" not in report:
                report["run"] = now["run"]
                mark("film started", run=now["run"])
            if not seen_still and now.get("preview") in ("still", "engine"):
                seen_still = True
                mark("first picture in the panel", stage=now["stage"])
            key = (now["stage"], now["segment"], now["in_dit"])
            if key != last_stage:
                mark("stage", **{k: now[k] for k in ("state", "stage", "segment", "in_dit", "schedule", "available_gb", "line")})
                last_stage = key
            if now["state"] not in ("done", "failed", "stopped"):
                report["live_seen"] = True
            elif args.resume and not report.get("live_seen"):
                await asyncio.sleep(1.0)  # "keep going": the film that had stopped starts again
                continue
            if now["state"] in ("done", "failed", "stopped"):
                mark("film ended", state=now["state"])
                break
            if now["in_dit"] and time.perf_counter() - last_dit_turn > args.gap:
                if not steered and args.steer and (now["segment"] or 0) == 1:
                    await turn(str(args.steer), "steer during the DiT pass")
                    steered = True
                    mark("steer said", reply=report["turns"][-1]["reply"])
                elif questions:
                    await turn(questions[asked % len(questions)], "DiT pass")
                    asked += 1
                last_dit_turn = time.perf_counter()
            elif not now["in_dit"] and now["state"] in ("rendering", "keyframe", "checking", "planning") and args.other_every \
                    and time.perf_counter() - last_dit_turn > args.other_every and questions:
                await turn(questions[asked % len(questions)], f"film at work ({now['stage']})")
                asked += 1
                last_dit_turn = time.perf_counter()
            await asyncio.sleep(1.0)
        report["updates"] = [{"t": round(e["t"] - began, 1), "text": e.get("text"), "source": e.get("source")}
                             for e in ear.events if e.get("type") == "reply" and e.get("source")]
        ear.task.cancel()
    for where in ("idle", "DiT pass", "steer during the DiT pass"):
        values = [t["latency_s"] for t in report["turns"] if t["where"] == where and t["latency_s"] is not None]
        report.setdefault("latency", {})[where] = {"n": len(values), "p50": pct(values, 0.5), "p95": pct(values, 0.95),
                                                     "mean": round(statistics.mean(values), 3) if values else None}
    during = [t["latency_s"] for t in report["turns"] if t["dit"] and t["latency_s"] is not None]
    report["latency"]["all turns during a DiT pass"] = {"n": len(during), "p50": pct(during, 0.5), "p95": pct(during, 0.95)}
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["latency"], indent=1))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ws", default="ws://127.0.0.1:8196/ws")
    parser.add_argument("--http", default="http://127.0.0.1:8196")
    parser.add_argument("--request", help="the spoken request (a synthetic WAV, 16 kHz mono)")
    parser.add_argument("--clip", help="instead: one typed clip of this idea (a latency test during its DiT pass)")
    parser.add_argument("--clip-seconds", type=float, default=10)
    parser.add_argument("--steer", help="one steer, said during piece 1's DiT pass")
    parser.add_argument("--questions", nargs="*", default=[], help="questions asked while the film renders")
    parser.add_argument("--idle", type=int, default=2, help="questions asked before the request (a baseline)")
    parser.add_argument("--gap", type=float, default=6.0, help="seconds between turns during a DiT pass")
    parser.add_argument("--other-every", type=float, default=0.0, help="also ask during other stages, this often (0: never)")
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--resume", action="store_true", help="the request carries on the panel's film (\"keep going\")")
    parser.add_argument("--report", required=True)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
