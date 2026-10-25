"""Talks to the agent like a person: streams WAV files (16 kHz mono) in real time, then silence, and measures the
voice-to-voice latency (the end of the person's speech to the first reply audio received). With --barge, it speaks the
next utterance one second into each reply and measures how fast the agent stops.

    python -m voice_agent.bench ws://127.0.0.1:8095/ws asrset/u00.wav asrset/u01.wav [--barge] [--agent NAME]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import numpy as np
import soundfile as sf
import websockets

FRAME = 512  # 32 ms


async def stream(ws, audio: np.ndarray, silence_s: float) -> float:
    """Sends audio then silence in real time; returns when the last speech sample left (perf_counter)."""

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    t = time.perf_counter()
    for i in range(0, len(pcm), FRAME):
        await ws.send(pcm[i : i + FRAME].tobytes())
        t += FRAME / 16000
        await asyncio.sleep(max(0, t - time.perf_counter()))
    end = time.perf_counter()
    quiet = np.zeros(FRAME, dtype=np.int16).tobytes()
    for _ in range(int(silence_s * 16000 / FRAME)):
        await ws.send(quiet)
        t += FRAME / 16000
        await asyncio.sleep(max(0, t - time.perf_counter()))
    return end


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("wavs", nargs="+")
    parser.add_argument("--barge", action="store_true")
    parser.add_argument("--agent", help="hand deep requests to this agent (e.g. a tool-free direct one) instead of the profile's")
    args = parser.parse_args()
    results = []
    async with websockets.connect(args.url, max_size=2**24) as ws:
        assert json.loads(await ws.recv())["type"] == "ready"
        if args.agent:
            await ws.send(json.dumps({"type": "agent", "name": args.agent}))
        for n, path in enumerate(args.wavs):
            audio, rate = sf.read(path, dtype="float32")
            assert rate == 16000, path
            got = {"audio_bytes": 0, "rate": 44100}
            first_audio = asyncio.get_running_loop().create_future()
            done = asyncio.get_running_loop().create_future()
            interrupted = asyncio.get_running_loop().create_future()

            async def reader():
                while True:
                    m = await ws.recv()
                    if isinstance(m, bytes):
                        got["audio_bytes"] += len(m)
                        if not first_audio.done():
                            first_audio.set_result(time.perf_counter())
                        continue
                    e = json.loads(m)
                    if e["type"] == "audio_start":
                        got["rate"] = e.get("sample_rate", 44100)
                    if e["type"] in ("transcript", "reply"):
                        got.setdefault(e["type"], []).append(e["text"])
                    if e["type"] == "speech_end":
                        got["ends"] = got.get("ends", 0) + 1
                    if e["type"] == "interrupt" and not interrupted.done():
                        interrupted.set_result(time.perf_counter())
                    if e["type"] == "metrics":
                        got["server"] = {k: v for k, v in e.items() if k != "type"}
                        if not done.done():
                            done.set_result(True)

            task = asyncio.create_task(reader())
            sender = asyncio.create_task(stream(ws, audio, 1.0))
            end = await sender
            try:
                t_audio = await asyncio.wait_for(first_audio, 30)
            except asyncio.TimeoutError:
                t_audio = None
            row = {"wav": path.split("/")[-1], "voice_to_voice_ms": round((t_audio - end) * 1000) if t_audio else None}
            if args.barge and n + 1 < len(args.wavs) and t_audio:
                await asyncio.sleep(1.0)
                nxt, _ = sf.read(args.wavs[n + 1], dtype="float32")
                t_barge = time.perf_counter()
                barge = asyncio.create_task(stream(ws, nxt[: 16000], 0.0))  # the first second of the next utterance
                try:
                    t_int = await asyncio.wait_for(interrupted, 5)
                    row["barge_in_stop_ms"] = round((t_int - t_barge) * 1000)
                except asyncio.TimeoutError:
                    row["barge_in_stop_ms"] = None
                await barge
                await ws.send(np.zeros(16000, dtype=np.int16).tobytes())
                await asyncio.sleep(2.5)
                await ws.send(json.dumps({"type": "stop"}))
                await asyncio.sleep(0.3)
            else:
                try:
                    await asyncio.wait_for(done, 60)
                except asyncio.TimeoutError:
                    pass
            task.cancel()
            row.update({"ends": got.get("ends", 0), "heard": " ".join(got.get("transcript", [])), "said": " ".join(got.get("reply", [])),
                        "reply_audio_s": round(got["audio_bytes"] / 2 / got["rate"], 2), **{"srv_" + k: v for k, v in got.get("server", {}).items()}})
            results.append(row)
            print(json.dumps(row), flush=True)
            await asyncio.sleep(0.5)
    v2v = [r["voice_to_voice_ms"] for r in results if r["voice_to_voice_ms"]]
    if v2v:
        print(json.dumps({"turns": len(results), "voice_to_voice_ms_median": statistics.median(v2v), "min": min(v2v), "max": max(v2v)}))


if __name__ == "__main__":
    asyncio.run(main())
