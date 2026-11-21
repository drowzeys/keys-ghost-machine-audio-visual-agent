"""Synthetic spoken turns for latency tests: short questions spoken by a stock Kokoro-82M voice (never a person's
recording), resampled to 16 kHz mono WAV for voice_agent.bench.

    python tools/synth_turns.py http://127.0.0.1:8197/v1 out_dir [--voice am_michael]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

LINES = ("Hey, how are you doing today?", "What's a good name for a cat?", "Tell me something nice about the moon.",
         "Can you suggest a quick breakfast?", "What colour is the sky at sunset?", "Say hello to my friend.",
         "What should I call my new plant?", "Give me a fun fact about owls.")


def to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return pcm
    n = int(round(len(pcm) * 16000 / rate))
    return np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("--voice", default="am_michael")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(LINES):
        reply = httpx.post(args.url.rstrip("/") + "/audio/speech", timeout=60,
                           json={"model": "kokoro-82m", "input": line, "voice": args.voice, "response_format": "pcm"})
        reply.raise_for_status()
        rate = int(reply.headers.get("x-sample-rate", 24000))
        pcm = np.frombuffer(reply.content, dtype=np.int16).astype(np.float32) / 32768
        sf.write(out / f"t{i:02d}.wav", to_16k(pcm, rate), 16000)
        print(out / f"t{i:02d}.wav", f"{len(pcm) / rate:.2f}s", line)


if __name__ == "__main__":
    main()
