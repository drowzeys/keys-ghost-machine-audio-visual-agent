"""One speech recognizer shared by the voice agents on a machine (optional).

Each agent loads its own Parakeet by default: about 1.2 GB of weights plus MLX's buffer cache on a Mac, 5.4 GB of GPU
memory with NeMo on a DGX Spark. With this server running, the agents' profiles name it (``"asr": {"kind": ...,
"model": ..., "url": "http://127.0.0.1:8192"}``) and send it their utterances instead; an agent whose server does not
answer recognizes in process (engines.Recognizer). One model, one thread: two agents that finish a turn at the same
moment are recognized one after the other (Parakeet takes 70-100 ms an utterance).

    python -m voice_agent.asr_server --kind parakeet-mlx --model mlx-community/parakeet-tdt-0.6b-v3 --port 8192

POST /v1/transcribe: the utterance as float32 little-endian mono samples at 16 kHz (application/octet-stream),
optional ?language=; answers {"text": ..., "ms": ...}. GET /health: {"ok": true, "kind": ..., "model": ...}.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from .engines import SAMPLE_RATE, Recognizer

MAX_SECONDS = 120  # the longest utterance taken (a turn is cut long before this)


def app_for(recognizer: Recognizer) -> FastAPI:
    app = FastAPI()
    pool = ThreadPoolExecutor(1, thread_name_prefix="asr")  # one recognizer, one thread (MLX, CUDA)
    served = {"utterances": 0, "seconds": 0.0}

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "kind": recognizer.kind, "model": recognizer.model_id, **served}

    @app.post("/v1/transcribe")
    async def transcribe(request: Request) -> dict:
        body = await request.body()
        if len(body) % 4 or len(body) > MAX_SECONDS * SAMPLE_RATE * 4:
            raise HTTPException(400, f"send float32 samples at {SAMPLE_RATE} Hz, at most {MAX_SECONDS} s")
        audio = np.frombuffer(body, dtype="<f4").astype(np.float32)
        language = request.query_params.get("language")
        started = time.perf_counter()

        def run() -> str:
            before = recognizer.language
            recognizer.language = language or before
            try:
                return recognizer(audio)
            finally:
                recognizer.language = before

        text = await asyncio.get_running_loop().run_in_executor(pool, run)
        served["utterances"] += 1
        served["seconds"] = round(served["seconds"] + len(audio) / SAMPLE_RATE, 1)
        return {"text": text, "ms": round(1000 * (time.perf_counter() - started), 1)}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="One speech recognizer shared by the voice agents on this machine.")
    parser.add_argument("--kind", default="parakeet-mlx", choices=["parakeet-mlx", "mlx-audio", "nemo"])
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v3")
    parser.add_argument("--language", default=None)
    parser.add_argument("--mlx-cache-mb", type=float, default=256, help="MLX buffer cache cap on a Mac; -1 leaves it alone")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8192)
    args = parser.parse_args()
    recognizer = Recognizer({"kind": args.kind, "model": args.model, "language": args.language, "mlx_cache_mb": args.mlx_cache_mb})
    uvicorn.run(app_for(recognizer), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
