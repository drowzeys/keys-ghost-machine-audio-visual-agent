"""Optional "lite voice": Kokoro-82M (hexgrad, Apache-2.0) behind the same HTTP API the agent's Voice class uses for
Drama 3 (POST /v1/audio/speech streamed as int16 PCM with an x-sample-rate header, GET /v1/voices), so a profile only
points voice.url at it. Kokoro is not autoregressive: a sentence's audio comes all at once, ~50-60 ms for a short first
chunk on a GPU, so streaming within a sentence is not needed, and it needs no keep-warm (profiles set
turn.keep_warm.voice false: an instant "Mm." makes the warm loop spin). No emotion tags, no cloning: 54 stock voices only.

    python -m voice_agent.kokoro_server --backend mlx --port 8197                  # Mac (mlx-audio, Metal)
    python -m voice_agent.kokoro_server --backend torch --device cuda --port 8197  # DGX Spark (kokoro, PyTorch CUDA)

Needs its own Python environment (requirements-kokoro-mac.txt or requirements-kokoro-spark.txt; install.sh --voice lite
makes it): mlx-audio (Mac) or kokoro + PyTorch (CUDA), misaki[en], the spaCy en_core_web_sm model, fastapi, uvicorn,
with preshed==3.0.10 cymem==2.0.11 murmurhash==1.0.13 pinned (newer preshed makes spaCy 3.8's tokenizer raise
OverflowError). English only here: American (a) and British (b) voices.
"""

from __future__ import annotations

import argparse
import asyncio
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

RATE = 24_000
VOICES = {  # the English voices; grades are hexgrad's own (VOICES.md), British first
    "bf_emma": "British female, warm and clear (B-)", "bf_isabella": "British female, brighter (C)",
    "bf_alice": "British female, crisp (D)", "bf_lily": "British female, soft (D)",
    "bm_george": "British male, deep (C)", "bm_fable": "British male, storyteller (C)",
    "bm_lewis": "British male (D+)", "bm_daniel": "British male (D)",
    "af_heart": "American female, Kokoro's best voice (A)", "af_bella": "American female, lively (A-)",
    "af_nicole": "American female, close and quiet (B-)", "af_sarah": "American female (C+)",
    "am_michael": "American male (C+)", "am_fenrir": "American male (C+)", "am_puck": "American male (C+)",
}


class Engine:
    def __init__(self, backend: str, device: str) -> None:
        self.lock = threading.Lock()  # one synthesis at a time on the GPU
        if backend == "mlx":
            from mlx_audio.tts.utils import load_model

            model = load_model("mlx-community/Kokoro-82M-bf16")

            def synth(text: str, voice: str) -> np.ndarray:
                parts = [np.asarray(r.audio, dtype=np.float32).reshape(-1)
                         for r in model.generate(text=text, voice=voice, speed=1.0, lang_code=voice[0])]
                return np.concatenate(parts) if parts else np.zeros(0, np.float32)
        else:
            import torch
            from kokoro import KModel, KPipeline

            km = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
            pipes = {lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=km) for lang in "ab"}

            def synth(text: str, voice: str) -> np.ndarray:
                with torch.inference_mode():
                    parts = [r.audio.float().cpu().numpy().reshape(-1) for r in pipes[voice[0]](text, voice=voice, speed=1.0)]
                return np.concatenate(parts) if parts else np.zeros(0, np.float32)
        self._synth = synth
        self(" Hello there.", "bf_emma")  # warm up: first call builds kernels and loads the G2P

    def __call__(self, text: str, voice: str) -> bytes:
        with self.lock:
            audio = self._synth(text, voice)
        # Kokoro pads each segment with ~250 ms of silence before and ~600 ms after: cut it to 20 ms before (that
        # silence is pure added latency) and 150 ms after (a natural gap between the agent's sentences)
        loud = np.nonzero(np.abs(audio) > 0.005)[0]
        if len(loud):
            audio = audio[max(0, loud[0] - RATE // 50): loud[-1] + RATE * 15 // 100]
        if not len(audio):  # nothing to say ("Mm." can come back empty): a little silence, never an empty stream
            audio = np.zeros(RATE // 20, np.float32)
        return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def build(engine: Engine, default_voice: str) -> FastAPI:
    app = FastAPI(title="kokoro (Drama-compatible API)")

    @app.get("/v1/voices")
    def voices():
        return {"data": [{"name": v} for v in VOICES], "details": [{"id": v, "description": d} for v, d in VOICES.items()]}

    @app.get("/health")
    def health():
        return {"ok": True, "model": "kokoro-82m"}

    @app.get("/v1/models")
    def models():  # what the installer's and Docker's readiness checks ask every engine
        return {"object": "list", "data": [{"id": "kokoro-82m", "object": "model", "owned_by": "hexgrad"}]}

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        body = await request.json()
        voice = body.get("voice") if body.get("voice") in VOICES else default_voice
        text = str(body.get("input", "")).strip()
        if not text:
            return JSONResponse({"error": "empty input"}, status_code=400)
        pcm = await asyncio.get_running_loop().run_in_executor(None, engine, text, voice)
        if body.get("response_format", "pcm") != "pcm":
            return JSONResponse({"error": "only pcm"}, status_code=400)

        async def chunks():
            for i in range(0, len(pcm), 9600):  # 200 ms pieces
                yield pcm[i: i + 9600]

        return StreamingResponse(chunks(), media_type="audio/pcm", headers={"x-sample-rate": str(RATE)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voice", default="bf_emma")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8197)
    args = parser.parse_args()
    uvicorn.run(build(Engine(args.backend, args.device), args.voice), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
