"""Ghost Studio's clip worker: FastH3 V2 on FastVideo's FP4 sparse-attention recipe, one resident process.

Every clip of Ghost Studio is rendered here, and only here: MiniMax H3 image-to-video with its stereo audio from the
piece's keyframe (FL2VA, the first frame), or from text alone, with the eight-forward FastH3 V2 transformer in NVFP4,
the NVFP4 Qwen3-VL text encoder, the light H3 video VAE and the audio VAE all resident, and MiniMax H3 video sparse
attention (VSA, sparsity 0.8, 64-token tiles) with its FP4 attention kernel. The recipe is FastVideo's
``examples/inference/basic/basic_fasth3_spark_v2_nvfp4.yaml`` and docs/getting_started/installation/spark_performance.md
("FastH3 V2 NVFP4 on one Spark"), FastVideo (hao-ai-lab/FastVideo, Apache-2.0) at commit a01c50a8; the weights are
FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer. Nothing here falls back to another renderer: a clip that cannot
be made fails with its reason.

It runs in its own minimal virtual environment (torch cu130, FlashInfer, fastvideo-kernel built for sm_121a,
FastVideo's package), never in the app's: the app starts it as a managed engine (gma/registry.py), loads it for a
clip and releases it when a keyframe or the film's sound needs the room. This file uses only the standard library and
FastVideo, so it runs in that environment as ``python gma/fp4_worker.py``.

    GET  /health                 200 {"loaded": true, ...} once the model is resident (503 while it loads)
    POST /v1/clips               {"prompt", "image"?, "width", "height", "frames", "seed", "output"} -> 202 {"id"}
    GET  /v1/clips/{id}          {"status": queued|running|done|failed, "stage", "at", "of", "timings", "error", "output"}

One clip at a time (a second is refused with 409 while one runs). ``stage`` is "conditioning", "denoising" (the DiT
pass: ``at`` of ``of`` forwards), "decoding the video", "decoding the audio", "saving"; the app releases the brain for
the DiT pass on that word.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# the recipe's environment (spark_performance.md), set before FastVideo is imported
RECIPE_ENV = {
    "FASTVIDEO_H3_VSA_FP4": "1",
    "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
    "FASTVIDEO_NVFP4_MM_BACKEND": "cudnn",
    "FASTVIDEO_NVFP4_MM_ROW_SPLIT": "table",
    "FASTVIDEO_H3_VAE_TILE_BATCH": "12",
    "FASTVIDEO_H3_VAE_INT8_OVERLAY": "0",  # the dense VAE decoder; the int8 overlay file is not downloaded
    "FASTVIDEO_VSA_TRITON": "1",
    "FASTVIDEO_VSA_PTX_BLACKWELL": "0",
    "FASTVIDEO_FA4": "0",
    "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
    "FASTVIDEO_STAGE_LOGGING": "1",
    # a first-use JIT build never runs unlimited nvcc beside the resident model (on the DGX Spark that took MemAvailable to 0)
    "MAX_JOBS": "4",
    "FLASHINFER_JIT_PREBUILD_MAX_JOBS": "4",
    "CMAKE_BUILD_PARALLEL_LEVEL": "4",
    "TORCHINDUCTOR_COMPILE_THREADS": "4",
}

STEPS = 9  # nine sigma points, eight DiT forwards (the trained ladder in fastvideo_inference.json)
FORWARDS = STEPS - 1


def generator_config(model: str, compile_vae: bool = True) -> dict:
    """basic_fasth3_spark_v2_nvfp4.yaml's generator, image-to-video: everything resident, eager DiT, NVFP4 DiT with
    the h3_dit_vsa profile, MiniMax H3 VSA with FP4 attention at sparsity 0.8, the light H3 VAE decode."""

    return {
        "model_path": model,
        "engine": {
            "num_gpus": 1, "use_fsdp_inference": False,
            "quantization": {"transformer_quant": "NVFP4", "layer_profile": "h3_dit_vsa"},
            "parallelism": {"tp_size": 1, "sp_size": 1},
            "offload": {"dit": False, "dit_layerwise": False, "text_encoder": False, "image_encoder": False, "vae": False,
                        "pin_cpu_memory": False, "lazy_module_load": False},
            "compile": {"enabled": False, "vae_enabled": bool(compile_vae)},
        },
        "pipeline": {
            "workload_type": "i2v",
            "vae_tiling": True,
            "experimental": {
                "attention_backend": "VIDEO_SPARSE_ATTN_H3", "h3_vsa_fp4": {"enabled": True},
                "VSA_sparsity": 0.8, "VSA_tile_size": 64, "h3_sequential_load": False,
                "inference_torch_compile": False, "vae_parallel_decode": False, "video_decode_backend": "h3-vae",
            },
        },
    }


def materialize_safetensors() -> None:
    """safetensors' CPU tensors copied out of their memory map as they are read. On GB10 a tensor that still points
    into the file's mmap moves to the GPU ~300x slower than a plain one (measured on the DGX Spark: 4.8 GB in 28 s against 1 s
    to copy and 0.1 s to move), and FastVideo's loader moves the whole DiT that way: 70 of a 93 s load. Called before
    FastVideo is imported, so every ``from safetensors import safe_open`` sees this one."""

    import safetensors

    if getattr(safetensors.safe_open, "_ghost", False):
        return
    plain = safetensors.safe_open

    class Materialized:
        _ghost = True

        def __init__(self, *args, **kwargs):
            device = kwargs.get("device", args[2] if len(args) > 2 else "cpu")
            self._file, self._cpu = plain(*args, **kwargs), str(device) == "cpu"

        def __enter__(self):
            self._file.__enter__()
            return self

        def __exit__(self, *exc):
            return self._file.__exit__(*exc)

        def get_tensor(self, name):
            tensor = self._file.get_tensor(name)
            return tensor.clone() if self._cpu else tensor

        def __getattr__(self, name):
            return getattr(self._file, name)

    safetensors.safe_open = Materialized


def share_cuda_graphs() -> None:
    """One set of CUDA-graph trees for every thread of this process. torch keeps a compiled module's graph trees in
    thread-local storage, set up in the thread that first imports them, and FastVideo runs each clip's forward pass in
    a thread of its own: a clip replays the graphs an earlier clip recorded, but a size first seen in a later clip has
    to record a new graph, looks for the trees in its own thread and fails (``assert torch._C._is_key_in_tls``; the
    compiled video decoder's tile helpers at 1376x768 after 864x480). One clip renders at a time, so they are shared."""

    import collections
    import types

    import torch._inductor.cudagraph_trees as trees

    held = trees.local
    trees.local = types.SimpleNamespace(
        tree_manager_containers=getattr(held, "tree_manager_containers", {}),
        tree_manager_locks=getattr(held, "tree_manager_locks", collections.defaultdict(threading.Lock)))


class Worker:
    def __init__(self, model: str, compile_vae: bool = True) -> None:
        self.model, self.compile_vae = model, compile_vae
        self.gen = None
        self.loaded = False
        self.load_error: str | None = None
        self.load_s: float | None = None
        self.clips: dict[str, dict] = {}
        self.current: dict | None = None
        self.lock = threading.Lock()
        self.started = time.time()

    # ----- the model -----

    def load(self) -> None:
        try:
            t0 = time.perf_counter()
            materialize_safetensors()
            self.hook_stages()
            if self.compile_vae:
                share_cuda_graphs()
            from fastvideo import VideoGenerator
            self.gen = VideoGenerator.from_config(generator_config(self.model, self.compile_vae))
            self.load_s = round(time.perf_counter() - t0, 1)
            self.loaded = True
            log(f"loaded in {self.load_s} s")
        except BaseException as error:  # noqa: BLE001 - said on /health, and the process ends: the app sees it fail
            self.load_error = f"{type(error).__name__}: {str(error)[:400]}"
            log("load failed: " + self.load_error + "\n" + traceback.format_exc())
            os._exit(3)

    def hook_stages(self) -> None:
        """The clip's stage and its DiT forward count, from FastVideo's own H3 stages (wrapped, not changed)."""

        from fastvideo.models.dits import minimax_h3 as dit
        from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_decoding as dec
        from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_denoising as den

        worker = self

        def staged(cls, name):
            inner = cls.forward

            def forward(self, *args, **kwargs):
                worker.stage(name)
                return inner(self, *args, **kwargs)
            cls.forward = forward

        staged(den.MiniMaxH3DenoisingStage, "denoising")
        staged(dec.MiniMaxH3VideoDecodingStage, "decoding the video")
        staged(dec.MiniMaxH3AudioDecodingStage, "decoding the audio")
        inner = dit.MiniMaxH3Transformer3DModel.forward

        def forward(self, *args, **kwargs):
            out = inner(self, *args, **kwargs)
            clip = worker.current
            if clip is not None:
                clip["at"] = min(FORWARDS, int(clip.get("at") or 0) + 1)
            return out
        dit.MiniMaxH3Transformer3DModel.forward = forward

    def stage(self, name: str) -> None:
        clip = self.current
        if clip is None:
            return
        now = time.monotonic()
        clip["stage"] = name
        clip["stages_at"].setdefault(name, round(now - clip["t0"], 2))

    # ----- a clip -----

    def submit(self, body: dict) -> tuple[int, dict]:
        if not self.loaded:
            return 503, {"error": "the clip model is still loading" if not self.load_error else self.load_error}
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return 400, {"error": "a clip needs its prompt"}
        image = body.get("image")
        if image and not Path(image).is_file():
            return 400, {"error": f"the keyframe {image} is not a file"}
        last = body.get("last")
        if last and not Path(last).is_file():
            return 400, {"error": f"the end frame {last} is not a file"}
        width, height, frames = int(body.get("width") or 864), int(body.get("height") or 480), int(body.get("frames") or 124)
        if (frames - 5) % 17:
            return 400, {"error": f"MiniMax H3 makes 17n+5 frames, not {frames}"}
        output = body.get("output")
        if not output:
            return 400, {"error": "a clip needs its output path"}
        with self.lock:
            if self.current is not None:
                return 409, {"error": "a clip is rendering already"}
            handle = uuid.uuid4().hex[:12]
            clip = {"id": handle, "status": "queued", "stage": "queued", "at": 0, "of": FORWARDS, "error": None,
                    "output": None, "timings": {}, "stages_at": {}, "t0": time.monotonic(),
                    "request": {"prompt": prompt, "image": image, "last": last, "width": width, "height": height, "frames": frames,
                                "seed": int(body.get("seed") or 0), "output": str(output)}}
            self.clips[handle] = clip
            self.current = clip
            for old in list(self.clips)[:-20]:
                self.clips.pop(old, None)
        threading.Thread(target=self.run, args=(clip,), name=f"clip-{handle}", daemon=True).start()
        return 202, {"id": handle, "status": "queued"}

    @staticmethod
    def picture(path: str | None):
        """The clip's last frame, when one is given (MiniMax H3 holds a first and a last keyframe)."""

        if not path:
            return None
        from PIL import Image
        return Image.open(path).convert("RGB")

    def run(self, clip: dict) -> None:
        import torch
        from fastvideo.api.schema import GenerationRequest, InputConfig, OutputConfig, SamplingConfig

        r = clip["request"]
        clip["status"], clip["stage"] = "running", "conditioning"
        try:
            out = Path(r["output"])
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.stem + ".part.mp4")
            torch.cuda.reset_peak_memory_stats()
            request = GenerationRequest(
                prompt=r["prompt"], negative_prompt="",
                inputs=InputConfig(image_path=r["image"] or None, last_image=self.picture(r.get("last"))),
                sampling=SamplingConfig(seed=r["seed"], height=r["height"], width=r["width"], num_frames=r["frames"],
                                        fps=24, num_inference_steps=STEPS, guidance_scale=1.0, batch_cfg=False),
                output=OutputConfig(output_path=str(part), save_video=True, return_frames=False))
            t0 = time.perf_counter()
            result = self.gen.generate(request)
            wall = time.perf_counter() - t0
            made = Path(getattr(result, "video_path", None) or part)
            if not made.is_file():
                raise RuntimeError("FastVideo finished without writing the clip")
            made.replace(out)
            timings = {"render_s": round(wall, 1), "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
                       "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 1)}
            stages = getattr(getattr(result, "logging_info", None), "stages", None) or {}
            for name, info in stages.items():
                if isinstance(info, dict) and info.get("execution_time") is not None:
                    timings[name] = round(float(info["execution_time"]), 2)
            at = clip["stages_at"]
            if "denoising" in at:
                timings["dit_s"] = round(at.get("decoding the video", wall) - at["denoising"], 1)
                timings["decode_s"] = round(wall - at.get("decoding the video", wall), 1)
            clip.update(status="done", stage="done", output=str(out), timings=timings)
            log(f"clip {clip['id']} done in {wall:.1f} s: {json.dumps(timings)}")
        except BaseException as error:  # noqa: BLE001 - the clip fails with its reason; the worker stays for the next
            clip.update(status="failed", stage="failed", error=f"{type(error).__name__}: {str(error)[:400]}")
            log(f"clip {clip['id']} failed: {clip['error']}\n{traceback.format_exc()}")
        finally:
            try:
                torch.cuda.empty_cache()  # between clips the worker holds its weights, not the last clip's activations
            except Exception:  # noqa: BLE001
                pass
            with self.lock:
                self.current = None

    def view(self, clip: dict) -> dict:
        return {k: v for k, v in clip.items() if k not in ("t0", "request")}

    def health(self) -> dict:
        out = {"loaded": self.loaded, "load_s": self.load_s, "error": self.load_error, "busy": self.current is not None,
               "pid": os.getpid(), "up_s": round(time.time() - self.started, 1), "recipe": "FastH3 V2 NVFP4, VSA FP4 0.8"}
        if self.loaded:
            try:
                import torch
                out["allocated_gb"] = round(torch.cuda.memory_allocated() / 2 ** 30, 1)
                out["reserved_gb"] = round(torch.cuda.memory_reserved() / 2 ** 30, 1)
            except Exception:  # noqa: BLE001
                pass
        return out


def log(text: str) -> None:
    print(f"[fp4_worker {time.strftime('%H:%M:%S')}] {text}", flush=True)


def serve(worker: Worker, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: dict) -> None:
            raw = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                said = worker.health()
                return self.send(200 if said["loaded"] else 503, said)
            if self.path.startswith("/v1/clips/"):
                clip = worker.clips.get(self.path.rsplit("/", 1)[-1])
                return self.send(200, worker.view(clip)) if clip else self.send(404, {"error": "no such clip"})
            self.send(404, {"error": "no such route"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/clips":
                return self.send(404, {"error": "no such route"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self.send(400, {"error": "the body is not JSON"})
            code, said = worker.submit(body if isinstance(body, dict) else {})
            self.send(code, said)

        def log_message(self, *args):  # the app polls twice a second: no access log
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    log(f"listening on {host}:{port}")
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="the local FastVideo-FastH3-8-Step-V2-NVFP4-Consumer folder")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--no-compile-vae", action="store_true",
                   help="decode eagerly, without torch.compile (measured on the DGX Spark, 2026-11-16, 243 frames: 44 s against "
                        "32.5 s compiled at 864x480, 73 s against 57 s at 1376x768; a size's first clip compiles for ~6 s)")
    p.add_argument("--cache", default="~/ghost-studio/cache", help="FlashInfer, Triton and Inductor build caches")
    p.add_argument("--jit-only", action="store_true", help="build FlashInfer's NVFP4 kernels (4 jobs) and exit")
    a = p.parse_args(argv)
    for key, value in RECIPE_ENV.items():
        os.environ.setdefault(key, value)
    cache = Path(os.path.expanduser(a.cache))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(cache))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
    # ninja from this environment and nvcc from CUDA 13 for those builds
    os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), "/usr/local/cuda/bin", os.environ.get("PATH", "")])
    if a.jit_only:
        sys.exit(jit_warm())
    worker = Worker(a.model, compile_vae=not a.no_compile_vae)
    threading.Thread(target=serve, args=(worker, a.host, a.port), name="http", daemon=True).start()
    worker.load()
    threading.Event().wait()


def jit_warm() -> int:
    """FlashInfer's NVFP4 quantize and cuDNN GEMM once on small tensors, before any weights are resident: a first-use
    JIT build happens here (MAX_JOBS=4), never beside the 60 GB model."""

    import torch
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    a = torch.randn(256, 5376, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 5376, device="cuda", dtype=torch.bfloat16)
    ga, gb = (448 * 6) / a.float().abs().max(), (448 * 6) / b.float().abs().max()
    t0 = time.time()
    aq, asf = nvfp4_quantize(a, ga, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    bq, bsf = nvfp4_quantize(b, gb, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    mm_fp4(aq, bq.T, asf, bsf.T, 1.0 / (ga * gb), torch.bfloat16, None, backend=os.environ["FASTVIDEO_NVFP4_MM_BACKEND"])
    torch.cuda.synchronize()
    log(f"FlashInfer NVFP4 kernels ready in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    main()
