"""NVFP4 GEMMs for the fast FastFrames pipeline on a GB10 (FlashInfer's quantize and GEMM kernels).

A ``Weight`` is quantized once. Activations are quantized on every call with a scale fixed after one calibration clip
(twice the largest value the layer saw: the block scales are FP8, so the headroom costs no precision, and no pass over
the activation is spent finding its maximum). ``freeze`` also times every GEMM backend on the layer's real shape and
keeps the fastest; where BF16 wins (small square layers) the layer stays BF16.
"""
import time
import torch, torch.nn.functional as F
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

TOP = 448.0 * 6.0
BACKENDS = ("cudnn", "b12x", "cutlass")
ALL = []


def quant(x, g):
    return nvfp4_quantize(x, g, sfLayout=SfLayout.layout_128x4, do_shuffle=False, backend="cuda")


class Weight:
    def __init__(self, weight):
        self.w, self.shape = weight, tuple(weight.shape)
        self.gw = TOP / weight.abs().amax().float().clamp_min(1e-12)
        self.wq, self.wsf = quant(weight.contiguous(), self.gw)
        self.wq_t, self.wsf_t = self.wq.T, self.wsf.T
        self.peak, self.ga, self.alpha, self.backend, self.rows = None, None, None, "cudnn", 0
        ALL.append(self)

    def __call__(self, x):
        shape = x.shape
        a = x.reshape(-1, shape[-1])
        if a.dtype != torch.bfloat16: a = a.to(torch.bfloat16)
        a = a.contiguous()
        if self.backend == "bf16":
            return F.linear(a, self.w).view(*shape[:-1], -1)
        if self.ga is None:  # calibrating: the scale from this very activation
            amax = a.abs().amax().float().clamp_min(1e-12)
            self.peak = amax if self.peak is None else torch.maximum(self.peak, amax)
            self.rows = a.shape[0]
            ga = TOP / amax; alpha = 1.0 / (ga * self.gw)
        else:
            ga, alpha = self.ga, self.alpha
        aq, asf = quant(a, ga)
        y = mm_fp4(aq, self.wq_t, asf, self.wsf_t, alpha, torch.bfloat16, None, backend=self.backend)
        return y.view(*shape[:-1], -1)

    def freeze(self, tune=True):
        self.ga = TOP / (2.0 * self.peak); self.alpha = 1.0 / (self.ga * self.gw)
        if not tune: return
        n, k = self.shape
        a = torch.randn(self.rows, k, device=self.wq.device, dtype=torch.bfloat16) * (float(self.peak) / 4)
        best, took = "bf16", (_time(lambda: F.linear(a, self.w)) if self.w is not None else float("inf"))
        self.bf16_ms = took
        for backend in BACKENDS:
            self.backend = backend
            try:
                t = _time(lambda: self(a))
            except Exception:
                continue
            if t < took: best, took = backend, t
        self.backend, self.ms = best, took
        if best == "bf16":
            self.wq = self.wsf = self.wq_t = self.wsf_t = None


def _time(fn, n=12, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3


def freeze(tune=True):
    """After the calibration clip: fix every layer's scale, pick its backend. Returns {shape: (backend, ms, bf16 ms)}."""
    table, cache = {}, {}
    for w in ALL:
        key = (w.rows, w.shape)
        if tune and key in cache:
            w.freeze(tune=False); w.backend, w.ms, w.bf16_ms = cache[key]
            if w.backend == "bf16": w.wq = w.wsf = w.wq_t = w.wsf_t = None
        else:
            w.freeze(tune)
            if tune: cache[key] = (w.backend, w.ms, w.bf16_ms)
        if tune: table[str(key)] = (w.backend, round(w.ms, 3), round(w.bf16_ms, 3))
    return table
