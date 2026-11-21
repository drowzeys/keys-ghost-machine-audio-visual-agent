"""NVFP4 self-attention over FastFrames' frame cache on a GB10 (the SageAttention-derived SM120 kernel in FlashInfer).

The kernel wants K and V packed in 4 bits (K in rows, V transposed, scale factors swizzled). Quantizing the whole cache
on every call would cost more than the kernel saves, so ``Packed`` keeps a packed copy beside the BF16 cache and
requantizes only the 128-token blocks a newly written frame touches (the quantize kernels work block by block).
Q is quantized on every call. Nothing is centred: the QK correction the kernel adds is zero.
"""
import torch
from flashinfer.nvfp4_attention_sm120 import get_nvfp4_attention_sm120_module

B = 128
STRIDE = True
STRIDED = [None]


def strided_module():
    """FlashInfer's SM120 NVFP4 attention module rebuilt to take K and V as strided views (the kernel already addresses
    them through their strides; only the binding insisted on contiguous tensors). With it a prefix of the packed cache
    goes to the kernel in place instead of being copied on every call. None if the build fails."""
    if STRIDED[0] is None:
        try:
            import os
            from pathlib import Path
            import flashinfer.jit.nvfp4_attention_sm120 as j
            from flashinfer.jit.core import gen_jit_spec, current_compilation_context, sm120a_nvcc_flags
            src = [j._nvfp4_attention_sm120_source_path(f) for f in j._NVFP4_ATTENTION_SM120_SOURCE_FILES]
            code = src[0].read_text()
            for name in ("k_fp4", "v_fp4_t"):
                assert f"CHECK_INPUT({name});" in code
                code = code.replace(f"CHECK_INPUT({name});", f"CHECK_CUDA({name});")
            out = Path(os.environ.get("FF_KERNEL_DIR", "/tmp/ff-kern")); out.mkdir(parents=True, exist_ok=True)
            patched = out / "ff_nvfp4_attention_sm120_strided_binding.cu"
            if not patched.exists() or patched.read_text() != code: patched.write_text(code)
            try: flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
            except RuntimeError: flags = sm120a_nvcc_flags
            spec = gen_jit_spec("ff_nvfp4_attention_sm120_strided", [patched, src[1]], extra_cuda_cflags=flags + j._NVFP4_ATTENTION_SM120_CUDA_FLAGS,
                                extra_include_paths=j._nvfp4_attention_sm120_include_paths())
            STRIDED[0] = spec.build_and_load()
        except Exception as error:
            print("fattn: the strided build failed, copying instead:", str(error)[-400:], flush=True)
            STRIDED[0] = False
    return STRIDED[0] or None


def up(n): return -(-n // B) * B


class Packed:
    """The packed K/V of one layer for a cache of ``tokens`` tokens, and the buffers of one query frame of ``frame`` tokens."""

    def __init__(self, heads, dim, tokens, frame, device):
        self.h, self.d, self.n, self.m = heads, dim, up(tokens), up(frame)
        f8, u8 = torch.float8_e4m3fn, torch.uint8
        self.k4 = torch.zeros(1, heads, self.n, dim // 2, dtype=u8, device=device)
        self.ks = torch.zeros(1, heads, self.n, dim // 16, dtype=f8, device=device)
        self.v4 = torch.zeros(1, heads, dim, self.n // 2, dtype=u8, device=device)
        size = heads * dim * self.n // 16  # with slack after it: write() addresses windows of it through a wider view
        self.vs = torch.zeros(size + dim * self.n // 16, dtype=f8, device=device)[:size].view(1, heads, dim, self.n // 16)
        self.q4 = torch.zeros(1, heads, self.m, dim // 2, dtype=u8, device=device)
        self.qs = torch.zeros(1, heads, self.m, dim // 16, dtype=f8, device=device)
        self.out = torch.empty(1, heads, self.m, dim, dtype=torch.bfloat16, device=device)
        self.zero = {}
        self.mod = get_nvfp4_attention_sm120_module()
        strided = strided_module() if STRIDE else None
        self.fwd = strided.fwd if strided is not None else self.mod.fwd

    def write(self, k, v, start, end):
        """Requantize the blocks holding tokens [start, end) from the BF16 caches ``k``, ``v`` ([1, tokens, heads, dim]).
        Everything after ``end`` in those blocks must be zero in ``k`` and ``v`` (it is: the frames are written in order)."""
        a, b = start // B * B, min(up(end), self.n)
        w = b - a
        self.mod.scaled_fp4_quant_permute(k[:, a:b], self.k4[:, :, a:b].permute(0, 2, 1, 3), self.ks[:, :, a:b].permute(0, 2, 1, 3), 0)
        # V's scale factors are stored swizzled over the whole width: a window of them starts 4 bytes a token further on
        sh = self.vs.stride(1)
        vs = torch.as_strided(self.vs, (1, self.d, self.h, w // 16), (self.vs.stride(0), self.n // 16, sh, 1), 4 * a)
        self.mod.scaled_fp4_quant_trans(v[:, a:b], self.v4[..., a // 2:b // 2].permute(0, 2, 1, 3), vs, 0)

    def attend(self, q, end, scale=None):
        """q: [1, frame, heads, dim] BF16 -> attention over the first ``end`` cached tokens, [1, frame, heads, dim]."""
        length = q.shape[1]
        self.mod.scaled_fp4_quant(q, self.q4.permute(0, 2, 1, 3)[:, :length], self.qs.permute(0, 2, 1, 3)[:, :length], 0)
        n = up(end)
        if n == self.n:
            k4, ks, v4, vs = self.k4, self.ks, self.v4, self.vs
        else:  # the kernel walks the physical length: give it only the blocks in use (the scale factors densely)
            ks = self.ks[:, :, :n].contiguous()
            if self.fwd is not self.mod.fwd: k4, v4 = self.k4[:, :, :n], self.v4[..., :n // 2]
            else: k4, v4 = self.k4[:, :, :n].contiguous(), self.v4[..., :n // 2].contiguous()
            vs = torch.as_strided(self.vs, (1, self.h, 2, 4 * n), (self.vs.stride(0), self.vs.stride(1), 4 * self.n, 1)).contiguous().view(1, self.h, self.d, n // 16)
        if n not in self.zero:
            self.zero[n] = torch.zeros(1, self.h, 1, n, dtype=torch.float32, device=q.device)
        self.fwd(self.q4, k4, v4, self.qs, ks, vs, self.zero[n], self.out, None,
                     float(scale or self.d ** -0.5), False, False, int(end))
        return self.out[:, :, :length].transpose(1, 2)
