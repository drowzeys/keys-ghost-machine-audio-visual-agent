"""A faster FastFrames forward for one GB10 (DGX Spark): same weights, same pipeline, the per-frame DiT block rewritten.

    import fastpipe; fastpipe.install(model, fp4=False)

What changes against the released per-frame path:
* every Linear with a bias is a GEMM and then an add (the GB10's addmm path runs the 8960->1536 layer at 12 TFLOPS, the
  GEMM alone at 86);
* one fused QKV GEMM per layer, RoPE from FP32 (cos, sin) tables, RMS-norm and RoPE in one FP32 pass (FastFrames-S ships
  with complex128 RoPE and three separate projections);
* the timestep modulation of a layer is worked out once per timestep, not once per call;
* optionally (fp4=True) the large GEMMs in NVFP4: weights quantized once, activations by a CUDA kernel on every call.
"""
import torch, torch.nn.functional as F
import fastframes.wan.causal_model as cm
from fastframes.wan.attention import attention, cross_attention

OPT = {"fp4": False, "attn": None}


def lin(x, w, b):
    y = F.linear(x, w)
    return y if b is None else y.add_(b)


class Dense:
    """A Linear layer's weights as this pipeline runs them (BF16 GEMM + add, or NVFP4)."""

    def __init__(self, weight, bias, fp4=False, lean=False):
        self.w, self.b, self.q = weight, bias, None
        if fp4 and min(weight.shape) >= 1024:
            import fp4
            self.q = fp4.Weight(weight)
            if lean: self.w = self.q.w = None  # the BF16 weight is not kept

    def __call__(self, x):
        if self.q is not None:
            y = self.q(x)
            return y if self.b is None else y.add_(self.b)
        return lin(x, self.w, self.b)


def rms_rope(x, weight, table, shape, eps):
    """RMS-norm over the whole width, then RoPE per head, in FP32; BF16 out. x: [B, L, C]."""
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight
    pairs = x.view(*shape[:-1], shape[-1] // 2, 2)
    re, im = pairs[..., 0], pairs[..., 1]
    cos, sin = table[..., 0], table[..., 1]
    return torch.stack((re * cos - im * sin, re * sin + im * cos), dim=-1).view(shape).to(torch.bfloat16)


def pre(x, scale, shift, eps: float):
    """LayerNorm (no affine) and the timestep modulation."""
    return F.layer_norm(x, (x.shape[-1],), None, None, eps) * scale + shift


def post(x, y, gate):
    return x + y * gate


def rms(x, weight, eps: float):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight).to(torch.bfloat16)


def gelu(x):
    return F.gelu(x, approximate="tanh")


FN = {"pre": pre, "post": post, "rms": rms, "rms_rope": rms_rope, "gelu": gelu}


def compile_pieces():
    """Each small elementwise chain as one generated kernel (they are launch-bound: ~60 tiny kernels a layer otherwise)."""
    import torch._inductor.config as inductor
    inductor.emulate_precision_casts = True
    for name in list(FN):
        FN[name] = torch.compile(FN[name], dynamic=False)


def refine_gemm(self, nK, nV, phi_v, e, layers=slice(None)):
    """GroupGEMM.forward of the KV refinement network with each baddbmm as a bmm and an add (the GB10's fused-bias GEMM is slow)."""
    def mm(bias, x, w): return torch.bmm(x, w[layers]).add_(bias[layers])
    b = nK.shape[0]
    pk, pv = mm(self.proj_k_b, nK, self.proj_k_w), mm(self.proj_v_b, nV, self.proj_v_w)
    h = mm(self.fuse_b, torch.cat([pk, pv, phi_v], dim=-1), self.fuse_w)
    e1 = F.silu(e).reshape(1, 1, self.d_a).expand(b, 1, self.d_a).contiguous()
    shift, scale = mm(self.mod_b, e1, self.mod_w).chunk(2, dim=-1)
    h = F.layer_norm(h, (self.d_v,)) * (1 + scale) + shift
    h = mm(self.m0_b, h, self.m0_w)
    h = mm(self.m2_b, F.gelu(h, approximate="tanh"), self.m2_w)
    k = self._renorm_per_head(nK + mm(self.dK_b, h, self.dK_w), ref=nK)
    v = (nV + mm(self.dV_b, h, self.dV_w)) * torch.exp(self.v_logscale[layers])
    return k, v


class State:
    def __init__(self, block, fp4, lean):
        sa, ca = block.self_attn, block.cross_attn
        self.qkv = Dense(torch.cat([sa.q.weight, sa.k.weight, sa.v.weight]).contiguous(),
                         torch.cat([sa.q.bias, sa.k.bias, sa.v.bias]).contiguous(), fp4, lean)
        self.o, self.cq, self.co = Dense(sa.o.weight, sa.o.bias, fp4, lean), Dense(ca.q.weight, ca.q.bias, fp4, lean), Dense(ca.o.weight, ca.o.bias, fp4, lean)
        self.f0, self.f2 = Dense(block.ffn[0].weight, block.ffn[0].bias, fp4, lean), Dense(block.ffn[2].weight, block.ffn[2].bias, fp4, lean)
        if fp4 and lean:  # this pipeline no longer reads the layers' own BF16 weights: free them
            for layer in (sa.q, sa.k, sa.v, sa.o, ca.q, ca.o, block.ffn[0], block.ffn[2]):
                if min(layer.weight.shape) >= 1024: layer.weight = torch.nn.Parameter(layer.weight.new_empty(0), requires_grad=False)
        self.nq, self.nk, self.cnq = sa.norm_q.weight.float(), sa.norm_k.weight.float(), ca.norm_q.weight.float()
        self.eps = sa.norm_q.eps
        self.mods = {}

    def mod(self, block, e):
        key = e.data_ptr()
        if key not in self.mods:
            m = [u.flatten(1, 2) for u in (block.modulation.unsqueeze(1) + e).chunk(6, dim=2)]  # six [B, 1, C]
            self.mods[key] = (m[0], 1 + m[1], m[2], m[3], 1 + m[4], m[5], e)  # e kept alive: its address is the key
        return self.mods[key]


def forward_frame(self, x, e, grid_sizes, freqs, cross_kv, kv_cache, frame, fast=None, layer=0):
    st = self._ff
    tokens = x.shape[1]
    start, end = frame * tokens, (frame + 1) * tokens
    m = st.mod(self, e)
    sa, ca = self.self_attn, self.cross_attn
    shape = (x.shape[0], tokens, sa.num_heads, sa.head_dim)
    q, k, v = st.qkv(FN["pre"](x, m[1], m[0], self.norm1.eps)).chunk(3, dim=-1)
    q = FN["rms_rope"](q, st.nq, fast.rope[frame], shape, st.eps)
    kv_cache["k"][:, start:end].copy_(FN["rms_rope"](k, st.nk, fast.rope[frame], shape, st.eps))
    kv_cache["v"][:, start:end].copy_(v.view(shape))
    if OPT.get("dump") and frame == 10 and layer in OPT["dump"][1]:
        import os
        path = f"{OPT['dump'][0]}-{layer}.pt"
        if not os.path.exists(path): torch.save({"q": q.cpu(), "k": kv_cache["k"][:, :end].cpu(), "v": kv_cache["v"][:, :end].cpu()}, path)
    mode = ATTN.get(layer)
    if mode is None:
        y = attention(q, kv_cache["k"][:, :end], kv_cache["v"][:, :end])
    else:
        P = packed(layer, sa.num_heads, sa.head_dim, kv_cache["k"].shape[1], tokens, x.device)
        P.write(kv_cache["k"], kv_cache["v"], start, end)
        y = P.attend(q, end)
        if mode == "measure":  # the calibration clip: BF16 result used, the NVFP4 one's error kept
            exact = attention(q, kv_cache["k"][:, :end], kv_cache["v"][:, :end])
            ERR.setdefault(layer, []).append(((y.float() - exact.float()).norm() / exact.float().norm()))
            y = exact
    x = FN["post"](x, st.o(y.flatten(2)), m[2])
    cq = FN["rms"](st.cq(self.norm3(x)), st.cnq, st.eps).view(shape)
    x = x + st.co(cross_attention(cq, cross_kv["k"], cross_kv["v"]).flatten(2))
    y = st.f2(FN["gelu"](st.f0(FN["pre"](x, m[4], m[3], self.norm2.eps))))
    return FN["post"](x, y, m[5])


ATTN, ERR, POOL = {}, {}, {}


def packed(layer, heads, dim, tokens, frame, device):
    if layer not in POOL:
        import fattn
        POOL[layer] = fattn.Packed(heads, dim, tokens, frame, device)
    return POOL[layer]


def attention_fp4(model, limit=None):
    """Self-attention in NVFP4. With no ``limit``: measure every layer's error during the next clip (BF16 results are
    used meanwhile). With ``limit``: keep NVFP4 on the layers whose mean relative error was under it. Returns the errors."""
    layers = range(len(model.dit.blocks))
    if limit is None:
        ERR.clear(); ATTN.update({i: "measure" for i in layers})
        if not hasattr(model, "_ff_refine"):
            model._ff_refine = model._refine
            def refine(x0, frame, kv_cache):  # the refiner rewrites the frame's K/V: pack them again
                model._ff_refine(x0, frame, kv_cache)
                tokens = model.frame_tokens
                for i, mode in ATTN.items():
                    if mode is not None: POOL[i].write(kv_cache[i]["k"], kv_cache[i]["v"], frame * tokens, (frame + 1) * tokens)
            model._refine = refine
        return {}
    errors = {i: float(torch.stack(ERR[i]).mean()) for i in layers}
    ATTN.update({i: ("fp4" if errors[i] < limit else None) for i in layers})
    for i in layers:
        if ATTN[i] is None: POOL.pop(i, None)
    return errors


def install(model, fp4=False, attn=None, lean=False, jit=False, refine=True):
    """Make ``model`` (a fastframes.pipeline.FastFrames) use this forward. Returns the number of NVFP4 layers."""
    OPT["fp4"], OPT["attn"] = fp4, attn
    cfg = model.config
    if model.fast is None:
        model.fast = cm.FastOps(model.dit, *cfg.latent_size, cfg.num_frames)
    model.fast.qkv = None  # FastOps' own fused QKV copies are not used here
    count = 0
    for block in model.dit.blocks:
        block._ff = State(block, fp4, lean)
        count += sum(d.q is not None for d in (block._ff.qkv, block._ff.o, block._ff.cq, block._ff.co, block._ff.f0, block._ff.f2))
    cm.CausalWanAttentionBlock.forward_frame = forward_frame
    if jit: compile_pieces()
    if refine:
        import fastframes.models.kv_refinement as kvr
        kvr.GroupGEMM.forward = refine_gemm
    if lean: torch.cuda.empty_cache()
    return count
