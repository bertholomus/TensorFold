"""Ling-3.1-flash's shared-expert SwiGLU with the config's clamp (share_expert_swiglu_limit_list).

On the bf16 [gate | up] rows: ``bf16(min(bf16(silu(g)), L) * clip(u, -L, L))``, torch's
``clamp(silu(g), max=L) * clamp(u, -L, L)`` on bf16 tensors: silu rounded once, the clamps exact (L is a bf16 value;
config.py refuses others), the product rounded once. The clamp sits after silu, as the serving stacks apply it
(GLM's swiglu_limit clamps the gate before silu: glm5_next glue.swiglu). NaN passes through, as torch's clamp keeps
it. At L = inf the bits are glm_moe_dsa glue.silu_mul's (the same operations and 64-group sums), so a layer without
a clamp gets the plain kernel's output. The routed experts apply the same rule in their epilogue
(tensorfold.cuda.exl3.experts.ACT_BF16_POST).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_post(GU, OUT, XS, LIMIT, W: tl.constexpr, BLOCK: tl.constexpr):
    """Row r, BLOCK columns: the clamped activation and its 64-group sums (fp32, for the next EXL3 linear)."""

    r = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GU + r * (2 * W) + d).to(tl.float32)
    u = tl.load(GU + r * (2 * W) + W + d).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    s = tl.where(s > LIMIT, LIMIT, s)
    u = tl.where(u > LIMIT, LIMIT, tl.where(u < -LIMIT, -LIMIT, u))
    a = (s * u).to(tl.bfloat16)
    tl.store(OUT + r * W + d, a)
    t = tl.sum(tl.reshape(a.to(tl.float32), (BLOCK // 64, 64)), axis=1)
    tl.store(XS + r * (W // 64) + cb * (BLOCK // 64) + tl.arange(0, BLOCK // 64), t)


def silu_mul(gu: torch.Tensor, out: torch.Tensor, xs: torch.Tensor, limit: float = math.inf) -> None:
    """gu [R, 2W] bf16 -> out [R, W] bf16 and xs [R, W / 64] fp32, clamped at ``limit`` (inf: no clamp)."""

    rows, w = out.shape
    block = math.gcd(512, w)
    if block < 64:
        raise ValueError(f"silu_mul: width {w} is not a multiple of 64")
    _swiglu_post[(rows, w // block)](gu, out, xs, float(limit), W=w, BLOCK=block, num_warps=4)
