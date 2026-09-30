"""Row-invariant kernels for the pieces GLM-5.3 needs that Flash's glue does not provide: LayerNorm, unclamped swiglu."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# -- LayerNorm (biased, the indexer key norm) -----------------------------------------------------------------
@triton.jit
def _layernorm(X, x_stride, W, B, OUT, o_stride, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    ok = d < D
    x = tl.load(X + r * x_stride + d, mask=ok, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(ok, x, 0.0), axis=0) / D
    xc = tl.where(ok, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    y = xc / tl.sqrt(var + eps) * tl.load(W + d, mask=ok, other=0.0).to(tl.float32) \
        + tl.load(B + d, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * o_stride + d, y.to(tl.bfloat16), mask=ok)


def layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float, out: torch.Tensor) -> torch.Tensor:
    rows, d = x.shape
    block = triton.next_power_of_2(d)
    if out.stride(-1) != 1:
        raise ValueError("layernorm: rows of out must be contiguous")
    _layernorm[(rows,)](x, x.stride(0), w, b, out, out.stride(0), eps, D=d, BLOCK=block,
                        num_warps=4 if block <= 2048 else 8)
    return out


# -- swiglu without the clamp ---------------------------------------------------------------------------------
@triton.jit
def _swiglu_plain(GU, OUT, XS, W: tl.constexpr, BLOCK: tl.constexpr):
    """Dense MLP without swiglu_limit (GLM-5.3 has none): bf16(bf16(silu(g)) * u) and its 64-group sums."""

    r = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GU + r * (2 * W) + d).to(tl.float32)
    u = tl.load(GU + r * (2 * W) + W + d).to(tl.float32)
    a = ((g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16)
    tl.store(OUT + r * W + d, a)
    s = tl.sum(tl.reshape(a.to(tl.float32), (BLOCK // 64, 64)), axis=1)
    tl.store(XS + r * (W // 64) + cb * (BLOCK // 64) + tl.arange(0, BLOCK // 64), s)


def silu_mul(gu: torch.Tensor, out: torch.Tensor, xs: torch.Tensor) -> None:
    """GLM-5.3's activation: silu(gate) * up, no limit (act_limit 0 in the reference)."""

    rows, w = out.shape
    block = math.gcd(512, w)
    if block < 64:
        raise ValueError(f"silu_mul: width {w} is not a multiple of 64")
    _swiglu_plain[(rows, w // block)](gu, out, xs, W=w, BLOCK=block, num_warps=4)
