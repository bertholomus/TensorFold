"""Row-invariant kernels for the pieces GLM-5.3 needs that Flash's glue does not provide: LayerNorm, unclamped swiglu."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# Flash's row-invariant kernels that GLM-5.3 uses unchanged: RMSNorm (+64-group sums), the plain residual add of a
# rank-gathered fp32 partial, the fp32 router matmul, and sigmoid noaux_tc top-k with the shared expert appended.
from tensorfold.families.glm5_next.cuda.glue import residual_add, rmsnorm, router, select  # noqa: F401

# -- prompt chunks' partials summed by row shares: residual_add's branch arithmetic, a share of the rows on each rank ---
@triton.jit
def _rank_sum(QR, BR, n, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    """bf16(every rank's fp32 partial summed rank 0 first): QR [world, n] -> BR [n]."""

    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < n
    acc = tl.load(QR + i, mask=ok, other=0.0)
    for k in tl.static_range(1, WORLD):
        acc = acc + tl.load(QR + k * n + i, mask=ok, other=0.0)
    tl.store(BR + i, acc.to(tl.bfloat16), mask=ok)


def rank_sum(qr: torch.Tensor, br: torch.Tensor) -> torch.Tensor:
    """qr [world, n] fp32 -> br [n] bf16: residual_add's branch (the partials summed in rank order) for these values."""

    n = br.numel()
    block = 1024
    _rank_sum[(triton.cdiv(n, block),)](qr, br, n, WORLD=qr.shape[0], BLOCK=block, num_warps=4)
    return br


# -- a prompt chunk's residual add and the next RMSNorm in one pass (residual_add's and rmsnorm's arithmetic) ----------
@triton.jit
def _residual_rmsnorm(X, G, W, XS, eps, D: tl.constexpr, BLOCK: tl.constexpr, OUT, o_stride):
    """Row r: X = bf16(X + bf16(G)) (residual_add with one bf16 branch), OUT = rmsnorm(X) and its 64-group sums."""

    r = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    ok = d < D
    x = tl.load(X + r * D + d, mask=ok, other=0.0).to(tl.float32)
    g = tl.load(G + r * D + d, mask=ok, other=0.0).to(tl.bfloat16).to(tl.float32)
    xn = (x + g).to(tl.bfloat16)
    tl.store(X + r * D + d, xn, mask=ok)
    xf = xn.to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / D + eps)
    w = tl.load(W + d, mask=ok, other=0.0).to(tl.float32)
    y = (w * (xf * rinv).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + r * o_stride + d, y, mask=ok)
    gs = tl.sum(tl.reshape(tl.where(ok, y.to(tl.float32), 0.0), (BLOCK // 64, 64)), axis=1)
    gi = tl.arange(0, BLOCK // 64)
    tl.store(XS + r * (D // 64) + gi, gs, mask=gi < D // 64)


def residual_rmsnorm(x: torch.Tensor, branch: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor,
                     xs: torch.Tensor) -> torch.Tensor:
    """x += the bf16 branch (residual_add), then out / xs = rmsnorm(x) (Flash's rmsnorm with its sums), one launch."""

    rows, d = x.shape
    if not (x.is_contiguous() and branch.is_contiguous() and out.stride(-1) == 1):
        raise ValueError("residual_rmsnorm: contiguous rows")
    block = triton.next_power_of_2(d)
    _residual_rmsnorm[(rows,)](x, branch, w, xs, eps, D=d, BLOCK=block, OUT=out, o_stride=out.stride(0),
                               num_warps=4 if block <= 2048 else 8)
    return out


# -- the embedding split by vocabulary (weights.embed_span): a rank's own rows, then each token's from its holder ------
@triton.jit
def _embed_span(IDS, W, OUT, lo, n, D: tl.constexpr):
    """Row r, group g: 64 values of token IDS[r]'s row when this rank holds it (table rows lo .. lo + n), else zeros."""

    row = tl.program_id(0)
    d = tl.program_id(1) * 64 + tl.arange(0, 64)
    tok = tl.load(IDS + row).to(tl.int64) - lo
    mine = (tok >= 0) & (tok < n)
    v = tl.load(W + tl.where(mine, tok, 0) * D + d, mask=(d < D) & mine, other=0.0)
    tl.store(OUT + row * D + d, v)


def embed_span(ids: torch.Tensor, table: torch.Tensor, lo: int, out: torch.Tensor) -> torch.Tensor:
    """out [R, D] bf16: the rows of the tokens this rank's table slice holds (its first token ``lo``), zeros for the
    rest."""

    rows, dims = out.shape
    if not out.is_contiguous() or dims % 64:
        raise ValueError("embed_span: out must be contiguous rows of a multiple of 64 values")
    _embed_span[(rows, dims // 64)](ids, table, out, lo, table.shape[0], D=dims, num_warps=1)
    return out


@triton.jit
def _embed_pick(IDS, G, OUT, R, SPAN, BIG, EXTRA, DIV, D: tl.constexpr):
    """Row r, group g: token IDS[r]'s row from G [world, R, D], every rank's rows, taking the holder's: blocks of 128
    tokens, the first EXTRA ranks SPAN blocks each (BIG blocks in all) and the rest DIV (weights.embed_owner)."""

    row = tl.program_id(0)
    d = tl.program_id(1) * 64 + tl.arange(0, 64)
    block = tl.load(IDS + row).to(tl.int64) // 128
    owner = tl.where(block < BIG, block // SPAN, EXTRA + (block - BIG) // DIV)
    tl.store(OUT + row * D + d, tl.load(G + (owner * R + row) * D + d))


def embed_pick(ids: torch.Tensor, gathered: torch.Tensor, out: torch.Tensor, vocab: int) -> torch.Tensor:
    """out [R, D] bf16: each token's row from the rank that holds it, out of gathered [world, R, D] (every rank's
    embed_span output)."""

    world, rows, dims = gathered.shape
    if not (out.is_contiguous() and gathered.is_contiguous()) or dims % 64:
        raise ValueError("embed_pick: contiguous rows of a multiple of 64 values")
    base, extra = divmod(-(-vocab // 128), world)
    _embed_pick[(rows, dims // 64)](ids, gathered, out, rows, base + 1, extra * (base + 1), extra, max(base, 1),
                                    D=dims, num_warps=1)
    return out


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
