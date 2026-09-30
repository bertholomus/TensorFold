"""GLM-5.3's interleaved (GPT-J) rope: one table over the query's rope slice, the shared rope key, and the indexer heads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _table(COS, SIN, POS, R, n, THETA: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    """Rows pos..pos + R - 1 of cos/sin [R, HALF]: position * (theta ** (-2i / D))."""

    r = tl.program_id(0)
    P = tl.load(POS).to(tl.float32) + r
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = i < HALF
    inv = tl.exp2(-(2.0 * i.to(tl.float32) / (2 * HALF)) * tl.log2(THETA))
    phase = P * inv
    tl.store(COS + r * HALF + i, tl.cos(phase), mask=ok)
    tl.store(SIN + r * HALF + i, tl.sin(phase), mask=ok)


def table(cos: torch.Tensor, sin: torch.Tensor, pos: int, R: int, theta: float, rope_dim: int) -> None:
    half = rope_dim // 2
    if half == 0:
        return
    pos_dev = torch.zeros((1,), dtype=torch.int32, device=cos.device)
    pos_dev.fill_(pos)
    _table[(R, triton.cdiv(half, 128))](cos, sin, pos_dev, R, half, THETA=theta, HALF=half, BLOCK=128, num_warps=4)


@triton.jit
def _apply(X, COS, SIN, R, stride, HALF: tl.constexpr, PER: tl.constexpr):
    """One row: x[r, 2i] = x[r, 2i] * cos - x[r, 2i + 1] * sin, x[r, 2i + 1] likewise (interleaved pairing)."""

    r = tl.program_id(0)
    g = tl.program_id(1)
    i = g * PER + tl.arange(0, PER)
    ok = i < HALF
    c = tl.load(COS + r * HALF + i, mask=ok, other=1.0)
    s = tl.load(SIN + r * HALF + i, mask=ok, other=0.0)
    a = tl.load(X + r * stride + 2 * i, mask=ok, other=0.0).to(tl.float32)
    b = tl.load(X + r * stride + 2 * i + 1, mask=ok, other=0.0).to(tl.float32)
    tl.store(X + r * stride + 2 * i, (a * c - b * s).to(tl.bfloat16), mask=ok)
    tl.store(X + r * stride + 2 * i + 1, (b * c + a * s).to(tl.bfloat16), mask=ok)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cols: int) -> torch.Tensor:
    """Interleaved rope over each row's first ``cols`` columns, in place; the rest of the row is untouched."""

    rows, wide = x.shape
    if cols == 0 or wide < cols:
        return x
    view = x[:, :cols]
    if not view.is_contiguous():
        raise ValueError("rope: the rope slice must be contiguous columns")
    half = cols // 2
    per = min(256, triton.next_power_of_2(half))
    _apply[(rows, triton.cdiv(half, per))](view, cos, sin, rows, x.stride(0), HALF=half, PER=per, num_warps=4)
    return x


def apply(qr: torch.Tensor, rope_key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cfg) -> None:
    """Rotate the last qk_rope dims of every head of q [R, H * qk_dim] and the rows' shared rope key [R, qk_rope]."""

    R, heads = qr.shape[0], qr.shape[1] // cfg.qk_dim
    d = cfg.qk_dim
    head_rows = qr.view(R * heads, d)
    _rotate(head_rows, cos, sin, cfg.qk_rope)
    _rotate(rope_key, cos, sin, cfg.qk_rope)
