"""GLM-5.3's interleaved (GPT-J) rope: one table for the queries' rope slices, the shared rope key and the indexer heads.

Layouts (ExLlamaV3 ``mla_attn`` is the reference):
- query heads are [qk_nope | qk_rope]: the rope slice is each head's LAST ``qk_rope`` columns (after q_b);
- the shared rope key is kv_a's output past the latent (``lat[:, kv_lora:]``, a strided view);
- indexer heads (keys and queries, ``index_dim`` wide) rotate their LEADING ``qk_rope`` columns.
"""

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


def table(cos: torch.Tensor, sin: torch.Tensor, pos, R: int, theta: float, rope_dim: int) -> None:
    """cos/sin rows for positions pos .. pos + R - 1; ``pos`` an int or a device int32 [1] (graph-capturable)."""

    half = rope_dim // 2
    if half == 0:
        return
    if isinstance(pos, torch.Tensor):
        pos_dev = pos
    else:
        pos_dev = torch.full((1,), int(pos), dtype=torch.int32, device=cos.device)
    _table[(R, triton.cdiv(half, 128))](cos, sin, pos_dev, R, half, THETA=theta, HALF=half, BLOCK=128, num_warps=4)


@triton.jit
def _apply_heads(X, COS, SIN, row_stride, head_stride, off, HALF: tl.constexpr, PER: tl.constexpr):
    """Program (row r, head j): rotate columns off .. off + 2 HALF of head j of row r by row r's angles."""

    r = tl.program_id(0)
    j = tl.program_id(1)
    i = tl.arange(0, PER)
    ok = i < HALF
    base = X + r * row_stride + j * head_stride + off
    c = tl.load(COS + r * HALF + i, mask=ok, other=1.0)
    s = tl.load(SIN + r * HALF + i, mask=ok, other=0.0)
    a = tl.load(base + 2 * i, mask=ok, other=0.0).to(tl.float32)
    b = tl.load(base + 2 * i + 1, mask=ok, other=0.0).to(tl.float32)
    tl.store(base + 2 * i, (a * c - b * s).to(tl.bfloat16), mask=ok)
    tl.store(base + 2 * i + 1, (b * c + a * s).to(tl.bfloat16), mask=ok)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, heads: int, head_dim: int, off: int,
           cols: int) -> torch.Tensor:
    """In place: rows [R, >= heads * head_dim] (row stride may exceed the width), each head's columns off .. off + cols.

    Row r uses cos/sin row r, so a window's rows rotate exactly as they would alone (row-invariant).
    """

    if cols == 0 or heads == 0:
        return x
    if x.dim() != 2 or x.stride(-1) != 1:
        raise ValueError("rope: rows must have unit column stride")
    rows = x.shape[0]
    if x.shape[1] < heads * head_dim or off + cols > head_dim or cols % 2:
        raise ValueError(f"rope: {heads} heads of {head_dim} (cols {off}..{off + cols}) do not fit {tuple(x.shape)}")
    half = cols // 2
    _apply_heads[(rows, heads)](x, cos, sin, x.stride(0), head_dim, off, HALF=half,
                                PER=triton.next_power_of_2(half), num_warps=1)
    return x


def apply_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cfg, heads: int) -> None:
    """Query rows [R, heads * qk_dim] (after q_b): each head's last qk_rope columns."""

    rotate(q, cos, sin, heads=heads, head_dim=cfg.qk_dim, off=cfg.qk_nope, cols=cfg.qk_rope)


def apply_key(rope_key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cfg) -> None:
    """The shared rope key rows [R, qk_rope] (a strided view of kv_a's output)."""

    rotate(rope_key, cos, sin, heads=1, head_dim=cfg.qk_rope, off=0, cols=cfg.qk_rope)


def apply_index(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cfg, heads: int) -> None:
    """Indexer rows [R, heads * index_dim]: each head's leading qk_rope columns."""

    rotate(x, cos, sin, heads=heads, head_dim=cfg.index_dim, off=0, cols=cfg.qk_rope)
