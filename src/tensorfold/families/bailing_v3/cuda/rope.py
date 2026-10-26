"""Ling-3.1-flash's interleaved RoPE with the reference's roundings (modeling 248-281, 541-576).

The reference builds ``inv_freq = 1 / theta ** (arange(0, 64, 2) / 64)`` in fp32 on the CPU at load, takes
``phase = position * inv_freq`` in fp32 (a one-term matmul: the rounded product), and hands the attention
``cos(phase)`` and ``sin(phase)`` rounded to bf16. It de-interleaves the rope slice (pair i's two values to
columns i and i + 32) and rotates in bf16 ops: ``bf16(bf16(a c) - bf16(b s))`` and ``bf16(bf16(b c) + bf16(a s))``
for the pair (a, b) with c, s the bf16 cos and sin of angle i.

Here the table is made with the same torch ops on the device (``table``), and ``apply`` rotates each pair in
place in that arithmetic. The pairs stay interleaved: the reference's layout is a fixed permutation of the
rope columns applied to queries and keys alike, so every q . k score is a sum of the same products (test 4).
glm_moe_dsa's rope (GLM-5.3) computes its angles inside its kernel (exp2 of a log) and rotates in fp32 with one
rounding; at long contexts its inverse frequencies' last bits move the phase, so Ling uses this one.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def inv_freq(theta: float, dim: int) -> torch.Tensor:
    """The reference's inverse frequencies (transformers' default rope init): fp32 [dim / 2], made on the CPU."""

    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))


class Table:
    """Static cos / sin rows (bf16 [rows, dim / 2]) and their scratch, for positions given on the device."""

    def __init__(self, rows: int, theta: float, dim: int, device) -> None:
        half = dim // 2
        self.inv = inv_freq(theta, dim).to(device)
        self.step = torch.arange(rows, dtype=torch.int32, device=device)
        self.pos = torch.empty((rows,), dtype=torch.int32, device=device)
        self.phase = torch.empty((rows, half), dtype=torch.float32, device=device)
        self.f32 = torch.empty((rows, half), dtype=torch.float32, device=device)
        self.cos = torch.empty((rows, half), dtype=torch.bfloat16, device=device)
        self.sin = torch.empty((rows, half), dtype=torch.bfloat16, device=device)

    def fill(self, pos_dev: torch.Tensor, R: int) -> None:
        """Rows 0 .. R - 1 for positions pos_dev[0] + r (capturable: the position is read on the device)."""

        torch.add(self.step[:R], pos_dev, out=self.pos[:R])
        self.phase[:R].copy_(self.pos[:R, None].expand(R, self.inv.shape[0]))          # int -> fp32: exact
        self.phase[:R].mul_(self.inv)                                                    # position * inv, rounded
        torch.cos(self.phase[:R], out=self.f32[:R])
        self.cos[:R].copy_(self.f32[:R])                                                 # the reference's .to(bf16)
        torch.sin(self.phase[:R], out=self.f32[:R])
        self.sin[:R].copy_(self.f32[:R])


@triton.jit
def _rotate(X, COS, SIN, row_stride, head_stride, off, HALF: tl.constexpr, PER: tl.constexpr):
    """Program (row r, head j): the pairs (off + 2i, off + 2i + 1) of head j, rotated by row r's angle i."""

    r = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    i = tl.arange(0, PER)
    ok = i < HALF
    base = X + r * row_stride + j * head_stride + off
    c = tl.load(COS + r * HALF + i, mask=ok, other=1.0).to(tl.float32)
    s = tl.load(SIN + r * HALF + i, mask=ok, other=0.0).to(tl.float32)
    a = tl.load(base + 2 * i, mask=ok, other=0.0).to(tl.float32)
    b = tl.load(base + 2 * i + 1, mask=ok, other=0.0).to(tl.float32)
    ac = (a * c).to(tl.bfloat16).to(tl.float32)            # each bf16 product exact in fp32, then rounded
    bs = (b * s).to(tl.bfloat16).to(tl.float32)
    bc = (b * c).to(tl.bfloat16).to(tl.float32)
    as_ = (a * s).to(tl.bfloat16).to(tl.float32)
    tl.store(base + 2 * i, (ac - bs).to(tl.bfloat16), mask=ok)
    tl.store(base + 2 * i + 1, (bc + as_).to(tl.bfloat16), mask=ok)


def rotate(x: torch.Tensor, t: Table, R: int, *, heads: int, head_dim: int, off: int, cols: int) -> torch.Tensor:
    """In place: rows [R, >= heads * head_dim] bf16 (any row stride), each head's columns off .. off + cols."""

    if cols == 0 or heads == 0 or R == 0:
        return x
    if x.dim() != 2 or x.stride(-1) != 1 or x.dtype != torch.bfloat16:
        raise ValueError("rope: bf16 rows with unit column stride")
    if x.shape[0] < R or x.shape[1] < heads * head_dim or off + cols > head_dim or cols % 2 \
            or cols // 2 != t.cos.shape[1]:
        raise ValueError(f"rope: {heads} heads of {head_dim} (cols {off}..{off + cols}) do not fit {tuple(x.shape)}")
    half = cols // 2
    # enable_fp_fusion=False: Triton on sm_121 otherwise contracts bf16(a c) - bf16(b s) into one fma.rn.bf16 that
    # drops the a c rounding; the reference rounds every product (the KDA ext's --fmad=false, here)
    _rotate[(R, heads)](x, t.cos, t.sin, x.stride(0), head_dim, off, HALF=half, PER=triton.next_power_of_2(half),
                        num_warps=1, enable_fp_fusion=False)
    return x


def apply_q(q: torch.Tensor, t: Table, R: int, cfg, heads: int) -> None:
    """Query rows [R, heads * qk_dim] (after q_b): each head's last qk_rope columns."""

    rotate(q, t, R, heads=heads, head_dim=cfg.qk_dim, off=cfg.qk_nope, cols=cfg.qk_rope)


def apply_key(rope_key: torch.Tensor, t: Table, R: int, cfg) -> None:
    """The shared rope key rows [R, qk_rope] (a strided view of kv_a's output)."""

    rotate(rope_key, t, R, heads=1, head_dim=cfg.qk_rope, off=0, cols=cfg.qk_rope)
