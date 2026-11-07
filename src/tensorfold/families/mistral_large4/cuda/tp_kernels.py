"""The decode step's small kernels (tp.py): a step is a few rows, so what costs is the launches. Each of
these is one pass over a row (or a row's head) that the torch form in tp.py spends five to fifteen kernels on, same math,
fp32 inside, with reductions in their own order.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from .cp4_kernels import _lat_part

ATTEND_SPLIT = int(os.environ.get("TF_ML4_ATTEND_SPLIT") or 8)            # a row's selection in this many parts, whatever the rows: its bits stay the same
D_BLOCK = 8192                                                 # the next power of two over the model width


@triton.jit
def _add_norm(E, X, NW, XO, N32, N16, R, D: tl.constexpr, W: tl.constexpr, EPS, BLOCK: tl.constexpr, HAS_E: tl.constexpr):
    """x += every rank's partial, in rank order; then RMS norm of the new x, as fp32 and as fp16."""

    r = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, BLOCK)
    ok = i < D
    x = tl.load(X + r * D + i, mask=ok, other=0.0)
    if HAS_E:
        s = tl.load(E + r * D + i, mask=ok, other=0.0).to(tl.float32)
        for k in tl.static_range(1, W):
            s += tl.load(E + (k * R + r) * D + i, mask=ok, other=0.0).to(tl.float32)
        x = x + s
        tl.store(XO + r * D + i, x, mask=ok)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
    y = x * inv * tl.load(NW + i, mask=ok, other=0.0)
    tl.store(N32 + r * D + i, y, mask=ok)
    tl.store(N16 + r * D + i, y.to(tl.float16), mask=ok)


def add_norm(every: torch.Tensor | None, x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple:
    """(x + sum of ``every`` [W, R, d] in rank order, written to x; its RMS norm in fp32; the norm in fp16)."""

    R, d = x.shape
    n32 = torch.empty_like(x)
    n16 = torch.empty((R, d), dtype=torch.float16, device=x.device)
    w = 1 if every is None else every.shape[0]
    _add_norm[(R,)](every if every is not None else x, x, weight, x, n32, n16, R, D=d, W=w, EPS=eps, BLOCK=D_BLOCK,
                    HAS_E=every is not None, num_warps=8)
    return x, n32, n16


@triton.jit
def _own_rows(IDS, TABLE, OUT, first, V, D: tl.constexpr, BLOCK: tl.constexpr):
    """Row r of OUT: token IDS[r]'s row if this rank's part of the table holds it, zeros if not."""

    r = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    at = tl.load(IDS + r) - first
    tl.store(OUT + r * D + i, tl.load(TABLE + at * D + i, mask=(at >= 0) & (at < V), other=0.0))


def own_rows(ids: torch.Tensor, table: torch.Tensor, first: int) -> torch.Tensor:
    d = table.shape[1]
    out = torch.empty((ids.numel(), d), dtype=table.dtype, device=table.device)
    _own_rows[(ids.numel(), d // 1024)](ids, table, out, first, table.shape[0], D=d, BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _rows_of(E, IDS, X, R, V, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    who = tl.load(IDS + r) // V
    tl.store(X + r * D + i, tl.load(E + (who * R + r) * D + i).to(tl.float32))


def rows_of(every: torch.Tensor, ids: torch.Tensor, per_rank: int) -> torch.Tensor:
    """x [R, d] fp32 of ``own_rows``'s parts from every rank, [W, R, d]: each row from the rank that holds its token."""

    _, R, d = every.shape
    x = torch.empty((R, d), dtype=torch.float32, device=every.device)
    _rows_of[(R, d // 1024)](every, ids, x, R, per_rank, D=d, BLOCK=1024, num_warps=4)
    return x


@triton.jit
def _fill_rows(T, DRAFTS, OUT, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    ok = i < n
    t = tl.load(T + i, mask=ok, other=0)
    tl.store(OUT + i, tl.where(t < 0, tl.load(DRAFTS + (-1 - t), mask=ok & (t < 0), other=0), t), mask=ok)


def fill_rows(tokens: torch.Tensor, drafted: torch.Tensor, out: torch.Tensor) -> None:
    """A step's tokens: those that are -1 - a place in ``drafted`` from there, the rest as they are."""

    _fill_rows[(1,)](tokens, drafted, out, tokens.numel(), BLOCK=64, num_warps=1)


@triton.jit
def _layer_norm(X, NW, NB, OUT, D: tl.constexpr, EPS, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    ok = i < D
    x = tl.load(X + r * D + i, mask=ok, other=0.0)
    mean = tl.sum(x, axis=0) / D
    c = tl.where(ok, x - mean, 0.0)
    inv = tl.rsqrt(tl.sum(c * c, axis=0) / D + EPS)
    y = c * inv * tl.load(NW + i, mask=ok, other=0.0) + tl.load(NB + i, mask=ok, other=0.0)
    tl.store(OUT + r * D + i, y.to(tl.float16), mask=ok)


def layer_norm16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    _layer_norm[(x.shape[0],)](x, weight, bias, out, D=x.shape[1], EPS=eps, BLOCK=D_BLOCK, num_warps=8)
    return out


@triton.jit
def _q_rope(Q, POS, S, B, INV, PE, q_stride, p_stride, QK: tl.constexpr, NW: tl.constexpr, RD: tl.constexpr):
    """A row's head: the rope part, with the ssmax scale, rotated in GPT-J pairs, fp16 out."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(POS + r).to(tl.float32)
    sc = tl.log(1.0 + pos) * tl.load(S + h) + tl.load(B + h)
    base = Q + r * q_stride + h * QK + NW
    j = tl.arange(0, RD // 2)
    a = tl.load(base + 2 * j).to(tl.float32) * sc
    b = tl.load(base + 2 * j + 1).to(tl.float32) * sc
    ang = pos * tl.load(INV + j)
    c, s = tl.cos(ang), tl.sin(ang)
    tl.store(PE + r * p_stride + h * RD + 2 * j, (a * c - b * s).to(tl.float16))
    tl.store(PE + r * p_stride + h * RD + 2 * j + 1, (a * s + b * c).to(tl.float16))


def q_rope(q: torch.Tensor, pos: torch.Tensor, s: torch.Tensor, b: torch.Tensor, inv: torch.Tensor, heads: int, qk: int,
           rd: int, pe: torch.Tensor) -> None:
    """q [R, heads * qk] fp16 -> ``pe`` [R, heads * rd] fp16 (rows may be strided)."""

    _q_rope[(q.shape[0], heads)](q, pos, s, b, inv, pe, q.stride(0), pe.stride(0), QK=qk, NW=qk - rd, RD=rd, num_warps=1)


@triton.jit
def _q_absorb(Q, POS, S, B, WUK, OUT, rows, q_stride, o_stride, wh, wn, QK: tl.constexpr, NW: tl.constexpr, C: tl.constexpr,
              BC: tl.constexpr):
    """out[r, h] = fp16(q_nope[r, h] * ssmax) @ W_UK[h]: a head's weight tile stays put while the rows go by."""

    h = tl.program_id(0)
    cc = tl.program_id(1) * BC + tl.arange(0, BC)
    i = tl.arange(0, NW)
    w = tl.load(WUK + h.to(tl.int64) * wh + i[:, None] * wn + cc[None, :]).to(tl.float32)
    s, b = tl.load(S + h), tl.load(B + h)
    for r in range(rows):
        sc = tl.log(1.0 + tl.load(POS + r).to(tl.float32)) * s + b
        q = (tl.load(Q + r * q_stride + h * QK + i).to(tl.float32) * sc).to(tl.float16).to(tl.float32)
        tl.store(OUT + r * o_stride + h * C + cc, tl.sum(q[:, None] * w, axis=0).to(tl.float16))


def q_absorb(q: torch.Tensor, pos: torch.Tensor, s: torch.Tensor, b: torch.Tensor, wuk: torch.Tensor, qk: int,
             out: torch.Tensor) -> None:
    """q [R, h * qk] fp16, wuk [h, nope, C] -> ``out`` [R, h * C] fp16 (rows may be strided)."""

    heads, nope, c = wuk.shape
    _q_absorb[(heads, c // 64)](q, pos, s, b, wuk, out, q.shape[0], q.stride(0), out.stride(0), wuk.stride(0), wuk.stride(1),
                                QK=qk, NW=nope, C=c, BC=64, num_warps=4)


@triton.jit
def _qi_prep(Q, POS, S, B, INV, OUT, o_stride, ID: tl.constexpr, RD: tl.constexpr, HL: tl.constexpr):
    """The indexer's query head: ssmax, then its LEADING RD dims rotated, the rest as they are."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(POS + r).to(tl.float32)
    sc = tl.log(1.0 + pos) * tl.load(S + h) + tl.load(B + h)
    base = Q + (r * HL + h) * ID
    out = OUT + r * o_stride + h * ID
    if RD > 0:
        j = tl.arange(0, RD // 2)
        a = tl.load(base + 2 * j).to(tl.float32) * sc
        b = tl.load(base + 2 * j + 1).to(tl.float32) * sc
        ang = pos * tl.load(INV + j)
        c, s = tl.cos(ang), tl.sin(ang)
        tl.store(out + 2 * j, (a * c - b * s).to(tl.float16))
        tl.store(out + 2 * j + 1, (a * s + b * c).to(tl.float16))
    i = tl.arange(0, ID)
    tl.store(out + i, (tl.load(base + i).to(tl.float32) * sc).to(tl.float16), mask=i >= RD)


def qi_prep(q: torch.Tensor, pos: torch.Tensor, s: torch.Tensor, b: torch.Tensor, inv: torch.Tensor, heads: int, dim: int,
            rd: int, out: torch.Tensor) -> None:
    _qi_prep[(q.shape[0], heads)](q, pos, s, b, inv, out, out.stride(0), ID=dim, RD=rd, HL=heads, num_warps=1)


@triton.jit
def _kv_row(CKV, NW, POS, WR, INV, PLANE, c_stride, row_bytes, EPS, LORA: tl.constexpr, RD: tl.constexpr):
    """One row's new KV: RMS norm of the latent, fp8 by groups of 128 with fp32 scales, k_pe rotated: its pool row, whole."""

    r = tl.program_id(0)
    i = tl.arange(0, LORA)
    x = tl.load(CKV + r * c_stride + i)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / LORA + EPS)
    y = x * inv * tl.load(NW + i)
    g = tl.reshape(y, (4, 128))
    amax = tl.max(tl.abs(g), axis=1)
    scale = tl.maximum(amax / 448.0, 1.1754943508222875e-38)
    q = tl.minimum(tl.maximum(g / scale[:, None], -448.0), 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    row = PLANE + tl.load(WR + r) * row_bytes
    gi = tl.arange(0, 4)
    ci = tl.arange(0, 128)
    tl.store(row + gi[:, None] * 128 + ci[None, :], q)
    tl.store((row + LORA).to(tl.pointer_type(tl.float32)) + gi, scale)
    if RD > 0:
        j = tl.arange(0, RD // 2)
        pos = tl.load(POS + r).to(tl.float32)
        a = tl.load(CKV + r * c_stride + LORA + 2 * j)
        b = tl.load(CKV + r * c_stride + LORA + 2 * j + 1)
        ang = pos * tl.load(INV + j)
        c, s = tl.cos(ang), tl.sin(ang)
        pe = (row + LORA + 16).to(tl.pointer_type(tl.float16))
        tl.store(pe + 2 * j, (a * c - b * s).to(tl.float16))
        tl.store(pe + 2 * j + 1, (a * s + b * c).to(tl.float16))


def kv_row(ckv: torch.Tensor, weight: torch.Tensor, pos: torch.Tensor, wr: torch.Tensor, inv: torch.Tensor,
           plane: torch.Tensor, eps: float, lora: int, rd: int) -> None:
    _kv_row[(ckv.shape[0],)](ckv, weight, pos, wr, inv, plane, ckv.stride(0), plane.stride(0), EPS=eps, LORA=lora, RD=rd,
                             num_warps=4)


@triton.jit
def _key_row(K, NW, NB, POS, WR, INV, PLANE, EPS, ID: tl.constexpr, RD: tl.constexpr):
    """One row's indexer key: biased LayerNorm, leading RD dims rotated, fp16 in the key plane."""

    r = tl.program_id(0)
    i = tl.arange(0, ID)
    x = tl.load(K + r * ID + i)
    mean = tl.sum(x, axis=0) / ID
    c = x - mean
    inv = tl.rsqrt(tl.sum(c * c, axis=0) / ID + EPS)
    row = PLANE + tl.load(WR + r) * ID
    tl.store(row + i, (c * inv * tl.load(NW + i) + tl.load(NB + i)).to(tl.float16), mask=i >= RD)
    if RD > 0:
        j = tl.arange(0, RD // 2)
        a = (tl.load(K + r * ID + 2 * j) - mean) * inv * tl.load(NW + 2 * j) + tl.load(NB + 2 * j)
        b = (tl.load(K + r * ID + 2 * j + 1) - mean) * inv * tl.load(NW + 2 * j + 1) + tl.load(NB + 2 * j + 1)
        ang = tl.load(POS + r).to(tl.float32) * tl.load(INV + j)
        co, si = tl.cos(ang), tl.sin(ang)
        tl.store(row + 2 * j, (a * co - b * si).to(tl.float16))
        tl.store(row + 2 * j + 1, (a * si + b * co).to(tl.float16))


def key_row(k: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, pos: torch.Tensor, wr: torch.Tensor,
            inv: torch.Tensor, plane: torch.Tensor, eps: float, dim: int, rd: int) -> None:
    _key_row[(k.shape[0],)](k, weight, bias, pos, wr, inv, plane, EPS=eps, ID=dim, RD=rd, num_warps=1)


@triton.jit
def _merge_expand(BUF, WUV, OUT, rows, stride, lse_at, wh, wv, HL: tl.constexpr, C: tl.constexpr, V: tl.constexpr,
                  BV: tl.constexpr, W: tl.constexpr):
    """A head from every rank's (normalized partial, log-sum-exp), fp32, added in rank order, as fp16, then its W_UV tile.
    ``BUF`` holds a block for each rank: partials [rows, HL, C] fp16, then the lse values [rows, HL] fp32."""

    h = tl.program_id(0)
    v = tl.program_id(1) * BV + tl.arange(0, BV)
    c = tl.arange(0, C)
    w = tl.load(WUV + h.to(tl.int64) * wh + v[:, None] * wv + c[None, :]).to(tl.float32)
    for r in range(rows):
        at = r * HL + h
        m = tl.load((BUF + lse_at).to(tl.pointer_type(tl.float32)) + at)
        for k in tl.static_range(1, W):
            m = tl.maximum(m, tl.load((BUF + k * stride + lse_at).to(tl.pointer_type(tl.float32)) + at))
        den = 0.0
        acc = tl.zeros((C,), dtype=tl.float32)
        for k in tl.static_range(W):
            lse = tl.load((BUF + k * stride + lse_at).to(tl.pointer_type(tl.float32)) + at)
            a = tl.where(lse > float("-inf"), tl.exp(lse - m), 0.0)
            den += a
            acc += a * tl.load(BUF + k * stride + at * C + c).to(tl.float32)
        o = (acc / den).to(tl.float16).to(tl.float32)
        tl.store(OUT + at * V + v, tl.sum(w * o[None, :], axis=1).to(tl.float16))


def merge_expand(buf: torch.Tensor, wuv: torch.Tensor, rows: int, out: torch.Tensor) -> torch.Tensor:
    """buf [W, stride] fp16 of ``attend_step``'s blocks, wuv [h, V, C] -> out [rows, h * V] fp16."""

    W, stride = buf.shape
    heads, v, c = wuv.shape
    _merge_expand[(heads, v // 32)](buf, wuv, out, rows, stride, rows * heads * c, wuv.stride(0), wuv.stride(1), HL=heads, C=c,
                                    V=v, BV=32, W=W, num_warps=8)
    return out


@triton.jit
def _score_step(EV, Wt, Kp, BASE, VIS, OUT, n, q_w, q_r, q_at, WS, IH: tl.constexpr, IL: tl.constexpr, ID: tl.constexpr,
                BN: tl.constexpr):
    """A decode row's indexer scores over this rank's keys. Its query heads come straight from the gathered buffer, IL heads
    a rank, and the head weights take their constant here; tiles past the row's visible keys read nothing."""

    r = tl.program_id(0)
    t = tl.program_id(1) * BN + tl.arange(0, BN)
    vis = tl.load(VIS + r)
    tok = t < vis
    d = tl.arange(0, ID)
    h = tl.arange(0, IH)
    q = tl.load(EV + (h // IL)[:, None].to(tl.int64) * q_w + r * q_r + q_at + (h % IL)[:, None] * ID + d[None, :])
    at = tl.load(BASE + r) + t.to(tl.int64)
    k = tl.load(Kp + at[:, None] * ID + d[None, :], mask=tok[:, None], other=0.0)
    w = tl.load(Wt + r * IH + h) * WS
    s = tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * w[:, None], axis=0)
    tl.store(OUT + r.to(tl.int64) * n + t, tl.where(tok, s, float("-inf")), mask=t < n)


def score_step(every: torch.Tensor, q_at: int, wts: torch.Tensor, scale: float, il: int, idim: int, keys: torch.Tensor,
               base: torch.Tensor, vis: torch.Tensor, n: int, out: torch.Tensor) -> None:
    """every [W, R, T] fp16 from the query all-gather, its indexer part at element ``q_at`` of a row."""

    W, R, T = every.shape
    _score_step[(R, triton.cdiv(n, 128))](every, wts, keys, base, vis, out, n, R * T, T, q_at, scale, IH=W * il, IL=il, ID=idim,
                                          BN=128, num_warps=4)


@triton.jit
def _attend_part(EV, ROWS, CNT, BASE, PLANE, PO, PM, sm, ks, q_w, q_r, pe_at, row_bytes, S, HB: tl.constexpr, RD: tl.constexpr,
                 BN: tl.constexpr):
    """cp4_kernels._sparse_mla for a decode step: a program's HB heads are one rank's, read from the gathered queries, over its
    share of a row's selection (a decode step has few programs otherwise). PO its sums, PM its (max, sum of weights)."""

    hb = tl.program_id(0)
    r = tl.program_id(1)
    sp = tl.program_id(2)
    c = tl.load(CNT + r)
    per = tl.cdiv(tl.cdiv(c, S), BN) * BN
    lo = sp * per
    hi = tl.minimum(c, lo + per)
    base0 = tl.load(BASE + r)
    h = tl.arange(0, HB)
    cc = tl.arange(0, 128)
    n = tl.arange(0, BN)
    src = EV + hb.to(tl.int64) * q_w + r * q_r
    qb = src + h[:, None] * 512 + cc[None, :]
    q0 = tl.load(qb)
    q1 = tl.load(qb + 128)
    q2 = tl.load(qb + 256)
    q3 = tl.load(qb + 384)
    if RD > 0:
        rr = tl.arange(0, RD)
        qp = tl.load(src + pe_at + h[:, None] * RD + rr[None, :])
    m_i = tl.full((HB,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((HB,), dtype=tl.float32)
    a0 = tl.zeros((HB, 128), dtype=tl.float32)
    a1 = tl.zeros((HB, 128), dtype=tl.float32)
    a2 = tl.zeros((HB, 128), dtype=tl.float32)
    a3 = tl.zeros((HB, 128), dtype=tl.float32)
    for t0 in range(lo, hi, BN):
        i = t0 + n
        ok = i < hi
        row = tl.load(ROWS + r * ks + i, mask=ok, other=0).to(tl.int64) + base0
        base = PLANE + row * row_bytes
        l0 = _lat_part(base, 0, cc, ok)
        l1 = _lat_part(base, 1, cc, ok)
        l2 = _lat_part(base, 2, cc, ok)
        l3 = _lat_part(base, 3, cc, ok)
        s = tl.dot(q0, tl.trans(l0))
        s = tl.dot(q1, tl.trans(l1), s)
        s = tl.dot(q2, tl.trans(l2), s)
        s = tl.dot(q3, tl.trans(l3), s)
        if RD > 0:
            kp = tl.load((base + 528).to(tl.pointer_type(tl.float16))[:, None] + rr[None, :], mask=ok[:, None], other=0.0)
            s = tl.dot(qp, tl.trans(kp), s)
        s = tl.where(ok[None, :], s * sm, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p16 = p.to(tl.float16)
        a0 = tl.dot(p16, l0, a0 * alpha[:, None])
        a1 = tl.dot(p16, l1, a1 * alpha[:, None])
        a2 = tl.dot(p16, l2, a2 * alpha[:, None])
        a3 = tl.dot(p16, l3, a3 * alpha[:, None])
        m_i = m_new
    at = (hb * tl.num_programs(1) + r) * S + sp
    ob = PO + at.to(tl.int64) * (HB * 512) + h[:, None] * 512 + cc[None, :]
    tl.store(ob, a0)
    tl.store(ob + 128, a1)
    tl.store(ob + 256, a2)
    tl.store(ob + 384, a3)
    tl.store(PM + (at * 2) * HB + h, m_i)
    tl.store(PM + (at * 2 + 1) * HB + h, l_i)


@triton.jit
def _attend_join(PO, PM, OUT, stride, lse_at, S: tl.constexpr, HB: tl.constexpr):
    """A head's parts into rank hb's block of the all-to-all: normalized partials, then lse."""

    hb = tl.program_id(0)
    r = tl.program_id(1)
    h = tl.program_id(2)
    first = (hb * tl.num_programs(1) + r) * S
    s = tl.arange(0, S)
    m = tl.load(PM + ((first + s) * 2) * HB + h)
    l = tl.load(PM + ((first + s) * 2 + 1) * HB + h)
    top = tl.max(m, axis=0)
    w = tl.exp(m - top)
    total = tl.sum(l * w, axis=0)
    cc = tl.arange(0, 512)
    acc = tl.sum(tl.load(PO + ((first + s) * HB + h).to(tl.int64)[:, None] * 512 + cc[None, :]) * w[:, None], axis=0)
    inv = tl.where(total > 0.0, 1.0 / total, 0.0)
    dst = OUT + hb.to(tl.int64) * stride
    tl.store(dst + (r * HB + h) * 512 + cc, (acc * inv).to(tl.float16))
    tl.store((dst + lse_at).to(tl.pointer_type(tl.float32)) + r * HB + h, tl.where(total > 0.0, top + tl.log(total), float("-inf")))


def attend_step(every: torch.Tensor, heads: int, rd: int, rows: torch.Tensor, cnt: torch.Tensor, base: torch.Tensor,
                plane: torch.Tensor, sm: float, out: torch.Tensor) -> None:
    """every [W, R, T] fp16 (latent queries of a rank's ``heads``, then their k_pe queries), rows [R, K] int32 local rows,
    plane the layer's uint8 rows. out [W, stride] fp16: rank k's block holds what k merges."""

    W, R, T = every.shape
    S = ATTEND_SPLIT
    po = torch.empty((W * R * S * heads * 512,), dtype=torch.float32, device=every.device)
    pm = torch.empty((W * R * S * 2 * heads,), dtype=torch.float32, device=every.device)
    _attend_part[(W, R, S)](every, rows, cnt, base, plane, po, pm, sm, rows.stride(0), R * T, T, heads * 512, plane.stride(0),
                            S, HB=heads, RD=rd, BN=32, num_warps=4, num_stages=1)
    _attend_join[(W, R, heads)](po, pm, out, out.shape[1], R * heads * 512, S=S, HB=heads, num_warps=4)


@triton.jit
def _route(LG, LUT, PICK, WTS, E: tl.constexpr, TOPK: tl.constexpr, SKIP):
    """Softmax over the routed logits, the top ``TOPK``, renormalized; a pick of another rank's expert becomes SKIP, weight 0."""

    r = tl.program_id(0)
    i = tl.arange(0, E)
    x = tl.load(LG + r * E + i)
    p = tl.exp(x - tl.max(x, axis=0))
    p = p / tl.sum(p, axis=0)
    total = 0.0
    q = p
    for s in tl.static_range(TOPK):
        best = tl.max(q, axis=0)
        total += best
        q = tl.where(i == tl.min(tl.where(q == best, i, E), axis=0), -1.0, q)
    q = p
    for s in tl.static_range(TOPK):
        best = tl.max(q, axis=0)
        at = tl.min(tl.where(q == best, i, E), axis=0)
        q = tl.where(i == at, -1.0, q)
        mine = tl.load(LUT + at)
        tl.store(PICK + r * TOPK + s, mine)
        tl.store(WTS + r * TOPK + s, tl.where(mine < SKIP, best / total, 0.0))


def route(logits: torch.Tensor, lut: torch.Tensor, k: int, skip: int) -> tuple:
    R, e = logits.shape
    pick = torch.empty((R, k), dtype=torch.int32, device=logits.device)
    wts = torch.empty((R, k), dtype=torch.float32, device=logits.device)
    _route[(R,)](logits, lut, pick, wts, E=e, TOPK=k, SKIP=skip, num_warps=1)
    return pick, wts


@triton.jit
def _act(G, U, OUT, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < N
    g = tl.load(G + i, mask=ok, other=0.0)
    u = tl.load(U + i, mask=ok, other=0.0)
    tl.store(OUT + i, (g / (1.0 + tl.exp(-g)) * u).to(tl.float16), mask=ok)


def silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """silu(g) * u, fp32 in, fp16 out."""

    out = torch.empty(g.shape, dtype=torch.float16, device=g.device)
    n = g.numel()
    _act[(triton.cdiv(n, 1024),)](g, u, out, n, BLOCK=1024, num_warps=4)
    return out


@lru_cache(maxsize=1)
def _select():
    from tensorfold.cuda.build import load

    return load(name="tensorfold_ml4_select_v4", sources=[str(Path(__file__).parent / "select_step.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def select_ready() -> None:
    """Build the selection's kernels now: a capture cannot wait on nvcc."""

    _select()


def local_top(score: torch.Tensor, vis: torch.Tensor, k: int, stripe, rank: int, keys: torch.Tensor | None = None,
              rows: torch.Tensor | None = None) -> tuple:
    """score [R, n], ``vis`` a row's visible scores: this rank's best k as (int64 keys [R, k], int32 rows [R, k]), in row order,
    PAD_KEY and NO_ROW past the end. The keys are cp4.score_keys'."""

    keys = torch.empty((score.shape[0], k), dtype=torch.int64, device=score.device) if keys is None else keys
    rows = torch.empty((score.shape[0], k), dtype=torch.int32, device=score.device) if rows is None else rows
    _select().local_top(score, vis, k, stripe.world, stripe.block, rank, keys, rows)
    return keys, rows


def top8(score: torch.Tensor, largest: torch.Tensor, vis: torch.Tensor, k: int, stripe, rank: int, keys: torch.Tensor,
         rows: torch.Tensor, slow: torch.Tensor | None = None) -> None:
    """local_top of a long prompt's rows, scores [R, >= n] and the largest of every eight [R, >= n / 8] from score8. ``slow`` [1]
    int32 counts the rows it read through."""

    _select().top8(score, largest, vis, k, stripe.world, stripe.block, rank, keys, rows, slow)


def pick_rows(every: torch.Tensor, rows: torch.Tensor, rank: int) -> tuple:
    """every [W, R, k] the gathered keys, rows this rank's: cp4.filter_selection of the k-th best key, (rows, count)."""

    out = torch.empty_like(rows)
    count = torch.empty((rows.shape[0],), dtype=torch.int32, device=rows.device)
    _select().pick_rows(every, rows, rank, out, count)
    return out, count


def kth_rows(every: torch.Tensor) -> torch.Tensor:
    """every [W, R, k] keys of R rows from every rank: a key per row that the same keys pass as cp4.kth_key's."""

    out = torch.empty((every.shape[1],), dtype=torch.int64, device=every.device)
    _select().kth_rows(every, out)
    return out


def keep_rows(keys: torch.Tensor, rows: torch.Tensor, thresh: torch.Tensor, base: int) -> tuple:
    """cp4.filter_selection, the rows offset by ``base``: (rows [R, k] ascending, NO_ROW padded; count [R])."""

    out = torch.empty_like(rows)
    count = torch.empty((rows.shape[0],), dtype=torch.int32, device=rows.device)
    _select().keep_rows(keys, rows, thresh, base, out, count)
    return out, count


# -- a round's rows (cp4_fused.FusedCp) ---------------------------------------------------------------------------------------

@triton.jit
def _q_scale(Q, POS, S, B, OUT, q_stride, o_head, m, H: tl.constexpr, QK: tl.constexpr, NW: tl.constexpr):
    """A round's query: the ssmax scale on the part the latents absorb, heads leading, for the batched matmul."""

    r = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, H)
    i = tl.arange(0, NW)
    sc = tl.log(1.0 + tl.load(POS + r).to(tl.float32)) * tl.load(S + h) + tl.load(B + h)
    q = tl.load(Q + r * q_stride + h[:, None] * QK + i[None, :]).to(tl.float32) * sc[:, None]
    tl.store(OUT + h[:, None].to(tl.int64) * o_head + r * NW + i[None, :], q.to(tl.float16))


def q_scale(q: torch.Tensor, pos: torch.Tensor, s: torch.Tensor, b: torch.Tensor, qk: int, out: torch.Tensor) -> None:
    """q [m, H * qk] fp16 -> ``out`` [H, m, nope] fp16."""

    heads, m, nope = out.shape
    _q_scale[(m,)](q, pos, s, b, out, q.stride(0), out.stride(0), m, H=heads, QK=qk, NW=nope, num_warps=4)


@triton.jit
def _merge_rows(O, L, OUT, n, H: tl.constexpr, V: tl.constexpr, W: tl.constexpr):
    """cp4.merge_partials of one row of a round: every rank's normalized part and log-sum-exp, fp32, added in rank order."""

    r = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, H)
    i = tl.arange(0, V)
    m = tl.load(L + r * H + h)
    for k in tl.static_range(1, W):
        m = tl.maximum(m, tl.load(L + (k * n + r) * H + h))
    den = tl.zeros((H,), dtype=tl.float32)
    acc = tl.zeros((H, V), dtype=tl.float32)
    for k in tl.static_range(W):
        lse = tl.load(L + (k * n + r) * H + h)
        a = tl.where(lse > float("-inf"), tl.exp(lse - m), 0.0)
        den += a
        acc += a[:, None] * tl.load(O + ((k * n + r) * H + h[:, None]) * V + i[None, :]).to(tl.float32)
    tl.store(OUT + (r * H + h[:, None]) * V + i[None, :], (acc / den[:, None]).to(tl.float16))


def merge_rows(o: torch.Tensor, lse: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """o [W, n, H, V] fp16, lse [W, n, H] fp32 -> [n, H * V] fp16."""

    W, n, H, V = o.shape
    out = torch.empty((n, H * V), dtype=torch.float16, device=o.device) if out is None else out
    _merge_rows[(n,)](o, lse, out, n, H=H, V=V, W=W, num_warps=8)
    return out
