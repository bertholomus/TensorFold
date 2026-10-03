"""Concurrent decode rounds (``--parallel``): the kernels that read or write a cache, each row at its own stream's
position and cache base, read from device tables.

A round verifies every live stream's window in one forward over a pool of cache slots: row r belongs to a stream
whose token 0 sits at pool row ``base[r]`` and is at position ``pos[r]`` of that stream. Each kernel here is its
single-stream twin (rope._table, latent._lat_write, kv8._write, kvq._write, mla_pe._dense_chunks_pe and
_sparse_chunks_pe, select._scores) with only the addressing changed: a row's key tiles, chunks and merge order are
counted from its own stream's token 0, so a row gets the bits the single-stream kernels give it at that position
(tests/cuda/test_glm_rows.py). Prompts never come here: a stream's prompt fills its slot through the single-stream
path on the slot's views.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda.latent import CHUNK, HB, KT, _merge

from . import kv8 as kv8_mod
from .kv8 import Kv8
from .kvq import KvQ, _put_plane, rotate as _q_rotate, tile as _q_tile
from .mla_pe import _keys


class Tables:
    """A round's per-row tables on the device, with pinned host mirrors: ``pos`` int32 [rows] (each row's position in
    its stream), ``base`` int64 [rows] (its stream's first pool row), and ``seg`` int32 [streams, 2] (each stream's
    first row in the window and its row count; unused entries 0 rows). ``dense`` / ``sparse``: whether some row sits
    below / at or past the dense limit (the kernels a round runs); ``bucket``: the scored-token bucket of its deepest
    row past the limit."""

    def __init__(self, rows: int, streams: int, device) -> None:
        self.rows, self.streams = rows, streams
        self.pos = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.base = torch.zeros((rows,), dtype=torch.int64, device=device)
        self.seg = torch.zeros((streams, 2), dtype=torch.int32, device=device)
        pin = device is not None and torch.device(device).type == "cuda"
        self.pos_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=pin)
        self.base_host = torch.zeros((rows,), dtype=torch.int64, pin_memory=pin)
        self.seg_host = torch.zeros((streams, 2), dtype=torch.int32, pin_memory=pin)
        self.dense = self.sparse = False
        self.bucket = 0
        self.R = 0

    def fill(self, windows: list[tuple[int, int, int]], dense_limit: int) -> int:
        """``windows``: each stream's (pool base, first position, rows), in window order. Returns the rows."""

        from .select import sparse_bucket

        if len(windows) > self.streams:
            raise ValueError(f"{len(windows)} streams in a round, tables for {self.streams}")
        pos, base, seg = self.pos_host.numpy(), self.base_host.numpy(), self.seg_host.numpy()
        seg[:] = 0
        r = 0
        deepest = -1
        self.dense = self.sparse = False
        for i, (b0, p0, n) in enumerate(windows):
            if r + n > self.rows:
                raise ValueError(f"a round of more than {self.rows} rows")
            pos[r:r + n] = range(p0, p0 + n)
            base[r:r + n] = b0
            seg[i] = (r, n)
            if p0 < dense_limit:
                self.dense = True
            if p0 + n - 1 >= dense_limit:
                self.sparse = True
                deepest = max(deepest, p0 + n - 1)
            r += n
        self.R = r
        self.bucket = sparse_bucket(deepest, 1) if self.sparse else 0
        self.pos[:r].copy_(self.pos_host[:r], non_blocking=True)
        self.base[:r].copy_(self.base_host[:r], non_blocking=True)
        self.seg.copy_(self.seg_host, non_blocking=True)
        return r

    def key(self) -> tuple[int, bool, bool, int]:
        """What a captured round's kernels depend on: rows, dense / sparse passes, bucket."""

        return self.R, self.dense, self.sparse, self.bucket


# -- rope ------------------------------------------------------------------------------------------------------------
@triton.jit
def _rope_rows(COS, SIN, POS, THETA: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    """rope._table with row r at POS[r]."""

    r = tl.program_id(0)
    P = tl.load(POS + r).to(tl.float32)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = i < HALF
    inv = tl.exp2(-(2.0 * i.to(tl.float32) / (2 * HALF)) * tl.log2(THETA))
    phase = P * inv
    tl.store(COS + r * HALF + i, tl.cos(phase), mask=ok)
    tl.store(SIN + r * HALF + i, tl.sin(phase), mask=ok)


def rope(cos: torch.Tensor, sin: torch.Tensor, t: Tables, R: int, theta: float, rope_dim: int) -> None:
    half = rope_dim // 2
    if half:
        _rope_rows[(R, triton.cdiv(half, 128))](cos, sin, t.pos, THETA=theta, HALF=half, BLOCK=128, num_warps=4)


# -- writers ---------------------------------------------------------------------------------------------------------
@triton.jit
def _write_bf16(X, x_stride, C, POS, BASE, LW: tl.constexpr):
    """latent._lat_write with row r at pool row BASE[r] + POS[r]."""

    r = tl.program_id(0)
    at = tl.load(BASE + r) + tl.load(POS + r).to(tl.int64)
    k = tl.arange(0, LW)
    tl.store(C + at * LW + k, tl.load(X + r * x_stride + k))


@triton.jit
def _write_kv8(X, x_stride, C, S, POS, BASE, W: tl.constexpr, SHIFT: tl.constexpr):
    """kv8._write with row r at pool row BASE[r] + POS[r]."""

    r = tl.program_id(0)
    at = tl.load(BASE + r) + tl.load(POS + r).to(tl.int64)
    k = tl.arange(0, W)
    x = tl.load(X + r * x_stride + k).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    bits = amax.to(tl.int32, bitcast=True)
    expo = (bits >> 23) & 0xFF
    up = ((bits & 0x7FFFFF) != 0).to(tl.int32)
    e = tl.where(expo == 0, tl.where(bits == 0, 0, -126), tl.maximum(expo - 127 + up - SHIFT, -126))
    scale = ((e + 127) << 23).to(tl.float32, bitcast=True)
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    codes = (x * inv).to(tl.float8e4nv)
    tl.store(C + at * W + k, codes.to(tl.uint8, bitcast=True))
    tl.store(S + at, scale)


@triton.jit
def _write_q(X, x_stride, QW, QS, H, POS, BASE, LW: tl.constexpr, QB: tl.constexpr):
    """kvq._write with row r at pool row BASE[r] + POS[r]."""

    r = tl.program_id(0)
    at = tl.load(BASE + r) + tl.load(POS + r).to(tl.int64)
    G: tl.constexpr = LW // 32
    x = tl.load(X + r * x_stride + tl.arange(0, LW)).to(tl.float32)
    h = tl.load(H + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :])
    if G >= 16:
        v = tl.dot(tl.reshape(x, (G, 32)), h, input_precision="ieee") * 0.17677669529663688110
    else:
        v = tl.sum(tl.reshape(x, (G, 32))[:, :, None] * h[None, :, :], axis=1) * 0.17677669529663688110
    s16 = (tl.max(tl.abs(v), 1) + 1e-10).to(tl.float16)
    inv = 1.0 / s16.to(tl.float32)
    MF: tl.constexpr = 1 << (QB - 1)
    q = tl.floor(v * inv[:, None] * MF + MF)
    q = tl.minimum(tl.maximum(q, 0.0), (1 << QB) - 1).to(tl.uint32)
    base = at * (G * QB)
    if QB & 8:
        _put_plane(q, QW, base, 0, QB - 8, 8, G, QB)
    if QB & 4:
        _put_plane(q, QW, base, QB & 8, QB & 3, 4, G, QB)
    if QB & 2:
        _put_plane(q, QW, base, QB & 12, QB & 1, 2, G, QB)
    if QB & 1:
        _put_plane(q, QW, base, QB & 14, 0, 1, G, QB)
    tl.store(QS + at * G + tl.arange(0, G), s16)


def write(rows: torch.Tensor, cache, t: Tables) -> None:
    """Window rows [R, width] (unit column stride) into a pool plane at each row's BASE + POS."""

    R, W = rows.shape
    if rows.stride(1) != 1:
        raise ValueError("rows.write: rows must have unit column stride")
    if isinstance(cache, Kv8):
        _write_kv8[(R,)](rows, rows.stride(0), cache.codes, cache.scales, t.pos, t.base, W=W, SHIFT=kv8_mod.SHIFT,
                         num_warps=4)
    elif isinstance(cache, KvQ):
        _write_q[(R,)](rows, rows.stride(0), cache.codes, cache.scales, cache.h, t.pos, t.base, LW=W, QB=cache.bits,
                       num_warps=4)
    else:
        _write_bf16[(R,)](rows, rows.stride(0), cache, t.pos, t.base, LW=W, num_warps=4)


# -- attention -------------------------------------------------------------------------------------------------------
@triton.jit
def _dense_rows(QA, QP, LC, LS, HQ, PC, POS, BASE, PO, PM, PL, R, DL, H: tl.constexpr, LW: tl.constexpr,
                PW: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr,
                KV8: tl.constexpr, QB: tl.constexpr):
    """mla_pe._dense_chunks_pe with row r at POS[r] over its stream's keys from BASE[r]; rows at or past DL (the dense
    limit: their sparse pass overwrites them) attend nothing here."""

    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    limit = tl.load(POS + r)
    B = tl.load(BASE + r)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    kq = tl.arange(0, PW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    start = c * CH
    if (start <= limit) & (limit < DL):
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        qp = tl.load(QP + (r * H + hh[:, None]) * PW + kq[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        qr = q
        if QB > 0:
            qr = _q_rotate(q.to(tl.float32), HQ, HBT, LW).to(tl.float16)
        for t in range(tl.minimum(CH // KTT, (limit - start) // KTT + 1)):
            ki = start + t * KTT + tl.arange(0, KTT)
            ok = ki <= limit
            m, l, o = _keys(q, qr, qp, LC, LS, PC, B + ki.to(tl.int64), ok, k, kq, m, l, o, LW, PW, SCALE, KTT, KV8,
                            QB)
        if QB > 0:
            o = _q_rotate(o, HQ, HBT, LW)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _sparse_rows(QA, QP, LC, LS, HQ, PC, TOK, CNT, BASE, PO, PM, PL, R, W: tl.constexpr, H: tl.constexpr,
                 LW: tl.constexpr, PW: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr,
                 KTT: tl.constexpr, KV8: tl.constexpr, QB: tl.constexpr):
    """mla_pe._sparse_chunks_pe with row r's selected tokens (its stream's positions) read from BASE[r]."""

    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(CNT + r)
    B = tl.load(BASE + r)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    kq = tl.arange(0, PW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    if c * CH < n:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        qp = tl.load(QP + (r * H + hh[:, None]) * PW + kq[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        qr = q
        if QB > 0:
            qr = _q_rotate(q.to(tl.float32), HQ, HBT, LW).to(tl.float16)
        for t in range(CH // KTT):
            idx = c * CH + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            tok = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
            m, l, o = _keys(q, qr, qp, LC, LS, PC, B + tok, ok, k, kq, m, l, o, LW, PW, SCALE, KTT, KV8, QB)
        if QB > 0:
            o = _q_rotate(o, HQ, HBT, LW)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


def dense_attention(qa: torch.Tensor, qp: torch.Tensor, cache, pcache: torch.Tensor, t: Tables, s, *, scale: float,
                    dense_limit: int, out: torch.Tensor) -> torch.Tensor:
    """mla_pe.attention for the round's rows below the dense limit (s.nch chunks: the dense window's)."""

    from .mla_pe import _planes

    R, H, LW = qa.shape
    PW = qp.shape[2]
    nch = s.nch
    if R > s.part_rows or LW != s.lw:
        raise ValueError(f"rows.dense_attention: {R} rows, width {LW} past the scratch's {s.part_rows}, {s.lw}")
    n = nch * R * H
    rows, scales, h32, kv8, qb = _planes(cache)
    _dense_rows[(R, triton.cdiv(H, HB), nch)](qa, qp, rows, scales, h32, pcache, t.pos, t.base, s.po[:n * LW],
                                              s.pm[:n], s.pl[:n], R, dense_limit, H=H, LW=LW, PW=PW, CH=CHUNK,
                                              SCALE=scale, HBT=HB, KTT=KT, KV8=kv8, QB=qb, num_warps=4, num_stages=3)
    _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    return out


def sparse_attention(qa: torch.Tensor, qp: torch.Tensor, cache, pcache: torch.Tensor, tokens: torch.Tensor,
                     counts: torch.Tensor, t: Tables, out: torch.Tensor, scale: float) -> None:
    """mla_pe.sparse_attention's decode-window path (a program a chunk, then _merge) over each row's stream."""

    from .mla_pe import _planes

    R, H, LW = qa.shape
    PW = qp.shape[2]
    W = tokens.shape[1]
    rows, scales, h32, kv8, qb = _planes(cache)
    nch = triton.cdiv(W, CHUNK)
    n = nch * R * H
    po = torch.empty((n * LW,), dtype=torch.float32, device=qa.device)
    pm = torch.empty((n,), dtype=torch.float32, device=qa.device)
    pl = torch.empty((n,), dtype=torch.float32, device=qa.device)
    _sparse_rows[(R, triton.cdiv(H, HB), nch)](qa, qp, rows, scales, h32, pcache, tokens, counts, t.base, po, pm, pl,
                                               R, W=W, H=H, LW=LW, PW=PW, CH=CHUNK, SCALE=scale, HBT=HB, KTT=KT,
                                               KV8=kv8, QB=qb, num_warps=4, num_stages=3)
    _merge[(R, H)](po, pm, pl, out, counts, R, H=H, LW=LW, NCH=nch, SPARSE=True, num_warps=4)


# -- selection -------------------------------------------------------------------------------------------------------
@triton.jit
def _scores_rows(QI, Wt, w_stride, IK, IS, HQ, OUT, POS, BASE, SEG, NT, scale, wscale, H: tl.constexpr,
                 HP: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, RB: tl.constexpr, RBLK: tl.constexpr,
                 KV8: tl.constexpr, QB: tl.constexpr):
    """select._scores over a round's streams: program (stream s's row block, token block) scores up to RB of stream
    s's rows (SEG[s] = its first row and row count) against one tile of its keys, as select._scores scores a row
    block of one stream (each key tile read once for the block; a row's scores are its own)."""

    pid = tl.program_id(0)
    s = pid // RBLK
    blk = pid % RBLK
    lo = tl.load(SEG + 2 * s)
    n = tl.load(SEG + 2 * s + 1)
    r0 = lo + blk * RB
    nr = tl.minimum(n - blk * RB, RB)
    if nr > 0:
        tb = tl.program_id(1)
        B = tl.load(BASE + r0)
        top = tl.load(POS + r0 + nr - 1) + 1                 # the block's last row's bound: past it every score is -inf
        t = tb * BT + tl.arange(0, BT)
        if tb * BT < top:
            d = tl.arange(0, D)
            hh = tl.arange(0, HP)
            hok = hh < H
            kt = (B + t.to(tl.int64))
            if KV8:
                k = tl.load(IK + kt[:, None] * D + d[None, :], mask=(t < top)[:, None],
                            other=0).to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
                ks = tl.load(IS + kt, mask=t < top, other=0.0)
            elif QB > 0:
                k = _q_tile(IK, IS, kt, t < top, D, QB, BT)
            else:
                k = tl.load(IK + kt[:, None] * D + d[None, :], mask=(t < top)[:, None], other=0.0).to(tl.bfloat16)
            for i in tl.static_range(RB):
                if i < nr:
                    r = r0 + i
                    bound = tl.load(POS + r) + 1
                    q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None],
                                other=0.0).to(tl.bfloat16)
                    if QB > 0:
                        q = _q_rotate(q.to(tl.float32), HQ, HP, D).to(tl.float16)
                    kr = tl.where((t < bound)[:, None], k, 0.0)
                    dots = tl.dot(q, tl.trans(kr))
                    if KV8:
                        dots = dots * ks[None, :]
                    w = tl.load(Wt + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
                    sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
                    sc = tl.where(t < bound, sc, float("-inf"))
                    tl.store(OUT + r * NT + t, sc, mask=t < NT)
        else:
            for i in tl.static_range(RB):
                if i < nr:
                    tl.store(OUT + (r0 + i) * NT + t, tl.full((BT,), float("-inf"), tl.float32), mask=t < NT)


def select(qi: torch.Tensor, wts: torch.Tensor, keys, R: int, topk: int, t: Tables, *, tokens: torch.Tensor,
           counts: torch.Tensor, max_rows: int) -> None:
    """select.select_tokens for a round's rows (each over its own stream's keys) at the round's bucket; ``max_rows``:
    the most rows a stream's window holds."""

    from . import select as sel

    if qi.stride(0) != qi.shape[1] or wts.stride(1) != 1:
        raise ValueError("rows.select: index queries must be contiguous rows, weights unit-stride columns")
    H = wts.shape[1]
    D = qi.shape[1] // H
    np_max = t.bucket
    scores = torch.empty((R, np_max), dtype=torch.float32, device=qi.device)
    kv8, qb = isinstance(keys, Kv8), keys.bits if isinstance(keys, KvQ) else 0
    ik, isc = (keys.codes, keys.scales) if kv8 or qb else (keys, keys)
    hq = keys.h if qb else isc
    rb = min(16, max_rows)
    rblk = triton.cdiv(max_rows, rb)
    _scores_rows[(t.streams * rblk, triton.cdiv(np_max, 64))](qi, wts, wts.stride(0), ik, isc, hq, scores, t.pos,
                                                              t.base, t.seg, np_max, D ** -0.5, H ** -0.5, H=H,
                                                              HP=max(16, triton.next_power_of_2(H)), D=D, BT=64,
                                                              RB=rb, RBLK=rblk, KV8=kv8, QB=qb, num_warps=4)
    width = tokens.shape[1]
    k = min(topk, np_max)
    if R >= sel.RADIX_ROWS and tokens.stride(1) == 1:
        tokens.zero_()
        sel._radix_topk[(R,)](scores, scores.stride(0), tokens, tokens.stride(0), np_max, K=k, BLOCK=1024,
                              num_warps=4)
    else:
        tokens.zero_()
        sel._split_topk(scores, tokens, k, np_max)
    del scores
    if width > k:
        tokens[:, k:] = -1
    q = t.pos[:R].to(torch.int64)
    counts.copy_(torch.where(q + 1 > topk, tokens.ne(-1).sum(1).clamp(max=width), torch.zeros_like(counts)))
