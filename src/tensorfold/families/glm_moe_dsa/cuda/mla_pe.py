"""GLM-5.3's MLA scores with the rope term: q_lat . c_kv + q_pe . k_pe over the latent cache plus a rope-key cache.

Flash Next (GLM-5.3-Flash) has qk_rope = 0, so its latent kernels (``glm5_next.cuda.latent``) score with the latent
alone. Full GLM-5.3 keeps a 64-wide shared rope key per token: these kernels add its dot product to every score.
Values stay the latent (the rope key is never a value), so the chunk partials and their merge are Flash's
(``latent._merge``, same layout): only the scoring differs.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda.latent import CHUNK, HB, HB_WIDE, KT, LatentScratch, _merge, head_block

from .kv8 import Kv8
from .kvq import KvQ, rotate as _q_rotate, tile as _q_tile

# sparse attention of windows this wide or wider (prompt chunks) runs _sparse_rows_pe, partials merged in registers;
# decode windows keep a program a chunk (more programs a row) and _merge (TF_GLM_SPARSE_FUSED=0: every window does)
FUSED_ROWS = 64 if os.environ.get("TF_GLM_SPARSE_FUSED", "1") != "0" else 1 << 30
FUSED_LAUNCH = (4, 3)             # _sparse_rows_pe's warps and load stages


@triton.jit
def _tile_pe(q, qp, kv, kp, m, l, o, valid, SCALE: tl.constexpr):
    """One key tile: scores = (q . kv + qp . kp) * scale; values are the latent kv."""

    scores = (tl.dot(q, tl.trans(kv)) + tl.dot(qp, tl.trans(kp))).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _tile_peq(qr, qp, kv, kp, m, l, o, valid, SCALE: tl.constexpr):
    """_tile_pe over a quantized latent tile (kvq): qr the H32-rotated query and kv the rotated values, both fp16, so
    q . k is the latent's score and o sums rotated values (rotated back once a chunk)."""

    scores = (tl.dot(qr, tl.trans(kv)) + tl.dot(qp, tl.trans(kp))).to(tl.float32) * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.float16), kv)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _keys(q, qr, qp, LC, LS, PC, rows, ok, k, kq, m, l, o, LW: tl.constexpr, PW: tl.constexpr, SCALE: tl.constexpr,
          KTT: tl.constexpr, KV8: tl.constexpr, QB: tl.constexpr):
    """One tile of key rows ``rows`` [KTT] (int64) folded into (m, l, o): bf16 latent rows; FP8 rows dequantized to
    bf16 in registers (exact: an e4m3 code times a power of two) through the same _tile_pe; or quantized rows (QB
    bits) through _tile_peq against the rotated query qr."""

    kp = tl.load(PC + rows[:, None] * PW + kq[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
    if KV8:
        kc = tl.load(LC + rows[:, None] * LW + k[None, :], mask=ok[:, None], other=0)
        ks = tl.load(LS + rows, mask=ok, other=0.0)
        kv = (kc.to(tl.float8e4nv, bitcast=True).to(tl.float32) * ks[:, None]).to(tl.bfloat16)
        m, l, o = _tile_pe(q, qp, kv, kp, m, l, o, ok, SCALE)
    elif QB > 0:
        kv = _q_tile(LC, LS, rows, ok, LW, QB, KTT)
        m, l, o = _tile_peq(qr, qp, kv, kp, m, l, o, ok, SCALE)
    else:
        kv = tl.load(LC + rows[:, None] * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
        m, l, o = _tile_pe(q, qp, kv, kp, m, l, o, ok, SCALE)
    return m, l, o


@triton.jit
def _dense_chunks_pe(QA, QP, LC, LS, HQ, PC, POS, PO, PM, PL, R, H: tl.constexpr, LW: tl.constexpr,
                     PW: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr,
                     KV8: tl.constexpr, QB: tl.constexpr):
    """Program (row, head block, chunk): causal attention of HB heads of row r over keys [c CH, (c + 1) CH).

    The latent rows are bf16 (LC), FP8 codes with a scale a row (KV8: LC, LS) or packed QB-bit codes with fp16 group
    scales (QB: LC, LS, HQ the +-1 H32), the rope keys bf16 (PC)."""

    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    P = tl.load(POS)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    kq = tl.arange(0, PW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    start = c * CH
    limit = P + r
    if start <= limit:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        qp = tl.load(QP + (r * H + hh[:, None]) * PW + kq[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        qr = q
        if QB > 0:
            qr = _q_rotate(q.to(tl.float32), HQ, HBT, LW).to(tl.float16)
        # a tile wholly past the row's last key leaves m, l and o as they are (alpha 1, p 0): stop before it
        for t in range(tl.minimum(CH // KTT, (limit - start) // KTT + 1)):
            ki = start + t * KTT + tl.arange(0, KTT)
            ok = ki <= limit
            m, l, o = _keys(q, qr, qp, LC, LS, PC, ki.to(tl.int64), ok, k, kq, m, l, o, LW, PW, SCALE, KTT, KV8, QB)
        if QB > 0:
            o = _q_rotate(o, HQ, HBT, LW)                       # the chunk's value sum back from the rotated domain
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _sparse_chunks_pe(QA, QP, LC, LS, HQ, PC, TOK, CNT, PO, PM, PL, R, W: tl.constexpr, H: tl.constexpr,
                      LW: tl.constexpr, PW: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr,
                      KTT: tl.constexpr, KV8: tl.constexpr, QB: tl.constexpr):
    """Program (row, head block, chunk): HB heads of row r over its selected tokens [c CH, (c + 1) CH) in list order
    (cache formats as in _dense_chunks_pe)."""

    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(CNT + r)
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
            m, l, o = _keys(q, qr, qp, LC, LS, PC, tok, ok, k, kq, m, l, o, LW, PW, SCALE, KTT, KV8, QB)
        if QB > 0:
            o = _q_rotate(o, HQ, HBT, LW)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _sparse_rows_pe(QA, QP, LC, LS, HQ, PC, TOK, CNT, OUT, W: tl.constexpr, H: tl.constexpr, LW: tl.constexpr,
                    PW: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr,
                    KV8: tl.constexpr, QB: tl.constexpr):
    """Program (row, head block): _sparse_chunks_pe's chunk partials one after another, each folded in as _merge folds
    it (the same arithmetic in the same order), so the partials never leave registers. Rows with CNT 0 are skipped,
    chunks past the row's count and key tiles past its last token change nothing there and are not run."""

    r = tl.program_id(0)
    hb = tl.program_id(1)
    n = tl.load(CNT + r)
    if n > 0:
        hh = hb * HBT + tl.arange(0, HBT)
        hok = hh < H
        k = tl.arange(0, LW)
        kq = tl.arange(0, PW)
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        qp = tl.load(QP + (r * H + hh[:, None]) * PW + kq[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        qr = q
        if QB > 0:
            qr = _q_rotate(q.to(tl.float32), HQ, HBT, LW).to(tl.float16)
        mm = tl.full((HBT,), float("-inf"), tl.float32)
        ll = tl.zeros((HBT,), tl.float32)
        oo = tl.zeros((HBT, LW), tl.float32)
        for c in range(tl.cdiv(n, CH)):
            m = tl.full((HBT,), float("-inf"), tl.float32)
            l = tl.zeros((HBT,), tl.float32)
            o = tl.zeros((HBT, LW), tl.float32)
            for t in range(tl.minimum(CH // KTT, tl.cdiv(n - c * CH, KTT))):
                idx = c * CH + t * KTT + tl.arange(0, KTT)
                ok = idx < n
                tok = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
                m, l, o = _keys(q, qr, qp, LC, LS, PC, tok, ok, k, kq, m, l, o, LW, PW, SCALE, KTT, KV8, QB)
            if QB > 0:
                o = _q_rotate(o, HQ, HBT, LW)                   # as _sparse_chunks_pe stores it
            # _merge's step for chunk c
            active = l > 0.0
            next_m = tl.where(active, tl.maximum(mm, m), mm)
            a = tl.where(active, tl.where(mm == float("-inf"), 0.0, tl.exp(mm - next_m)), 1.0)
            b = tl.where(active, tl.exp(m - next_m), 0.0)
            oo = oo * a[:, None] + o * b[:, None]
            ll = ll * a + l * b
            mm = next_m
        tl.store(OUT + (r * H + hh[:, None]) * LW + k[None, :], (oo / ll[:, None]).to(tl.bfloat16), mask=hok[:, None])


@triton.jit
def _gather_pe(Q, QP, R, H: tl.constexpr, D: tl.constexpr, OFF: tl.constexpr, PW: tl.constexpr):
    """QP[r, h] = Q[r, h, OFF:OFF + PW] (each head's rope slice into a contiguous [R, H, PW])."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    k = tl.arange(0, PW)
    tl.store(QP + (r * H + h) * PW + k, tl.load(Q + (r * H + h) * D + OFF + k))


def gather_pe(q: torch.Tensor, qk_nope: int, out: torch.Tensor) -> torch.Tensor:
    """q [R, H, qk_dim] -> out [R, H, qk_rope] (the heads' rope slices, contiguous)."""

    R, H, D = q.shape
    PW = out.shape[2]
    _gather_pe[(R, H)](q, out, R, H=H, D=D, OFF=qk_nope, PW=PW, num_warps=1)
    return out


def _planes(cache):
    """A latent cache as the kernels take it: (rows, scales, H32, FP8?, bits) for a bf16 tensor, an FP8 plane
    (kv8.Kv8) or a quantized plane (kvq.KvQ)."""

    if isinstance(cache, Kv8):
        return cache.codes, cache.scales, cache.scales, True, 0
    if isinstance(cache, KvQ):
        return cache.codes, cache.scales, cache.h, False, cache.bits
    return cache, cache, cache, False, 0


def attention(qa: torch.Tensor, qp: torch.Tensor, cache, pcache: torch.Tensor, pos: torch.Tensor,
              s: LatentScratch, *, scale: float, nch: int, out: torch.Tensor, hb: int | None = None) -> torch.Tensor:
    """Dense causal attention of rows (qa [R, H, 512], qp [R, H, 64]) through pos + R - 1 in nch 512-key chunks.

    ``cache`` is the layer's bf16 latent rows, its FP8 plane (kv8.Kv8: codes and a scale a row) or its quantized
    plane (kvq.KvQ), read in place.

    A prompt chunk's scratch holds ``part_rows`` rows of partials: longer windows run in row blocks, each at its first
    row's device position (same bits a row as one call: rows never see each other's partials).
    """

    R = qa.shape[0]
    # 16 heads a program always: the rope operand pushes HB_WIDE's 32-head tile past GB10's 99 KiB shared memory,
    # and a TP4 rank holds 16 heads, so one program still reads each key once (16- and 32-head tiles: same bits)
    hb = HB if hb is None else hb
    step = s.part_rows
    if R <= step:
        return _attention(qa, qp, cache, pcache, pos, s, scale=scale, nch=nch, out=out, hb=hb)
    for r0 in range(0, R, step):
        r1 = min(R, r0 + step)
        at = pos if r0 == 0 else pos + r0
        _attention(qa[r0:r1], qp[r0:r1], cache, pcache, at, s, scale=scale, nch=nch, out=out[r0:r1], hb=hb)
    return out


def _attention(qa: torch.Tensor, qp: torch.Tensor, cache, pcache: torch.Tensor, pos: torch.Tensor,
               s: LatentScratch, *, scale: float, nch: int, out: torch.Tensor, hb: int) -> torch.Tensor:
    R, H, LW = qa.shape
    PW = qp.shape[2]
    if nch > s.nch or R > s.part_rows or LW != s.lw:
        raise ValueError(f"latent attention: {R} rows, {nch} chunks, width {LW} past the scratch's "
                         f"{s.part_rows}, {s.nch}, {s.lw}")
    n = nch * R * H
    if hb not in (HB, HB_WIDE):
        raise ValueError(f"latent attention: {hb} heads a program, not {HB} or {HB_WIDE}")
    # 4 warps with 3 load stages keep the 8-warp single-stage launch's bits (tools/check_mla_cfg.py) and take 21 instead
    # of 35 us a decode row at 1-2k tokens on GB10
    rows, scales, h32, kv8, qb = _planes(cache)
    _dense_chunks_pe[(R, triton.cdiv(H, hb), nch)](qa, qp, rows, scales, h32, pcache, pos, s.po[:n * LW], s.pm[:n],
                                                   s.pl[:n], R, H=H, LW=LW, PW=PW, CH=CHUNK, SCALE=scale, HBT=hb,
                                                   KTT=KT, KV8=kv8, QB=qb, num_warps=4, num_stages=3)
    _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    return out


def sparse_attention(qa: torch.Tensor, qp: torch.Tensor, cache, pcache: torch.Tensor,
                     tokens: torch.Tensor, counts: torch.Tensor, out: torch.Tensor, scale: float) -> None:
    """Rows with counts > 0 over their selected tokens (ascending, -1 padded) into out; other rows untouched (``cache``
    as in attention())."""

    R, H, LW = qa.shape
    PW = qp.shape[2]
    W = tokens.shape[1]
    rows, scales, h32, kv8, qb = _planes(cache)
    if R >= FUSED_ROWS:
        # a prompt chunk: a program a row and head block, partials merged in registers (the same bits)
        _sparse_rows_pe[(R, triton.cdiv(H, HB))](qa, qp, rows, scales, h32, pcache, tokens, counts, out, W=W, H=H,
                                                 LW=LW, PW=PW, CH=CHUNK, SCALE=scale, HBT=HB, KTT=KT, KV8=kv8, QB=qb,
                                                 num_warps=FUSED_LAUNCH[0], num_stages=FUSED_LAUNCH[1])
        return
    nch = triton.cdiv(W, CHUNK)
    n = nch * R * H
    po = torch.empty((n * LW,), dtype=torch.float32, device=qa.device)
    pm = torch.empty((n,), dtype=torch.float32, device=qa.device)
    pl = torch.empty((n,), dtype=torch.float32, device=qa.device)
    hb = HB                                       # see attention(): the rope tile stays within shared memory
    # 4 warps with 3 load stages keep the 8-warp single-stage launch's bits (tools/check_sparse_cfg.py) and take 29
    # instead of 40 us a decode row over 2,048 selected tokens on GB10
    _sparse_chunks_pe[(R, triton.cdiv(H, hb), nch)](qa, qp, rows, scales, h32, pcache, tokens, counts, po, pm, pl,
                                                    R, W=W, H=H, LW=LW, PW=PW, CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT,
                                                    KV8=kv8, QB=qb, num_warps=4, num_stages=3)
    _merge[(R, H)](po, pm, pl, out, counts, R, H=H, LW=LW, NCH=nch, SPARSE=True, num_warps=4)


# -- prompt chunks' absorb (q_nope . W_UK) and expand (o_lat . W_UV): latent's kernels, a row block a program ------------
# latent._absorb_q / _expand_v run a program per (head, column block) over every row of the window: 256 programs for a
# whole prompt chunk. These are the same programs over RB rows each (program id 2): each row's tile, product and tl.sum
# are latent's (same shapes, BN and warps), so every row gets the same bits; a chunk gets R / RB times the programs.
PROMPT_RB = 64
# "cuda": latent_rows.cu, the Triton kernels' sums in their order without a barrier a row (bf16 weights, 256 -> 512 ->
# 256 per head); "triton": _absorb_rows / _expand_rows (TF_GLM_ABSORB)
ABSORB = os.environ.get("TF_GLM_ABSORB") or "cuda"


@lru_cache(maxsize=1)
def _rows_ext():
    from pathlib import Path

    from tensorfold.cuda.build import load

    return load(name="tensorfold_glm_latent_rows_v1", sources=[str(Path(__file__).parent / "latent_rows.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def _exact_shapes(a) -> bool:
    return (a.wk.dtype == torch.bfloat16 and a.wv.dtype == torch.bfloat16 and a.wk.shape[1] == 256
            and a.lw == 512 and a.v_dim % 16 == 0)


@triton.jit
def _absorb_rows(Q, WK, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr,
                 RB: tl.constexpr):
    """Program (head, column block, row block): QA[r, h, n] = sum_k Q[r, h, k] WK[h, k, n], latent._absorb_q's sum."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    r0 = tl.program_id(2) * RB
    k = tl.arange(0, D)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WK + (h * D + k[:, None]) * LW + n[None, :]).to(tl.float32)            # [D, BN]
    for r in range(r0, tl.minimum(r0 + RB, R)):
        q = tl.load(Q + (r * H + h) * D + k).to(tl.float32)
        acc = tl.sum(q[:, None] * w, axis=0)
        tl.store(QA + (r * H + h) * LW + n, acc.to(tl.bfloat16))


@triton.jit
def _expand_rows(OL, WV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr,
                 RB: tl.constexpr):
    """Program (head, output block, row block): OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k], latent._expand_v's sum."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    r0 = tl.program_id(2) * RB
    k = tl.arange(0, LW)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.float32)          # [BN, LW]
    for r in range(r0, tl.minimum(r0 + RB, R)):
        o = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        acc = tl.sum(w * o[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))


def absorb_q(q: torch.Tensor, a, out: torch.Tensor) -> torch.Tensor:
    """latent.absorb_q (q [R, H, qk_dim] -> out [R, H, latent]); a prompt chunk's rows split over programs."""

    from tensorfold.families.glm5_next.cuda import latent

    R, H, D = q.shape
    if R <= PROMPT_RB or isinstance(a, latent.AbsorbQ4):
        return latent.absorb_q(q, a, out)
    if ABSORB == "cuda" and _exact_shapes(a) and q.is_contiguous() and out.is_contiguous():
        _rows_ext().absorb(q, a.wk, out, PROMPT_RB)
        return out
    BN = 32                                              # latent.absorb_q's: the same tile, the same sum
    _absorb_rows[(H, a.lw // BN, triton.cdiv(R, PROMPT_RB))](q, a.wk, out, R, H=H, D=D, LW=a.lw, BN=BN,
                                                             RB=PROMPT_RB, num_warps=4)
    return out


def expand_v(o_lat: torch.Tensor, a, out: torch.Tensor) -> torch.Tensor:
    """latent.expand_v (o_lat [R, H, latent] -> out [R, H, v_dim]); a prompt chunk's rows split over programs."""

    from tensorfold.families.glm5_next.cuda import latent

    R, H, _ = o_lat.shape
    if R <= PROMPT_RB or isinstance(a, latent.AbsorbQ4):
        return latent.expand_v(o_lat, a, out)
    if ABSORB == "cuda" and _exact_shapes(a) and o_lat.is_contiguous() and out.is_contiguous():
        _rows_ext().expand(o_lat, a.wv, out, PROMPT_RB)
        return out
    BN = 16                                              # latent.expand_v's
    _expand_rows[(H, a.v_dim // BN, triton.cdiv(R, PROMPT_RB))](o_lat, a.wv, out, R, H=H, DV=a.v_dim, LW=a.lw,
                                                                BN=BN, RB=PROMPT_RB, num_warps=4)
    return out
