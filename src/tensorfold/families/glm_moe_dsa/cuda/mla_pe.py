"""GLM-5.3's MLA scores with the rope term: q_lat . c_kv + q_pe . k_pe over the latent cache plus a rope-key cache.

Flash Next (GLM-5.3-Flash) has qk_rope = 0, so its latent kernels (``glm5_next.cuda.latent``) score with the latent
alone. Full GLM-5.3 keeps a 64-wide shared rope key per token: these kernels add its dot product to every score.
Values stay the latent (the rope key is never a value), so the chunk partials and their merge are Flash's
(``latent._merge``, same layout): only the scoring differs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda.latent import CHUNK, HB, HB_WIDE, KT, LatentScratch, _merge, head_block


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
def _dense_chunks_pe(QA, QP, LC, PC, POS, PO, PM, PL, R, H: tl.constexpr, LW: tl.constexpr, PW: tl.constexpr,
                     CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr):
    """Program (row, head block, chunk): causal attention of HB heads of row r over keys [c CH, (c + 1) CH)."""

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
        for t in range(CH // KTT):
            ki = start + t * KTT + tl.arange(0, KTT)
            ok = ki <= limit
            kv = tl.load(LC + ki[:, None].to(tl.int64) * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            kp = tl.load(PC + ki[:, None].to(tl.int64) * PW + kq[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile_pe(q, qp, kv, kp, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _sparse_chunks_pe(QA, QP, LC, PC, TOK, CNT, PO, PM, PL, R, W: tl.constexpr, H: tl.constexpr, LW: tl.constexpr,
                      PW: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr):
    """Program (row, head block, chunk): HB heads of row r over its selected tokens [c CH, (c + 1) CH) in list order."""

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
        for t in range(CH // KTT):
            idx = c * CH + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            tok = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
            kv = tl.load(LC + tok[:, None] * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            kp = tl.load(PC + tok[:, None] * PW + kq[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile_pe(q, qp, kv, kp, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


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


def attention(qa: torch.Tensor, qp: torch.Tensor, cache: torch.Tensor, pcache: torch.Tensor, pos: torch.Tensor,
              s: LatentScratch, *, scale: float, nch: int, out: torch.Tensor, hb: int | None = None) -> torch.Tensor:
    """Dense causal attention of rows (qa [R, H, 512], qp [R, H, 64]) through pos + R - 1 in nch 512-key chunks."""

    R, H, LW = qa.shape
    PW = qp.shape[2]
    if nch > s.nch or R > s.part_rows or LW != s.lw:
        raise ValueError(f"latent attention: {R} rows, {nch} chunks, width {LW} past the scratch's "
                         f"{s.part_rows}, {s.nch}, {s.lw}")
    n = nch * R * H
    hb = head_block(R) if hb is None else hb
    if hb not in (HB, HB_WIDE):
        raise ValueError(f"latent attention: {hb} heads a program, not {HB} or {HB_WIDE}")
    _dense_chunks_pe[(R, triton.cdiv(H, hb), nch)](qa, qp, cache, pcache, pos, s.po[:n * LW], s.pm[:n], s.pl[:n], R,
                                                   H=H, LW=LW, PW=PW, CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT,
                                                   num_warps=8, num_stages=1)
    _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    return out


def sparse_attention(qa: torch.Tensor, qp: torch.Tensor, cache: torch.Tensor, pcache: torch.Tensor,
                     tokens: torch.Tensor, counts: torch.Tensor, out: torch.Tensor, scale: float) -> None:
    """Rows with counts > 0 over their selected tokens (ascending, -1 padded) into out; other rows untouched."""

    R, H, LW = qa.shape
    PW = qp.shape[2]
    W = tokens.shape[1]
    nch = triton.cdiv(W, CHUNK)
    n = nch * R * H
    po = torch.empty((n * LW,), dtype=torch.float32, device=qa.device)
    pm = torch.empty((n,), dtype=torch.float32, device=qa.device)
    pl = torch.empty((n,), dtype=torch.float32, device=qa.device)
    hb = head_block(R)
    _sparse_chunks_pe[(R, triton.cdiv(H, hb), nch)](qa, qp, cache, pcache, tokens, counts, po, pm, pl, R, W=W, H=H,
                                                    LW=LW, PW=PW, CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT,
                                                    num_warps=8, num_stages=1)
    _merge[(R, H)](po, pm, pl, out, counts, R, H=H, LW=LW, NCH=nch, SPARSE=True, num_warps=4)
