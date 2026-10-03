"""Decode context parallelism (TF_GLM_DCP=4, a long-context startup mode): the cache's positions interleaved over the
ranks, so each rank holds a quarter of every plane and the window reaches ~4x further.

The design follows ashhart/TensorFold PR #159 (dc43ca6, drowzeys: fused._attention_dcp, _merge_lse, _dcp_combine,
dcp_gather, select), restated for this family's kernels and cache formats:
- Position p lives on rank p % G at local slot p // G: latent, rope key, indexer keys and the MTP head's alike; every
  writer stores only its own rows.
- Selection (full indexer layers, rows past the dense limit): each rank scores its own slots (a token's score is the
  replicated scorer's, bit for bit), keeps its top index_topk (ties to the lower position), the ranks all-gather those
  candidates as (score, position) keys and every rank takes the same global top index_topk: the replicated selection
  exactly (positions are unique, so the merge is tie-free). A rank keeps the selected positions it holds.
- Attention: the ranks all-gather every head's absorbed query; each rank attends all heads over its own keys (rows
  below the dense limit: every visible key it holds; past it: its share of the selection), normalizes each head's
  partial and keeps its log-sum-exp; each head's partials go to the rank that owns the head, merged in rank order.
  Only the slots a rank holds are read: nothing is masked full width (vllm#58980).
Rows never meet, so a verify window's row keeps the serial step's bits (drafted == serial); the bits differ from the
replicated mode's (the merge order differs), so this mode has its own serial reference.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .kvq import rotate as _q_rotate, tile as _q_tile
from .mla_pe import _keys

DCP = int(os.environ.get("TF_GLM_DCP") or 1)


def local_slots(capacity: int, world: int) -> int:
    """Cache rows one rank holds for ``capacity`` positions interleaved over ``world`` ranks."""

    return -(-capacity // world) + 1


# -- writers ---------------------------------------------------------------------------------------------------------
@triton.jit
def _write_bf16(X, x_stride, C, POS, W: tl.constexpr, G: tl.constexpr, RANK: tl.constexpr):
    """Program r: row r (position POS + r) into slot (POS + r) // G when this rank holds it."""

    r = tl.program_id(0)
    p = (tl.load(POS) + r).to(tl.int64)
    if p % G == RANK:
        k = tl.arange(0, W)
        tl.store(C + (p // G) * W + k, tl.load(X + r * x_stride + k))


def write_bf16(rows: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor, G: int, rank: int) -> None:
    _write_bf16[(rows.shape[0],)](rows, rows.stride(0), cache, pos, W=cache.shape[1], G=G, RANK=rank, num_warps=4)


# -- selection -------------------------------------------------------------------------------------------------------
@triton.jit
def _scores_dcp(QI, W, w_stride, IK, IS, HQ, OUT, POS, R, NT, scale, wscale, H: tl.constexpr, HP: tl.constexpr,
                D: tl.constexpr, BT: tl.constexpr, RB: tl.constexpr, KV8: tl.constexpr, QB: tl.constexpr,
                G: tl.constexpr, RANK: tl.constexpr):
    """select._scores over this rank's slots t (position t G + RANK): s_t = sum_h w_h relu(scale qi_h . k_t), -inf past
    each row's position. A token's arithmetic is the replicated scorer's (the same dot, head order and scales)."""

    rb = tl.program_id(0)
    tb = tl.program_id(1)
    P = tl.load(POS)
    t = tb * BT + tl.arange(0, BT)
    g = t * G + RANK
    last = P + rb * RB + RB                                   # past every row of the block
    if tb * BT * G + RANK < last:
        d = tl.arange(0, D)
        hh = tl.arange(0, HP)
        hok = hh < H
        if KV8:
            k = tl.load(IK + t[:, None].to(tl.int64) * D + d[None, :], mask=(g < last)[:, None],
                        other=0).to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            ks = tl.load(IS + t.to(tl.int64), mask=g < last, other=0.0)
        elif QB > 0:
            k = _q_tile(IK, IS, t.to(tl.int64), g < last, D, QB, BT)
        else:
            k = tl.load(IK + t[:, None].to(tl.int64) * D + d[None, :], mask=(g < last)[:, None],
                        other=0.0).to(tl.bfloat16)
        for i in tl.static_range(RB):
            r = rb * RB + i
            if r < R:
                bound = P + r + 1
                q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None],
                            other=0.0).to(tl.bfloat16)
                if QB > 0:
                    q = _q_rotate(q.to(tl.float32), HQ, HP, D).to(tl.float16)
                kr = tl.where((g < bound)[:, None], k, 0.0)
                dots = tl.dot(q, tl.trans(kr))
                if KV8:
                    dots = dots * ks[None, :]
                w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
                sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
                sc = tl.where(g < bound, sc, float("-inf"))
                tl.store(OUT + r * NT + t, sc, mask=t < NT)
    else:
        for i in tl.static_range(RB):
            r = rb * RB + i
            if r < R:
                tl.store(OUT + r * NT + t, tl.full((BT,), float("-inf"), tl.float32), mask=t < NT)


@triton.jit
def _candidates(S, s_stride, TOK, t_stride, KEYS, k_stride, K, G: tl.constexpr, RANK: tl.constexpr,
                BLOCK: tl.constexpr):
    """Program r: row r's K local picks (slots, ascending) as int64 keys in select._order_key's order, the score then
    the lower position first: (signed ordered score bits << 32) | (0xFFFFFFFF - position); a pick of an invisible slot
    (-inf) is no candidate (int64 min)."""

    r = tl.program_id(0).to(tl.int64)
    for i0 in range(0, K, BLOCK):
        i = i0 + tl.arange(0, BLOCK)
        ok = i < K
        slot = tl.load(TOK + r * t_stride + i, mask=ok, other=0)
        s = tl.load(S + r * s_stride + slot, mask=ok, other=float("-inf"))
        bits = (s + 0.0).to(tl.int32, bitcast=True)              # select._order_key's: signed, negatives flipped
        order = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
        key = (order.to(tl.int64) << 32) | (0xFFFFFFFF - (slot.to(tl.int64) * G + RANK))
        tl.store(KEYS + r * k_stride + i, tl.where(s == float("-inf"), -9223372036854775807 - 1, key), mask=ok)


def candidates(qi: torch.Tensor, wts: torch.Tensor, keys, pos_dev: torch.Tensor, n: int, np_max: int, topk: int,
               G: int, rank: int) -> torch.Tensor:
    """Rows pos_dev .. + n: this rank's top-k slots (of np_max scored) as candidate keys [n, topk] int64 (no
    candidate: int64 min)."""

    from . import select as sel
    from .kv8 import Kv8
    from .kvq import KvQ

    H = wts.shape[1]
    D = qi.shape[1] // H
    kv8, qb = isinstance(keys, Kv8), keys.bits if isinstance(keys, KvQ) else 0
    ik, isc = (keys.codes, keys.scales) if kv8 or qb else (keys, keys)
    hq = keys.h if qb else isc
    scores = torch.empty((n, np_max), dtype=torch.float32, device=qi.device)
    rb = 16 if n >= 16 else n
    _scores_dcp[(triton.cdiv(n, rb), triton.cdiv(np_max, 64))](
        qi, wts, wts.stride(0), ik, isc, hq, scores, pos_dev, n, np_max, D ** -0.5, H ** -0.5, H=H,
        HP=max(16, triton.next_power_of_2(H)), D=D, BT=64, RB=rb, KV8=kv8, QB=qb, G=G, RANK=rank, num_warps=4)
    k = min(topk, np_max)
    local = torch.zeros((n, topk), dtype=torch.int32, device=qi.device)
    if n >= sel.RADIX_ROWS:
        sel._radix_topk[(n,)](scores, scores.stride(0), local, local.stride(0), np_max, K=k, BLOCK=1024, num_warps=4)
    else:
        sel._split_topk(scores, local, k, np_max)
    cand = torch.full((n, topk), -9223372036854775807 - 1, dtype=torch.int64, device=qi.device)
    _candidates[(n,)](scores, scores.stride(0), local, local.stride(0), cand, cand.stride(0), k, G=G, RANK=rank,
                      BLOCK=1024, num_warps=4)
    return cand


def merge(every: torch.Tensor, G: int, rank: int, topk: int, tokens: torch.Tensor, counts: torch.Tensor) -> None:
    """Every rank's candidates [G, n, topk] -> the global top-k of each row (the replicated selection: positions are
    unique, so the keys never tie) and, into tokens [n, width] / counts [n], the slots this rank holds of it,
    ascending, -1 past them."""

    n = every.shape[1]
    top = torch.topk(every.permute(1, 0, 2).reshape(n, G * topk), topk, dim=1, sorted=False).values
    p = 0xFFFFFFFF - (top & 0xFFFFFFFF)                       # the global positions
    own = ((p % G) == rank) & (top != -9223372036854775807 - 1)
    slot = torch.where(own, p // G, torch.full_like(p, 1 << 40))
    tokens[:, :topk].copy_(torch.sort(slot, dim=1).values.clamp(max=(1 << 31) - 1).to(torch.int32))
    cnt = own.sum(1).to(torch.int32)
    tokens[:, :topk].masked_fill_(torch.arange(topk, device=every.device)[None, :] >= cnt[:, None], -1)
    if tokens.shape[1] > topk:
        tokens[:, topk:] = -1
    counts.copy_(cnt)


def select(w, qi: torch.Tensor, wts: torch.Tensor, keys, pos: int | None, R: int, topk: int, pos_dev: torch.Tensor,
           *, tokens: torch.Tensor, counts: torch.Tensor, bucket: int | None = None) -> None:
    """select.select_tokens across the ranks' slots: tokens[r, :counts[r]] this rank's slots of row r's global top-k
    (ascending), the rest -1. Rows at or below the dense limit get no use of theirs (attention reads their range).
    ``bucket`` fixes the slots scored (captured graphs, no host position); else they follow the host position."""

    from . import select as sel

    G, rank = w.world, w.rank
    if bucket is not None:
        np_max = bucket
    else:
        start = int(pos) if pos is not None else int(pos_dev.item())
        np_max = sel.sparse_bucket(start, R, G)
        if pos is not None and R >= sel.RADIX_ROWS:            # prompt chunks: up to the last visible slot
            np_max = min(np_max, -(-(-(-(start + R) // G)) // 64) * 64)
    np_max = min(np_max, keys.shape[0])
    rows = max(1, min(R, sel.SELECT_BYTES // (4 * np_max)))
    for r0 in range(0, R, rows):
        n = min(rows, R - r0)
        cand = candidates(qi[r0:r0 + n], wts[r0:r0 + n], keys, pos_dev + r0 if r0 else pos_dev, n, np_max, topk, G,
                          rank)
        every = torch.empty((G, n, topk), dtype=torch.int64, device=qi.device)
        w.comm.all_gather(cand.view(-1).view(torch.float32), every.view(-1).view(torch.float32))
        merge(every, G, rank, topk, tokens[r0:r0 + n], counts[r0:r0 + n])


# -- attention -------------------------------------------------------------------------------------------------------
@triton.jit
def _attend_dcp(QALL, LC, LS, HQ, PC, TOK, CNT, POS, SEND, R, W: tl.constexpr, H: tl.constexpr, LW: tl.constexpr,
                PW: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, KTT: tl.constexpr, K: tl.constexpr,
                G: tl.constexpr, RANK: tl.constexpr, KV8: tl.constexpr, QB: tl.constexpr):
    """Program (row, head group g): rank g's H heads (QALL [G, R, H, LW + PW]) of row r over this rank's keys of the
    row: its slots of positions 0 .. p while p < K (the dense limit), else its share of the selection (TOK, CNT), in
    chunks of CH merged as _merge merges them. Writes each head's normalized partial (bf16 pairs as fp32 words) and
    log-sum-exp into SEND [G, R, H, LW / 2 + 1] at group g (no key: 0 and -inf)."""

    r = tl.program_id(0)
    grp = tl.program_id(1)
    p = tl.load(POS) + r
    dense = p < K
    if dense:
        n = tl.where(p >= RANK, (p - RANK) // G + 1, 0)
    else:
        n = tl.load(CNT + r)
    hh = tl.arange(0, H)
    k = tl.arange(0, LW)
    kq = tl.arange(0, PW)
    qb = QALL + ((grp * R + r) * H + hh[:, None]) * (LW + PW)
    q = tl.load(qb + k[None, :]).to(tl.bfloat16)
    qp = tl.load(qb + LW + kq[None, :]).to(tl.bfloat16)
    qr = q
    if QB > 0:
        qr = _q_rotate(q.to(tl.float32), HQ, H, LW).to(tl.float16)
    mm = tl.full((H,), float("-inf"), tl.float32)
    ll = tl.zeros((H,), tl.float32)
    oo = tl.zeros((H, LW), tl.float32)
    for c in range(tl.cdiv(n, CH)):
        m = tl.full((H,), float("-inf"), tl.float32)
        l = tl.zeros((H,), tl.float32)
        o = tl.zeros((H, LW), tl.float32)
        for t in range(tl.minimum(CH // KTT, tl.cdiv(n - c * CH, KTT))):
            idx = c * CH + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            if dense:
                rows = idx.to(tl.int64)
            else:
                rows = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
            m, l, o = _keys(q, qr, qp, LC, LS, PC, rows, ok, k, kq, m, l, o, LW, PW, SCALE, KTT, KV8, QB)
        if QB > 0:
            o = _q_rotate(o, HQ, H, LW)
        active = l > 0.0
        next_m = tl.where(active, tl.maximum(mm, m), mm)
        a = tl.where(active, tl.where(mm == float("-inf"), 0.0, tl.exp(mm - next_m)), 1.0)
        b = tl.where(active, tl.exp(m - next_m), 0.0)
        oo = oo * a[:, None] + o * b[:, None]
        ll = ll * a + l * b
        mm = next_m
    has = ll > 0.0
    out = tl.where(has[:, None], oo / tl.where(has, ll, 1.0)[:, None], 0.0).to(tl.bfloat16)
    WORDS: tl.constexpr = LW // 2 + 1
    base = SEND + ((grp * R + r) * H + hh) * WORDS
    pairs = tl.reshape(out, (H, LW // 2, 2))
    lo, hi = tl.split(pairs)
    word = lo.to(tl.uint16, bitcast=True).to(tl.uint32) | (hi.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
    tl.store(base[:, None] + tl.arange(0, LW // 2)[None, :], word.to(tl.float32, bitcast=True))
    tl.store(base + LW // 2, tl.where(has, mm + tl.log(tl.where(has, ll, 1.0)), float("-inf")))


@triton.jit
def _combine(RECV, OUT, R, src_stride, H: tl.constexpr, LW: tl.constexpr, G: tl.constexpr):
    """Program (row, own head): every rank's normalized partial of the head (RECV + src * src_stride, laid out [R, H,
    LW / 2 + 1] words) merged by their log-sum-exps in rank order -> OUT [R, H, LW] bf16."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    WORDS: tl.constexpr = LW // 2 + 1
    j = tl.arange(0, LW // 2)
    mx = float("-inf")
    for s in tl.static_range(G):
        mx = tl.maximum(mx, tl.load(RECV + s * src_stride + (r * H + h) * WORDS + LW // 2))
    acc_lo = tl.zeros((LW // 2,), tl.float32)
    acc_hi = tl.zeros((LW // 2,), tl.float32)
    den = 0.0
    for s in tl.static_range(G):
        base = RECV + s * src_stride + (r * H + h) * WORDS
        ls = tl.load(base + LW // 2)
        wgt = tl.where(ls == float("-inf"), 0.0, tl.exp(ls - mx))
        word = tl.load(base + j).to(tl.uint32, bitcast=True)
        lo = (word & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        hi = (word >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        acc_lo += wgt * lo
        acc_hi += wgt * hi
        den += wgt
    d = tl.where(den > 0.0, den, 1.0)
    out = tl.join(acc_lo / d, acc_hi / d)                     # [LW / 2, 2]: back to the value order
    tl.store(OUT + (r * H + h) * LW + tl.arange(0, LW), tl.reshape(out, (LW,)).to(tl.bfloat16))


class Scratch:
    """A window's DCP exchange buffers (flat, viewed contiguous for each window's rows): every head's absorbed query,
    the partials sent, the partials received."""

    def __init__(self, rows: int, heads: int, lw: int, pw: int, world: int, device) -> None:
        self.rows, self.heads, self.lw, self.pw, self.world = rows, heads, lw, pw, world
        words = lw // 2 + 1
        self.qpack = torch.empty((rows * heads * (lw + pw),), dtype=torch.bfloat16, device=device)
        self.qall = torch.empty((world * rows * heads * (lw + pw),), dtype=torch.bfloat16, device=device)
        self.send = torch.empty((world * rows * heads * words,), dtype=torch.float32, device=device)
        # decode windows gather every rank's whole send (a rank reads its own group); prompt chunks exchange groups
        self.recv = torch.empty(((world if rows <= 64 else 1) * world * rows * heads * words,), dtype=torch.float32,
                                device=device)


def attend(qall: torch.Tensor, cache, pcache: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
           pos: torch.Tensor, send: torch.Tensor, R: int, H: int, G: int, rank: int, scale: float, topk: int) -> None:
    """Every rank's heads (qall [G, R, H, LW + PW]) of the window's rows over the keys this rank holds -> send
    [G, R, H, LW / 2 + 1] (each head's normalized partial and log-sum-exp, laid out by the head's owner)."""

    from .mla_pe import FUSED_LAUNCH, KT, _planes

    PW = pcache.shape[1]
    LW = qall.shape[-1] - PW
    rows, scales, h32, kv8, qb = _planes(cache)
    _attend_dcp[(R, G)](qall, rows, scales, h32, pcache, tokens, counts, pos, send, R, W=tokens.shape[1], H=H, LW=LW,
                        PW=PW, CH=512, SCALE=scale, KTT=KT, K=topk, G=G, RANK=rank, KV8=kv8, QB=qb,
                        num_warps=FUSED_LAUNCH[0], num_stages=FUSED_LAUNCH[1])


def combine(src: torch.Tensor, stride: int, out: torch.Tensor, G: int) -> None:
    """Every rank's partials of this rank's heads (src, a [R, H, LW / 2 + 1] block every ``stride`` words, in rank
    order) merged by their log-sum-exps -> out [R, H, LW] bf16."""

    R, H, LW = out.shape
    _combine[(R, H)](src, out, R, stride, H=H, LW=LW, G=G, num_warps=4)


def attention(w, qa: torch.Tensor, qp: torch.Tensor, cache, pcache: torch.Tensor, tokens: torch.Tensor,
              counts: torch.Tensor, pos: torch.Tensor, out: torch.Tensor, scale: float, d: Scratch, topk: int) -> None:
    """Every head's attention over the keys this rank holds, merged across the ranks into this rank's heads (out)."""

    R, H, LW = qa.shape
    PW = qp.shape[2]
    G, rank = w.world, w.rank
    qpack = d.qpack[:R * H * (LW + PW)].view(R, H, LW + PW)
    qpack[:, :, :LW].copy_(qa)
    qpack[:, :, LW:].copy_(qp)
    qall = d.qall[:G * R * H * (LW + PW)]
    if R <= 64:                                           # bf16 pairs as fp32 words: the RDMA gather takes them
        w.comm.all_gather(qpack.view(-1).view(torch.float32), qall.view(torch.float32))
    else:
        w.comm.all_gather(qpack.view(-1), qall)
    words = LW // 2 + 1
    send = d.send[:G * R * H * words].view(G, R, H, words)
    attend(qall.view(G, R, H, LW + PW), cache, pcache, tokens, counts, pos, send, R, H, G, rank, scale, topk)
    if R <= 64:                                           # decode windows: gather everything, read our group
        recv = d.recv[:G * G * R * H * words].view(G, G, R, H, words)
        w.comm.all_gather(send.view(-1), recv.view(-1))
        combine(recv[:, rank], G * R * H * words, out, G)
    else:                                                 # prompt chunks: each group to its owner
        recv = d.recv[:G * R * H * words].view(G, R, H, words)
        w.comm.grouped([(send[g], g) for g in range(G)], [(recv[g], g) for g in range(G)])
        combine(recv, R * H * words, out, G)
