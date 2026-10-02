"""GLM-5.3's raw-token DSA selection: per-row top-index_topk lightning-indexer scores, exact ties, ascending order."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .kvq import rotate as _q_rotate, tile as _q_tile


@triton.jit
def _scores(QI, W, w_stride, IK, IS, HQ, OUT, POS, R, NT, scale, wscale, H: tl.constexpr, HP: tl.constexpr,
            D: tl.constexpr, BT: tl.constexpr, RB: tl.constexpr, KV8: tl.constexpr, QB: tl.constexpr):
    """Program (RB rows, token block): s_t = sum_h w_h relu(scale * qi_h . k_t) up to each row's position.

    The reference (exllamav3's dsa_indexer_scores) folds D_i**-0.5 and H_i**-0.5 into one scale and
    sums relu(q.k) * w_h over the heads in head order; the same order here keeps a row's bits fixed.
    KV8: the keys are FP8 codes (IK, bf16 in registers) with a power-of-two scale a token (IS) folded into each
    token's dot products. QB: the keys are kvq's QB-bit groups (IK codes, IS fp16 scales, HQ the +-1 H32), read as
    rotated fp16 values against each query head rotated alike.
    """

    rb = tl.program_id(0)
    tb = tl.program_id(1)
    P = tl.load(POS)
    t = tb * BT + tl.arange(0, BT)
    if tb * BT < P + rb * RB + RB:
        d = tl.arange(0, D)
        hh = tl.arange(0, HP)
        hok = hh < H
        if KV8:
            k = tl.load(IK + t[:, None].to(tl.int64) * D + d[None, :], mask=(t < (P + rb * RB + RB))[:, None],
                        other=0).to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)       # [BT, D]
            ks = tl.load(IS + t.to(tl.int64), mask=t < (P + rb * RB + RB), other=0.0)
        elif QB > 0:
            k = _q_tile(IK, IS, t.to(tl.int64), t < (P + rb * RB + RB), D, QB, BT)    # [BT, D] fp16, rotated
        else:
            k = tl.load(IK + t[:, None].to(tl.int64) * D + d[None, :], mask=(t < (P + rb * RB + RB))[:, None],
                        other=0.0).to(tl.bfloat16)                                     # [BT, D]
        for i in tl.static_range(RB):
            r = rb * RB + i
            if r < R:
                bound = P + r + 1
                q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None],
                            other=0.0).to(tl.bfloat16)
                if QB > 0:
                    q = _q_rotate(q.to(tl.float32), HQ, HP, D).to(tl.float16)
                kr = tl.where((t < bound)[:, None], k, 0.0)
                dots = tl.dot(q, tl.trans(kr))                                         # [HP, BT] fp32
                if KV8:
                    dots = dots * ks[None, :]
                w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
                sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
                sc = tl.where(t < bound, sc, float("-inf"))
                tl.store(OUT + r * NT + t, sc, mask=t < NT)
    else:
        # every token of the block is past every row's last visible one: the scores the dot would give, -inf
        for i in tl.static_range(RB):
            r = rb * RB + i
            if r < R:
                tl.store(OUT + r * NT + t, tl.full((BT,), float("-inf"), tl.float32), mask=t < NT)


def _order_key(scores: torch.Tensor) -> torch.Tensor:
    """fp32 scores as int64 keys whose sort is score-ascending with ties to the lower token (Flash's trick)."""

    bits = (scores + 0.0).view(torch.int32)                    # + 0.0: -0 becomes +0, as the sort ties them
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)   # IEEE order as signed ints (negatives flipped)
    keys = ordered.to(torch.int64).bitwise_left_shift_(32)
    keys.bitwise_or_(0xFFFFFFFF - torch.arange(scores.shape[1], device=scores.device, dtype=torch.int64))
    return keys


@triton.jit
def _ordered(s):
    """fp32 scores as uint32 keys in score order (-0 as +0: the sort ties them)."""

    bits = (s + 0.0).to(tl.uint32, bitcast=True)
    return tl.where(bits >= 0x80000000, ~bits, bits | 0x80000000)


@triton.jit
def _radix_topk(S, s_stride, TOK, t_stride, NP, K: tl.constexpr, BLOCK: tl.constexpr):
    """Program r: row r's K highest scores (ties to the lower token) into TOK[r, :K] in ascending token order.

    Four 8-bit radix passes find the K-th key and how many of its ties to keep; a last pass in token order keeps
    every key above it and the first such ties: the set and order torch.topk + sort produce on _order_key.
    """

    r = tl.program_id(0).to(tl.int64)
    row = S + r * s_stride
    bins = tl.arange(0, 512)
    real = bins < 256
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = tl.full((), K, dtype=tl.int32)
    for p in tl.static_range(4):
        shift = 24 - 8 * p
        hist = tl.zeros((512,), dtype=tl.int32)
        for start in range(0, NP, BLOCK):
            i = start + tl.arange(0, BLOCK)
            ok = i < NP
            key = _ordered(tl.load(row + i, mask=ok, other=0.0))
            match = ok & ((key & fixed) == prefix)
            digit = tl.where(match, ((key >> shift) & 0xFF).to(tl.int32), 256)
            hist += tl.histogram(digit, 512)
        hist = tl.where(real, hist, 0)
        above = tl.cumsum(hist, 0, reverse=True)                     # keys with this digit or a higher one
        d = tl.max(tl.where(real & (above >= need), bins, -1), 0)
        at_d = tl.sum(tl.where(bins == d, hist, 0), 0)
        need = need - (tl.sum(tl.where(bins == d, above, 0), 0) - at_d)
        prefix = prefix | (d.to(tl.uint32) << shift)
        fixed = fixed | (0xFF << shift)
    out = tl.zeros((), dtype=tl.int32)
    ties = tl.zeros((), dtype=tl.int32)
    for start in range(0, NP, BLOCK):
        i = start + tl.arange(0, BLOCK)
        ok = i < NP
        key = _ordered(tl.load(row + i, mask=ok, other=0.0))
        gt = ok & (key > prefix)
        eq = ok & (key == prefix)
        e = eq.to(tl.int32)
        take = gt | (eq & (ties + tl.cumsum(e, 0) - e < need))
        t = take.to(tl.int32)
        tl.store(TOK + r * t_stride + out + tl.cumsum(t, 0) - t, i.to(tl.int32), mask=take)
        out += tl.sum(t, 0)
        ties += tl.sum(e, 0)


@triton.jit
def _split_hist(S, s_stride, STATE, HIST, NP, SEG: tl.constexpr, BLOCK: tl.constexpr, SHIFT: tl.constexpr):
    """Program (row r, segment g): the digit at SHIFT of the segment's keys under row r's fixed prefix, into HIST[r]."""

    r = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    prefix = tl.load(STATE + r * 4).to(tl.uint32)
    fixed = tl.load(STATE + r * 4 + 1).to(tl.uint32)
    row = S + r * s_stride
    hist = tl.zeros((512,), dtype=tl.int32)
    for t in range(0, SEG, BLOCK):
        i = g * SEG + t + tl.arange(0, BLOCK)
        ok = i < NP
        key = _ordered(tl.load(row + i, mask=ok, other=0.0))
        match = ok & ((key & fixed) == prefix)
        hist += tl.histogram(tl.where(match, ((key >> SHIFT) & 0xFF).to(tl.int32), 256), 512)
    bins = tl.arange(0, 512)
    tl.atomic_add(HIST + r * 256 + bins, hist, mask=(bins < 256) & (hist > 0))


@triton.jit
def _split_resolve(STATE, HIST, SHIFT: tl.constexpr):
    """Program r: the digit at SHIFT of row r's K-th key (prefix, fixed mask and ties still needed in STATE[r])."""

    r = tl.program_id(0).to(tl.int64)
    bins = tl.arange(0, 256)
    hist = tl.load(HIST + r * 256 + bins)
    tl.store(HIST + r * 256 + bins, tl.zeros((256,), dtype=tl.int32))
    need = tl.load(STATE + r * 4 + 2)
    above = tl.cumsum(hist, 0, reverse=True).to(tl.int64)
    d = tl.max(tl.where(above >= need, bins, -1), 0)
    at_d = tl.sum(tl.where(bins == d, hist, 0), 0).to(tl.int64)
    need = need - (tl.sum(tl.where(bins == d, above, 0), 0) - at_d)
    tl.store(STATE + r * 4, tl.load(STATE + r * 4) | (d.to(tl.int64) << SHIFT))
    tl.store(STATE + r * 4 + 1, tl.load(STATE + r * 4 + 1) | (0xFF << SHIFT))
    tl.store(STATE + r * 4 + 2, need)


@triton.jit
def _split_count(S, s_stride, STATE, COUNTS, NP, NSEG, SEG: tl.constexpr, BLOCK: tl.constexpr):
    """Program (r, g): the segment's keys above row r's K-th key and equal to it."""

    r = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    kth = tl.load(STATE + r * 4).to(tl.uint32)
    row = S + r * s_stride
    gt = tl.zeros((), dtype=tl.int32)
    eq = tl.zeros((), dtype=tl.int32)
    for t in range(0, SEG, BLOCK):
        i = g * SEG + t + tl.arange(0, BLOCK)
        ok = i < NP
        key = _ordered(tl.load(row + i, mask=ok, other=0.0))
        gt += tl.sum((ok & (key > kth)).to(tl.int32), 0)
        eq += tl.sum((ok & (key == kth)).to(tl.int32), 0)
    tl.store(COUNTS + (r * NSEG + g) * 2, gt)
    tl.store(COUNTS + (r * NSEG + g) * 2 + 1, eq)


@triton.jit
def _split_offsets(STATE, COUNTS, NSEG, NSP: tl.constexpr):
    """Program r: each segment's first output slot and the ties before it (in place of its counts)."""

    r = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, NSP)
    ok = s < NSEG
    gt = tl.load(COUNTS + (r * NSEG + s) * 2, mask=ok, other=0)
    eq = tl.load(COUNTS + (r * NSEG + s) * 2 + 1, mask=ok, other=0)
    need = tl.load(STATE + r * 4 + 2).to(tl.int32)
    before = tl.cumsum(eq, 0) - eq
    take = gt + tl.minimum(tl.maximum(need - before, 0), eq)
    tl.store(COUNTS + (r * NSEG + s) * 2, tl.cumsum(take, 0) - take, mask=ok)
    tl.store(COUNTS + (r * NSEG + s) * 2 + 1, before, mask=ok)


@triton.jit
def _split_write(S, s_stride, STATE, COUNTS, TOK, t_stride, NP, NSEG, SEG: tl.constexpr, BLOCK: tl.constexpr):
    """Program (r, g): the segment's kept tokens, in token order, at its slots of TOK[r]."""

    r = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    kth = tl.load(STATE + r * 4).to(tl.uint32)
    need = tl.load(STATE + r * 4 + 2).to(tl.int32)
    out = tl.load(COUNTS + (r * NSEG + g) * 2)
    ties = tl.load(COUNTS + (r * NSEG + g) * 2 + 1)
    row = S + r * s_stride
    for t in range(0, SEG, BLOCK):
        i = g * SEG + t + tl.arange(0, BLOCK)
        ok = i < NP
        key = _ordered(tl.load(row + i, mask=ok, other=0.0))
        e = (ok & (key == kth)).to(tl.int32)
        take = (ok & (key > kth)) | ((e > 0) & (ties + tl.cumsum(e, 0) - e < need))
        x = take.to(tl.int32)
        tl.store(TOK + r * t_stride + out + tl.cumsum(x, 0) - x, i.to(tl.int32), mask=take)
        out += tl.sum(x, 0)
        ties += tl.sum(e, 0)


SPLIT_SEG = 16384           # scores a program of the decode-row selection reads


def _split_topk(scores: torch.Tensor, tokens: torch.Tensor, k: int, np_max: int) -> None:
    """Decode rows: _radix_topk's selection with every row's scores split over programs (a few launches, no host sync)."""

    R = scores.shape[0]
    nseg = triton.cdiv(np_max, SPLIT_SEG)
    state = torch.zeros((R, 4), dtype=torch.int64, device=scores.device)
    state[:, 2] = k
    hist = torch.zeros((R, 256), dtype=torch.int32, device=scores.device)
    counts = torch.empty((R, nseg, 2), dtype=torch.int32, device=scores.device)
    for p in range(4):
        _split_hist[(R, nseg)](scores, scores.stride(0), state, hist, np_max, SEG=SPLIT_SEG, BLOCK=1024,
                               SHIFT=24 - 8 * p, num_warps=4)
        _split_resolve[(R,)](state, hist, SHIFT=24 - 8 * p, num_warps=4)
    _split_count[(R, nseg)](scores, scores.stride(0), state, counts, np_max, nseg, SEG=SPLIT_SEG, BLOCK=1024,
                            num_warps=4)
    _split_offsets[(R,)](state, counts, nseg, NSP=max(16, triton.next_power_of_2(nseg)), num_warps=4)
    _split_write[(R, nseg)](scores, scores.stride(0), state, counts, tokens, tokens.stride(0), np_max, nseg,
                            SEG=SPLIT_SEG, BLOCK=1024, num_warps=4)


def sparse_bucket(pos: int, R: int) -> int:
    """Tokens scored for rows pos .. pos + R - 1: the visible ones rounded up to a power of two (at least 2048)."""

    visible = pos + R
    return max(2048, 1 << (visible - 1).bit_length())


def sparse_buckets(capacity: int, dense_limit: int) -> list[int]:
    """Every value ``sparse_bucket`` gives a window starting at or past the dense limit within ``capacity`` slots."""

    out, bucket = [], sparse_bucket(dense_limit, 1)
    while True:
        out.append(bucket)
        if bucket >= capacity:
            return out
        bucket *= 2


SELECT_BYTES = 96 << 20     # a selection's scores and sort keys held at once: a prompt chunk selects in row blocks
TRIM = os.environ.get("TF_GLM_SELECT_TRIM", "1") != "0"   # prompt chunks score up to their last visible token only
RADIX_ROWS = 16             # blocks of this many rows or more select with _radix_topk (a program a row), fewer rows
                            # (decode windows) with _split_topk: the tokens torch.topk + sort pick, 5x faster for a
                            # 2,048-row chunk at 128k and ~10x for a decode row (check_select.py)


def select_tokens(qi: torch.Tensor, wts: torch.Tensor, keys, pos: int | None, R: int,
                  topk: int, pos_dev: torch.Tensor, *, tokens: torch.Tensor, counts: torch.Tensor,
                  bucket: int | None = None) -> None:
    """Each row's attended tokens [R, width] ascending (-1 padded) and their count past the dense limit.

    ``keys`` is the layer group's bf16 indexer key plane, its FP8 plane (kv8.Kv8) or its quantized one (kvq.KvQ).

    ``bucket`` fixes the scored token count for captured graphs; without it the visible count is
    rounded up to a power of two so the allocator reuses a few sizes. Scores, top-k membership and
    ascending order match the reference exactly: any global top-k member is in its own tile's top-k,
    ties go to the lower token, and a row below the dense limit keeps every token (count 0 marks it).
    A row's selection never depends on the others, so a prompt chunk's rows go in blocks of at most
    SELECT_BYTES of scores and keys (a 2,048-row chunk at 128k tokens would otherwise hold ~3 GiB),
    and blocks of 16 rows or more score 16 rows a program, reading each key tile once for them.
    """

    if qi.stride(0) != qi.shape[1] or wts.stride(1) != 1:
        raise ValueError("select_tokens: index queries must be contiguous rows, weights unit-stride columns")
    np_max = bucket if bucket is not None else sparse_bucket(int(pos) if pos is not None
                                                             else int(pos_dev.item()), R)
    if TRIM and bucket is None and pos is not None and R >= RADIX_ROWS:
        # a prompt chunk (eager, host position known) scores only up to its last row's visible tokens: past them
        # every score is -inf, which neither a selected row's top-k nor its order can contain (it sees > topk tokens)
        np_max = min(np_max, -(-(int(pos) + R) // 64) * 64)
    np_max = min(np_max, keys.shape[0])
    per_row = 4 * np_max if R >= RADIX_ROWS else 12 * np_max      # the radix path keeps only the fp32 scores
    rows = max(1, min(R, SELECT_BYTES // per_row))
    if rows >= R:
        _select(qi, wts, keys, R, topk, pos_dev, tokens, counts, np_max)
        return
    for r0 in range(0, R, rows):
        r1 = min(R, r0 + rows)
        _select(qi[r0:r1], wts[r0:r1], keys, r1 - r0, topk, pos_dev + r0, tokens[r0:r1], counts[r0:r1], np_max)


def _select(qi: torch.Tensor, wts: torch.Tensor, keys, R: int, topk: int, pos_dev: torch.Tensor,
            tokens: torch.Tensor, counts: torch.Tensor, np_max: int) -> None:
    """select_tokens for rows at pos_dev .. pos_dev + R - 1 over np_max scored tokens."""

    from .kv8 import Kv8
    from .kvq import KvQ

    H = wts.shape[1]                                            # wts [R, heads]: each row's per-head weights
    D = qi.shape[1] // H
    wscale = H ** -0.5
    rb = 16 if R >= 16 else R                                   # rows a scoring program: each key tile read once (same bits a row)
    scores = torch.empty((R, np_max), dtype=torch.float32, device=qi.device)
    kv8, qb = isinstance(keys, Kv8), keys.bits if isinstance(keys, KvQ) else 0
    ik, isc = (keys.codes, keys.scales) if kv8 or qb else (keys, keys)
    hq = keys.h if qb else isc
    _scores[(triton.cdiv(R, rb), triton.cdiv(np_max, 64))](qi, wts, wts.stride(0), ik, isc, hq, scores, pos_dev, R,
                                                           np_max, D ** -0.5, wscale, H=H,
                                                           HP=max(16, triton.next_power_of_2(H)), D=D, BT=64, RB=rb,
                                                           KV8=kv8, QB=qb, num_warps=4)
    width = tokens.shape[1]
    k = min(topk, np_max)
    if R >= RADIX_ROWS and tokens.stride(1) == 1:
        tokens.zero_()
        _radix_topk[(R,)](scores, scores.stride(0), tokens, tokens.stride(0), np_max, K=k, BLOCK=1024, num_warps=4)
    elif tokens.stride(1) == 1:
        tokens.zero_()
        _split_topk(scores, tokens, k, np_max)
    else:
        picked = torch.topk(_order_key(scores), k, dim=1, sorted=False).indices     # ties to the lower token
        picked = torch.sort(picked, dim=1).values                                   # ascending token order
        tokens.zero_()
        tokens[:, :k] = picked.to(torch.int32)
    del scores
    if width > k:
        tokens[:, k:] = -1
    # rows whose whole visible context fits the budget attend densely: their sparse pass is skipped
    q = pos_dev.to(torch.int64) + torch.arange(R, device=qi.device)
    counts.copy_(torch.where(q + 1 > topk, tokens.ne(-1).sum(1).clamp(max=width), torch.zeros_like(counts)))
