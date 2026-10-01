"""GLM-5.3's raw-token DSA selection: per-row top-index_topk lightning-indexer scores, exact ties, ascending order."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scores(QI, W, w_stride, IK, OUT, POS, R, NT, scale, wscale, H: tl.constexpr, HP: tl.constexpr,
            D: tl.constexpr, BT: tl.constexpr, RB: tl.constexpr):
    """Program (RB rows, token block): s_t = sum_h w_h relu(scale * qi_h . k_t) up to each row's position.

    The reference (exllamav3's dsa_indexer_scores) folds D_i**-0.5 and H_i**-0.5 into one scale and
    sums relu(q.k) * w_h over the heads in head order; the same order here keeps a row's bits fixed.
    """

    rb = tl.program_id(0)
    tb = tl.program_id(1)
    P = tl.load(POS)
    t = tb * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    hh = tl.arange(0, HP)
    hok = hh < H
    k = tl.load(IK + t[:, None].to(tl.int64) * D + d[None, :], mask=(t < (P + rb * RB + RB))[:, None],
                other=0.0).to(tl.bfloat16)                                             # [BT, D]
    for i in tl.static_range(RB):
        r = rb * RB + i
        if r < R:
            bound = P + r + 1
            q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None], other=0.0).to(tl.bfloat16)
            kr = tl.where((t < bound)[:, None], k, 0.0)
            dots = tl.dot(q, tl.trans(kr))                                             # [HP, BT] fp32
            w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
            sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
            sc = tl.where(t < bound, sc, float("-inf"))
            tl.store(OUT + r * NT + t, sc, mask=t < NT)


def _order_key(scores: torch.Tensor) -> torch.Tensor:
    """fp32 scores as int64 keys whose sort is score-ascending with ties to the lower token (Flash's trick)."""

    bits = (scores + 0.0).view(torch.int32)                    # + 0.0: -0 becomes +0, as the sort ties them
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)   # IEEE order as signed ints (negatives flipped)
    keys = ordered.to(torch.int64).bitwise_left_shift_(32)
    keys.bitwise_or_(0xFFFFFFFF - torch.arange(scores.shape[1], device=scores.device, dtype=torch.int64))
    return keys


def sparse_bucket(pos: int, R: int) -> int:
    """Tokens scored for rows pos .. pos + R - 1: the visible ones rounded up to a power of two (at least 2048)."""

    visible = pos + R
    return max(2048, 1 << (visible - 1).bit_length())


def select_tokens(qi: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, pos: int | None, R: int,
                  topk: int, pos_dev: torch.Tensor, *, tokens: torch.Tensor, counts: torch.Tensor,
                  bucket: int | None = None) -> None:
    """Each row's attended tokens [R, width] ascending (-1 padded) and their count past the dense limit.

    ``bucket`` fixes the scored token count for captured graphs; without it the visible count is
    rounded up to a power of two so the allocator reuses a few sizes. Scores, top-k membership and
    ascending order match the reference exactly: any global top-k member is in its own tile's top-k,
    ties go to the lower token, and a row below the dense limit keeps every token (count 0 marks it).
    """

    if qi.stride(0) != qi.shape[1] or wts.stride(1) != 1:
        raise ValueError("select_tokens: index queries must be contiguous rows, weights unit-stride columns")
    H = wts.shape[1]                                            # wts [R, heads]: each row's per-head weights
    D = qi.shape[1] // H
    wscale = H ** -0.5
    np_max = bucket if bucket is not None else sparse_bucket(int(pos) if pos is not None
                                                             else int(pos_dev.item()), R)
    np_max = min(np_max, keys.shape[0])
    scores = torch.empty((R, np_max), dtype=torch.float32, device=qi.device)
    _scores[(R, triton.cdiv(np_max, 64))](qi, wts, wts.stride(0), keys, scores, pos_dev, R, np_max,
                                          D ** -0.5, wscale, H=H, HP=max(16, triton.next_power_of_2(H)),
                                          D=D, BT=64, RB=1, num_warps=4)
    width = tokens.shape[1]
    k = min(topk, np_max)
    picked = torch.topk(_order_key(scores), k, dim=1, sorted=False).indices     # ties to the lower token
    picked = torch.sort(picked, dim=1).values                                   # ascending token order
    tokens.zero_()
    tokens[:, :k] = picked.to(torch.int32)
    if width > k:
        tokens[:, k:] = -1
    # rows whose whole visible context fits the budget attend densely: their sparse pass is skipped
    q = pos_dev.to(torch.int64) + torch.arange(R, device=qi.device)
    counts.copy_(torch.where(q + 1 > topk, tokens.ne(-1).sum(1).clamp(max=width), torch.zeros_like(counts)))
