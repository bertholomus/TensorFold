"""Ling-3.1-flash's router: grouped sigmoid top-k (the reference's BailingMoeV3Gate), one row a program.

For each row, with fp32 router logits (glue.router):

- scores ``s = sigmoid(logits)``: torch.sigmoid, the reference's own op, so the scores carry its bits;
- ``sb = s + expert_bias`` (the checkpoint's fp32 bias, never rounded to bf16);
- a group's score is the sum of its two largest ``sb`` (``n_group`` groups of E / n_group experts);
- keep the ``topk_group`` groups with the largest group scores;
- the ``top_k`` experts with the largest ``sb`` inside the kept groups, in descending order;
- weights ``s[pick] / (sum + 1e-20) * routed_scale`` on the unbiased scores, the sum taken in pick order and
  the quotient rounded once (IEEE division); the reference normalizes whenever top_k > 1, whatever
  norm_topk_prob says, and so does this.

Ties are deterministic: the lower index wins at the group step and at the expert step (the reference's
torch.topk leaves them unspecified). Each row is one program over its own values only, so a window's rows get
the bits of the serial one-row steps. The shared expert is appended as slot ``top_k`` (id E, weight 1): the
slot layout glm_moe_dsa's routed-experts launch reads (picks >= E are skipped there).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_topk(S, BIAS, PICK, WTS, scale, NE: tl.constexpr, NG: tl.constexpr, GS: tl.constexpr,
                  TOPG: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr, SLOTP: tl.constexpr,
                  NORM: tl.constexpr):
    """Row r: scores S[r] [NE] fp32 as [NG groups, GS experts] -> PICK[r] / WTS[r] [SLOTS] (the shared slot last)."""

    r = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, NG)
    e = tl.arange(0, GS)
    flat = g[:, None] * GS + e[None, :]                                    # [NG, GS] expert ids
    s = tl.load(S + r * NE + flat)
    sb = s + tl.load(BIAS + flat)
    neg = float("-inf")
    # each group's top-2 sum: the largest value, then the largest of the rest (a tie gives the same value twice)
    m1 = tl.max(sb, axis=1)
    i1 = tl.min(tl.where(sb == m1[:, None], e[None, :], GS), axis=1)
    m2 = tl.max(tl.where(e[None, :] == i1[:, None], neg, sb), axis=1)
    gsum = m1 + m2
    # the TOPG best groups, the lower group index first on equal scores
    keep = tl.zeros((NG,), dtype=tl.int32)
    left = gsum
    for _ in tl.static_range(TOPG):
        mx = tl.max(left, axis=0)
        gi = tl.min(tl.where(left == mx, g, NG), axis=0)
        keep = tl.where(g == gi, 1, keep)
        left = tl.where(g == gi, neg, left)
    choice = tl.where(keep[:, None] != 0, sb, neg)
    k = tl.arange(0, SLOTP)
    picks = tl.zeros((SLOTP,), dtype=tl.int32)
    wts = tl.zeros((SLOTP,), dtype=tl.float32)
    total = 0.0
    for j in tl.static_range(TOPK):
        mx = tl.max(tl.max(choice, axis=1), axis=0)
        idx = tl.min(tl.min(tl.where(choice == mx, flat, NE), axis=1), axis=0)
        sj = tl.sum(tl.sum(tl.where(flat == idx, s, 0.0), axis=1), axis=0)   # s[idx] (one non-zero term)
        picks = tl.where(k == j, idx, picks)
        wts = tl.where(k == j, sj, wts)
        total = total + sj
        choice = tl.where(flat == idx, neg, choice)
    if NORM:
        wts = tl.math.div_rn(wts, total + tl.full((SLOTP,), 1e-20, tl.float32))
    wts = wts * scale
    picks = tl.where(k == TOPK, NE, picks)
    wts = tl.where(k == TOPK, 1.0, wts)
    tl.store(PICK + r * SLOTS + k, picks, mask=k < SLOTS)
    tl.store(WTS + r * SLOTS + k, wts, mask=k < SLOTS)


def check_shape(experts: int, n_group: int, topk_group: int, top_k: int) -> None:
    """The kernel's shape rules: power-of-two groups and group sizes, enough experts in the kept groups."""

    if experts % n_group:
        raise ValueError(f"route: {experts} experts do not split into {n_group} groups")
    gs = experts // n_group
    for name, v in (("n_group", n_group), ("experts a group", gs)):
        if v & (v - 1):
            raise ValueError(f"route: {name} {v} is not a power of two")
    if not 1 <= topk_group <= n_group or top_k > topk_group * gs or gs < 2:
        raise ValueError(f"route: top-{top_k} of {topk_group} kept groups of {gs} experts cannot be picked")


def select(scores: torch.Tensor, bias: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor, *, n_group: int,
           topk_group: int, top_k: int, scale: float) -> None:
    """scores [R, E] fp32 (sigmoid already applied), bias [E] fp32 -> pick [R, top_k + 1] int32, wts fp32."""

    rows, experts = scores.shape
    check_shape(experts, n_group, topk_group, top_k)
    if scores.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("route: scores and expert_bias are fp32")
    if not scores.is_contiguous() or pick.shape[1] != top_k + 1 or wts.shape[1] != top_k + 1:
        raise ValueError("route: contiguous scores, top_k + 1 slots (the shared expert last)")
    if rows == 0:
        return
    _grouped_topk[(rows,)](scores, bias, pick, wts, float(scale), NE=experts, NG=n_group, GS=experts // n_group,
                           TOPG=topk_group, TOPK=top_k, SLOTS=top_k + 1, SLOTP=triton.next_power_of_2(top_k + 1),
                           NORM=top_k > 1, num_warps=4)


def route(logits: torch.Tensor, bias: torch.Tensor, scores: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor, *,
          n_group: int, topk_group: int, top_k: int, scale: float) -> None:
    """fp32 router logits [R, E] -> scores (sigmoid, into ``scores``), then the grouped selection (capturable)."""

    torch.sigmoid(logits, out=scores)
    select(scores, bias, pick, wts, n_group=n_group, topk_group=topk_group, top_k=top_k, scale=scale)
