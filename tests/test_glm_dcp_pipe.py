"""dcp.py's pipe host logic (Ported-from MiaAI-Lab patch 0131, adapted): the owner split and the
part sizes, and the selection the owner path must give — the same tokens / counts as dcp.merge over
the all-gathered candidates (positions are unique, so the radix threshold's keep set is the top-k).

The triton kernels (_threshold / _keep) run on the nodes (tests/cuda/test_glm_dcp.py's bit checks);
here their contract is checked against a pure-torch mirror of the same rule, on CPU, including the
key flip (our candidate keys are float-ordered as signed int64; the kernels compare uint64).
"""

from __future__ import annotations

import torch

from tensorfold.families.glm_moe_dsa.cuda import dcp

G, TOPK = 4, 64
SENTINEL = -9223372036854775807 - 1


def candidate_keys(seed: int, n: int, rank: int, real_frac: float = 0.9) -> torch.Tensor:
    """One rank's [n, topk] candidates in _candidates' layout: (float-ordered bits << 32) |
    (0xFFFFFFFF - position); some slots the sentinel (no candidate). The positions are the rank's
    own (p % G == rank, unique a row), as the cache's layout makes them — so the G ranks' keys of a
    row never tie, which is what makes the threshold's keep set exactly the top-k."""

    g = torch.Generator().manual_seed(seed * 8 + rank)
    scores = torch.randn((n, TOPK), generator=g)
    order = (scores + 0.0).view(torch.int32)
    order = torch.where(order < 0, order ^ 0x7FFFFFFF, order).to(torch.int64)
    pos = (torch.argsort(torch.rand((n, TOPK), generator=g), dim=1) * G + rank).to(torch.int64)
    keys = (order << 32) | (0xFFFFFFFF - pos)
    drop = torch.rand((n, TOPK), generator=g) > real_frac
    return torch.where(drop, torch.full_like(keys, SENTINEL), keys)


def merge_reference(every: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """dcp.merge's tokens / counts for one rank (its own slots of the global top-k, ascending, -1)."""

    n = every.shape[1]
    top = torch.topk(every.permute(1, 0, 2).reshape(n, G * TOPK), TOPK, dim=1, sorted=False).values
    p = 0xFFFFFFFF - (top & 0xFFFFFFFF)
    own = ((p % G) == rank) & (top != SENTINEL)
    slot = torch.where(own, p // G, torch.full_like(p, 1 << 40))
    tokens = torch.sort(slot, dim=1).values.clamp(max=(1 << 31) - 1).to(torch.int32)
    cnt = own.sum(1).to(torch.int32)
    tokens[:, :TOPK].masked_fill_(torch.arange(TOPK)[None, :] >= cnt[:, None], -1)
    return tokens, cnt


def owner_reference(all_cands: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The owner path's rule, in torch, per row: the row's threshold the K-th largest of its G
    ranks' candidates (uint64 order — the keys flipped by the top bit, the sentinel the least), the
    keep this rank's real candidates at or above it, slots ascending, -1 past the count. The
    owner-block padding of owner_select (fixed G * M sends) only ever adds sentinel rows past n,
    which the keep empties and the [:n] slice drops."""

    n = all_cands.shape[1]
    # the rule in the keys' own signed order (the same order the kernels' uint64 flip gives): the
    # threshold the K-th largest of the row's G ranks' candidates, the keep the real ones at/above
    pools = all_cands.permute(1, 0, 2).reshape(n, G * TOPK)
    th = pools.sort(dim=1, descending=True).values[:, TOPK - 1]
    keep = (all_cands[rank] >= th[:, None]) & (all_cands[rank] != SENTINEL)
    big = torch.iinfo(torch.int64).max
    pos = torch.where(keep, 0xFFFFFFFF - (all_cands[rank] & 0xFFFFFFFF), big)
    pos = pos.sort(dim=1).values[:, :TOPK]
    slots = torch.where(pos != big, pos // G, -1).to(torch.int32)
    cnt = keep.sum(1).to(torch.int32)
    return slots, cnt


def test_owner_split_and_parts() -> None:
    assert dcp.owner_rows(1, G) == 1
    assert dcp.owner_rows(1000, G) == 250
    assert dcp.owner_rows(1001, G) == 251
    for R in (1, 2, 511, 512, 513, 1024, 2048, 8192):
        parts = dcp.cp_parts(R, 512)
        assert parts[0][0] == 0 and parts[-1][1] == R
        assert all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))
        assert all(0 < r1 - r0 <= 512 for r0, r1 in parts)
        if len(parts) > 1:
            assert parts[0][1] - parts[0][0] == 256 and R - parts[-1][0] <= 512
    assert dcp.cp_parts(100, 512) == [(0, 100)]


def test_flip_gives_the_uint64_score_order() -> None:
    """Our candidate keys sort as the scores in signed int64 (dcp.merge's torch.topk); flipped by the
    top bit they sort as the scores in uint64 (the kernels' order)."""

    import numpy as np

    # (0.0 and -0.0 are left out: _order_key ties them to +0 and the position breaks the tie, which
    # a bare float argsort does not)
    scores = torch.tensor([1.0, -1.0, 3.5, -2.0, 5e-40, -5e-40, 7.25, -0.25], dtype=torch.float32)
    keys = (scores + 0.0).view(torch.int32)
    keys = torch.where(keys < 0, keys ^ 0x7FFFFFFF, keys).to(torch.int64) << 32
    keys = keys | (0xFFFFFFFF - torch.arange(len(scores), dtype=torch.int64))
    order = scores.argsort().tolist()
    assert keys.argsort().tolist() == order                     # the signed order the topk uses
    flipped = (keys ^ (1 << 63)).numpy().view(np.uint64)
    assert flipped.argsort().tolist() == order                  # the uint64 order the kernels use


def test_owner_path_is_merge() -> None:
    """The owner rule gives dcp.merge's tokens / counts for every rank, for whole and ragged row
    blocks, dense and thin candidates alike."""

    for seed, n in ((1, 1000), (2, 1001), (3, 251), (4, 7), (5, 4 * 251)):
        every = torch.stack([candidate_keys(seed * 10 + r, n, r) for r in range(G)])
        for rank in range(G):
            want_t, want_c = merge_reference(every, rank)
            got_t, got_c = owner_reference(every, rank)
            assert torch.equal(got_t, want_t), f"rank {rank}, n {n}: tokens differ"
            assert torch.equal(got_c, want_c), f"rank {rank}, n {n}: counts differ"
