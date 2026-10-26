"""The grouped top-k router: the routing spec's torch transcription against a direct
transcription of the reference's BailingMoeV3Gate (modeling 368-406) on 64K random rows, plus the tie rule.

The select kernel (route.py) is held against the same spec by tests/bailing_kernel_checks.py: identical
picks, weights bit-equal to the pick-order fp32 transcription and within 9 eps of the reference's (any order of
the eight-term sum). The oracle's real layer-4 and layer-68 rows: test_real_router_rows_match_the_dumps.

Run: python -m pytest tests/test_bailing_6_router.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")

E, HIDDEN, N_GROUP, TOPK_GROUP, TOP_K, SCALE = 512, 4096, 8, 4, 8, 2.5


def spec_select(sb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The routing spec: group score = the sum of each group of 64's top-2; keep the top-4 groups; the top-8
    experts by sb inside them. Deterministic ties: the lower index wins at both steps (a stable descending
    sort). Returns (indices, order) — the weights are applied to the unbiased scores by spec_weights."""

    rows = sb.shape[0]
    groups = sb.view(rows, N_GROUP, E // N_GROUP)
    # top-2 per group, lower index first on ties: a stable sort of the negated scores
    order = torch.argsort(-groups, dim=-1, stable=True)[..., :2]
    group_scores = torch.gather(groups, -1, order).sum(-1)                     # [rows, N_GROUP]
    keep = torch.argsort(-group_scores, dim=-1, stable=True)[..., :TOPK_GROUP]  # the top-4 groups
    mask = torch.zeros_like(group_scores, dtype=torch.bool)
    mask.scatter_(1, keep, True)
    score_mask = mask.unsqueeze(-1).expand(rows, N_GROUP, E // N_GROUP).reshape(rows, E)
    masked = torch.where(score_mask, sb, torch.full_like(sb, float("-inf")))
    top = torch.argsort(-masked, dim=-1, stable=True)[..., :TOP_K]
    return top, masked


def spec_weights(scores: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
    """s[top8] / (sum + 1e-20) * 2.5 on the unbiased sigmoid scores (modeling 402-406)."""

    s = torch.gather(scores, 1, top)
    return s / (s.sum(-1, keepdim=True) + 1e-20) * SCALE


def reference_select(logits: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BailingMoeV3Gate.forward verbatim (modeling 394-406): fp32 logits, sigmoid, +bias, group_limited_topk."""

    scores = torch.sigmoid(logits.float())
    scores_for_routing = scores + bias
    num_tokens = scores.shape[0]
    group_scores = scores_for_routing.view(num_tokens, N_GROUP, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (group_mask.unsqueeze(-1).expand(num_tokens, N_GROUP, E // N_GROUP).reshape(num_tokens, -1))
    masked_scores = scores_for_routing.masked_fill(~score_mask.bool(), float("-inf"))
    probs, top_indices = torch.topk(masked_scores, k=TOP_K, dim=-1)
    scores = torch.gather(scores, dim=1, index=top_indices)
    topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    return top_indices, topk_weight * SCALE


def test_spec_matches_the_reference_on_random_rows():
    torch.manual_seed(11)
    logits = torch.randn(64_000, E)
    bias = torch.randn(E) * 0.01
    scores = torch.sigmoid(logits.float())
    sb = scores + bias
    top_spec = spec_select(sb)[0]
    idx_ref, w_ref = reference_select(logits, bias)
    assert torch.equal(torch.sort(top_spec, dim=-1).values, torch.sort(idx_ref, dim=-1).values)
    w_spec = spec_weights(scores, top_spec)
    gathered_ref = torch.gather(torch.zeros_like(sb).scatter(1, idx_ref, w_ref), 1, top_spec)
    # the two transcriptions sum the same eight values in different orders: within a few fp32 ulp
    assert (w_spec - gathered_ref).abs().max() <= 16 * torch.finfo(torch.float32).eps * SCALE


def test_ties_take_the_lower_index():
    """All-equal scores: the groups and experts keep index order (the spec's determinism rule)."""

    sb = torch.zeros(3, E)
    top, _ = spec_select(sb)
    assert torch.equal(top, torch.arange(TOP_K).expand(3, -1))
    # and a two-way tie at the group boundary keeps the lower group
    sb = torch.zeros(1, E)
    sb[0, 63] = 1.0                       # group 0's top-2 sum = 1
    sb[0, 127] = 1.0                      # group 1's top-2 sum = 1 (its top-1; the second is 0)
    kept, _ = spec_select(sb)
    masked_groups = set((kept[0] // (E // N_GROUP)).tolist())
    assert 0 in masked_groups             # the tie at group score 1 keeps the lower group's expert


def test_route_kernel_against_the_spec_and_the_reference():
    """bailing_v3/cuda/route.py (in a child process: natively on CUDA, else Triton's interpreter): picks equal the
    stable-sort spec in order, weights equal the pick-order fp32 transcription bit for bit and lie within 9 eps of
    the reference's (the sum-order bound); the shared slot is (E, 1); rows are window-independent; ties keep the
    lower index."""

    import os
    import subprocess

    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = dict(os.environ)
    if dev == "cpu":
        env["TRITON_INTERPRET"] = "1"
    r = subprocess.run([sys.executable, str(REPO / "tests" / "bailing_kernel_checks.py"), "route", dev], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-4000:] + r.stderr[-4000:]




def test_real_router_rows_match_the_dumps():
    """The oracle's real layer-4 and layer-68 router rows (oracle/o2/router_L4.npz, router_L68.npz), in a child
    process like the kernel check: natively on CUDA, in Triton's interpreter on a CPU host. Per dumped row the same
    expert set as the picks (order free: torch.topk leaves ties unordered) and weights within 9 eps of the dump's,
    matched by expert id. The scores are the reference gate's own fp32 scores from the dump (else sigmoid of its
    logits, else F.linear on the bank's gate weight); the bias the dump's fp32 expert_bias or the bank's, never
    rounded. Skips cleanly until the dumps land; a failing row prints its nine largest sigmoid+bias values."""

    import os
    import subprocess

    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = dict(os.environ)
    if dev == "cpu":
        env["TRITON_INTERPRET"] = "1"
    r = subprocess.run([sys.executable, str(REPO / "tests" / "bailing_kernel_checks.py"), "router_rows", dev],
                       env=env, capture_output=True, text=True, timeout=1800)
    if "no router dumps under" in (r.stdout + r.stderr):
        pytest.skip("no router dumps under LING_ORACLE_O2")
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-4000:] + r.stderr[-4000:]
    print(r.stdout[-2000:])
