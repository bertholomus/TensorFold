"""RoPE: Ling's de-interleaved rotary form and our GPT-J table give the same scores.

The reference de-interleaves q and k (pairs to halves), applies rotate_half, and scores in that order;
our engine rotates the interleaved pairs in place (glm_moe_dsa's rope). Both are the same rotation up to a
fixed permutation of q and k, so every score must agree — here in fp32 to 1e-6 relative, at the positions
the design names (0, 1, 4095, 262143) with Ling's theta 6e6 on the 64-dim rope slice.

Run: python -m pytest tests/test_bailing_4_rope.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")

THETA = 6_000_000.0
HALF = 32                    # the rope slice's cos/sin pairs (qk_rope 64)
POSITIONS = (0, 1, 4095, 262143)


def _angles(pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin rows at one position: the GPT-J table's formula (theta ** (-2i / 64), phase = pos * inv)."""

    i = torch.arange(HALF, dtype=torch.float64)
    inv = torch.exp2(-(2.0 * i / (2 * HALF)) * math.log2(THETA))
    phase = float(pos) * inv
    return torch.cos(phase).float(), torch.sin(phase).float()


def _rotate_gptj(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Our engine's form: rows [..., 64] rotated in place, each pair (2i, 2i + 1) by angle i."""

    a, b = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = b * cos + a * sin
    return out


def _rotate_ling(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The reference's form (modeling 541-576): de-interleave to halves, rotate_half there.

    x [..., 64] -> [..., 2, 32] (evens first), rotate_half([x1, x2]) = [x1 c - x2 s, x2 c + x1 s].
    """

    v = x.view(*x.shape[:-1], HALF, 2).transpose(-2, -1)          # [..., 2, 32]: evens, odds
    x1, x2 = v[..., 0, :], v[..., 1, :]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


@pytest.mark.parametrize("pos", POSITIONS)
def test_the_two_forms_score_identically(pos):
    torch.manual_seed(pos)
    heads, rows = 4, 7
    q = torch.randn(rows, heads, 64)
    k = torch.randn(rows, heads, 64)
    cos, sin = _angles(pos)

    qg, kg = _rotate_gptj(q, cos, sin), _rotate_gptj(k, cos, sin)
    ql, kl = _rotate_ling(q, cos, sin), _rotate_ling(k, cos, sin)

    # the outputs are the same rotation up to the de-interleaving permutation (evens to the first half)
    perm = torch.arange(64)
    perm = torch.cat((perm[0::2], perm[1::2]))
    assert torch.equal(ql, qg[..., perm]) and torch.equal(kl, kg[..., perm])

    # and so the scores agree: the permutation cancels in the dot product
    scores_gptj = torch.einsum("rhd,rhd->rh", qg, kg)
    scores_ling = torch.einsum("rhd,rhd->rh", ql, kl)
    scale = scores_gptj.abs().clamp_min(1e-30)
    assert (scores_gptj - scores_ling).abs().max() <= 1e-6 * scale.max()


def test_scores_are_position_dependent_and_bounded():
    """Sanity: position 0 is identity, and the rotation preserves each pair's norm (a rotation, not a scale)."""

    torch.manual_seed(0)
    q = torch.randn(1, 1, 64)
    cos0, sin0 = _angles(0)
    assert torch.equal(_rotate_gptj(q, cos0, sin0), q)
    cos, sin = _angles(4095)
    rotated = _rotate_gptj(q, cos, sin)
    norms = (rotated[..., 0::2] ** 2 + rotated[..., 1::2] ** 2).sqrt()
    orig = (q[..., 0::2] ** 2 + q[..., 1::2] ** 2).sqrt()
    assert torch.allclose(norms, orig, rtol=1e-5)


def _kernel_check(what: str) -> None:
    """tests/bailing_kernel_checks.py in a child process: natively on a CUDA host, else in Triton's interpreter
    (TRITON_INTERPRET must be set before triton loads, so never in this process)."""

    import os
    import subprocess

    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = dict(os.environ)
    if dev == "cpu":
        env["TRITON_INTERPRET"] = "1"
    r = subprocess.run([sys.executable, str(REPO / "tests" / "bailing_kernel_checks.py"), what, dev], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-4000:] + r.stderr[-4000:]


def test_engine_rope_matches_the_reference_bit_for_bit():
    """bailing_v3/cuda/rope.py: the table equals the reference's bf16 cos / sin and the in-place rotation its bf16
    ops (de-interleave, q cos + rotate_half(q) sin, re-interleaved) bit for bit, for queries and the strided rope
    key, at positions 0, 1, 4095 and 262143; a window's rows equal one-row tables."""

    _kernel_check("rope")
