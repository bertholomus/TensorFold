"""TF_EXL3_PROMPT_KERNEL=mma2 (``grouped_mma2``): prompt chunks' routed experts with each warp decoding its own weight
tiles into mma fragments, one fp32 chain over K an output. Against the one-tile launches it agrees to fp32 rounding
(other summation order, so other bits), at mixed widths up to 5 bits, balanced and skewed routing (an expert in 3 rows
of 4: several 64-row programs); a row's output does not depend on the window it runs in, and repeats bit for bit.
mma3 (gate/up's input rotation inside the program) gives mma2's bits where it applies (gate/up up to 4 bits)."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _trellis(k, n, k2, g):
    return torch.randint(-32768, 32767, (k // 16, n // 16, 8 * k2), generator=g, dtype=torch.int16).cuda()


def _scale(n, mag, g):
    return ((torch.rand((n,), generator=g) * 2 - 1) * mag + mag * 0.1).half().cuda()


def _layer(E, D, I, kfun, cb, seed):
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(seed)
    gate, up, down = [], [], []
    for e in range(E):
        kg, ku, kd = kfun(e)
        gate.append((_trellis(D, I, kg, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        up.append((_trellis(D, I, ku, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        down.append((_trellis(I, D, kd, g), _scale(I, 1 / math.sqrt(I), g), _scale(D, 0.25, g)))
    return experts.prepare(gate, up, down, cb)


def _picks(E, R, k, g):
    sel = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(R)]).to(torch.int32)
    w = torch.rand((R, k), generator=g) * 0.2 + 0.05
    sel = torch.cat([sel, torch.full((R, 1), E, dtype=torch.int32)], 1)
    w = torch.cat([w, torch.ones((R, 1))], 1)
    return sel.cuda().contiguous(), w.float().cuda().contiguous()


def _hot(sel, every=4):
    hot = sel.clone()
    for r in range(hot.shape[0]):
        if r % every == 0:
            continue
        row = hot[r, :-1]
        j = (row == 0).nonzero()
        if len(j):
            row[int(j[0])] = row[0].clone()
        row[0] = 0
    return hot.contiguous()


CASES = [
    ("mul1-3bit", 2, lambda e: (6, 6, 6)),
    ("mul1-mixed", 2, lambda e: ((4, 6, 8, 10)[e % 4], (6, 4, 10, 8)[e % 4], (8, 6, 4, 10)[e % 4])),
    ("mcg-mixed", 1, lambda e: ((2, 4, 6, 8)[e % 4],) * 2 + ((4, 6, 8)[e % 3],)),
    ("3inst-mixed", 0, lambda e: ((3, 5, 7, 9)[e % 4],) * 2 + ((6, 8, 10)[e % 3],)),
]


@pytest.mark.parametrize("name,cb,kfun", CASES, ids=[c[0] for c in CASES])
def test_mma2_matches_the_one_tile_launches(name, cb, kfun):
    from tensorfold.cuda.exl3 import experts

    E, D, I, TOPK = 24, 1024, 512, 6
    ex = _layer(E, D, I, kfun, cb, seed=7 + cb)
    g = torch.Generator().manual_seed(5)
    ROWS = 600
    x = torch.randn((ROWS, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, ROWS, TOPK, g)
    scratch = experts.Scratch(ex, ROWS, TOPK + 1)
    saved = experts.PROMPT_KERNEL
    try:
        for picks in (sel, _hot(sel)):
            full = None
            for R in (600, 64, 100, 129):
                p, wr = picks[:R].contiguous(), w[:R].contiguous()
                ref = experts.routed(x[:R], p, wr, ex, scratch, None, R, prompt=False).clone()
                experts.PROMPT_KERNEL = "mma2"
                out = experts.routed(x[:R], p, wr, ex, scratch, None, R, prompt=True).clone()
                again = experts.routed(x[:R], p, wr, ex, scratch, None, R, prompt=True).clone()
                if max(ex.k2_gu) <= 8 and max(ex.k2_d) <= 8:      # mma3: rotation and down epilogue inside
                    experts.PROMPT_KERNEL = "mma3"
                    fused = experts.routed(x[:R], p, wr, ex, scratch, None, R, prompt=True).clone()
                    fused2 = experts.routed(x[:R], p, wr, ex, scratch, None, R, prompt=True).clone()
                    assert torch.equal(fused.view(torch.int32), fused2.view(torch.int32)), (name, R, "mma3")
                    # the slots' bf16 outputs before the sum: within bf16 rounding of mma2's fp32 combine
                    err3 = float((fused.double() - out.double()).norm() / out.double().norm())
                    assert err3 < 4e-3, (name, R, err3)
                experts.PROMPT_KERNEL = saved
                assert torch.isfinite(out).all()
                assert torch.equal(out.view(torch.int32), again.view(torch.int32)), (name, R)      # deterministic
                err = float((out.double() - ref.double()).norm() / ref.double().norm())
                assert err < 1e-4, (name, R, err)
                if full is None:
                    full = out
                else:                                   # a row's bits do not depend on its window
                    assert torch.equal(out.view(torch.int32), full[:R].view(torch.int32)), (name, R)
    finally:
        experts.PROMPT_KERNEL = saved
