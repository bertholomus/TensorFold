"""Same-bits checks of the concurrent rounds' kernel variants: each gives today's outputs bit for bit.

- group2 == group, entry for entry (uids, count, members), 1-64 rows, 12 / 256 / 288 experts.
- routed() with the expert-major order, the shared-memory staged programs (3 / 4 / 6 stages) and group2 equals routed()
  with today's settings: GLM's 3-bit shapes and mixed widths, 1-16 rows; y_unused leaves y alone, the same rows.
- decode-window absorb / expand on latent_rows.cu == latent's Triton kernels, 1-64 rows.
- the fused RDMA collect's residual arithmetic (rank_residual) == residual_add, edge values planted.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _trellis(k, n, k2, g):
    v = torch.randint(-32768, 32768, (k // 16, n // 16, 8 * k2), dtype=torch.int32, generator=g)
    return v.to(torch.int16).cuda().contiguous()


def _scale(n, mag, g):
    sign = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return (sign * (torch.rand((n,), generator=g) + 0.5) * mag).half().cuda()


def _layer(E, D, I, k2s, cb, seed):
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(seed)
    gate, up, down = [], [], []
    for e in range(E):
        kg, ku, kd = k2s[e]
        gate.append((_trellis(D, I, kg, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        up.append((_trellis(D, I, ku, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        down.append((_trellis(I, D, kd, g), _scale(I, 1 / math.sqrt(I), g), _scale(D, 0.25, g)))
    return experts.prepare(gate, up, down, cb)


def _picks(E, R, k, g, shared=True):
    sel = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(R)]).to(torch.int32)
    w = torch.rand((R, k), generator=g) * 0.2 + 0.05
    if shared:
        sel = torch.cat([sel, torch.full((R, 1), E, dtype=torch.int32)], 1)
        w = torch.cat([w, torch.ones((R, 1))], 1)
    return sel.cuda().contiguous(), w.float().cuda().contiguous()


@pytest.mark.parametrize("E", (12, 256, 288))
def test_group2_equals_group(E):
    from tensorfold.cuda.exl3 import experts

    ext = experts._ext()
    g = torch.Generator().manual_seed(E)
    slots = 9
    for R in (1, 2, 3, 4, 8, 13, 16, 32, 64):
        sel, _ = _picks(E, R, 8, g)
        maxu = min(R * slots, E)
        out = []
        for fn in (ext.group, ext.group2):
            u = torch.full((maxu,), -9, dtype=torch.int32, device="cuda")
            c = torch.zeros((1,), dtype=torch.int32, device="cuda")
            m = torch.full((maxu, R), -9, dtype=torch.int32, device="cuda")
            fn(sel, u, c, m, R, slots, E)
            out.append((u, c, m))
        (u0, c0, m0), (u1, c1, m1) = out
        n = int(c0)
        assert int(c1) == n, (E, R)
        assert torch.equal(u0[:n], u1[:n]) and torch.equal(m0[:n], m1[:n]), (E, R)


GLM = ("glm", 2, lambda e: (6, 6, 6))
MIX = ("mix", 2, lambda e: ((2, 3, 5, 6, 7, 8, 10, 12, 14, 16)[e % 10],) * 2 + ((3, 5, 7, 8, 16)[e % 5],))


@pytest.mark.parametrize("name,cb,kfun", (GLM, MIX), ids=("glm3", "mixed"))
def test_routed_variants_keep_every_bit(name, cb, kfun):
    from tensorfold.cuda.exl3 import experts

    E, D, I = (64, 1024, 512) if name == "glm" else (40, 512, 256)
    ex = _layer(E, D, I, [kfun(e) for e in range(E)], cb, seed=11)
    g = torch.Generator().manual_seed(5)
    saved = (experts.GROUPED_ORDER, experts.GROUP2, experts.GROUPED_STAGES, experts.DECODE_Y)
    try:
        for R in (1, 2, 4, 7, 16):
            x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
            sel, w = _picks(E, R, 8, g)
            sc = experts.Scratch(ex, 16, 9)
            experts.GROUPED_ORDER, experts.GROUP2, experts.GROUPED_STAGES, experts.DECODE_Y = 0, False, 0, True
            ref = experts.routed(x, sel, w, ex, sc, None, R, act_mode=experts.ACT_BF16).clone()
            ref_y = experts.routed(x, sel, None, ex, sc, None, R, act_mode=experts.ACT_BF16).clone()
            for order, g2, stages in ((1, False, 0), (0, True, 0), (1, True, 3), (1, True, 4), (0, True, 6)):
                experts.GROUPED_ORDER, experts.GROUP2, experts.GROUPED_STAGES = order, g2, stages
                out = experts.routed(x, sel, w, ex, sc, None, R, act_mode=experts.ACT_BF16)
                assert _same(out, ref), (name, R, order, g2, stages)
                y = experts.routed(x, sel, None, ex, sc, None, R, act_mode=experts.ACT_BF16)
                assert _same(y, ref_y), ("y", name, R, order, g2, stages)
            # a caller that never reads y: y is left alone (DECODE_Y off), the combined rows are the same
            experts.DECODE_Y = False
            sc.y.fill_(3.0)
            out = experts.routed(x, sel, w, ex, sc, None, R, act_mode=experts.ACT_BF16, y_unused=True)
            assert bool((sc.y == 3.0).all()), (name, R)
            sc.y.zero_()
            experts.DECODE_Y = True
            ref0 = experts.routed(x, sel, w, ex, sc, None, R, act_mode=experts.ACT_BF16)
            assert _same(out, ref0), ("y_unused", name, R)
    finally:
        experts.GROUPED_ORDER, experts.GROUP2, experts.GROUPED_STAGES, experts.DECODE_Y = saved


def test_decode_absorb_and_expand_on_latent_rows_keep_every_bit():
    from tensorfold.families.glm5_next.cuda import latent
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    H, QK, LW, DV = 16, 256, 512, 256
    g = torch.Generator().manual_seed(2)

    class A:
        pass

    a = A()
    a.wk = (torch.randn((H, QK, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    a.wk[:, 192:] = 0
    a.wv = (torch.randn((H, DV, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    a.lw, a.v_dim, a.heads, a.qk_dim = LW, DV, H, QK
    saved = mla_pe.DECODE_ABSORB
    try:
        for R in (1, 2, 3, 4, 8, 13, 16, 32, 64):
            q = torch.randn((R, H, QK), generator=g).to(torch.bfloat16).cuda()
            ol = torch.randn((R, H, LW), generator=g).to(torch.bfloat16).cuda()
            outs = []
            for mode in ("triton", "cuda"):
                mla_pe.DECODE_ABSORB = mode
                oa = torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda")
                ov = torch.empty((R, H, DV), dtype=torch.bfloat16, device="cuda")
                mla_pe.absorb_q(q, a, oa)
                mla_pe.expand_v(ol, a, ov)
                outs.append((oa, ov))
            ref_a = latent.absorb_q(q, a, torch.empty_like(outs[0][0]))
            assert _same(outs[0][0], outs[1][0]) and _same(outs[0][0], ref_a), ("absorb", R)
            assert _same(outs[0][1], outs[1][1]), ("expand", R)
    finally:
        mla_pe.DECODE_ABSORB = saved


def test_fused_collect_residual_arithmetic_equals_residual_add():
    try:
        from tensorfold.cuda import rdma

        ext = rdma._ext()
    except Exception as exc:                                       # noqa: BLE001  (no RDMA verbs in this container)
        pytest.skip(f"rdma extension unavailable: {exc}")
    from tensorfold.families.glm5_next.cuda import glue

    g = torch.Generator().manual_seed(3)
    for world in (2, 4):
        for R in (1, 3, 16):
            for scale in (1e-3, 1.0, 30.0):
                parts = (torch.randn((world, R, 6144), generator=g) * scale).cuda()
                parts[:, :, :8] = torch.tensor([0.0, -0.0, 1e-38, -1e-38, 3.4e38, -3.4e38, 1e-45, 65504.0])
                x = (torch.randn((R, 6144), generator=g) * 4).to(torch.bfloat16).cuda()
                a, b = torch.empty_like(x), torch.empty_like(x)
                glue.residual_add(x, a, parts)
                ext.rank_residual(parts, x, b)
                torch.cuda.synchronize()
                assert _same(a, b), (world, R, scale)
