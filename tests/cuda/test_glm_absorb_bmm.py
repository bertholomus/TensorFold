"""TF_GLM_ABSORB=bmm: a prompt chunk's absorb (q_nope . W_UK) and expand (o_lat . W_UV) as one batched tensor-core
matmul a head match a float64 reference as closely as the exact-sum kernels do (fp32 sums, bf16 out), at GLM-5.3's
TP4 shapes (16 heads, 256-wide heads, 512-wide latent)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

H, D, LW, DV = 16, 256, 512, 256


@pytest.mark.parametrize("R", [65, 300, 2048])
def test_bmm_absorb_and_expand_match_the_reference(R, monkeypatch):
    from types import SimpleNamespace

    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(R)
    q = (torch.randn((R, H, D), generator=g) * 0.5).to(torch.bfloat16).cuda()
    o = (torch.randn((R, H, LW), generator=g) * 0.5).to(torch.bfloat16).cuda()
    wk = (torch.randn((H, D, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    wv = (torch.randn((H, DV, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    a = SimpleNamespace(wk=wk, wv=wv, lw=LW, v_dim=DV)
    want_a = torch.einsum("rhd,hdn->rhn", q.double(), wk.double())
    want_e = torch.einsum("rhk,hnk->rhn", o.double(), wv.double())
    for mode in ("cuda", "bmm"):
        monkeypatch.setattr(mla_pe, "ABSORB", mode)
        if mode == "cuda" and not mla_pe._exact_shapes(a):
            continue
        qa = torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda")
        ex = torch.empty((R, H, DV), dtype=torch.bfloat16, device="cuda")
        mla_pe.absorb_q(q, a, qa)
        mla_pe.expand_v(o, a, ex)
        ea = float((qa.double() - want_a).norm() / want_a.norm())
        ee = float((ex.double() - want_e).norm() / want_e.norm())
        assert ea < 4e-3 and ee < 4e-3, (mode, ea, ee)            # bf16 output rounding: ~2e-3
