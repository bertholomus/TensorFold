"""GLM-5.3 prompt chunks' absorb (q_nope . W_UK) and expand (o_lat . W_UV) split their rows over programs
(``mla_pe.absorb_q`` / ``expand_v``, Triton row blocks and latent_rows.cu): every row's output is bit-identical to
latent's one-program-per-column-block kernels, at GLM-5.3's TP4 shapes (16 heads, 256 -> 512 -> 256), for windows of
17 to 4,096 rows."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@pytest.mark.parametrize("R", (17, 64, 65, 300, 2016, 4096))
def test_row_blocks_give_latent_bits(R):
    from tensorfold.families.glm5_next.cuda import latent
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(R)
    H, D, LW, DV = 16, 256, 512, 256
    a = latent.AbsorbW((torch.randn((H, D, LW), generator=g) * 0.05).cuda(),
                       (torch.randn((H, DV, LW), generator=g) * 0.05).cuda())
    q = (torch.randn((R, H, D), generator=g) * 2).to(torch.bfloat16).cuda()
    ol = (torch.randn((R, H, LW), generator=g) * 2).to(torch.bfloat16).cuda()
    want_qa = latent.absorb_q(q, a, torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda"))
    want_v = latent.expand_v(ol, a, torch.empty((R, H, DV), dtype=torch.bfloat16, device="cuda"))
    saved = mla_pe.ABSORB
    try:
        for kind in ("triton", "cuda"):
            mla_pe.ABSORB = kind
            got_qa = mla_pe.absorb_q(q, a, torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda"))
            got_v = mla_pe.expand_v(ol, a, torch.empty((R, H, DV), dtype=torch.bfloat16, device="cuda"))
            assert torch.equal(got_qa.view(torch.int16), want_qa.view(torch.int16)), (R, kind)
            assert torch.equal(got_v.view(torch.int16), want_v.view(torch.int16)), (R, kind)
    finally:
        mla_pe.ABSORB = saved
