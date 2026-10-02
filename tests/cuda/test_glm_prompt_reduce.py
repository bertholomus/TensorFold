"""GLM-5.3 prompt chunks' rank partials by row shares (forward.gather in rows mode), simulated on one GPU: each rank sums
every rank's fp32 partial of its share of the rows (glue.rank_sum) and the bf16 sums gathered back in row order give
residual_add's result bit for bit, at 2, 3 and 4 ranks, row counts that split unevenly, with exact zeros, signed zeros,
cancelling partials and values on bf16 rounding ties."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _partials(world: int, R: int, D: int, g: torch.Generator) -> torch.Tensor:
    p = torch.randn((world, R, D), generator=g) * torch.logspace(-3, 3, D)[None, None, torch.randperm(D, generator=g)]
    if R >= 4:
        p[:, 0] = 0.0                                      # a row of exact zeros
        p[:, 1] = -0.0                                     # signed zeros: rank order decides the sign
        p[1:, 2] = -p[0, 2] / max(world - 1, 1)            # partials that cancel
        tie = torch.tensor([1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8, -(1.0 + 2.0 ** -8)])
        p[0, 3, :3], p[1:, 3, :3] = tie, 0.0               # sums exactly halfway between two bf16 values
    return p.float().cuda()


@pytest.mark.parametrize("world", (2, 3, 4))
def test_row_shares_give_residual_add_bits(world):
    from tensorfold.families.glm_moe_dsa.cuda import glue
    from tensorfold.families.glm_moe_dsa.cuda.forward import Rows, residual

    g = torch.Generator().manual_seed(world)
    for R, D in ((1, 6144), (7, 6144), (130, 3072), (1430, 6144), (4096, 6144)):    # widths residual_add covers
        p = _partials(world, R, D, g)
        x = (torch.randn((R, D), generator=g) * 3).to(torch.bfloat16).cuda()
        want = torch.empty_like(x)
        glue.residual_add(x, want, p)
        cut = [R * k // world for k in range(world + 1)]
        bg = torch.empty((R, D), dtype=torch.bfloat16, device="cuda")
        for k in range(world):                             # rank k: every rank's partial of its rows, summed
            rows = p[:, cut[k]:cut[k + 1]].contiguous()
            glue.rank_sum(rows.view(world, -1), bg[cut[k]:cut[k + 1]].view(-1))
        got = torch.empty_like(x)
        residual(x, got, Rows(bg))
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (world, R, D)
