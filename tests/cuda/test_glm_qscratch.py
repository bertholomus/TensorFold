"""TF_GLM_QSCRATCH: a prompt chunk over a quantized latent (kvq) dequantizes the slots it can see once into an fp16
scratch (mla_pe.unpack_q) and its attention kernels read those rows: unpack_q holds exactly the rotated-domain values
the readers build in registers, and the dense pass and the sparse prompt kernel give the same bits with the scratch as
without (prompt == decode is test_glm_kvq's); the bf16 mode (TF_GLM_QSCRATCH=2) holds the latent rotated back."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

H, LW, PW = 16, 512, 64


def latent(n: int, g: torch.Generator) -> torch.Tensor:
    gain = torch.rand((LW,), generator=g) * 3 + 0.05
    x = torch.randn((n, LW), generator=g) * gain
    x[::97] *= 40
    x[3] = 0                                                   # an all-zero row: zero scales, signed-zero values
    return x.to(torch.bfloat16).cuda()


def plane_of(x: torch.Tensor, bits: int, extra: int = 0):
    from tensorfold.families.glm_moe_dsa.cuda import kvq

    plane = kvq.KvQ(x.shape[0] + extra, LW, bits, "cuda")
    kvq.write(x, plane, torch.zeros((1,), dtype=torch.int32, device="cuda"))
    return plane


def scratch_of(plane, n: int) -> torch.Tensor:
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    out = torch.full((plane.shape[0], LW), float("nan"), dtype=torch.float16, device="cuda")
    return mla_pe.unpack_q(plane, n, out)


@pytest.mark.parametrize("bits", (8, 6, 5, 4))
def test_unpack_holds_the_tiles_values(bits):
    from tensorfold.families.glm_moe_dsa.cuda import kvq

    g = torch.Generator().manual_seed(bits)
    n = 1000
    plane = plane_of(latent(n, g), bits, extra=100)
    got = scratch_of(plane, n - 7)
    want = kvq.dequantize_reference(plane.codes[:n - 7], plane.scales[:n - 7], bits, LW, rotated=True).to(torch.float16)
    assert torch.equal(got[:n - 7].view(torch.int16), want.view(torch.int16))
    assert bool(got[n - 7:].isnan().all())                     # slots past n untouched


@pytest.mark.parametrize("bits", (5, 4, 8))
@pytest.mark.parametrize("R,pos", [(64, 5000), (129, 2500), (300, 1800)])
def test_sparse_prompt_kernels_same_bits_with_the_scratch(bits, R, pos):
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(bits * 1000 + R * 11 + pos)
    n = pos + R
    plane = plane_of(latent(n, g), bits)
    pc = (torch.randn((n, PW), generator=g) * 2).to(torch.bfloat16).cuda()
    topk = 2048
    tokens = torch.full((R, topk + 1), -1, dtype=torch.int32, device="cuda")
    counts = torch.zeros((R,), dtype=torch.int32, device="cuda")
    for r in range(R):
        vis = pos + r + 1
        if vis > topk:
            take = topk if r % 5 else topk - 37 * (r % 3)             # some short lists: partial last tiles
            tokens[r, :take] = torch.randperm(vis, generator=g)[:take].sort().values.to(torch.int32).cuda()
            counts[r] = take
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    junk = torch.randn((R, H, LW), generator=g).to(torch.bfloat16).cuda()
    a, b = junk.clone(), junk.clone()
    mla_pe.sparse_attention(qa, qp, plane, pc, tokens, counts, a, 256 ** -0.5)
    mla_pe.sparse_attention(qa, qp, plane, pc, tokens, counts, b, 256 ** -0.5, qscratch=scratch_of(plane, n))
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("bits", (5, 8))
@pytest.mark.parametrize("R,pos", [(300, 1200), (64, 0), (129, 1919)])
def test_dense_pass_same_bits_with_the_scratch(bits, R, pos):
    from tensorfold.families.glm5_next.cuda.latent import LatentScratch, chunks_for
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(bits * 1000 + R * 7 + pos)
    n = pos + R
    plane = plane_of(latent(n, g), bits)
    pc = (torch.randn((n, PW), generator=g) * 2).to(torch.bfloat16).cuda()
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    s = LatentScratch(R, H, chunks_for(n + 512), "cuda", lw=LW, part_rows=min(R, 256))
    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    outs = []
    for qs in (None, scratch_of(plane, n)):
        outs.append(mla_pe.attention(qa, qp, plane, pc, at, s, scale=256 ** -0.5, nch=chunks_for(n),
                                     out=torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda"), qscratch=qs))
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


def test_decode_windows_ignore_the_scratch():
    """Windows narrower than a prompt chunk (decode, verify) keep their chunk programs: the scratch is not read."""

    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(9)
    R, pos, n = 4, 6000, 6004
    plane = plane_of(latent(n, g), 5)
    pc = (torch.randn((n, PW), generator=g) * 2).to(torch.bfloat16).cuda()
    tokens = torch.stack([torch.randperm(pos + r + 1, generator=g)[:2048].sort().values for r in range(R)])
    tokens = torch.cat([tokens, torch.full((R, 1), -1)], 1).to(torch.int32).cuda()
    counts = torch.full((R,), 2048, dtype=torch.int32, device="cuda")
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    bad = torch.full((n, LW), float("nan"), dtype=torch.float16, device="cuda")         # would poison every score
    a = torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    mla_pe.sparse_attention(qa, qp, plane, pc, tokens, counts, a, 256 ** -0.5)
    mla_pe.sparse_attention(qa, qp, plane, pc, tokens, counts, b, 256 ** -0.5, qscratch=bad)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("bits", (5, 8))
def test_bf16_unpack_holds_the_latent(bits):
    """TF_GLM_QSCRATCH=2: the scratch holds bf16 latent rows rotated back (the plane's own values, bf16-rounded)."""

    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(bits + 50)
    n = 1000
    plane = plane_of(latent(n, g), bits)
    out = torch.empty((n, LW), dtype=torch.bfloat16, device="cuda")
    mla_pe.unpack_q(plane, n, out)
    want = plane.dequant(n)                                    # fp32 from a float64 rotation
    # bf16 rounding (2^-9 relative on average) over the fp16 tile values' own rounding
    err = (out.float() - want).norm(dim=1) / want.norm(dim=1).clamp(min=1e-6)
    assert float(err.max()) < 4e-3, float(err.max())


@pytest.mark.parametrize("bits", (5, 4))
@pytest.mark.parametrize("R,pos", [(64, 5000), (300, 1800)])
def test_bf16_scratch_attention_matches_the_reference(bits, R, pos):
    from tensorfold.families.glm5_next.cuda.latent import LatentScratch, chunks_for
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(bits * 100 + R + pos)
    n = pos + R
    plane = plane_of(latent(n, g), bits)
    rows = plane.dequant(n)
    pc = (torch.randn((n, PW), generator=g) * 2).to(torch.bfloat16).cuda()
    topk = 2048
    tokens = torch.full((R, topk + 1), -1, dtype=torch.int32, device="cuda")
    counts = torch.zeros((R,), dtype=torch.int32, device="cuda")
    for r in range(R):
        if pos + r + 1 > topk:
            tokens[r, :topk] = torch.randperm(pos + r + 1, generator=g)[:topk].sort().values.to(torch.int32).cuda()
            counts[r] = topk
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    scr = torch.empty((n, LW), dtype=torch.bfloat16, device="cuda")
    mla_pe.unpack_q(plane, n, scr)
    out = torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda")
    s = LatentScratch(R, H, chunks_for(n + 512), "cuda", lw=LW, part_rows=min(R, 256))
    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    mla_pe.attention(qa, qp, plane, pc, at, s, scale=256 ** -0.5, nch=chunks_for(n), out=out, qscratch=scr)
    mla_pe.sparse_attention(qa, qp, plane, pc, tokens, counts, out, 256 ** -0.5, qscratch=scr)
    for r in range(0, R, max(1, R // 7)):
        sel = tokens[r, :topk].long() if int(counts[r]) else torch.arange(pos + r + 1, device="cuda")
        k = rows[sel].double()
        sc = (qa[r].double() @ k.T + qp[r].double() @ pc[sel].double().T) * 256 ** -0.5
        want = torch.softmax(sc, dim=-1) @ k
        err = float((out[r].double() - want).norm() / want.norm())
        assert err < 8e-3, (r, err)                            # the bf16 cache's 6e-3 plus the latent's bf16 rounding
