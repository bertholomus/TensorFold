"""GLM-5.3's quantized latent planes (TF_GLM_KV=q8 / q6 / q5 / q4, ``kvq``: exllamav3's MLA cache format): the writer
quantizes like the torch reference, the reference dequantizer inverts the packing, and the attention kernels over
packed rows (dense, sparse for decode windows and prompt chunks) match a float64 attention over the dequantized rows,
with the prompt kernel bit-equal to the decode windows' chunk programs + merge (prompt == decode)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

H, LW, PW = 16, 512, 64
BITS = (8, 6, 5, 4)


def latent(n: int, g: torch.Generator) -> torch.Tensor:
    gain = torch.rand((LW,), generator=g) * 3 + 0.05
    x = torch.randn((n, LW), generator=g) * gain
    x[::97] *= 40
    if n > 3:
        x[3] = 0
    return x.to(torch.bfloat16).cuda()


def plane_of(x: torch.Tensor, bits: int):
    from tensorfold.families.glm_moe_dsa.cuda import kvq

    plane = kvq.KvQ(x.shape[0], LW, bits, "cuda")
    kvq.write(x, plane, torch.zeros((1,), dtype=torch.int32, device="cuda"))
    return plane


@pytest.mark.parametrize("bits", BITS + (7, 3, 2))
def test_writer_matches_the_reference_quantizer(bits):
    from tensorfold.families.glm_moe_dsa.cuda import kvq

    g = torch.Generator().manual_seed(bits)
    x = latent(3000, g)
    plane = kvq.KvQ(3200, LW, bits, "cuda")
    kvq.write(x, plane, torch.tensor([100], dtype=torch.int32, device="cuda"))
    codes, scales = kvq.quantize_reference(x, bits)
    got = kvq.unpack_reference(plane.codes[100:3100], bits, LW)
    want = kvq.unpack_reference(codes, bits, LW)
    # the rotation's fp32 sums may land a value on the other side of a grid boundary: rare, and one step at most
    assert float((got - want).abs().max()) <= 1
    assert float((got != want).float().mean()) < 1e-3
    assert float(((plane.scales[100:3100].float() - scales.float()).abs() / scales.float().clamp(min=1e-6)).max()) \
        < 2e-3
    # every value within half a grid step (plus the scale's fp16 rounding) of its rotated original
    m = 1 << (bits - 1)
    rot = ((x.double().view(-1, LW // 32, 32) @ kvq.hadamard32(x.device).double()) * kvq.R32).view(-1, LW)
    deq = kvq.dequantize_reference(plane.codes[100:3100], plane.scales[100:3100], bits, LW, rotated=True).double()
    step = plane.scales[100:3100].double().repeat_interleave(32, dim=1) / m
    assert bool(((deq - rot).abs() <= step * 0.5 + rot.abs() * 2e-3 + 1e-6).all())


def reference(qa, qp, rows, pc, tokens, scale):
    """float64 attention of one row's heads over the given token ids."""

    k = rows[tokens].double()
    s = (qa.double() @ k.T + qp.double() @ pc[tokens].double().T) * scale
    return torch.softmax(s, dim=-1) @ k


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("R,pos", [(1, 700), (4, 2040), (300, 1200)])
def test_dense_attention_over_packed_rows(bits, R, pos):
    from tensorfold.families.glm5_next.cuda.latent import LatentScratch, chunks_for
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(bits * 1000 + R * 7 + pos)
    n = pos + R
    plane = plane_of(latent(n, g), bits)
    rows = plane.dequant(n)
    pc = (torch.randn((n, PW), generator=g) * 2).to(torch.bfloat16).cuda()
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    s = LatentScratch(R, H, chunks_for(n + 512), "cuda", lw=LW, part_rows=min(R, 256))
    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    got = mla_pe.attention(qa, qp, plane, pc, at, s, scale=256 ** -0.5, nch=chunks_for(n),
                           out=torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda"))
    for r in range(0, R, max(1, R // 7)):
        want = reference(qa[r], qp[r], rows, pc, torch.arange(pos + r + 1, device="cuda"), 256 ** -0.5)
        err = (got[r].double() - want).norm() / want.norm()
        assert float(err) < 6e-3, (r, float(err))


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("R,pos", [(1, 3000), (4, 9000), (64, 5000), (300, 2500)])
def test_sparse_attention_over_packed_rows(bits, R, pos):
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(bits * 1000 + R * 11 + pos)
    n = pos + R
    plane = plane_of(latent(n, g), bits)
    rows = plane.dequant(n)
    pc = (torch.randn((n, PW), generator=g) * 2).to(torch.bfloat16).cuda()
    topk = 2048
    tokens = torch.full((R, topk + 1), -1, dtype=torch.int32, device="cuda")
    counts = torch.zeros((R,), dtype=torch.int32, device="cuda")
    for r in range(R):
        vis = pos + r + 1
        if vis > topk:
            tokens[r, :topk] = torch.randperm(vis, generator=g)[:topk].sort().values.to(torch.int32).cuda()
            counts[r] = topk
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    junk = torch.randn((R, H, LW), generator=g).to(torch.bfloat16).cuda()
    fused, kt = mla_pe.FUSED_ROWS, mla_pe.PROMPT_KT
    outs = []
    try:
        mla_pe.PROMPT_KT = mla_pe.KT                       # the decode windows' key tile: their bits
        for rows_cut in (1 << 30, 1, 1):                   # chunk programs + _merge, the prompt kernel, at PROMPT_KT
            mla_pe.FUSED_ROWS = rows_cut
            if len(outs) == 2:
                mla_pe.PROMPT_KT = kt
            out = junk.clone()
            mla_pe.sparse_attention(qa, qp, plane, pc, tokens, counts, out, 256 ** -0.5)
            outs.append(out)
    finally:
        mla_pe.FUSED_ROWS, mla_pe.PROMPT_KT = fused, kt
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))        # prompt == decode at KT
    for r in range(0, R, max(1, R // 7)):
        if int(counts[r]) == 0:
            assert torch.equal(outs[0][r].view(torch.int16), junk[r].view(torch.int16))
            continue
        want = reference(qa[r], qp[r], rows, pc, tokens[r, :topk].long(), 256 ** -0.5)
        for o in (outs[0], outs[2]):
            err = (o[r].double() - want).norm() / want.norm()
            assert float(err) < 6e-3, (r, float(err))


def test_slot_bytes_of_the_quantized_formats():
    from tensorfold.families.glm_moe_dsa.cuda import kv8

    # 79 latent + rope rows (the rope key bf16) and 22 indexer key rows, FP8 or (the "b" modes) bf16
    assert {m: kv8.slot_bytes(79, 512, 64, 22, 128, m) for m in ("q8", "q6", "q5", "q4")} == \
        {"q8": 55992, "q6": 45880, "q5": 40824, "q4": 35768}
    assert {m: kv8.slot_bytes(79, 512, 64, 22, 128, m) for m in ("fp8b", "q8b", "q6b", "q5b", "q4b")} == \
        {"fp8b": 56508, "q8b": 58720, "q6b": 48608, "q5b": 43552, "q4b": 38496}


@pytest.mark.parametrize("R,pos", [(1, 2048), (4, 30000), (2048, 14000)])
def test_selection_over_q8_index_keys(R, pos):
    """Index keys in kvq's 8-bit format (LATENT/q8): the selection over the packed keys is the selection over their
    dequantized values up to near-ties (the rotated fp16 dots round differently), and nearly the bf16 keys' own."""

    from tensorfold.families.glm_moe_dsa.cuda import kvq, select

    g = torch.Generator().manual_seed(R * 17 + pos)
    HI, D, topk = 32, 128, 2048
    n = pos + R
    keys = (torch.randn((n, D), generator=g) * (torch.rand((D,), generator=g) + 0.2)).to(torch.bfloat16).cuda()
    plane = kvq.KvQ(n, D, 8, "cuda")
    kvq.write(keys, plane, torch.zeros((1,), dtype=torch.int32, device="cuda"))
    codes, scales = kvq.quantize_reference(keys, 8)
    assert float((kvq.unpack_reference(plane.codes, 8, D) - kvq.unpack_reference(codes, 8, D)).abs().max()) <= 1
    deq = plane.dequant(n).to(torch.bfloat16)
    qi = torch.randn((R, HI * D), generator=g).to(torch.bfloat16).cuda()
    wts = torch.randn((R, HI), generator=g).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    out = []
    for k in (deq, plane, keys):
        tokens = torch.empty((R, topk + 1), dtype=torch.int32, device="cuda")
        counts = torch.empty((R,), dtype=torch.int32, device="cuda")
        select.select_tokens(qi, wts, k, pos, R, topk, pos_dev, tokens=tokens, counts=counts)
        out.append((tokens, counts))

    def moved(a, b):
        return sum(len(set(a[0][r, :topk].tolist()) - set(b[0][r, :topk].tolist())) for r in range(R))

    assert torch.equal(out[0][1], out[1][1])
    # vs the dequantized keys (rounded to bf16 here, and dotted in bf16 where the packed path dots rotated fp16):
    # near-ties only; vs the bf16 keys themselves: the format's own rounding
    print(f"R {R} pos {pos}: moved vs dequantized {moved(out[0], out[1]) / (R * topk):.5f}, vs bf16 keys "
          f"{moved(out[2], out[1]) / (R * topk):.5f}")
    assert moved(out[0], out[1]) <= max(2, R * topk // 200)
    assert moved(out[2], out[1]) <= R * topk // 50


def test_mode_names():
    from tensorfold.families.glm_moe_dsa.cuda import kv8

    assert kv8.parse("bf16") == ("bf16", "bf16") and kv8.parse("idx8") == ("bf16", "fp8")
    assert kv8.parse("q6") == ("q6", "fp8") and kv8.parse("q6b") == ("q6", "bf16") and kv8.parse("q6/q8") == ("q6", "q8")
    assert kv8.slot_bytes(79, 512, 64, 22, 128, "q6/q8") == 79 * (6 * 64 + 32 + 128) + 22 * 136
    with pytest.raises(ValueError):
        kv8.parse("q9")
    assert len({kv8.mode_code(m) for m in ("bf16", "idx8", "fp8", "q8", "q8b", "q8/q8", "bf16/q8")}) == 7
