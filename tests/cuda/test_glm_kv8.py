"""GLM-5.3's FP8 cache planes (TF_GLM_KV=fp8 / idx8, ``kv8``): a row's codes and power-of-two scale match torch's
e4m3 cast of the row over its scale, the dequantized rows are exact in bf16, and every reader (dense MLA, sparse MLA for
decode windows and prompt chunks, DSA scoring and selection) matches a float64 computation over the dequantized rows,
the prompt kernel bit-equal to the decode windows' chunk programs + merge, at GLM-5.3's TP4 shapes (16 heads, 512-wide
latent, 64-wide rope key, 32 indexer heads of 128).

(Not bit-equal to the bf16 kernels over the dequantized rows: a tile dequantized from 8-bit loads reaches tl.dot in
Triton's 8-bit operand layout, which orders a tensor-core step's k values differently, so the fp32 sums round
differently. tools/check_select.py measures what that and the FP8 rounding move in the selection.)"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

H, LW, PW = 16, 512, 64


def rows(n: int, width: int, g: torch.Generator) -> torch.Tensor:
    """bf16 rows shaped like a normed latent: a per-column gain, a few outliers, one zero row, one tiny row."""

    gain = torch.rand((width,), generator=g) * 3 + 0.05
    x = torch.randn((n, width), generator=g) * gain
    x[::97] *= 40                                         # rows with a large amax
    if n > 5:
        x[3] = 0
        x[5] *= 1e-30                                     # a row near fp32's smallest normals
    return x.to(torch.bfloat16).cuda()


def reference(qa, qp, rows, pc, tokens, scale):
    """float64 attention of one row's heads over the given token ids."""

    k = rows[tokens].double()
    s = (qa.double() @ k.T + qp.double() @ pc[tokens].double().T) * scale
    return torch.softmax(s, dim=-1) @ k


def planes(x: torch.Tensor):
    from tensorfold.families.glm_moe_dsa.cuda import kv8

    plane = kv8.Kv8(x.shape[0], x.shape[1], "cuda")
    kv8.write(x, plane, torch.zeros((1,), dtype=torch.int32, device="cuda"))
    return plane, plane.dequant()


def test_write_is_the_reference_cast():
    from tensorfold.families.glm_moe_dsa.cuda import kv8

    g = torch.Generator().manual_seed(1)
    x = rows(1000, LW, g)
    plane = kv8.Kv8(1200, LW, "cuda")
    pos = torch.tensor([150], dtype=torch.int32, device="cuda")
    kv8.write(x[:, :LW], plane, pos)                      # rows at slots 150 .. 1149
    codes, scales = kv8.quantize_reference(x)
    assert torch.equal(plane.codes[150:1150], codes)
    assert torch.equal(plane.scales[150:1150].view(torch.int32), scales.view(torch.int32))
    assert torch.equal(plane.codes[:150], torch.zeros_like(plane.codes[:150]))       # nothing else written
    # no code saturates (|x| / scale <= 256 < 448) and every scale is a power of two
    q = codes.view(torch.float8_e4m3fn).to(torch.float32)
    assert float(q.abs().max()) <= 256
    assert torch.equal(scales.view(torch.int32) & 0x7FFFFF, torch.zeros_like(scales, dtype=torch.int32))
    # e4m3 keeps 3 mantissa bits: a normal code is within 1/16 of its value
    dq = plane.dequant(1150)[150:].to(torch.float32)
    xf = x.to(torch.float32)
    big = xf.abs() >= scales[:, None] * 2.0 ** -6
    assert bool(((dq - xf).abs() <= xf.abs() / 16)[big].all())
    assert float(dq[3].abs().max()) == 0.0


def test_strided_rows_write_like_contiguous_ones():
    from tensorfold.families.glm_moe_dsa.cuda import kv8

    g = torch.Generator().manual_seed(2)
    lat = rows(64, LW + PW, g)                            # kv_a's output: the latent then the rope key
    a = kv8.Kv8(64, LW, "cuda")
    b = kv8.Kv8(64, LW, "cuda")
    zero = torch.zeros((1,), dtype=torch.int32, device="cuda")
    kv8.write(lat[:, :LW], a, zero)
    kv8.write(lat[:, :LW].contiguous(), b, zero)
    assert torch.equal(a.codes, b.codes) and torch.equal(a.scales, b.scales)


@pytest.mark.parametrize("R,pos", [(1, 0), (1, 700), (4, 2040), (6, 1500), (300, 1200)])
def test_dense_attention_over_fp8_rows(R, pos):
    from tensorfold.families.glm5_next.cuda.latent import LatentScratch, chunks_for
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(R * 7 + pos)
    n = pos + R
    plane, dq = planes(rows(n, LW, g))
    pc = (torch.randn((n, PW), generator=g) * 2).to(torch.bfloat16).cuda()
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    s = LatentScratch(R, H, chunks_for(n + 512), "cuda", lw=LW, part_rows=min(R, 256))
    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    got = mla_pe.attention(qa, qp, plane, pc, at, s, scale=256 ** -0.5, nch=chunks_for(n),
                           out=torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda"))
    for r in range(0, R, max(1, R // 7)):
        want = reference(qa[r], qp[r], dq, pc, torch.arange(pos + r + 1, device="cuda"), 256 ** -0.5)
        assert float((got[r].double() - want).norm() / want.norm()) < 6e-3


@pytest.mark.parametrize("R,pos", [(1, 3000), (4, 9000), (6, 2100), (64, 5000), (300, 2500)])
def test_sparse_attention_over_fp8_rows(R, pos):
    """Decode windows (a program a chunk + _merge) and prompt chunks (a program a row, merged in registers)."""

    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    g = torch.Generator().manual_seed(R * 11 + pos)
    n = pos + R
    plane, dq = planes(rows(n, LW, g))
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
        for cut in (1 << 30, 1, 1):                        # chunk programs + _merge, the prompt kernel, at PROMPT_KT
            mla_pe.FUSED_ROWS = cut
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
        want = reference(qa[r], qp[r], dq, pc, tokens[r, :topk].long(), 256 ** -0.5)
        for o in (outs[0], outs[2]):
            assert float((o[r].double() - want).norm() / want.norm()) < 6e-3


@pytest.mark.parametrize("R,pos", [(1, 2048), (4, 30000), (6, 6677), (2048, 14000), (300, 4096)])
def test_selection_over_fp8_keys_is_the_selection_over_the_dequantized_keys(R, pos):
    from tensorfold.families.glm_moe_dsa.cuda import select

    g = torch.Generator().manual_seed(R * 13 + pos)
    HI, D, topk = 32, 128, 2048
    n = pos + R
    plane, dq = planes(rows(n, D, g))
    qi = (torch.randn((R, HI * D), generator=g)).to(torch.bfloat16).cuda()
    wts = (torch.randn((R, HI), generator=g)).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    out = []
    for keys in (dq, plane):
        tokens = torch.empty((R, topk + 1), dtype=torch.int32, device="cuda")
        counts = torch.empty((R,), dtype=torch.int32, device="cuda")
        select.select_tokens(qi, wts, keys, pos, R, topk, pos_dev, tokens=tokens, counts=counts)
        out.append((tokens, counts))
    assert torch.equal(out[0][1], out[1][1])
    # the same scores up to the fp32 sums' order: a near tie may swap, nothing more
    moved = sum(len(set(out[0][0][r].tolist()) ^ set(out[1][0][r].tolist())) for r in range(R)) / 2
    assert moved <= max(1, R * topk // 2000), moved


def test_slot_bytes():
    from tensorfold.families.glm_moe_dsa.cuda import kv8

    # GLM-5.3 at TP4: 78 layers + the MTP head's latent and rope key, 21 full indexer layers + the MTP head's keys
    assert kv8.slot_bytes(79, 512, 64, 22, 128, "bf16") == 96640
    assert kv8.slot_bytes(79, 512, 64, 22, 128, "fp8") == 53780
    assert kv8.slot_bytes(79, 512, 64, 22, 128, "idx8") == 93912
