"""GLM-5.3's decode context parallelism (TF_GLM_DCP=4, ``dcp``) on one GPU, the four ranks emulated: each rank's
scores are the replicated scorer's for the positions it holds, bit for bit; the merged selection is the replicated
selection exactly; and the attention each rank computes over its share, exchanged and merged by log-sum-exp, matches a
float64 attention over every key (bf16, FP8 and Q8 latents), for rows below and past the dense limit."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

G, H, LW, PW, TOPK = 4, 16, 512, 64, 2048
HI, D = 32, 128


def gen(n: int, width: int, g: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    gain = torch.rand((width,), generator=g) * 3 + 0.05
    return (torch.randn((n, width), generator=g) * gain * scale).to(torch.bfloat16).cuda()


def held(x: torch.Tensor, rank: int) -> torch.Tensor:
    return x[rank::G].contiguous()


def plane_of(rows: torch.Tensor, fmt: str):
    """A cache plane of these rows in a format ('bf16', 'fp8', 'q8') and its values as the kernels read them."""

    from tensorfold.families.glm_moe_dsa.cuda import kv8, kvq

    zero = torch.zeros((1,), dtype=torch.int32, device="cuda")
    if fmt == "bf16":
        return rows, rows.float()
    if fmt == "fp8":
        p = kv8.Kv8(rows.shape[0], rows.shape[1], "cuda")
        kv8.write(rows, p, zero)
        return p, p.dequant().float()
    p = kvq.KvQ(rows.shape[0], rows.shape[1], 8, "cuda")
    kvq.write(rows, p, zero)
    return p, p.dequant()


@pytest.mark.parametrize("fmt", ["bf16", "fp8", "q8"])
@pytest.mark.parametrize("R,pos", [(1, 9000), (4, 30001), (64, 6000)])
def test_scores_and_selection_are_the_replicated_ones(fmt, R, pos):
    from tensorfold.families.glm_moe_dsa.cuda import dcp, select

    g = torch.Generator().manual_seed(R * 31 + pos)
    n = pos + R
    keys = gen(n, D, g)
    qi = torch.randn((R, HI * D), generator=g).to(torch.bfloat16).cuda()
    wts = torch.randn((R, HI), generator=g).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    full, _ = plane_of(keys, fmt)
    tokens = torch.empty((R, TOPK + 1), dtype=torch.int32, device="cuda")
    counts = torch.empty((R,), dtype=torch.int32, device="cuda")
    select.select_tokens(qi, wts, full, pos, R, TOPK, pos_dev, tokens=tokens, counts=counts)
    want = [set(tokens[r, :TOPK].tolist()) for r in range(R)]
    np_max = select.sparse_bucket(pos, R, G)
    cands = []
    for rank in range(G):
        part, _ = plane_of(held(keys, rank), fmt)
        cands.append(dcp.candidates(qi, wts, part, pos_dev, R, np_max, TOPK, G, rank))
    every = torch.stack(cands)
    got = [set() for _ in range(R)]
    total = torch.zeros((R,), dtype=torch.int64, device="cuda")
    for rank in range(G):
        t = torch.empty((R, TOPK + 1), dtype=torch.int32, device="cuda")
        c = torch.empty((R,), dtype=torch.int32, device="cuda")
        dcp.merge(every, G, rank, TOPK, t, c)
        total += c
        for r in range(R):
            slots = t[r, :int(c[r])].tolist()
            assert slots == sorted(slots) and all(x >= 0 for x in slots)
            got[r] |= {s * G + rank for s in slots}
    assert torch.equal(total, torch.full_like(total, TOPK))
    assert got == want                                       # the replicated selection, token for token


def reference(qa, qp, rows, pc, idx, scale):
    k = rows[idx].double()
    s = (qa.double() @ k.T + qp.double() @ pc[idx].double().T) * scale
    return torch.softmax(s, dim=-1) @ k


@pytest.mark.parametrize("fmt", ["bf16", "fp8", "q8"])
@pytest.mark.parametrize("R,pos", [(1, 700), (3, 2046), (4, 9000), (80, 5000)])
def test_attention_over_the_ranks_shares(fmt, R, pos):
    from tensorfold.families.glm_moe_dsa.cuda import dcp

    g = torch.Generator().manual_seed(R * 7 + pos + len(fmt))
    n = pos + R
    lat = gen(n, LW, g)
    pc = gen(n, PW, g)
    qa = (torch.randn((R, G * H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()      # every head of the window
    qp = (torch.randn((R, G * H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    scale = 256 ** -0.5
    # each row's keys past the dense limit: a random global selection (ascending), split by owner
    sel = []
    for r in range(R):
        p = pos + r
        sel.append(torch.arange(p + 1) if p < TOPK else torch.randperm(p + 1, generator=g)[:TOPK].sort().values)
    qall = torch.cat([qa.view(R, G, H, LW), qp.view(R, G, H, PW)], dim=3).permute(1, 0, 2, 3).contiguous()
    words = LW // 2 + 1
    sends = []
    _, values = plane_of(lat, fmt)
    for rank in range(G):
        cache, _ = plane_of(held(lat, rank), fmt)
        tokens = torch.full((R, TOPK + 1), -1, dtype=torch.int32, device="cuda")
        counts = torch.zeros((R,), dtype=torch.int32, device="cuda")
        for r in range(R):
            mine = sel[r][sel[r] % G == rank] // G
            tokens[r, :len(mine)] = mine.to(torch.int32).cuda()
            counts[r] = len(mine)
        send = torch.empty((G, R, H, words), dtype=torch.float32, device="cuda")
        dcp.attend(qall, cache, held(pc, rank), tokens, counts, pos_dev, send, R, H, G, rank, scale, TOPK)
        sends.append(send)
    for owner in range(G):
        recv = torch.stack([s[owner] for s in sends])                       # [G src, R, H, words]
        out = torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda")
        dcp.combine(recv, R * H * words, out, G)
        for r in range(R):
            idx = sel[r].cuda()
            want = reference(qa[r, owner * H:(owner + 1) * H], qp[r, owner * H:(owner + 1) * H], values, pc, idx, scale)
            err = float((out[r].double() - want).norm() / want.norm())
            assert err < 6e-3, (owner, r, err)
