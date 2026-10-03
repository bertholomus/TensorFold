"""GLM-5.3's concurrent rounds (``rows``): every kernel that reads or writes a cache, run over a pool of stream slots
with per-row tables, gives each row exactly the bits the single-stream kernel gives it on its own slot (views of the
same pool) at the same position: writers, rope, dense attention, DSA selection (tokens and counts) and sparse
attention, for streams below, across and past the dense limit, in every cache format, at GLM-5.3's TP4 shapes."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

H, LW, PW, HI, D, TOPK = 16, 512, 64, 32, 128, 2048
CAP = 8192                        # slot rows (a multiple of 64: key tiles stay aligned to each stream's token 0)
DL = 2048                         # GLM-5.3's dense limit (index_topk)


def gen(n: int, width: int, g: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    gain = torch.rand((width,), generator=g) * 3 + 0.05
    return (torch.randn((n, width), generator=g) * gain * scale).to(torch.bfloat16).cuda()


def plane(fmt: str, rows: int, width: int):
    from tensorfold.families.glm_moe_dsa.cuda import kv8, kvq

    if fmt == "bf16":
        return torch.zeros((rows, width), dtype=torch.bfloat16, device="cuda")
    if fmt == "fp8":
        return kv8.Kv8(rows, width, "cuda")
    return kvq.KvQ(rows, width, int(fmt[1:]), "cuda")


def view(p, lo: int, hi: int):
    return p[lo:hi] if isinstance(p, torch.Tensor) else p.rows(lo, hi)


def same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
                           b.view(torch.int16) if b.dtype == torch.bfloat16 else b)
    return torch.equal(a.codes, b.codes) and torch.equal(a.scales.view(torch.int16) if a.scales.dtype == torch.float16
                                                          else a.scales.view(torch.int32),
                                                          b.scales.view(torch.int16) if b.scales.dtype == torch.float16
                                                          else b.scales.view(torch.int32))


def solo_write(rows: torch.Tensor, p, pos: int) -> None:
    from tensorfold.families.glm5_next.cuda import latent
    from tensorfold.families.glm_moe_dsa.cuda import kv8, kvq
    from tensorfold.families.glm_moe_dsa.cuda.kv8 import Kv8
    from tensorfold.families.glm_moe_dsa.cuda.kvq import KvQ

    at = torch.tensor([pos], dtype=torch.int32, device="cuda")
    if isinstance(p, Kv8):
        kv8.write(rows, p, at)
    elif isinstance(p, KvQ):
        kvq.write(rows, p, at)
    else:
        latent.latent_write(rows, p, at)


# streams: (first position of the window, window rows); slots in this order
MIXES = {
    "dense": [(5, 4), (700, 1), (1200, 3)],
    "across": [(2046, 4), (100, 2), (3000, 4)],
    "sparse": [(2048, 1), (5000, 4), (2500, 3), (8000, 2)],
    "mixed": [(9, 1), (2047, 2), (6001, 4), (3333, 1)],
}


@pytest.mark.parametrize("fmt,ifmt", [("bf16", "bf16"), ("fp8", "fp8"), ("q5", "fp8"), ("q8", "q8")])
@pytest.mark.parametrize("mix", list(MIXES))
def test_round_rows_equal_each_streams_solo_rows(fmt, ifmt, mix):
    from tensorfold.families.glm5_next.cuda import latent
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe, rope, rows, select

    streams = MIXES[mix]
    S = len(streams)
    g = torch.Generator().manual_seed(len(mix) * 7 + len(fmt) + 31 * len(ifmt))
    lat, pc, idx = plane(fmt, S * CAP, LW), plane("bf16", S * CAP, PW), plane(ifmt, S * CAP, D)
    # each slot's committed rows (0 .. p - 1), written through the single-stream writers on its views
    for i, (p, n) in enumerate(streams):
        if p:
            solo_write(gen(p, LW, g), view(lat, i * CAP, (i + 1) * CAP), 0)
            solo_write(gen(p, PW, g), view(pc, i * CAP, (i + 1) * CAP), 0)
            solo_write(gen(p, D, g), view(idx, i * CAP, (i + 1) * CAP), 0)
    R = sum(n for _, n in streams)
    t = rows.Tables(32, 4, "cuda")
    assert t.fill([(i * CAP, p, n) for i, (p, n) in enumerate(streams)], DL) == R
    # the round's new rows: written by the round's writers on the pool, by the single-stream writers on copies
    new_lat, new_pc, new_idx = gen(R, LW, g), gen(R, PW, g), gen(R, D, g)
    solo = [plane(fmt, CAP, LW) for _ in streams], [plane("bf16", CAP, PW) for _ in streams], \
        [plane(ifmt, CAP, D) for _ in streams]
    r0 = 0
    for i, (p, n) in enumerate(streams):
        for k, (pool, src) in enumerate(((lat, new_lat), (pc, new_pc), (idx, new_idx))):
            v = view(pool, i * CAP, (i + 1) * CAP)
            mine = solo[k][i]
            if isinstance(v, torch.Tensor):
                mine.copy_(v)
            else:
                mine.codes.copy_(v.codes)
                mine.scales.copy_(v.scales)
            solo_write(src[r0:r0 + n], mine, p)
        r0 += n
    rows.write(new_lat, lat, t)
    rows.write(new_pc, pc, t)
    rows.write(new_idx, idx, t)
    for i in range(S):
        for k, pool in enumerate((lat, pc, idx)):
            assert same(view(pool, i * CAP, (i + 1) * CAP), solo[k][i]), (i, k)
    # rope tables
    cos, sin = torch.empty((R, PW // 2), device="cuda"), torch.empty((R, PW // 2), device="cuda")
    rows.rope(cos, sin, t, R, 1e6, PW)
    r0 = 0
    for p, n in streams:
        c1, s1 = torch.empty((n, PW // 2), device="cuda"), torch.empty((n, PW // 2), device="cuda")
        rope.table(c1, s1, torch.tensor([p], dtype=torch.int32, device="cuda"), n, 1e6, PW)
        assert torch.equal(c1, cos[r0:r0 + n]) and torch.equal(s1, sin[r0:r0 + n])
        r0 += n
    # attention and selection: the round's way (rows.*) against each stream's single-stream calls on its views
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qi = torch.randn((R, HI * D), generator=g).to(torch.bfloat16).cuda()
    wts = torch.randn((R, HI), generator=g).to(torch.bfloat16).cuda()
    scale = 256 ** -0.5
    s = latent.LatentScratch(32, H, latent.chunks_for(2560 + 32), "cuda")
    got = torch.full((R, H, LW), float("nan"), dtype=torch.bfloat16, device="cuda")
    tokens = torch.empty((R, TOPK + 1), dtype=torch.int32, device="cuda")
    counts = torch.empty((R,), dtype=torch.int32, device="cuda")
    if t.dense:
        rows.dense_attention(qa, qp, lat, pc, t, s, scale=scale, dense_limit=DL, out=got)
    if t.sparse:
        rows.select(qi, wts, idx, R, TOPK, t, tokens=tokens, counts=counts, max_rows=4)
        rows.sparse_attention(qa, qp, lat, pc, tokens, counts, t, got, scale)
    r0 = 0
    for i, (p, n) in enumerate(streams):
        lv, pv, iv = (view(x, i * CAP, (i + 1) * CAP) for x in (lat, pc, idx))
        pos = torch.tensor([p], dtype=torch.int32, device="cuda")
        out = torch.full((n, H, LW), float("nan"), dtype=torch.bfloat16, device="cuda")
        s1 = latent.LatentScratch(n, H, latent.chunks_for(2560 + n), "cuda")
        if p < DL:                                                   # forward.dsa_block's single-stream choices
            mla_pe.attention(qa[r0:r0 + n], qp[r0:r0 + n], lv, pv, pos, s1, scale=scale,
                             nch=-(-(p + n) // latent.CHUNK), out=out)
        if p + n - 1 >= DL:
            tk = torch.empty((n, TOPK + 1), dtype=torch.int32, device="cuda")
            ct = torch.empty((n,), dtype=torch.int32, device="cuda")
            select.select_tokens(qi[r0:r0 + n].contiguous(), wts[r0:r0 + n], iv, p, n, TOPK, pos, tokens=tk,
                                 counts=ct)
            assert torch.equal(ct, counts[r0:r0 + n]), (mix, i)
            for r in range(n):
                k = int(ct[r])
                assert torch.equal(tk[r, :k], tokens[r0 + r, :k]), (mix, i, r)
            mla_pe.sparse_attention(qa[r0:r0 + n], qp[r0:r0 + n], lv, pv, tk, ct, out, scale)
        assert torch.equal(out.view(torch.int16), got[r0:r0 + n].view(torch.int16)), (mix, i)
        r0 += n
