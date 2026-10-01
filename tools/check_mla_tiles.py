"""Bit-compare glm_moe_dsa's dense MLA attention (mla_pe.attention) against the full-tile loop it replaced.

usage (one GPU, in the tf container): python3 tools/check_mla_tiles.py
Random latents, rope keys and queries at GLM-5.3's TP4 shapes (16 heads, 512 + 64), windows of 1-8 rows at positions
that start, straddle and end chunks and tiles; every output must match the old kernel bit for bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.families.glm_moe_dsa.cuda import mla_pe
from tensorfold.families.glm_moe_dsa.cuda.mla_pe import _tile_pe
from tensorfold.families.glm5_next.cuda.latent import CHUNK, HB, KT, LatentScratch, _merge


@triton.jit
def _dense_chunks_full(QA, QP, LC, PC, POS, PO, PM, PL, R, H: tl.constexpr, LW: tl.constexpr, PW: tl.constexpr,
                       CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr):
    """The kernel as of 75e6e4b: every tile of an active chunk, masked past the row's last key."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    P = tl.load(POS)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    kq = tl.arange(0, PW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    start = c * CH
    limit = P + r
    if start <= limit:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        qp = tl.load(QP + (r * H + hh[:, None]) * PW + kq[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        for t in range(CH // KTT):
            ki = start + t * KTT + tl.arange(0, KTT)
            ok = ki <= limit
            kv = tl.load(LC + ki[:, None].to(tl.int64) * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            kp = tl.load(PC + ki[:, None].to(tl.int64) * PW + kq[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile_pe(q, qp, kv, kp, m, l, o, ok, SCALE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


def old_attention(qa, qp, cache, pcache, pos, s, *, scale, nch, out):
    R, H, LW = qa.shape
    PW = qp.shape[2]
    n = nch * R * H
    _dense_chunks_full[(R, triton.cdiv(H, HB), nch)](qa, qp, cache, pcache, pos, s.po[:n * LW], s.pm[:n], s.pl[:n], R,
                                                     H=H, LW=LW, PW=PW, CH=CHUNK, SCALE=scale, HBT=HB, KTT=KT,
                                                     num_warps=8, num_stages=1)
    _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    return out


def staged_attention(stages: int, warps: int = 8):
    def run(qa, qp, cache, pcache, pos, s, *, scale, nch, out):
        R, H, LW = qa.shape
        PW = qp.shape[2]
        n = nch * R * H
        mla_pe._dense_chunks_pe[(R, triton.cdiv(H, HB), nch)](qa, qp, cache, pcache, pos, s.po[:n * LW], s.pm[:n],
                                                              s.pl[:n], R, H=H, LW=LW, PW=PW, CH=CHUNK, SCALE=scale,
                                                              HBT=HB, KTT=KT, num_warps=warps, num_stages=stages)
        _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
        return out
    return run


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    H, LW, PW, cap = 16, 512, 64, 2560 + 8
    nch_cap = triton.cdiv(cap, CHUNK)
    cache = (torch.randn((cap, LW), device=dev) * 0.5).to(torch.bfloat16)
    pcache = (torch.randn((cap, PW), device=dev) * 0.5).to(torch.bfloat16)
    s = LatentScratch(8, H, nch_cap, dev, lw=LW, part_rows=8)
    scale = 256 ** -0.5
    checked = mismatched = 0
    for pos in (0, 1, 7, 31, 32, 33, 160, 511, 512, 513, 1000, 1535, 2040, 2047, 2048, 2550):
        for R in (1, 2, 3, 4, 6, 8):
            if pos + R > cap:
                continue
            qa = torch.randn((R, H, LW), device=dev).to(torch.bfloat16)
            qp = torch.randn((R, H, PW), device=dev).to(torch.bfloat16)
            pos_dev = torch.tensor([pos], dtype=torch.int32, device=dev)
            for nch in sorted({triton.cdiv(pos + R, CHUNK), nch_cap}):        # eager's chunk count and the graphs'
                a = torch.empty((R, H, LW), dtype=torch.bfloat16, device=dev)
                b = torch.empty((R, H, LW), dtype=torch.bfloat16, device=dev)
                old_attention(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch, out=a)
                mla_pe.attention(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch, out=b)
                torch.cuda.synchronize()
                checked += 1
                if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                    mismatched += 1
                    print(f"MISMATCH pos {pos} R {R} nch {nch}: max |diff| "
                          f"{(a.float() - b.float()).abs().max().item():.3e}", flush=True)
    print(f"mla tiles: {checked} cases, {mismatched} mismatched", flush=True)

    # timing at decode shapes: one row, the graphs' chunk count
    import time

    for pos in (160, 1000, 2040):
        qa = torch.randn((1, H, LW), device=dev).to(torch.bfloat16)
        qp = torch.randn((1, H, PW), device=dev).to(torch.bfloat16)
        pos_dev = torch.tensor([pos], dtype=torch.int32, device=dev)
        out = torch.empty((1, H, LW), dtype=torch.bfloat16, device=dev)
        variants = [("old", old_attention), ("new", mla_pe.attention), ("new s2", staged_attention(2)),
                    ("new s3", staged_attention(3)), ("new w4 s2", staged_attention(2, 4))]
        for name, fn in variants:
            if name.startswith("new "):
                ref = torch.empty_like(out)
                old_attention(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch_cap, out=ref)
                fn(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch_cap, out=out)
                torch.cuda.synchronize()
                print(f"pos {pos} {name}: same bits {torch.equal(ref.view(torch.int16), out.view(torch.int16))}")
            fn(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch_cap, out=out)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(50):
                    fn(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch_cap, out=out)
            g.replay()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            print(f"pos {pos} {name}: {(time.perf_counter() - t) / 500 * 1e6:.1f} us a call (graph)", flush=True)


if __name__ == "__main__":
    main()
