"""Same-bits checks and timings of a concurrent round's kernel variants on one GB10 (GLM-5.3 TP4 rank shapes, random
weights: the arithmetic is data-independent). Every variant must give the baseline's outputs bit for bit; then its
time a call (CUDA graph replays, cold weights cycling through many copies so L2 never holds the next one).

- experts: routed experts (exl3.experts.routed, decode windows) at 4 / 8 / 16 rows, a stream's 4 rows drawing their 8
  experts from a pool of 24 (consecutive tokens share experts) or independently; variants TF_EXL3_GROUPED_ORDER
  (expert-major programs), TF_EXL3_GROUP2 (the shared-memory grouping) and TF_EXL3_GROUPED_STAGES (each warp's
  weights through shared-memory stages).
- group: group vs group2 alone (uids, count and members equal entry for entry) for 1-64 rows.
- absorb: latent's Triton absorb / expand vs latent_rows.cu's (TF_GLM_DECODE_ABSORB=cuda) at 1-64 rows.

usage (one GPU, tf container): python3 tools/bench_c4_kernels.py [PARTS=experts,group,absorb,residual] [L=24]
"""

from __future__ import annotations

import sys
import time

import torch

E, D, I, K, BITS = 256, 6144, 512, 8, 3


def graph_us(fns, reps: int = 10) -> float:
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for fn in fns:
            fn()
    g.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (len(fns) * reps) * 1e6


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def picks_for(L: int, R: int, pool: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(L):
        rs = []
        for s in range(-(-R // 4)):
            own = torch.randperm(E, generator=g)[:max(pool, K)]
            for _ in range(min(4, R - 4 * s)):
                rs.append(torch.cat([own[torch.randperm(len(own), generator=g)[:K]], torch.tensor([E])]))
        out.append(torch.stack(rs).to(torch.int32).cuda())
    return out


def experts(L: int) -> None:
    from tensorfold.cuda.exl3 import experts as x3

    dev = "cuda"
    layers = []
    for _ in range(L):
        mats = {}
        for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
            mats[p] = [(torch.randint(-32768, 32767, (k // 16, n // 16, 16 * BITS), dtype=torch.int16, device=dev),
                        (torch.randn(k, device=dev) * 0.1 + 1).half(), (torch.randn(n, device=dev) * 0.01).half())
                       for _ in range(E)]
        layers.append(x3.prepare(mats["gate"], mats["up"], mats["down"], "mul1", device=dev))
    ebytes = 3 * D * I * BITS / 8
    print(f"-- routed experts: {L} layers of {E} x {ebytes / 2**20:.2f} MiB (3 bits, a rank's 6,144 x 512)", flush=True)
    variants = [("base", 0, False, 0, True), ("order1+group2", 1, True, 0, True), ("sm4", 1, True, 4, True),
                ("sm4 noy", 1, True, 4, False), ("sm6 noy", 1, True, 6, False)]
    for pool in (24, 256):
        for R in (4, 8, 16):
            picks = picks_for(L, R, pool, R * 1000 + pool)
            distinct = sum(len(set(p[:, :K].reshape(-1).tolist())) for p in picks) / L
            x = torch.randn((R, D), device=dev).to(torch.bfloat16)
            wts = torch.rand((R, K + 1), device=dev)
            sc = x3.Scratch(layers[0], R, K + 1, device=dev)
            outs = torch.empty((L, R, D), device=dev, dtype=torch.float32)
            ref, line = None, []
            for name, order, g2, stages, keep_y in variants:
                x3.GROUPED_ORDER, x3.GROUP2, x3.GROUPED_STAGES, x3.DECODE_Y = order, g2, stages, keep_y
                outs.zero_()
                us = graph_us([(lambda i=i, ex=ex: x3.routed(x, picks[i], wts, ex, sc, outs[i], R,
                                                             act_mode=x3.ACT_BF16, kernel="mma", y_unused=True))
                               for i, ex in enumerate(layers)])
                ok = True if ref is None else same(outs, ref)
                if ref is None:
                    ref = outs.clone()
                line.append(f"{name} {us:7.1f} us ({distinct * ebytes / us / 1e3:5.1f} GB/s){'' if ok else ' DIFFERS'}")
            x3.GROUPED_ORDER, x3.GROUP2, x3.GROUPED_STAGES, x3.DECODE_Y = 0, False, 0, True
            print(f"   pool {pool:3d}, {R:2d} rows, {distinct:5.1f} experts a layer: " + "; ".join(line), flush=True)
            if pool == 24 and R == 16:                   # the best variant's kernels, eager, by name
                from torch.profiler import ProfilerActivity, profile

                x3.GROUPED_ORDER, x3.GROUP2, x3.GROUPED_STAGES, x3.DECODE_Y = 1, True, 4, False
                for i, ex in enumerate(layers):
                    x3.routed(x, picks[i], wts, ex, sc, outs[i], R, act_mode=x3.ACT_BF16, kernel="mma", y_unused=True)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        for i, ex in enumerate(layers):
                            x3.routed(x, picks[i], wts, ex, sc, outs[i], R, act_mode=x3.ACT_BF16, kernel="mma",
                                      y_unused=True)
                    torch.cuda.synchronize()
                x3.GROUPED_ORDER, x3.GROUP2, x3.GROUPED_STAGES, x3.DECODE_Y = 0, False, 0, True
                agg: dict = {}
                for ev in prof.events():
                    if ev.device_type == torch.autograd.DeviceType.CUDA:
                        agg[ev.name[:60]] = agg.get(ev.name[:60], 0.0) + ev.time_range.elapsed_us()
                n = 3 * L
                print("   kernels a layer (eager, us): " + "; ".join(f"{k} {v / n:.1f}" for k, v in
                                                                   sorted(agg.items(), key=lambda kv: -kv[1])),
                      flush=True)
    del layers
    torch.cuda.empty_cache()


def group() -> None:
    from tensorfold.cuda.exl3 import experts as x3

    ext = x3._ext()
    print("-- grouping: group vs group2", flush=True)
    for R in (1, 4, 8, 12, 16, 32, 64):
        slots = K + 1
        picks = picks_for(75, R, 24, 7 + R)
        maxu = min(R * slots, E)
        bufs = [(torch.zeros((maxu,), dtype=torch.int32, device="cuda"), torch.zeros((1,), dtype=torch.int32,
                 device="cuda"), torch.full((maxu, R), -7, dtype=torch.int32, device="cuda")) for _ in range(2)]
        ok = True
        for p in picks:
            for fn, (u, c, m) in zip((ext.group, ext.group2), bufs):
                u.zero_()
                m.fill_(-7)
                fn(p, u, c, m, R, slots, E)
            (u0, c0, m0), (u1, c1, m1) = bufs
            n = int(c0)
            ok &= int(c1) == n and torch.equal(u0[:n], u1[:n]) and torch.equal(m0[:n], m1[:n])
        t0 = graph_us([(lambda p=p: ext.group(p, *bufs[0], R, slots, E)) for p in picks])
        t1 = graph_us([(lambda p=p: ext.group2(p, *bufs[1], R, slots, E)) for p in picks])
        print(f"   {R:2d} rows: group {t0:6.2f} us, group2 {t1:6.2f} us, outputs equal: {ok}", flush=True)


def absorb() -> None:
    from tensorfold.families.glm5_next.cuda import latent
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    H, QK, LW, DV = 16, 256, 512, 256
    print("-- absorb / expand: latent's Triton vs latent_rows.cu at decode rows", flush=True)
    ext = mla_pe._rows_ext()
    copies = 24                                                        # a layer's weights each call, cold
    wks, wvs = [], []
    for _ in range(copies):
        wk = (torch.randn((H, QK, LW), device="cuda") * 0.05).to(torch.bfloat16)
        wk[:, 192:] = 0                                                # rope rows of wk are zero
        wks.append(wk.contiguous())
        wvs.append((torch.randn((H, DV, LW), device="cuda") * 0.05).to(torch.bfloat16).contiguous())

    class A:
        def __init__(self, wk, wv):
            self.wk, self.wv, self.lw, self.v_dim, self.heads, self.qk_dim = wk, wv, LW, DV, H, QK

    absorbs = [A(k, v) for k, v in zip(wks, wvs)]
    for R in (1, 2, 4, 8, 12, 16, 32, 64):
        q = (torch.randn((R, H, QK), device="cuda")).to(torch.bfloat16)
        ol = (torch.randn((R, H, LW), device="cuda")).to(torch.bfloat16)
        oa = [torch.empty((R, H, LW), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        ov = [torch.empty((R, H, DV), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        latent.absorb_q(q, absorbs[0], oa[0])
        ext.absorb(q, absorbs[0].wk, oa[1], R)
        latent.expand_v(ol, absorbs[0], ov[0])
        ext.expand(ol, absorbs[0].wv, ov[1], R)
        ta0 = graph_us([(lambda a=a: latent.absorb_q(q, a, oa[0])) for a in absorbs])
        ta1 = graph_us([(lambda a=a: ext.absorb(q, a.wk, oa[1], R)) for a in absorbs])
        te0 = graph_us([(lambda a=a: latent.expand_v(ol, a, ov[0])) for a in absorbs])
        te1 = graph_us([(lambda a=a: ext.expand(ol, a.wv, ov[1], R)) for a in absorbs])
        print(f"   {R:2d} rows: absorb {ta0:6.1f} -> {ta1:6.1f} us (same bits: {same(oa[0], oa[1])}); expand "
              f"{te0:6.1f} -> {te1:6.1f} us (same bits: {same(ov[0], ov[1])})", flush=True)


def residual() -> None:
    from tensorfold.cuda import rdma
    from tensorfold.families.glm5_next.cuda import glue

    ext = rdma._ext()
    D, W = 6144, 4
    print("-- the fused collect's residual arithmetic vs residual_add (4 ranks' fp32 partials)", flush=True)
    g = torch.Generator(device="cpu").manual_seed(3)
    for R in (1, 2, 3, 4, 8, 12, 16):
        ok = True
        for scale in (1e-3, 1.0, 30.0):
            parts = (torch.randn((W, R, D), generator=g) * scale).cuda()
            parts[:, :, :8] = torch.tensor([0.0, -0.0, 1e-38, -1e-38, 3.4e38, -3.4e38, 1e-45, 65504.0])
            x = (torch.randn((R, D), generator=g) * 4).to(torch.bfloat16).cuda()
            a, b = torch.empty_like(x), torch.empty_like(x)
            glue.residual_add(x, a, parts)
            ext.rank_residual(parts, x, b)
            torch.cuda.synchronize()
            ok &= same(a, b)
        xs = [x.clone() for _ in range(48)]
        t0 = graph_us([(lambda xx=xx: glue.residual_add(xx, xx, parts)) for xx in xs])
        t1 = graph_us([(lambda xx=xx: ext.rank_residual(parts, xx, xx)) for xx in xs])
        print(f"   {R:2d} rows: same bits {ok}; residual_add {t0:5.1f} us, fused arithmetic {t1:5.1f} us (from device "
              f"memory)", flush=True)


def main() -> None:
    torch.manual_seed(0)
    parts = (sys.argv[1] if len(sys.argv) > 1 else "experts,group,absorb,residual").split(",")
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    print(f"{torch.cuda.get_device_name()}, L2 {torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB",
          flush=True)
    for part in parts:
        {"experts": lambda: experts(L), "group": group, "absorb": absorb, "residual": residual}[part]()


if __name__ == "__main__":
    main()
