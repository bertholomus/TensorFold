"""The routed experts of a concurrent round on one GB10 (GLM-5.3 TP4 rank shapes: 256 experts of 6,144 x 512 per rank,
3-bit trellis), as the rows path calls them (kernel "mma", a decode window's grouping), over cold weights cycling
through L layers: us a layer and the weight bytes it reads (distinct experts x an expert's bytes) per second, for R
rows from S streams. Rows of one stream draw their 8 experts from a pool of POOL experts (consecutive tokens share
experts); streams' pools are independent (POOL 256: every row's picks independent).

usage (one GPU, tf container): python3 tools/bench_rows_moe.py [L=24] [POOLS=12,24,256] [ROWS=4,8,16] [CFGS]
  CFGS: tile settings to compare, "gu:NT,W,SK,PF" or "d:NT,W,SK,PF" joined by "/" (gate/up's or down's setting, the other
  at its default; "default" for both defaults). A setting is per shape, never per row count (rows keep their bits), so
  a new one changes every decode window's bits (new serial references).
"""

from __future__ import annotations

import sys
import time

import torch

from tensorfold.cuda.exl3 import experts as x3experts

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


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    pools = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "12,24,256").split(",")]
    rows = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "4,8,16").split(",")]
    layers = []
    for _ in range(L):
        mats = {}
        for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
            mats[p] = [(torch.randint(-32768, 32767, (k // 16, n // 16, 16 * BITS), dtype=torch.int16, device=dev),
                        torch.ones(k, device=dev).half(), torch.full((n,), 0.01, device=dev).half()) for _ in range(E)]
        layers.append(x3experts.prepare(mats["gate"], mats["up"], mats["down"], "mul1", device=dev))
    expert_bytes = 3 * D * I * BITS / 8
    print(f"{L} layers of {E} experts, {expert_bytes / 2**20:.2f} MiB an expert (rank share, {BITS} bits)", flush=True)
    cfgs = (sys.argv[4] if len(sys.argv) > 4 else "default").split("/")
    for cfg in cfgs:
        gu = dn = None
        if cfg.startswith("gu:"):
            gu = tuple(int(v) for v in cfg[3:].split(","))
        elif cfg.startswith("d:"):
            dn = tuple(int(v) for v in cfg[2:].split(","))
        print(f"-- tile setting: gate/up {gu or x3experts.default_config(D, I, True)}, down "
              f"{dn or x3experts.default_config(I, D, False)}", flush=True)
        for pool in pools:
            for R in rows:
                run_one(layers, pool, R, gu, dn, expert_bytes, dev)


def run_one(layers, pool: int, R: int, gu, dn, expert_bytes: float, dev: str) -> None:
    L = len(layers)
    slots = K + 1
    g = torch.Generator(device="cpu").manual_seed(R * 1000 + pool)
    picks, distinct = [], 0
    for _ in range(L):
        rs = []
        for s in range(-(-R // 4)):                         # a stream's window: 4 rows (MTP-3)
            own = torch.randperm(E, generator=g)[:max(pool, K)]
            for _ in range(min(4, R - 4 * s)):
                rs.append(torch.cat([own[torch.randperm(len(own), generator=g)[:K]], torch.tensor([E])]))
        p = torch.stack(rs).to(torch.int32)
        distinct += len(set(p[:, :K].reshape(-1).tolist()))
        picks.append(p.to(dev))
    x = torch.randn((R, D), device=dev).to(torch.bfloat16)
    wts = torch.rand((R, slots), device=dev)
    sc = x3experts.Scratch(layers[0], R, slots, cfg_gu=gu, cfg_d=dn, device=dev)
    part = torch.empty((R, D), device=dev, dtype=torch.float32)

    def routed(i, ex):
        x3experts.routed(x, picks[i], wts, ex, sc, part, R, act_mode=x3experts.ACT_BF16, kernel="mma")

    try:
        us = graph_us([(lambda i=i, ex=ex: routed(i, ex)) for i, ex in enumerate(layers)])
    except Exception as exc:                              # noqa: BLE001  (a setting the extension does not build)
        print(f"pool {pool:3d}, {R:2d} rows: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return
    n = distinct / L
    print(f"pool {pool:3d}, {R:2d} rows: {n:5.1f} distinct experts a layer, {us:7.1f} us a layer, "
          f"{n * expert_bytes / us / 1e3:6.1f} GB/s of weights, {us / n:5.2f} us an expert", flush=True)


if __name__ == "__main__":
    main()
