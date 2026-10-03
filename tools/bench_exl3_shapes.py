"""GB/s of GLM-5.3's TP4 EXL3 matrices on one GB10, decode rows: each linear shape (rot_in + linear) and the routed
experts (gate/up, down) at their tile settings, against a plain read of the same bytes. Random trellis words (the
decode is data-independent); a variant with the same arithmetic order is bit-compared to the shipped setting.

usage (one GPU, in the tf container): python3 tools/bench_exl3_shapes.py [ROWS=1,4] [PARTS=linear,experts]
  (a concurrent round's verify window is 4 rows a stream: 16 at four streams)
"""

from __future__ import annotations

import sys
import time

import torch

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3.linear import Exl3Linear


def graph_time(fns, reps: int = 10) -> float:
    """us a call, the calls captured in one CUDA graph (each once, in order) and replayed."""

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


def linear(k: int, n: int, bits: int) -> Exl3Linear:
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), dtype=torch.int16)
    suh = torch.randn(k).half() * 0.1 + 1
    svh = torch.randn(n).half() * 0.01
    return Exl3Linear.from_tensors(trellis, suh, svh, "mul1", device="cuda")


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    rows = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,4").split(",")]
    parts = (sys.argv[2] if len(sys.argv) > 2 else "linear,experts").split(",")
    print(f"L2 {torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB", flush=True)
    copy_src = torch.empty((256 * 1024 * 1024,), dtype=torch.int32, device=dev)       # 1 GiB
    copy_dst = torch.empty_like(copy_src)
    us = graph_time([lambda: copy_dst.copy_(copy_src)], reps=5)
    print(f"device copy of 1 GiB: {2 * copy_src.numel() * 4 / us / 1e3:.0f} GB/s (read + write)", flush=True)
    del copy_src, copy_dst

    shapes = [("q_a", 6144, 2048, 5), ("kv_a", 6144, 640, 5), ("q_b", 2048, 4096, 5), ("o_proj", 4096, 6144, 5),
              ("wq_b", 2048, 4096, 5), ("shared gate/up", 6144, 512, 5), ("shared down", 512, 6144, 5),
              ("dense gate/up", 6144, 3072, 4), ("dense down", 3072, 6144, 4), ("lm_head", 6144, 38784, 6)]
    for R in (rows if "linear" in parts else ()):
        print(f"-- EXL3 linear, {R} row(s)", flush=True)
        for name, k, n, bits in shapes:
            one = linear(k, n, bits)
            copies = max(1, min(64, int(512e6 // one.nbytes())))      # distinct weights: L2 never holds the next
            lins = [one] + [linear(k, n, bits) for _ in range(copies - 1)]
            x = torch.randn((R, k), device=dev).to(torch.bfloat16)
            y = torch.empty((R, n), device=dev, dtype=torch.bfloat16)
            xh = torch.empty((R, k), device=dev, dtype=torch.float16)
            sk = one.split[0]
            z = torch.empty((max(1, sk) * R * n,), device=dev, dtype=torch.float32)
            t = graph_time([(lambda lin=lin: lin(x, out=y, xh=xh, z=z if sk > 1 else None)) for lin in lins])
            mb = one.nbytes() / 1e6
            print(f"   {name:15s} K {k:5d} N {n:5d} {bits}b split {one.split} x{copies:2d}: {t:6.1f} us, {mb:6.2f} MB, "
                  f"{mb / t * 1e3:4.0f} GB/s", flush=True)
            del lins

    if "experts" not in parts:
        return
    print("-- routed experts (8 of 256, 3 bits, D 6144, I 512 a rank)", flush=True)
    E, D, I = 256, 6144, 512
    mats = {}
    for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
        mats[p] = [(torch.randint(-32768, 32767, (k // 16, n // 16, 48), dtype=torch.int16, device=dev),
                    (torch.randn(k, device=dev) * 0.1 + 1).half(), (torch.randn(n, device=dev) * 0.01).half())
                   for _ in range(E)]
    ex = x3experts.prepare(mats["gate"], mats["up"], mats["down"], "mul1", device=dev)
    per_expert = 3 * D * I * 3 / 8
    for R in rows:
        slots = 9
        x = torch.randn((R, D), device=dev).to(torch.bfloat16)
        g = torch.Generator(device="cpu").manual_seed(R)
        picks = []
        for _ in range(24):                     # a different set of experts each call, as successive layers read
            p = torch.stack([torch.cat([torch.randperm(E, generator=g)[:8], torch.tensor([E])]) for _ in range(R)])
            picks.append(p.to(torch.int32).to(dev))
        wts = torch.rand((R, slots), device=dev)
        uniq = sum(len(set(p[:, :8].flatten().tolist())) for p in picks) / len(picks)
        res = {}
        for label, cfg_gu, cfg_d in (("shipped (8,4,4,1)/(8,4,1,1)", None, None),
                                     ("pf2 (8,4,4,2)/(8,4,1,2)", (8, 4, 4, 2), (8, 4, 1, 2))):
            s = x3experts.Scratch(ex, R, slots, cfg_gu=cfg_gu, cfg_d=cfg_d, device=dev)
            o = torch.empty((len(picks), R, D), device=dev, dtype=torch.float32)
            t = graph_time([(lambda i=i, p=p: x3experts.routed(x, p, wts, ex, s, o[i], R, act_mode=x3experts.ACT_BF16))
                            for i, p in enumerate(picks)])
            res[label] = o.clone()
            mb = uniq * per_expert / 1e6
            print(f"   R={R} ({uniq:.1f} experts) {label}: {t:6.1f} us, {mb:5.1f} MB, {mb / t * 1e3:4.0f} GB/s",
                  flush=True)
        a, b = res.values()
        print(f"   R={R} pf2 bits == shipped: {torch.equal(a.view(torch.int32), b.view(torch.int32))}", flush=True)


if __name__ == "__main__":
    main()
