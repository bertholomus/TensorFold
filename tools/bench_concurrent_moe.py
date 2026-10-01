"""Shared expert on a side stream while the routed experts run, vs one after the other (one GB10, decode rows, cold
weights cycling through 24 layers' worth), and small-kernel dispatch cost in a graph.
usage (one GPU, in the tf container): python3 tools/bench_concurrent_moe.py"""

from __future__ import annotations

import time

import torch

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3.linear import Exl3Linear


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


def linear(k: int, n: int, bits: int) -> Exl3Linear:
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), dtype=torch.int16)
    return Exl3Linear.from_tensors(trellis, torch.ones(k).half(), torch.full((n,), 0.01).half(), "mul1", device="cuda")


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    E, D, I, L = 256, 6144, 512, 24
    layers = []
    for _ in range(L):
        mats = {}
        for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
            mats[p] = [(torch.randint(-32768, 32767, (k // 16, n // 16, 48), dtype=torch.int16, device=dev),
                        torch.ones(k, device=dev).half(), torch.full((n,), 0.01, device=dev).half()) for _ in range(E)]
        ex = x3experts.prepare(mats["gate"], mats["up"], mats["down"], "mul1", device=dev)
        layers.append((ex, linear(D, I, 5), linear(D, I, 5), linear(I, D, 5)))
    side = torch.cuda.Stream()
    for R in (1, 4):
        slots = 9
        x = torch.randn((R, D), device=dev).to(torch.bfloat16)
        g = torch.Generator(device="cpu").manual_seed(R)
        picks = [torch.stack([torch.cat([torch.randperm(E, generator=g)[:8], torch.tensor([E])]) for _ in range(R)])
                 .to(torch.int32).to(dev) for _ in range(L)]
        wts = torch.rand((R, slots), device=dev)
        s = x3experts.Scratch(layers[0][0], R, slots, device=dev)
        part = torch.empty((R, D), device=dev, dtype=torch.float32)
        go = torch.empty((R, I), device=dev, dtype=torch.bfloat16)
        uo = torch.empty((R, I), device=dev, dtype=torch.bfloat16)
        sy = torch.empty((R, D), device=dev, dtype=torch.float32)
        xh = torch.empty((R, D), device=dev, dtype=torch.float16)
        xh2 = torch.empty((R, I), device=dev, dtype=torch.float16)
        z = torch.empty((8 * R * D,), device=dev, dtype=torch.float32)

        def shared(layer):
            _, gate, up, down = layer
            gate(x, out=go, xh=xh, z=z)
            up(x, out=uo, xh=xh, z=z)
            down(go, out=sy, xh=xh2, z=z)

        def routed(i, layer):
            x3experts.routed(x, picks[i], wts, layer[0], s, part, R, act_mode=x3experts.ACT_BF16)

        def sequential(i, layer):
            def run():
                routed(i, layer)
                shared(layer)
            return run

        def concurrent(i, layer):
            def run():
                main = torch.cuda.current_stream()
                fork = torch.cuda.Event()
                fork.record(main)
                with torch.cuda.stream(side):
                    side.wait_event(fork)
                    shared(layer)
                    done = torch.cuda.Event()
                    done.record(side)
                routed(i, layer)
                main.wait_event(done)
            return run

        t_seq = graph_us([sequential(i, l) for i, l in enumerate(layers)])
        t_con = graph_us([concurrent(i, l) for i, l in enumerate(layers)])
        t_r = graph_us([(lambda i=i, l=l: routed(i, l)) for i, l in enumerate(layers)])
        t_s = graph_us([(lambda l=l: shared(l)) for l in layers])
        print(f"R={R}: routed {t_r:.1f} us + shared {t_s:.1f} us; one after the other {t_seq:.1f} us; "
              f"shared on a side stream {t_con:.1f} us a layer", flush=True)

    a = torch.zeros((256,), device=dev)
    us = graph_us([lambda: a.add_(1.0)] * 3500)
    print(f"a tiny kernel in a graph of 3,500: {us:.2f} us each", flush=True)


if __name__ == "__main__":
    main()
