"""Can an L2 prefetch on a side stream, run while the main stream waits (an all-gather's ~35 us), speed up the next
EXL3 linear? One GB10, cold weights (64 distinct q_a-shaped matrices), CUDA graphs like the decode forward's.

usage (one GPU, in the tf container): python3 tools/bench_prefetch.py

Result (2026-10-01): alone, yes (q_a 42 -> 22 us after an overlapped read). In the TP4 forward, no: reading the MoE's
router and shared expert during the attention's all-gather and the next q_a / kv_a during the MoE's, at 2-12 MB and
32-96 programs a window, left R=1 at 43.1-45.3 ms against 43.3 without and R=4 at 66.4-67.4 against 67.0 (on GB10 the
NIC, the NCCL proxy and the GPU share one DRAM, so the all-gathers slowed by what the reads saved). Not shipped.
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from tensorfold.cuda.exl3.linear import Exl3Linear


@triton.jit
def _touch(X, n, OUT, BLOCK: tl.constexpr, STEP: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for start in range(pid * BLOCK, n, STEP):
        i = start + tl.arange(0, BLOCK)
        acc += tl.load(X + i, mask=i < n, other=0, eviction_policy="evict_last")
    tl.store(OUT + pid, tl.sum(acc, axis=0))


def touch(words: torch.Tensor, out: torch.Tensor, programs: int, block: int, warps: int, stages: int) -> None:
    n = words.numel()
    _touch[(programs,)](words, n, out, BLOCK=block, STEP=programs * block, num_warps=warps, num_stages=stages)


def linear(k: int, n: int, bits: int) -> Exl3Linear:
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), dtype=torch.int16)
    return Exl3Linear.from_tensors(trellis, torch.ones(k).half(), torch.full((n,), 0.01).half(), "mul1", device="cuda")


def timed(fn, reps=10) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e6


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    out = torch.zeros((1024,), dtype=torch.int32, device=dev)
    side = torch.cuda.Stream()
    for name, k, n in (("q_a", 6144, 2048), ("o_proj", 4096, 6144), ("shared gate", 6144, 512)):
        lins = [linear(k, n, 5) for _ in range(48)]
        x = torch.randn((1, k), device=dev).to(torch.bfloat16)
        y = torch.empty((1, n), device=dev, dtype=torch.bfloat16)
        xh = torch.empty((1, k), device=dev, dtype=torch.float16)
        sk = lins[0].split[0]
        z = torch.empty((max(1, sk) * n,), device=dev, dtype=torch.float32)
        mb = lins[0].nbytes() / 1e6
        cycles = 35 * 1900                                        # ~35 us of spinning at ~1.9 GHz

        def wait_then_linear():
            for lin in lins:
                torch.cuda._sleep(cycles)
                lin(x, out=y, xh=xh, z=z if sk > 1 else None)

        def sleep_only():
            for _ in lins:
                torch.cuda._sleep(cycles)

        def touch_only(programs, block, warps, stages):
            def run():
                for lin in lins:
                    touch(lin.words.view(-1), out, programs, block, warps, stages)
            return run

        def overlapped(programs, block, warps, stages):
            def run():
                main = torch.cuda.current_stream()
                for lin in lins:
                    fork = torch.cuda.Event()
                    fork.record(main)
                    with torch.cuda.stream(side):
                        side.wait_event(fork)
                        touch(lin.words.view(-1), out, programs, block, warps, stages)
                        done = torch.cuda.Event()
                        done.record(side)
                    torch.cuda._sleep(cycles)
                    main.wait_event(done)
                    lin(x, out=y, xh=xh, z=z if sk > 1 else None)
            return run

        base = timed(wait_then_linear) / len(lins)
        sl = timed(sleep_only) / len(lins)
        print(f"{name} ({mb:.2f} MB): wait {sl:.1f} us; wait + cold linear {base:.1f} us (linear {base - sl:.1f})",
              flush=True)
        for programs, block, warps, stages in ((48, 1024, 4, 1), (48, 2048, 4, 3), (96, 1024, 4, 3), (24, 2048, 4, 3)):
            tt = timed(touch_only(programs, block, warps, stages)) / len(lins)
            ov = timed(overlapped(programs, block, warps, stages)) / len(lins)
            print(f"   touch {programs}x{block} w{warps} s{stages}: alone {tt:.1f} us ({mb / tt * 1e3:.0f} GB/s); "
                  f"overlapped with the wait, wait + linear {ov:.1f} us (linear {ov - sl:.1f})", flush=True)
        del lins


if __name__ == "__main__":
    main()
