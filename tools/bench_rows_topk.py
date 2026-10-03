"""The DSA top-k of a concurrent round's rows on one GB10: select._radix_topk (a program a row; the rows path at 16 rows
or more) against select._split_topk (each row's scores split over programs; fewer rows), ms a call and whether both
pick the same tokens in the same order. Scores are bf16-rounded (ties, as real indexer scores have them).

usage (one GPU, tf container): python3 tools/bench_rows_topk.py [ROWS=4,8,16,32,64] [TOKENS=8192,32768,131072]
"""

from __future__ import annotations

import sys
import time

import torch

TOPK = 2048


def timed(fn, reps: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main() -> None:
    from tensorfold.families.glm_moe_dsa.cuda import select

    rows = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "4,8,16,32,64").split(",")]
    sizes = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "8192,32768,131072").split(",")]
    g = torch.Generator(device="cuda").manual_seed(0)
    for n in sizes:
        for R in rows:
            scores = torch.randn((R, n), generator=g, device="cuda").to(torch.bfloat16).float()
            a = torch.zeros((R, TOPK + 1), dtype=torch.int32, device="cuda")
            b = torch.zeros_like(a)

            def radix():
                a.zero_()
                select._radix_topk[(R,)](scores, scores.stride(0), a, a.stride(0), n, K=TOPK, BLOCK=1024, num_warps=4)

            def split():
                b.zero_()
                select._split_topk(scores, b, TOPK, n)

            tr, ts = timed(radix), timed(split)
            same = torch.equal(a[:, :TOPK], b[:, :TOPK])
            sets = torch.equal(a[:, :TOPK].sort(1).values, b[:, :TOPK].sort(1).values)
            print(f"{R:3d} rows x {n:6d} tokens: radix {tr:6.3f} ms, split {ts:6.3f} ms; same tokens in order {same}, "
                  f"same sets {sets}", flush=True)


if __name__ == "__main__":
    main()
