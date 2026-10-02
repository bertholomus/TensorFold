"""us a call of glm_moe_dsa's raw-token DSA selection (select.select_tokens) at decode windows past the dense limit,
by scored-token bucket, with the time of its scoring kernel alone. usage (one GPU): python3 tools/bench_select.py"""

from __future__ import annotations

import time

import torch
import triton

from tensorfold.families.glm_moe_dsa.cuda import select


def graph_us(fn, reps: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (10 * reps) * 1e6


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    H, D, topk = 32, 128, 2048
    cap = 1 << 20
    keys = torch.randn((cap, D), device=dev).to(torch.bfloat16)
    for R in (1, 4):
        qi = torch.randn((R, H * D), device=dev).to(torch.bfloat16)
        wts = torch.randn((R, H), device=dev).to(torch.bfloat16)
        tokens = torch.empty((R, topk + 1), dtype=torch.int32, device=dev)
        counts = torch.empty((R,), dtype=torch.int32, device=dev)
        for pos in (3000, 6677, 30000, 120000, 500000, 1000000):
            if pos + R > cap:
                continue
            bucket = select.sparse_bucket(pos, R)
            pos_dev = torch.tensor([pos], dtype=torch.int32, device=dev)
            whole = graph_us(lambda: select.select_tokens(qi, wts, keys, None, R, topk, pos_dev, tokens=tokens,
                                                          counts=counts, bucket=bucket))
            np_max = min(bucket, cap)
            scores = torch.empty((R, np_max), dtype=torch.float32, device=dev)
            scoring = graph_us(lambda: select._scores[(R, triton.cdiv(np_max, 64))](
                qi, wts, wts.stride(0), keys, keys, keys, scores, pos_dev, R, np_max, D ** -0.5, H ** -0.5, H=H,
                HP=32, D=D, BT=64, RB=1, KV8=False, QB=0, num_warps=4))
            print(f"R={R} pos {pos:7d} bucket {bucket:7d}: select_tokens {whole:8.1f} us (scores kernel {scoring:7.1f})",
                  flush=True)


if __name__ == "__main__":
    main()
