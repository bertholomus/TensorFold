"""The EXL3 prompt GEMM (exl3.prefill._gemm) at GLM-5.3's TP4 dense shapes on one GB10: ms and TFLOPS for tile
settings (rows a program, K step, warps, stages, raster group), against cuBLAS on the same fp16 operands.

usage (one GPU, tf container): python3 tools/bench_prefill_gemm.py [M=2048]
"""

from __future__ import annotations

import sys
import time

import torch
import triton

from tensorfold.cuda.exl3 import prefill

SHAPES = {"q_a": (6144, 2048), "q_b": (2048, 4096), "kv_a": (6144, 640), "o_proj": (4096, 6144),
          "shared gate/up": (6144, 1024), "shared down": (512, 6144), "dense gate/up": (6144, 6144),
          "dense down": (3072, 6144)}
TILES = [(128, 32, 8, 4, 8), (128, 64, 8, 3, 8), (128, 64, 4, 3, 8), (64, 64, 4, 4, 8), (128, 32, 4, 4, 8),
         (256, 32, 8, 3, 8), (128, 64, 8, 4, 8), (64, 32, 4, 4, 8)]


def timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    ws = prefill.Workspace()
    H = ws.hadamard("cuda")
    for name, (K, N) in SHAPES.items():
        x = (torch.randn((M, K), device="cuda") * 0.1).half()
        w = (torch.randn((K, N), device="cuda") * 0.02).half()
        svh = torch.ones((N,), device="cuda").half()
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        fl = 2 * M * K * N
        line = f"{name} {M}x{K}->{N}: cuBLAS {fl / timed(lambda: x @ w) / 1e9:.0f} TFLOPS;"
        for bm, bk, warps, stages, group in TILES:
            def run():
                prefill._gemm[(triton.cdiv(M, bm) * (N // 128),)](x, w, H, svh, svh, out, M, out.stride(0), K=K, N=N,
                                                                  BM=bm, BK=bk, GROUP=group, HAS_BIAS=False,
                                                                  SCALE=prefill.HAD_SCALE, num_warps=warps,
                                                                  num_stages=stages)
            try:
                ms = timed(run)
                line += f" {bm}/{bk}/{warps}/{stages} {fl / ms / 1e9:.0f}"
            except Exception as err:                      # noqa: BLE001  (shared memory past the limit)
                line += f" {bm}/{bk}/{warps}/{stages} -"
        print(line, flush=True)


if __name__ == "__main__":
    main()
