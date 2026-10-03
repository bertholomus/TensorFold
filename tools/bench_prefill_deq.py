"""The EXL3 prompt GEMM's two modes at GLM-5.3's TP4 dense shapes on one GB10, ms a call: the rotated path (rot_in +
the fixed-tile GEMM with its H128 epilogue, its unpack apart) against the dequantized one (_dequant over the decoded
W_q, then cuBLAS bf16 or _gemm_deq on fp16 at a few tile settings), with what each step costs alone.

usage (one GPU, tf container): python3 tools/bench_prefill_deq.py [M=4096]
"""

from __future__ import annotations

import sys
import time

import torch
import triton

from tensorfold.cuda.exl3 import prefill

SHAPES = {"q_a": (6144, 2048), "q_b": (2048, 4096), "kv_a": (6144, 640), "o_proj": (4096, 6144),
          "shared gate/up": (6144, 1024), "shared down": (512, 6144), "dense gate/up": (6144, 6144),
          "dense down": (3072, 6144), "indexer wq_b": (2048, 4096)}
DEQ_TILES = [(128, 128, 64, 8, 3, 8), (128, 256, 64, 8, 3, 8), (128, 128, 32, 4, 4, 8), (64, 128, 64, 4, 4, 8),
             (256, 128, 64, 8, 3, 8)]


def timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    ws = prefill.Workspace()
    H = ws.hadamard("cuda")
    ext = prefill._ext()
    for name, (K, N) in SHAPES.items():
        x = (torch.randn((M, K), device="cuda") * 0.5).to(torch.bfloat16)
        wq = (torch.randn((K, N), device="cuda") * 0.5).half()
        suh = (torch.randn((K,), device="cuda").sign() * 0.05).half()
        svh = (torch.rand((N,), device="cuda") * 0.02).half()
        xh = torch.empty((M, K), dtype=torch.float16, device="cuda")
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        out32 = torch.empty((M, N), dtype=torch.float32, device="cuda")
        fl = 2 * M * K * N
        bm, bk, warps, stages, group = prefill.tiles(K, N, True)

        def rotated():
            ext.rot_in(x, suh, xh)
            prefill._gemm[(triton.cdiv(M, bm) * (N // 128),)](xh, wq, H, svh, svh, out, M, out.stride(0), K=K, N=N,
                                                              BM=bm, BK=bk, GROUP=group, HAS_BIAS=False,
                                                              SCALE=prefill.HAD_SCALE, num_warps=warps,
                                                              num_stages=stages)

        t_rot = timed(lambda: ext.rot_in(x, suh, xh))
        t_rotated = timed(rotated)
        wb = torch.empty((K, N), dtype=torch.bfloat16, device="cuda")
        wf = torch.empty((K, N), dtype=torch.float16, device="cuda")
        t_deq = timed(lambda: prefill._dequant[(K // 128, N // 128)](wq, wb, H, suh, svh, N=N, SCALE=prefill.DEQ_SCALE,
                                                                     num_warps=8))
        prefill._dequant[(K // 128, N // 128)](wq, wf, H, suh, svh, N=N, SCALE=prefill.DEQ_SCALE, num_warps=8)
        t_cublas = timed(lambda: torch.mm(x, wb, out=out))
        f32 = "-"
        try:
            f32 = f"{timed(lambda: torch.mm(x, wb, out_dtype=torch.float32, out=out32)):.2f}"
        except (TypeError, RuntimeError):
            pass
        line = (f"{name} {M}x{K}->{N}: rotated {t_rotated:.2f} ms (rot_in {t_rot:.2f}); dequant {t_deq:.2f} ms a "
                f"chunk; cuBLAS bf16 {t_cublas:.2f} ({fl / t_cublas / 1e9:.0f} TFLOPS), fp32 out {f32}; _gemm_deq fp16")
        for tbm, tbn, tbk, tw, ts, tg in DEQ_TILES:
            if N % tbn:
                continue

            def run():
                prefill._gemm_deq[(triton.cdiv(M, tbm) * (N // tbn),)](x, x.stride(0), wf, svh, out32, M,
                                                                       out32.stride(0), K=K, N=N, BM=tbm, BN=tbn,
                                                                       BK=tbk, GROUP=tg, HAS_BIAS=False, num_warps=tw,
                                                                       num_stages=ts)
            try:
                ms = timed(run)
                line += f" {tbm}/{tbn}/{tbk}/{tw}/{ts} {ms:.2f} ({fl / ms / 1e9:.0f})"
            except Exception:                              # noqa: BLE001  (shared memory past the limit)
                line += f" {tbm}/{tbn}/{tbk}/{tw}/{ts} -"
        print(line, flush=True)


if __name__ == "__main__":
    main()
