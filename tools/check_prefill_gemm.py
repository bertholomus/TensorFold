"""The EXL3 prompt GEMM (cuda/exl3/prefill._gemm) at other launch settings: bits against the shipped (BM 128, BK 32,
8 warps, 4 stages, group 8) and ms a call, at GLM-5.3 TP4 projection shapes, 2,048 rows. A setting that keeps every
row's bits at every shape can replace tiles() without changing the prompt path's arithmetic.
usage (one GPU, tf container): python3 tools/check_prefill_gemm.py [ROWS=2048]
"""

from __future__ import annotations

import sys
import time

import torch
import triton

from tensorfold.cuda.exl3 import prefill

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SHAPES = [(6144, 2048), (2048, 4096), (4096, 6144), (6144, 512), (512, 6144), (6144, 640)]
SETTINGS = [(128, 32, 8, 4, 8), (128, 64, 8, 3, 8), (128, 32, 4, 4, 8), (64, 32, 4, 4, 8), (128, 64, 4, 3, 8),
            (256, 32, 8, 3, 8), (128, 32, 8, 5, 8), (64, 64, 4, 3, 8), (128, 32, 8, 4, 4)]


def run(xh, wq, h, svh, out, k, n, s):
    bm, bk, warps, stages, group = s
    m = xh.shape[0]
    prefill._gemm[(triton.cdiv(m, bm) * (n // prefill.BN),)](xh, wq, h, svh, svh, out, m, out.stride(0), K=k, N=n,
                                                            BM=bm, BK=bk, GROUP=group, HAS_BIAS=False,
                                                            SCALE=prefill.HAD_SCALE, num_warps=warps,
                                                            num_stages=stages)


def main() -> None:
    torch.manual_seed(0)
    ws = prefill.Workspace()
    h = ws.hadamard("cuda")
    totals = {s: 0.0 for s in SETTINGS}
    same = {s: True for s in SETTINGS}
    for k, n in SHAPES:
        xh = (torch.randn((ROWS, k), device="cuda") * 0.3).half()
        wq = (torch.randn((k, n), device="cuda") * 0.02).half()
        svh = (torch.rand((n,), device="cuda") + 0.5).half()
        ref = torch.empty((ROWS, n), dtype=torch.float32, device="cuda")
        run(xh, wq, h, svh, ref, k, n, SETTINGS[0])
        line = f"K={k:5d} N={n:5d}:"
        for s in SETTINGS:
            out = torch.empty_like(ref)
            try:
                run(xh, wq, h, svh, out, k, n, s)
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(10):
                    run(xh, wq, h, svh, out, k, n, s)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) / 10 * 1e3
            except Exception as exc:                  # noqa: BLE001  (shared memory past the device's)
                line += f"  {s}: {type(exc).__name__}"
                same[s] = False
                continue
            eq = torch.equal(out.view(torch.int32), ref.view(torch.int32))
            same[s] &= eq
            totals[s] += ms
            line += f"  {s[:4]} {ms:.3f}{'' if eq else ' DIFF'}"
        print(line, flush=True)
    for s in SETTINGS:
        print(f"{s}: bit-equal at every shape {same[s]}, total {totals[s]:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
