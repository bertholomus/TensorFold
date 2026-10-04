"""A few eager decode-window calls of the routed experts (16 rows, a 24-expert pool a stream) for ncu: the grouped
gate/up and down launches of layers past the first ones (cold weights).

usage (one GPU, a container with --cap-add SYS_ADMIN):
  ncu --kernel-name regex:grouped_kernel --launch-skip 8 --launch-count 2 --set full python3 tools/ncu_c4.py [R=16]
"""

from __future__ import annotations

import sys

import torch

from bench_c4_kernels import BITS, D, E, I, K, picks_for


def main() -> None:
    from tensorfold.cuda.exl3 import experts as x3

    R = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    L = 12
    torch.manual_seed(0)
    layers = []
    for _ in range(L):
        mats = {}
        for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
            mats[p] = [(torch.randint(-32768, 32767, (k // 16, n // 16, 16 * BITS), dtype=torch.int16, device="cuda"),
                        (torch.randn(k, device="cuda") * 0.1 + 1).half(),
                        (torch.randn(n, device="cuda") * 0.01).half()) for _ in range(E)]
        layers.append(x3.prepare(mats["gate"], mats["up"], mats["down"], "mul1", device="cuda"))
    picks = picks_for(L, R, 24, R * 1000 + 24)
    x = torch.randn((R, D), device="cuda").to(torch.bfloat16)
    wts = torch.rand((R, K + 1), device="cuda")
    sc = x3.Scratch(layers[0], R, K + 1, device="cuda")
    out = torch.empty((R, D), device="cuda", dtype=torch.float32)
    for i, ex in enumerate(layers):
        x3.routed(x, picks[i], wts, ex, sc, out, R, act_mode=x3.ACT_BF16, kernel="mma")
    torch.cuda.synchronize()
    print("distinct experts a layer:", sum(len(set(p[:, :K].reshape(-1).tolist())) for p in picks) / L, flush=True)


if __name__ == "__main__":
    main()
