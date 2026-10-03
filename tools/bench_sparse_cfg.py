"""_sparse_rows_pe launch settings on one GB10 for a prompt chunk over a bf16 latent (GLM-5.3 TP4 rank shapes): ms
for (warps, stages, key tile) settings, each checked against the shipped setting's output (rel. error).

usage (one GPU, tf container): python3 tools/bench_sparse_cfg.py [ROWS=2048] [CONTEXT=38000] [W/S/KT,...]
"""

from __future__ import annotations

import sys
import time

import torch
import triton

H, LW, PW, TOPK = 16, 512, 64, 2048


def timed(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    from tensorfold.families.glm5_next.cuda.latent import CHUNK, HB
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 38000
    g = torch.Generator().manual_seed(0)
    lat = (torch.randn((n, LW), generator=g) * 2).to(torch.bfloat16).cuda()
    pc = torch.randn((n, PW), generator=g).to(torch.bfloat16).cuda()
    pos = n - R
    tokens = torch.stack([torch.randperm(pos + r, generator=g)[:TOPK].sort().values for r in range(R)])
    tokens = torch.cat([tokens, torch.full((R, 1), -1)], 1).to(torch.int32).cuda()
    counts = torch.full((R,), TOPK, dtype=torch.int32, device="cuda")
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    W = tokens.shape[1]
    ref = torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(ref)

    def run(o, warps, stages, ktt):
        mla_pe._sparse_rows_pe[(R, triton.cdiv(H, HB))](qa, qp, lat, lat, lat, pc, tokens, counts, o, W=W, H=H, LW=LW,
                                                         PW=PW, CH=CHUNK, SCALE=256 ** -0.5, HBT=HB, KTT=ktt, KV8=False,
                                                         QB=0, num_warps=warps, num_stages=stages)

    run(ref, 4, 3, mla_pe.KT)
    print(f"{R} rows x {TOPK} tokens, context {n}; shipped (4 warps, 3 stages, KT {mla_pe.KT}): "
          f"{timed(lambda: run(out, 4, 3, mla_pe.KT)):.2f} ms", flush=True)
    cfgs = [tuple(int(v) for v in c.split("/")) for c in (sys.argv[3] if len(sys.argv) > 3 else
            "4/2/32,8/3/32,8/2/64,4/3/16,4/2/16,4/4/16,8/3/16,8/2/16,2/3/16,2/2/16,1/3/16").split(",")]
    for warps, stages, ktt in cfgs:
        try:
            ms = timed(lambda: run(out, warps, stages, ktt))
            err = float((out.double() - ref.double()).norm() / ref.double().norm())
            print(f"  warps {warps}, stages {stages}, KT {ktt}: {ms:.2f} ms (rel. diff {err:.1e})", flush=True)
        except Exception as e:                            # noqa: BLE001  (shared memory past the limit)
            print(f"  warps {warps}, stages {stages}, KT {ktt}: {type(e).__name__}", flush=True)


if __name__ == "__main__":
    main()
