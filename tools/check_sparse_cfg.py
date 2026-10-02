"""Launch settings of glm_moe_dsa's sparse MLA kernel (rows past the dense limit) against the shipped 8 warps, 1 stage:
bits over random selections and graph-timed us a call. usage (one GPU, in the tf container): python3 tools/check_sparse_cfg.py"""

from __future__ import annotations

import time

import torch
import triton

from tensorfold.families.glm_moe_dsa.cuda import mla_pe
from tensorfold.families.glm5_next.cuda.latent import CHUNK, HB, KT, _merge


def sparse(stages: int, warps: int):
    def run(qa, qp, cache, pcache, tokens, counts, out, scale, scratch):
        R, H, LW = qa.shape
        PW = qp.shape[2]
        W = tokens.shape[1]
        nch = triton.cdiv(W, CHUNK)
        po, pm, pl = scratch
        mla_pe._sparse_chunks_pe[(R, triton.cdiv(H, HB), nch)](qa, qp, cache, cache, cache, pcache, tokens, counts,
                                                               po, pm, pl, R, W=W, H=H, LW=LW, PW=PW, CH=CHUNK,
                                                               SCALE=scale, HBT=HB, KTT=KT, KV8=False, QB=0,
                                                               num_warps=warps, num_stages=stages)
        _merge[(R, H)](po, pm, pl, out, counts, R, H=H, LW=LW, NCH=nch, SPARSE=True, num_warps=4)
    return run


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    H, LW, PW, topk = 16, 512, 64, 2048
    cap = 131072
    cache = (torch.randn((cap, LW), device=dev) * 0.5).to(torch.bfloat16)
    pcache = (torch.randn((cap, PW), device=dev) * 0.5).to(torch.bfloat16)
    W = topk + 1
    scale = 256 ** -0.5
    cases = []
    for pos in (2048, 3000, 6677, 40000, 131000):
        for R in (1, 2, 4, 6):
            tokens = torch.full((R, W), -1, dtype=torch.int32, device=dev)
            counts = torch.zeros((R,), dtype=torch.int32, device=dev)
            for r in range(R):
                visible = pos + r + 1
                n = min(topk, visible)
                pick = torch.randperm(visible, device=dev)[:n].sort().values.to(torch.int32)
                tokens[r, :n] = pick
                counts[r] = n if visible > topk else 0
            cases.append((pos, R, tokens, counts, torch.randn((R, H, LW), device=dev).to(torch.bfloat16),
                          torch.randn((R, H, PW), device=dev).to(torch.bfloat16)))
    nmax = triton.cdiv(W, CHUNK) * 6 * H
    scratch = (torch.empty((nmax * LW,), device=dev), torch.empty((nmax,), device=dev), torch.empty((nmax,), device=dev))
    ref = sparse(1, 8)
    for warps, stages in ((8, 1), (4, 1), (4, 2), (4, 3), (8, 2), (8, 3)):
        fn = sparse(stages, warps)
        bad = 0
        try:
            for pos, R, tokens, counts, qa, qp in cases:
                a = torch.zeros((R, H, LW), dtype=torch.bfloat16, device=dev)
                b = torch.zeros((R, H, LW), dtype=torch.bfloat16, device=dev)
                ref(qa, qp, cache, pcache, tokens, counts, a, scale, scratch)
                fn(qa, qp, cache, pcache, tokens, counts, b, scale, scratch)
                torch.cuda.synchronize()
                bad += not torch.equal(a.view(torch.int16), b.view(torch.int16))
            times = []
            for R in (1, 4):
                pos, _, tokens, counts, qa, qp = next(c for c in cases if c[0] == 40000 and c[1] == R)
                out = torch.empty((R, H, LW), dtype=torch.bfloat16, device=dev)
                fn(qa, qp, cache, pcache, tokens, counts, out, scale, scratch)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(50):
                        fn(qa, qp, cache, pcache, tokens, counts, out, scale, scratch)
                g.replay()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(10):
                    g.replay()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t) / 500 * 1e6)
            print(f"warps {warps} stages {stages}: {len(cases)} cases, {bad} differ; us R=1 / R=4 at 40k: "
                  + " / ".join(f"{x:.1f}" for x in times), flush=True)
        except Exception as exc:             # noqa: BLE001  (a launch past shared memory)
            print(f"warps {warps} stages {stages}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)


if __name__ == "__main__":
    main()
