"""Every (warps, stages) launch of the new dense MLA kernel vs the old full-tile kernel: bits over the 192-case matrix and
graph-timed us a call at decode positions. usage: python3 tools/check_mla_cfg.py"""

import time

import torch
import triton

import check_mla_tiles as base
from tensorfold.families.glm5_next.cuda.latent import CHUNK, LatentScratch


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    H, LW, PW, cap = 16, 512, 64, 2560 + 8
    nch_cap = triton.cdiv(cap, CHUNK)
    cache = (torch.randn((cap, LW), device=dev) * 0.5).to(torch.bfloat16)
    pcache = (torch.randn((cap, PW), device=dev) * 0.5).to(torch.bfloat16)
    s = LatentScratch(8, H, nch_cap, dev, lw=LW, part_rows=8)
    scale = 256 ** -0.5
    cases = []
    for pos in (0, 1, 7, 31, 32, 33, 160, 511, 512, 513, 1000, 1535, 2040, 2047, 2048, 2550):
        for R in (1, 2, 3, 4, 6, 8):
            if pos + R <= cap:
                cases.append((pos, R, torch.randn((R, H, LW), device=dev).to(torch.bfloat16),
                              torch.randn((R, H, PW), device=dev).to(torch.bfloat16)))
    for warps, stages in ((8, 1), (4, 1), (4, 2), (4, 3), (2, 2), (2, 3), (8, 3), (4, 4)):
        fn = base.staged_attention(stages, warps)
        bad = n = 0
        try:
            for pos, R, qa, qp in cases:
                pos_dev = torch.tensor([pos], dtype=torch.int32, device=dev)
                for nch in sorted({triton.cdiv(pos + R, CHUNK), nch_cap}):
                    a = torch.empty((R, H, LW), dtype=torch.bfloat16, device=dev)
                    b = torch.empty((R, H, LW), dtype=torch.bfloat16, device=dev)
                    base.old_attention(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch, out=a)
                    fn(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch, out=b)
                    torch.cuda.synchronize()
                    n += 1
                    bad += not torch.equal(a.view(torch.int16), b.view(torch.int16))
            times = []
            for pos in (160, 1000, 2040):
                qa = torch.randn((1, H, LW), device=dev).to(torch.bfloat16)
                qp = torch.randn((1, H, PW), device=dev).to(torch.bfloat16)
                pos_dev = torch.tensor([pos], dtype=torch.int32, device=dev)
                out = torch.empty((1, H, LW), dtype=torch.bfloat16, device=dev)
                fn(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch_cap, out=out)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(50):
                        fn(qa, qp, cache, pcache, pos_dev, s, scale=scale, nch=nch_cap, out=out)
                g.replay()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(10):
                    g.replay()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t) / 500 * 1e6)
            print(f"warps {warps} stages {stages}: {n} cases, {bad} differ; us at pos 160/1000/2040: "
                  + " / ".join(f"{x:.1f}" for x in times), flush=True)
        except Exception as exc:          # noqa: BLE001  (a launch that does not fit shared memory)
            print(f"warps {warps} stages {stages}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)


if __name__ == "__main__":
    main()
