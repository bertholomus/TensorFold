"""Prompt chunks' mHC through hc_pre2 (switch "hc_pf": 8 rows a program share each mixing weight tile, the attention
post fused into the FFN mixes) against the kernels they replace (hc_post then hc_pre, one row a program): the posted
streams, the normed input, pre / post / comb, bit for bit, at prompt-chunk row counts; then their times.

  python3 tools/dsv41/hc_prefill_test.py [--out F]      (one GPU, lane idle)
Exits 1 when a check fails.
"""

import argparse
import json
import sys
import time

import torch


def inputs(R, D, gen, dev):
    h = (torch.randn((R, 4, D), generator=gen) * 0.7).to(torch.bfloat16).to(dev)
    fn = (torch.randn((24, 4 * D), generator=gen) * 0.02).to(dev)
    scale = (torch.rand((3,), generator=gen) + 0.5).to(dev)
    base = (torch.randn((24,), generator=gen) * 0.5).to(dev)
    pre_in = (torch.rand((R, 4), generator=gen) * 0.9 + 0.05).to(dev)
    norm = (torch.rand((D,), generator=gen) + 0.5).to(torch.bfloat16).to(dev)
    g = (torch.randn((2, R, D), generator=gen) * 0.3).to(dev)
    post = (torch.rand((R, 4), generator=gen) * 2.0).to(dev)
    comb = torch.rand((R, 4, 4), generator=gen)
    comb = (comb / comb.sum(1, keepdim=True)).to(dev)
    return h, fn, scale, base, pre_in, norm, g, post, comb


def outs(R, D, dev, nb):
    return (torch.empty((R, D), dtype=torch.bfloat16, device=dev), torch.empty((R, 4), device=dev),
            torch.empty((R * nb * 32,), device=dev))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    dev = "cuda"
    gen = torch.Generator().manual_seed(5)
    D, eps, hc_eps, iters = 5120, 1e-6, 1e-6, 20
    res: dict = {"checks": {}, "times_ms": {}}
    ok = res["checks"]
    for R in (2048, 1919, 128, 33, 17):
        h, fn, scale, base, pre_in, norm, g, post0, comb0 = inputs(R, D, gen, dev)
        # the attention post, then the FFN mixes: hc_post + hc_pre (old) vs hc_pre2 with gathered (new)
        h1, post1, comb1 = h.clone(), post0.clone(), comb0.clone()
        x1, pre1, part1 = outs(R, D, dev, K.HC_BLOCKS)
        K.hc_post(g, h1, post1, comb1, h1)
        K.hc_pre(h1, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x1, pre1, post1, comb1, part1)
        h2, post2, comb2 = h.clone(), post0.clone(), comb0.clone()
        x2, pre2, part2 = outs(R, D, dev, K.HC_BLOCKS)
        h_out = torch.empty_like(h2)
        src = K.hc_pre2(h2, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x2, pre2, post2, comb2, part2,
                        gathered=g, h_out=h_out)
        torch.cuda.synchronize()
        ok[f"fused/{R}/streams"] = bool(torch.equal(src, h1)) and src.data_ptr() == h_out.data_ptr()
        ok[f"fused/{R}/x"] = bool(torch.equal(x2, x1))
        ok[f"fused/{R}/pre_post_comb"] = bool(torch.equal(pre2, pre1) and torch.equal(post2, post1)
                                              and torch.equal(comb2, comb1))
        ok[f"fused/{R}/input_untouched"] = bool(torch.equal(h2, h))
        # the attention mixes alone: hc_pre (old) vs hc_pre2 without gathered (new)
        post3, comb3, post4, comb4 = post0.clone(), comb0.clone(), post0.clone(), comb0.clone()
        x3, pre3, part3 = outs(R, D, dev, K.HC_BLOCKS)
        x4, pre4, part4 = outs(R, D, dev, K.HC_BLOCKS)
        K.hc_pre(h, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x3, pre3, post3, comb3, part3)
        K.hc_pre2(h, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x4, pre4, post4, comb4, part4)
        torch.cuda.synchronize()
        ok[f"plain/{R}/x_pre_post_comb"] = bool(torch.equal(x4, x3) and torch.equal(pre4, pre3)
                                                and torch.equal(post4, post3) and torch.equal(comb4, comb3))

    def timed(fn_, reps=20):
        for _ in range(3):
            fn_()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn_()
            torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t0))
        return round(sorted(ts)[len(ts) // 2], 3)

    R = 2048
    h, fn, scale, base, pre_in, norm, g, post0, comb0 = inputs(R, D, gen, dev)
    x, pre, part = outs(R, D, dev, K.HC_BLOCKS)
    hb = h.clone()
    hb2 = torch.empty_like(h)
    post, comb = post0.clone(), comb0.clone()

    def old_pair():
        K.hc_post(g, hb, post, comb, hb)
        K.hc_pre(hb, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x, pre, post, comb, part)

    def new_pair():
        K.hc_pre2(hb, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x, pre, post, comb, part, gathered=g,
                  h_out=hb2)

    res["times_ms"]["2048 rows: post + mixes (attention post, FFN mixes)"] = {"old": timed(old_pair),
                                                                             "new": timed(new_pair)}
    res["times_ms"]["2048 rows: mixes alone"] = {
        "old": timed(lambda: K.hc_pre(hb, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x, pre, post, comb,
                                      part)),
        "new": timed(lambda: K.hc_pre2(hb, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x, pre, post, comb,
                                       part))}
    res["passed"] = all(ok.values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
