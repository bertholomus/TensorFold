"""Prompt chunks' routed experts on one real layer (rank 0's half of a TP2 split): the down projection through the mma
kernel at 2 or 6 k tiles a step (TF_EXL3_MMA_KB2 / TF_EXL3_MMA_KB6, experts.MMA_KB2 / MMA_KB6) against grouped_rows
(switches off) and against kernel "rows" for both projections: the same fp32 outputs bit for bit, which kernel each
call took, and the times.

  python3 tools/dsv41/expert_prompt_test.py --model M [--layer 10] [--rows 2048,1000,64] [--out F]   (one GPU, idle)
Exits 1 when a check fails.
"""

import argparse
import json
import sys

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--rows", default="2048,1000,64")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--require-faster", action="store_true",
                   help="also fail unless the 2-a-step down is faster than the served path at the largest row count")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.cuda.exl3 import experts as X
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, load_block

    cfg = Cfg.read(a.model)
    lay = load_block(Shards(a.model), cfg, f"layers.{a.layer}", a.layer, 0, 2, cfg.n_routed)
    ex = lay.experts
    E, D = ex.count - 1, cfg.dim
    ext = X._ext()
    calls = {}
    for name in ("grouped_rows", "grouped_mma"):                 # count which kernel each projection took
        f = getattr(ext, name)

        def wrap(*args, _f=f, _n=name):
            calls[_n] = calls.get(_n, 0) + 1
            return _f(*args)
        setattr(ext, name, wrap)
    g = torch.Generator(device="cuda").manual_seed(0)
    rows = [int(r) for r in a.rows.split(",")]
    s = X.Scratch(ex, rows=max(rows), slots=cfg.topk + 1, cfg_gu=(8, 4, 4, 1), cfg_d=(8, 4, 1, 1), prompt=True)
    res: dict = {"checks": {}, "times_ms": {}, "kernels": {}}
    variants = {"rows (both)": ("rows", False, False), "mma gate/up + rows down (served)": ("mma", False, False),
                "mma gate/up + mma down at 2 a step": ("mma", True, False),
                "mma gate/up + mma down at 6 a step": ("mma", False, True)}
    for R in rows:
        x = (torch.randn((R, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        pick = torch.stack([torch.randperm(E, generator=g, device="cuda")[:cfg.topk] for _ in range(R)]).to(torch.int32)
        pick = torch.cat([pick, torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1).contiguous()
        wts = torch.rand((R, cfg.topk + 1), generator=g, device="cuda")
        wts[:, -1] = 1.0
        outs = {}
        for name, (kern, kb2, kb6) in variants.items():
            X.MMA_KB2, X.MMA_KB6 = kb2, kb6
            run = lambda: X.routed(x, pick, wts, ex, s, None, R, limit=cfg.swiglu_limit, act_mode=X.ACT_F32,
                                   kernel=kern)
            calls.clear()
            out = run().clone()
            res["kernels"][f"R{R} {name}"] = dict(calls)
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(a.iters):
                run()
            t1.record()
            torch.cuda.synchronize()
            res["times_ms"][f"R{R} {name}"] = round(t0.elapsed_time(t1) / a.iters, 3)
            outs[name] = out
        base = outs["rows (both)"]
        for name in variants:
            res["checks"][f"R{R} {name} == rows"] = bool(torch.equal(outs[name], base))
        for step in (2, 6):
            k = res["kernels"][f"R{R} mma gate/up + mma down at {step} a step"]
            res["checks"][f"R{R} down took mma at {step} a step"] = k.get("grouped_mma", 0) == 2 and "grouped_rows" not in k
        print(json.dumps({k: v for k, v in res["times_ms"].items() if k.startswith(f"R{R} ")}), flush=True)
    if a.require_faster:
        R = max(rows)
        res["checks"][f"R{R} faster than served"] = (min(res["times_ms"][f"R{R} mma gate/up + mma down at {st} a step"]
                                                         for st in (2, 6))
                                                     < res["times_ms"][f"R{R} mma gate/up + rows down (served)"])
    res["passed"] = all(res["checks"].values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
