"""How much of a prompt chunk's routed-expert time is the grids' size: the grouped launches size every expert's member
groups for the busiest expert, and the shared expert (the last slot) has every row. Times on one real layer (rank 0's
half of a TP2 split, served kernels): all slots (as served), the routed slots alone (shared slot skipped), the shared
slot alone (routed slots skipped). If routed + shared alone is far below all slots, a grid sized per expert pays.

  python3 tools/dsv41/expert_shared_probe.py --model M [--layer 10] [--rows 2048] [--out F]   (one GPU, idle)
"""

import argparse
import json
import sys

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.cuda.exl3 import experts as X
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, load_block

    cfg = Cfg.read(a.model)
    lay = load_block(Shards(a.model), cfg, f"layers.{a.layer}", a.layer, 0, 2, cfg.n_routed)
    ex = lay.experts
    E, D, R = ex.count - 1, cfg.dim, a.rows
    s = X.Scratch(ex, rows=R, slots=cfg.topk + 1, prompt=True)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn((R, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    routed = torch.stack([torch.randperm(E, generator=g, device="cuda")[:cfg.topk] for _ in range(R)]).to(torch.int32)
    wts = torch.rand((R, cfg.topk + 1), generator=g, device="cuda")
    wts[:, -1] = 1.0
    skip = ex.count                                       # picks >= the expert count are skipped
    picks = {"all slots (served)": torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1),
             "routed slots alone": torch.cat([routed, torch.full((R, 1), skip, dtype=torch.int32, device="cuda")], 1),
             "shared slot alone": torch.cat([torch.full_like(routed, skip),
                                             torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1)}
    res: dict = {"rows": R, "times_ms": {}}
    for name, pick in picks.items():
        pick = pick.contiguous()
        run = lambda: X.routed(x, pick, wts, ex, s, None, R, limit=cfg.swiglu_limit, act_mode=X.ACT_F32)
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(a.iters):
            run()
        t1.record()
        torch.cuda.synchronize()
        res["times_ms"][name] = round(t0.elapsed_time(t1) / a.iters, 3)
        print(json.dumps({name: res["times_ms"][name]}), flush=True)
    t = res["times_ms"]
    res["all - (routed + shared)"] = round(t["all slots (served)"] - t["routed slots alone"] - t["shared slot alone"], 3)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
