"""One prompt chunk's routed experts on one real layer (rank 0's half of a TP2 split), as served (gate / up through
grouped_mma, down through grouped_rows): a warm-up call, then one call between cudaProfilerStart / Stop, for
ncu --profile-from-start off (tools/dsv41/expert_ncu.sh).

  python3 tools/dsv41/expert_one.py --model M [--layer 10] [--rows 2048] [--out F]
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
    pick = torch.stack([torch.randperm(E, generator=g, device="cuda")[:cfg.topk] for _ in range(R)]).to(torch.int32)
    pick = torch.cat([pick, torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1).contiguous()
    wts = torch.rand((R, cfg.topk + 1), generator=g, device="cuda")
    wts[:, -1] = 1.0
    run = lambda: X.routed(x, pick, wts, ex, s, None, R, limit=cfg.swiglu_limit, act_mode=X.ACT_F32)
    run()
    torch.cuda.synchronize()
    counts = torch.bincount(pick[:, :-1].reshape(-1).long(), minlength=E).float()
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    res = {"rows": R, "members_per_expert": {"mean": round(counts.mean().item(), 1), "max": int(counts.max()),
                                             "min": int(counts.min()),
                                             "over_32": int((counts > 32).sum()), "over_64": int((counts > 64).sum())}}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
