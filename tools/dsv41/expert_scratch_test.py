"""The prompt expert scratch sized for folded sums (experts.prompt_z) against a full-size one: the same routed outputs bit
for bit at prompt-chunk row counts on one real layer, and no write past the folded size (a sentinel tail stays as it
was); prints both sizes.

  python3 tools/dsv41/expert_scratch_test.py --model M [--layer 10] [--rows 2048,1000,64] [--out F]   (one GPU, idle)
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
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.cuda.exl3 import experts as X
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, load_block

    cfg = Cfg.read(a.model)
    lay = load_block(Shards(a.model), cfg, f"layers.{a.layer}", a.layer, 0, 2, cfg.n_routed)
    ex = lay.experts
    E, D, I = ex.count - 1, cfg.dim, ex.width
    slots = cfg.topk + 1
    rows = [int(r) for r in a.rows.split(",")]
    Rm = max(rows)
    small = X.Scratch(ex, rows=Rm, slots=slots, prompt=True)
    full = X.Scratch(ex, rows=Rm, slots=slots, prompt=True)
    n_small = small.z.numel()
    n_full = max(2 * full.cfg_gu[2] * I, full.cfg_d[2] * D) * Rm * slots
    full.z = torch.zeros((n_full,), dtype=torch.float32, device="cuda")      # the size before the change
    canary = 1 << 20
    sentinel = 1234.5
    small.z = torch.full((n_small + canary,), sentinel, dtype=torch.float32, device="cuda")
    small.z[:n_small] = 0.0                                                   # as allocated; the tail is the canary
    res: dict = {"z_MiB": {"before": round(n_full * 4 / 2**20, 1), "after": round(n_small * 4 / 2**20, 1)},
                 "checks": {}}
    g = torch.Generator(device="cuda").manual_seed(3)
    for R in rows:
        x = (torch.randn((R, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        pick = torch.stack([torch.randperm(E, generator=g, device="cuda")[:cfg.topk] for _ in range(R)]).to(torch.int32)
        pick = torch.cat([pick, torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1).contiguous()
        wts = torch.rand((R, slots), generator=g, device="cuda")
        wts[:, -1] = 1.0
        o_small = X.routed(x, pick, wts, ex, small, None, R, limit=cfg.swiglu_limit, act_mode=X.ACT_F32).clone()
        o_full = X.routed(x, pick, wts, ex, full, None, R, limit=cfg.swiglu_limit, act_mode=X.ACT_F32).clone()
        torch.cuda.synchronize()
        res["checks"][f"R{R} outputs equal"] = bool(torch.equal(o_small, o_full))
        res["checks"][f"R{R} not regrown"] = small.z.numel() == n_small + canary
        res["checks"][f"R{R} tail untouched"] = bool((small.z[n_small:] == sentinel).all())
    res["passed"] = all(res["checks"].values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
