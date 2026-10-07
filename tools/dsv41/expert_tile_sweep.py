"""Prompt chunks' routed experts on one real layer (rank 0's half of a TP2 split): every compiled grouped_rows tile
setting for the down projection (experts.PROMPT_TILES["down"] = (n tiles a block, tiles in flight, member tiles a
program); warps and K splits stay the window's, which fix each row's bits) with gate/up through the mma kernel as
served: outputs bit-compared with the served setting, and times. A setting that is faster with the same bits is an
exact speedup.

  python3 tools/dsv41/expert_tile_sweep.py --model M [--layers 10,2] [--rows 2048,1024] [--out F]   (one GPU, idle)
Exits 1 when a setting's bits differ from the served one.
"""

import argparse
import json
import sys

import torch

SETTINGS = [(8, 1, 2), (8, 1, 1), (8, 2, 1), (4, 2, 2)]       # grouped_rows_launch's compiled (nt, pf, g) with 4 warps


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="10,2")
    p.add_argument("--rows", default="2048,1024")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.cuda.exl3 import experts as X
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, load_block

    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    served = tuple(X.PROMPT_TILES["down"])
    res: dict = {"served_down": served, "checks": {}, "times_ms": {}}
    rows = [int(r) for r in a.rows.split(",")]
    for L in (int(x) for x in a.layers.split(",")):
        lay = load_block(sh, cfg, f"layers.{L}", L, 0, 2, cfg.n_routed)
        ex = lay.experts
        E, D = ex.count - 1, cfg.dim
        s = X.Scratch(ex, rows=max(rows), slots=cfg.topk + 1, cfg_gu=(8, 4, 4, 1), cfg_d=(8, 4, 1, 1), prompt=True)
        g = torch.Generator(device="cuda").manual_seed(L)
        for R in rows:
            x = (torch.randn((R, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
            pick = torch.stack([torch.randperm(E, generator=g, device="cuda")[:cfg.topk]
                                for _ in range(R)]).to(torch.int32)
            pick = torch.cat([pick, torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1).contiguous()
            wts = torch.rand((R, cfg.topk + 1), generator=g, device="cuda")
            wts[:, -1] = 1.0
            base = None
            for st in SETTINGS:
                X.PROMPT_TILES["down"] = st
                run = lambda: X.routed(x, pick, wts, ex, s, None, R, limit=cfg.swiglu_limit, act_mode=X.ACT_F32,
                                       kernel="mma")
                key = f"L{L} R{R} down {st}"
                try:
                    out = run().clone()
                    for _ in range(3):
                        run()
                    torch.cuda.synchronize()
                    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for _ in range(a.iters):
                        run()
                    t1.record()
                    torch.cuda.synchronize()
                    res["times_ms"][key] = round(t0.elapsed_time(t1) / a.iters, 3)
                    if st == served:
                        base = out
                    else:
                        res["checks"][key + " == served"] = bool(torch.equal(out, base))
                except Exception as exc:                      # noqa: BLE001
                    res["times_ms"][key] = f"error: {str(exc)[:160]}"
                print(json.dumps({key: res["times_ms"][key]}), flush=True)
            X.PROMPT_TILES["down"] = served
        del lay, ex, s
        torch.cuda.empty_cache()
    res["passed"] = all(res["checks"].values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
