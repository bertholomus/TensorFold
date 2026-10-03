"""The decode windows' routed-expert kernel on one layer (one GPU, rank 0's halves of a TP2 split): time at R rows of
independent picks (6 of 384 experts a row, plus the shared one) for each tile setting, outputs bit-compared across
settings and against a saved run (``--save`` / ``--against``: before and after a kernel change).

  python3 expert_bench.py --model M [--layer 10] [--rows 1,4,8,16] [--save F | --against F] --out F
"""

import argparse
import json

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--rows", default="1,4,8,16")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--save")
    p.add_argument("--against")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from tensorfold.cuda.exl3 import experts as X
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, load_block

    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    lay = load_block(sh, cfg, f"layers.{a.layer}", a.layer, 0, 2, cfg.n_routed)
    ex = lay.experts
    E = ex.count - 1
    D = cfg.dim
    g = torch.Generator(device="cuda").manual_seed(0)
    settings = {"gu8441_d8411 (today)": ((8, 4, 4, 1), (8, 4, 1, 1)), "gu8442_d8412": ((8, 4, 4, 2), (8, 4, 1, 2)),
                "gu4442_d4412": ((4, 4, 4, 2), (4, 4, 1, 2))}
    rows = [int(r) for r in a.rows.split(",")]
    res, outs = {}, {}
    for R in rows:
        x = (torch.randn((R, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        pick = torch.stack([torch.randperm(E, generator=g, device="cuda")[:cfg.topk] for _ in range(R)]).to(torch.int32)
        pick = torch.cat([pick, torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1).contiguous()
        wts = torch.rand((R, cfg.topk + 1), generator=g, device="cuda")
        wts[:, -1] = 1.0
        union = int(torch.unique(pick[:, :-1]).numel())
        for name, (gu, dn) in settings.items():
            try:
                s = X.Scratch(ex, rows=64, slots=cfg.topk + 1, cfg_gu=gu, cfg_d=dn)
                run = lambda: X.routed(x, pick, wts, ex, s, None, R, limit=cfg.swiglu_limit, act_mode=X.ACT_F32)
                for _ in range(5):
                    out = run()
                torch.cuda.synchronize()
                t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0.record()
                for _ in range(a.iters):
                    out = run()
                t1.record()
                torch.cuda.synchronize()
                ms = t0.elapsed_time(t1) / a.iters
                outs[(R, name)] = out.clone()
                gb = (ex_bytes(ex, union) + ex_bytes(ex, 1, shared=True)) / 1e9
                res[f"R{R} {name}"] = {"ms": round(ms, 4), "experts_touched": union, "GBps": round(gb / (ms / 1000), 1)}
            except Exception as exc:                      # noqa: BLE001
                res[f"R{R} {name}"] = {"error": str(exc)[:200]}
            print(json.dumps({f"R{R} {name}": res[f"R{R} {name}"]}), flush=True)
        base = outs.get((R, "gu8441_d8411 (today)"))
        for name in settings:
            if (R, name) in outs and base is not None:
                res[f"R{R} {name}"]["bit_equal_to_today"] = bool(torch.equal(outs[(R, name)], base))
    if a.save:
        torch.save({f"{R}|{n}": t.cpu() for (R, n), t in outs.items()}, a.save)
    if a.against:
        old = torch.load(a.against)
        for (R, n), t in outs.items():
            k = f"{R}|{n}"
            if k in old:
                res[f"R{R} {n}"]["bit_equal_to_saved"] = bool(torch.equal(t.cpu(), old[k]))
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def ex_bytes(ex, n, shared=False):
    """The trellis bytes of n average routed experts (or the shared one) of this layer, gate + up + down."""

    import math

    k2 = ex.k2_gu[1] if hasattr(ex, "k2_gu") else 6
    per = (2 * ex.dims * ex.width + ex.width * ex.dims) * (k2 / 2) / 8      # K2 is half-bits per weight
    return per * (1 if shared else n)


if __name__ == "__main__":
    main()
