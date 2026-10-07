"""Prompt chunks' grouped expert launches over a work list (TF_EXL3_GROUP_LIST, experts.GROUP_LIST) against the
grids they replace (every expert place x the busiest expert's member groups): the same routed outputs bit for bit on
one real layer (rank 0's half of a TP2 split), served kernels and kernel "rows", with and without the shared slot in
the launch; the device-built lists (ext.work_list) equal to the torch reference (experts._work_list) pair for pair, the
tail all sentinels; then the times.

  python3 tools/dsv41/expert_worklist_test.py --model M [--layer 10] [--rows 2048,1000,64] [--require-faster] [--out F]
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
    p.add_argument("--require-faster", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.cuda.exl3 import experts as X
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, load_block

    cfg = Cfg.read(a.model)
    lay = load_block(Shards(a.model), cfg, f"layers.{a.layer}", a.layer, 0, 2, cfg.n_routed)
    ex = lay.experts
    E, D = ex.count - 1, cfg.dim
    rows = [int(r) for r in a.rows.split(",")]
    s = X.Scratch(ex, rows=max(rows), slots=cfg.topk + 1, prompt=True)
    g = torch.Generator(device="cuda").manual_seed(4)
    res: dict = {"checks": {}, "times_ms": {}}
    for R in rows:
        x = (torch.randn((R, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        routed = torch.stack([torch.randperm(E, generator=g, device="cuda")[:cfg.topk]
                              for _ in range(R)]).to(torch.int32)
        wts = torch.rand((R, cfg.topk + 1), generator=g, device="cuda")
        wts[:, -1] = 1.0
        for shared in (True, False):
            last = torch.full((R, 1), E if shared else ex.count, dtype=torch.int32, device="cuda")
            pick = torch.cat([routed, last], 1).contiguous()
            for kern in ("mma", "rows"):
                outs = {}
                for gl in (False, True):
                    X.GROUP_LIST = gl
                    run = lambda: X.routed(x, pick, wts, ex, s, None, R, limit=cfg.swiglu_limit,
                                           act_mode=X.ACT_F32, kernel=kern)
                    outs[gl] = run().clone()
                    for _ in range(3):
                        run()
                    torch.cuda.synchronize()
                    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for _ in range(a.iters):
                        run()
                    t1.record()
                    torch.cuda.synchronize()
                    key = f"R{R} {'with' if shared else 'without'} shared, {kern}, list {'on' if gl else 'off'}"
                    res["times_ms"][key] = round(t0.elapsed_time(t1) / a.iters, 3)
                    print(json.dumps({key: res["times_ms"][key]}), flush=True)
                res["checks"][f"R{R} {'with' if shared else 'without'} shared, {kern}: list == grid"] = bool(
                    torch.equal(outs[True], outs[False]))
            # the lists of the call just made (its counts, places and used count are in the scratch)
            ids = s.window(R)[0]
            for grp in (64, 32):
                total = int(X._groups_per_place(s, ids, grp).sum())
                ref = X._work_list(s, ids, grp, total)
                got = torch.empty((X._list_len(R * (cfg.topk + 1), ids.shape[0], grp), 2), dtype=torch.int32,
                                  device="cuda")
                X._ext().work_list(s.counts, ids, s.count, got, grp, s.no_work, grp)
                tail = got[total:]
                res["checks"][f"R{R} {'with' if shared else 'without'} shared, {grp}-row groups: device list == "
                              f"torch list ({total} of {got.shape[0]})"] = bool(
                    torch.equal(got[:total], ref) and bool((tail[:, 0] == 0x7fffffff).all())
                    and bool((tail[:, 1] == 0).all()))
    X.GROUP_LIST = True
    if a.require_faster:
        R = max(rows)
        res["checks"][f"R{R} served faster with the list"] = (res["times_ms"][f"R{R} with shared, mma, list on"]
                                                             < res["times_ms"][f"R{R} with shared, mma, list off"])
    res["passed"] = all(res["checks"].values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
