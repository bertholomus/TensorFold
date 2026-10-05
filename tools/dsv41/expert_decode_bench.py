"""Decode-window routed experts (exl3.experts.routed, fewer than 64 rows) on one layer of DeepSeek-V4.1-Flash, rank 0's
halves of the TP2 split: the decode paths side by side at R rows (routed(decode=...): old = grouped_kernel in six
launches, cp = grouped_cp_kernel in six launches, fused = three launches, the default), each call a CUDA graph replay
(as the engine runs decode), the L2 flushed before every replay (a layer's experts are never L2-resident in a real
forward), the variants interleaved replay by replay; outputs bit-compared against the first variant and, with
--against, against outputs saved by --save on another tree (the base commit: variant "base"); then the row-invariance
check (each of 64 rows alone against inside windows of 2..16 random other rows in random order: torch.equal).

  python3 expert_decode_bench.py --model M [--layer 10] [--rows 1,2,4,6,8,16] [--picks random,disjoint,clustered,gate]
                                 [--variants old,cp,fused] [--iters 12] [--sets 4] [--invariance 200]
                                 [--save F.pt | --against F.pt] --out F.json

Pick modes (each (mode, R) seeded on its own, so --save / --against line up whatever else runs): random (6 of the
routed experts a row, uniform), disjoint (no expert shared between rows: the most a window can touch), clustered (each
row's 6 from a pool of 12: verify windows of one stream share most), gate (the layer's own gate and bias on random
rows: its popularity skew), real (--real F: the layer's picks of saved served verify windows of R rows, R = 1 the first
row of a 2-row window: their per-expert row counts). Every row also takes the shared expert (the layer's last), as the
family routes it. --flush 0 keeps the L2 between replays (default 1: flushed).
"""

import argparse
import json
import math
import os
import random
import time

import torch


REAL: dict = {}


def picks_for(mode, R, E, topk, g, gate=None, x=None):
    if mode == "real":
        lst = REAL.get(R) or []
        if not lst:
            raise ValueError(f"no saved windows of {R} rows")
        k = int(torch.randint(len(lst), (1,), generator=torch.Generator().manual_seed(int(g.initial_seed()) +
                                                                                     len(REAL.get("_n", [])))))
        REAL.setdefault("_n", []).append(k)
        return lst[k].to(device="cuda", dtype=torch.int32)
    if mode == "random":
        p = torch.stack([torch.randperm(E, generator=g, device="cuda")[:topk] for _ in range(R)])
    elif mode == "disjoint":
        p = torch.randperm(E, generator=g, device="cuda")[:R * topk].view(R, topk)
    elif mode == "clustered":
        pool = torch.randperm(E, generator=g, device="cuda")[:2 * topk]
        p = torch.stack([pool[torch.randperm(2 * topk, generator=g, device="cuda")[:topk]] for _ in range(R)])
    elif mode == "gate":
        w, b = gate
        sc = torch.nn.functional.softplus(x.float() @ w.float().t()).sqrt()
        p = (sc + b.float()).topk(topk, dim=-1).indices
    else:
        raise ValueError(mode)
    return p.to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--rows", default="1,2,4,6,8,16")
    ap.add_argument("--picks", default="random,disjoint,clustered,gate")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--sets", type=int, default=4)
    ap.add_argument("--invariance", type=int, default=200, help="windows checked for row invariance (0: skip)")
    ap.add_argument("--variants", default="old,cp,fused", help="routed(decode=...): old (grouped_kernel, six "
                    "launches), cp (grouped_cp_kernel, six launches), fused (three launches); base = the tree's default")
    ap.add_argument("--save", help="save every output (torch.save) for --against")
    ap.add_argument("--against", help="outputs saved by --save (e.g. on the base commit): bit-compare each variant")
    ap.add_argument("--real", help="saved routed picks (list of (layer, int32 [rows, topk])) for --picks real")
    ap.add_argument("--flush", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tensorfold.cuda.exl3 import experts as X
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, load_block

    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    t0 = time.time()
    lay = load_block(sh, cfg, f"layers.{a.layer}", a.layer, 0, 2, cfg.n_routed)
    ex = lay.experts
    E, D, topk, slots = ex.count - 1, ex.dims, cfg.topk, cfg.topk + 1
    print(f"layer {a.layer} loaded in {time.time() - t0:.1f} s: E={E} D={D} I={ex.width} k2_gu={ex.k2_gu} "
          f"k2_d={ex.k2_d} aligned16={getattr(ex, 'aligned16', None)}", flush=True)
    gate = (lay.gate_w, lay.gate_b)
    if a.real:
        for li, t in torch.load(a.real):
            if int(li) == a.layer:
                REAL.setdefault(int(t.shape[0]), []).append(t)
        REAL[1] = [t[:1] for t in REAL.get(2, [])]
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush_buf = torch.empty(max(4 * l2, 128 << 20) // 4, dtype=torch.int32, device="cuda")
    sink = torch.empty((), dtype=torch.int64, device="cuda")

    def flush():
        torch.sum(flush_buf, 0, dtype=torch.int64, out=sink)

    def variant(name):
        """(routed kwargs, module flags set while the variant's graphs are captured): base = the tree's default (the
        base commit's only decode path); old / cp / fused (the defaults: programmatic dependent launch and per-expert
        readiness); fused-nopdl, fused-pdl, fused-ready: fused with DECODE_PDL off / on without / on with DECODE_READY."""
        if name == "base":
            return {}, {}
        if name.startswith("fused-"):
            flags = {"nopdl": {"DECODE_PDL": False, "DECODE_READY": False},
                     "pdl": {"DECODE_PDL": True, "DECODE_READY": False},
                     "ready": {"DECODE_PDL": True, "DECODE_READY": True}}[name[6:]]
            return {"decode": "fused"}, flags
        return {"decode": name}, {}

    import contextlib

    @contextlib.contextmanager
    def flags_set(flags):
        old = {k: getattr(X, k) for k in flags}
        for k, v in flags.items():
            setattr(X, k, v)
        try:
            yield
        finally:
            for k, v in old.items():
                setattr(X, k, v)

    variants = a.variants.split(",")
    modes = ["random", "disjoint", "clustered", "gate", "real"]
    res = {"layer": a.layer, "bench": []}
    saved = {}
    against = torch.load(a.against) if a.against else None
    for mode in a.picks.split(","):
        for R in [int(r) for r in a.rows.split(",")]:
            if mode == "real" and not REAL.get(R):
                continue                             # no saved windows of R rows
            # each (picks, R) its own seed: the same sets whichever other modes and rows run (--against)
            g = torch.Generator(device="cuda").manual_seed(1234 + 1000 * modes.index(mode) + R)
            sets = []
            for _ in range(a.sets):
                x = (torch.randn((R, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
                p = picks_for(mode, R, E, topk, g, gate, x)
                pick = torch.cat([p, torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1).contiguous()
                wts = torch.rand((R, slots), generator=g, device="cuda")
                wts[:, -1] = 1.0
                sets.append((x, pick, wts))
            union = sum(int(torch.unique(st[1][:, :-1]).numel()) for st in sets) / len(sets)
            nbytes = sum(ex.nbytes_read(torch.unique(st[1]).tolist()) for st in sets) / len(sets)
            graphs, outs = {}, {}
            for vn in variants:
                kw, stages = variant(vn)
                s = X.Scratch(ex, rows=64, slots=slots)
                for si, (x, pick, wts) in enumerate(sets):
                    with flags_set(stages):
                        call = lambda: X.routed(x, pick, wts, ex, s, None, R, limit=cfg.swiglu_limit,
                                                act_mode=X.ACT_F32, **kw)
                        for _ in range(3):
                            call()
                        torch.cuda.synchronize()
                        gr = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(gr):
                            o = call()
                    graphs[(vn, si)] = (gr, o, s)     # the scratch stays alive with its graphs (they write into it)
            times = {vn: [] for vn in variants}
            for it in range(a.iters + 2):
                for si in range(len(sets)):
                    for vn in (variants if it % 2 == 0 else variants[::-1]):
                        gr, o, _ = graphs[(vn, si)]
                        if a.flush:
                            flush()
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        gr.replay()
                        e1.record()
                        torch.cuda.synchronize()
                        if it >= 2:
                            times[vn].append(e0.elapsed_time(e1))
                        if it == 0:
                            outs[(vn, si)] = o.clone()
            row = {"picks": mode, "R": R, "experts": union, "MB": round(nbytes / 1e6, 1)}
            for vn in variants:
                t = sorted(times[vn])
                ms = sum(t) / len(t)
                row[f"{vn}_ms"] = round(ms, 4)
                row[f"{vn}_med_ms"] = round(t[len(t) // 2], 4)
                row[f"{vn}_GBps"] = round(nbytes / ms / 1e6, 1)
                if vn != variants[0]:
                    row[f"{vn}_bit_equal_{variants[0]}"] = all(torch.equal(outs[(vn, si)], outs[(variants[0], si)])
                                                               for si in range(len(sets)))
            if len(variants) > 1:
                row["saving_ms"] = round(row[f"{variants[0]}_ms"] - row[f"{variants[-1]}_ms"], 4)
                row["saving_pct"] = round(100 * row["saving_ms"] / row[f"{variants[0]}_ms"], 1)
            for vn in variants:
                for si in range(len(sets)):
                    saved[f"{mode}|{R}|{si}|{vn}"] = outs[(vn, si)].cpu()
            if against is not None:
                for vn in variants:
                    ks = [f"{mode}|{R}|{si}|" for si in range(len(sets))]
                    ref = [k for k in against if k.startswith(ks[0])]
                    if ref:
                        rv = ref[0].split("|")[-1]
                        row[f"{vn}_bit_equal_saved_{rv}"] = all(
                            torch.equal(outs[(vn, si)].cpu(), against[ks[si] + rv]) for si in range(len(sets)))
            print(json.dumps(row), flush=True)
            res["bench"].append(row)
            del graphs
    if a.save:
        torch.save(saved, a.save)
    if a.invariance:
        res["invariance"] = invariance(X, ex, cfg, a.invariance, variants, variant, flags_set)
        print(json.dumps(res["invariance"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def invariance(X, ex, cfg, windows, variants, variant, flags_set):
    """Each of 64 random rows alone (R 1) against inside windows of 2..16 rows drawn from the pool in random order (rows
    with random and with shared picks): torch.equal on the row, for every variant, and each variant's solo rows against
    the first variant's."""

    E, D, topk, slots = ex.count - 1, ex.dims, cfg.topk, cfg.topk + 1
    g = torch.Generator(device="cuda").manual_seed(99)
    rnd = random.Random(7)
    n = 64
    x = (torch.randn((n, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    pool = torch.randperm(E, generator=g, device="cuda")[:12]
    p = []
    for i in range(n):
        if i % 2:
            p.append(pool[torch.randperm(12, generator=g, device="cuda")[:topk]])        # shared picks
        else:
            p.append(torch.randperm(E, generator=g, device="cuda")[:topk])
    pick = torch.cat([torch.stack(p).to(torch.int32), torch.full((n, 1), E, dtype=torch.int32, device="cuda")], 1)
    wts = torch.rand((n, slots), generator=g, device="cuda")
    wts[:, -1] = 1.0
    out = {}
    solo = {}
    for vn in variants:
        kw, flags = variant(vn)
        with flags_set(flags):
            s = X.Scratch(ex, rows=64, slots=slots)
            run = lambda idx: X.routed(x[idx].contiguous(), pick[idx].contiguous(), wts[idx].contiguous(), ex, s, None,
                                       len(idx), limit=cfg.swiglu_limit, act_mode=X.ACT_F32, **kw).clone()
            solo[vn] = torch.cat([run([i]) for i in range(n)], 0)
            bad, checked = 0, 0
            for wi in range(windows):
                R = 2 + wi % 15
                idx = rnd.sample(range(n), R)
                o = run(idx)
                for j, i in enumerate(idx):
                    checked += 1
                    bad += not torch.equal(o[j], solo[vn][i])
            out[vn] = {"windows": windows, "rows_checked": checked, "rows_differing": bad}
    for vn in variants[1:]:
        out[vn]["solo_bit_equal_" + variants[0]] = bool(torch.equal(solo[vn], solo[variants[0]]))
    return out


if __name__ == "__main__":
    main()
