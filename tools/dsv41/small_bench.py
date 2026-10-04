"""The decode round's small kernels on one GPU (rank 0's half of a TP2 split, real weights): CUDA-graph replays of
RoundDecoder forwards (rounds.py) at 1 .. 16 rows, the old path against the new one (module switches, see MODES),
interleaved, plus bit checks (old == new logits and taps) and end-to-end row invariance (a row alone against the same
row inside rounds of 2 .. 16 rows of several streams, any order).

One GPU stands in for the pair: the gathers are a local stand-in (FakeComm: two copy kernels, both ranks' partials
the same), the experts of every layer are one of a few donor layers' (memory), Engram is off. Everything else is the
real decode round: all 40 layers' attention (CSA2, indexer, compressor, mHC), MoE gate + routed/shared experts, head.

  python3 small_bench.py --model M [--layers 40] [--donors 0,20] [--rows 1,2,6,16] [--modes old,new] [--out F]
"""

import argparse
import json
import random
import statistics
import time

import torch

BF16, F32 = torch.bfloat16, torch.float32


class FakeComm:
    """world 2 on one GPU: gather(x) = [x, x] (a stage and a collect copy, as the RDMA gather's two kernels)."""

    def __init__(self):
        self.world = 2
        self.nccl = None

    def gather(self, x):
        x = x.contiguous()
        out = torch.empty((2, *x.shape), dtype=x.dtype, device=x.device)
        out[0].copy_(x)
        out[1].copy_(x)
        return out

    def sum(self, x):
        g = self.gather(x)
        acc = g[0].clone()
        acc += g[1]
        return acc


def load_model(model_dir, n_layers, donors, rank=0, world=2):
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.model import Model

    cfg = Cfg.read(model_dir)
    sh = W.Shards(model_dir)
    ex = {}
    for d in donors:
        ex[d] = W.load_block(sh, cfg, f"layers.{d}", d, rank, world, cfg.n_routed).experts
    layers = []
    for i in range(n_layers):
        lay = W.load_block(sh, cfg, f"layers.{i}", i, rank, world, 0)       # attention, gate, shared expert only
        lay.experts = ex[donors[i % len(donors)]]
        layers.append(lay)
    vocab_l = cfg.vocab // world
    w = W.Weights(cfg, rank, world, embed=sh.get("embed.weight").cuda(), norm=sh.get("norm.weight").cuda(),
                  head=W.linear(sh, "head", cols=(rank * vocab_l, (rank + 1) * vocab_l)), vocab_lo=rank * vocab_l,
                  vocab_hi=(rank + 1) * vocab_l, layers=layers)
    sh.close()
    return Model(w, FakeComm())


# module switches of each path; "new" turns on every switch the patch adds (absent on the base commit: old only)
def modes():
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    sw = getattr(K, "SMALL_SWITCHES", None)
    out = {"old": {}}
    if sw:
        out["old"] = {k: False for k in sw}
        out["new"] = {k: True for k in sw}
        for k in sw:                                   # every switch but one (attribution: what each adds)
            out[f"without_{k}"] = {j: (j != k) for j in sw}
        out["new_nopdl"] = {j: (j != "pdl") for j in sw}
    return out


def set_mode(flags):
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    for k, v in flags.items():
        K.set_switch(k, v)


def windows_for(streams, rng, R, probe=None):
    """R rows as streams' windows: (slot, base, size, first position, tokens, host); ``probe`` (slot, pos, tok) leads
    its stream's window. Streams in random order."""

    order = list(streams)
    rng.shuffle(order)
    k = rng.randint(1, min(len(order), R))                 # streams in the round
    used = order[:k]
    if probe is not None and probe[0] not in used:
        used[rng.randrange(k)] = probe[0]
    cuts = sorted(rng.sample(range(1, R), k - 1)) if k > 1 else []
    sizes = [b - a for a, b in zip([0] + cuts, cuts + [R])]   # a random composition of R rows into k windows
    wins = []
    for s, n in zip(used, sizes):
        slot, base, size, p0 = streams[s]
        toks = [rng.randrange(1000, 100000) for _ in range(n)]
        if probe is not None and s == probe[0]:
            p0 = probe[1]
            toks[0] = probe[2]
        wins.append((slot, base, size, p0, toks, None))
    return wins


def probe_row(wins, probe):
    i = 0
    for slot, _, _, p0, toks, _ in wins:
        if slot == probe[0] and p0 == probe[1]:
            return i
        i += len(toks)
    raise AssertionError("probe row missing")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--donors", default="0,20")
    p.add_argument("--rows", default="1,2,6,16")
    p.add_argument("--prompt", type=int, default=700)
    p.add_argument("--extent", type=int, default=1024)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--modes", default="old,new")
    p.add_argument("--invariance", type=int, default=2, help="probe rows a mode for the end-to-end row check (0: off)")
    p.add_argument("--profile", default="", help="mode:rows to profile graph replays per kernel, e.g. old:1")
    p.add_argument("--attrib", action="store_true", help="also time each switch alone at 1 row")
    p.add_argument("--sequence", default="", help="write the profiled replay's kernels in order to this file")
    p.add_argument("--save-logits", default="", help="save the first mode's graph logits and taps a row count (.pt)")
    p.add_argument("--out")
    a = p.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    m = load_model(a.model, a.layers, [int(d) for d in a.donors.split(",")])
    torch.cuda.synchronize()
    res = {"load_s": round(time.time() - t0, 1), "gpu_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
           "layers": a.layers, "rows": {}, "bits": {}, "invariance": {}}
    print(json.dumps({"load_s": res["load_s"], "gpu_gib": res["gpu_gib"]}), flush=True)

    from tensorfold.families.deepseek_v41.cuda.rounds import RoundRunner

    pool = m.new_pool(a.slots, a.slots * a.extent)
    g = torch.Generator().manual_seed(7)
    streams = {}
    set_mode(modes()["old"])
    for s in range(a.slots):
        n = a.prompt + 37 * s
        v = m.pool_view(pool, s, s * a.extent, a.extent)
        ids = torch.randint(1000, 100000, (n,), generator=g)
        m.forward(v, ids.cuda(), 0)
        streams[s] = (s, s * a.extent, a.extent, n)
    torch.cuda.synchronize()
    print(json.dumps({"prompts_filled": a.slots, "gpu_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}),
          flush=True)

    all_modes = modes()
    want = [md for md in a.modes.split(",") if md in all_modes]
    rows = [int(r) for r in a.rows.split(",")]
    rng = random.Random(11)

    # -- graphs a mode a row count (the same windows for every mode) ----------------------------------------------
    runners = {md: RoundRunner(m, pool, graphs=True) for md in want}
    wins_of = {R: windows_for(streams, rng, R) for R in rows}
    decs = {}
    for R in rows:
        for md in want:
            set_mode(all_modes[md])
            runners[md].forward(wins_of[R], replay=False)
            decs[(md, R)] = next(iter(v for k, v in runners[md].graphs.items() if k[0] == R))
        torch.cuda.synchronize()
    print(json.dumps({"captured": len(decs), "gpu_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                      "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2)}), flush=True)

    def run(md, R, wins):
        set_mode(all_modes[md])
        lg, taps = runners[md].forward(wins)
        return lg.clone(), None if taps is None else taps.clone()

    if a.save_logits:                                  # for comparing trees (e.g. the base commit's run against this one)
        saved = {}
        for R in rows:
            lg, taps = run(want[0], R, wins_of[R])
            saved[R] = (lg.cpu(), None if taps is None else taps.cpu())
        torch.save(saved, a.save_logits)

    # -- bits: every mode's graph against old's, eager against graph ------------------------------------------------
    for R in (rows if "old" in want else []):
        ref = run("old", R, wins_of[R])
        for md in want:
            got = run(md, R, wins_of[R])
            eq = torch.equal(ref[0], got[0]) and (ref[1] is None or torch.equal(ref[1], got[1]))
            res["bits"][f"{md}_vs_old_rows{R}"] = bool(eq)
        # eager (no graph) of the newest mode
        md = want[-1]
        set_mode(all_modes[md])
        eager = RoundRunner(m, pool, graphs=False)
        lg, taps = eager.forward(wins_of[R])
        res["bits"][f"{md}_eager_vs_old_graph_rows{R}"] = bool(torch.equal(lg, ref[0]) and
                                                               (taps is None or torch.equal(taps, ref[1])))
        print(json.dumps({f"bits rows{R}": {k: v for k, v in res["bits"].items() if k.endswith(f"rows{R}")}}),
              flush=True)

    # -- end-to-end row invariance (eager rounds; every kernel is the graph's) ---------------------------------------
    if a.invariance:
        for md in want:
            set_mode(all_modes[md])
            eager = RoundRunner(m, pool, graphs=False)
            ok, checks = True, 0
            for t in range(a.invariance):
                s = rng.randrange(a.slots)
                pos = streams[s][3] + rng.randrange(0, 40)
                probe = (s, pos, rng.randrange(1000, 100000))
                solo = [(streams[s][0], streams[s][1], streams[s][2], pos, [probe[2]], None)]
                alone = eager.forward(solo)[0][0].clone()
                for R in range(2, 17):
                    wins = windows_for(streams, rng, R, probe=probe)
                    lg = eager.forward(wins)[0]
                    i = probe_row(wins, probe)
                    ok &= bool(torch.equal(lg[i], alone))
                    checks += 1
            res["invariance"][md] = {"rounds": checks, "all_equal": bool(ok)}
            print(json.dumps({f"invariance {md}": res["invariance"][md]}), flush=True)

    # -- time: interleaved graph replays ---------------------------------------------------------------------------
    def time_graph(dec, iters):
        gr = dec.graphs[0] if len(dec.graphs) == 1 else None
        for _ in range(2):
            for x in dec.graphs:
                x.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            if gr is not None:
                gr.replay()
            else:
                for x in dec.graphs:
                    x.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters

    for R in rows:
        ts = {md: [] for md in want}
        for md in want:
            decs[(md, R)].set(*_set_args(wins_of[R]))
        for _ in range(a.rounds):
            for md in want:
                ts[md].append(time_graph(decs[(md, R)], a.iters))
        r = {md: round(statistics.median(v), 3) for md, v in ts.items()}
        r.update({f"{md}_min": round(min(v), 3) for md, v in ts.items()})
        if "new" in r and "old" in r:
            r["saving_ms"] = round(r["old"] - r["new"], 3)
        res["rows"][R] = r
        print(json.dumps({f"ms a forward, rows{R}": r}), flush=True)

    if a.attrib and len(want) > 1:
        res["attrib_rows1"] = {}
        R = rows[0]
        sws = [md for md in all_modes if md.startswith("without_")]
        extra = {}
        for md in sws:
            runners[md] = RoundRunner(m, pool, graphs=True)
            set_mode(all_modes[md])
            runners[md].forward(wins_of[R], replay=False)
            extra[md] = next(iter(runners[md].graphs.values()))
            extra[md].set(*_set_args(wins_of[R]))
        ts = {md: [] for md in ["new"] + sws}
        for _ in range(a.rounds):
            for md in ts:
                ts[md].append(time_graph(decs[("new", R)] if md == "new" else extra[md], a.iters))
        base = statistics.median(ts["new"])
        for md in sws:
            res["attrib_rows1"][md.replace("without_", "")] = round(statistics.median(ts[md]) - base, 3)
        print(json.dumps({f"what each switch adds to the new path (ms, off minus on), rows{R}": res["attrib_rows1"]}),
              flush=True)

    if a.profile:
        md, R = a.profile.split(":")
        R = int(R)
        dec = decs[(md, R)]
        from torch.profiler import ProfilerActivity, profile

        for _ in range(3):
            dec.graphs[0].replay()
        torch.cuda.synchronize()
        n = 5
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                dec.graphs[0].replay()
            torch.cuda.synchronize()
        ks = []
        for e in prof.key_averages():
            t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
            if t > 0:
                ks.append((e.key[:100], round(t / n / 1000, 4), e.count // n))
        ks.sort(key=lambda r: -r[1])
        tot = sum(k[1] for k in ks)
        cnt = sum(k[2] for k in ks)
        res["profile"] = {"mode": md, "rows": R, "kernel_ms": round(tot, 3), "kernels": cnt, "top": ks[:60]}
        print(json.dumps({"profile": md, "rows": R, "kernel_ms": round(tot, 3), "kernels": cnt}), flush=True)
        for k in ks[:60]:
            print(f"  {k[1]:8.4f} ms x{k[2]:5d}  {k[0]}", flush=True)
        # timeline of one replay: each kernel's busy time (its span minus the overlap with the one before) and the
        # idle gap before it, summed by kernel name
        import os
        import tempfile

        path = os.path.join(tempfile.gettempdir(), "small_bench_trace.json")
        prof.export_chrome_trace(path)
        ev = [e for e in json.load(open(path))["traceEvents"] if e.get("cat") == "kernel"]
        ev.sort(key=lambda e: e["ts"])
        per = len(ev) // n
        one = ev[per * (n - 1):]                       # the last replay
        agg: dict = {}
        end = one[0]["ts"]
        span0 = one[0]["ts"]
        for e in one:
            nm = e["name"][:70]
            gap = max(0.0, e["ts"] - end)
            busy = max(0.0, e["ts"] + e["dur"] - max(end, e["ts"]))
            r = agg.setdefault(nm, [0, 0.0, 0.0])
            r[0] += 1
            r[1] += busy
            r[2] += gap
            end = max(end, e["ts"] + e["dur"])
        wall = end - span0
        if a.sequence:
            with open(a.sequence, "w") as f:
                for e in one:
                    f.write(f"{e['ts'] - span0:10.1f} {e['dur']:8.1f}  {e['name'][:110]}\n")
        rowsx = sorted(agg.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))
        res["timeline"] = {"wall_us": round(wall, 1), "busy_us": round(sum(v[1] for v in agg.values()), 1),
                           "gap_us": round(sum(v[2] for v in agg.values()), 1),
                           "by_kernel": [(k, v[0], round(v[1], 1), round(v[2], 1)) for k, v in rowsx[:70]]}
        print(json.dumps({k: res["timeline"][k] for k in ("wall_us", "busy_us", "gap_us")}), flush=True)
        for k, v in rowsx[:70]:
            print(f"  busy {v[1]:8.1f} us  gap {v[2]:7.1f} us  x{v[0]:4d}  {k}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def _set_args(wins):
    ids, pos, slots, base, end = [], [], [], [], []
    for slot, b0, size, p0, toks, _ in wins:
        n = len(toks)
        ids += toks
        pos += range(p0, p0 + n)
        slots += [slot] * n
        base += [b0] * n
        end += [b0 + size] * n
    return ids, pos, slots, base, end


if __name__ == "__main__":
    main()
