"""A one-GPU proxy of the decode round: rounds.RoundDecoder (--body rounds, the --parallel engine's body) or
graph.StaticDecoder (--body static) over all 40 layers with rank 0's TP2 weights (attention, indexer, compressor, head:
the real linears; MoE: the real gate and experts, but only the first --experts routed experts of each layer loaded and
the gate cut to them, so 40 layers fit in ~12 GB), the TP gathers replaced by a local stand-in (the partial in slot 0,
zeros in slot 1: two small kernels, as the RDMA gather's stage + collect, plus an optional spin of --gather-us
microseconds for the peer's round trip), no Engram rows. Every decode kernel of the round runs in its real order inside
CUDA graphs, so a change to one kernel, or to what overlaps it, shows up as it would in the served forward, minus the
peer's waits (emulate them with --gather-us).

Variants are module flags set while a variant's graphs are captured (--variants name=mod.ATTR:val;mod.ATTR2:val2,...;
an empty spec = the tree's defaults); every variant's graphs are replayed interleaved (median of rounds), their logits
and taps compared (torch.equal) step by step and, with --against, against logits saved by --save on another tree (the
base commit). --invariance N: rows of 16 streams (a pool slot each), each row alone against the same row inside N
windows of 2..16 rows in random order, every variant. --profile R:variant: a torch.profiler kernel dump of that graph.

  python3 forward_proxy.py --model M [--rows 1,2,6,16] [--experts 32] [--gather-us 0] [--body rounds|static]
                           [--variants old=M.FLAG:False,new=] [--invariance 75] [--save F.pt | --against F.pt] --out F
"""

import argparse
import importlib
import json
import statistics
import time

import torch


class LocalComm:
    """Comm stand-in: gather -> [2, *x.shape] with x in slot 0 and zeros in slot 1 (two small kernels, as the RDMA
    gather's stage + collect), then a spin of ``us`` microseconds (the peer's round trip)."""

    def __init__(self, world: int = 2, us: float = 0.0):
        self.world, self.us = world, us
        self.cycles = 0
        if us:
            clk = torch.cuda.get_device_properties(0).clock_rate if hasattr(torch.cuda.get_device_properties(0),
                                                                            "clock_rate") else 2_400_000
            self.cycles = int(us * clk / 1000)

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty((self.world, *x.shape), dtype=x.dtype, device=x.device)
        out[0].copy_(x)
        out[1:].zero_()
        if self.cycles:
            torch.cuda._sleep(self.cycles)
        return out

    def sum(self, x: torch.Tensor) -> torch.Tensor:
        return x


def load(model_dir, n_layers, n_experts, rank=0, world=2):
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, Weights, linear, load_block

    cfg = Cfg.read(model_dir)
    sh = Shards(model_dir)
    vocab_l = cfg.vocab // world
    w = Weights(cfg, rank, world, embed=sh.get("embed.weight").contiguous().cuda(),
                norm=sh.get("norm.weight").contiguous().cuda(),
                head=linear(sh, "head", cols=(rank * vocab_l, (rank + 1) * vocab_l)),
                vocab_lo=rank * vocab_l, vocab_hi=(rank + 1) * vocab_l)
    for i in range(n_layers):
        lay = load_block(sh, cfg, f"layers.{i}", i, rank, world, n_experts)
        lay.gate_w = lay.gate_w[:n_experts].contiguous()          # the gate picks among the loaded experts
        lay.gate_b = lay.gate_b[:n_experts].contiguous()
        w.layers.append(lay)
    sh.close()
    return cfg, w


def set_flags(spec):
    """'mod.ATTR:val;mod2.ATTR2:val2' -> {(module, attr): value}; values parsed as Python literals."""

    import ast

    out = {}
    for part in filter(None, (spec or "").split(";")):
        lhs, val = part.split(":", 1)
        mod, attr = lhs.rsplit(".", 1)
        try:
            v = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            v = val
        out[(importlib.import_module(mod), attr)] = v
    return out


class StaticBody:
    """graph.StaticDecoder behind RoundDecoder's calls (set / capture / graphs / logits / taps), one sequence cache."""

    def __init__(self, m, sc, R, b, taps):
        from tensorfold.families.deepseek_v41.cuda.graph import StaticDecoder

        self.d = StaticDecoder(m, sc, R, b, taps)
        self.graphs = None

    def set(self, ids, pos, slots, base, end):
        self.d.ids.copy_(torch.tensor(ids, dtype=torch.long))
        self.d.pos.copy_(torch.tensor(pos, dtype=torch.long))

    def capture(self, pool=None):
        self.d.capture(pool)
        self.graphs = [self.d.graph]

    @property
    def logits(self):
        return self.d.logits

    @property
    def taps(self):
        return self.d.taps


class flags:
    def __init__(self, f):
        self.f = f

    def __enter__(self):
        self.old = {k: getattr(k[0], k[1]) for k in self.f}
        for (m, a), v in self.f.items():
            setattr(m, a, v)

    def __exit__(self, *e):
        for (m, a), v in self.old.items():
            setattr(m, a, v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--rows", default="1,2,6,16")
    p.add_argument("--cap", type=int, default=8192)
    p.add_argument("--pos", type=int, default=600, help="first position of the window (context so far)")
    p.add_argument("--gather-us", type=float, default=0.0)
    p.add_argument("--body", default="rounds", choices=["rounds", "static"],
                   help="rounds.RoundDecoder (the --parallel engine's body) or graph.StaticDecoder (the serial one)")
    p.add_argument("--variants", default="old", help="name=spec,... (spec: mod.ATTR:val;...); 'old' = no change")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--steps", type=int, default=6, help="bit-compare steps (different token ids) a row count")
    p.add_argument("--save")
    p.add_argument("--against")
    p.add_argument("--profile", help="rows:variant: only replay that graph a few times (torch.profiler kernel dump)")
    p.add_argument("--invariance", type=int, default=0, help="windows of 2..16 rows from different streams, each row "
                   "against itself alone (eager rounds, every variant)")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import model as M
    from tensorfold.families.deepseek_v41.cuda import rounds as RD

    t0 = time.time()
    cfg, w = load(a.model, a.layers, a.experts)
    m = M.Model(w, LocalComm(2, a.gather_us))
    pool = m.new_pool(slots=16 if a.invariance else 4, cap=a.cap)
    sc = m.new_cache(a.cap + 64) if a.body == "static" else None
    torch.cuda.synchronize()
    print(json.dumps({"load_s": round(time.time() - t0, 1),
                      "gpu_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}), flush=True)
    variants = {}
    for item in a.variants.split(","):
        name, _, spec = item.partition("=")
        variants[name] = set_flags(spec)
    gpool = torch.cuda.graph_pool_handle()
    rows = [int(r) for r in a.rows.split(",")]
    res = {"args": vars(a), "rows": {}}
    saved = {}
    against = torch.load(a.against) if a.against else None
    g = torch.Generator().manual_seed(5)
    for R in rows:
        pos = list(range(a.pos, a.pos + R))
        b = RD.bucket_for(a.pos + R, pool.cap)
        decs = {}
        ids0 = torch.randint(1000, 100000, (R,), generator=g).tolist()
        for name, f in variants.items():
            with flags(f):
                d = (RD.RoundDecoder(m, pool, R, b, taps=R > 1) if a.body == "rounds"
                     else StaticBody(m, sc, R, b, R > 1))
                d.set(ids0, pos, [0] * R, [0] * R, [a.cap] * R)
                d.capture(gpool)
            decs[name] = d
        torch.cuda.synchronize()
        if a.profile:
            pr, pv = a.profile.split(":")
            if int(pr) == R:
                d = decs[pv]
                for _ in range(5):
                    for gr in d.graphs:
                        gr.replay()
                torch.cuda.synchronize()
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        for gr in d.graphs:
                            gr.replay()
                    torch.cuda.synchronize()
                ev = [(e.name, e.time_range.start, e.time_range.end) for e in prof.events()
                      if e.device_type == torch.autograd.DeviceType.CUDA]
                ev.sort(key=lambda t: t[1])
                json.dump(ev, open((a.out or "/w/res/prof") + ".kern.json", "w"))
                print(f"profiled {len(ev)} kernels", flush=True)
            continue
        # bits: each step new ids, every variant replayed on the same inputs, logits compared
        eq = {name: True for name in variants}
        first = next(iter(variants))
        for st in range(a.steps):
            ids = torch.randint(1000, 100000, (R,), generator=g).tolist()
            outs = {}
            for name, d in decs.items():
                d.set(ids, pos, [0] * R, [0] * R, [a.cap] * R)
                for gr in d.graphs:
                    gr.replay()
                outs[name] = (d.logits.clone(), None if d.taps is None else d.taps.clone())
            torch.cuda.synchronize()
            for name in variants:
                eq[name] &= all(torch.equal(u, v) for u, v in zip(outs[name], outs[first]) if u is not None)
                saved[f"{R}|{st}|{name}"] = outs[name][0].cpu()
            if against is not None:
                for name in variants:
                    ref = against.get(f"{R}|{st}|old")
                    if ref is not None:
                        key = f"{name}_vs_saved"
                        eq[key] = eq.get(key, True) and torch.equal(outs[name][0].cpu(), ref)
            assert torch.isfinite(outs[first][0]).all(), "non-finite logits"
        # time: interleaved replays
        ts = {name: [] for name in variants}
        for rd in range(a.rounds):
            order = list(decs.items()) if rd % 2 == 0 else list(decs.items())[::-1]
            for name, d in order:
                for gr in d.graphs:
                    gr.replay()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(a.iters):
                    for gr in d.graphs:
                        gr.replay()
                e1.record()
                torch.cuda.synchronize()
                ts[name].append(e0.elapsed_time(e1) / a.iters)
        r = {name: {"ms_median": round(statistics.median(v), 3), "ms_min": round(min(v), 3)} for name, v in ts.items()}
        r["bits_equal_" + first] = {k: bool(v) for k, v in eq.items()}
        res["rows"][R] = r
        print(json.dumps({"R": R, **r}), flush=True)
        del decs
    if a.invariance:
        res["invariance"] = invariance(m, pool, RD, variants, a)
        print(json.dumps({"invariance": res["invariance"]}), flush=True)
    if a.save:
        torch.save(saved, a.save)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def invariance(m, pool, RD, variants, a):
    """Rows of 16 streams (a pool slot each, extents of cap / 16 positions): every row alone (an R = 1 round)
    against the same row inside windows of 2..16 rows of other streams in random order, eager rounds (the decode
    graphs launch the same kernels), logits and taps compared with torch.equal, for every variant; and each variant's
    solo rows against the first variant's."""

    import random

    rnd = random.Random(11)
    ext = a.cap // 16
    b = RD.bucket_for(512, pool.cap)                # every stream's positions are < 512
    out = {}
    solo_first = None
    # (slot, token id, the stream's own position: caches address it from the slot's extent base)
    pool_rows = [(s, rnd.randrange(1000, 100000), 200 + rnd.randrange(0, 200)) for s in range(16)]
    for name, f in variants.items():
        wr = random.Random(12)                      # the same windows for every variant
        with flags(f):
            solo = {}

            def run(rows):
                d = RD.RoundDecoder(m, pool, len(rows), b, True)
                ids = [r[1] for r in rows]
                pos = [r[2] for r in rows]
                d.run(ids, pos, [r[0] for r in rows], [r[0] * ext for r in rows], [(r[0] + 1) * ext for r in rows],
                      None)
                torch.cuda.synchronize()
                return d.logits.clone(), (None if d.taps is None else d.taps.clone())

            for r in pool_rows:
                solo[r] = run([r])
            bad, checked = 0, 0
            for w in range(a.invariance):
                R = 2 + w % 15
                rows = wr.sample(pool_rows, R)
                lg, tp = run(rows)
                for j, r in enumerate(rows):
                    checked += 1
                    ok = torch.equal(lg[j], solo[r][0][0])
                    if tp is not None and solo[r][1] is not None:
                        ok &= torch.equal(tp[j], solo[r][1][0])
                    bad += not ok
            out[name] = {"windows": a.invariance, "rows_checked": checked, "rows_differing": bad}
            if solo_first is None:
                solo_first = solo
            else:
                out[name]["solo_equal_first"] = all(torch.equal(solo[r][0], solo_first[r][0]) for r in solo)
    return out


if __name__ == "__main__":
    main()
