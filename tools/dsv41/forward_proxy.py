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

--picks F (a list of (layer, int32 [rows, topk]) routed picks of served verify windows, e.g. results/code100-picks.pt):
real routing. Each forward's routed picks are a saved window's (R = 1: the first row of the 2-row windows), remapped
layer by layer onto the --experts loaded experts (distinct experts stay distinct, shared ones shared: a window's per
expert row counts are the served ones), written over the gate's picks right after the route kernel (one small copy a
layer, in every variant). --windows N windows a row count are timed (the mean of their medians) and bit-compared.
--profile R:variant then sums the kernels' CUDA time by name (a forward's), and gaps. --full-gate (with --picks): the
gate keeps every expert's row, so the router matmul and the route run at the model's width (cut to --experts rows they
read ~10x fewer weight bytes than served); the routed picks are the saved windows' either way.

Variants may flip the small-kernel switches: 'sw.NAME:val' is kernels.set_switch(NAME, val).

--engram DIR: the model's Engram (rank 0's hash columns, rows read from DIR's tables, hashes from the window's tokens)
in the rounds, as served (TF_DS_ENGRAM_SPLIT: a graph a stretch). --host N: after the graph timing, N whole
rounds.RoundRunner.forward calls a row count and variant timed by the wall clock (host prep, Engram hashing and reads,
the stretch graphs, the logits, plus an argmax read back as the sampler's): the "forward" of the served round stats
minus the peer.

  python3 forward_proxy.py --model M [--rows 1,2,6,16] [--experts 32] [--gather-us 0] [--body rounds|static]
                           [--variants old=M.FLAG:False,new=] [--invariance 75] [--save F.pt | --against F.pt] --out F
                           [--picks F --windows 8]
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
        """model.Comm.sum's kernels (the gather, a clone, a rank-order add) on the stand-in gather."""

        g = self.gather(x)
        acc = g[0].clone()
        for r in range(1, self.world):
            acc += g[r]
        return acc


class ForcedRoute:
    """kernels.route wrapped: after the real launch, the routed slots' picks of a layer are overwritten with the rows of
    ``buf[layer]`` (a device buffer the graphs capture; set before each replay). Off: the gate's own picks."""

    def __init__(self, K, w, maxrows: int = 16):
        c = w.cfg
        self.K, self.topk = K, c.topk
        self.buf = torch.zeros((len(w.layers), maxrows, c.topk), dtype=torch.int32, device="cuda")
        self.layer_of = {lay.gate_b.data_ptr(): i for i, lay in enumerate(w.layers)}
        self.orig = K.route

        def route(logits, bias, topk, scale, shared_id, pick, wts):
            self.orig(logits, bias, topk, scale, shared_id, pick, wts)
            li = self.layer_of.get(bias.data_ptr())
            if li is not None:
                pick[:, :topk].copy_(self.buf[li, :pick.shape[0]])

        K.route = route

    def set(self, win: torch.Tensor) -> None:
        """win [L, R, topk] int32 (device)."""

        self.buf[:, :win.shape[1]].copy_(win)


def picks_windows(path, n_layers, pool, seed=7):
    """{R: [window [L, R, topk] int32 remapped onto experts 0..pool-1]} from a saved picks list: a window is a run of
    entries for layers 0..L-1 with the same row count (entries of other layers, e.g. the drafter's, are skipped); R = 1
    takes the first row of each 2-row window. Each layer's distinct experts map to distinct random pool experts."""

    import random

    raw = torch.load(path)
    wins, cur = [], {}
    for li, t in raw:
        li = int(li)
        if li >= n_layers:
            continue
        if li == 0:
            cur = {}
        if cur and t.shape[0] != next(iter(cur.values())).shape[0]:
            cur = {}
        cur[li] = t.to(torch.int32)
        if len(cur) == n_layers:
            wins.append(torch.stack([cur[i] for i in range(n_layers)]))
            cur = {}
    out: dict = {}
    for w in wins:
        out.setdefault(w.shape[1], []).append(w)
    out[1] = [w[:, :1] for w in out.get(2, [])]
    folded = 0
    for R in out:
        for k, w in enumerate(out[R]):
            rnd = random.Random(seed * 1000003 + R * 1009 + k)
            m = torch.empty_like(w)
            for li in range(w.shape[0]):
                ids = sorted(set(w[li].reshape(-1).tolist()))
                if len(ids) > pool:
                    folded += 1
                    tgt = [rnd.randrange(pool) for _ in ids]
                else:
                    tgt = rnd.sample(range(pool), len(ids))
                lut = dict(zip(ids, tgt))
                m[li] = torch.tensor([[lut[e] for e in row] for row in w[li].tolist()], dtype=torch.int32)
            out[R][k] = m
    if folded:
        print(f"[proxy] {folded} window layers had more distinct experts than the pool ({pool}): folded", flush=True)
    return out


def load(model_dir, n_layers, n_experts, rank=0, world=2, full_gate=False):
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
        if not full_gate:
            lay.gate_w = lay.gate_w[:n_experts].contiguous()      # the gate picks among the loaded experts
            lay.gate_b = lay.gate_b[:n_experts].contiguous()
        w.layers.append(lay)
    sh.close()
    return cfg, w


class _Switches:
    """kernels.SMALL_SWITCHES as attributes (a variant's "sw.NAME:val" flips kernels.set_switch(NAME, val))."""

    def __getattr__(self, name):
        from tensorfold.families.deepseek_v41.cuda import kernels as K

        return K.SMALL_SWITCHES[name]

    def __setattr__(self, name, value):
        from tensorfold.families.deepseek_v41.cuda import kernels as K

        K.set_switch(name, value)


def set_flags(spec):
    """'mod.ATTR:val;mod2.ATTR2:val2' -> {(module, attr): value}; values parsed as Python literals. 'sw.NAME:val' is
    the kernels switch NAME (kernels.SMALL_SWITCHES)."""

    import ast

    out = {}
    for part in filter(None, (spec or "").split(";")):
        lhs, val = part.split(":", 1)
        mod, attr = lhs.rsplit(".", 1)
        try:
            v = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            v = val
        out[(_Switches() if mod == "sw" else importlib.import_module(mod), attr)] = v
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
    p.add_argument("--profile", help="rows[,rows..]:variant: only replay those graphs a few times (torch.profiler "
                   "kernel dump and a summary by kernel name)")
    p.add_argument("--invariance", type=int, default=0, help="windows of 2..16 rows from different streams, each row "
                   "against itself alone (eager rounds, every variant)")
    p.add_argument("--picks", help="saved routed picks: real routing (see the docstring)")
    p.add_argument("--full-gate", action="store_true", help="with --picks: the gate keeps all its experts' rows (the "
                   "router matmul and route at the model's width; the routed picks are the saved windows')")
    p.add_argument("--engram", help="Engram tables folder (see the docstring)")
    p.add_argument("--host", type=int, default=0, help="whole RoundRunner.forward calls timed a row count and variant")
    p.add_argument("--draft-ms", type=float, default=0.0, help="--host: each forward follows a drafter stand-in of "
                   "this many ms on the GPU, after the round's first row's Engram rows are touched (multi.py)")
    p.add_argument("--windows", type=int, default=8, help="--picks: windows a row count (timed and compared)")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import model as M
    from tensorfold.families.deepseek_v41.cuda import rounds as RD

    t0 = time.time()
    assert not a.full_gate or a.picks, "--full-gate needs --picks (the routed picks must be loaded experts)"
    cfg, w = load(a.model, a.layers, a.experts, full_gate=a.full_gate)
    engram = None
    if a.engram:
        import os
        from pathlib import Path

        from tensorfold.families.deepseek_v41.ops import compressed_token_map

        tmf = Path(os.environ.get("TF_DS_TOKEN_MAP") or "/w/cache/dsv41_token_map.json")
        if tmf.exists():
            tm = json.loads(tmf.read_text())
        else:
            tm, _ = compressed_token_map(Path(a.model) / "tokenizer.json")
            tmf.write_text(json.dumps(tm))
        engram = M.Engram(a.engram, cfg, tm, 0, 2)
    m = M.Model(w, LocalComm(2, a.gather_us), engram)
    forced, wins = None, {}
    if a.picks:
        forced = ForcedRoute(M.K, w)
        allw = picks_windows(a.picks, len(w.layers), a.experts)
        for R, lst in allw.items():
            step = max(1, len(lst) // a.windows)
            wins[R] = [x.cuda() for x in lst[::step][:a.windows]]
        dist = {}
        for R, v in sorted(wins.items()):
            n = [len(set(x[li].reshape(-1).tolist())) for x in v for li in range(x.shape[0])]
            dist[R] = round(sum(n) / max(1, len(n)), 2)
        print(json.dumps({"windows": {R: len(v) for R, v in sorted(wins.items())}, "distinct_mean": dist}), flush=True)
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
        rw = wins.get(R) if forced is not None else None
        if forced is not None:
            if not rw:
                print(f"[proxy] no saved windows of {R} rows: skipped", flush=True)
                continue
            forced.set(rw[0])
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
            if R in [int(x) for x in pr.split(",")]:
                d = decs[pv]
                for _ in range(5):
                    for gr in d.graphs:
                        gr.replay()
                torch.cuda.synchronize()
                from torch.profiler import ProfilerActivity, profile
                nrep = 3 * (len(rw) if rw else 1)
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for i in range(nrep):
                        if rw:
                            forced.set(rw[i % len(rw)])
                        for gr in d.graphs:
                            gr.replay()
                    torch.cuda.synchronize()
                ev = [(e.name, e.time_range.start, e.time_range.end) for e in prof.events()
                      if e.device_type == torch.autograd.DeviceType.CUDA]
                ev.sort(key=lambda t: t[1])
                stem = (a.out or "/w/res/prof") + f".R{R}"
                json.dump(ev, open(stem + ".kern.json", "w"))
                summ = profile_summary(ev, nrep)
                json.dump(summ, open(stem + ".sum.json", "w"), indent=1)
                print(f"profiled {len(ev)} kernels", flush=True)
                print(json.dumps({k: v for k, v in summ.items() if k != "kernels"}), flush=True)
                for name, us, cnt in summ["kernels"][:45]:
                    print(f"  {us:9.1f} us  x{cnt:6.1f}  {name[:110]}", flush=True)
            continue
        # bits: each step new ids, every variant replayed on the same inputs, logits compared
        eq = {name: True for name in variants}
        first = next(iter(variants))
        for st in range(a.steps):
            ids = torch.randint(1000, 100000, (R,), generator=g).tolist()
            if rw:
                forced.set(rw[st % len(rw)])
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
        nw = len(rw) if rw else 1
        ts = {name: [[] for _ in range(nw)] for name in variants}
        for rd in range(a.rounds):
            order = list(decs.items()) if rd % 2 == 0 else list(decs.items())[::-1]
            for wi in range(nw):
                if rw:
                    forced.set(rw[wi])
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
                    ts[name][wi].append(e0.elapsed_time(e1) / a.iters)
        r = {name: {"ms_median": round(statistics.mean(statistics.median(x) for x in v), 3),
                    "ms_min": round(statistics.mean(min(x) for x in v), 3)} for name, v in ts.items()}
        if nw > 1:
            for name, v in ts.items():
                r[name]["ms_windows"] = [round(statistics.median(x), 3) for x in v]
        r["bits_equal_" + first] = {k: bool(v) for k, v in eq.items()}
        res["rows"][R] = r
        print(json.dumps({"R": R, **r}), flush=True)
        del decs
    if a.host:
        res["host"] = host_rounds(m, pool, RD, variants, a, rows, wins if forced is not None else None, forced, gpool)
    if a.invariance:
        res["invariance"] = invariance(m, pool, RD, variants, a)
        print(json.dumps({"invariance": res["invariance"]}), flush=True)
    if a.save:
        torch.save(saved, a.save)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def host_rounds(m, pool, RD, variants, a, rows, wins, forced, gpool):
    """Whole rounds.RoundRunner.forward calls (one stream in slot 0 at position --pos, its host tokens random, a
    window's routing when --picks), each followed by the logits' argmax read back; wall clock, synchronized before;
    with --draft-ms each call follows a round's start (its first row's Engram rows touched, a drafter stand-in on the
    GPU, the host waiting for it). Each variant runs blocks of 10 consecutive rounds (what one round leaves the next,
    e.g. the drive's idle time, stays the variant's own), the variants' blocks alternating. Returns {R: {variant:
    {ms_median, ms_mean, ms_min, GPU idle before each stretch (mean, median)}}}."""

    import random
    import time

    rnd = random.Random(3)
    host = [rnd.randrange(1000, 100000) for _ in range(a.pos)]
    out = {}
    # each stretch graph's replay between two events: the GPU's idle time before each stretch (waiting for the host)
    evs: list = []
    run0 = RD.RoundDecoder.run

    def run(self, ids, pos, slots, base, end, e_rows, e_start=None):
        """RoundDecoder.run (the same steps in the same order) with an event pair around each stretch's replay."""

        if self.graphs is None:
            return run0(self, ids, pos, slots, base, end, e_rows, e_start)

        def start(k):
            if e_start:
                nxt = self.stretches[k + 1][0] if k + 1 < len(self.stretches) else None
                for i, fn in e_start.items():
                    if (k < 0 and i < self.stretches[0][1]) or (k >= 0 and i == nxt):
                        fn()

        self.set(ids, pos, slots, base, end)
        start(-1)
        marks = []
        for k, (first, _) in enumerate(self.stretches):
            for i in self.e_in:
                if e_rows and (i == first or (k == 0 and i < self.stretches[0][1])):
                    t = e_rows[i]
                    self.e_in[i].copy_(t() if callable(t) else t)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            self.graphs[k].replay()
            e1.record()
            marks.append((e0, e1))
            start(k)
        evs.append(marks)
        return self.logits

    RD.RoundDecoder.run = run
    BLOCK = 10
    cyc = int(a.draft_ms * 1000 * torch.cuda.get_device_properties(0).clock_rate / 1000) if a.draft_ms else 0
    for R in rows:
        rw = wins.get(R) if wins is not None else None
        if wins is not None and not rw:
            continue
        runners = {}
        for name, f in variants.items():
            with flags(f):
                rr = RD.RoundRunner(m, pool, graphs=True, graph_pool=gpool)
                if rw:
                    forced.set(rw[0])
                rr.forward([(0, 0, pool.cap, a.pos, host[:R], host + host[:R])], replay=False)
            runners[name] = rr
        names = list(runners)
        ts = {name: [] for name in names}
        gaps = {name: [] for name in names}
        nblk = max(1, -(-a.host // BLOCK))
        for blk in range(nblk + 1):                      # (the first block of each: warm-up, not counted)
            for name in (names if blk % 2 == 0 else names[::-1]):
                rr = runners[name]
                for j in range(BLOCK):
                    toks = [rnd.randrange(1000, 100000) for _ in range(R)]
                    if rw:
                        forced.set(rw[(blk * BLOCK + j) % len(rw)])
                    with flags(variants[name]):
                        torch.cuda.synchronize()
                        if a.draft_ms:
                            rr.engram_touch([(host + toks[:1])[-9:]])
                            torch.cuda._sleep(cyc)
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        lg, _ = rr.forward([(0, 0, pool.cap, a.pos, toks, host + toks)])
                        lg.argmax(-1).tolist()
                        t1 = time.perf_counter()
                    if blk > 0:
                        ts[name].append(1000 * (t1 - t0))
                        mk = evs[-1]
                        gaps[name].append([mk[k - 1][1].elapsed_time(mk[k][0]) for k in range(1, len(mk))])
                    evs.clear()
        out[R] = {}
        for name in names:
            v, g = ts[name], gaps[name]
            ng = len(g[0]) if g and g[0] else 0
            out[R][name] = {"ms_median": round(statistics.median(v), 3), "ms_mean": round(statistics.mean(v), 3),
                            "ms_min": round(min(v), 3),
                            "idle_mean_ms": [round(statistics.mean(x[k] for x in g), 3) for k in range(ng)],
                            "idle_median_ms": [round(statistics.median(x[k] for x in g), 3) for k in range(ng)]}
        print(json.dumps({"host_R": R, **out[R]}), flush=True)
        del runners
    RD.RoundDecoder.run = run0
    return out


def profile_summary(ev, nrep):
    """Kernel events (name, start, end) of nrep forwards -> CUDA time and count a forward by kernel name (sorted), the
    span of a forward, the busy time (the union of kernel intervals) and the idle gaps."""

    by: dict = {}
    for name, t0, t1 in ev:
        x = by.setdefault(name, [0.0, 0])
        x[0] += t1 - t0
        x[1] += 1
    ks = sorted(((n, v[0] / nrep, v[1] / nrep) for n, v in by.items()), key=lambda r: -r[1])
    busy, end = 0.0, None
    for _, t0, t1 in ev:
        if end is None or t0 > end:
            busy += t1 - t0
            end = t1
        elif t1 > end:
            busy += t1 - end
            end = t1
    span = (ev[-1][2] - ev[0][1]) if ev else 0.0
    return {"forwards": nrep, "kernels_a_forward": round(len(ev) / nrep, 1),
            "sum_us": round(sum(r[1] for r in ks), 1), "busy_us": round(busy / nrep, 1),
            "span_us": round(span / nrep, 1), "idle_us": round((span - busy) / nrep, 1),
            "kernels": [(n, round(t, 1), round(c, 2)) for n, t, c in ks]}


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
