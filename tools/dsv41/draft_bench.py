"""A one-GPU proxy of the DSpark drafter graph (dspark.BatchDraftGraph) with rank 0's TP2 drafter weights (its three
stages, main_proj, the head's vocabulary half, the replicated Markov head / embedding / confidence head), each TP
gather replaced by a local stand-in (rank 0's tensor in every rank's slot: two small kernels, as the RDMA gather's
stage + collect, plus an optional spin of --gather-us for the peer's round trip). Variants are module flags set while
a variant's graphs are captured (forward_proxy's syntax); a variant named "...@allhit" gets a slot table that sends
every token to a cached row (timing only: every Markov step a cache hit), "...@allmiss" an empty one. The graphs are
replayed interleaved (median of rounds) and their outputs (drafts and confidences, bit for bit) compared with the first
variant's on random tokens.

  python3 draft_bench.py --model M [--streams 1,2,4] [--steps 5,5,3] [--variants old=MOD.FLAG:val,new=]
                         [--gather-us 0] [--profile N:variant] --out F
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_proxy import LocalComm, flags, set_flags  # noqa: E402


class DupComm(LocalComm):
    """gather -> [world, *x.shape] with x in every slot (two small kernels), then the optional spin."""

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty((self.world, *x.shape), dtype=x.dtype, device=x.device)
        out[0].copy_(x)
        out[1:].copy_(x.expand(self.world - 1, *x.shape))
        if self.cycles:
            torch.cuda._sleep(self.cycles)
        return out


def load(model_dir, rank=0, world=2, other_head=False):
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import DSparkWeights, Shards, Weights, linear, load_block

    cfg = Cfg.read(model_dir)
    sh = Shards(model_dir)
    vocab_l = cfg.vocab // world

    def plain(name):
        return sh.get(name).contiguous().cuda()

    w = Weights(cfg, rank, world, embed=plain("embed.weight"), norm=plain("norm.weight"),
                head=linear(sh, "head", cols=(rank * vocab_l, (rank + 1) * vocab_l)),
                vocab_lo=rank * vocab_l, vocab_hi=(rank + 1) * vocab_l)
    n = len(cfg.compress_ratios) - cfg.n_layers
    last = f"mtp.{n - 1}"
    blocks = [load_block(sh, cfg, f"mtp.{j}", cfg.n_layers + j, rank, world, cfg.dspark_routed) for j in range(n)]
    w.dspark = DSparkWeights(blocks, linear(sh, "mtp.0.main_proj"), plain("mtp.0.main_norm.weight"),
                             plain(f"{last}.norm.weight"), plain(f"{last}.markov_head.embed.weight"),
                             plain(f"{last}.markov_head.head.weight"), plain(f"{last}.confidence_head.proj.weight"))
    heads = None
    if other_head:
        heads = [linear(sh, "head", cols=(r * vocab_l, (r + 1) * vocab_l)) if r != rank else w.head
                 for r in range(world)]
    sh.close()
    return cfg, w, heads


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--streams", default="1,2,4")
    p.add_argument("--steps", default="5,5,3", help="Markov steps a stream count (multi.py: depth table)")
    p.add_argument("--variants", default="old=")
    p.add_argument("--gather-us", type=float, default=0.0)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--ctx", type=int, default=700)
    p.add_argument("--profile")
    p.add_argument("--serial", action="store_true", help="also the serial engine's dspark.DraftGraph (one stream): "
                   "every variant's outputs against the first's, and its time")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import dspark as D
    from tensorfold.families.deepseek_v41.cuda import model as M

    t0 = time.time()
    cfg, w, _ = load(a.model)
    m = M.Model(w, DupComm(2, a.gather_us))
    m.rope_cap = 8192
    drafter = D.Drafter(m)
    slots = 4
    dpool = D.DraftPool(drafter, slots)
    sc = M.SeqCache(cap=8192, ring_size=dpool.ring_size)
    g = torch.Generator(device="cuda").manual_seed(7)
    for s in range(slots):                                   # rings: a context of a.ctx absorbed positions a slot
        taps = (torch.randn((dpool.ring_size, 3 * cfg.dim), generator=g, device="cuda") * 0.7).to(torch.bfloat16)
        drafter.absorb_many(dpool, sc, [(s, taps, a.ctx - dpool.ring_size)])
    torch.cuda.synchronize()
    print(json.dumps({"load_s": round(time.time() - t0, 1), "gpu_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}),
          flush=True)
    variants = {}
    for item in a.variants.split(","):
        name, _, spec = item.partition("=")
        variants[name] = set_flags(spec)
    gpool = torch.cuda.graph_pool_handle()
    res = {"args": vars(a), "streams": {}}
    hg = torch.Generator().manual_seed(11)
    for N, steps in zip([int(x) for x in a.streams.split(",")], [int(x) for x in a.steps.split(",")]):
        graphs = {}
        tokens = torch.randint(100, 60000, (N,), generator=hg).tolist()
        q0 = [a.ctx] * N
        sl = list(range(N))
        for name, f in variants.items():
            with flags(f):
                bg = D.BatchDraftGraph(drafter, sc, dpool, N, steps=steps)
                if bg.mk is not None and "@" in name:
                    import copy
                    bg.mk = copy.copy(bg.mk)
                    Kc = bg.mk.cache.shape[0]
                    if name.endswith("@allhit"):
                        bg.mk.slot = (torch.arange(cfg.vocab, device="cuda") % Kc).to(torch.int32)
                    elif name.endswith("@allmiss"):
                        bg.mk.slot = bg.mk.none
                bg.tokens.copy_(torch.tensor(tokens))
                bg.q0.copy_(torch.tensor(q0))
                bg.slots.copy_(torch.tensor(sl))
                bg.capture(gpool)
            graphs[name] = bg
        torch.cuda.synchronize()
        # drafts: every variant on the same inputs, several token sets
        same = {name: 0 for name in graphs}
        trials = 40
        for t in range(trials):
            toks = torch.randint(100, 120000, (N,), generator=hg).tolist()
            outs = {}
            for name, bg in graphs.items():
                bg.run(toks, q0, sl)
                outs[name] = bg.packed.clone()                # drafts and confidences, fp32
            ref = outs[next(iter(outs))]
            for name, o in outs.items():
                same[name] += int(torch.equal(o, ref))
        if a.profile and int(a.profile.split(":")[0]) == N:
            bg = graphs[a.profile.split(":")[1]]
            from torch.profiler import ProfilerActivity, profile
            for _ in range(5):
                bg.graph.replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    bg.graph.replay()
                torch.cuda.synchronize()
            ev = {}
            for e in prof.events():
                if e.device_type.name != "CUDA":
                    continue
                d = ev.setdefault(e.name, [0, 0.0])
                d[0] += 1
                d[1] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            rows = sorted(ev.items(), key=lambda kv: -kv[1][1])
            print(f"profile N={N}: {sum(v[0] for v in ev.values()) / 5:.0f} kernels a graph, "
                  f"{sum(v[1] for v in ev.values()) / 5 / 1000:.3f} ms busy", flush=True)
            for name, (cnt, us) in rows[:40]:
                print(f"  {us / 5:9.1f} us  {cnt / 5:5.1f}x  {name[:110]}", flush=True)
        times = {name: [] for name in graphs}
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            for bg in graphs.values():
                bg.graph.replay()
        for r in range(a.rounds):
            for name, bg in graphs.items():
                torch.cuda.synchronize()
                e0.record()
                for _ in range(a.iters):
                    bg.graph.replay()
                e1.record()
                torch.cuda.synchronize()
                times[name].append(e0.elapsed_time(e1) / a.iters)
        row = {name: {"ms": round(statistics.median(v), 4), "min": round(min(v), 4), "same_packed": f"{same[name]}/{trials}"}
               for name, v in times.items()}
        res["streams"][N] = {"steps": steps, **row}
        print(json.dumps({"N": N, "steps": steps, **row}), flush=True)
    if a.serial:
        graphs = {}
        for name, f in variants.items():
            with flags(f):
                dg = D.DraftGraph(drafter, sc, dpool.views[0])
                if dg.mk is not None and "@" in name:
                    import copy
                    dg.mk = copy.copy(dg.mk)
                    Kc = dg.mk.cache.shape[0]
                    dg.mk.slot = ((torch.arange(cfg.vocab, device="cuda") % Kc).to(torch.int32)
                                  if name.endswith("@allhit") else dg.mk.none)
                dg.capture(gpool)
            graphs[name] = dg
        same = {name: 0 for name in graphs}
        trials = 40
        for t in range(trials):
            tok = int(torch.randint(100, 120000, (1,), generator=hg))
            outs = {}
            for name, dg in graphs.items():
                dg.run(tok, a.ctx)
                outs[name] = dg.packed.clone()
            ref = outs[next(iter(outs))]
            for name, o in outs.items():
                same[name] += int(torch.equal(o, ref))
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = {name: [] for name in graphs}
        for r in range(a.rounds):
            for name, dg in graphs.items():
                torch.cuda.synchronize()
                e0.record()
                for _ in range(a.iters):
                    dg.graph.replay()
                e1.record()
                torch.cuda.synchronize()
                times[name].append(e0.elapsed_time(e1) / a.iters)
        row = {name: {"ms": round(statistics.median(v), 4), "same_packed": f"{same[name]}/{trials}"}
               for name, v in times.items()}
        res["serial"] = row
        print(json.dumps({"serial DraftGraph": row}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
