"""The decode windows' EXL3 linears of DeepSeek-V4.1 on one GPU (rank 0's parts of a TP2 split, real weights): today's
per-layer path (rot_in + linear a layer, wo_a's slices copied and concatenated; model.GROUPED off) against the grouped
launches (linear_grouped.cu: one rot_many + one glinear a group; wq_a + wkv (+ the compressor's) as one group, wo_a's
four slices as one group read and written in place; programmatic dependent launches; model.GROUPED on), through the
family's own call helpers (model.mm / attn_in / wo_a_out).

  bits     every output of both paths compared (torch.equal) at each row count, and their SHA-256 saved (--save) or
           checked against a saved run (--against: e.g. one made at the base commit, before any change)
  rows     row invariance of the grouped kernels: a row alone against the same row inside windows of 2..16 (and 40)
           random other rows at a random place, torch.equal, for every kind of launch (wq_a group, wo_a group, wq_b,
           the indexer's, wo_b on a strided input, the head, Engram), and against today's path on the row alone
  time     CUDA-graph replay (many iterations, old and new interleaved, median of rounds) of one layer of each kind, of
           each kind of linear over all 40 layers, and of every linear of a forward (40 layers + Engram + head)

  python3 linear_bench.py --model M [--rows 1,2,4,6,8,16] [--layers 40] [--save F | --against F] --out F
"""

import argparse
import hashlib
import json
import statistics
import time
from types import SimpleNamespace

import torch

BF16, F32 = torch.bfloat16, torch.float32


def load_attn(sh, cfg, i, rank, world):
    """Layer i's linears as load_block takes them (no experts)."""

    from tensorfold.families.deepseek_v41.cuda.weights import linear

    p = f"layers.{i}"
    Hl, hd, gl = cfg.n_heads // world, cfg.head_dim, cfg.o_groups // world
    L = SimpleNamespace(idx=i, ratio=cfg.compress_ratios[i], comp_wkv=None, comp_wgate=None, idx_wq_b=None,
                        idx_wk=None, engram_wkv=None)
    L.wq_a = linear(sh, f"{p}.attn.wq_a")
    L.wq_b = linear(sh, f"{p}.attn.wq_b", cols=(rank * Hl * hd, (rank + 1) * Hl * hd))
    L.wkv = linear(sh, f"{p}.attn.wkv")
    L.wo_a = [linear(sh, f"{p}.attn.wo_a.slice.{g}") for g in range(rank * gl, (rank + 1) * gl)]
    L.wo_b = linear(sh, f"{p}.attn.wo_b", rows=(rank * gl * cfg.o_rank, (rank + 1) * gl * cfg.o_rank))
    if i in cfg.kv_sources:
        L.comp_wkv = linear(sh, f"{p}.attn.compressor.wkv")
        if L.ratio > 1:
            L.comp_wgate = linear(sh, f"{p}.attn.compressor.wgate")
    if i in cfg.index_sources:
        L.idx_wq_b = linear(sh, f"{p}.attn.indexer.wq_b")
        if i in cfg.kv_sources:
            L.idx_wk = linear(sh, f"{p}.attn.indexer.wk")
    if i in cfg.engram_layers:
        per = (cfg.engram_ngram - 1) * cfg.engram_heads // world * cfg.engram_dim
        L.engram_wkv = linear(sh, f"{p}.engram.wkv", rows=(rank * per, (rank + 1) * per))
    return L


def kind(L):
    return ("kv2" if L.comp_wgate is not None else "kv1" if L.comp_wkv is not None else
            "idx" if L.idx_wq_b is not None else "plain") + ("+engram" if L.engram_wkv is not None else "")


def layer_linears(M, L, inp, grouped):
    """One attention block's linears in the decode graph's order (graph.py / rounds.py): a dict of outputs."""

    M.GROUPED = grouped
    x, qr, lat, o, e = inp["x"], inp["qr"], inp["lat"], inp["o"], inp["e"]
    r = {}
    if L.engram_wkv is not None:
        r["engram"] = M.mm(L.engram_wkv, e, F32)
    if grouped:
        qa, kv, ck, cg = M.attn_in(L, x, comp=bool(L.ratio))
        u = None
    else:                                    # today's code, verbatim in effect: one mm a layer, slices copied + cat
        qa, kv = M.mm(L.wq_a, x), M.mm(L.wkv, x)
        ck = cg = None
        if L.ratio and L.comp_wkv is not None:
            ck = M.mm(L.comp_wkv, x) if L.ratio == 1 else M.mm(L.comp_wkv, x, F32)
            cg = M.mm(L.comp_wgate, x, F32) if L.ratio > 1 else None
    r["wq_a"], r["wkv"] = qa, kv
    if ck is not None:
        r["comp_wkv"] = ck
    if cg is not None:
        r["comp_wgate"] = cg
    r["wq_b"] = M.mm(L.wq_b, qr)
    if L.idx_wq_b is not None:
        r["idx_wq_b"] = M.mm(L.idx_wq_b, qr)
    if L.idx_wk is not None:
        r["idx_wk"] = M.mm(L.idx_wk, lat)
    if grouped:
        u = M.wo_a_out(L, o)
    else:
        n = o.shape[0]
        og = o.view(n, len(L.wo_a), -1)
        u = torch.cat([M.mm(wo, og[:, g].contiguous()) for g, wo in enumerate(L.wo_a)], -1)
    r["wo_a"] = u
    r["wo_b"] = M.mm(L.wo_b, u, F32)
    return r


def forward_linears(M, layers, head, inp, grouped):
    out = {}
    for L in layers:
        for k, v in layer_linears(M, L, inp, grouped).items():
            out[f"{L.idx}.{k}"] = v
    if head is not None:
        M.GROUPED = grouped
        out["head"] = M.mm(head, inp["xc"], F32)
    return out


def inputs(cfg, n, gen, world=2):
    Hl = cfg.n_heads // world
    per = (cfg.engram_ngram - 1) * cfg.engram_heads // world * cfg.engram_dim

    def r(*shape):
        return torch.randn(shape, generator=gen, device="cuda").to(BF16)

    return {"x": r(n, cfg.dim), "qr": r(n, cfg.q_rank), "lat": r(n, cfg.head_dim),
            "o": r(n, Hl, cfg.head_dim), "e": r(n, per), "xc": r(n, cfg.dim),
            "u": r(n, cfg.o_groups // world * cfg.o_rank)}


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:24]


def graph_of(fn, pool):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()                                 # warm-up outside the capture: groups, counters, kernels
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def time_graph(g, iters):
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / iters          # us a replay


def ab(fn_old, fn_new, iters, rounds, pool):
    """(old us, new us) medians over rounds of interleaved replays, and the graphs' outputs."""

    go, oo = graph_of(fn_old, None)          # each graph its own memory pool (a shared one outlives its graphs)
    gn, on = graph_of(fn_new, None)
    to, tn = [], []
    for _ in range(rounds):
        to.append(time_graph(go, iters))
        tn.append(time_graph(gn, iters))
    del go, gn
    return statistics.median(to), statistics.median(tn), min(to), min(tn), oo, on


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rows", default="1,2,4,6,8,16")
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--save")
    p.add_argument("--against")
    p.add_argument("--bits-only", action="store_true")
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--modes", default="old,new,new_nopdl")
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--pdl", type=int, default=1, help="programmatic dependent launches in the bits and rows checks")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import model as M
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, linear

    has_grouped = hasattr(M, "attn_in")
    if has_grouped:
        import tensorfold.cuda.exl3.linear as LIN0

        LIN0.PDL = bool(a.pdl)
    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    t0 = time.time()
    rank, world = 0, 2
    layers = [load_attn(sh, cfg, i, rank, world) for i in range(a.layers)]
    vocab_l = cfg.vocab // world
    head = linear(sh, "head", cols=(rank * vocab_l, (rank + 1) * vocab_l))
    torch.cuda.synchronize()
    res = {"load_s": round(time.time() - t0, 1), "gpu_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
           "grouped_available": has_grouped, "pdl": a.pdl, "bits": {}, "hashes": {}, "row_invariance": {},
           "time_us": {}}
    print(json.dumps({k: res[k] for k in ("load_s", "gpu_gib", "grouped_available")}), flush=True)
    rows = [int(r) for r in a.rows.split(",")]
    gen = torch.Generator(device="cuda").manual_seed(1234)
    ins = {n: inputs(cfg, n, gen) for n in rows}

    # -- bits: old against new, every output, every row count; hashes for --save / --against ----------------------
    for n in rows:
        old = forward_linears(M, layers, head, ins[n], False)
        torch.cuda.synchronize()
        res["hashes"][f"old.{n}"] = {k: sha(v) for k, v in old.items()}
        if has_grouped:
            new = forward_linears(M, layers, head, ins[n], True)
            torch.cuda.synchronize()
            res["hashes"][f"new.{n}"] = {k: sha(v) for k, v in new.items()}
            bad = [k for k in old if not torch.equal(old[k], new[k])]
            res["bits"][f"rows{n}"] = {"outputs": len(old), "equal": len(old) - len(bad), "differ": bad[:20]}
            print(json.dumps({f"bits rows{n}": res["bits"][f"rows{n}"]}), flush=True)
        del old
    if a.save:
        json.dump(res["hashes"], open(a.save, "w"))
    if a.against:
        ref = json.load(open(a.against))
        for n in rows:
            ro = ref.get(f"old.{n}", {})
            for path in ("old", "new"):
                h = res["hashes"].get(f"{path}.{n}")
                if h is None:
                    continue
                miss = [k for k in h if ro.get(k) != h[k]]
                res["bits"][f"{path}_vs_saved_rows{n}"] = {"outputs": len(h), "equal": len(h) - len(miss),
                                                            "differ": miss[:20]}
                print(json.dumps({f"{path} vs saved rows{n}": res["bits"][f"{path}_vs_saved_rows{n}"]}), flush=True)
    if not has_grouped or a.bits_only:
        json.dump(res, open(a.out, "w"), indent=1)
        return

    # -- row invariance of the grouped launches ---------------------------------------------------------------------
    M.GROUPED = True
    by_kind = {}
    for L in layers:
        by_kind.setdefault(kind(L), L)
    g2 = torch.Generator(device="cuda").manual_seed(99)
    probes = {
        "attn_in(kv2: wq_a+wkv+comp_wkv+comp_wgate)": lambda inp: M.attn_in(by_kind["kv2"], inp["x"]),
        "attn_in(kv1: wq_a+wkv+comp_wkv)": lambda inp: M.attn_in(by_kind["kv1"], inp["x"]),
        "attn_in(plain: wq_a+wkv)": lambda inp: M.attn_in(by_kind["plain"], inp["x"]),
        "wo_a_out(4 slices, strided)": lambda inp: (M.wo_a_out(by_kind["plain"], inp["o"]),),
        "wq_b": lambda inp: (M.mm(by_kind["plain"].wq_b, inp["qr"]),),
        "idx_wq_b": lambda inp: (M.mm(by_kind["idx"].idx_wq_b, inp["qr"]),),
        "idx_wk(8 bit)": lambda inp: (M.mm(by_kind["kv2"].idx_wk, inp["lat"]),),
        "wo_b(strided input)": lambda inp: (M.mm(by_kind["plain"].wo_b, inp["o"].view(inp["o"].shape[0], -1)[:, :4096],
                                                 F32),),
        "engram": lambda inp: (M.mm(next(L for L in layers if L.engram_wkv is not None).engram_wkv, inp["e"], F32),),
        "head(6 bit)": lambda inp: (M.mm(head, inp["xc"], F32),),
    }
    for name, fn in probes.items():
        ok, checks = True, 0
        for _ in range(a.trials):
            probe = inputs(cfg, 1, g2)
            alone = [t.clone() for t in fn(probe) if t is not None]
            M.GROUPED = False                # today's path on the probe row alone: the same bits
            old1 = [t.clone() for t in fn(probe) if t is not None]
            M.GROUPED = True
            ok &= all(torch.equal(u, v) for u, v in zip(alone, old1))
            for w in list(range(2, 17)) + [40]:
                others = inputs(cfg, w, g2)
                at = int(torch.randint(0, w, (1,), generator=g2, device="cuda"))
                win = {k: v.clone() for k, v in others.items()}
                for k in win:
                    win[k][at] = probe[k][0]
                outs = [t for t in fn(win) if t is not None]
                ok &= all(torch.equal(u[at:at + 1], v) for u, v in zip(outs, alone))
                checks += 1
        res["row_invariance"][name] = {"windows": checks, "all_equal": bool(ok)}
        print(json.dumps({f"rows {name}": res["row_invariance"][name]}), flush=True)

    if a.no_time:
        json.dump(res, open(a.out, "w"), indent=1)
        return

    # -- time: CUDA graph replays ------------------------------------------------------------------------------------
    import tensorfold.cuda.exl3.linear as LIN

    kinds = ["plain", "idx", "kv2", "kv1", "kv2+engram", "plain+engram"]
    res["layer_kinds"] = {k: sum(1 for L in layers if kind(L) == k) for k in kinds}
    modes = {"old": (False, True), "new": (True, True), "new_nopdl": (True, False)}
    which_all = tuple(a.modes.split(","))

    def variant(fn, mode):
        grouped, pdl = modes[mode]

        def run():
            M.GROUPED = grouped
            LIN.PDL = pdl
            return fn(grouped)
        return run

    def bench(fn, iters, which=None):
        which = which or which_all
        graphs = {}
        outs = {}
        for mode in which:
            graphs[mode], outs[mode] = graph_of(variant(fn, mode), None)
        ts = {mode: [] for mode in which}
        for _ in range(a.rounds):
            for mode in which:
                ts[mode].append(time_graph(graphs[mode], iters))
        del graphs
        LIN.PDL = True
        r = {mode: round(statistics.median(v), 1) for mode, v in ts.items()}
        r.update({f"{mode}_min": round(min(v), 1) for mode, v in ts.items()})
        return r, outs

    for n in rows:
        inp = ins[n]
        t = {}
        for k in ([] if a.forward_only else kinds):
            if k in by_kind:
                L = by_kind[k]
                t[f"layer {k} (L{L.idx})"], _ = bench(lambda g, L=L: layer_linears(M, L, inp, g), a.iters * 2)
        sel = {
            "wq_a+wkv(+comp)": lambda L, g: M.attn_in(L, inp["x"], comp=bool(L.ratio)) if g else (
                [M.mm(L.wq_a, inp["x"]), M.mm(L.wkv, inp["x"])] +
                ([M.mm(L.comp_wkv, inp["x"], BF16 if L.ratio == 1 else F32)] if L.comp_wkv else []) +
                ([M.mm(L.comp_wgate, inp["x"], F32)] if L.comp_wgate else [])),
            "wq_b": lambda L, g: M.mm(L.wq_b, inp["qr"]),
            "idx_wq_b": lambda L, g: M.mm(L.idx_wq_b, inp["qr"]) if L.idx_wq_b is not None else None,
            "idx_wk": lambda L, g: M.mm(L.idx_wk, inp["lat"]) if L.idx_wk is not None else None,
            "wo_a(4)": lambda L, g: M.wo_a_out(L, inp["o"]) if g else torch.cat(
                [M.mm(wo, inp["o"].view(n, 4, -1)[:, j].contiguous()) for j, wo in enumerate(L.wo_a)], -1),
            "wo_b": lambda L, g: M.mm(L.wo_b, inp["u"], F32),
            "engram": lambda L, g: M.mm(L.engram_wkv, inp["e"], F32) if L.engram_wkv is not None else None,
        }
        for name, f in ({} if a.forward_only else sel).items():
            cnt = sum(1 for L in layers if f(L, False) is not None)
            r, _ = bench(lambda g, f=f: [f(L, g) for L in layers], max(5, a.iters * 8 // max(cnt, 1)))
            r["layers"] = cnt
            t[f"{name} (all layers)"] = r
        if not a.forward_only:
            t["head"], _ = bench(lambda g: M.mm(head, inp["xc"], F32), a.iters)
        r, outs = bench(lambda g: forward_linears(M, layers, head, inp, g), max(5, a.iters // 5))
        r["saving_new"] = round(r["old"] - r["new"], 1) if "new" in r and "old" in r else None
        r["graph_outputs_equal"] = all(torch.equal(outs["old"][k], outs[m][k]) for m in outs for k in outs["old"])
        t[f"forward ({len(layers)} layers + engram + head)"] = r
        res["time_us"][f"rows{n}"] = t
        print(json.dumps({f"time rows{n}": t}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
