"""Model.moe of DeepSeek-V4.1 decode windows on real weights (one layer, rank 0's half of every expert, the real gate):
TF_DS_SHARED_OVERLAP (the shared expert on a side stream beside the gate, the routing and the routed gate/up) against
the plain sequence, and the L2 prefetch flag's MoE-side effect (none), bit for bit:

  bits        windows of 1..16 rows (random rows, the layer's own routing), overlap on vs off: torch.equal, eager and
              in CUDA graphs (each graph replayed on new inputs after capture)
  invariance  overlap on: each of 64 rows alone vs inside windows of 2..16 other rows in random order: torch.equal
  time        the MoE of R rows (gate, routing, experts) as a graph after a 20 us idle span (a gather's wait), L2
              flushed before each replay: off / off with the gate prefetched into L2 during the span (the decode
              bodies' "moe" fork) / on with it (what the decode bodies run), interleaved

  python3 overlap_check.py --model M [--layers 10,30] [--windows 200] --out F
"""

import argparse
import json
import random
import statistics
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="10,30")
    p.add_argument("--windows", type=int, default=200)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import model as M
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, Weights, load_block

    class Comm:
        world = 2

        def gather(self, x):
            return torch.stack([x, torch.zeros_like(x)])

        def sum(self, x):
            return x

    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    res = {"layers": {}}
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush_buf = torch.empty(max(4 * l2, 128 << 20) // 4, dtype=torch.int32, device="cuda")
    sink = torch.empty((), dtype=torch.int64, device="cuda")
    for li in [int(v) for v in a.layers.split(",")]:
        t0 = time.time()
        lay = load_block(sh, cfg, f"layers.{li}", li, 0, 2, cfg.n_routed)
        w = Weights(cfg, 0, 2, embed=torch.empty(0), norm=torch.empty(0), head=None, vocab_lo=0, vocab_hi=0,
                    layers=[lay])
        m = M.Model(w, Comm())
        D = cfg.dim
        g = torch.Generator(device="cuda").manual_seed(100 + li)
        xs = (torch.randn((64, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        r = {"load_s": round(time.time() - t0, 1)}

        def moe(x, on):
            M.SHARED_OVERLAP = on
            return m.moe(lay, x.contiguous(), shared_side=on)

        clk = getattr(torch.cuda.get_device_properties(0), "clock_rate", 2_400_000)

        def body(x, mode):
            if mode != "off":
                M.L2_PREFETCH = True
                M.l2_fork(m, lay, "moe")
            torch.cuda._sleep(int(20 * clk / 1000))         # the gather's wait
            out = moe(x, mode == "on_pf")
            M.l2_join(m)
            return out

        # bits, eager: windows of 1..16 random rows
        rnd = random.Random(li)
        bad = 0
        for wi in range(48):
            n = 1 + wi % 16
            idx = rnd.sample(range(64), n)
            o0 = moe(xs[idx], False).clone()
            o1 = moe(xs[idx], True).clone()
            torch.cuda.synchronize()
            bad += not torch.equal(o0, o1)
        r["eager_windows"] = 48
        r["eager_windows_differing"] = bad

        # bits in graphs: capture each R with its own static input, replay on new rows
        gbad, times = 0, {}
        modes = ("off", "off_pf", "on_pf")
        for n in (1, 2, 6, 16):
            graphs = {}
            for mode in modes:
                xin = torch.empty((n, D), dtype=torch.bfloat16, device="cuda")
                xin.copy_(xs[:n])
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    body(xin, mode)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    out = body(xin, mode)
                graphs[mode] = (gr, xin, out)
            for k in range(4):
                idx = rnd.sample(range(64), n)
                outs = []
                for mode in modes:
                    gr, xin, out = graphs[mode]
                    xin.copy_(xs[idx])
                    gr.replay()
                    torch.cuda.synchronize()
                    outs.append(out.clone())
                ref = moe(xs[idx], False)
                gbad += not all(torch.equal(o, ref) for o in outs)
            ts = {mode: [] for mode in modes}
            for it in range(a.iters):
                for mode in (modes if it % 2 == 0 else modes[::-1]):
                    gr = graphs[mode][0]
                    torch.sum(flush_buf, 0, dtype=torch.int64, out=sink)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    gr.replay()
                    e1.record()
                    torch.cuda.synchronize()
                    ts[mode].append(e0.elapsed_time(e1) * 1000)
            times[n] = {f"{mode}_us": round(statistics.median(v), 1) for mode, v in ts.items()}
            del graphs
        r["graph_checks_differing"] = gbad
        r["moe_graph_us"] = times

        # row invariance with the overlap on
        solo = torch.cat([moe(xs[i:i + 1], True).clone() for i in range(64)])
        rbad, checked = 0, 0
        for wi in range(a.windows):
            n = 2 + wi % 15
            idx = rnd.sample(range(64), n)
            o = moe(xs[idx], True)
            for j, i in enumerate(idx):
                checked += 1
                rbad += not torch.equal(o[j], solo[i])
        r["invariance"] = {"windows": a.windows, "rows_checked": checked, "rows_differing": rbad}
        res["layers"][li] = r
        print(json.dumps({"layer": li, **r}), flush=True)
        del lay, w, m
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
