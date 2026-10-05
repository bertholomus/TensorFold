"""Time the drafter's Markov loop alone (markov.Markov.steps, CUDA-graph captured) against 3b9818d's torch loop at
N = 1, 2, 4 streams (5, 5, 3 steps): every step a cache hit (a slot table sending every token to a cached row: timing
only) or a miss, split over two emulated ranks (the gather: two copies, as forward_proxy's stand-in) or not.

  python3 markov_bench.py --model M --out F
"""

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch


class Dup:
    world = 2

    def gather(self, x):
        out = torch.empty((2, *x.shape), dtype=x.dtype, device=x.device)
        out[0].copy_(x)
        out[1].copy_(x)
        return out


def timed(fn, iters=50, rounds=7):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(rounds):
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters * 1000)
    return round(statistics.median(ts), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import markov as MK
    from tensorfold.families.deepseek_v41.cuda.weights import Shards

    sh = Shards(a.model)
    head = sh.get("mtp.2.markov_head.head.weight").contiguous().cuda()
    emb = sh.get("mtp.2.markov_head.embed.weight").contiguous().cuda()
    V = head.shape[0]
    Vl = V // 2
    res = {}
    n = 5
    for N, steps in ((1, 5), (2, 5), (4, 3)):
        row = {}
        R = N * n
        local = torch.randn((R, Vl), device="cuda") * 3
        gathered = torch.stack([local, torch.randn((R, Vl), device="cuda") * 3])
        logits = gathered.permute(1, 0, 2).reshape(R, -1).view(N, n, -1)
        out = torch.empty((N, steps + 1), dtype=torch.long, device="cuda")
        out[:, 0] = torch.arange(N, device="cuda") + 300

        def old():
            o = out
            for i in range(steps):
                e = emb[o[:, i]]
                o[:, i + 1] = (logits[:, i] + (e.to(torch.float16) @ head.t()).float()).argmax(-1)
        row["old_loop_us"] = timed(old)

        def old_with_copy():                          # the old graph's permute copy of the gathered logits too
            lg = gathered.permute(1, 0, 2).reshape(R, -1).view(N, n, -1)
            o = out
            for i in range(steps):
                e = emb[o[:, i]]
                o[:, i + 1] = (lg[:, i] + (e.to(torch.float16) @ head.t()).float()).argmax(-1)
        row["old_loop+permute_us"] = timed(old_with_copy)
        for split in (True, False):
            w = SimpleNamespace(world=2, rank=0, dspark=SimpleNamespace(markov_head=head, markov_embed=emb))
            mk = MK.Markov(SimpleNamespace(w=w, cfg=SimpleNamespace(vocab=V), comm=Dup()), rows=4096, split=split)
            lg = local if split else gathered
            for mode in ("hit", "miss"):
                mk.slot = ((torch.arange(V, device="cuda") % 4096).to(torch.int32) if mode == "hit" else mk.none)
                row[f"{'split' if split else 'nosplit'}_{mode}_us"] = timed(lambda: mk.steps(lg, out, n, steps))
            del mk
        res[f"N{N}_steps{steps}"] = row
        print(json.dumps({f"N{N}_steps{steps}": row}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
