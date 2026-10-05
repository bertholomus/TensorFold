"""Both TP ranks of the DSpark drafter on one GPU: rank 0's and rank 1's drafter weights (their heads' and experts'
halves, their own Markov cache columns), each rank's batched drafter pass (dspark.BatchDraftGraph, eager) in its own
thread and stream, every TP gather a real exchange between the two threads (each rank's tensor in its slot, rank
order). Variants are module flags (forward_proxy's syntax); for each, both ranks' outputs (drafts and confidences, fp32
bits) must be equal, and every variant's must equal the first's.

  python3 draft_tp_emu.py --model M [--streams 1,2,4] [--steps 5,5,3] [--trials 40]
                          [--variants old=MOD.ON:False,new=] --out F
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draft_bench import load  # noqa: E402
from forward_proxy import flags, set_flags  # noqa: E402


class Hub:
    def __init__(self, world=2):
        self.world = world
        self.bar = threading.Barrier(world)
        self.box = [None] * world


class EmuComm:
    """gather -> [world, *x.shape]: every rank's x in rank order (the threads meet twice: posted, copied)."""

    def __init__(self, rank, hub):
        self.rank, self.hub, self.world = rank, hub, hub.world

    def gather(self, x):
        x = x.contiguous()
        torch.cuda.current_stream().synchronize()
        self.hub.box[self.rank] = x
        self.hub.bar.wait()
        out = torch.empty((self.world, *x.shape), dtype=x.dtype, device=x.device)
        for r in range(self.world):
            out[r].copy_(self.hub.box[r])
        torch.cuda.current_stream().synchronize()
        self.hub.bar.wait()
        return out

    def sum(self, x):
        g = self.gather(x)
        acc = g[0].clone()
        for r in range(1, self.world):
            acc += g[r]
        return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--streams", default="1,2,4")
    p.add_argument("--steps", default="5,5,3")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--ctx", type=int, default=700)
    p.add_argument("--variants", default="old=")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import dspark as D
    from tensorfold.families.deepseek_v41.cuda import model as M

    t0 = time.time()
    hub = Hub(2)
    ranks = []
    embed = None
    for rk in range(2):
        cfg, w, _ = load(a.model, rank=rk)
        if embed is None:
            embed = w.embed
        else:
            w.embed = embed                                # replicated: one copy
        m = M.Model(w, EmuComm(rk, hub))
        m.rope_cap = 8192
        drafter = D.Drafter(m)
        dpool = D.DraftPool(drafter, 4)
        sc = M.SeqCache(cap=8192, ring_size=dpool.ring_size)
        ranks.append((m, drafter, dpool, sc))
    g = torch.Generator(device="cuda").manual_seed(7)
    for s in range(4):                                      # the same absorbed taps on both ranks (replicated)
        taps = (torch.randn((ranks[0][2].ring_size, 3 * cfg.dim), generator=g, device="cuda") * 0.7).to(torch.bfloat16)
        for m, drafter, dpool, sc in ranks:
            drafter.absorb_many(dpool, sc, [(s, taps, a.ctx - dpool.ring_size)])
    torch.cuda.synchronize()
    print(json.dumps({"load_s": round(time.time() - t0, 1), "gpu_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}),
          flush=True)
    variants = []
    for item in a.variants.split(","):
        name, _, spec = item.partition("=")
        variants.append((name, set_flags(spec)))
    res = {"args": vars(a), "streams": {}}
    hg = torch.Generator().manual_seed(11)
    for N, steps in zip([int(x) for x in a.streams.split(",")], [int(x) for x in a.steps.split(",")]):
        cases = [torch.randint(100, 120000, (N,), generator=hg).tolist() for _ in range(a.trials)]
        q0, sl = [a.ctx] * N, list(range(N))
        outs = {}
        for name, f in variants:
            with flags(f):
                graphs = [D.BatchDraftGraph(drafter, sc, dpool, N, steps=steps) for m, drafter, dpool, sc in ranks]
            got = [[None] * a.trials for _ in range(2)]

            def work(rk):
                st = torch.cuda.Stream()
                with torch.cuda.stream(st):
                    for t, toks in enumerate(cases):
                        graphs[rk].run(toks, q0, sl)
                        got[rk][t] = graphs[rk].packed.clone()
                    st.synchronize()

            with flags(f):                                  # _body() reads the module flags when it runs, too
                th = [threading.Thread(target=work, args=(rk,)) for rk in range(2)]
                for x in th:
                    x.start()
                for x in th:
                    x.join()
            outs[name] = got
        first = next(iter(outs))
        row = {}
        for name, got in outs.items():
            ranks_eq = sum(int(torch.equal(got[0][t], got[1][t])) for t in range(a.trials))
            eq_first = sum(int(torch.equal(got[0][t], outs[first][0][t])) for t in range(a.trials))
            row[name] = {"rank0_eq_rank1": f"{ranks_eq}/{a.trials}", f"eq_{first}": f"{eq_first}/{a.trials}"}
        hits = None
        mk = ranks[0][1].markov()
        if mk is not None and mk.tokens:
            dr = torch.stack([o[:, :steps] for o in outs[first][0]]).long().cuda()
            hits = round(float((mk.slot[dr] >= 0).float().mean()), 3)
        res["streams"][N] = {"steps": steps, **row, "draft_hit_rate": hits}
        print(json.dumps({"N": N, "steps": steps, **row, "draft_hit_rate": hits}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
