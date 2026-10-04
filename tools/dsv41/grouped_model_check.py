"""TF_DS_GROUPED_LINEAR end to end on one GPU: the family's own decode paths with the grouped EXL3 launches (1) against
the per-layer path (0), through the call sites they change (model.py's eager attention, graph.py's StaticDecoder,
rounds.py's RoundDecoder, dspark.py's absorb / draft / DraftGraph / BatchDraftGraph). Rank 0's weights of a TP2 split,
the first --layers layers and the DSpark blocks, the other rank stood in by this rank's own partials (a gather that
repeats them: a rank's own arithmetic at the model's shapes, not the model's output); every logits / taps / drafts
tensor saved. Run once with each flag value, then --compare.

  TF_DS_GROUPED_LINEAR=0 python3 grouped_model_check.py --model M --out off.pt
  TF_DS_GROUPED_LINEAR=1 python3 grouped_model_check.py --model M --out on.pt
  python3 grouped_model_check.py --compare off.pt on.pt
"""

import argparse
import json
import os

import torch


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    keys = [k for k in a if not k.startswith("_")]
    res = {k: bool(k in b and a[k].shape == b[k].shape and torch.equal(a[k], b[k])) for k in keys}
    res["_all_equal"] = all(res.values()) and set(a) == set(b)
    res["_flags"] = [int(a["_flag"]), int(b["_flag"])]
    print(json.dumps(res, indent=1, default=str))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2)
    a = p.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    from tensorfold.families.deepseek_v41.cuda import model as M
    from tensorfold.families.deepseek_v41.cuda.dspark import BatchDraftGraph, DraftGraph, DraftPool, Drafter
    from tensorfold.families.deepseek_v41.cuda.graph import GraphRunner
    from tensorfold.families.deepseek_v41.cuda.rounds import RoundRunner
    from tensorfold.families.deepseek_v41.cuda.weights import load

    torch.cuda.set_device(0)
    class Repeat(M.Comm):
        """World 2 with rank 0's partials standing in for rank 1's: every shape as served, no second GPU."""

        def __init__(self):
            super().__init__(None, 1)
            self.world = 2

        def gather(self, x):
            x = x.contiguous()
            return torch.stack([x, x])

    w = load(a.model, 0, 2, n_layers=a.layers, dspark=True)
    m = M.Model(w, Repeat())
    m.rope_cap = 8192
    c = w.cfg
    print(f"loaded {a.layers} layers + DSpark: {torch.cuda.memory_allocated() / 2**30:.1f} GiB; grouped={M.GROUPED}",
          flush=True)
    g = torch.Generator().manual_seed(0)
    out = {"_flag": torch.tensor([int(M.GROUPED)])}

    def toks(n):
        return torch.randint(0, c.vocab, (n,), generator=g).tolist()

    def dev(ids):
        return torch.tensor(ids, dtype=torch.long, device="cuda")

    with torch.inference_mode():
        # one sequence: an eager prompt (300 rows: the prompt GEMM), eager windows (<= 128 rows: attention_k with the
        # grouped launches), then graph windows of 1..6 rows (StaticDecoder)
        sc = m.new_cache(4096)
        out["prefill"] = m.forward(sc, dev(toks(300)), 0).clone()
        out["eager100"] = m.forward(sc, dev(toks(100)), sc.length, all_logits=True).clone()
        out["eager5"] = m.forward(sc, dev(toks(5)), sc.length, all_logits=True).clone()
        gr = GraphRunner(m, 6)
        for n in (1, 2, 4, 6, 6, 1):
            lo, taps = gr.forward(sc, toks(n), sc.length, n > 1)
            out[f"graph{n}@{sc.length}"] = lo.clone()
        # three streams in one pool: eager prompts on their slot views, then concurrent rounds (RoundDecoder graphs)
        pool = m.new_pool(3, 3 * 1024)
        views = [m.pool_view(pool, s, s * 1024, 1024) for s in range(3)]
        lens = [120, 77, 64]
        for s, v in enumerate(views):
            out[f"pool_prompt{s}"] = m.forward(v, dev(toks(lens[s])), 0).clone()
        rr = RoundRunner(m, pool, graphs=True)
        for step, rows in enumerate([(1, 1, 1), (6, 1, 3), (6, 6, 4)]):
            windows = []
            for s, n in enumerate(rows):
                windows.append((s, s * 1024, 1024, lens[s], toks(n), []))
            lo, taps = rr.forward(windows)
            out[f"round{step}"] = lo.clone()
            if taps is not None:
                out[f"round{step}_taps"] = taps.clone()
            lens = [lens[s] + rows[s] for s in range(3)]
        # the drafter: absorb, an eager draft, its graph, and a batch of two streams
        d = Drafter(m)
        dc = d.new_cache()
        taps = (torch.randn((40, 3 * c.dim), generator=g) * 0.5).to(torch.bfloat16).cuda()
        d.absorb(dc, sc, taps, 0)
        drafts, conf = d.draft(dc, sc, 17, 40)
        out["draft_eager"] = torch.tensor(drafts)
        out["draft_eager_conf"] = conf.clone()
        dg = DraftGraph(d, sc, dc)
        dg.token.fill_(17)
        dg.q0.fill_(40)
        dg.capture()
        drafts, conf = dg.run(17, 40)
        out["draft_graph"] = dg.packed.clone()
        dp = DraftPool(d, 2)
        t2 = (torch.randn((2, 30, 3 * c.dim), generator=g) * 0.5).to(torch.bfloat16).cuda()
        d.absorb_many(dp, sc, [(0, t2[0], 0), (1, t2[1], 0)])
        bg = BatchDraftGraph(d, sc, dp, 2)
        bg.capture()
        bg.run([5, 9], [30, 30], [0, 1])
        out["draft_batch"] = bg.packed.clone()
    torch.cuda.synchronize()
    torch.save({k: v.cpu() for k, v in out.items()}, a.out)
    print(f"saved {len(out)} tensors to {a.out}", flush=True)


if __name__ == "__main__":
    main()
