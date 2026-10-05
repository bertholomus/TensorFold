"""Exactness of the drafter's Markov kernels (markov.py) on one GPU, both TP ranks emulated.

1. bias bits: markov._bias's rows (through the cache fill) against 3b9818d's cuBLAS ``e.to(fp16) @ head.t()`` at the
   served shapes (N = 1, 2, 3, 4 rows; the serial graph's matrix-vector product), the vocabulary halves (rank 0 / rank 1
   columns) against the whole, and a step's own staging rows (row counts 1..16) against the fill's; --full: every
   token of the vocabulary against cuBLAS (N = 1 eager and replayed from a captured graph, N = 4 batches).
2. drafts: the loop over N streams x steps with real-scale logits: unsplit (one rank, gathered logits) == split (two
   emulated ranks, their (value, index) bests stacked as the gather) == uncached == 3b9818d's torch loop, bit for bit;
   plus forced ties across the halves (the lower index wins, as torch.argmax).

  python3 markov_check.py --model M [--tokens 4096] [--trials 300] [--full] --out F
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def old_loop(logits, head, emb, tokens, steps):
    """3b9818d BatchDraftGraph's loop: logits [N, n, V] fp32."""

    N = logits.shape[0]
    out = torch.empty((N, steps + 1), dtype=torch.long, device="cuda")
    out[:, 0] = tokens
    for i in range(steps):
        e = emb[out[:, i]]
        out[:, i + 1] = (logits[:, i] + (e.to(torch.float16) @ head.t()).float()).argmax(-1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--full", action="store_true", help="also every token's row against cuBLAS (N = 1 eager and in a "
                   "CUDA graph, N = 4 batches)")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import markov as MK
    from tensorfold.families.deepseek_v41.cuda.weights import Shards

    sh = Shards(a.model)
    head = sh.get("mtp.2.markov_head.head.weight").contiguous().cuda()
    emb = sh.get("mtp.2.markov_head.embed.weight").contiguous().cuda()
    V = head.shape[0]
    res = {}

    def mk(rank, split, rows, tokens):
        # a Markov over the real head / embedding as TP rank ``rank`` of 2 (the split steps' gather is driven by hand)
        w = SimpleNamespace(world=2, rank=rank, dspark=SimpleNamespace(markov_head=head, markov_embed=emb))
        model = SimpleNamespace(w=w, cfg=SimpleNamespace(vocab=V), comm=None)
        return MK.Markov(model, tokens=tokens, rows=rows, split=split)

    g = torch.Generator().manual_seed(3)
    try:
        from tensorfold.families.deepseek_v41.cuda.markov_tokens import TOKENS
        freq = list(TOKENS)
    except ImportError:
        freq = []
    toks = list(dict.fromkeys(freq[:a.tokens // 2] + torch.randint(0, V, (a.tokens,), generator=g).tolist()))[:a.tokens]
    t0 = time.time()
    full = mk(0, False, len(toks), toks)            # unsplit: every column (cache = the fill's rows)
    r0 = mk(0, True, len(toks), toks)
    r1 = mk(1, True, len(toks), toks)
    torch.cuda.synchronize()
    res["fill_s"] = round(time.time() - t0, 2)
    # 1a. halves == the whole
    Vl = V // 2
    res["halves_equal_full"] = bool(torch.equal(r0.cache, full.cache[:, :Vl]) and torch.equal(r1.cache, full.cache[:, Vl:]))
    # 1b. against cuBLAS at the served shapes
    tt = torch.tensor(toks, device="cuda")
    for N in (1, 2, 3, 4):
        diff_el = diff_rows = 0
        for r0_ in range(0, len(toks) - N + 1, N):
            e = emb[tt[r0_:r0_ + N]].to(torch.float16)
            ref = e @ head.t()                                # [N, V] fp16: 3b9818d's bias rows
            mine = full.cache[r0_:r0_ + N]
            d = (ref.view(torch.int16) != mine.view(torch.int16))
            diff_el += int(d.sum())
            diff_rows += int(d.any(1).sum())
        res[f"cublas_N{N}"] = {"rows": len(toks) // N * N, "rows_differ": diff_rows, "elements_differ": diff_el}
    # also the serial DraftGraph's shape (head @ e: a matrix-vector product)
    diff_el = diff_rows = 0
    for j in range(min(len(toks), 1024)):
        ref = head @ emb[tt[j]].to(torch.float16)
        d = ref.view(torch.int16) != full.cache[j].view(torch.int16)
        diff_el += int(d.sum())
        diff_rows += int(d.any())
    res["cublas_mv"] = {"rows": min(len(toks), 1024), "rows_differ": diff_rows, "elements_differ": diff_el}
    # 1c. a step's staging rows (no cache: every row computed by the step's own launch) at row counts 1..16 == the
    # fill's rows; and with the cache on, a step leaves the staging rows of rows with cached rows untouched
    nocache = mk(0, False, 0, None)
    live_ok = True
    for N in (1, 2, 3, 4, 5, 8, 16):
        for start in (0, 37, len(toks) - N):
            out = torch.zeros((N, 2), dtype=torch.long, device="cuda")
            out[:, 0] = tt[start:start + N]
            lg = torch.zeros((2, N, Vl), dtype=torch.float32, device="cuda")   # gathered layout, one row a stream
            stage = torch.empty((N, V), dtype=torch.float16, device="cuda")
            nocache.local_best(lg, out, 1, 0, stage=stage)
            live_ok &= bool(torch.equal(stage, full.cache[start:start + N]))
            stage.fill_(7.0)
            full.local_best(lg, out, 1, 0, stage=stage)
            live_ok &= bool((stage == 7.0).all())
    res["live_bias_equal_fill"] = live_ok
    # 2. drafts: realistic logits = a scaled real bias row of another token + noise (peaked, with near ties)
    def run_split(lgfull, tokens, steps):
        N = lgfull.shape[0]
        lg_loc = [lgfull[..., :Vl].reshape(N * lgfull.shape[1], Vl).contiguous(),
                  lgfull[..., Vl:].reshape(N * lgfull.shape[1], Vl).contiguous()]
        outs = []
        for rk in (r0, r1):
            o = torch.empty((N, steps + 1), dtype=torch.long, device="cuda")
            o[:, 0] = tokens
            outs.append(o)
        for i in range(steps):
            sends = [rk.local_best(lg_loc[j], outs[j], lgfull.shape[1], i).clone() for j, rk in enumerate((r0, r1))]
            gat = torch.stack(sends)                          # the gather: [world, N, 4], rank order
            for j, rk in enumerate((r0, r1)):
                rk.pick(gat, outs[j], i)
        assert torch.equal(outs[0], outs[1])
        return outs[0]

    def run_full(mm, lgfull, tokens, steps):
        N, n = lgfull.shape[:2]
        gathered = torch.stack([lgfull[..., :Vl].reshape(N * n, Vl), lgfull[..., Vl:].reshape(N * n, Vl)])
        o = torch.empty((N, steps + 1), dtype=torch.long, device="cuda")
        o[:, 0] = tokens
        mm.steps(gathered, o, n, steps)
        return o

    stats = {"trials": 0, "split_eq_full": 0, "cached_eq_uncached": 0, "eq_old": 0, "old_rows_differ": 0,
             "steps": 0, "hits": 0}
    gg = torch.Generator(device="cuda").manual_seed(9)
    headf = head.float()
    for t in range(a.trials):
        N = (1, 2, 3, 4)[t % 4]
        n = 5
        steps = 5 if N < 4 else 3
        tok0 = tt[torch.randint(0, len(toks), (N,), generator=g)]
        src = tt[torch.randint(0, len(toks), (N * n,), generator=g)]
        scale = float((0.5, 1.0, 2.0)[t % 3])
        lgfull = (emb[src].float() @ headf.t()) * scale
        lgfull = lgfull + torch.randn(lgfull.shape, generator=gg, device="cuda") * 0.5
        lgfull = lgfull.view(N, n, V).contiguous()
        a_split = run_split(lgfull, tok0, steps)
        a_full = run_full(full, lgfull, tok0, steps)
        a_nc = run_full(nocache, lgfull, tok0, steps)
        a_old = old_loop(lgfull, head, emb, tok0, steps)
        stats["trials"] += 1
        stats["split_eq_full"] += int(torch.equal(a_split, a_full))
        stats["cached_eq_uncached"] += int(torch.equal(a_full, a_nc))
        stats["eq_old"] += int(torch.equal(a_full, a_old))
        stats["old_rows_differ"] += int((a_full != a_old).any(1).sum())
        stats["steps"] += N * steps
        stats["hits"] += int((full.slot[a_full[:, :steps]] >= 0).sum())
    res["drafts"] = stats
    # 2b. forced ties across the halves: equal scores at v (rank 0) and Vl + v (rank 1) -> the lower index
    ties_ok = True
    for v in (0, 5, Vl - 1, 1234):
        N, n, steps = 2, 1, 1
        tok0 = tt[:N]
        lgfull = torch.full((N, n, V), -1e4, dtype=torch.float32, device="cuda")
        b = (emb[tok0].to(torch.float16) @ head.t()).float()   # make score(v) == score(Vl + v) exactly
        lgfull[:, 0, v] = 100.0 - b[:, v]
        lgfull[:, 0, Vl + v] = 100.0 - b[:, Vl + v]
        ok_scores = bool(torch.equal(lgfull[:, 0, v] + b[:, v], lgfull[:, 0, Vl + v] + b[:, Vl + v]))
        a_split = run_split(lgfull, tok0, steps)
        a_old = old_loop(lgfull, head, emb, tok0, steps)
        ties_ok &= ok_scores and bool((a_split[:, 1] == v).all()) and bool(torch.equal(a_split, a_old))
    res["ties_lower_index"] = ties_ok
    if a.full:
        # every token of the vocabulary: the step's computed bias row (_bias through the fill, chunks of 4096 tokens)
        # against 3b9818d's cuBLAS rows at N = 1 (eager, and replayed from a captured graph like the drafter's) and
        # in batches of 4 (the 4-stream graph's shape)
        t0 = time.time()
        one = torch.zeros((1,), dtype=torch.long, device="cuda")
        g1 = torch.cuda.CUDAGraph()
        e1 = emb[one].to(torch.float16)
        ref1 = e1 @ head.t()
        torch.cuda.synchronize()
        with torch.cuda.graph(g1):
            ref1g = emb[one].to(torch.float16) @ head.t()
        full_stats = {"tokens": 0, "n1_rows_differ": 0, "n1_graph_rows_differ": 0, "n4_rows_differ": 0}
        for c0 in range(0, V, 4096):
            chunk = list(range(c0, min(V, c0 + 4096)))
            mkc = mk(0, False, len(chunk), chunk)
            ct = torch.tensor(chunk, device="cuda")
            for j, tkn in enumerate(chunk):
                one.fill_(tkn)
                g1.replay()
                ref = emb[ct[j:j + 1]].to(torch.float16) @ head.t()
                full_stats["n1_rows_differ"] += int(not torch.equal(ref[0], mkc.cache[j]))
                full_stats["n1_graph_rows_differ"] += int(not torch.equal(ref1g[0], mkc.cache[j]))
            for j in range(0, len(chunk) - 3, 4):
                ref = emb[ct[j:j + 4]].to(torch.float16) @ head.t()
                full_stats["n4_rows_differ"] += int((ref.view(torch.int16) != mkc.cache[j:j + 4].view(torch.int16))
                                                    .any(1).sum())
            full_stats["tokens"] += len(chunk)
            del mkc
        full_stats["s"] = round(time.time() - t0, 1)
        res["full_vocab"] = full_stats
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
