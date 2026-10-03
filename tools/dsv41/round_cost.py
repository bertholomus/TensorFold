"""Cost of one graph-captured forward of R rows (R = 1 .. 16) at a context bucket, on TP ranks (same command on each),
plus a per-kernel CUDA-time breakdown of eager forwards at a few row counts (rank 0 prints and writes).

  python3 round_cost.py --rank R --master <HEAD_IP> --model M [--rows 1,2,3,4,5,6,8,12,16] [--out F]

Rows carry random token ids (independent picks, like rows of different streams); each replay changes them.
"""

import argparse
import json
import statistics
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--master", required=True)
    p.add_argument("--port", type=int, default=29667)
    p.add_argument("--model", required=True)
    p.add_argument("--engram")
    p.add_argument("--rows", default="1,2,3,4,5,6,8,10,12,16")
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--profile-rows", default="1,4,16")
    p.add_argument("--out")
    a = p.parse_args()
    from pathlib import Path

    from tensorfold.families.deepseek_v41.cuda import engine as E
    from tensorfold.families.deepseek_v41.cuda.graph import GraphRunner, StaticDecoder

    E.WARM = False
    rows = [int(r) for r in a.rows.split(",") if r.strip()]
    eng = E.DsEngine(Path(a.model), rank=a.rank, world=a.world, master=a.master, port=a.port, drafts=3,
                     context=8192, engram_dir=a.engram)
    m = eng.model
    sc = m.new_cache(8192 + 64)
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(1000, 100000, (a.prompt_len,), generator=g).tolist()
    m.forward(sc, torch.tensor(prompt, device="cuda"), 0, host_ids=prompt)
    torch.cuda.synchronize()
    runner = GraphRunner(m, max(rows or [1]))
    sc.host = prompt[:]
    out = {"rows": {}, "profile": {}}
    P = a.prompt_len
    for R in rows:
        taps = R > 1
        times = []
        for i in range(a.reps + 3):
            ids = torch.randint(1000, 100000, (R,), generator=g).tolist()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            lg, _ = runner.forward(sc, ids, P, taps)
            torch.cuda.synchronize()
            if i >= 3:                         # the first calls capture the graph
                times.append(1000 * (time.perf_counter() - t0))
        out["rows"][R] = {"ms_median": round(statistics.median(times), 2), "ms_min": round(min(times), 2)}
        if a.rank == 0:
            print(json.dumps({"R": R, **out["rows"][R]}), flush=True)
    # drafter graph
    if eng.drafter is not None and rows:
        dc = eng.drafter.new_cache()
        dg = eng._draft_graph(sc, dc, prompt[-1], P)
        times = []
        for i in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dg.run(prompt[-1], P)
            times.append(1000 * (time.perf_counter() - t0))
        out["draft_ms_median"] = round(statistics.median(times), 2)
        if a.rank == 0:
            print(json.dumps({"draft_ms_median": out["draft_ms_median"]}), flush=True)
    # per-kernel CUDA time of eager forwards (kernel path) at a few row counts
    from torch.profiler import ProfilerActivity, profile

    for R in [int(r) for r in a.profile_rows.split(",")]:
        ids = torch.randint(1000, 100000, (R,), generator=g)
        for _ in range(2):
            sc.length = P
            m.forward(sc, ids.cuda(), P, host_ids=ids.tolist(), all_logits=True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                sc.length = P
                m.forward(sc, ids.cuda(), P, host_ids=ids.tolist(), all_logits=True)
            torch.cuda.synchronize()
        rows_k = []
        for e in prof.key_averages():
            t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
            if t > 0:
                rows_k.append((e.key[:90], round(t / 3 / 1000, 3), e.count // 3))
        rows_k.sort(key=lambda r: -r[1])
        stages: dict = {}
        for name, ms, cnt in rows_k:
            st = stage_of(name)
            stages[st] = round(stages.get(st, 0.0) + ms, 3)
        out["profile"][R] = {"total_ms": round(sum(r[1] for r in rows_k), 2), "stages_ms": stages,
                             "kernels": sum(r[2] for r in rows_k), "top": rows_k[:30]}
        if a.rank == 0:
            print(json.dumps({"profile_R": R, "total_ms": out["profile"][R]["total_ms"],
                              "kernels": out["profile"][R]["kernels"], "stages_ms": stages}), flush=True)
            for r in rows_k[:30]:
                print(f"  {r[1]:8.3f} ms  x{r[2]:5d}  {r[0]}", flush=True)
    # synchronized section times of eager forwards (model.TIMING): the stage split with launch gaps included
    from tensorfold.families.deepseek_v41.cuda import model as M

    out["sections"] = {}
    for R in [int(r) for r in a.profile_rows.split(",")]:
        ids = torch.randint(1000, 100000, (R,), generator=g)
        M.TIMING, M.TIMES = True, {}
        for _ in range(5):
            sc.length = P
            m.forward(sc, ids.cuda(), P, host_ids=ids.tolist(), all_logits=True)
        M.TIMING = False
        out["sections"][R] = {k: round(1000 * v / 5, 3) for k, v in sorted(M.TIMES.items())}
        if a.rank == 0:
            print(json.dumps({"sections_R": R, **out["sections"][R]}), flush=True)
    if a.rank == 0 and a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def stage_of(name: str) -> str:
    n = name.lower()
    for key, stage in (("grouped", "routed+shared experts"), ("gateup", "routed+shared experts"),
                       ("down_combine", "routed+shared experts"), ("down_epilogue", "routed+shared experts"),
                       ("rot_in", "routed+shared experts"), ("group_", "routed+shared experts"),
                       ("_route", "MoE gate"), ("_rowmm", "router/indexer weights (rowmm)"),
                       ("linear", "EXL3 linears (attention, head, engram)"), ("exl3", "EXL3 linears (attention, head, engram)"),
                       ("_sparse_attn", "attention kernels"), ("_index_score", "indexer scores"),
                       ("_kv_norm_rope", "attention kernels"), ("_rope_heads", "attention kernels"),
                       ("topk", "top-k / sort (torch)"), ("sort", "top-k / sort (torch)"), ("radix", "top-k / sort (torch)"),
                       ("_hc_", "mHC"), ("rdma", "collectives"), ("nccl", "collectives"), ("gather_", "collectives"),
                       ("_engram_gate", "engram + norms"), ("_collapse", "engram + norms"), ("_rmsnorm", "engram + norms")):
        if key in n:
            return stage
    return "torch glue (copies, elementwise, indexing)"


if __name__ == "__main__":
    main()
