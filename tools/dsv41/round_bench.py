"""A concurrent round's forward on TP ranks (same command on each): 16 rows as 4 streams x 4 rows on 4 slots against
16 rows of one stream, graph replays, Engram host time split out (rank 0 prints).

  python3 round_bench.py --rank R --master <HEAD_IP> --model M [--reps 20]
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
    p.add_argument("--port", type=int, default=29669)
    p.add_argument("--model", required=True)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out")
    a = p.parse_args()
    from pathlib import Path

    from tensorfold.families.deepseek_v41.cuda import engine as E

    E.WARM = False
    eng = E.DsEngine(Path(a.model), rank=a.rank, world=a.world, master=a.master, port=a.port, drafts=3,
                     context=8192, parallel=4)
    mu, m = eng.multi, eng.model
    g = torch.Generator().manual_seed(1)
    P = 300
    hosts = []
    for s in range(4):
        prompt = torch.randint(1000, 100000, (P,), generator=g).tolist()
        eng.prefill(mu.slots[s].sc, None, prompt)
        hosts.append(list(prompt))
    torch.cuda.synchronize()
    out = {}

    def bench(name, windows_fn):
        times, eng_t = [], []
        for i in range(a.reps + 3):
            wins = windows_fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hs = [m.engram.hashes(h, p0, len(t)) for _, p0, t, h in wins]
            t1 = time.perf_counter()
            mu.runner.forward(wins)
            torch.cuda.synchronize()
            if i >= 3:
                times.append(1000 * (time.perf_counter() - t0))
                eng_t.append(1000 * (t1 - t0))
        out[name] = {"ms_median": round(statistics.median(times), 2), "hash_ms": round(statistics.median(eng_t), 2)}
        if a.rank == 0:
            print(json.dumps({name: out[name]}), flush=True)

    def four():
        return [(s, P, torch.randint(1000, 100000, (4,), generator=g).tolist(), hosts[s] + [0] * 4) for s in range(4)]

    def one(rows):
        return lambda: [(0, P, torch.randint(1000, 100000, (rows,), generator=g).tolist(), hosts[0] + [0] * rows)]

    for name, fn in (("one_stream_16_rows", one(16)), ("four_streams_4_rows", four), ("one_stream_4_rows", one(4)),
                     ("one_stream_1_row", one(1))):
        bench(name, fn)
    # the same with the Engram rows read once ahead (the GPU part alone)
    import numpy as np

    wins = four()
    hs = np.concatenate([m.engram.hashes(h, p0, len(t)) for _, p0, t, h in wins], 0)
    lo, hi = m.engram.cols
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.reps):
        rows = {i: m.engram.rows(i, hs[:, m.cfg.engram_layers.index(i), lo:hi]) for i in m.cfg.engram_layers}
        torch.cuda.synchronize()
    out["engram_rows_16_ms"] = round(1000 * (time.perf_counter() - t0) / a.reps, 2)
    if a.rank == 0:
        print(json.dumps({"engram_rows_16_ms": out["engram_rows_16_ms"]}), flush=True)
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
