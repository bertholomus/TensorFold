"""Where an Engram row read's time goes (one rank, no model weights): hashes, the C++ gather, the copy and decode on
the GPU, per layer, at 1 / 6 / 16 rows, with fresh row ids (cold-ish) and with the same ids again (page-cache warm).

  python3 engram_profile.py --model M [--engram DIR] [--reps 50] --out F
"""

import argparse
import json
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--engram")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rows", default="1,6,16")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from pathlib import Path

    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.engine import _default_engram
    from tensorfold.families.deepseek_v41.cuda.model import Engram
    from tensorfold.families.deepseek_v41.ops import compressed_token_map

    cfg = Cfg.read(a.model)
    tm, _ = compressed_token_map(Path(a.model) / "tokenizer.json")
    eng = Engram(a.engram or _default_engram(a.model), cfg, tm, 0, 2)
    lo, hi = eng.cols
    rng = np.random.default_rng(a.seed)
    res = {}
    for rows in [int(x) for x in a.rows.split(",")]:
        for warm in (False, True):
            t_hash = t_gather = t_gpu = t_total = 0.0
            fixed = None
            for rep in range(a.reps):
                toks = (rng.integers(1000, 100000, size=rows + 64) if not warm or fixed is None else fixed).tolist()
                fixed = np.asarray(toks) if fixed is None else fixed
                t0 = time.perf_counter()
                hs = eng.hashes(toks, 64, rows)
                t1 = time.perf_counter()
                tg = tc = 0.0
                for i in cfg.engram_layers:
                    idx = hs[:, cfg.engram_layers.index(i), lo:hi]
                    fw, bw, rw = eng.files[f"layers.{i}.engram.embed.weight"]
                    fs, bs, rs = eng.files[f"layers.{i}.engram.embed.scale"]
                    flat = np.ascontiguousarray(idx.reshape(-1), dtype=np.int64)
                    m = flat.shape[0]
                    ow = torch.empty((m, rw), dtype=torch.uint8, pin_memory=True)
                    os_ = torch.empty((m, rs), dtype=torch.uint8, pin_memory=True)
                    g0 = time.perf_counter()
                    eng.io.gather_rows2(fw, bw, rw, fs, bs, rs, torch.from_numpy(flat), ow, os_, eng.threads)
                    g1 = time.perf_counter()
                    out = eng._decode(ow.cuda(non_blocking=True), os_.cuda(non_blocking=True), m, rw, idx.shape[0])
                    torch.cuda.synchronize()
                    tg += g1 - g0
                    tc += time.perf_counter() - g1
                t2 = time.perf_counter()
                if rep >= 3:
                    t_hash += t1 - t0
                    t_gather += tg
                    t_gpu += tc
                    t_total += t2 - t0
            n = a.reps - 3
            key = f"rows{rows}_{'warm' if warm else 'fresh'}"
            res[key] = {"hashes_ms": round(1000 * t_hash / n, 3), "gather_ms": round(1000 * t_gather / n, 3),
                        "copy_decode_ms": round(1000 * t_gpu / n, 3), "total_ms": round(1000 * t_total / n, 3)}
            print(json.dumps({key: res[key]}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
