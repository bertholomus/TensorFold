"""Engram reads started ahead (lanes 1 and 2, several outstanding a layer) give the same rows as direct reads."""

import argparse

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    a = p.parse_args()
    from pathlib import Path

    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.engine import _default_engram
    from tensorfold.families.deepseek_v41.cuda.model import Engram
    from tensorfold.families.deepseek_v41.ops import compressed_token_map

    cfg = Cfg.read(a.model)
    tm, _ = compressed_token_map(Path(a.model) / "tokenizer.json")
    eng = Engram(_default_engram(a.model), cfg, tm, 0, 2)
    lo, hi = eng.cols

    def direct(i, idx):
        fw, bw, rw = eng.files[f"layers.{i}.engram.embed.weight"]
        fs, bs, rs = eng.files[f"layers.{i}.engram.embed.scale"]
        flat = np.ascontiguousarray(idx.reshape(-1), dtype=np.int64)
        m = flat.shape[0]
        ow = torch.empty((m, rw), dtype=torch.uint8)
        os_ = torch.empty((m, rs), dtype=torch.uint8)
        eng.io.gather_rows2(fw, bw, rw, fs, bs, rs, torch.from_numpy(flat), ow, os_, eng.threads)
        return eng._decode(ow.cuda(), os_.cuda(), m, rw, idx.shape[0])

    rng = np.random.default_rng(7)
    ok = n = hits = 0
    for trial in range(6):
        sets = []
        for k in range(3):                                  # three reads outstanding a layer, mixed lanes
            rows = int(rng.choice([1, 6, 16, 2048]))
            toks = rng.integers(1000, 100000, size=rows + 8).tolist()
            hs = eng.hashes(toks, 8, rows)
            sets.append(hs)
            for i in cfg.engram_layers:
                eng.prefetch(i, hs[:, cfg.engram_layers.index(i), lo:hi], lane=1 + (k % 2))
        for hs in sets:
            for i in cfg.engram_layers:
                idx = hs[:, cfg.engram_layers.index(i), lo:hi]
                before = len(eng.ahead)
                got = eng.rows(i, idx).clone()
                hits += int(len(eng.ahead) == before - 1)
                ok += int(torch.equal(got, direct(i, idx)))
                n += 1
    torch.cuda.synchronize()
    print({"equal": ok, "of": n, "hits": hits, "slots": {k: len(v) for k, v in eng.ring.items()}})


if __name__ == "__main__":
    main()
