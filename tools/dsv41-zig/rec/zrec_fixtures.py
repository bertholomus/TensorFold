"""Host-side fixtures for the Zig port, from the served family's own code (no GPU): Engram's constants and hash rows.

  python zrec_fixtures.py MODEL_DIR TOKEN_MAP_JSON OUT_DIR

engram.json: the bucket primes, offsets, multipliers and pad as the engine builds them (ops.engram_primes,
engram_multipliers; numpy's generator, so the Zig side embeds the multipliers and checks the rest), and Engram.hashes
for fixed id sequences: a plain run from position 0, a continuation from 517, one with an image span (negative ids),
and short sequences at the start, each rank's columns.
"""

import hashlib
import json
import os
import sys

import numpy as np

from tensorfold.families.deepseek_v41.config import Cfg
from tensorfold.families.deepseek_v41.cuda.model import Engram
from tensorfold.families.deepseek_v41.ops import engram_multipliers, engram_primes


def hasher(cfg: Cfg, token_map: list[int]) -> Engram:
    """An Engram with only what hashes() reads (no files, no reader pools)."""

    e = object.__new__(Engram)
    primes = engram_primes(cfg)
    flat = [[p for per in layer for p in per] for layer in primes]
    e.cfg = cfg
    e.np_primes = np.array(flat, dtype=np.int64)
    e.np_offsets = np.array([np.cumsum([0, *f[:-1]]) for f in flat], dtype=np.int64)
    e.np_mult = engram_multipliers(cfg).numpy().astype(np.int64)
    e.np_map = np.asarray(token_map, dtype=np.int64)
    e.pad = int(token_map[cfg.engram_pad])
    return e


def main() -> None:
    model_dir, map_path, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    cfg = Cfg.read(model_dir)
    token_map = json.load(open(map_path))
    e = hasher(cfg, token_map)
    rng = np.random.default_rng(20261007)
    ids = rng.integers(0, cfg.vocab, size=1200).tolist()
    image = list(ids[:300])
    image[100:140] = [-1] * 40
    cases = []
    for name, seq, start, n in (("plain", ids, 0, 600), ("continued", ids, 517, 300), ("image", image, 0, 300),
                                ("first1", ids, 0, 1), ("first3", ids, 0, 3), ("at2", ids, 2, 5)):
        rows = e.hashes(seq, start, n)                                     # [n, L, cols]
        cases.append({"name": name, "ids": seq[:start + n], "start": start, "n": n, "rows": rows.reshape(-1).tolist()})
    doc = {"layers": cfg.engram_layers, "ngram": cfg.engram_ngram, "heads": cfg.engram_heads, "pad": e.pad,
           "primes": e.np_primes.tolist(), "offsets": e.np_offsets.tolist(), "multipliers": e.np_mult.tolist(),
           "token_map_sha256": hashlib.sha256(open(map_path, "rb").read()).hexdigest(), "token_map_len": len(token_map),
           "cases": cases}
    with open(os.path.join(out, "engram.json"), "w") as f:
        json.dump(doc, f)
    print(json.dumps({"engram_cases": len(cases), "multipliers": doc["multipliers"]}))


if __name__ == "__main__":
    main()
