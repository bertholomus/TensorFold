"""The served Engram's rows for a recorded chunk, for the Zig port's Engram check: the served code's Engram.hashes and
Engram.rows (its readers and its GPU decode). Writes OUT.ids.i64 (this rank's columns' row ids [n, k]) and
OUT.rows.bf16 ([n, k * head_dim]).

  python zrec_engram_ref.py MODEL_DIR ENGRAM_DIR TOKEN_MAP IDS_BIN RANK WORLD LAYER OUT_PREFIX
"""

import json
import sys

import numpy as np
import torch

from tensorfold.families.deepseek_v41.config import Cfg
from tensorfold.families.deepseek_v41.cuda.model import Engram


def main() -> None:
    model, edir, tmap, ids_bin, rank, world, layer, out = sys.argv[1:9]
    rank, world, layer = int(rank), int(world), int(layer)
    cfg = Cfg.read(model)
    eng = Engram(edir, cfg, json.load(open(tmap)), rank, world)
    ids = np.fromfile(ids_bin, dtype=np.int64).tolist()
    n = len(ids)
    h = eng.hashes(ids, 0, n)
    lo, hi = eng.cols
    idx = np.ascontiguousarray(h[:, list(cfg.engram_layers).index(layer), lo:hi]).astype(np.int64)
    idx.tofile(out + ".ids.i64")
    rows = eng.rows(layer, idx)
    torch.cuda.synchronize()
    rows.contiguous().view(torch.uint8).cpu().numpy().tofile(out + ".rows.bf16")
    print(json.dumps({"rank": rank, "n": n, "cols": [lo, hi], "ids": list(idx.shape), "rows": list(rows.shape),
                      "dtype": str(rows.dtype)}))


if __name__ == "__main__":
    main()
