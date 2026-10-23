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


def jit_table() -> dict:
    """Every Triton JITFunction the engine's modules define: its arguments and the ones Triton never specializes
    (aot_pack.py's --jit input), by module-qualified name."""

    import importlib
    import os

    import tensorfold
    from triton.runtime.jit import JITFunction

    # pkgutil.walk_packages misses a subpackage without __init__.py (iter_modules reports only dirs with one) and
    # the family's cuda/ is one: its markov.py (the drafter's _bias, _score, ...) never reached the table and
    # aot_pack.py died on the first recorded launch there. Import the two prefixes' modules from disk instead.
    def _import_tree(base: str) -> None:
        for p in list(importlib.import_module(base).__path__):
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = sorted(d for d in dirnames if d != "__pycache__" and not d.startswith("."))
                for fn in sorted(filenames):
                    if fn.endswith(".py") and not fn.startswith("_"):
                        name = base + "." + os.path.relpath(os.path.join(dirpath, fn), p)[:-3].replace(os.sep, ".")
                        try:
                            importlib.import_module(name)
                        except Exception as e:
                            raise SystemExit(f"{name}: the jit table cannot import it ({e!r}) — aot_pack.py would "
                                             "miss its kernels; fix the import and rerun zrec_fixtures.py")

    for base in ("tensorfold.families.deepseek_v41", "tensorfold.cuda"):
        _import_tree(base)
    out = {}
    for mod in list(sys.modules.values()):
        if not getattr(mod, "__name__", "").startswith("tensorfold."):
            continue
        for v in vars(mod).values():
            if isinstance(v, JITFunction):
                names = list(v.arg_names)
                nospec = [names[i] if isinstance(i, int) else i for i in (v.do_not_specialize or [])]
                out[f"{v.fn.__module__}.{v.fn.__qualname__}"] = {"args": names, "do_not_specialize": nospec}
    return out


def rope_tables(cfg: Cfg, out: str, rows: int = 17 << 16) -> dict:
    """Both RoPE kinds' (cos, sin) fp32 tables as the engine builds them (ops.rope_cs on the CPU: torch's own pow and
    polar, whose last bits the Zig side cannot recompute), `rows` rows each, as raw little-endian files with digests."""

    import torch

    from tensorfold.families.deepseek_v41.ops import RopeTables, rope_cs

    rt = RopeTables(cfg, device="cpu")
    doc = {"rows": rows, "half": cfg.rope_dim // 2}
    for kind, compressed in (("plain", False), ("compressed", True)):
        cos, sin = rope_cs(cfg.rope_dim, rows, *rt.args(compressed), device="cpu")
        for part, t in (("cos", cos), ("sin", sin)):
            raw = t.contiguous().view(torch.uint8).numpy().tobytes()
            name = f"rope-{kind}-{part}.f32"
            with open(os.path.join(out, name), "wb") as f:
                f.write(raw)
            doc[name] = hashlib.sha256(raw).hexdigest()
    with open(os.path.join(out, "rope.json"), "w") as f:
        json.dump(doc, f, indent=1)
    return doc


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
    rope = rope_tables(cfg, out)
    jit = jit_table()
    with open(os.path.join(out, "jit.json"), "w") as f:
        json.dump(jit, f, indent=1)
    print(json.dumps({"engram_cases": len(cases), "multipliers": doc["multipliers"], "jit_functions": len(jit),
                      "rope_rows": rope["rows"]}))


if __name__ == "__main__":
    main()
