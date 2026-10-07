"""ops.HostIds (a sequence's host ids as a growable int32 buffer) against the Python list it replaces: the same ids
after random runs of the engine's edits (set at a position <= the length, truncate, copy to a top, slices), and the
same Engram row ids from model.Engram.hashes (its source, run on the CPU with a random hasher state) on both, with image
spans (negative ids) and positions at the sequence's start and at buffer growth boundaries.

  python3 tools/dsv41/host_ids_test.py          (CPU: numpy + torch; no GPU, no Triton)
Exits 1 when a check fails.
"""

import ast
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tensorfold.families.deepseek_v41.ops import HostIds  # noqa: E402


def engram_hashes():
    """model.Engram.hashes, compiled from model.py's source (model.py itself imports Triton kernels)."""

    tree = ast.parse((ROOT / "src/tensorfold/families/deepseek_v41/cuda/model.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Engram":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "hashes":
                    f.returns = None
                    for a in f.args.args:
                        a.annotation = None
                    ns = {"np": np}
                    exec(compile(ast.Module(body=[f], type_ignores=[]), "model.py", "exec"), ns)
                    return ns["hashes"]
    raise LookupError("Engram.hashes")


def fake_engram(rng):
    ngram, heads, layers, vocab = 3, 4, 2, 5000
    cfg = types.SimpleNamespace(engram_ngram=ngram, engram_heads=heads)
    return types.SimpleNamespace(
        cfg=cfg, np_map=rng.integers(0, 3000, vocab).astype(np.int64), pad=2999,
        np_mult=rng.integers(1, 2**20, (layers, ngram)).astype(np.int64),
        np_primes=rng.choice([10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079], (layers, (ngram - 1) * heads)),
        np_offsets=rng.integers(0, 10**6, (layers, (ngram - 1) * heads)).astype(np.int64))


def ids_with_images(rng, n):
    a = rng.integers(0, 5000, n)
    for _ in range(rng.integers(0, 3)):
        s = int(rng.integers(0, max(1, n - 1)))
        a[s:s + int(rng.integers(1, 40))] = -1
    return a.tolist()


def main() -> int:
    rng = np.random.default_rng(5)
    hashes = engram_hashes()
    eng = fake_engram(rng)
    checks = {"ops_equal": True, "hashes_equal": True, "copies_independent": True}
    n_ops = n_hash = 0
    for trial in range(60):
        ref: list[int] = []
        got = HostIds()
        kept: list[tuple[list[int], HostIds]] = []
        for step in range(40):
            op = rng.integers(0, 10)
            if op < 6 or not ref:                     # set at a position <= length: a prompt chunk or a round's window
                start = int(rng.integers(0, len(ref) + 1)) if rng.random() < 0.3 else len(ref)
                if rng.random() < 0.1:
                    start = 0
                k = int(rng.choice([1, 2, 5, 17, 300, 1023, 1024, 1025, 2048, 5000]))
                ids = ids_with_images(rng, k)
                del ref[start:]
                ref.extend(int(t) for t in ids)
                got.set(start, ids if rng.random() < 0.5 else np.asarray(ids))
                n = len(ids)
                if n and start + n <= len(ref):
                    h_ref = hashes(eng, ref, start, n)
                    h_got = hashes(eng, got.view(), start, n)
                    checks["hashes_equal"] &= bool(np.array_equal(h_ref, h_got) and h_ref.dtype == h_got.dtype)
                    n_hash += 1
            elif op < 7:                              # truncate (engine.prefill_steps)
                start = int(rng.integers(0, len(ref) + 1))
                del ref[start:]
                got.truncate(start)
            elif op < 8:                              # a kept prompt's copy to its top, then a slot resumed from it
                top = int(rng.integers(0, len(ref) + 1))
                kept.append((list(ref[:top]), got.copy(top)))
                if rng.random() < 0.5:
                    lr, lg = kept[int(rng.integers(0, len(kept)))]
                    cut = int(rng.integers(0, len(lr) + 1))
                    ref, got = list(lr[:cut]), lg.copy(cut)
            else:                                     # engram_touch's tail
                L = len(ref)
                a = max(0, L - 8)
                tail_ref = ref[a:L] + [7]
                tail_got = [*got[a:L].tolist(), 7]
                checks["ops_equal"] &= tail_ref == tail_got
                if tail_ref:
                    checks["hashes_equal"] &= bool(np.array_equal(hashes(eng, tail_ref, len(tail_ref) - 1, 1),
                                                                  hashes(eng, tail_got, len(tail_got) - 1, 1)))
                    n_hash += 1
            checks["ops_equal"] &= len(got) == len(ref) and got.view().tolist() == ref
            n_ops += 1
        for lr, lg in kept:                           # later edits of the live ids never reach a kept copy
            checks["copies_independent"] &= lg.view().tolist() == lr
    res = {**checks, "ops": n_ops, "hash_checks": n_hash, "passed": all(checks.values())}
    print(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
