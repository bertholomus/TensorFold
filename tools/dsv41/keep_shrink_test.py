"""Kept-prompt shrinking (multi.py: MultiDecoder._shrink, switch TF_DS_KEEP_SHRINK) against the placement before it
(``--old``: a copy of multi.py without the switch): the window's extents stay a partition of the window, a request is
placed wherever the old code placed one, a kept prompt is cut back to its largest kept boundary that makes the room
instead of being forgotten, nothing changes when a request fits as it is, and every decision is a function of the
state (two runs on copies agree). The methods are compiled from multi.py's own source (multi.py imports Triton).

  python3 tools/dsv41/keep_shrink_test.py --old OLD_MULTI_PY        (CPU: numpy only)
Exits 1 when a check fails.
"""

import argparse
import ast
import copy
import json
import random
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tensorfold.families.deepseek_v41.ops import HostIds  # noqa: E402

C = 2048
METHODS = ("_marks", "_kept_marks", "_forget", "_room", "_place", "_shrink")


class _Engine(ast.NodeTransformer):
    """``from .engine import PREFILL_CHUNK as C`` -> ``C = 2048`` (engine.py imports CUDA code)."""

    def visit_ImportFrom(self, node):
        if node.module == "engine" and node.level == 1:
            return ast.parse(f"C = {C}").body[0]
        return node


def load(path):
    tree = _Engine().visit(ast.parse(Path(path).read_text()))
    ns = {"np": np, "KEEP_MARKS": 10, "os": __import__("os")}
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("Extents", "Kept"):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("ALIGN", "KEEP_SHRINK", "KEEP_SHRINK_MIN") for t in node.targets):
            keep.append(node)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MultiDecoder":
            funcs = [f for f in node.body if isinstance(f, ast.FunctionDef) and f.name in METHODS]
            for f in funcs:
                f.returns = None
                for a in f.args.args:
                    a.annotation = None
            cls = ast.ClassDef(name="Dec", bases=[], keywords=[], body=funcs, decorator_list=[])
            exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), ns)
    return ns


def fresh(ns, total):
    d = ns["Dec"]()
    d.extents = ns["Extents"](total)
    d.kept, d.keep_stats = {}, {}
    return d


def add_kept(ns, d, eid, n, tick, rng):
    """A finished prompt of n tokens kept at the first fit (as _keep leaves one); False when it does not fit."""

    marks = d._marks(n)
    snaps = {b: object() for b in marks}
    kept = d._kept_marks(snaps)
    top = kept[-1]
    size = d.extents.size(top)
    base = d.extents.take(size)
    if base is None:
        return False
    keys = np.asarray(rng.integers(0, 50000, top), dtype=np.int64)
    d.kept[eid] = ns["Kept"](eid, base, size, top, HostIds(keys), {b: snaps[b] for b in kept}, True, keys, tick)
    return True


def partition_ok(d, placed):
    spans = sorted([(a, b) for a, b in d.extents.gaps] + [(k.base, k.base + k.size) for k in d.kept.values()]
                   + placed)
    pos = 0
    for a, b in spans:
        if a != pos or b <= a:
            return False
        pos = b
    return pos == d.extents.total


def state(d):
    return (list(d.extents.gaps), sorted((k.eid, k.base, k.size, k.top, tuple(sorted(k.snaps))) for k in d.kept.values()))


def layout(d):
    """state without the boundary sets (the old code keeps fewer boundaries; the extents and tops are the same)"""

    g, ks = state(d)
    return g, [x[:4] for x in ks]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    a = p.parse_args()
    new = load(ROOT / "src/tensorfold/families/deepseek_v41/cuda/multi.py")
    old = load(a.old)
    assert new.get("KEEP_SHRINK") is True and "KEEP_SHRINK" not in old
    checks: dict = {}
    total = 1048576

    # 1. the case from the lane: one kept 1M prompt, a chat beside it (lane default reply 32,768)
    rng = np.random.default_rng(1)
    for need, want_top in ((20000 + 32768 + 28, 909312), (600000, 262144), (100000, 909312), (300000, 524288)):
        dn, do = fresh(new, total), fresh(old, total)
        add_kept(new, dn, 1, 1039836, 1, np.random.default_rng(1))
        add_kept(old, do, 1, 1039836, 1, np.random.default_rng(1))
        bn, how_n = dn._place(need, None)
        bo, how_o = do._place(need, None)
        k = dn.kept.get(1)
        checks[f"lane1M/{need}"] = (bn is not None and bo is not None and how_n == how_o == "fresh" and k is not None
                                    and k.top == want_top and len(k.keys) == want_top and len(k.host) == want_top
                                    and max(k.snaps) == want_top and not do.kept
                                    and partition_ok(dn, [(bn, bn + dn.extents.size(need))]))
    dn = fresh(new, total)
    add_kept(new, dn, 1, 1039836, 1, rng)
    bn, _ = dn._place(total - C, None)                   # the smallest boundary (2048) still makes this room
    checks["lane1M/cut_to_first_chunk"] = bn == C and dn.kept[1].top == C
    dn = fresh(new, total)
    add_kept(new, dn, 1, 1039836, 1, rng)
    bn, _ = dn._place(total - C + 1, None)               # no cut makes this room: forgotten, as before
    checks["lane1M/too_big_forgets"] = bn == 0 and not dn.kept

    # 2. random mixes: kept prompts at random sizes, then requests; old vs new decisions
    R = random.Random(7)
    same_when_fits = shrink_only = parts = det = True
    n_cases = n_shrunk = 0
    for trial in range(400):
        seed = R.randrange(1 << 30)
        dn, do = fresh(new, total), fresh(old, total)
        for i in range(R.randrange(1, 5)):
            n = R.choice([5000, 40000, 131072, 262144, 400000, 700000, 1039836])
            r1, r2 = np.random.default_rng(seed + i), np.random.default_rng(seed + i)
            ok_n = add_kept(new, dn, i, n, i, r1)
            ok_o = add_kept(old, do, i, n, i, r2)
            assert ok_n == ok_o
        for j in range(R.randrange(1, 4)):
            need = R.choice([3000, 40000, 60000, 200000, 500000, 900000])
            if not dn._room(need):
                continue
            n_cases += 1
            before_n, before_o = layout(dn), layout(do)
            fits = dn.extents.take(need) is not None
            if fits:                                       # undo the probe take
                dn.extents.gaps = before_n[0]
            twin = copy.deepcopy(dn)
            bn, how_n = dn._place(need, None)
            bt, how_t = twin._place(need, None)
            det &= (bn, how_n, state(dn)) == (bt, how_t, state(twin))
            if before_n == before_o:
                bo, how_o = do._place(need, None)
                if fits:
                    same_when_fits &= (bn, how_n, layout(dn)) == (bo, how_o, layout(do))
                else:
                    # the new placement forgets a subset of what the old one forgot, and keeps every kept prompt it
                    # did not forget at or below its old top
                    old_ids = {e for e, *_ in layout(do)[1]}
                    for e, base, size, top in layout(dn)[1]:
                        orig = {x[0]: x for x in before_n[1]}[e]
                        shrink_only &= top <= orig[3] and base == orig[1]
                        n_shrunk += top < orig[3]
                    shrink_only &= old_ids <= {e for e, *_ in layout(dn)[1]}
                    shrink_only &= bn == bo or bo is None or bn is not None
            parts &= bn is not None and partition_ok(dn, [(bn, bn + dn.extents.size(need))])
            if bn is not None:
                dn.extents.give(bn, need)
                if before_n == before_o and bo is not None:
                    do.extents.give(bo, need)
    checks.update({"random/same_when_it_fits": same_when_fits, "random/shrink_only": shrink_only,
                   "random/partition": parts, "random/deterministic": det})
    res = {**checks, "random_cases": n_cases, "random_shrinks": int(n_shrunk), "passed": all(checks.values())}
    print(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
