"""Summaries of a ztrace.jsonl: launches by phase and kernel, the unresolved ones, and the distinct launches of the
kernels whose names contain any of the given substrings (in one phase).

  python3 ztrace_sum.py ZTRACE.jsonl [--phase forward] [--names group_count,rot_in_kernel] [--max 6]
"""
import argparse
import collections
import json
import re


def short(name: str) -> str:
    m = re.search(r"(\d+)([A-Za-z_][A-Za-z0-9_]*?kernel[A-Za-z0-9_]*)", name)
    return m.group(2) if m else name[:60]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--names", default="")
    ap.add_argument("--max", type=int, default=6)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.trace)]
    summ = [r["summary"] for r in rows if "summary" in r]
    rows = [r for r in rows if "name" in r]
    print("summary:", summ[-1] if summ else None, "distinct lines:", len(rows))
    by_phase = collections.Counter(r["phase"] for r in rows)
    print("phases:", dict(by_phase.most_common(40)))
    unres = collections.Counter((r["phase"], r["name"][:90]) for r in rows if r["params"].startswith("?"))
    print("unresolved (phase, name):", unres.most_common(12))
    if a.names:
        want = a.names.split(",")
        for w in want:
            hits = [r for r in rows if w in r["name"] and (a.phase is None or r["phase"] == a.phase)]
            print(f"== {w}: {len(hits)} distinct")
            for r in hits[:a.max]:
                print("  ", r["phase"], short(r["name"]), r["grid"], r["block"], r["smem"], r["pdl"], r["params"][:300])


if __name__ == "__main__":
    main()
