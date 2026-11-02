"""Compact launch fixtures for the Zig wrappers' conformance tests, from a recording's launch logs (both ranks).

  python3 launch_fixtures.py REC_DIR OUT.json

For every Triton function: each distinct launch (grid, scalar arguments, constexprs, and every tensor argument's dtype,
shape, strides and 16-byte alignment), with the phase it came from; constexprs come from the launched variant.
"""

import json
import sys
from collections import defaultdict

rec = sys.argv[1]
seen = defaultdict(dict)
for r in (0, 1):
    d = json.load(open(f"{rec}/rank{r}/launches.json"))
    kern = d["kernels"]
    for e in d["log"]:
        if e["kind"] != "triton":
            continue
        k = kern[e["hash"]]
        tensors, scalars = {}, {}
        for name, v in e["args"].items():
            if name in k["constexprs"]:
                continue
            if isinstance(v, dict) and "dtype" in v:
                tensors[name] = [v["dtype"], v["shape"], v["stride"], v["align16"]]
            else:
                scalars[name] = v
        case = {"grid": list(e["grid"]) + [1] * (3 - len(e["grid"])), "scalars": scalars,
                "consts": k["constexprs"], "tensors": tensors, "num_warps": k["metadata"]["num_warps"]
                if "metadata" in k else None, "phase": e["phase"], "rank": r}
        key = json.dumps([case["grid"], scalars, k["constexprs"], tensors], sort_keys=True)
        seen[k["name"]].setdefault(key, case)
out = {fn: list(cases.values()) for fn, cases in sorted(seen.items())}
json.dump(out, open(sys.argv[2], "w"), indent=0, sort_keys=True)
print(json.dumps({fn: len(c) for fn, c in out.items()}))
