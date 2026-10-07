"""One recording's summary (both ranks): Triton kernels and launches by phase, extension calls by phase, ATen ops by
phase on CUDA and on the CPU, cuBLAS calls logged, and the extension modules (.so) loaded."""

import json
import os
import re
import sys
from collections import Counter, defaultdict

d = sys.argv[1]
out = {}
for r in (0, 1):
    base = os.path.join(d, f"rank{r}")
    rec = json.load(open(os.path.join(base, "launches.json")))
    kern = rec["kernels"]
    phases = defaultdict(lambda: Counter())
    for phase, counts in rec["phases"].items():
        for key, n in counts.items():
            kind = "triton" if key in kern else "ext"
            phases[phase][kind] += n
    aten = json.load(open(os.path.join(base, "aten.json")))
    by_phase = defaultdict(Counter)
    for row in aten["ops"]:
        dev = "cuda" if '"cuda"' in json.dumps(row["sig"]) else "cpu"     # any CUDA tensor argument
        by_phase[row["phase"]][(row["op"], dev)] += row["count"]
    cublas = Counter()
    for name in ("cublas.log", "cublaslt.log"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            for line in open(p, errors="replace"):
                m = re.search(r"(cublas\w+)\(", line)
                if m:
                    cublas[m.group(1)] += 1
    mods = json.load(open(os.path.join(base, "modules.json")))
    out[f"rank{r}"] = {
        "triton_kernels": len(kern),
        "triton_functions": sorted({k["function"] for k in kern.values()}),
        "launch_counts": {p: dict(c) for p, c in sorted(phases.items())},
        "aten_ops": {p: {f"{op} [{dev}]": n for (op, dev), n in sorted(c.items(), key=lambda kv: -kv[1])}
                     for p, c in sorted(by_phase.items())},
        "aten_folded": aten.get("folded", []),
        "cublas_calls": dict(cublas),
        "extension_modules": mods,
    }
json.dump(out, sys.stdout, indent=1)
