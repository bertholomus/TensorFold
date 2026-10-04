"""The decode round, the base commit against this tree: small_bench.py run in separate processes, alternating (base,
new, base, new, ...), the same rows, windows, prompts and weights; per-forward graph-replay times and the logits / taps
of every row count compared bit for bit across the two trees.

  python3 small_ab.py --base /path/to/base/TensorFold --model M [--rounds 3] [--rows 1,2,6,16] --out F
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="a git archive of the base commit (its src/ goes first on PYTHONPATH)")
    p.add_argument("--model", required=True)
    p.add_argument("--rows", default="1,2,6,16")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--donors", default="0,20")
    p.add_argument("--workdir", default="/tmp/small_ab")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    bench = os.path.join(here, "small_bench.py")
    times = {"base": {}, "new": {}}
    for k in range(a.rounds):
        for side in ("base", "new"):
            env = dict(os.environ)
            if side == "base":
                env["PYTHONPATH"] = os.path.join(a.base, "src") + os.pathsep + env.get("PYTHONPATH", "")
            out = os.path.join(a.workdir, f"{side}{k}.json")
            cmd = [sys.executable, bench, "--model", a.model, "--layers", str(a.layers), "--donors", a.donors,
                   "--rows", a.rows, "--modes", "old" if side == "base" else "new", "--invariance", "0",
                   "--save-logits", os.path.join(a.workdir, f"{side}{k}.pt"), "--out", out]
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
            res = json.load(open(out))
            mode = "old" if side == "base" else "new"
            for R, r in res["rows"].items():
                times[side].setdefault(R, []).append(r[mode])
            print(json.dumps({side: {R: r[mode] for R, r in res["rows"].items()}}), flush=True)
    bits = {}
    ref = torch.load(os.path.join(a.workdir, "base0.pt"))
    for k in range(a.rounds):
        for side in ("base", "new"):
            got = torch.load(os.path.join(a.workdir, f"{side}{k}.pt"))
            for R in ref:
                ok = torch.equal(ref[R][0], got[R][0]) and (ref[R][1] is None or torch.equal(ref[R][1], got[R][1]))
                bits[f"{side}{k}_rows{R}"] = bool(ok)
    table = {}
    for R in times["base"]:
        b, n = statistics.median(times["base"][R]), statistics.median(times["new"][R])
        table[R] = {"base_ms": round(b, 3), "new_ms": round(n, 3), "saving_ms": round(b - n, 3),
                    "base_runs": times["base"][R], "new_runs": times["new"][R]}
    res = {"forward_ms": table, "logits_taps_equal_to_base": bits, "all_equal": all(bits.values())}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
