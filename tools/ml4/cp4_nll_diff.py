"""Two ``cp4_run.py --nll`` runs of the same tokens, row by row: the mean loss of each, the paired difference with its error,
and how often the top id flips and how close those rows were.

    python3 tools/ml4/cp4_nll_diff.py BASE.nll.json NEW.nll.json
"""
from __future__ import annotations

import json
import math
import sys


def main() -> None:
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    for part in ("rounds", "steps"):
        x = {int(r[0]): r for r in a[part]}
        y = {int(r[0]): r for r in b[part]}
        at = sorted(set(x) & set(y))[:-1]                        # the last row has no next id
        if not at:
            continue
        d = [y[p][1] - x[p][1] for p in at]
        n = len(d)
        mean = sum(d) / n
        err = math.sqrt(sum((v - mean) ** 2 for v in d) / max(n - 1, 1) / n)
        flips = [p for p in at if int(x[p][2]) != int(y[p][2])]
        gaps = sorted(x[p][3] for p in flips)
        print(f"{part}: {n} rows, loss {sum(x[p][1] for p in at) / n:.5f} -> {sum(y[p][1] for p in at) / n:.5f}, "
              f"delta {mean:+.5f} +- {2 * err:.5f} (2 s.e.), mean |delta| {sum(abs(v) for v in d) / n:.5f}, top id flips "
              f"{len(flips)} ({100 * len(flips) / n:.2f} %)" + (f", base gap at a flip: median {gaps[len(gaps) // 2]:.3f}, "
                                                              f"max {gaps[-1]:.3f}" if gaps else ""))


if __name__ == "__main__":
    main()
