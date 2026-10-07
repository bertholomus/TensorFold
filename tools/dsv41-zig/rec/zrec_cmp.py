"""Compare two digest lists (zrec_weights.py's and tf-dsv41-load's) by tensor name: equal, different, one side only."""

import json
import sys


def read(path):
    return {r["name"]: r for r in map(json.loads, open(path))}


a, b = read(sys.argv[1]), read(sys.argv[2])
same = [n for n in a if n in b and a[n]["sha256"] == b[n]["sha256"] and a[n]["bytes"] == b[n]["bytes"]]
diff = [n for n in a if n in b and n not in same]
only_a = [n for n in a if n not in b]
only_b = [n for n in b if n not in a and ".ffn.experts.ptr" not in n]
print(json.dumps({"equal": len(same), "different": len(diff), "only_first": len(only_a), "only_second": len(only_b),
                  "equal_bytes": sum(a[n]["bytes"] for n in same), "first_diff": diff[:10], "first_only_first": only_a[:10],
                  "first_only_second": only_b[:10], "ok": not diff and not only_a and not only_b}))
sys.exit(0 if not diff and not only_a and not only_b else 1)
