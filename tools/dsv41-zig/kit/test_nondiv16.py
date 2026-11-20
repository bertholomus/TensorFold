#!/usr/bin/env python3
"""Checks the nondiv16-sm121 fragment and its cubins. Usage: python3 -B test_nondiv16.py [--kit DIR]"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAG = HERE / "nondiv16-sm121" / "aot-fragment.json"
CUBINS = FRAG.parent / "cubins"

EXPECTED = {
    "052d6c1ceb0c0134932d80c70ab723b84d5d13784d79962cbb47a4560c82c02d": {
        "fn": "_index_score",
        "cubin": "f11eef966275e317522a6ec3b441898acb1a7e102434f1215d5a333707a61edb",
        "size": 99880,
        "shared": 10240,
        "consts": {
            "IH": 32, "ID": 128, "BN": 64, "PACKED": 1, "HAS_BASE": 1, "PDL": 1,
            "HAS_CAND": 0, "CB": 8, "KEYS": 0, "HAS_TMAX": 0, "SKIP_DEAD": 1, "DEAD_PAD": 0,
        },
        "params": {
            "Q": "*bf16", "K": "*u8", "KS": "*u8", "Wt": "*bf16", "VIS": "*i64",
            "OUT": "*fp32", "n": "i32", "KB": "*i64", "CAND": "*i64", "cs": "i32",
            "TMAX": "*fp32", "nt": "i32",
        },
    },
    "9ab1ce45d139729e01bf8351bd87df01bed63459cd2c0d8072656fd178e6e88f": {
        "fn": "_score_keys",
        "cubin": "e9b6faf7cbb6b37736df75ddac645fdfb49ca52edc5c49ac55dc23e5027bd030",
        "size": 38904,
        "shared": 256,
        "consts": {"BN": 1024, "PDL": 1, "HAS_TMAX": 1, "TB": 64},
        "params": {"S": "*fp32", "OUT": "*i64", "n": "i32", "TMAX": "*i64", "nt": "i32"},
    },
}


def int_const(v: object) -> int:
    assert isinstance(v, dict) and isinstance(v.get("int"), int), v
    return v["int"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", help="an existing kit to check for hash collisions")
    a = ap.parse_args()

    frag = json.loads(FRAG.read_text())
    records = frag["kernels"]
    assert len(records) == 2, len(records)
    assert {k["hash"] for k in records} == set(EXPECTED)

    for k in records:
        want = EXPECTED[k["hash"]]
        assert k["fn"] == want["fn"], k
        assert k["name"] == want["fn"], k
        assert k["num_warps"] == 4 and k["num_ctas"] == 1, k
        assert k["shared"] == want["shared"], k
        assert k["global_scratch"] == 0 and k["profile_scratch"] == 0, k
        assert k["pdl"] is True, k
        assert {c: int_const(v) for c, v in k["consts"].items()} == want["consts"], k
        assert {p["name"]: p["type"] for p in k["params"]} == want["params"], k
        for p in k["params"]:
            if p["type"].startswith("*"):
                assert p["div16"] is True, (k["fn"], p)
        for p in k["params"]:
            if p["name"] in ("n", "nt"):
                assert p["div16"] is False, (k["fn"], p)
        cub = CUBINS / f"{k['hash']}.cubin"
        data = cub.read_bytes()
        assert len(data) == want["size"], (cub, len(data))
        assert hashlib.sha256(data).hexdigest() == want["cubin"], cub

    if a.kit:
        aot = json.loads((Path(a.kit) / "aot/aot.json").read_text())
        have = {k["hash"] for k in aot["kernels"]}
        assert not (have & set(EXPECTED)), sorted(have & set(EXPECTED))
        print(f"test_nondiv16: fragment ok; {len(aot['kernels'])} kit records, no collision")
    else:
        print("test_nondiv16: fragment and two cubins ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
