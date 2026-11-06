#!/usr/bin/env python3
"""compare_hidden.py — step-3 parity compare (phase 2b4). CPU-runnable (fp64 math, no CUDA).

Pairs the two dump files (tf_dump_hidden.py, parity_exl3_side.py; layout (1+nL, R, S, H) fp32,
keys h/rows/ids): asserts the fixture streams are byte-identical, then per layer (residual stream
after each transformer layer — embeddings reported but NOT gated, the two sides own that table
read path independently) computes global cosine, per-token cosine mean/min, and rel-L2, exactly
the metrics loader-test/parity_L00_07.py reported.

Gate (BRINGUP-PLAN step 3): worst LAYER cosine >= 0.9995. Prints the table, writes
parity-summary.json next to --out, exit 0/1 = gate pass/fail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # ml4_fixture


def metrics(a, b) -> dict:
    """(R, S, H) tensors, fp64: global cos, per-token cos mean/min, rel-L2 ||a-b||/||b||."""

    import torch
    a = a.to(torch.float64).flatten(0, -2)
    b = b.to(torch.float64).flatten(0, -2)
    dot, na, nb = (a * b).sum(), a.norm(), b.norm()
    tc = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return {"cos": float(dot / (na * nb)), "tok_cos_mean": float(tc.mean()),
            "tok_cos_min": float(tc.min()), "rel_l2": float((a - b).norm() / nb)}


def compare(tf: dict, ex: dict, gate: float = 0.9995, pack=None) -> dict:
    """dict summary from the two loaded dump dicts; raises on stream/shape mismatch."""

    import torch
    assert torch.equal(tf["ids"], ex["ids"]), "fixture token streams differ between the dumps"
    h1, h2 = tf["h"], ex["h"]
    assert h1.shape == h2.shape and h1.ndim == 4, f"dump shapes {tuple(h1.shape)} vs {tuple(h2.shape)}"
    nL = h1.shape[0] - 1
    res = {"rows": tf["rows"], "seqlen": int(h1.shape[2]), "hidden": int(h1.shape[3]),
           "layers": [], "embeddings": metrics(h1[0], h2[0]), "gate": gate}
    for L in range(nL):
        res["layers"].append({"layer": L, **metrics(h1[L + 1], h2[L + 1])})
    res["worst_layer_cos"] = min(l["cos"] for l in res["layers"])
    res["pass"] = bool(res["worst_layer_cos"] >= gate)
    if pack is not None:
        res["pack"] = str(pack)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="/w/results/tf_hidden_L00_03.pt")
    ap.add_argument("--exl3", default="/w/results/exl3_hidden_L00_03.pt")
    ap.add_argument("--out", default="/w/results/parity-summary.json")
    ap.add_argument("--gate", type=float, default=0.9995)
    args = ap.parse_args()

    import torch
    from ml4_fixture import fixture_ids
    tf = torch.load(args.tf, map_location="cpu", weights_only=False)
    ex = torch.load(args.exl3, map_location="cpu", weights_only=False)
    if tf["pack"] == ex["pack"] and (Path(tf["pack"]) / "config.json").is_file():
        ref = fixture_ids(tf["pack"], rows=len(tf["rows"]), seqlen=int(tf["h"].shape[2]))
        assert torch.equal(tf["ids"], ref), "dump ids differ from the ml4_fixture re-derivation"
    res = compare(tf, ex, gate=args.gate, pack=tf["pack"])
    print(f"{'layer':>5} | {'cos':>9} | {'tok cos mean':>12} | {'tok cos min':>11} | {'rel-L2':>8}")
    print("-" * 55)
    for l in res["layers"]:
        print(f"{l['layer']:>5} | {l['cos']:>9.6f} | {l['tok_cos_mean']:>12.6f} "
              f"| {l['tok_cos_min']:>11.5f} | {l['rel_l2']:>8.5f}")
    e = res["embeddings"]
    print(f"embed: cos {e['cos']:.6f} (reported, not gated)")
    print(f"worst layer cos {res['worst_layer_cos']:.6f} vs gate {args.gate} -> "
          f"{'PASS' if res['pass'] else 'FAIL'}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[compare] wrote {args.out}", flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
