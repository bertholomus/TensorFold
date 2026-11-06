"""CPU tests for the step-3 parity tooling (phase 2b4): tools/parity/{ml4_fixture,
tf_dump_hidden,compare_hidden}.py. The compare gate is exercised against synthetic dumps (identical
PASS, single-layer perturbation FAILs under 0.9995, mismatches raise) and the TF dump plumbing runs
the tiny fixture model end to end — no GPU, no real pack needed anywhere here.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
for p in (SRC, Path(__file__).resolve().parents[1] / "tools" / "parity", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch                                                     # noqa: E402

import ml4_fixture as FX                                         # noqa: E402
import compare_hidden as CH                                      # noqa: E402
from tf_dump_hidden import dump_hidden                           # noqa: E402
import test_ml4_model as TM                                      # noqa: E402  (gate_pair, new_cache, M)
from refs import tiny_model as TMOD                              # noqa: E402  (TinyConfig)


def _mini_pack(root: Path, vocab: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(f'{{"vocab_size": {vocab}}}')
    return root


def test_fixture_deterministic_and_bounded():
    """Same (seed, vocab, rows, seqlen) -> byte-identical ids across calls and processes (pure
    sha256 stream, no RNG library to drift); ids in range; the parity shape is 4x2048."""

    with tempfile.TemporaryDirectory() as td:
        pack = _mini_pack(Path(td) / "p", 204_800)
        a, b = FX.fixture_ids(pack), FX.fixture_ids(pack)
        assert torch.equal(a, b), "fixture drifted between two calls"
        assert a.shape == (FX.ROWS, FX.SEQLEN) and a.dtype == torch.int64
        assert int(a.min()) >= 0 and int(a.max()) < 204_800
        c = FX.fixture_ids(pack, rows=2, seqlen=8, seed=7)
        assert c.shape == (2, 8) and not torch.equal(c[0], a[0][:8]), "seed/shape must bind the stream"
    return {"shape": list(a.shape), "deterministic": 1}


def _synthetic(seed: int = 1105, layers: int = 4, rows: int = 2, seq: int = 16, hid: int = 32):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(layers + 1, rows, seq, hid, generator=g)
    ids = torch.randint(0, 1000, (rows, seq), generator=g, dtype=torch.int64)
    return h, ids


def test_compare_gate_on_synthetic_dumps():
    """Identical dumps: PASS, worst cos 1.0. A 20%-rel-L2 perturbation of ONE row of ONE layer must
    pull the worst-layer cosine under the 0.9995 gate (per-layer attribution must catch it)."""

    h, ids = _synthetic()
    tf = {"h": h, "rows": [0, 1], "ids": ids, "pack": "p"}
    res = CH.compare(tf, {"h": h.clone(), "rows": [0, 1], "ids": ids.clone(), "pack": "p"})
    assert res["pass"] and abs(res["worst_layer_cos"] - 1.0) < 1e-9, res["worst_layer_cos"]
    bad = h.clone()
    bad[3][1] += 0.2 * bad[3].std()                                # layer 2, row 1, 20% noise
    res = CH.compare(tf, {"h": bad, "rows": [0, 1], "ids": ids.clone(), "pack": "p"})
    assert not res["pass"] and res["worst_layer_cos"] < 0.9995, res["worst_layer_cos"]
    assert res["layers"][2]["cos"] == res["worst_layer_cos"], "worst layer must be the perturbed one"
    for mut, why in (({"h": h[:, :1], "rows": [0], "ids": ids[:1]}, "shape"),
                     ({"h": h, "rows": [0, 1], "ids": ids + 1}, "ids")):
        caught = False
        try:
            CH.compare(tf, {**mut, "pack": "p"})
        except AssertionError as e:
            assert "differ" in str(e) or "shapes" in str(e), f"wrong error for {why}: {e}"
            caught = True
        assert caught, f"mismatch not caught: {why}"
    return {"identical_worst_cos": 1.0, "perturbed_worst_cos": round(res["worst_layer_cos"], 6)}


def test_tf_dump_plumbing_cpu():
    """dump_hidden() on the tiny fixture model: (1+nL, R, S, H) fp32, h[0] = the embedding rows,
    every layer state finite and matching a manual per-layer collect."""

    tc = TMOD.TinyConfig()
    ref, mine, _ = TM.gate_pair(tc)
    with tempfile.TemporaryDirectory() as td:
        pack = _mini_pack(Path(td) / "p", tc.vocab)
        ids = FX.fixture_ids(pack, rows=2, seqlen=5, seed=FX.SEED)
        h = dump_hidden(mine, ids, torch.device("cpu"))
        nL, R, S = mine.w.cfg.n_layers, ids.shape[0], ids.shape[1]
        assert h.shape == (nL + 1, R, S, mine.w.cfg.dim) and torch.isfinite(h).all()
        assert torch.equal(h[0][0], mine.w.embed[ids[0]].float()), "h[0] must be the embedding rows"
        collect = []
        with torch.inference_mode():
            mine.forward(ids[1:2], TM.M.new_cache(mine.w.cfg), collect=collect)
        assert torch.equal(h[-1][1], collect[-1][0].float()), "last dump row must equal the collect tail"
    return {"shape": list(h.shape), "embed_and_layers": 1}


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        try:
            r = t()
            print(f"PASS {t.__name__}: {r}")
        except Exception:
            fails += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed")
    sys.exit(1 if fails else 0)
