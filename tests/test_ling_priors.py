"""tools/ling_memory_model.py and tools/ling_decode_model.py, the Ling priors node results are judged against: the
recipe's weights a rank, admission run as the engine runs it, and the decode model's measured-cost bookkeeping.

Torch-free. Run: python -m pytest tests/test_ling_priors.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorfold.cuda.capacity import GIB, Weights  # noqa: E402


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # dataclasses look their module up while the class is made
    spec.loader.exec_module(mod)
    return mod


mem = _tool("ling_memory_model")
dec = _tool("ling_decode_model")


def test_recipe_weights_a_rank():
    text = mem.ling_text()
    w2, g2 = mem.prior_weights(text, 2, 0, mtp=True)
    # experts 102.68 GB + their suh/svh 0.99 + the MTP layer's experts 2.03 + the rest 5.83 (~111 GB)
    assert 111.4e9 < w2.resident < 111.7e9
    # the 4-bit MTP layer is the largest group: it sets the staging admission adds
    assert max(g2, key=g2.get) == "layers.72" and w2.staging == 3 * g2["layers.72"]
    w2n, g2n = mem.prior_weights(text, 2, 0, mtp=False)
    assert "layers.72" not in g2n and w2.resident - w2n.resident == g2["layers.72"]
    w4, _ = mem.prior_weights(text, 4, 0, mtp=True)
    w4b, _ = mem.prior_weights(text, 4, 2, mtp=True)
    assert w4b.resident < w4.resident < 0.62 * w2.resident        # rank 0 holds 384 of 1280 expert columns


def test_admission_as_the_engine_runs_it():
    text = mem.ling_text()
    w, _ = mem.prior_weights(text, 2, 0, mtp=True)
    avail = int(116.9 * GIB)
    # the default reserve (a tenth of the memory): weights and staging pass the budget
    assert mem.admitted(text, 2, 0, True, w, avail - 12 * GIB, 262144, None) == 0
    window = mem.admitted(text, 2, 0, True, w, avail - 7 * GIB, 262144, None)
    assert 200_000 < window < 262_144
    geo = mem.memory.geometry(text, 2, 0, mtp=True, rows=mem.ROWS)
    assert w.resident + geo.needed(window) <= avail - 7 * GIB < w.resident + geo.needed(window + 1)
    # TP4 rank 0: the whole native window at the default reserve
    w4, _ = mem.prior_weights(text, 4, 0, mtp=True)
    assert mem.admitted(text, 4, 0, True, w4, avail - 12 * GIB, 262144, None) == 262144
    assert mem.admitted(text, 2, 0, True, Weights(w.resident, 0), avail - 7 * GIB, 262144, None) == window


def test_measured_costs_are_kept_and_the_walk_is_taken_from_them():
    verify = [40.0, 50.6, 61.1, 71.5, 81.7, 91.9, 101.9, 111.8]
    short = [37.9, 46.5, 55.0, 63.3, 71.5, 79.6, 87.6, 95.5]
    slots = 229930
    m = dec.Model(2, 205.0, "birthday", verify_ms=verify, slots=slots)
    assert [m.forward_ms(R) for R in range(1, 9)] == verify          # the calibration's own depth and walk
    ab = dec.Model(2, 205.0, "birthday", verify_ms=verify, slots=slots, short_ms=short)
    walked = dec.chunks_for(slots) - dec.chunks_for(dec.SHORT_SLOTS)
    for R in range(1, 9):
        assert abs(ab.per_chunk[R - 1] * walked - (verify[R - 1] - short[R - 1])) < 1e-9
        fixed = dec.Model(2, 205.0, "birthday", verify_ms=verify, slots=slots, short_ms=short, merge="filled")
        want = verify[R - 1] - ab.per_chunk[R - 1] * (dec.chunks_for(slots) - dec.chunks_for(dec.CAL_DEPTH + R))
        assert abs(fixed.forward_ms(R) - want) < 1e-9


def test_depth_and_walk_terms():
    for world in (2, 4):
        for depth in (4096, 131072, 229922):
            today = dec.Model(world, 205.0, "birthday", slots=229930, depth=depth)
            fixed = dec.Model(world, 205.0, "birthday", slots=229930, depth=depth, merge="filled")
            for R in (1, 4, 8):
                assert fixed.forward_ms(R) <= today.forward_ms(R)
        shallow = dec.Model(world, 205.0, "birthday", slots=229930, depth=4096)
        deep = dec.Model(world, 205.0, "birthday", slots=229930, depth=131072)
        # the latent cache is replicated: TP4 reads as much of it as TP2 a token
        read = deep.forward_ms(1) - shallow.forward_ms(1)
        assert abs(read - 9 * (131072 - 4096) * 1152 / deep.bw) < 1e-6 * read
    assert abs(dec.Model(2, 205.0, "birthday").kda_ms(1) - 2 * dec.Model(4, 205.0, "birthday").kda_ms(1)) < 1e-12
    assert dec.chunks_for(262144 + 8) == 513
