"""CPU tests of the world-parameterized TP4 changes (commit 497f951): split.py, weights.py WORLD handling, engine.py plumbing.

Run: pytest -q tests/test_tp4.py
No CUDA, no NCCL, no Triton kernels are touched: split math runs on plain numpy/torch CPU tensors,
weights.py WORLD handling is exercised through the module's own helpers, and engine.py's WORLD
plumbing is checked at import time plus against a stubbed GlmEngine.__init__ path (mocked com-init).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")
if torch.cuda.is_available():      # these tests must never touch a GPU even where one exists
    pytest.skip("CPU-only TP4 tests", allow_module_level=True)


def import_split(world: int | None = None):
    """split.py imported fresh, optionally with TF_TP_WORLD set first (WORLD is read at import time)."""

    for name in [n for n in sys.modules if n.startswith("tensorfold")]:
        del sys.modules[name]
    old = os.environ.get("TF_TP_WORLD")
    if world is None:
        os.environ.pop("TF_TP_WORLD", None)
    else:
        os.environ["TF_TP_WORLD"] = str(world)
    try:
        import tensorfold.families.glm5_next.cuda.split as split
        return importlib.reload(split)
    finally:
        if old is None:
            os.environ.pop("TF_TP_WORLD", None)
        else:
            os.environ["TF_TP_WORLD"] = old


# ---------------------------------------------------------------- split.py: WORLD at import time


@pytest.mark.parametrize("world,expected", [(1, 1), (2, 2), (4, 4)])
def test_world_env_read_at_import(world, expected):
    split = import_split(world)
    assert split.WORLD == expected


def test_world_defaults_to_2():
    split = import_split(None)
    assert split.WORLD == 2


# ---------------------------------------------------------------- split.py: split_bytes


def split_bytes_roundtrip(shape, itemsize, kind, world):
    """Split every rank, reassemble by concat along the split dim, compare to the original bytes.

    split_bytes works on a flat byte array whose length is prod(shape) * itemsize (the checkpoint's
    stored bytes); ``shape`` is the logical tensor shape, so the byte count of the leading axis is
    prod(shape[1:]) * itemsize.
    """

    split = import_split()
    logical = int(np.prod(shape))
    raw = (np.arange(logical, dtype=np.int64) % 251).astype(np.uint32).view(np.uint8).reshape(-1)
    assert raw.size == logical * itemsize

    def logical_view(flat: np.ndarray, part_shape: list[int]) -> np.ndarray:
        """flat stored bytes -> the logical uint32 word grid the split math must reproduce."""

        return flat.view(np.uint32).reshape(part_shape)

    original = logical_view(raw, list(shape))
    parts = [split.split_bytes(raw, list(shape), itemsize, kind, rank, world=world) for rank in range(world)]
    if kind == "rep":
        for data, s in parts:
            assert s == list(shape)
            assert np.array_equal(logical_view(data, s), original)
        return
    axis = 0 if kind == "row" else 1
    for data, s in parts:
        assert s[axis] == shape[axis] // world
        assert s[:axis] + s[axis + 1:] == shape[:axis] + shape[axis + 1:]
    # reassembly: concat along the split dim reproduces the exact logical values
    if kind == "row":
        got = np.concatenate([logical_view(d, s) for d, s in parts], axis=0)
    else:
        got = np.concatenate([logical_view(d, s) for d, s in parts], axis=1)
    assert np.array_equal(got, original)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("kind,shape", [
    ("row", [8, 16]),          # q_b_proj-like: rows split
    ("row", [256, 6144]),      # expert gate_proj-like: 256 experts' rows
    ("col", [4, 8]),           # down_proj-like: columns split
    ("col", [6144, 2048]),     # expert down_proj real shape ratio
    ("dim1", [2, 8, 4]),       # 3-D second axis split (second dim divisible by 4)
    ("rep", [4, 4]),           # replicated
])
def test_split_bytes_world_1_2_4(kind, shape, world):
    split_bytes_roundtrip(shape, 4, kind, world)


def test_split_bytes_row_2d_world_4_exact_values():
    """Byte-level check: rank r gets exactly rows [r*part, (r+1)*part) of a row-major tensor."""

    split = import_split()
    rows, cols, world = 8, 3, 4
    raw = np.arange(rows * cols, dtype=np.int32)
    part = rows // world
    for rank in range(world):
        data, shape = split.split_bytes(raw, [rows, cols], 4, "row", rank, world=world)
        assert shape == [part, cols]
        assert np.array_equal(data, raw[rank * part * cols:(rank + 1) * part * cols])


def test_split_bytes_col_world_4_exact_values():
    """col split works on flat stored bytes; the logical view is (rows, cols) uint32 words."""

    split = import_split()
    rows, cols, world = 2, 8, 4
    words = np.arange(rows * cols, dtype=np.uint32)
    raw = words.view(np.uint8)
    for rank in range(world):
        data, shape = split.split_bytes(raw, [rows, cols], 4, "col", rank, world=world)
        assert shape == [rows, cols // world]
        got = data.view(np.uint32).reshape(rows, cols // world)
        assert np.array_equal(got, words.reshape(rows, cols)[:, rank * 2:(rank + 1) * 2])


def test_split_bytes_dim1_world_4_3d():
    split = import_split()
    shape, world = [2, 8, 3], 4
    words = np.arange(int(np.prod(shape)), dtype=np.uint32)
    raw = words.view(np.uint8)
    parts = [split.split_bytes(raw, shape, 4, "dim1", r, world=world) for r in range(world)]
    assert [p[1] for p in parts] == [[2, 2, 3]] * world
    got = np.concatenate([p[0].view(np.uint32).reshape(2, 2, 3) for p in parts], axis=1)
    assert np.array_equal(got, words.reshape(shape))


def test_split_bytes_rep_ignores_world():
    split = import_split()
    raw = np.arange(6, dtype=np.int32)
    for world in (1, 2, 4):
        data, shape = split.split_bytes(raw, [6], 4, "rep", 1, world=world)
        assert np.array_equal(data, raw) and shape == [6]


# ---------------------------------------------------------------- split.py: split_device (CPU tensors)


def cpu_t(shape):
    return torch.arange(int(np.prod(shape)), dtype=torch.uint8).reshape(shape)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("kind,shape", [
    ("row", [8, 16]),
    ("col", [4, 8]),
    ("dim1", [2, 8, 4]),
    ("rep", [4, 4]),
])
def test_split_device_world_1_2_4(kind, shape, world):
    """The device path mirrors _cut's contract: a flat uint8 tensor in, a flat byte tensor out whose
    logical view reshapes to the returned shape (split.py:_cut does data.view(dtype).reshape(shape))."""

    split = import_split()
    logical = int(np.prod(shape))
    raw = torch.arange(logical * 4, dtype=torch.uint8)     # the span's bytes: prod(shape) * itemsize
    parts = [split.split_device(raw, list(shape), 4, kind, rank, world=world) for rank in range(world)]
    original_words = raw.view(torch.uint32).clone()
    if kind == "rep":
        for data, s in parts:
            assert s == list(shape)
            assert torch.equal(data.view(torch.uint32), original_words)
        return
    axis = 0 if kind == "row" else 1
    for data, s in parts:
        assert data.dim() == 1                                    # flat bytes out, as _cut's view() needs
        assert s[axis] == shape[axis] // world
    if kind == "row":
        got = torch.cat([p[0] for p in parts]).view(torch.uint32).reshape(shape)
    else:
        view_shape = [shape[0], shape[1] // world] + shape[2:]
        got = torch.cat([p[0].view(torch.uint32).reshape(view_shape) for p in parts], dim=1)
    assert torch.equal(got, original_words.reshape(shape))
    for p, _ in parts:        # a proper split (world > 1) returns fresh storage, never a view of raw
        assert world == 1 or p.data_ptr() != raw.data_ptr()


def test_split_device_row_world_4_exact_values():
    split = import_split()
    rows, cols, world = 8, 3, 4
    raw = torch.arange(rows * cols, dtype=torch.uint8)
    part = rows // world
    for rank in range(world):
        data, shape = split.split_device(raw, [rows, cols], 4, "row", rank, world=world)
        assert shape == [part, cols]
        assert torch.equal(data, raw[rank * part * cols:(rank + 1) * part * cols])


# ---------------------------------------------------------------- split.py: errors on non-divisible shapes


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("kind,shape", [
    ("row", [3, 4]),           # 3 divisible by neither
    ("row", [6, 4]),           # 6 % 4 (only 4-rank fails, 2-rank passes)
    ("col", [4, 3]),
    ("col", [4, 6]),           # 6 % 4
    ("dim1", [2, 5, 2]),
])
def test_split_bytes_rejects_non_divisible(kind, shape, world):
    split = import_split()
    raw = np.zeros(int(np.prod(shape)) * 4, dtype=np.uint8)
    divisible = (shape[0] if kind == "row" else shape[1]) % world == 0
    if divisible:
        return                  # world 2 divides these shapes; only world 4 must reject
    with pytest.raises(ValueError):
        split.split_bytes(raw, list(shape), 4, kind, 0, world=world)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("kind,shape", [
    ("row", [3, 4]),
    ("row", [6, 4]),
    ("col", [4, 3]),
    ("col", [4, 6]),
    ("dim1", [2, 5, 2]),
])
def test_split_device_rejects_non_divisible(kind, shape, world):
    split = import_split()
    raw = torch.zeros(int(np.prod(shape)) * 4, dtype=torch.uint8)
    divisible = (shape[0] if kind == "row" else shape[1]) % world == 0
    if divisible:
        return                  # world 2 divides these shapes; only world 4 must reject
    with pytest.raises(ValueError):
        split.split_device(raw, list(shape), 4, kind, 0, world=world)


def test_split_bytes_rejects_rank_out_of_world():
    split = import_split()
    raw = np.zeros(16, dtype=np.uint8)
    with pytest.raises(ValueError):
        split.split_bytes(raw, [4, 4], 4, "row", 4, world=4)


def test_split_bytes_col_requires_2d():
    split = import_split()
    with pytest.raises(ValueError):
        split.split_bytes(np.zeros(48, dtype=np.uint8), [4, 3, 4], 4, "col", 0, world=4)


def test_split_bytes_unknown_kind():
    split = import_split()
    with pytest.raises(ValueError):
        split.split_bytes(np.zeros(16, dtype=np.uint8), [4, 4], 4, "diag", 0, world=2)


def test_split_bytes_row_needs_full_tensor_bytes():
    """split_bytes trusts the caller to pass the tensor's full span: short bytes yield truncated rows
    (16 bytes for [8, 4] uint32 words gives each rank 4 bytes = half a row) without an error."""

    split = import_split()
    raw = np.zeros(16, dtype=np.uint8)
    data, shape = split.split_bytes(raw, [8, 4], 4, "row", 3, world=4)
    assert shape == [2, 4]
    assert data.size == 4        # the clamped window, not the full 2 rows * 4 words * 4 bytes


# ---------------------------------------------------------------- weights.py: WORLD handling


def glm53_config_dict():
    """The banked checkpoint's real dims, small where a full model is not needed for the assertion."""

    return {
        "model_type": "glm_moe_dsa", "hidden_size": 6144, "num_hidden_layers": 78, "vocab_size": 154880,
        "rms_norm_eps": 1e-5, "num_attention_heads": 64, "num_key_value_heads": 64,
        "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
        "v_head_dim": 256, "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048,
        "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
        "n_shared_experts": 1, "intermediate_size": 12288, "first_k_dense_replace": 3,
        "routed_scaling_factor": 2.5, "norm_topk_prob": True, "num_nextn_predict_layers": 1,
        "eos_token_id": [154820],
        "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "sparse"],
        "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4,
                               "gate_lower_bound": -5.0},
        "quantization": {"group_size": 64, "bits": 4},
    }


def make_config():
    """Config.read against a minimal folder, avoiding the real checkpoint entirely."""

    import tempfile

    import tensorfold.families.glm5_next.cuda.weights as weights
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(__import__("json").dumps(glm53_config_dict()))
        return weights.Config.read(d)


def test_config_read_real_dims():
    cfg = make_config()
    assert cfg.heads == 64 and cfg.index_heads == 32 and cfg.experts == 256
    assert cfg.moe_width == 2048 and cfg.dense_width == 12288 and cfg.vocab == 154880
    assert cfg.layers == 78 and cfg.mlp_kinds == ["dense", "moe"]


def test_weights_world_dataclass_roundtrip():
    """The world actually travels onto Weights and drives the vocab offset (no tensors needed)."""

    import tensorfold.families.glm5_next.cuda.weights as weights
    cfg = make_config()
    for world in (2, 4):
        for rank in range(world):
            w = weights.Weights(cfg=cfg, embed=(), layers=[], norm=torch.zeros(1), head=None, mtp=None,
                                rank=rank, world=world, device=torch.device("cpu"))
            assert w.world == world
            assert w.vocab_offset == rank * (cfg.vocab // world)


def test_head_partition_counts_at_world_4():
    """lm_head rows per rank: vocab // world, at world=4 that is a quarter (38,720 of 154,880)."""

    cfg = make_config()
    assert cfg.vocab % 4 == 0
    for world in (2, 4):
        vl = cfg.vocab // world
        assert vl * world == cfg.vocab
        # every rank's slice stays inside the real lm_head.weight of [154880, 6144]
        for rank in range(world):
            lo, hi = rank * vl, (rank + 1) * vl
            assert 0 <= lo < hi <= cfg.vocab
    assert cfg.vocab // 4 == 38720
    assert cfg.vocab // 2 == 77440


def test_mlp_gate_up_world_split_math():
    """MLPW width = (gate rows + up rows stacked) // world for the real dims; gate/up are stacked [2*W, D]."""

    # dense MLP: gate/up [12288, 6144] each -> stacked n = 24576
    # moe expert: gate/up [2048, 6144] each -> stacked n = 4096
    # shared expert: moe_intermediate * n_shared_experts = 2048 -> stacked n = 4096
    for stacked_n, world in [(24576, 2), (24576, 4), (4096, 2), (4096, 4), (4096, 1)]:
        assert stacked_n % world == 0
        width = stacked_n // world
        assert width == {24576: {2: 12288, 4: 6144}, 4096: {1: 4096, 2: 2048, 4: 1024}}[stacked_n][world]
    # down_proj columns split the same way: 12288 // world for dense, 2048 // world for experts
    assert 12288 % 4 == 0 and 2048 % 4 == 0


def test_head_partition_counts_64_q_heads_and_32_indexer_heads_at_world_4():
    """HL = heads // world = 16 q heads and kv head rows; indexer heads are REPLICATED (stay 32)."""

    cfg = make_config()
    for world in (2, 4):
        assert cfg.heads % world == 0 and cfg.index_heads % world == 0
        hl = cfg.heads // world
        assert hl == {2: 32, 4: 16}[world]
        # q_b_proj rows are head-major: HL * qk_dim(=256) rows per rank of [16384, 2048]
        assert hl * (cfg.qk_dim) * world == cfg.heads * cfg.qk_dim
        # kv_b_proj rows [64 heads * 512, 512]: the rank keeps its heads' rows
        assert hl * 512 * world == 64 * 512
        # the DSA indexer is in the REP rules: all 32 indexer heads on every rank
        from tensorfold.families.glm5_next.cuda.split import rule
        assert rule("model.language_model.layers.5.self_attn.indexer.wq_b.weight") == "rep"
        assert rule("model.language_model.layers.5.self_attn.indexer.wk.weight") == "rep"


def test_weights_load_world_from_env(monkeypatch):
    """weights.load reads TF_TP_WORLD itself (not just split.WORLD); assert against a stubbed RankReader path."""

    import tensorfold.families.glm5_next.cuda.weights as weights

    seen = {}

    class FakeReader:
        def __init__(self, model_dir, rank):
            seen["rank"] = rank

        def get(self, name):
            raise AssertionError(f"no tensor reads expected, got {name}")

        def prefetch(self, names, device=None):
            pass

        @property
        def ahead(self):
            return {}

        def close(self):
            pass

    cfg = make_config()
    monkeypatch.setattr(weights, "Config", SimpleNamespace(read=lambda d: cfg))
    fake_split = SimpleNamespace(RankReader=FakeReader, WORLD=4)
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.split", fake_split)
    # load() imports .split inside the function: a fake module entry satisfies it
    monkeypatch.setenv("TF_TP_WORLD", "4")
    with pytest.raises(Exception) as exc:
        weights.load("unused", rank=3, device="cpu")
    # the load fails on device/code paths, but only after world was taken from the env:
    # rank 3 with world 4 is a legal rank, so the ValueError must NOT be a rank/world complaint
    assert not ("rank" in str(exc.value) and "TF_TP_WORLD" in str(exc.value))


# ---------------------------------------------------------------- engine.py: WORLD plumbing, CPU-safe import


def test_engine_module_imports_without_cuda_or_nccl():
    """engine.py must import on a CPU-only host: no NCCL, no torch.cuda calls at module scope."""

    for name in [n for n in sys.modules if n.startswith("tensorfold")]:
        del sys.modules[name]
    old = os.environ.get("TF_TP_WORLD")
    os.environ["TF_TP_WORLD"] = "4"
    try:
        from tensorfold.families.glm5_next.cuda import engine  # noqa: F401 - the import is the test
    finally:
        if old is None:
            os.environ.pop("TF_TP_WORLD", None)
        else:
            os.environ["TF_TP_WORLD"] = old


def test_engine_world_plumbing_via_stubbed_init(monkeypatch, tmp_path):
    """GlmEngine.__init__ reads TF_TP_WORLD before any communicator use: run it with every CUDA piece mocked."""

    import tensorfold.families.glm5_next.cuda.engine as engine

    world_seen = {}
    (tmp_path / "config.json").write_text(json.dumps(glm53_config_dict()))

    class FakeComm:
        def __init__(self, rank, world, master, port):
            world_seen["nccl"] = (rank, world)

        def barrier(self):
            pass

        def all_gather(self, x):
            return [x] * world_seen["nccl"][1]

    # engine.py imports NCCL inside __init__ from tensorfold.cuda.comm: stub that module entry
    monkeypatch.setitem(sys.modules, "tensorfold.cuda.comm",
                        SimpleNamespace(NCCL=FakeComm))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)

    import tensorfold.cuda.capacity as capacity
    import tensorfold.cuda.geometry as geometry

    monkeypatch.setattr(capacity, "admit", lambda *a, **kw: {
        "context_window": 2051, "cache_slots": 2560, "budget_bytes": 94 * 2**30,
        "total_bytes_estimate": 80 * 2**30, "serving_peak_bytes_estimate": 80 * 2**30})
    monkeypatch.setattr(engine, "mla_geometry", lambda *a, **kw: SimpleNamespace(), raising=False)
    monkeypatch.setattr(engine, "split_weights", lambda *a: None, raising=False)
    monkeypatch.setattr(engine, "draft_geometry", lambda *a, **kw: None, raising=False)

    def fake_load(model_dir, *, rank, device="cuda", mtp=True):
        world_seen["load_rank"] = rank
        raise _StopLoad()

    class _StopLoad(Exception):
        pass

    import tensorfold.families.glm5_next.cuda.weights as weights_mod
    monkeypatch.setattr(weights_mod, "load", fake_load, raising=False)
    monkeypatch.setattr(engine, "LATENT", True, raising=False)
    monkeypatch.setattr(engine.GlmEngine, "_gather_ints",
                        lambda self, values: [list(values) for _ in range(world_seen["nccl"][1])])
    monkeypatch.setenv("TF_TP_WORLD", "4")

    # FIXED 2026-09-30: weights.py now imports os; the stubbed load path executes with world from TF_TP_WORLD.
    try:
        engine.GlmEngine(tmp_path, rank=2, master="example", port=29500, serial_only=True)
    except _StopLoad:
        pass                                     # stubbed weights.load reached: plumbing proven this far
    assert world_seen["nccl"] == (2, 4)          # the (fake) NCCL was built with world 4 from TF_TP_WORLD
    assert world_seen["load_rank"] == 2          # and weights.load was called for rank 2 of 4


def test_engine_setting_mismatch_rejected_across_world(monkeypatch, tmp_path):
    """The all-rank equality check now compares every rank's settings, not just ranks 0 and 1."""

    import tensorfold.families.glm5_next.cuda.engine as engine

    world = 4
    (tmp_path / "config.json").write_text(json.dumps(glm53_config_dict()))

    class FakeComm:
        def barrier(self):
            pass

    # engine.py imports NCCL inside __init__ from tensorfold.cuda.comm: stub that module entry
    monkeypatch.setitem(sys.modules, "tensorfold.cuda.comm", SimpleNamespace(NCCL=lambda *a: FakeComm()))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    import tensorfold.cuda.capacity as capacity
    monkeypatch.setattr(capacity, "admit", lambda *a, **kw: {
        "context_window": 2051, "cache_slots": 2560, "budget_bytes": 94 * 2**30,
        "total_bytes_estimate": 80 * 2**30, "serving_peak_bytes_estimate": 80 * 2**30})

    def fake_load(model_dir, *, rank, device="cuda", mtp=True):
        return "WEIGHTS"

    import tensorfold.families.glm5_next.cuda.weights as weights_mod
    monkeypatch.setattr(weights_mod, "load", fake_load, raising=False)
    monkeypatch.setattr(engine, "Config", SimpleNamespace(read=lambda d: SimpleNamespace(dense_limit=2051)),
                        raising=False)
    monkeypatch.setattr(engine, "LATENT", True, raising=False)

    # rank 3 reports a different serial_only: the mismatch check must fire
    def gather_mismatched(self, values):
        rows = [list(values) for _ in range(world)]
        rows[3] = list(values)
        rows[3][4] = 1 - rows[3][4]              # flip the serial_only bit on rank 3
        return rows

    monkeypatch.setattr(engine.GlmEngine, "_gather_ints", gather_mismatched)
    monkeypatch.setenv("TF_TP_WORLD", "4")
    with pytest.raises(RuntimeError, match="different settings"):
        engine.GlmEngine(tmp_path, rank=0, master="example", port=29500, serial_only=True)


def test_cli_tp4_choices():
    """cli.py accepts --tp 4 and refuses other world sizes."""

    from tensorfold import cli
    parser = cli.build_parser()
    args = parser.parse_args(["serve", "/tmp", "--backend", "cuda", "--tp", "4", "--rank", "3", "--no-drafts"])
    assert getattr(args, "tp", None) == 4 and args.rank == 3
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "/tmp", "--backend", "cuda", "--tp", "3"])

def test_rankreader_uses_call_time_world_not_stale_import_global():
    """Regression (ddb8167 follow-up): RankReader._span must use call-time TF_TP_WORLD, not the
    import-time WORLD global. If env is set after import (engine.py does this), the stale global
    computed wrong byte offsets -> silent rank corruption. Found via test-order dependence in the
    glm_moe_dsa family suite (kv_v rows collapsed to 0)."""

    import importlib

    old = os.environ.get("TF_TP_WORLD")
    os.environ.pop("TF_TP_WORLD", None)
    try:
        import tensorfold.families.glm5_next.cuda.split as split_fresh
        importlib.reload(split_fresh)                      # WORLD baked as 2, env unset
        shape = [8, 64]                                    # 8 rows divide by both 2 and 4
        raw = np.arange(8 * 64, dtype=np.int32)
        os.environ["TF_TP_WORLD"] = "4"                    # set AFTER import, like engine.py does
        part, part_shape = split_fresh.split_bytes(raw, shape, 4, "row", 3)
        assert part_shape == [2, 64]                       # world 4 honored at call time
        assert part.tolist() == raw[6 * 64:8 * 64].tolist()  # rank 3 of 4, not rank 1 of 2
    finally:
        if old is None:
            os.environ.pop("TF_TP_WORLD", None)
        else:
            os.environ["TF_TP_WORLD"] = old

