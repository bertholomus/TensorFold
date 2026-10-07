"""DeepSeek CUDA warm-up covers prefill tails without loading PyTorch, Triton or a checkpoint."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def warmup(monkeypatch):
    cuda = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None, memory_reserved=lambda: 0)
    torch = ModuleType("torch")
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)
    kernels = ModuleType("tensorfold.families.deepseek_v41.cuda.kernels")
    kernels.DECODE_ROWS = 16
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    path = Path(__file__).parents[1] / "src/tensorfold/families/deepseek_v41/cuda/engine.py"
    spec = importlib.util.spec_from_file_location("tensorfold.families.deepseek_v41.cuda._warmup_test_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WARM_LENGTHS", (1, 17, 33, 131, 514, 1024, 2113))
    monkeypatch.setattr(module, "PREFILL_CHUNK", 2048)

    def run(*, concurrent=True, limit=262144):
        calls = []
        engine = object.__new__(module.DsEngine)
        engine.model = SimpleNamespace(new_cache=lambda capacity: object())
        engine.concurrent, engine.limit, engine.max_rows = concurrent, limit, 6
        engine.drafter = engine.runner = engine.vcfg = None
        engine.sc, engine.dc, engine.rank = object(), None, 0
        engine.prefill = lambda sc, dc, ids: calls.append(len(ids))
        engine._sample = lambda *args: None
        engine._run = lambda ids, *args: calls.append(len(ids))
        engine.warm()
        assert engine.quiet is False
        return calls

    return module, kernels, run


@pytest.mark.parametrize("concurrent", [False, True])
def test_abbreviated_warmup_covers_the_failed_thirteen_row_prefill_tail(warmup, concurrent):
    module, _, run = warmup
    lengths = run(concurrent=concurrent)
    assert 2048 + (20493 % 2048) in lengths
    assert set(range(2049, 2065)) <= set(lengths)
    assert set(range(1, 17)) <= set(lengths)
    assert lengths[:len(module.WARM_LENGTHS)] == list(module.WARM_LENGTHS)


@pytest.mark.parametrize("chunk, rows", [(512, 16), (1024, 8), (8, 16)])
def test_tail_coverage_uses_configured_chunk_and_decode_row_bound(warmup, chunk, rows):
    module, kernels, run = warmup
    module.PREFILL_CHUNK, kernels.DECODE_ROWS = chunk, rows
    module.WARM_LENGTHS = (1, 17)
    lengths = run()
    tails = range(1, min(chunk, rows) + 1)
    assert set(tails) <= set(lengths)
    assert {chunk + n for n in tails} <= set(lengths)
    assert len(lengths) == len(set(lengths))


def test_explicit_lengths_are_preserved_and_required_lengths_are_not_repeated(warmup):
    module, _, run = warmup
    module.WARM_LENGTHS = (2113, 1, 2061, 16, 2113, 2064)
    lengths = run()
    assert lengths[:5] == [2113, 1, 2061, 16, 2064]
    assert len(lengths) == len(set(lengths))


@pytest.mark.parametrize("limit", [40, 2056, 2069, 2072])
def test_warmup_respects_context_and_decode_headroom(warmup, limit, capsys):
    _, _, run = warmup
    lengths = run(limit=limit)
    assert all(n + 8 <= limit for n in lengths)
    assert {2048 + n for n in range(1, 17) if 2048 + n + 8 <= limit} <= set(lengths)
    assert f"warm-up: {len(lengths)} prompt lengths" in capsys.readouterr().out


def test_one_rank_reports_actual_warmup_count(warmup, capsys):
    _, _, run = warmup
    lengths = run()
    assert f"warm-up: {len(lengths)} prompt lengths" in capsys.readouterr().out
