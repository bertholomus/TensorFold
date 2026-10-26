"""The routed experts' trellises packed one allocation a projection (weights.Slab), on CPU: each slot holds its expert's
bytes, dtype and shape exactly, slots start on 512 bytes, a piece that does not fit gets its own allocation, and a
packed load of the tiny checkpoint holds the same bits as the per-expert load on both ranks. The GPU half (the same
logits and states from both loads) is tests/test_bailing_10_engine.py's.

Run: python -m pytest tests/test_bailing_16_pack.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "tests"))

torch = pytest.importorskip("torch")
if torch.cuda.is_available():          # these tests must never touch a GPU even where one exists
    pytest.skip("CPU-only bailing_v3 packing tests", allow_module_level=True)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _storage(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def test_equal_pieces_share_one_allocation_on_512_byte_slots():
    from tensorfold.families.bailing_v3.cuda.weights import SLOT_ALIGN, Slab

    g = torch.Generator().manual_seed(3)
    pieces = [torch.randint(-32768, 32767, (4, 3, 48), dtype=torch.int16, generator=g) for _ in range(5)]
    slab = Slab(len(pieces), torch.device("cpu"))
    got = [slab.put(p) for p in pieces]
    assert slab.slot == 1536                                      # 1,152 bytes up to a 512-byte slot
    base = slab.buf.data_ptr()
    for e, (src, out) in enumerate(zip(pieces, got)):
        assert _same(out, src) and out.is_contiguous()
        assert _storage(out) == _storage(slab.buf)                # one allocation behind every piece
        assert out.data_ptr() - base == e * slab.slot             # slot e, in the order put
        assert (out.data_ptr() - base) % SLOT_ALIGN == 0
    assert slab.buf.numel() == len(pieces) * slab.slot


def test_a_larger_piece_or_one_past_the_count_gets_its_own_allocation():
    from tensorfold.families.bailing_v3.cuda.weights import Slab

    slab = Slab(2, torch.device("cpu"))
    small = torch.arange(256, dtype=torch.int16).reshape(4, 64)  # 512 bytes: the slot exactly
    big = torch.arange(512, dtype=torch.int16).reshape(8, 64)
    a = slab.put(small)
    b = slab.put(big)                                             # larger than the first piece's slot
    assert slab.slot == 512
    c = slab.put(small.flip(0))
    d = slab.put(small)                                           # past the count
    assert _same(a, small) and _same(b, big) and _same(c, small.flip(0)) and _same(d, small)
    assert _storage(a) == _storage(c) == _storage(slab.buf)
    assert _storage(b) != _storage(slab.buf) and _storage(d) != _storage(slab.buf)


@pytest.mark.parametrize("rank", [0, 1])
def test_the_packed_load_holds_the_per_expert_loads_bits(tmp_path, monkeypatch, rank):
    from bailing_fakes import write_checkpoint

    from tensorfold.families.bailing_v3.cuda import weights as W

    folder = write_checkpoint(tmp_path)
    monkeypatch.setenv("TF_TP_WORLD", "2")
    old = W.load(folder, rank=rank, device="cpu", pack=False)
    new = W.load(folder, rank=rank, device="cpu", pack=True)
    moes = [(a.moe, b.moe) for a, b in zip(old.layers, new.layers) if a.moe is not None]
    moes.append((old.mtp.layer.moe, new.mtp.layer.moe))
    assert len(moes) == 3                                         # layers 1 and 2, and the MTP layer's
    for a, b in moes:
        assert (a.experts.count, a.experts.dims, a.experts.width, a.experts.codebook) == \
            (b.experts.count, b.experts.dims, b.experts.width, b.experts.codebook)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            pa, pb = a.experts.parts[proj], b.experts.parts[proj]
            assert len(pa) == len(pb) == a.experts.count
            for (ta, ua, va), (tb, ub, vb) in zip(pa, pb):
                assert _same(ta, tb) and _same(ua, ub) and _same(va, vb)
            assert len({_storage(t) for t, _, _ in pb}) == 1      # the projection's one allocation
            assert len({_storage(t) for t, _, _ in pa}) == len(pa)   # the old way: one an expert


def test_the_memory_line_reads_torch_and_the_node(tmp_path):
    from tensorfold.families.bailing_v3.cuda.engine import memory_line

    stats = {"allocated_bytes.all.current": 3 << 30, "reserved_bytes.all.current": 5 << 30,
             "reserved_bytes.small_pool.current": 1 << 29}
    fake = SimpleNamespace(cuda=SimpleNamespace(memory_stats=lambda: stats))
    info = tmp_path / "meminfo"
    info.write_text("MemTotal:       129804288 kB\nMemAvailable:    8388608 kB\n")
    line = memory_line(fake, 1, "loaded", meminfo=str(info))
    assert line == ("[tensorfold] rank 1 loaded: torch allocated 3.00 GiB, reserved 5.00 (small pool 0.50), "
                    "MemAvailable 8.00 GiB")
    assert memory_line(fake, 0, "built", meminfo=str(tmp_path / "none")).endswith("(small pool 0.50)")


def test_the_prompt_chunk_rows_come_from_the_environment(monkeypatch):
    from tensorfold.families.bailing_v3.cuda.engine import prompt_chunk_rows

    monkeypatch.delenv("TF_LING_PREFILL_ROWS", raising=False)
    assert prompt_chunk_rows() == 2048                            # geometry.PREFILL_ROWS, as before
    monkeypatch.setenv("TF_LING_PREFILL_ROWS", "512")
    assert prompt_chunk_rows() == 512
    for bad in ("0", "64", "100", "-512", "512.0", "1e3", "32768"):
        with pytest.raises(ValueError):
            prompt_chunk_rows(bad)
    assert prompt_chunk_rows(" 128 ") == 128 and prompt_chunk_rows("16384") == 16384
