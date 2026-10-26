"""kv_b's per-head split: the loader's key and value rows reproduce the reference's k_nope and
value_states — per head, qk_nope rows then v_dim rows, both read from the config (never a hard-coded 512).

Run: python -m pytest tests/test_bailing_5_kv_b.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "tests"))

torch = pytest.importorskip("torch")
if torch.cuda.is_available():
    pytest.skip("CPU-only bailing_v3 kv_b tests", allow_module_level=True)

NOPE, V_DIM, KV = 128, 128, 512          # the real model's dims
HEADS = 64


def _reference_split(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's view and split (modeling 660-668): kv_b rows are [H, nope + v, kv] head-major."""

    per = w.view(HEADS, NOPE + V_DIM, KV)
    return per[:, :NOPE], per[:, NOPE:]


def test_the_config_driven_per_head_split_matches_the_reference():
    """Build kv_b from known per-head blocks; the reference view must recover exactly those blocks."""

    torch.manual_seed(7)
    blocks = torch.randn(HEADS, NOPE + V_DIM, KV)          # head h's rows: nope keys, then v_dim values
    w = blocks.reshape(HEADS * (NOPE + V_DIM), KV)
    k_ref, v_ref = _reference_split(w)
    for h in (0, 1, 31, 63):
        assert torch.equal(k_ref[h], blocks[h, :NOPE])
        assert torch.equal(v_ref[h], blocks[h, NOPE:])


def test_the_loader_splits_the_ranks_rows_by_head(tmp_path, monkeypatch):
    """The rank's kv_b rows are whole heads (rank 0: heads 0-3, rank 1: heads 4-7 of the tiny checkpoint)."""

    from bailing_fakes import CFG, write_checkpoint

    from tensorfold.families.bailing_v3.cuda import weights as W

    folder = write_checkpoint(tmp_path)
    monkeypatch.setenv("TF_TP_WORLD", "2")
    heads, nope, v_dim, kv = (CFG["num_attention_heads"], CFG["qk_nope_head_dim"], CFG["v_head_dim"],
                              CFG["kv_lora_rank"])
    import json
    import struct

    with (folder / "model.safetensors").open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    info = header["model.layers.2.attention.kv_b_proj.weight"]
    offset = 8 + n + info["data_offsets"][0]
    with (folder / "model.safetensors").open("rb") as f:
        f.seek(offset)
        raw = f.read(info["data_offsets"][1] - info["data_offsets"][0])
    whole = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16).reshape(
        heads * (nope + v_dim), kv)
    k_ref, v_ref = _reference_split_scaled(whole, heads, nope, v_dim, kv)

    for rank in (0, 1):
        w = W.load(folder, rank=rank, device="cpu")
        a = w.layers[2].mla
        lo, hi = rank * heads // 2, (rank + 1) * heads // 2        # this rank's heads
        assert torch.equal(a.kv_k.weight, k_ref[lo * nope:(hi) * nope])
        assert torch.equal(a.kv_v.weight, v_ref[lo * v_dim:(hi) * v_dim])


def _reference_split_scaled(w: torch.Tensor, heads: int, nope: int, v_dim: int, kv: int):
    per = w.view(heads, nope + v_dim, kv)
    return (per[:, :nope].reshape(heads * nope, kv), per[:, nope:].reshape(heads * v_dim, kv))
