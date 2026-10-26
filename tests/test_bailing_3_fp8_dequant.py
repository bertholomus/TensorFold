"""FP8 block dequant: tools/ling_fp8_dequant.py against an independent blockwise implementation.

Synthetic weights cover every grid shape the checkpoint has (whole blocks, kv_a's half-full fifth block row, the MTP
g_proj's single block row, ragged columns), the chunking, the bf16 copy and the refusals; ``check`` runs on a small
FP8 checkpoint written here. Where the Ling-3.1-flash FP8 checkpoint is present (LING31_FP8_DIR), kv_a's fifth block
row and the MTP g_proj are checked on the real tensors too.

Run: python -m pytest tests/test_bailing_3_fp8_dequant.py -q
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TOOL = REPO / "tools" / "ling_fp8_dequant.py"
BANK = Path(os.environ.get("LING31_FP8_DIR", "/models/Ling-3.1-flash"))

if not TOOL.is_file():
    pytest.skip("tools/ling_fp8_dequant.py does not exist yet", allow_module_level=True)

torch = pytest.importorskip("torch")
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("this torch has no float8_e4m3fn", allow_module_level=True)


def _load_tool():
    """The tool by its path (tools/ is a folder of scripts, not a package)."""

    import importlib.util

    spec = importlib.util.spec_from_file_location("ling_fp8_dequant", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fp8tool = _load_tool()


def _blockwise(fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The independent block dequant: W = fp32(fp8) * scale[row // 128, col // 128], one block at a time."""

    out = torch.empty(fp8.shape, dtype=torch.float32)
    br, bc = scales.shape
    for r in range(br):
        for c in range(bc):
            out[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128] = \
                fp8[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128].to(torch.float32) * scales[r, c]
    return out


def _random_fp8(rows: int, cols: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite e4m3 codes (every code but the two NaNs) and positive fp32 scales of the matching grid."""

    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (rows, cols), generator=g, dtype=torch.int32)
    codes = torch.where((codes & 0x7F) == 0x7F, codes & 0xF0, codes).to(torch.uint8)    # 0x7F / 0xFF are NaN
    fp8 = codes.view(torch.float8_e4m3fn)
    gr, gc = -(-rows // 128), -(-cols // 128)
    scales = torch.rand((gr, gc), generator=g, dtype=torch.float32) * 1e-3 + 1e-5
    return fp8, scales


@pytest.mark.parametrize("rows,cols", [(256, 384), (576, 4096), (64, 4096), (300, 200), (1, 1)])
def test_dequant_matches_the_blockwise_rule_bit_for_bit(rows, cols):
    fp8, scales = _random_fp8(rows, cols, rows * 7 + cols)
    want = _blockwise(fp8, scales)
    assert torch.equal(fp8tool.dequant(fp8, scales), want)
    # the row chunking (whole block rows at a time) does not change a value
    for chunk in (1, 128, 300, 1 << 20):
        assert torch.equal(fp8tool.dequant(fp8, scales, row_chunk=chunk), want), chunk
    # a bf16 copy is the fp32 product rounded once
    assert torch.equal(fp8tool.dequant(fp8, scales, dtype=torch.bfloat16), want.to(torch.bfloat16))


def test_partial_blocks_use_their_own_grid_rows_and_columns():
    """kv_a's [576, 4096] weight has a [5, 32] grid: rows 512..575 take grid row 4 and nothing else."""

    fp8, scales = _random_fp8(576, 4096, 1)
    w = fp8tool.dequant(fp8, scales)
    tail = fp8[512:].to(torch.float32) * scales[4].repeat_interleave(128)[None, :]
    assert torch.equal(w[512:], tail)
    # ragged columns: [64, 200] has a [1, 2] grid, columns 128..199 take grid column 1
    fp8, scales = _random_fp8(64, 200, 2)
    w = fp8tool.dequant(fp8, scales)
    assert torch.equal(w[:, 128:], fp8[:, 128:].to(torch.float32) * scales[0, 1])
    assert torch.equal(w[:, :128], fp8[:, :128].to(torch.float32) * scales[0, 0])


def test_refusals():
    fp8, scales = _random_fp8(576, 4096, 3)
    with pytest.raises(ValueError, match="scale grid"):
        fp8tool.dequant(fp8, scales[:4])                     # a [4, 32] grid for 576 rows: refused, not broadcast
    with pytest.raises(ValueError, match="scale grid"):
        fp8tool.dequant(fp8, scales[:, :1].expand(5, 32).contiguous()[:, :31].contiguous())
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        fp8tool.dequant(fp8.to(torch.float32), scales)
    with pytest.raises(ValueError, match="float32"):
        fp8tool.dequant(fp8, scales.to(torch.float64))
    bad = fp8.view(torch.uint8).clone()
    bad[517, 9] = 0x7F                                       # an e4m3 NaN code
    with pytest.raises(ValueError, match="row 517, column 9"):
        fp8tool.dequant(bad.view(torch.float8_e4m3fn), scales)
    s = scales.clone()
    s[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        fp8tool.dequant(fp8, s)
    assert torch.isinf(fp8tool.dequant(fp8, s, finite=False)).any()


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, offset, payload = {}, 0, b""
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(payload)


def test_check_and_load_on_a_small_fp8_checkpoint(tmp_path):
    """check() counts FP8 weights, lists partial blocks, flags a wrong grid and an orphan grid; load() dequantizes."""

    t: dict[str, tuple[str, list[int], bytes]] = {}

    def put(name: str, fp8: torch.Tensor, scales: torch.Tensor | None) -> None:
        t[name + ".weight"] = ("F8_E4M3", list(fp8.shape), fp8.view(torch.uint8).numpy().tobytes())
        if scales is not None:
            t[name + ".weight_scale_inv"] = ("F32", list(scales.shape), scales.numpy().tobytes())

    kv_a, kv_s = _random_fp8(576, 256, 4)
    put("model.layers.7.attention.kv_a_proj_with_mqa", kv_a, kv_s)
    q, q_s = _random_fp8(256, 256, 5)
    put("model.layers.7.attention.q_a_proj", q, q_s)
    bad, bad_s = _random_fp8(256, 256, 6)
    put("model.layers.8.attention.q_a_proj", bad, bad_s[:1])            # the wrong grid
    t["model.layers.9.mlp.gate.weight_scale_inv"] = ("F32", [1, 1], b"\0\0\x80?")     # no weight: an orphan
    norm = torch.ones(256, dtype=torch.bfloat16)
    t["model.norm.weight"] = ("BF16", [256], norm.view(torch.uint16).numpy().tobytes())
    _write_safetensors(tmp_path / "model.safetensors", t)

    rep = fp8tool.check(tmp_path)
    assert rep["fp8"] == 3
    assert [k for k, _, _ in rep["bad"]] == ["model.layers.8.attention.q_a_proj.weight"]
    assert rep["orphans"] == ["model.layers.9.mlp.gate.weight_scale_inv"]
    assert rep["partial"] == {"attention.kv_a_proj_with_mqa.weight 576x256 grid [5, 2]": 1}
    assert fp8tool.main(["--check", str(tmp_path)]) == 1

    w = fp8tool.load(tmp_path, "model.layers.7.attention.kv_a_proj_with_mqa.weight")
    assert torch.equal(w, _blockwise(kv_a, kv_s))
    assert torch.equal(fp8tool.load(tmp_path, "model.norm.weight"), norm.float())


def _bank_tensor(name: str):
    """(fp8 weight, scale grid) of one banked tensor, read from its shard."""

    return (fp8tool.read_tensor(BANK, f"{name}.weight"), fp8tool.read_tensor(BANK, f"{name}.weight_scale_inv"))


def _bank_raw(name: str) -> torch.Tensor:
    """The same tensor read independently of the tool (header parse and byte view here)."""

    index = json.loads((BANK / "model.safetensors.index.json").read_text())["weight_map"]
    file = BANK / index[name]
    with file.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        info = header[name]
        f.seek(8 + n + info["data_offsets"][0])
        raw = bytearray(f.read(info["data_offsets"][1] - info["data_offsets"][0]))
    if info["dtype"] == "F8_E4M3":
        return torch.frombuffer(raw, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(info["shape"])
    return torch.frombuffer(raw, dtype=torch.float32).reshape(info["shape"])


@pytest.mark.skipif(not BANK.is_dir(), reason="the banked Ling-3.1-flash checkpoint is not on this host")
def test_kv_a_partial_block_row_and_mtp_g_proj():
    """kv_a's scale grid is [5, 32] (its last block row is half full) and the MTP g_proj's is [1, 32]: the
    tool's dequant matches the explicit blockwise rule there, bit for bit in fp32."""

    fp8, scales = _bank_tensor("model.layers.7.attention.kv_a_proj_with_mqa")
    assert torch.equal(fp8.view(torch.uint8),
                       _bank_raw("model.layers.7.attention.kv_a_proj_with_mqa.weight").view(torch.uint8))
    assert fp8.shape == (576, 4096) and scales.shape == (5, 32)
    row = _blockwise(fp8, scales)[4 * 128:576]          # the fifth block row: its 64 rows
    assert torch.equal(fp8tool.dequant(fp8, scales)[4 * 128:576], row)
    fp8, scales = _bank_tensor("model.layers.72.attention.g_proj")
    assert fp8.shape == (64, 4096) and scales.shape == (1, 32)
    assert torch.equal(fp8tool.dequant(fp8, scales), _blockwise(fp8, scales))
