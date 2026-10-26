"""The pass-through restore tool (tools/ling_restore_passthrough.py) on a synthetic pair: the output's own
name-to-shard map, the partition check, BF16 pass-throughs compared and restored by sha256, the index's
total_size, and the fused conv1d restored as one tensor from the source's three, every pass-through byte-equal
to the source.

Run: python -m pytest tests/test_bailing_11_restore.py -q
CPU only, no model on disk: the pair is written from raw bytes.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "tests", REPO / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from ling_restore_passthrough import _header, restore_dir  # noqa: E402

D = 128                 # hidden
LD = 1024               # one conv / projection's rows (8 KDA heads x 128)
KV_ROWS = 2 * LD        # kv_b's rows (head-major qk_nope + v_dim)
P = "model.layers.0.attention."


def _shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> int:
    """One safetensors file: name -> (dtype, shape, raw bytes), header then data; returns the data bytes."""
    header, payload, offset = {}, b"", 0
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    blob = json.dumps(header, separators = (",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)
    return offset


def _write_pair(tmp_path: Path, *, drop_norm: bool = False, fused_in_input: bool = False):
    """A 2-shard BF16 input and a 1-shard EXL3 output under a different file name.

    Input shard 1: the norm (BF16), A_log (F32), kv_b (BF16) and the conv weight(s) (BF16, three parts unless
    ``fused_in_input``); shard 2: the q/k/v projections and one more BF16 Linear (g_proj). Output: the norm F16,
    A_log F32 already holding the source's bytes, kv_b F16, the fused conv1d BF16, the fused qkv_proj group and
    g_proj's own group.
    """
    rng = np.random.default_rng(11)

    def raw(n: int) -> bytes:
        return rng.integers(0, 256, n, dtype = np.uint8).tobytes()

    convs = ({P + "conv1d.weight": ("BF16", [3 * LD, 1, 4], raw(3 * LD * 4 * 2))} if fused_in_input else
             {P + x + "_conv1d.weight": ("BF16", [LD, 1, 4], raw(LD * 4 * 2)) for x in "qkv"})
    src = {
        "model.layers.0.input_layernorm.weight": ("BF16", [D], raw(D * 2)),
        P + "A_log": ("F32", [8], raw(8 * 4)),
        P + "kv_b_proj.weight": ("BF16", [KV_ROWS, D], raw(KV_ROWS * D * 2)),
        **convs,
        P + "q_proj.weight": ("BF16", [LD, D], raw(LD * D * 2)),
        P + "k_proj.weight": ("BF16", [LD, D], raw(LD * D * 2)),
        P + "v_proj.weight": ("BF16", [LD, D], raw(LD * D * 2)),
        P + "g_proj.weight": ("BF16", [LD, D], raw(LD * D * 2)),
    }
    conv_src = (convs[P + "conv1d.weight"][2] if fused_in_input else
                b"".join(convs[P + x + "_conv1d.weight"][2] for x in "qkv"))
    first = ["model.layers.0.input_layernorm.weight", P + "A_log", P + "kv_b_proj.weight", *convs]
    in_shards = {"model-00001-of-00002.safetensors": {n: src[n] for n in first},
                 "model-00002-of-00002.safetensors": {n: src[n] for n in src if n not in first}}
    inp = tmp_path / "bf16-input"
    inp.mkdir()
    weight_map: dict[str, str] = {}
    for fname, tensors in in_shards.items():
        _shard(inp / fname, tensors)
        weight_map.update({n: fname for n in tensors})
    (inp / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(len(v[2]) for v in src.values())}, "weight_map": weight_map}))

    out = tmp_path / "exl3-output"
    out.mkdir()
    out_tensors = {
        "model.layers.0.input_layernorm.weight": ("F16", [D], raw(D * 2)),       # exl3 wrote F16: rewritten
        P + "A_log": ("F32", [8], src[P + "A_log"][2]),                          # already the source's bytes
        P + "kv_b_proj.weight": ("F16", [KV_ROWS, D], raw(KV_ROWS * D * 2)),     # exl3 wrote F16: rewritten
        P + "conv1d.weight": ("BF16", [3 * LD, 1, 4], raw(3 * LD * 4 * 2)),      # fused: rewritten
        P + "qkv_proj.trellis": ("I16", [D // 16, 3 * LD // 16, 64],
                                 raw((D // 16) * (3 * LD // 16) * 64 * 2)),
        P + "qkv_proj.suh": ("F16", [D], raw(D * 2)),
        P + "qkv_proj.svh": ("F16", [3 * LD], raw(3 * LD * 2)),
        P + "g_proj.trellis": ("I16", [D // 16, LD // 16, 64], raw((D // 16) * (LD // 16) * 64 * 2)),
        P + "g_proj.suh": ("F16", [D], raw(D * 2)),
        P + "g_proj.svh": ("F16", [LD], raw(LD * 2)),
    }
    if drop_norm:
        del out_tensors["model.layers.0.input_layernorm.weight"]
    _shard(out / "model.safetensors", out_tensors)
    # a deliberately wrong total_size: the restore recomputes it
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 1}, "weight_map": {n: "model.safetensors" for n in out_tensors}}))
    return inp, out, src, conv_src, out_tensors


def test_the_restore_passes_the_synthetic_pair_and_hashes_every_pass_through(tmp_path):
    inp, out, src, conv_src, out_tensors = _write_pair(tmp_path)
    report_file = tmp_path / "report.json"
    assert restore_dir(inp, out, report_file) == 0
    report = json.loads(report_file.read_text())
    assert report["result"] == "PASS"
    assert sorted(report["restored"]) == [P + "A_log", P + "conv1d.weight", P + "kv_b_proj.weight",
                                          "model.layers.0.input_layernorm.weight"]
    # every pass-through's sha256 is the source's: by name, and the fused conv as cat(q, k, v) rows
    for name, got in report["restored"].items():
        want = conv_src if name == P + "conv1d.weight" else src[name][2]
        assert got == hashlib.sha256(want).hexdigest(), name
    assert report["verify_bad"] == [] and report["unaccounted"] == [] and report["mismatch"] == []
    assert report["already_identical"] == 1        # A_log was written with the source's bytes already
    # the dtypes exllamav3 wrote (recorded before the overwrite), and what the output's header holds after
    assert report["written_dtype"] == {P + "A_log": "F32", P + "conv1d.weight": "BF16",
                                       P + "kv_b_proj.weight": "F16",
                                       "model.layers.0.input_layernorm.weight": "F16"}
    assert report["dtype_changed"] == {P + "kv_b_proj.weight": ["F16", "BF16"],
                                       "model.layers.0.input_layernorm.weight": ["F16", "BF16"]}
    head, base = _header(out / "model.safetensors")
    for name, dtype, shape in (("model.layers.0.input_layernorm.weight", "BF16", [D]),
                               (P + "kv_b_proj.weight", "BF16", [KV_ROWS, D]),
                               (P + "A_log", "F32", [8]),
                               (P + "conv1d.weight", "BF16", [3 * LD, 1, 4])):
        assert head[name]["dtype"] == dtype and head[name]["shape"] == shape, name
    # the quantized groups are untouched: no input counterpart, still I16
    assert head[P + "qkv_proj.trellis"]["dtype"] == "I16" and head[P + "g_proj.trellis"]["dtype"] == "I16"
    # the restore's own verify reads the output through the output's own shard map: 1 shard, another name
    blob = (out / "model.safetensors").read_bytes()
    for name, want in report["restored"].items():
        lo, hi = head[name]["data_offsets"]
        assert hashlib.sha256(blob[base + lo:base + hi]).hexdigest() == want, name
    # the index's total_size was recomputed from the restored shards
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["total_size"] == sum(len(v[2]) for v in out_tensors.values())
    assert report["skipped_quantized"] == 2        # the two trellis packs


def test_a_fused_source_conv_restores_by_name_too(tmp_path):
    """The same output against a source that already stored conv1d.weight fused: the by-name path restores it."""
    inp, out, src, conv_src, _ = _write_pair(tmp_path, fused_in_input = True)
    report_file = tmp_path / "report.json"
    assert restore_dir(inp, out, report_file) == 0
    report = json.loads(report_file.read_text())
    assert report["result"] == "PASS" and report["verify_bad"] == []
    assert report["restored"][P + "conv1d.weight"] == hashlib.sha256(conv_src).hexdigest()


def test_a_dropped_norm_fails_the_partition_check_by_name(tmp_path):
    inp, out, src, conv_src, out_tensors = _write_pair(tmp_path, drop_norm = True)
    report_file = tmp_path / "report.json"
    assert restore_dir(inp, out, report_file) == 1
    report = json.loads(report_file.read_text())
    assert report["result"] == "FAIL"
    assert report["unaccounted"] == ["model.layers.0.input_layernorm.weight"]
