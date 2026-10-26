#!/usr/bin/env python3
"""Pass-through restore for the EXL3 conversion output.

exllamav3 writes unquantized tensors in its own dtype (F16, or BF16/F32 where the source was). After the
conversion, every tensor of the output that exists in the BF16 conversion input is overwritten with the input's
exact bytes and dtype (norms, embedding, mlp.gate.weight, gate.expert_bias / A_log / dt_bias, b_proj, the MLA
head g_proj, kv_b, kv_a/q_a layernorms, enorm / hnorm / final_layernorm, ...), so bf16 stays bf16 and fp32 stays
fp32; the report records the dtype exllamav3 WROTE for every such tensor. The KDA's three short convs are one
fused `attention.conv1d.weight` in the output: it is restored as one tensor, cat(q_conv1d, k_conv1d,
v_conv1d) of the BF16 source along rows. Quantized Linears are packed `.trellis` groups with no input
counterpart and are left alone. The partition check fails the run, listing names, when a BF16 input tensor is
neither restored by name nor consumed by such a group: q/k/v_proj under the fused `qkv_proj` name, every other
Linear (f/g_proj, the experts' gate/up/down) under its own, and the conv parts under the fused conv1d the
restore wrote. BF16 output tensors are compared and restored by sha256 like F16/F32; tensors already
byte-identical are counted, not rewritten. Verification reads every restored tensor back through the OUTPUT's
own name-to-shard map, built from the output shard headers (never the input's shard names). If the output has a
model.safetensors.index.json its total_size is recomputed after the restore. Exit 0 only if every restored
tensor matches, the partition check passes and nothing expected is missing.

usage:
  python3 ling_restore_passthrough.py BF16_INPUT_DIR EXL3_OUTPUT_DIR [--report PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

UNQUANTIZED_OUT_DTYPES = {"F16", "F32", "BF16"}   # what exllamav3 may store unquantized weights as
PASS_INPUT_DTYPES = {"BF16", "F32", "F16"}
CONV_PARTS = ("q_conv1d", "k_conv1d", "v_conv1d")   # the source's three short convs


def _header(file: Path) -> tuple[dict, int]:
    with file.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _pack_header(meta: dict, tensors: dict) -> bytes:
    body = json.dumps({"__metadata__": meta, **tensors}, separators = (",", ":")).encode("utf-8")
    body += b" " * ((8 - len(body) % 8) % 8)
    return struct.pack("<Q", len(body)) + body


def _read_at(file: Path, base: int, lo: int, hi: int) -> bytes:
    with file.open("rb") as f:
        f.seek(base + lo)
        return f.read(hi - lo)


def _weight_map(input_dir: Path) -> dict[str, str]:
    """The BF16 input's name-to-shard map: its index when it has one, else every shard's own header."""
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    return {name: shard.name for shard in sorted(input_dir.glob("*.safetensors"))
            for name in _header(shard)[0] if name != "__metadata__"}


def _consumed_by(name: str, out_names: set[str], restored: dict) -> str | None:
    """The output tensor that consumed a BF16 source tensor, or None when nothing accounts for it.

    A quantized Linear's source weight lives on as the module's `.trellis` group: q/k/v_proj under the fused
    `qkv_proj` name, every other Linear (f/g_proj, the experts' gate/up/down) under its own. The three conv
    parts are consumed by the fused `conv1d.weight`, which the restore must have written.
    """
    base = name[:-len(".weight")] if name.endswith(".weight") else name
    module, _, leaf = base.rpartition(".")
    if leaf in ("q_proj", "k_proj", "v_proj"):
        owner = module + ".qkv_proj.trellis"
        return owner if owner in out_names else None
    if leaf.endswith("_conv1d"):
        owner = module + ".conv1d.weight"
        return owner if owner in out_names and owner in restored else None
    return base + ".trellis" if base + ".trellis" in out_names else None


def _absorb(raw: bytes, info: dict, src_raw: bytes, src_dtype: str, src_shape: list[int],
            off: int, report: dict, name: str) -> tuple[bytes, dict, bool]:
    """One restored tensor: the source's bytes and dtype in place of what the output had.

    Returns the new raw bytes, the header entry to write, and whether the file changed. A tensor the output
    already holds byte-identically with the source's dtype is only counted (report["already_identical"]).
    """
    want = hashlib.sha256(src_raw).hexdigest()
    report["written_dtype"][name] = info["dtype"]
    if hashlib.sha256(raw).hexdigest() == want and info["dtype"] == src_dtype:
        report["already_identical"] += 1
        return raw, info, False
    if info["dtype"] != src_dtype:
        report["dtype_changed"][name] = [info["dtype"], src_dtype]
    return src_raw, {"dtype": src_dtype, "shape": src_shape, "data_offsets": [off, off + len(src_raw)]}, True


def restore_dir(input_dir: Path, output_dir: Path, report_path: Path | None) -> int:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    weight_map = _weight_map(input_dir)
    in_meta: dict[str, tuple[str, list[int]]] = {}          # the input's dtype and shape, per name
    in_heads: dict[str, tuple[dict, int]] = {}              # input shard file -> (header, header bytes)
    for shard in sorted(set(weight_map.values())):
        head, base = _header(input_dir / shard)
        in_heads[shard] = (head, base)
        for name, info in head.items():
            if name != "__metadata__":
                in_meta[name] = (info["dtype"], list(info["shape"]))

    out_shards = sorted(output_dir.glob("*.safetensors"))
    if not out_shards:
        raise SystemExit(f"no safetensors shards under {output_dir}")

    # the output's own name-to-shard map, from the output shard headers
    out_index: dict[str, str] = {}
    out_names: set[str] = set()
    for shard in out_shards:
        for name in _header(shard)[0]:
            if name != "__metadata__":
                out_index[name] = shard.name
                out_names.add(name)

    def input_bytes(name: str) -> tuple[bytes, str, list[int]]:
        shard = weight_map[name]
        head, base = in_heads[shard]
        info = head[name]
        lo, hi = info["data_offsets"]
        return _read_at(input_dir / shard, base, lo, hi), info["dtype"], list(info["shape"])

    report = {"restored": {}, "written_dtype": {}, "dtype_changed": {}, "already_identical": 0,
              "skipped_quantized": 0, "missing_from_input": [], "mismatch": [], "unaccounted": [],
              "verify_bad": [], "restored_count": 0, "result": "FAIL"}
    n_restored = 0

    for shard in out_shards:
        head, base = _header(shard)
        meta = head.get("__metadata__", {})
        tensors = {k: v for k, v in head.items() if k != "__metadata__"}
        order = sorted(tensors, key = lambda n: tensors[n]["data_offsets"][0])
        blobs = []
        out = {}
        off = 0
        changed = False
        with shard.open("rb") as f:
            for name in order:
                info = tensors[name]
                lo, hi = info["data_offsets"]
                f.seek(base + lo)
                raw = f.read(hi - lo)

                if name in weight_map and info["dtype"] in UNQUANTIZED_OUT_DTYPES:
                    if name not in in_meta:
                        report["missing_from_input"].append(name)
                    else:
                        src_raw, src_dtype, src_shape = input_bytes(name)
                        if src_dtype not in PASS_INPUT_DTYPES:
                            # an fp8-source weight that exl3 left unquantized (should not happen with the
                            # BF16 input): flag, do not touch
                            report["missing_from_input"].append(name + " (input is " + src_dtype + ")")
                        elif list(info["shape"]) != src_shape:
                            report["mismatch"].append({"name": name,
                                                       "why": f"shape {info['shape']} vs input {src_shape}"})
                        else:
                            raw, entry, did = _absorb(raw, info, src_raw, src_dtype, src_shape, off, report, name)
                            changed |= did
                            info = entry
                            n_restored += 1
                            report["restored"][name] = hashlib.sha256(src_raw).hexdigest()
                elif (name.endswith(".conv1d.weight") and info["dtype"] in UNQUANTIZED_OUT_DTYPES
                        and name not in weight_map):
                    # the conversion's one fused conv: cat(q_conv1d, k_conv1d, v_conv1d) along rows
                    module = name[:-len(".weight")].rsplit(".", 1)[0]
                    parts = [module + "." + p + ".weight" for p in CONV_PARTS]
                    if all(p in weight_map for p in parts):
                        srcs = [input_bytes(p) for p in parts]
                        if len({s[1] for s in srcs}) != 1 or len({tuple(s[2][1:]) for s in srcs}) != 1:
                            report["mismatch"].append({"name": name, "why": "the three conv parts disagree"})
                        else:
                            rows = sum(s[2][0] for s in srcs)
                            src_shape = [rows] + list(srcs[0][2][1:])
                            if list(info["shape"]) != src_shape:
                                report["mismatch"].append({"name": name,
                                                           "why": f"shape {info['shape']} vs 3 convs {src_shape}"})
                            else:
                                src_raw = b"".join(s[0] for s in srcs)
                                raw, entry, did = _absorb(raw, info, src_raw, srcs[0][1], src_shape, off,
                                                          report, name)
                                changed |= did
                                info = entry
                                n_restored += 1
                                report["restored"][name] = hashlib.sha256(src_raw).hexdigest()

                out[name] = {"dtype": info["dtype"], "shape": list(info["shape"]),
                             "data_offsets": [off, off + len(raw)]}
                if info["dtype"] not in UNQUANTIZED_OUT_DTYPES:
                    report["skipped_quantized"] += 1
                blobs.append(raw)
                off += len(raw)
        if changed:
            tmp = shard.with_suffix(".safetensors.restore-tmp")
            tmp.write_bytes(_pack_header(meta, out) + b"".join(blobs))
            tmp.replace(shard)

    # the partition check: every BF16 input tensor is restored by name or consumed by a quantized group
    for name, (dtype, _shape) in sorted(in_meta.items()):
        if dtype == "BF16" and name not in report["restored"] and _consumed_by(name, out_names,
                                                                               report["restored"]) is None:
            report["unaccounted"].append(name)

    # verify every restored tensor's bytes against the source, reading the output back through its own map
    bad = []
    verify_heads: dict[str, tuple[dict, int]] = {}
    for name, want in sorted(report["restored"].items()):
        shard = output_dir / out_index[name]
        if shard.name not in verify_heads:
            verify_heads[shard.name] = _header(shard)
        ohead, obase = verify_heads[shard.name]
        lo, hi = ohead[name]["data_offsets"]
        if hashlib.sha256(_read_at(shard, obase, lo, hi)).hexdigest() != want:
            bad.append(name)
    report["verify_bad"] = bad
    report["restored_count"] = n_restored

    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():                          # the total_size follows the restored dtypes
        index = json.loads(index_path.read_text())
        total = 0
        for shard in out_shards:
            for info in _header(shard)[0].values():
                if isinstance(info, dict) and "data_offsets" in info:
                    lo, hi = info["data_offsets"]
                    total += hi - lo
        index["total_size"] = total
        index_path.write_text(json.dumps(index, indent = 2) + "\n")

    report["result"] = "PASS" if (not bad and not report["mismatch"] and not report["unaccounted"]
                                  and not report["missing_from_input"]) else "FAIL"

    text = json.dumps(report, indent = 1)
    if report_path:
        report_path.write_text(text)
    print(text)
    return 0 if report["result"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type = str)
    ap.add_argument("output_dir", type = str)
    ap.add_argument("--report", type = str, default = None)
    args = ap.parse_args()
    return restore_dir(Path(args.input_dir), Path(args.output_dir),
                       Path(args.report) if args.report else None)


if __name__ == "__main__":
    sys.exit(main())
