#!/usr/bin/env python3
"""Write a BF16 copy of the FP8 checkpoint for the EXL3 conversion: ling_fp8_write_bf16.py FP8_DIR BF16_DIR.

Every FP8 weight is written dequanted to bf16 (the fp32 product rounded once, via ling_fp8_dequant.load);
every other tensor is copied byte for byte; the *_scale_inv tensors are dropped; config.json loses its
quantization_config; the shard layout and the index keep their names (the index's total_size is recomputed).
"""
from __future__ import annotations
import json, shutil, struct, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ling_fp8_dequant import SCALE_SUFFIX, _header, load

def _pack_header(meta: dict, tensors: dict) -> bytes:
    body = json.dumps({"__metadata__": meta, **tensors}, separators=(",", ":")).encode("utf-8")
    body += b" " * ((8 - len(body) % 8) % 8)
    return struct.pack("<Q", len(body)) + body

def write_shard(src_file: Path, dst_file: Path, names: list[str]) -> int:
    """One shard: FP8 weights -> bf16 via load(); everything else byte-copied. Returns tensor bytes written."""
    head, base = _header(src_file)
    meta = head.get("__metadata__", {})
    order = sorted(names, key=lambda n: head[n]["data_offsets"][0])
    out, blobs, off, total = {}, [], 0, 0
    with src_file.open("rb") as f:
        for name in order:
            info = head[name]
            dt, shape = info["dtype"], list(info["shape"])
            lo, hi = info["data_offsets"]
            f.seek(base + lo)
            if dt == "F8_E4M3":
                w = load(src_file.parent, name, dtype=torch.bfloat16)
                raw = w.contiguous().view(torch.uint16).numpy().tobytes()
                odt = "BF16"
            else:
                raw = f.read(hi - lo)
                odt = dt
            out[name] = {"dtype": odt, "shape": shape, "data_offsets": [off, off + len(raw)]}
            blobs.append(raw); off += len(raw); total += len(raw)
    dst_file.write_bytes(_pack_header(meta, out) + b"".join(blobs))
    return total

def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    index = json.loads((src / "model.safetensors.index.json").read_text())
    keep = {k: v for k, v in index["weight_map"].items() if not k.endswith(SCALE_SUFFIX)}
    dst.mkdir(parents=True, exist_ok=True)
    total, shards = 0, sorted(set(keep.values()))
    for i, shard in enumerate(shards, 1):
        names = sorted(k for k, v in keep.items() if v == shard)
        n = write_shard(src / shard, dst / shard, names)
        total += n
        print(f"[{i}/{len(shards)}] {shard}: {len(names)} tensors, {n/2**30:.2f} GiB", flush=True)
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": keep}, indent=2))
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    skip = {p.name for p in dst.iterdir()} | {"model.safetensors.index.json"}
    for p in src.iterdir():
        if p.is_file() and p.name not in skip and not p.name.startswith("model-"):
            shutil.copyfile(p, dst / p.name)
    print(f"DONE: {len(shards)} shards, {total/2**30:.1f} GiB tensor bytes", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
