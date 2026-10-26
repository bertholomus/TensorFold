"""Ling-3.1-flash's FP8 block dequant: the one function the oracle and the EXL3 conversion both read.

The banked checkpoint stores every quantized linear as ``weight`` float8_e4m3fn [out, in] plus an fp32
``weight_scale_inv`` grid [ceil(out / 128), ceil(in / 128)], and the model's weight is

    W[r, c] = fp32(fp8[r, c]) * weight_scale_inv[r // 128, c // 128]

one fp32 multiply per element: the e4m3 code converts to fp32 exactly, so the product is rounded once and the
result does not depend on how the work is blocked. Partial blocks follow the same rule: kv_a_proj_with_mqa
[576, 4096] has a [5, 32] grid whose fifth block row covers rows 512..575 only, and the MTP layer's g_proj
[64, 4096] a [1, 32] grid. A grid of any other shape is refused, never broadcast.

usage:
  python tools/ling_fp8_dequant.py --check MODEL_DIR        every FP8 weight's grid shape, partial blocks listed
  python tools/ling_fp8_dequant.py --show MODEL_DIR NAME    one tensor's dequantized statistics
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

BLOCK = 128
SCALE_SUFFIX = "_scale_inv"           # weight -> weight_scale_inv
DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}


def grid_shape(rows: int, cols: int, block: int = BLOCK) -> tuple[int, int]:
    """The scale grid a [rows, cols] FP8 weight must carry."""

    return -(-rows // block), -(-cols // block)


def dequant(fp8: torch.Tensor, scales: torch.Tensor, *, block: int = BLOCK, dtype: torch.dtype = torch.float32,
            row_chunk: int = 8192, finite: bool = True) -> torch.Tensor:
    """fp8 [rows, cols] float8_e4m3fn with scales [ceil(rows / block), ceil(cols / block)] fp32 -> W [rows, cols].

    W is computed in fp32 (``fp32(code) * scale``, rounded once) and returned as ``dtype``: fp32 by default, or
    rounded once more to ``dtype`` (bf16 for a BF16 copy such as kv_b) after the fp32 product. ``row_chunk`` bounds
    the fp32 scratch (whole block rows at a time; the result does not depend on it). ``finite``: refuse a NaN or
    infinite result (an e4m3 NaN code, or a corrupt scale) instead of passing it on.
    """

    if fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(f"dequant: weight dtype {fp8.dtype}, expected torch.float8_e4m3fn")
    if scales.dtype != torch.float32:
        raise ValueError(f"dequant: scale dtype {scales.dtype}, expected torch.float32 (weight_scale_inv)")
    if fp8.dim() != 2 or scales.dim() != 2:
        raise ValueError(f"dequant: expected 2-D tensors, got {tuple(fp8.shape)} and {tuple(scales.shape)}")
    rows, cols = fp8.shape
    want = grid_shape(rows, cols, block)
    if tuple(scales.shape) != want:
        raise ValueError(f"dequant: a [{rows}, {cols}] weight needs a {list(want)} scale grid at block {block}, "
                         f"got {list(scales.shape)}")
    step = max(block, row_chunk // block * block)
    out = torch.empty((rows, cols), dtype=dtype, device=fp8.device)
    # the grid's columns repeated over each block's 128 columns once (the last block cropped to the weight)
    col_scale = scales.repeat_interleave(block, dim=1)[:, :cols]                     # [grid rows, cols] fp32
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        s = col_scale[r0 // block:-(-r1 // block)].repeat_interleave(block, dim=0)[:r1 - r0]
        w = fp8[r0:r1].to(torch.float32) * s                                         # one fp32 multiply a value
        if finite and not torch.isfinite(w).all():
            bad = (~torch.isfinite(w)).nonzero()[0].tolist()
            raise ValueError(f"dequant: non-finite value at row {r0 + bad[0]}, column {bad[1]}")
        out[r0:r1] = w if dtype == torch.float32 else w.to(dtype)
    return out


# -- reading the banked shards (header parsing only: no safetensors dependency) -----------------------------
def _header(file: Path) -> tuple[dict, int]:
    with file.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _index(model_dir: Path) -> dict[str, str]:
    idx = model_dir / "model.safetensors.index.json"
    if idx.is_file():
        return json.loads(idx.read_text())["weight_map"]
    single = model_dir / "model.safetensors"
    head, _ = _header(single)
    return {k: single.name for k in head if k != "__metadata__"}


def read_tensor(model_dir: str | Path, name: str) -> torch.Tensor:
    """One tensor of a checkpoint folder (sharded or single-file), on the CPU."""

    model_dir = Path(model_dir)
    file = model_dir / _index(model_dir)[name]
    head, base = _header(file)
    info = head[name]
    dt = DTYPES.get(info["dtype"])
    if dt is None:
        raise ValueError(f"{name}: dtype {info['dtype']} is not one this tool reads")
    lo, hi = info["data_offsets"]
    with file.open("rb") as f:
        f.seek(base + lo)
        raw = bytearray(f.read(hi - lo))
    if dt == torch.float8_e4m3fn:
        return torch.frombuffer(raw, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(info["shape"])
    return torch.frombuffer(raw, dtype=dt).reshape(info["shape"]) if raw else torch.empty(info["shape"], dtype=dt)


def load(model_dir: str | Path, name: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``name`` (a ``...weight`` key) as the model's weight: dequantized when it is FP8, else as stored."""

    w = read_tensor(model_dir, name)
    if w.dtype != torch.float8_e4m3fn:
        return w.to(dtype)
    return dequant(w, read_tensor(model_dir, name + SCALE_SUFFIX), dtype=dtype)


def check(model_dir: str | Path) -> dict:
    """Every FP8 weight of the checkpoint has a grid of the right shape; partial blocks counted and listed."""

    model_dir = Path(model_dir)
    index = _index(model_dir)
    shapes: dict[str, tuple[str, list[int]]] = {}
    for file in sorted(set(index.values())):
        head, _ = _header(model_dir / file)
        for k, info in head.items():
            if k != "__metadata__":
                shapes[k] = (info["dtype"], info["shape"])
    fp8 = [k for k, (dt, _) in shapes.items() if dt == "F8_E4M3"]
    bad, partial, orphans = [], {}, []
    for k in fp8:
        rows, cols = shapes[k][1]
        g = shapes.get(k + SCALE_SUFFIX)
        want = list(grid_shape(rows, cols))
        if g is None or g[0] != "F32" or g[1] != want:
            bad.append((k, want, g))
            continue
        if rows % BLOCK or cols % BLOCK:
            pattern = k.split(".layers.")[-1].split(".", 1)[-1] if ".layers." in k else k
            partial.setdefault(f"{pattern} {rows}x{cols} grid {want}", []).append(k)
    for k, (dt, _) in shapes.items():
        if k.endswith(SCALE_SUFFIX) and k[:-len(SCALE_SUFFIX)] not in shapes:
            orphans.append(k)
    return {"fp8": len(fp8), "bad": bad, "partial": {p: len(v) for p, v in partial.items()}, "orphans": orphans}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", metavar="MODEL_DIR")
    ap.add_argument("--show", nargs=2, metavar=("MODEL_DIR", "NAME"))
    args = ap.parse_args(argv)
    if args.check:
        rep = check(args.check)
        print(f"FP8 weights: {rep['fp8']}; grid problems: {len(rep['bad'])}; orphan grids: {len(rep['orphans'])}")
        for pattern, n in sorted(rep["partial"].items()):
            print(f"  partial blocks: {pattern} x {n}")
        for k, want, got in rep["bad"][:20]:
            print(f"  BAD {k}: want F32 {want}, got {got}")
        return 1 if rep["bad"] or rep["orphans"] else 0
    if args.show:
        d, name = args.show
        w = load(d, name if name.endswith(".weight") else name + ".weight")
        print(f"{name}: {tuple(w.shape)} fp32 min {w.min().item():.6g} max {w.max().item():.6g} "
              f"mean {w.mean().item():.6g} abs-mean {w.abs().mean().item():.6g}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
