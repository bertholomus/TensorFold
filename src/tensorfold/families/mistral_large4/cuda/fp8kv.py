"""The fp8-latent KV row (DESIGN-CP4 §6, D2 §6's 51,440 B/token layout): one row per writer layer and position.

    rope layer  [ 512 e4m3 latent | 16 B scales | 64 x fp16 k_pe ]   656 B
    NoPE layer  [ 512 e4m3 latent | 16 B scales ]                    528 B

The latent is the only quantized part: k_pe stays fp16 and the indexer keys live in their own fp16 plane (D2 §6:
selection is already fp16-borderline). ``SCALES`` are the 16-byte scale layouts gate 2 compares; a row is written once,
when its owner stores it, and the query attends through the stored row, so a position reads the same bits at every
later step. Pure torch: the CPU gates and the CUDA store share this file (the Triton reader decodes the same bytes).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

F32 = torch.float32
E4M3_MAX = 448.0


@dataclass(frozen=True)
class RowFormat:
    """How the 16 scale bytes split a latent of ``kv_lora`` values: ``groups`` scales of ``kind``."""

    kind: str                     # "f32" | "f16" | "e8m0"
    group: int                    # latent values one scale covers
    kv_lora: int = 512
    rd: int = 64                  # k_pe width; 0 on a NoPE layer

    @property
    def groups(self) -> int:
        return self.kv_lora // self.group

    @property
    def scale_bytes(self) -> int:
        return self.groups * {"f32": 4, "f16": 2, "e8m0": 1}[self.kind]

    @property
    def row_bytes(self) -> int:
        return self.kv_lora + self.scale_bytes + 2 * self.rd

    def nope(self) -> "RowFormat":
        return RowFormat(self.kind, self.group, self.kv_lora, 0)


#: gate 2's candidates (all 16 B of scales at kv_lora 512)
SCALES = {"f32x128": ("f32", 128), "f16x64": ("f16", 64), "e8m0x32": ("e8m0", 32)}


def row_format(name: str, kv_lora: int = 512, rd: int = 64) -> RowFormat:
    kind, group = SCALES[name]
    group = min(group, kv_lora)
    return RowFormat(kind, group, kv_lora, rd)


def _scales(amax: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(fp32 scale the values divide by, its stored bytes [n, groups * item]) for per-group ``amax``."""

    want = (amax / E4M3_MAX).clamp_min(2.0 ** -126)
    if kind == "f32":
        return want, want.contiguous().view(torch.uint8).view(want.shape[0], -1)
    if kind == "f16":
        s = want.to(torch.float16)
        s = torch.where(s.float() < want, torch.nextafter(s, torch.full_like(s, float("inf"))), s)
        s = s.clamp_min(2.0 ** -24)                         # fp16's smallest step: an all-tiny group stays finite
        return s.float(), s.contiguous().view(torch.uint8).view(s.shape[0], -1)
    if kind == "e8m0":                                      # 2 ** (byte - 127), rounded up: dequant is exact in fp16
        e = torch.ceil(torch.log2(want)).clamp(-127, 127)
        return torch.exp2(e), (e + 127).to(torch.uint8)
    raise ValueError(f"unknown scale kind {kind!r}")


def encode(lat: torch.Tensor, kpe: torch.Tensor | None, fmt: RowFormat) -> torch.Tensor:
    """fp32 latent rows [n, kv_lora] (+ k_pe [n, rd]) -> uint8 rows [n, row_bytes]."""

    n = lat.shape[0]
    x = lat.to(F32).view(n, fmt.groups, fmt.group)
    scale, sbytes = _scales(x.abs().amax(-1), fmt.kind)
    q = (x / scale[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8).view(n, -1)
    parts = [q, sbytes]
    if fmt.rd:
        parts.append(kpe.to(torch.float16).contiguous().view(torch.uint8).view(n, -1))
    return torch.cat(parts, 1)


def decode_scales(raw: torch.Tensor, fmt: RowFormat) -> torch.Tensor:
    """The stored scale bytes [n, scale_bytes] -> fp32 [n, groups]."""

    n = raw.shape[0]
    raw = raw.contiguous()
    if fmt.kind == "f32":
        return raw.view(F32).view(n, fmt.groups)
    if fmt.kind == "f16":
        return raw.view(torch.float16).view(n, fmt.groups).float()
    return torch.exp2(raw.float() - 127.0)


def decode(rows: torch.Tensor, fmt: RowFormat) -> tuple[torch.Tensor, torch.Tensor | None]:
    """uint8 rows [n, row_bytes] -> (fp32 latent [n, kv_lora], fp32 k_pe [n, rd] or None)."""

    n = rows.shape[0]
    a, b = fmt.kv_lora, fmt.kv_lora + fmt.scale_bytes
    q = rows[:, :a].contiguous().view(torch.float8_e4m3fn).to(F32).view(n, fmt.groups, fmt.group)
    lat = (q * decode_scales(rows[:, a:b], fmt)[..., None]).view(n, fmt.kv_lora)
    if not fmt.rd:
        return lat, None
    return lat, rows[:, b:].contiguous().view(torch.float16).view(n, fmt.rd).float()
