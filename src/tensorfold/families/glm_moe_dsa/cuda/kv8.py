"""The cache's formats (TF_GLM_KV) and its FP8 planes: a token row as e4m3 codes plus one power-of-two fp32 scale.

``TF_GLM_KV`` picks the cache format, LATENT/INDEX (every rank must agree; the engine checks): the 512-wide latent in
bf16, fp8 or exllamav3's quantized MLA format at 2-8 bits (q2 .. q8, kvq), the 128-wide indexer keys in bf16, fp8 or
q8 (kvq's format again). The 64-wide rope key stays bf16 in all. Short names:
  bf16  bf16/bf16 (the default; the shipped bits)
  idx8  bf16/fp8 (decode at depth reads every visible indexer key a step)
  fp8   fp8/fp8;   fp8b  fp8/bf16
  qN    qN/fp8;    qNb   qN/bf16

A row's bytes are a function of its bf16 values alone (``write`` serves prompt chunks, decode windows and the MTP head
alike), so drafted == serial and prompt == decode still hold. The FP8 scale is 2^(ceil(log2 amax) - SHIFT): every code
stays within e4m3's finite range without saturating, and the dequantized row (code * scale) is exact in bf16. Readers
never stage a bf16 copy of the cache: a tile's codes dequantize in registers and the bf16 kernels' own arithmetic
follows. (Not bit for bit the bf16 kernels over the dequantized rows: Triton feeds a tile that came from 8-bit loads to
tl.dot in its 8-bit operand layout, which orders the k values of a tensor-core step differently; tests/cuda/
test_glm_kv8.py holds the kernels to float64 references and prompt kernels to decode kernels bit for bit.)
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

LATENTS = ("bf16", "fp8", "q8", "q7", "q6", "q5", "q4", "q3", "q2")
INDEXES = ("bf16", "fp8", "q8")
ALIASES = {"bf16": "bf16/bf16", "idx8": "bf16/fp8", "fp8": "fp8/fp8", "fp8b": "fp8/bf16",
           **{f"q{n}": f"q{n}/fp8" for n in range(2, 9)}, **{f"q{n}b": f"q{n}/bf16" for n in range(2, 9)}}
MODE = os.environ.get("TF_GLM_KV") or "bf16"
SHIFT = 8          # |x| / scale <= 2^8 = 256 < 448, e4m3's largest finite value: no row ever saturates
SCALE_BYTES = 4    # one fp32 scale a row


def parse(mode: str) -> tuple[str, str]:
    """A cache mode (a short name or LATENT/INDEX) -> (latent format, index format)."""

    lat, _, idx = ALIASES.get(mode, mode).partition("/")
    if lat not in LATENTS or idx not in INDEXES:
        raise ValueError(f"TF_GLM_KV={mode!r}: LATENT/INDEX with LATENT one of {', '.join(LATENTS)} and INDEX one of "
                         f"{', '.join(INDEXES)}, or a short name ({', '.join(ALIASES)})")
    return lat, idx


def check_mode(mode: str) -> str:
    parse(mode)
    return mode


def mode_code(mode: str) -> int:
    """An int every rank compares at startup."""

    lat, idx = parse(mode)
    return LATENTS.index(lat) * 10 + INDEXES.index(idx)


def latent_fp8(mode: str) -> bool:
    return parse(mode)[0] == "fp8"


def latent_bits(mode: str) -> int:
    """The quantized latent's bits a value (kvq), 0 when the latent is bf16 or FP8."""

    lat = parse(mode)[0]
    return int(lat[1:]) if lat.startswith("q") else 0


def index_fp8(mode: str) -> bool:
    return parse(mode)[1] == "fp8"


def index_bits(mode: str) -> int:
    """The quantized indexer keys' bits a value (kvq), 0 when they are bf16 or FP8."""

    idx = parse(mode)[1]
    return int(idx[1:]) if idx.startswith("q") else 0


def slot_bytes(layers: int, kv_lora: int, qk_rope: int, index_planes: int, index_dim: int, mode: str) -> int:
    """Bytes a cache slot (one token) takes on a rank: ``layers`` latent + rope-key rows (the MTP head's included) and
    ``index_planes`` indexer key rows, as ``State`` allocates them in this mode."""

    from .kvq import row_bytes

    bits, ibits = latent_bits(mode), index_bits(mode)
    lat = row_bytes(kv_lora, bits) if bits else kv_lora + SCALE_BYTES if latent_fp8(mode) else 2 * kv_lora
    idx = row_bytes(index_dim, ibits) if ibits else index_dim + SCALE_BYTES if index_fp8(mode) else 2 * index_dim
    return layers * (lat + 2 * qk_rope) + index_planes * idx


class Kv8:
    """One FP8 cache plane: ``codes`` [capacity, width] uint8 (e4m3 bits) and ``scales`` [capacity] fp32."""

    def __init__(self, capacity: int, width: int, device) -> None:
        self.codes = torch.zeros((capacity, width), dtype=torch.uint8, device=device)
        self.scales = torch.ones((capacity,), dtype=torch.float32, device=device)

    @property
    def shape(self) -> torch.Size:
        return self.codes.shape

    @property
    def device(self) -> torch.device:
        return self.codes.device

    def nbytes(self) -> int:
        return self.codes.numel() + self.scales.numel() * SCALE_BYTES

    def clone(self) -> "Kv8":
        other = Kv8.__new__(Kv8)
        other.codes, other.scales = self.codes.clone(), self.scales.clone()
        return other

    def dequant(self, n: int | None = None) -> torch.Tensor:
        """Rows [0, n) as bf16 (exact: an e4m3 code times a power of two), for checks and tools."""

        n = self.codes.shape[0] if n is None else n
        x = self.codes[:n].view(torch.float8_e4m3fn).to(torch.float32) * self.scales[:n, None]
        return x.to(torch.bfloat16)

    def checksum(self, n: int) -> torch.Tensor:
        """An int64 sum over rows [0, n): codes and scale bits (the prefill A/B's cache checksum)."""

        return (self.codes[:n].to(torch.int64).sum()
                + self.scales[:n].view(torch.int32).to(torch.int64).sum())


@triton.jit
def _write(X, x_stride, C, S, POS, W: tl.constexpr, SHIFT: tl.constexpr, G: tl.constexpr = 1,
           RANK: tl.constexpr = 0):
    """Program r: row r of X (bf16) into slot POS + r: e4m3 codes of x / 2^e and the scale 2^e, e from amax's bits.
    G > 1 (dcp): position POS + r is stored at slot (POS + r) // G by rank (POS + r) % G only."""

    r = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    if G > 1:
        if (P + r) % G != RANK:
            return
        P = (P + r) // G - r
    k = tl.arange(0, W)
    x = tl.load(X + r * x_stride + k).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    bits = amax.to(tl.int32, bitcast=True)
    expo = (bits >> 23) & 0xFF
    up = ((bits & 0x7FFFFF) != 0).to(tl.int32)
    # ceil(log2 amax) - SHIFT for a normal amax, kept a normal fp32 power (a zero row: scale 1, codes 0)
    e = tl.where(expo == 0, tl.where(bits == 0, 0, -126), tl.maximum(expo - 127 + up - SHIFT, -126))
    scale = ((e + 127) << 23).to(tl.float32, bitcast=True)
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    codes = (x * inv).to(tl.float8e4nv)
    tl.store(C + (P + r) * W + k, codes.to(tl.uint8, bitcast=True))
    tl.store(S + P + r, scale)


def write(rows: torch.Tensor, plane: Kv8, pos: torch.Tensor, G: int = 1, rank: int = 0) -> None:
    """rows [R, width] bf16 (rows may be strided) into ``plane`` slots pos .. pos + R - 1 (pos read on the device;
    G > 1: the positions this rank holds, at their dcp slots)."""

    R, W = rows.shape
    if W != plane.codes.shape[1] or rows.stride(1) != 1:
        raise ValueError(f"kv8.write: rows of {W} (unit stride) into a plane of {plane.codes.shape[1]}")
    _write[(R,)](rows, rows.stride(0), plane.codes, plane.scales, pos, W=W, SHIFT=SHIFT, G=G, RANK=rank, num_warps=4)


def quantize_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch's version of ``_write`` for checks: rows [R, W] -> (codes uint8 [R, W], scales fp32 [R])."""

    xf = x.to(torch.float32)
    amax = xf.abs().amax(dim=1)
    bits = amax.view(torch.int32)
    expo = (bits >> 23) & 0xFF
    up = ((bits & 0x7FFFFF) != 0).to(torch.int32)
    e = torch.where(expo == 0, torch.where(bits == 0, torch.zeros_like(bits), torch.full_like(bits, -126)),
                    torch.clamp(expo - 127 + up - SHIFT, min=-126))
    scale = ((e + 127) << 23).view(torch.float32)
    inv = ((127 - e) << 23).view(torch.float32)
    codes = (xf * inv[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    return codes, scale
