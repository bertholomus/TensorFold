"""ExLlamaV3's quantized MLA cache format for GLM-5.3's latent and indexer keys (TF_GLM_KV=qN, LAT/q8; kv8.parse).

The format is exllamav3's ``CacheLayer_MLA_quant`` (exllamav3/cache/mla.py, exllamav3_ext/cache/q_cache_kernels.cuh,
modules/attention_fn/triton_paged.py; MIT, (c) turboderp): a token's 512-wide latent in 16 groups of 32 values, each
group rotated by the orthonormal Hadamard H32 (it pulls a group toward a Gaussian, which is what lets a few bits do), its
absmax kept as an fp16 scale, the values on the midpoint grid ((2q + 1) / 2^bits - 1) * absmax, and the codes packed
in power-of-two bit planes (8 / 4 / 2 / 1; 6 bits = a 4-bit plane then a 2-bit one): value j of a plane of W bits sits
at bits [j W, (j + 1) W) of the plane's W words, a group's QB words holding its planes widest first.
Per token and layer: 64 * bits bytes of codes + 32 of scales (+ the 128-byte bf16 rope key, kept as it is).

Readers never rotate a key back: a query is rotated once (H32 is symmetric and orthonormal, so q . k = H q . H k), the
softmax-weighted sum of rotated values is rotated back once per chunk, and the tiles in between are the codes times
their scale in fp16 (exllamav3's online dequantization; fp16 keeps the 8-bit grid exactly where bf16 would round it).
Here the rotations are fp32 sums of +-1 terms (an IEEE dot) times 1/sqrt(32), and a row is quantized from its bf16
values (exllamav3 quantizes fp16 rows in CUDA); the scale a group uses to quantize is the fp16 one it stores.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

GROUP = 32
R32 = 0.17677669529663688110        # 1 / sqrt(32)


@lru_cache(maxsize=4)
def hadamard32(device) -> torch.Tensor:
    """The +-1 Sylvester Hadamard matrix of order 32 (fp32; H H = 32 I)."""

    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < GROUP:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device).contiguous()


def row_bytes(width: int, bits: int) -> int:
    """Bytes of a token row: its packed codes and fp16 group scales."""

    return width // GROUP * (bits * 4 + 2)


class KvQ:
    """One quantized latent plane: ``codes`` [capacity, G * bits] int32 (a group's bits words: its bit planes) and
    ``scales`` [capacity, G] fp16, G = width / 32."""

    def __init__(self, capacity: int, width: int, bits: int, device) -> None:
        if width % GROUP or not 2 <= bits <= 8:
            raise ValueError(f"KvQ: a width of whole 32-value groups and 2-8 bits, not {width} x {bits}")
        self.bits, self.width = bits, width
        self.codes = torch.zeros((capacity, width // GROUP * bits), dtype=torch.int32, device=device)
        self.scales = torch.zeros((capacity, width // GROUP), dtype=torch.float16, device=device)
        self.h = hadamard32(torch.device(device) if not isinstance(device, torch.device) else device)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.codes.shape[0], self.width)

    @property
    def device(self) -> torch.device:
        return self.codes.device

    def nbytes(self) -> int:
        return self.codes.numel() * 4 + self.scales.numel() * 2

    def clone(self) -> "KvQ":
        other = KvQ.__new__(KvQ)
        other.bits, other.width, other.h = self.bits, self.width, self.h
        other.codes, other.scales = self.codes.clone(), self.scales.clone()
        return other

    def rows(self, lo: int, hi: int) -> "KvQ":
        """A plane over rows [lo, hi) of this one (views: a concurrent stream's slot of the pool)."""

        other = KvQ.__new__(KvQ)
        other.bits, other.width, other.h = self.bits, self.width, self.h
        other.codes, other.scales = self.codes[lo:hi], self.scales[lo:hi]
        return other

    def dequant(self, n: int | None = None) -> torch.Tensor:
        """Rows [0, n) back in the latent's own domain (fp32), for checks and tools."""

        n = self.codes.shape[0] if n is None else n
        return dequantize_reference(self.codes[:n], self.scales[:n], self.bits, self.width)

    def checksum(self, n: int) -> torch.Tensor:
        return (self.codes[:n].to(torch.int64).sum() + self.scales[:n].view(torch.int16).to(torch.int64).sum())


# -- writing --------------------------------------------------------------------------------------------------------
@triton.jit
def _put_plane(q, QW, base, PB: tl.constexpr, SH: tl.constexpr, W: tl.constexpr, G: tl.constexpr, QB: tl.constexpr):
    """The W-bit field at bit SH of every code [G, 32] into words [PB, PB + W) of each group's QB words at QW + base."""

    VPW: tl.constexpr = 32 // W
    f = tl.reshape((q >> SH) & ((1 << W) - 1), (G, W, VPW))
    words = tl.sum(f << (tl.arange(0, VPW) * W).to(tl.uint32)[None, None, :], axis=2)   # disjoint fields: their OR
    g = tl.arange(0, G)
    tl.store(QW + base + g[:, None] * QB + PB + tl.arange(0, W)[None, :], words.to(tl.int32, bitcast=True))


@triton.jit
def _write(X, x_stride, QW, QS, H, POS, LW: tl.constexpr, QB: tl.constexpr, DG: tl.constexpr = 1,
           RANK: tl.constexpr = 0):
    """Program r: row r of X (bf16) into slot POS + r: each group rotated by H32 / sqrt(32), its absmax as an fp16
    scale, its codes on the midpoint grid in bit planes. DG > 1 (dcp): rank (POS + r) % DG stores it at
    (POS + r) // DG."""

    r = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    if DG > 1:
        if (P + r) % DG != RANK:
            return
        P = (P + r) // DG - r
    G: tl.constexpr = LW // 32
    x = tl.load(X + r * x_stride + tl.arange(0, LW)).to(tl.float32)
    h = tl.load(H + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :])
    if G >= 16:
        v = tl.dot(tl.reshape(x, (G, 32)), h, input_precision="ieee") * 0.17677669529663688110
    else:                                     # a 128-wide indexer key: 4 groups, too few rows for a dot
        v = tl.sum(tl.reshape(x, (G, 32))[:, :, None] * h[None, :, :], axis=1) * 0.17677669529663688110
    s16 = (tl.max(tl.abs(v), 1) + 1e-10).to(tl.float16)
    inv = 1.0 / s16.to(tl.float32)
    MF: tl.constexpr = 1 << (QB - 1)
    q = tl.floor(v * inv[:, None] * MF + MF)
    q = tl.minimum(tl.maximum(q, 0.0), (1 << QB) - 1).to(tl.uint32)
    base = (P + r) * (G * QB)
    if QB & 8:
        _put_plane(q, QW, base, 0, QB - 8, 8, G, QB)
    if QB & 4:
        _put_plane(q, QW, base, QB & 8, QB & 3, 4, G, QB)
    if QB & 2:
        _put_plane(q, QW, base, QB & 12, QB & 1, 2, G, QB)
    if QB & 1:
        _put_plane(q, QW, base, QB & 14, 0, 1, G, QB)
    tl.store(QS + (P + r) * G + tl.arange(0, G), s16)


def write(rows: torch.Tensor, plane: KvQ, pos: torch.Tensor, G: int = 1, rank: int = 0) -> None:
    """rows [R, width] bf16 (unit column stride) into ``plane`` slots pos .. pos + R - 1 (pos read on the device;
    G > 1: the positions this rank holds, at their dcp slots)."""

    R, W = rows.shape
    if W != plane.width or rows.stride(1) != 1:
        raise ValueError(f"kvq.write: rows of {W} (unit stride) into a plane of {plane.width}")
    _write[(R,)](rows, rows.stride(0), plane.codes, plane.scales, plane.h, pos, LW=W, QB=plane.bits, DG=G,
                 RANK=rank, num_warps=4)


# -- reading (inside the attention kernels) --------------------------------------------------------------------------
@triton.jit
def _get_plane(QW, rows, ok, PB: tl.constexpr, W: tl.constexpr, G: tl.constexpr, QB: tl.constexpr,
               KT: tl.constexpr):
    """The W-bit plane of token rows ``rows`` [KT] (int64) as codes [KT, G * 32] (int32), value order."""

    VPW: tl.constexpr = 32 // W
    wi = tl.arange(0, G * W)
    cols = (wi // W) * QB + PB + (wi % W)
    w = tl.load(QW + rows[:, None] * (G * QB) + cols[None, :], mask=ok[:, None], other=0)
    nib = (w[:, :, None] >> (tl.arange(0, VPW) * W)[None, None, :]) & ((1 << W) - 1)
    return tl.reshape(nib, (KT, G * 32))


@triton.jit
def tile(QW, QS, rows, ok, LW: tl.constexpr, QB: tl.constexpr, KT: tl.constexpr):
    """Token rows [KT] of a quantized plane as fp16 values in the rotated domain [KT, LW] (0 where not ok)."""

    G: tl.constexpr = LW // 32
    if QB == 8:
        raw = _get_plane(QW, rows, ok, 0, 8, G, QB, KT)
    elif QB == 7:
        raw = ((_get_plane(QW, rows, ok, 0, 4, G, QB, KT) << 3) | (_get_plane(QW, rows, ok, 4, 2, G, QB, KT) << 1)
               | _get_plane(QW, rows, ok, 6, 1, G, QB, KT))
    elif QB == 6:
        raw = (_get_plane(QW, rows, ok, 0, 4, G, QB, KT) << 2) | _get_plane(QW, rows, ok, 4, 2, G, QB, KT)
    elif QB == 5:
        raw = (_get_plane(QW, rows, ok, 0, 4, G, QB, KT) << 1) | _get_plane(QW, rows, ok, 4, 1, G, QB, KT)
    elif QB == 4:
        raw = _get_plane(QW, rows, ok, 0, 4, G, QB, KT)
    elif QB == 3:
        raw = (_get_plane(QW, rows, ok, 0, 2, G, QB, KT) << 1) | _get_plane(QW, rows, ok, 2, 1, G, QB, KT)
    else:
        raw = _get_plane(QW, rows, ok, 0, 2, G, QB, KT)
    sc = tl.load(QS + rows[:, None] * G + tl.arange(0, G)[None, :], mask=ok[:, None], other=0.0)
    scx = tl.reshape(tl.broadcast_to(sc[:, :, None], (KT, G, 32)), (KT, LW))
    MH: tl.constexpr = (1 << (QB - 1)) - 0.5
    INV: tl.constexpr = 1.0 / (1 << (QB - 1))
    return ((raw.to(tl.float32) - MH) * (scx.to(tl.float32) * INV)).to(tl.float16)


@triton.jit
def rotate(x, H, ROWS: tl.constexpr, LW: tl.constexpr):
    """x [ROWS, LW] (fp32) times the block-diagonal H32 / sqrt(32), fp32 (its own inverse: rotates and unrotates)."""

    h = tl.load(H + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :])
    y = tl.dot(tl.reshape(x, (ROWS * (LW // 32), 32)), h, input_precision="ieee") * 0.17677669529663688110
    return tl.reshape(y, (ROWS, LW))


# -- torch references (tests and tools) ------------------------------------------------------------------------------
def quantize_reference(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """torch's version of ``_write``: rows [R, W] -> (codes int32 [R, W / 32 * bits], scales fp16 [R, W / 32])."""

    R, W = x.shape
    G = W // GROUP
    h = hadamard32(x.device).double()
    v = ((x.double().view(R, G, GROUP) @ h) * R32).float()
    s16 = (v.abs().amax(2) + 1e-10).to(torch.float16)
    m = 1 << (bits - 1)
    q = torch.floor(v * (1.0 / s16.to(torch.float32))[:, :, None] * m + m).clamp(0, (1 << bits) - 1).to(torch.int64)
    words = []
    rem = bits
    for w in (8, 4, 2, 1):
        if bits & w:
            rem -= w
            f = ((q >> rem) & ((1 << w) - 1)).view(R, G, w, GROUP // w)
            shifts = torch.arange(GROUP // w, device=x.device, dtype=torch.int64) * w
            words.append((f << shifts).sum(3))                                         # [R, G, w]
    packed = torch.cat(words, 2).view(R, G * bits)
    packed = torch.where(packed >= 1 << 31, packed - (1 << 32), packed).to(torch.int32)
    return packed, s16


def unpack_reference(codes: torch.Tensor, bits: int, width: int) -> torch.Tensor:
    """Packed rows -> codes [R, width] (int64, value order)."""

    R = codes.shape[0]
    G = width // GROUP
    words = codes.to(torch.int64).view(R, G, bits) & 0xFFFFFFFF
    q = torch.zeros((R, G, GROUP), dtype=torch.int64, device=codes.device)
    base = 0
    for w in (8, 4, 2, 1):
        if bits & w:
            ww = words[:, :, base:base + w]                                              # [R, G, w]
            shifts = torch.arange(GROUP // w, device=codes.device, dtype=torch.int64) * w
            f = (ww[:, :, :, None] >> shifts) & ((1 << w) - 1)                          # [R, G, w, VPW]
            q = (q << w) | f.reshape(R, G, GROUP)
            base += w
    return q.view(R, width)


def dequantize_reference(codes: torch.Tensor, scales: torch.Tensor, bits: int, width: int,
                         rotated: bool = False) -> torch.Tensor:
    """Packed rows -> fp32 values, in the rotated domain or (default) rotated back to the latent's own."""

    R = codes.shape[0]
    G = width // GROUP
    q = unpack_reference(codes, bits, width).to(torch.float32).view(R, G, GROUP)
    m = 1 << (bits - 1)
    v = (q - (m - 0.5)) * (scales.to(torch.float32)[:, :, None] / m)
    if rotated:
        return v.view(R, width)
    return ((v.double() @ hadamard32(codes.device).double()) * R32).float().view(R, width)
