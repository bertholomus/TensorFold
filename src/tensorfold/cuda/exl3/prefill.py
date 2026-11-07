"""EXL3 prompt matmuls: W_q decoded once a chunk into a fixed-tile fp16 GEMM whose epilogue rotates each 128-column block."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .linear import CODEBOOK_IDS, Exl3Linear, _ext

HAD_SCALE = 0.08838834764831845          # 1 / sqrt(128)
BN = 128                                  # a program's columns: one Hadamard block


@triton.jit(do_not_specialize=["M"])
def _gemm(X, W, H, SVH, BIAS, OUT, M, o_stride, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
          BK: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr, SCALE: tl.constexpr):
    """OUT[m, block] = ((xh[m] @ W_q[:, block]) @ H) * SCALE * svh + bias; K in BK steps in order, a row alone."""

    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    per = GROUP * (N // 128)
    first = (pid // per) * GROUP
    rows = tl.minimum(nm - first, GROUP)
    pm = first + (pid % per) % rows
    pn = (pid % per) // rows
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * 128 + tl.arange(0, 128)
    rk = tl.arange(0, BK)
    ok = rm < M
    acc = tl.zeros((BM, 128), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=ok[:, None], other=0.0)
        w = tl.load(W + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(x, w, acc)
    hi = tl.arange(0, 128)
    h = tl.load(H + hi[:, None] * 128 + hi[None, :])
    top = acc.to(tl.bfloat16)
    rest = (acc - top.to(tl.float32)).to(tl.bfloat16)
    y = tl.dot(rest, h, tl.dot(top, h))
    y = y * SCALE * tl.load(SVH + rn).to(tl.float32)[None, :]
    if HAS_BIAS:
        y += tl.load(BIAS + rn).to(tl.float32)[None, :]
    tl.store(OUT + rm[:, None] * o_stride + rn[None, :], y.to(OUT.dtype.element_ty), mask=ok[:, None])


@triton.jit
def _rows8(XH, Q, RA, K: tl.constexpr, BLOCK: tl.constexpr, UNIT: tl.constexpr):
    """A rotated row as int8, its largest at 127, RA the factor back (mul1's int8 unit inside)."""

    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    ok = c < K
    x = tl.load(XH + r * K + c, mask=ok, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(x), axis=0)
    s = tl.where(m > 0, 127.0 / m, 0.0)
    v = x * s
    q = tl.where(v >= 0, v + 0.5, v - 0.5).to(tl.int8)
    tl.store(Q + r * K + c, q, mask=ok)
    tl.store(RA + r, m * (UNIT / 127.0))


@triton.jit(do_not_specialize=["M"])
def _gemm8(X, W, RA, H, SVH, OUT, M, o_stride, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
           BK: tl.constexpr, GROUP: tl.constexpr, SCALE: tl.constexpr):
    """_gemm in int8: exact sums over K of a row's int8 and W_q's, W_q [N, K]; the rotation back as _gemm's."""

    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    per = GROUP * (N // 128)
    first = (pid // per) * GROUP
    rows = tl.minimum(nm - first, GROUP)
    pm = first + (pid % per) % rows
    pn = (pid % per) // rows
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * 128 + tl.arange(0, 128)
    rk = tl.arange(0, BK)
    ok = rm < M
    acc = tl.zeros((BM, 128), dtype=tl.int32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=ok[:, None], other=0)
        w = tl.load(W + rn[None, :] * K + (k0 + rk)[:, None])
        acc = tl.dot(x, w, acc, out_dtype=tl.int32)
    z = acc.to(tl.float32) * tl.load(RA + rm, mask=ok, other=0.0)[:, None]
    hi = tl.arange(0, 128)
    h = tl.load(H + hi[:, None] * 128 + hi[None, :])
    top = z.to(tl.bfloat16)
    rest = (z - top.to(tl.float32)).to(tl.bfloat16)
    y = tl.dot(rest, h, tl.dot(top, h))
    y = y * SCALE * tl.load(SVH + rn).to(tl.float32)[None, :]
    tl.store(OUT + rm[:, None] * o_stride + rn[None, :], y.to(OUT.dtype.element_ty), mask=ok[:, None])


# tiles by shape (matmul's by_shape, default TF_EXL3_PREFILL_TILES=1; GB10, tools/bench_prefill_gemm.py at 1,024 and
# 2,048 rows: 64-row programs for outputs up to 1,024 wide, 49 -> 70 TFLOPS on GLM-5.3's shared expert gate/up; a
# 64-wide K step elsewhere, 0-5 %) or 128 / 64 / 8 / 4 for every shape (the default: on GB10 8-10 % over a 32-wide K
# step, same bits). Either way the shape's alone: a row never depends on its chunk.
TILES_BY_SHAPE = os.environ.get("TF_EXL3_PREFILL_TILES", "0") == "1"


def tiles(k: int, n: int, by_shape: bool | None = None) -> tuple[int, int, int, int, int]:
    """(rows a program, K step, warps, stages, row blocks a raster group): the shape's alone, so a row never depends on its chunk."""

    if not (TILES_BY_SHAPE if by_shape is None else by_shape):
        return 128, 64, 8, 4, 8
    if n <= 1024:
        return 64, 64, 4, 4, 8
    return 128, 64, 8, 3, 8


TILES8 = tuple(int(v) for v in (os.environ.get("TF_EXL3_PREFILL_TILES8") or "128,128,8,3,8").split(","))
UNIT8 = 4.0 * 0.00676727294921875         # mul1's value for 1 of its int8


class Workspace:
    """One decoded W_q and one rotated input, grown to the largest call and reused (calls run in order on one stream)."""

    def __init__(self) -> None:
        self.w: torch.Tensor | None = None
        self.xh: torch.Tensor | None = None
        self.h: torch.Tensor | None = None
        self.w8: dict = {}                    # a layer's int8 W_q by its shape, while the next call may want it again
        self.kept: dict = {}                  # the same in fp16
        self.q8: torch.Tensor | None = None
        self.ra: torch.Tensor | None = None

    def _grow(self, name: str, numel: int, device) -> torch.Tensor:
        t = getattr(self, name)
        if t is None or t.numel() < numel:
            t = torch.empty((numel,), dtype=torch.float16, device=device)
            setattr(self, name, t)
        return t

    def hadamard(self, device) -> torch.Tensor:
        if self.h is None:
            i = torch.arange(128, device=device)
            parity = torch.tensor([bin(v).count("1") & 1 for v in range(128)], device=device)[i[:, None] & i[None, :]]
            self.h = (1.0 - 2.0 * parity.float()).to(torch.bfloat16).contiguous()
        return self.h

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w, self.xh, self.h) if t is not None)


def matmul8(layer: Exl3Linear, x: torch.Tensor, out: torch.Tensor, ws: Workspace) -> torch.Tensor:
    """matmul in int8, mul1 only: rows and W_q as the experts' mma8 rounds them. W_q stays decoded for the next round of its layer."""

    m, k, n = x.shape[0], layer.k, layer.n
    if layer.codebook != "mul1" or layer.bias is not None:
        raise ValueError(f"prefill matmul8: {layer.codebook}, bias {layer.bias is not None}")
    ext = _ext()
    xh = ws._grow("xh", m * k, x.device)[:m * k].view(m, k)
    ext.rot_in(x.contiguous(), layer.suh, xh)
    if ws.q8 is None or ws.q8.numel() < m * k:
        ws.q8 = torch.empty((m * k,), dtype=torch.int8, device=x.device)
    if ws.ra is None or ws.ra.numel() < m:
        ws.ra = torch.empty((m,), dtype=torch.float32, device=x.device)
    q8 = ws.q8[:m * k].view(m, k)
    _rows8[(m,)](xh, q8, ws.ra, K=k, BLOCK=triton.next_power_of_2(k), UNIT=UNIT8, num_warps=8)
    held = ws.w8.get((k, n))
    if held is None or held[0] is not layer:
        w8 = held[1] if held is not None else torch.empty((n, k), dtype=torch.int8, device=x.device)
        ext.unpack8(layer.words, w8, *layer.strides, layer.k2)
        ws.w8[k, n] = held = (layer, w8)
    bm, bk, warps, stages, group = TILES8
    _gemm8[(triton.cdiv(m, bm) * (n // BN),)](q8, held[1], ws.ra, ws.hadamard(x.device), layer.svh, out, m, out.stride(0),
                                              K=k, N=n, BM=bm, BK=bk, GROUP=group, SCALE=HAD_SCALE, num_warps=warps,
                                              num_stages=stages)
    return out


def matmul(layer: Exl3Linear, x: torch.Tensor, out: torch.Tensor, ws: Workspace,
           by_shape: bool | None = None, keep: bool = False) -> torch.Tensor:
    """out [M, N] (row stride free) = x [M, K] @ W + bias for any M, the prompt path's arithmetic (by_shape: tiles()). keep: W_q stays
    decoded for the next call of this layer, one buffer a shape."""

    m, k, n = x.shape[0], layer.k, layer.n
    if x.shape[1] != k or out.shape != (m, n) or out.stride(1) != 1:
        raise ValueError(f"prefill matmul: x {tuple(x.shape)} and out {tuple(out.shape)} do not match K={k}, N={n}")
    ext = _ext()
    xh = ws._grow("xh", m * k, x.device)[:m * k].view(m, k)
    ext.rot_in(x.contiguous(), layer.suh, xh)
    held = ws.kept.get((k, n)) if keep else None
    if held is not None and held[0] is layer:
        wq = held[1]
    else:
        wq = held[1] if held is not None else (torch.empty((k, n), dtype=torch.float16, device=x.device) if keep else
                                               ws._grow("w", k * n, x.device)[:k * n].view(k, n))
        ext.unpack(layer.words, wq, *layer.strides, layer.k2, CODEBOOK_IDS[layer.codebook])
        if keep:
            ws.kept[k, n] = (layer, wq)
    bm, bk, warps, stages, group = tiles(k, n, by_shape)
    bias = layer.bias if layer.bias is not None else layer.svh
    _gemm[(triton.cdiv(m, bm) * (n // BN),)](xh, wq, ws.hadamard(x.device), layer.svh, bias, out, m, out.stride(0),
                                             K=k, N=n, BM=bm, BK=bk, GROUP=group, HAS_BIAS=layer.bias is not None,
                                             SCALE=HAD_SCALE, num_warps=warps, num_stages=stages)
    return out
