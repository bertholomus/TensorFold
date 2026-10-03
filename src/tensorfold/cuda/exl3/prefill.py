"""EXL3 prompt matmuls: W_q decoded once a chunk into a fixed-tile fp16 GEMM whose epilogue rotates each 128-column
block, or (``deq``) the weight decoded once a chunk into the model's basis and multiplied as a plain GEMM."""

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


# tiles by shape (matmul's by_shape, default TF_EXL3_PREFILL_TILES=1; GB10, tools/bench_prefill_gemm.py at 1,024 and
# 2,048 rows: 64-row programs for outputs up to 1,024 wide, 49 -> 70 TFLOPS on GLM-5.3's shared expert gate/up; a
# 64-wide K step elsewhere, 0-5 %) or 128 / 32 / 8 / 4 for every shape (the default). Either way the shape's alone: a
# row never depends on its chunk.
TILES_BY_SHAPE = os.environ.get("TF_EXL3_PREFILL_TILES", "0") == "1"


def tiles(k: int, n: int, by_shape: bool | None = None) -> tuple[int, int, int, int, int]:
    """(rows a program, K step, warps, stages, row blocks a raster group): the shape's alone, so a row never depends on its chunk."""

    if not (TILES_BY_SHAPE if by_shape is None else by_shape):
        return 128, 32, 8, 4, 8
    if n <= 1024:
        return 64, 64, 4, 4, 8
    return 128, 64, 8, 3, 8


# The dequantized mode (matmul's ``deq``, default TF_EXL3_PREFILL_DEQ=0): y = x @ W with W = diag(suh) H W_q H diag(svh)
# / 128 (H the 128-block Hadamard on each side), made once a chunk from the decoded W_q in place (_dequant: fp32 sums,
# one rounding), so neither the rows' rotation (rot_in: a pass over every row of every input, 1.4 s of a 32.5k prefill
# on TP4) nor the epilogue's H128 runs. "bf16": W in bf16 and a cuBLAS bf16 GEMM (x as it is); "fp16": W in fp16 and
# _gemm_deq (x converted in registers); "auto": each call's faster one, bf16 outputs on cuBLAS bf16, fp32 outputs (rank
# partials) on _gemm_deq fp16 (one GB10, M 4,096: q_a 1.08 vs 1.25 ms, o_proj fp32 out 2.27 vs 2.49 with cuBLAS'
# out_dtype; tools/bench_prefill_deq.py). Other bits than the rotated path's (sums in another basis and order).
DEQ = os.environ.get("TF_EXL3_PREFILL_DEQ") or "0"
DEQ_SCALE = 0.0078125                     # (1 / sqrt(128)) ** 2


@triton.jit
def _dequant(W, OUT, H, SUH, SVH, N: tl.constexpr, SCALE: tl.constexpr):
    """OUT[kb, nb] = suh[k] (H W_q[kb, nb] H)[k, n] svh[n] SCALE for one 128 x 128 block (OUT may alias W: a program
    reads its whole block before it writes): H W_q exact products in fp32 sums, then H on the right in two bf16
    halves (the rotated epilogue's split), one rounding to OUT's type."""

    kb = tl.program_id(0)
    nb = tl.program_id(1)
    i = tl.arange(0, 128)
    rn = nb * 128 + i
    w = tl.load(W + (kb * 128 + i)[:, None].to(tl.int64) * N + rn[None, :])     # fp16, the decoded trellis values
    h = tl.load(H + i[:, None] * 128 + i[None, :])         # bf16 +-1
    sv = tl.load(SVH + rn).to(tl.float32)
    for half in tl.static_range(2):                        # 64 output rows at a time (registers); w is loaded whole
        j = half * 64 + tl.arange(0, 64)
        hk = tl.load(H + j[:, None] * 128 + i[None, :])    # H's rows j: (H W_q)'s rows j
        t = tl.dot(hk.to(tl.float16), w)
        top = t.to(tl.bfloat16)
        rest = (t - top.to(tl.float32)).to(tl.bfloat16)
        u = tl.dot(rest, h, tl.dot(top, h))
        rk = kb * 128 + j
        su = tl.load(SUH + rk).to(tl.float32) * SCALE
        tl.store(OUT + rk[:, None].to(tl.int64) * N + rn[None, :],
                 (u * su[:, None] * sv[None, :]).to(OUT.dtype.element_ty))


@triton.jit(do_not_specialize=["M"])
def _gemm_deq(X, x_stride, W, BIAS, OUT, M, o_stride, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr):
    """OUT[m, n] = x[m] @ W[:, n] (+ bias): x converted to W's type in registers, fp32 sums over K in BK steps."""

    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    per = GROUP * (N // BN)
    first = (pid // per) * GROUP
    rows = tl.minimum(nm - first, GROUP)
    pm = first + (pid % per) % rows
    pn = (pid % per) // rows
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ok = rm < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None].to(tl.int64) * x_stride + (k0 + rk)[None, :], mask=ok[:, None], other=0.0)
        w = tl.load(W + (k0 + rk)[:, None].to(tl.int64) * N + rn[None, :])
        acc = tl.dot(x.to(W.dtype.element_ty), w, acc)
    if HAS_BIAS:
        acc += tl.load(BIAS + rn).to(tl.float32)[None, :]
    tl.store(OUT + rm[:, None].to(tl.int64) * o_stride + rn[None, :], acc.to(OUT.dtype.element_ty), mask=ok[:, None])


def deq_tiles(k: int, n: int) -> tuple[int, int, int, int, int, int]:
    """_gemm_deq's (rows a program, columns, K step, warps, stages, row blocks a raster group): the shape's alone (one
    GB10 at M 4,096: 128 x 128 programs, 32-deep K steps, 4 warps, 4 stages the fastest of five on every GLM shape)."""

    return 128, 128, 32, 4, 4, 8


_OUT_DTYPE: list = []                     # whether torch.mm takes out_dtype (fp32 sums of bf16 inputs written as fp32)


def _out_dtype_ok(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether this torch's mm takes out_dtype (fp32 sums of bf16 inputs written as fp32) with out=."""

    if not _OUT_DTYPE:
        try:
            torch.mm(x[:1], w, out_dtype=torch.float32,
                     out=torch.empty((1, w.shape[1]), dtype=torch.float32, device=x.device))
            _OUT_DTYPE.append(True)
        except (TypeError, RuntimeError):
            _OUT_DTYPE.append(False)
    return _OUT_DTYPE[0]


def _mm_f32(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    torch.mm(x, w, out_dtype=torch.float32, out=out)


class Workspace:
    """One decoded W_q and one rotated input, grown to the largest call and reused (calls run in order on one stream)."""

    def __init__(self) -> None:
        self.w: torch.Tensor | None = None
        self.xh: torch.Tensor | None = None
        self.h: torch.Tensor | None = None

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


def matmul(layer: Exl3Linear, x: torch.Tensor, out: torch.Tensor, ws: Workspace,
           by_shape: bool | None = None, deq: str | None = None, blocks=None) -> torch.Tensor:
    """out [M, N] (row stride free) = x [M, K] @ W + bias for any M, the prompt path's arithmetic (by_shape: tiles();
    deq: DEQ's modes). ``blocks``: a caller's fixed-row-block runner (fn, M, ins, outs) for the cuBLAS products, so a
    row's bits do not depend on M (cuBLAS picks its kernel by M)."""

    m, k, n = x.shape[0], layer.k, layer.n
    if x.shape[1] != k or out.shape != (m, n) or out.stride(1) != 1:
        raise ValueError(f"prefill matmul: x {tuple(x.shape)} and out {tuple(out.shape)} do not match K={k}, N={n}")
    mode = DEQ if deq is None else deq
    if mode not in ("0", "bf16", "fp16", "auto"):
        raise ValueError(f"prefill matmul: deq {mode!r} is not 0, bf16, fp16 or auto")
    if mode == "auto":
        mode = "bf16" if out.dtype == torch.bfloat16 and layer.bias is None else "fp16"
    ext = _ext()
    if mode != "0":
        wq = ws._grow("w", k * n, x.device)[:k * n].view(k, n)
        ext.unpack(layer.words, wq, *layer.strides, layer.k2, CODEBOOK_IDS[layer.codebook])
        wd = wq.view(torch.bfloat16) if mode == "bf16" else wq
        _dequant[(k // 128, n // 128)](wq, wd, ws.hadamard(x.device), layer.suh, layer.svh, N=n, SCALE=DEQ_SCALE,
                                       num_warps=8)
        if x.stride(1) != 1:
            x = x.contiguous()
        if mode == "bf16" and layer.bias is None and x.dtype == torch.bfloat16:
            if out.dtype == torch.bfloat16:
                if blocks is not None:
                    blocks(lambda a, y: torch.mm(a, wd, out=y), m, [x], [out])
                else:
                    torch.mm(x, wd, out=out)
                return out
            if out.dtype == torch.float32 and _out_dtype_ok(x, wd):
                if blocks is not None:
                    blocks(lambda a, y: _mm_f32(a, wd, y), m, [x], [out])
                else:
                    _mm_f32(x, wd, out)
                return out
        bm, bn, bk, warps, stages, group = deq_tiles(k, n)
        bias = layer.bias if layer.bias is not None else layer.svh
        _gemm_deq[(triton.cdiv(m, bm) * (n // bn),)](x, x.stride(0), wd, bias, out, m, out.stride(0), K=k, N=n, BM=bm,
                                                     BN=bn, BK=bk, GROUP=group, HAS_BIAS=layer.bias is not None,
                                                     num_warps=warps, num_stages=stages)
        return out
    xh = ws._grow("xh", m * k, x.device)[:m * k].view(m, k)
    ext.rot_in(x.contiguous(), layer.suh, xh)
    wq = ws._grow("w", k * n, x.device)[:k * n].view(k, n)
    ext.unpack(layer.words, wq, *layer.strides, layer.k2, CODEBOOK_IDS[layer.codebook])
    bm, bk, warps, stages, group = tiles(k, n, by_shape)
    bias = layer.bias if layer.bias is not None else layer.svh
    _gemm[(triton.cdiv(m, bm) * (n // BN),)](xh, wq, ws.hadamard(x.device), layer.svh, bias, out, m, out.stride(0),
                                             K=k, N=n, BM=bm, BK=bk, GROUP=group, HAS_BIAS=layer.bias is not None,
                                             SCALE=HAD_SCALE, num_warps=warps, num_stages=stages)
    return out
