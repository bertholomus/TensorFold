"""The kernels CP4 adds (DESIGN-CP4 §3): the indexer's scores over a rank's key prefix, sparse MLA over a query's selected
fp8 rows, and the indexer's wq_b read from its fp8 bytes (§10). Patterns from deepseek_v41/cuda/kernels.py (_index_score_rows, _sparse_attn_part); that file's own
kernels stay as they are, ML4 differs in three ways that matter: no compressed rows, K != V (absorbed latents, so the
row is the 512-wide latent and the kernel carries o_lat), and fp16 where DS used bf16 (values fit, and 3 more bits).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

SCORE_QB = int(os.environ.get("TF_ML4_CP_SCORE_QB") or 64)     # queries a program shares a key tile with
SCORE_BN = int(os.environ.get("TF_ML4_CP_SCORE_BN") or 128)
SCORE_WARPS = int(os.environ.get("TF_ML4_CP_SCORE_WARPS") or 8)
ROW_QUERIES = 4                                                # up to here: one query a program (decode)
ATTN_HB = int(os.environ.get("TF_ML4_CP_ATTN_HB") or 64)       # heads a program: a row is read and widened once for all of them
ATTN_BN = int(os.environ.get("TF_ML4_CP_ATTN_BN") or 64)
ATTN_WARPS = int(os.environ.get("TF_ML4_CP_ATTN_WARPS") or 8)
ATTN = os.environ.get("TF_ML4_CP_ATTN") or "hand"               # "hand": mla_rows.cu; "triton": _sparse_mla
SCORE8 = int(os.environ.get("TF_ML4_CP_SCORE8", "256"))        # a part's queries from which the indexer runs in e4m3 (0: never)
SCORE4_ROW = 80                                                # bytes of a row in FP4, score4.cu's
FP8_STAGES = int(os.environ.get("TF_ML4_FP8_STAGES") or 4)      # few rows: a tile's bytes are few, so more of them in flight
FP8_FEW = 16                                                   # up to here: many programs, each weight byte read once
FP8_WIDE = int(os.environ.get("TF_ML4_CP_FP8_WIDE") or 256)    # from here: the weight widened to fp16 for cuBLAS; 0: never
FP8_BM = int(os.environ.get("TF_ML4_CP_FP8_BM") or 64)
FP8_BN = int(os.environ.get("TF_ML4_CP_FP8_BN") or 128)
FP8_WARPS = int(os.environ.get("TF_ML4_CP_FP8_WARPS") or 4)


@triton.jit
def _score_tile(Q, K, Wt, VIS, OUT, n, rows, IH: tl.constexpr, ID: tl.constexpr, QB: tl.constexpr, BN: tl.constexpr):
    """score[r, t] = sum_h w[r, h] relu(q[r, h] . k[t]), -inf at t >= vis[r]. One key tile stays put while the heads go
    by: QB x BN outputs accumulate in registers, so nothing is reduced across lanes and the heads cost only their dots."""

    r = tl.program_id(0) * QB + tl.arange(0, QB)
    t = tl.program_id(1) * BN + tl.arange(0, BN)
    rok = r < rows
    tok = t < n
    d = tl.arange(0, ID)
    r64 = r.to(tl.int64)
    k = tl.load(K + t[:, None].to(tl.int64) * ID + d[None, :], mask=tok[:, None], other=0.0)
    kt = tl.trans(k)
    acc = tl.zeros((QB, BN), dtype=tl.float32)
    for h in range(IH):
        q = tl.load(Q + (r64[:, None] * IH + h) * ID + d[None, :], mask=rok[:, None], other=0.0)
        w = tl.load(Wt + r64 * IH + h, mask=rok, other=0.0)
        acc += tl.maximum(tl.dot(q, kt), 0.0) * w[:, None]
    vis = tl.load(VIS + r, mask=rok, other=0)
    acc = tl.where(t[None, :] < vis[:, None], acc, float("-inf"))
    tl.store(OUT + r64[:, None] * n + t[None, :], acc, mask=rok[:, None] & tok[None, :])


@triton.jit
def _score_row(Q, K, Wt, VIS, OUT, n, IH: tl.constexpr, ID: tl.constexpr, BN: tl.constexpr):
    """_score_tile for a decode row: the heads are the dot's rows, the keys stream through once."""

    r = tl.program_id(0)
    t = tl.program_id(1) * BN + tl.arange(0, BN)
    tok = t < n
    d = tl.arange(0, ID)
    h = tl.arange(0, IH)
    r64 = r.to(tl.int64)
    q = tl.load(Q + (r64 * IH + h[:, None]) * ID + d[None, :])
    k = tl.load(K + t[:, None].to(tl.int64) * ID + d[None, :], mask=tok[:, None], other=0.0)
    w = tl.load(Wt + r64 * IH + h)
    s = tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * w[:, None], axis=0)
    s = tl.where(t < tl.load(VIS + r), s, float("-inf"))
    tl.store(OUT + r64 * n + t, s, mask=tok)


def index_scores(q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor, vis: torch.Tensor, n: int,
                 out: torch.Tensor) -> torch.Tensor:
    """q [R, IH, ID] fp16, w [R, IH] fp32, keys [>= n, ID] fp16, vis [R] int32 -> out [R, n] fp32."""

    rows, ih, idim = q.shape
    if rows <= ROW_QUERIES:
        _score_row[(rows, triton.cdiv(n, 128))](q, keys, w, vis, out, n, IH=ih, ID=idim, BN=128, num_warps=4)
    else:
        grid = (triton.cdiv(rows, SCORE_QB), triton.cdiv(n, SCORE_BN))
        _score_tile[grid](q, keys, w, vis, out, n, rows, IH=ih, ID=idim, QB=SCORE_QB, BN=SCORE_BN,
                          num_warps=SCORE_WARPS)
    return out


@lru_cache(maxsize=1)
def _score8():
    from tensorfold.cuda.build import load

    return load(name="tensorfold_ml4_score8_v1", sources=[str(Path(__file__).parent / "score8.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def score8_ready() -> None:
    """Build the kernels now: a round should not wait on nvcc."""

    _score8()


def score8_query(q: torch.Tensor, w: torch.Tensor) -> tuple:
    """q [R, IH, ID] fp16, w [R, IH] fp32 -> (q in e4m3 as bytes, [IH, >= R] w times a head's scale)."""

    rows, ih, _ = q.shape
    q8 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
    qw = torch.empty((ih, (rows + 3) // 4 * 4), dtype=torch.float32, device=q.device)
    _score8().quant_q(q, w, q8, qw)
    return q8, qw


def score8(q8: torch.Tensor, qw: torch.Tensor, first: int, keys: torch.Tensor, vis: torch.Tensor, n: int, out: torch.Tensor,
           top: torch.Tensor) -> None:
    """index_scores in e4m3 of out.shape[0] queries from ``first`` (a multiple of 64) of score8_query's q8 and qw: out [R, >= n]
    and the largest of every eight scores, top [R, >= n / 8]."""

    _score8().score8(q8, qw, first, out.shape[0], keys, vis, n, out, top)


@lru_cache(maxsize=1)
def _score4():
    from tensorfold.cuda.build import load

    return load(name="tensorfold_ml4_score4_v2", sources=[str(Path(__file__).parent / "score4.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], arch_specific=True, verbose=False)


def score4_ready() -> None:
    _score4()


def score4_rows(x: torch.Tensor, w: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [..., ID] fp16 as FP4, 80 bytes a row: [..., 80]; w [...] fp32 goes into a row's float."""

    if out is None:
        out = torch.empty((*x.shape[:-1], SCORE4_ROW), dtype=torch.uint8, device=x.device)
    _score4().quant4(x, w, out)
    return out


def score4(q4: torch.Tensor, first: int, keys: torch.Tensor, vis: torch.Tensor, n: int, out: torch.Tensor, top: torch.Tensor) -> None:
    """score8 in FP4, of out.shape[0] queries from ``first`` (a multiple of 64) of q4 [Q, IH, 80] over keys [>= n, 80]."""

    _score4().score4(q4, first, out.shape[0], keys, vis, n, out, top)


@triton.jit
def _score_rows(Q, K, Wt, BASE, VIS, OUT, n, IH: tl.constexpr, ID: tl.constexpr, BN: tl.constexpr):
    """_score_row for a decode step: row r's keys start at BASE[r], and tiles past its visible rows read nothing."""

    r = tl.program_id(0)
    t = tl.program_id(1) * BN + tl.arange(0, BN)
    vis = tl.load(VIS + r)
    tok = t < vis
    d = tl.arange(0, ID)
    h = tl.arange(0, IH)
    r64 = r.to(tl.int64)
    q = tl.load(Q + (r64 * IH + h[:, None]) * ID + d[None, :])
    at = tl.load(BASE + r) + t.to(tl.int64)
    k = tl.load(K + at[:, None] * ID + d[None, :], mask=tok[:, None], other=0.0)
    w = tl.load(Wt + r64 * IH + h)
    s = tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * w[:, None], axis=0)
    tl.store(OUT + r64 * n + t, tl.where(tok, s, float("-inf")), mask=t < n)


def index_rows(q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor, base: torch.Tensor, vis: torch.Tensor, n: int,
               out: torch.Tensor) -> torch.Tensor:
    """q [R, IH, ID] fp16, w [R, IH] fp32, keys the layer's plane, base [R] int64, vis [R] int32 -> out [R, n] fp32."""

    rows, ih, idim = q.shape
    _score_rows[(rows, triton.cdiv(n, 128))](q, keys, w, base, vis, out, n, IH=ih, ID=idim, BN=128, num_warps=4)
    return out


@triton.jit
def _small_mat(X, W, OUT, n, K: tl.constexpr, R: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[r, c] = sum_k x[r, k] w[c, k] in fp32, for a few rows: each weight byte read once. The sum over k waits until the
    end, so the loop is loads and multiplies (one tl.sum an iteration held GB10 to 67 GB/s, this reads at DRAM speed)."""

    c = tl.program_id(0) * BN + tl.arange(0, BN)
    r = tl.arange(0, R)
    acc = tl.zeros((R, BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        d = k0 + tl.arange(0, BK)
        w = tl.load(W + c[:, None].to(tl.int64) * K + d[None, :]).to(tl.float32)
        x = tl.load(X + r[:, None] * K + d[None, :])
        acc += x[:, None, :] * w[None, :, :]
    tl.store(OUT + r[:, None] * n + c[None, :], tl.sum(acc, axis=2))


def small_mat(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [R, K] fp32 @ w.T, w [N, K] bf16 or fp32, R a power of two up to 8: fp32 [R, N]."""

    rows, k = x.shape
    n = w.shape[0]
    assert k % 128 == 0 and n % 4 == 0 and w.is_contiguous() and x.is_contiguous(), f"small_mat: {x.shape} {w.shape}"
    out = torch.empty((rows, n), dtype=torch.float32, device=x.device)
    bk = next(b for b in (1024, 512, 256, 128) if k % b == 0)
    bn, warps = 2, 4                                         # best of a sweep on GB10 at K 6144, N 32 to 576; one tile keeps a row's bits at any R
    _small_mat[(n // bn,)](x, w, out, n, K=k, R=rows, BN=bn, BK=bk, num_warps=warps, num_stages=2)
    return out


@triton.jit
def _lat_part(base, g: tl.constexpr, cc, ok):
    """Latent values g*128 .. g*128+127 of a tile of rows: e4m3 codes times the group's fp32 scale, as fp16."""

    code = tl.load(base[:, None] + (g * 128 + cc)[None, :], mask=ok[:, None], other=0)
    s = tl.load((base + 512 + 4 * g).to(tl.pointer_type(tl.float32)), mask=ok, other=0.0)
    return (code.to(tl.float8e4nv, bitcast=True).to(tl.float32) * s[:, None]).to(tl.float16)


@triton.jit
def _sparse_mla(QL, QP, ROWS, CNT, PLANE, OUT, LSE, sm, ks, qh, qr, ph, pr, oh, orow, row_bytes, H: tl.constexpr,
                HB: tl.constexpr, RD: tl.constexpr, BN: tl.constexpr):
    """One query's HB heads over its selected rows on this rank, online softmax: OUT the normalized o_lat, LSE its
    log-sum-exp. Rows hold 512 e4m3 + 4 fp32 scales (+ RD fp16 k_pe). A query with no rows here: zeros and -inf."""

    hb = tl.program_id(0)
    r = tl.program_id(1).to(tl.int64)
    c = tl.load(CNT + r)
    h = hb * HB + tl.arange(0, HB)
    h64 = h.to(tl.int64)
    cc = tl.arange(0, 128)
    n = tl.arange(0, BN)
    qb = QL + h64[:, None] * qh + r * qr + cc[None, :]
    q0 = tl.load(qb)
    q1 = tl.load(qb + 128)
    q2 = tl.load(qb + 256)
    q3 = tl.load(qb + 384)
    if RD > 0:
        rr = tl.arange(0, RD)
        qp = tl.load(QP + h64[:, None] * ph + r * pr + rr[None, :])
    m_i = tl.full((HB,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((HB,), dtype=tl.float32)
    a0 = tl.zeros((HB, 128), dtype=tl.float32)
    a1 = tl.zeros((HB, 128), dtype=tl.float32)
    a2 = tl.zeros((HB, 128), dtype=tl.float32)
    a3 = tl.zeros((HB, 128), dtype=tl.float32)
    for t0 in range(0, c, BN):
        i = t0 + n
        ok = i < c
        row = tl.load(ROWS + r * ks + i, mask=ok, other=0).to(tl.int64)
        base = PLANE + row * row_bytes
        l0 = _lat_part(base, 0, cc, ok)
        l1 = _lat_part(base, 1, cc, ok)
        l2 = _lat_part(base, 2, cc, ok)
        l3 = _lat_part(base, 3, cc, ok)
        s = tl.dot(q0, tl.trans(l0))
        s = tl.dot(q1, tl.trans(l1), s)
        s = tl.dot(q2, tl.trans(l2), s)
        s = tl.dot(q3, tl.trans(l3), s)
        if RD > 0:
            kp = tl.load((base + 528).to(tl.pointer_type(tl.float16))[:, None] + rr[None, :], mask=ok[:, None], other=0.0)
            s = tl.dot(qp, tl.trans(kp), s)
        s = tl.where(ok[None, :], s * sm, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p16 = p.to(tl.float16)
        a0 = tl.dot(p16, l0, a0 * alpha[:, None])
        a1 = tl.dot(p16, l1, a1 * alpha[:, None])
        a2 = tl.dot(p16, l2, a2 * alpha[:, None])
        a3 = tl.dot(p16, l3, a3 * alpha[:, None])
        m_i = m_new
    inv = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)[:, None]
    ob = OUT + h64[:, None] * oh + r * orow + cc[None, :]
    tl.store(ob, (a0 * inv).to(tl.float16))
    tl.store(ob + 128, (a1 * inv).to(tl.float16))
    tl.store(ob + 256, (a2 * inv).to(tl.float16))
    tl.store(ob + 384, (a3 * inv).to(tl.float16))
    tl.store(LSE + r * H + h, tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf")))


@lru_cache(maxsize=1)
def _mla():
    from tensorfold.cuda.build import load

    return load(name="tensorfold_ml4_mla_v3", sources=[str(Path(__file__).parent / "mla_rows.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def sparse_mla(q_lat: torch.Tensor, q_pe: torch.Tensor | None, rows: torch.Tensor, cnt: torch.Tensor, plane: torch.Tensor,
               sm: float, out: torch.Tensor, lse: torch.Tensor) -> None:
    """q_lat [Q, H, 512] fp16 (any row and head stride, last dim dense), q_pe [Q, H, RD] fp16 or None, rows [Q, K] int32
    pool rows ascending, cnt [Q] int32, plane the layer's uint8 rows [cap, row_bytes]; out like q_lat, lse [Q, H] fp32."""

    Q, H, C = q_lat.shape
    if ATTN == "hand" and H == 64 and rows.shape[1] <= 2048:
        _mla().mla_rows(q_lat, q_pe, rows, cnt, plane, sm, out, lse)
        return
    hb = min(ATTN_HB, H)
    assert C == 512 and H % hb == 0, "the row layout is 4 x 128 latent values"
    rd = 0 if q_pe is None else q_pe.shape[-1]
    p = q_lat if q_pe is None else q_pe
    _sparse_mla[(H // hb, Q)](q_lat, p, rows, cnt, plane, out, lse, sm, rows.stride(0), q_lat.stride(1),
                              q_lat.stride(0), p.stride(1), p.stride(0), out.stride(1), out.stride(0),
                              plane.stride(0), H=H, HB=hb, RD=rd, BN=ATTN_BN, num_warps=ATTN_WARPS, num_stages=1)


# -- wq_b from its fp8 bytes (§10) ---------------------------------------------------------------------------------------

@triton.jit
def _fp8_linear(X, W, S, OUT, m, n, K: tl.constexpr, sk, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[r, c] = sum_k x[r, k] fp16(w[c, k] s[c / 128, k / 128]). A tile sits inside one 128 x 128 scale block, and the
    product rounds to the fp16 the loader's dequant_fp8_block gives, so this is the fp16 weight's matmul at half the bytes."""

    c = tl.program_id(0) * BN + tl.arange(0, BN)
    r = tl.program_id(1) * BM + tl.arange(0, BM)
    rok = r < m
    r64, c64 = r.to(tl.int64), c.to(tl.int64)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    block = (tl.program_id(0) * BN // 128) * sk
    for k0 in range(0, K, BK):
        d = k0 + tl.arange(0, BK)
        x = tl.load(X + r64[:, None] * K + d[None, :], mask=rok[:, None], other=0.0)
        w = tl.load(W + c64[:, None] * K + d[None, :])
        s = tl.load(S + block + k0 // 128)
        w16 = (w.to(tl.float32) * s).to(tl.float16)
        acc = tl.dot(x, tl.trans(w16), acc)
    tl.store(OUT + r64[:, None] * n + c64[None, :], acc.to(tl.float16), mask=rok[:, None])


@triton.jit
def _fp8_widen(W, S, OUT, K: tl.constexpr, sk):
    """A 128 x 128 block of the weight as fp16, the same rounding as ``_fp8_linear``'s."""

    c = (tl.program_id(0) * 128 + tl.arange(0, 128)).to(tl.int64)
    d = tl.program_id(1) * 128 + tl.arange(0, 128)
    s = tl.load(S + tl.program_id(0) * sk + tl.program_id(1))
    w = tl.load(W + c[:, None] * K + d[None, :])
    tl.store(OUT + c[:, None] * K + d[None, :], (w.to(tl.float32) * s).to(tl.float16))


def fp8_linear(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x [M, K] fp16 @ w.T, w e4m3 [N, K] with its fp32 grid ``scale`` [N / 128, K / 128]: fp16 [M, N]."""

    m, k = x.shape
    n = w.shape[0]
    assert n % 128 == 0 and k % 128 == 0 and x.is_contiguous() and w.is_contiguous(), f"fp8_linear: {x.shape} {w.shape}"
    if FP8_WIDE and m >= FP8_WIDE:                             # a round: the weight in fp16 once, for cuBLAS
        wide = torch.empty((n, k), dtype=torch.float16, device=x.device)
        _fp8_widen[(n // 128, k // 128)](w, scale, wide, K=k, sk=scale.stride(0), num_warps=8)
        return F.linear(x, wide)
    out = torch.empty((m, n), dtype=torch.float16, device=x.device)
    bm, bn, warps, stages = (16, 32, 4, FP8_STAGES) if m <= FP8_FEW else (FP8_BM, FP8_BN, FP8_WARPS, 2)
    _fp8_linear[(n // bn, triton.cdiv(m, bm))](x, w, scale, out, m, n, K=k, sk=scale.stride(0), BM=bm, BN=bn, BK=128,
                                               num_warps=warps, num_stages=stages)
    return out
