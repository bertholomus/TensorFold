"""Row-independent Triton kernels for DeepSeek-V4.1 decode and prompt chunks.

Every kernel works on one row at a time (or a fixed tile of heads within a row), with fixed-order reductions, so a
row's bits never depend on how many rows share the call: a verify window reproduces serial decode exactly.
Rounding follows DeepSeek's reference (fp32 math, bf16 where its tensors are bf16).
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

HC_BLOCKS = 40          # fixed K split of the mHC mixing dots (a function of the shape only)
# decode / verify windows: at most this many rows take the row-invariant kernels (one row a program, split keys);
# larger calls (prompt chunks) take the tiled ones. Concurrent rounds stay within it (rounds.MAX_ROWS)
DECODE_ROWS = int(os.environ.get("TF_DS_DECODE_ROWS") or 16)


# -- mHC: mixes of the stream (for the next sublayer) + collapse with the carried pre-mix + RMSNorm ---------------
@triton.jit
def _hc_partial(X, FN, PART, WIDE: tl.constexpr, NB: tl.constexpr, SUB: tl.constexpr):
    r = tl.program_id(0)
    b = tl.program_id(1)
    KB: tl.constexpr = WIDE // NB
    m = tl.arange(0, 32)
    k = tl.arange(0, SUB)
    acc = tl.zeros((32,), dtype=tl.float32)
    ss = tl.zeros((SUB,), dtype=tl.float32)
    for t in range(KB // SUB):
        base = b * KB + t * SUB
        x = tl.load(X + r * WIDE + base + k).to(tl.float32)
        w = tl.load(FN + m[:, None] * WIDE + base + k[None, :], mask=m[:, None] < 24, other=0.0)
        acc += tl.sum(w * x[None, :], axis=1)
        ss += x * x
    tl.store(PART + (r * NB + b) * 32 + m, acc, mask=m < 24)
    tl.store(PART + (r * NB + b) * 32 + 24, tl.sum(ss, axis=0))


@triton.jit
def _hc_finish(X, PART, BASE, SCALE, PRE_IN, NW, OUT, PRE_OUT, POST, COMB, eps, hc_eps,
               D: tl.constexpr, NB: tl.constexpr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    m = tl.arange(0, 32)
    mix = tl.zeros((32,), dtype=tl.float32)
    ss = 0.0
    for b in range(NB):
        mix += tl.load(PART + (r * NB + b) * 32 + m)
        ss += tl.load(PART + (r * NB + b) * 32 + 24)
    mix = mix * (1.0 / tl.sqrt(ss / (4 * D) + eps))
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    base = tl.load(BASE + m, mask=m < 24, other=0.0)
    sv = tl.arange(0, 4)
    pre_l = tl.sum(tl.where(m[None, :] == sv[:, None], (mix * s0 + base)[None, :], 0.0), axis=1)
    post_l = tl.sum(tl.where(m[None, :] == (sv[:, None] + 4), (mix * s1 + base)[None, :], 0.0), axis=1)
    pre = 1.0 / (1.0 + tl.exp(-pre_l)) + hc_eps
    post = 2.0 * (1.0 / (1.0 + tl.exp(-post_l)))
    ii = tl.arange(0, 4)[:, None]
    jj = tl.arange(0, 4)[None, :]
    flat = 8 + ii * 4 + jj
    cl = tl.sum(tl.where(m[None, None, :] == flat[:, :, None], (mix * s2 + base)[None, None, :], 0.0), axis=2)
    cmax = tl.max(cl, axis=1)
    ce = tl.exp(cl - cmax[:, None])
    comb = ce / tl.sum(ce, axis=1)[:, None] + hc_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    for _ in range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    tl.store(PRE_OUT + r * 4 + sv, pre)
    tl.store(POST + r * 4 + sv, post)
    tl.store(COMB + r * 16 + ii * 4 + jj, comb)
    # collapse with the carried pre-mix (the previous sublayer's), then RMSNorm, BLOCK columns at a time
    p0 = tl.load(PRE_IN + r * 4 + 0)
    p1 = tl.load(PRE_IN + r * 4 + 1)
    p2 = tl.load(PRE_IN + r * 4 + 2)
    p3 = tl.load(PRE_IN + r * 4 + 3)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c0 in range(0, D, BLOCK):
        d = c0 + tl.arange(0, BLOCK)
        x0 = tl.load(X + r * (4 * D) + d).to(tl.float32)
        x1 = tl.load(X + r * (4 * D) + D + d).to(tl.float32)
        x2 = tl.load(X + r * (4 * D) + 2 * D + d).to(tl.float32)
        x3 = tl.load(X + r * (4 * D) + 3 * D + d).to(tl.float32)
        c = (((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3).to(tl.bfloat16).to(tl.float32)
        acc += c * c
    rinv = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / D + eps)
    for c0 in range(0, D, BLOCK):
        d = c0 + tl.arange(0, BLOCK)
        x0 = tl.load(X + r * (4 * D) + d).to(tl.float32)
        x1 = tl.load(X + r * (4 * D) + D + d).to(tl.float32)
        x2 = tl.load(X + r * (4 * D) + 2 * D + d).to(tl.float32)
        x3 = tl.load(X + r * (4 * D) + 3 * D + d).to(tl.float32)
        c = (((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3).to(tl.bfloat16).to(tl.float32)
        w = tl.load(NW + d).to(tl.float32)
        tl.store(OUT + r * D + d, (w * (c * rinv)).to(tl.bfloat16))


def hc_pre(h: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, pre_in: torch.Tensor,
           norm_w: torch.Tensor, eps: float, hc_eps: float, iters: int, out: torch.Tensor, pre_out: torch.Tensor,
           post: torch.Tensor, comb: torch.Tensor, part: torch.Tensor) -> None:
    """h [R, 4, D] bf16 -> out [R, D] = RMSNorm(sum_j pre_in[j] h_j); pre_out/post/comb from h's own mixes."""

    rows = h.shape[0]
    d = h.shape[-1]
    wide = 4 * d
    _hc_partial[(rows, HC_BLOCKS)](h, fn, part, WIDE=wide, NB=HC_BLOCKS, SUB=128, num_warps=2)
    _hc_finish[(rows,)](h, part, base, scale, pre_in, norm_w, out, pre_out, post, comb, eps, hc_eps, D=d,
                        NB=HC_BLOCKS, ITERS=iters, BLOCK=1024, num_warps=8)


@triton.jit
def _hc_post(G, RS, X, XOUT, POST, COMB, D: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr,
             PDL: tl.constexpr = False):
    """y = bf16(rank-ordered sum of partials); out_k = post_k y + sum_j comb[j, k] x_j (fp32) -> bf16."""

    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(G + r * D + d)
    for k in tl.static_range(1, WORLD):
        acc = acc + tl.load(G + k * RS + r * D + d)
    y = acc.to(tl.bfloat16).to(tl.float32)
    x0 = tl.load(X + r * (4 * D) + d).to(tl.float32)
    x1 = tl.load(X + r * (4 * D) + D + d).to(tl.float32)
    x2 = tl.load(X + r * (4 * D) + 2 * D + d).to(tl.float32)
    x3 = tl.load(X + r * (4 * D) + 3 * D + d).to(tl.float32)
    for s in tl.static_range(4):
        c0 = tl.load(COMB + r * 16 + 0 * 4 + s)
        c1 = tl.load(COMB + r * 16 + 1 * 4 + s)
        c2 = tl.load(COMB + r * 16 + 2 * 4 + s)
        c3 = tl.load(COMB + r * 16 + 3 * 4 + s)
        ps = tl.load(POST + r * 4 + s)
        v = ps * y + (((c0 * x0 + c1 * x1) + c2 * x2) + c3 * x3)
        tl.store(XOUT + r * (4 * D) + s * D + d, v.to(tl.bfloat16))


def hc_post(gathered: torch.Tensor, h: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
            out: torch.Tensor) -> None:
    """gathered [world, R, D] fp32 partials; h [R, 4, D] bf16 residual streams -> out [R, 4, D] (may alias h)."""

    world, rows, d = gathered.shape
    block = 1024
    _hc_post[(rows, d // block)](gathered, rows * d, h, out, post, comb, D=d, WORLD=world, BLOCK=block,
                                 num_warps=4, **_pdl())


# -- the decode round's small kernels, faster and with the same bits (SMALL_SWITCHES) --------------------------------
# Each switch replaces kernels of the decode round with ones that keep every output's arithmetic: the same per-output
# reduction trees (each tile keeps the old tile's per-row layout: a row's dot is one warp's, the same elements a lane,
# the same butterfly), the same fp32 / bf16 roundings and the same fused multiply-adds, only more programs, shared
# loads, unrolled loops and fewer launches. So a row's bits are those of 88ff611 whatever shares the launch. Every
# switch is on unless its environment variable is "0" (A/B and debugging); set_switch() flips one at run time (a graph
# keeps what it captured).
SMALL_SWITCHES = {
    "hc": os.environ.get("TF_DS_HC_FUSED", "1") != "0",          # hc_post fused into the next hc_pre's mixes, 120
                                                                 # programs a row, an unrolled finish
    "rowmm": os.environ.get("TF_DS_ROWMM2", "1") != "0",         # router / indexer weights: 4 outputs a program,
                                                                 # rows share each weight tile, unrolled K
    "glue": os.environ.get("TF_DS_ROUND_GLUE", "1") != "0",      # a round's index arithmetic once, not per layer
    "rot_q": os.environ.get("TF_DS_ROT_Q", "1") != "0",          # the q RMSNorm writes wq_b's (and the indexer
                                                                 # wq_b's) rotated input rows (no rot_many launch),
                                                                 # wq_b's epilogue applies q's RoPE (no rope launch)
    "rot_wob": os.environ.get("TF_DS_ROT_WOB", "1") != "0",      # wo_a's epilogue writes wo_b's rotated input rows
    "rot_attn": os.environ.get("TF_DS_ROT_ATTN", "1") != "0",    # the attention merge also applies the inverse RoPE
                                                                 # and writes wo_a's rotated input rows
    "idx": os.environ.get("TF_DS_INDEXER_FUSED", "1") != "0",    # the indexer's glue in four launches: fp4_qd of q,
                                                                 # the weights' rowmm + bf16 + scale, the scores with
                                                                 # the candidate mask as top-k keys, one sort + mask
    "comp": os.environ.get("TF_DS_COMP_FUSED", "1") != "0",      # the compressor's FP4 cache writes in one launch
                                                                 # each (fp4_pack + the row writes)
    "pdl": os.environ.get("TF_DS_TRITON_PDL", "1") != "0",       # the decode kernels as programmatic dependent
                                                                 # launches (griddepcontrol): each waits for the one
                                                                 # before it before reading anything it wrote
    # the verify window's small kernels (2-16 rows; c100 small-kernel patch, each bit-identical to b768469):
    "hc_split": os.environ.get("TF_DS_HC_SPLIT", "1") != "0",    # the mHC finish as two programs a row side by
                                                                 # side (the Sinkhorn | the collapse + RMSNorm),
                                                                 # the collapse's loads issued together up front
    "hc_rot": os.environ.get("TF_DS_HC_ROT", "0") != "0",        # (default off: level on the proxy; with hc_split) the attention mHC finish also
                                                                 # writes the input projections' rotated rows (no
                                                                 # rot_many launch before wq_a / wkv / compressor)
    "qkv_split": os.environ.get("TF_DS_QKV_SPLIT", "1") != "0",  # q_kv_norm's input rotations each in a program of
                                                                 # its own (the norm recomputed), beside the others
    "hc_rb": os.environ.get("TF_DS_HC_RB_ON", "1") != "0",       # the mHC mixes' rows a program by window rows
    "attn_pf": os.environ.get("TF_DS_ATTN_PF", "1") != "0",      # the decode attention's window keys into L2
                                                                 # before its PDL wait
    "hc_xpf": os.environ.get("TF_DS_HC_XPF", "1") != "0",        # the posted mHC mixes prefetch the old streams
                                                                 # (and post / comb) into L2 before their PDL wait
    "rowmm_parts": os.environ.get("TF_DS_ROWMM_PARTS_ON", "1") != "0",   # the MoE gate's matmul as chunk sums over
                                                                 # (outputs, chunks) programs, summed in K order by
                                                                 # the route kernel
    # round 2 of the verify window's small kernels (each bit-identical to b0d2fb8):
    "hc_defer": os.environ.get("TF_DS_HC_DEFER", "1") != "0",    # the mHC finish's Sinkhorn half (pre / post /
                                                                 # comb, read sublayers later) as its own kernel on a
                                                                 # side stream beside the collapse + RMSNorm (rounds.py
                                                                 # joins it before the sublayer's gather), off the
                                                                 # critical path
    "merge_reg": os.environ.get("TF_DS_MERGE_REG", "1") != "0",  # the decode attention merge keeps the head's
                                                                 # output in registers for the inverse RoPE and wo_a's
                                                                 # input rotation (one store of o, one barrier) instead
                                                                 # of storing it and reading it back twice
    "topk_fused": os.environ.get("TF_DS_TOPK_FUSED", "1") != "0",   # the indexer's selection (topk_select) in one
                                                                 # launch a row: its unique int64 keys sorted, the k
                                                                 # largest kept, their indices sorted and masked (the
                                                                 # same set as torch's top-k: a total order), instead
                                                                 # of torch.topk + _topk_finish
    "hc_dots": os.environ.get("TF_DS_HC_DOTS", "1") != "0",      # (with "hc_defer") a posted mHC step's main-
                                                                 # stream kernel is the post alone (the new streams);
                                                                 # the mixes' partial dots of them (_hc_mix_part on the
                                                                 # stored streams) run on the side stream before the
                                                                 # Sinkhorn: the collapse + RMSNorm no longer waits for
                                                                 # the mixing weights' 2 MB and the dots
}


def set_switch(name: str, on: bool) -> None:
    if name not in SMALL_SWITCHES:
        raise KeyError(name)
    SMALL_SWITCHES[name] = bool(on)


def on(name: str) -> bool:
    return SMALL_SWITCHES[name]


def _pdl() -> dict:
    """Launch options of a decode kernel that takes PDL: a programmatic dependent launch (sm_90+) when the "pdl"
    switch is on. Such a kernel runs griddepcontrol.wait before it reads or writes anything the kernel just before it
    may still touch (weights, and inputs only kernels further back write, may be read before it: every PDL kernel waits
    before it lets the next one launch, so only the immediately preceding kernel can be in flight), then lets the next
    kernel launch; launched plainly, both are no-ops."""

    return {"PDL": True, "launch_pdl": True} if SMALL_SWITCHES["pdl"] else {}


@triton.jit
def _bfly(v, e, BIT: tl.constexpr):
    o = tl.gather(v, e ^ BIT, axis=1)
    return tl.where((e & BIT) != 0, o - v, v + o)


@triton.jit
def _rot128(v, s, M: tl.constexpr):
    """EXL3's input rotation of M 128-wide blocks (v, s [M, 128] fp32: values and the layer's suh): rot128_in's bits
    (linear_common.cuh): the first butterfly as fma(v0, s0, +-(v1 s1)), then six butterflies lo + hi / lo - hi in bit
    order, then the scale."""

    e = tl.broadcast_to(tl.arange(0, 128)[None, :], (M, 128))
    p = v * s
    vo = tl.gather(v, e ^ 1, axis=1)
    so = tl.gather(s, e ^ 1, axis=1)
    po = tl.gather(p, e ^ 1, axis=1)
    v = tl.where((e & 1) != 0, tl.fma(vo, so, -p), tl.fma(v, s, po))
    v = _bfly(v, e, 2)
    v = _bfly(v, e, 4)
    v = _bfly(v, e, 8)
    v = _bfly(v, e, 16)
    v = _bfly(v, e, 32)
    v = _bfly(v, e, 64)
    return v * 0.08838834764831845


# mHC, fused: the hc_post of the sublayer just gathered written into a second stream buffer and, in the same programs,
# the mixes' partial dots of the new streams (what _hc_partial computes from them). A program takes one K block of one
# row and 8 of the 24 mixes (an [8, 128] tile keeps _hc_partial's per-mix layout: 4 columns a lane, one warp a mix);
# the first group of each block also writes the streams and the block's sum of squares (_hc_partial's 2-warp layout).
@triton.jit
def _hc_mix_rows(X, XO, G, RS, POST, COMB, PART, ws, r, b, mg, m, k, WIDE: tl.constexpr, D: tl.constexpr,
                 NB: tl.constexpr, KB: tl.constexpr, SUB: tl.constexpr, MB: tl.constexpr, WORLD: tl.constexpr,
                 POSTED: tl.constexpr):
    """One row of _hc_mix_part (the weight tiles ws already loaded)."""

    s = (b * KB) // D
    col0 = b * KB - s * D
    acc = tl.zeros((MB,), dtype=tl.float32)
    ss = tl.zeros((SUB,), dtype=tl.float32)
    if POSTED:
        c0 = tl.load(COMB + r * 16 + 0 * 4 + s)
        c1 = tl.load(COMB + r * 16 + 1 * 4 + s)
        c2 = tl.load(COMB + r * 16 + 2 * 4 + s)
        c3 = tl.load(COMB + r * 16 + 3 * 4 + s)
        ps = tl.load(POST + r * 4 + s)
    for t in tl.static_range(KB // SUB):
        d = col0 + t * SUB + k
        if POSTED:
            ya = tl.load(G + r * D + d)
            for q in tl.static_range(1, WORLD):
                ya = ya + tl.load(G + q * RS + r * D + d)
            y = ya.to(tl.bfloat16).to(tl.float32)
            x0 = tl.load(X + r * WIDE + d).to(tl.float32)
            x1 = tl.load(X + r * WIDE + D + d).to(tl.float32)
            x2 = tl.load(X + r * WIDE + 2 * D + d).to(tl.float32)
            x3 = tl.load(X + r * WIDE + 3 * D + d).to(tl.float32)
            # _hc_post's contraction, spelled out: fma(ps, y, fma(c3, x3, fma(c2, x2, fma(c0, x0, c1 x1))))
            v = tl.fma(ps, y, tl.fma(c3, x3, tl.fma(c2, x2, tl.fma(c0, x0, c1 * x1))))
            xb = v.to(tl.bfloat16)
            if mg == 0:
                tl.store(XO + r * WIDE + s * D + d, xb)
            x = xb.to(tl.float32)
        else:
            x = tl.load(X + r * WIDE + s * D + d).to(tl.float32)
        acc += tl.sum(ws[t] * x[None, :], axis=1)
        ss = tl.fma(x, x, ss)                                   # _hc_partial's fma(x, x, ss)
    tl.store(PART + (r * NB + b) * 32 + m, acc)
    if mg == 0:
        tl.store(PART + (r * NB + b) * 32 + 24, tl.sum(ss, axis=0))


# mHC, fused: the hc_post of the sublayer just gathered written into a second stream buffer and, in the same programs,
# the mixes' partial dots of the new streams (what _hc_partial computes from them). A program takes one K block of RB
# rows and 8 of the 24 mixes (an [8, 128] tile keeps _hc_partial's per-mix layout: 4 columns a lane, one warp a mix),
# its weight tiles loaded once for its rows; the first group of each block also writes the streams and the block's
# sum of squares (_hc_partial's 2-warp layout).
@triton.jit
def _hc_mix_part(X, XO, G, RS, POST, COMB, FN, PART, rows, WIDE: tl.constexpr, D: tl.constexpr, NB: tl.constexpr,
                 SUB: tl.constexpr, MB: tl.constexpr, WORLD: tl.constexpr, POSTED: tl.constexpr, RB: tl.constexpr = 1,
                 PDL: tl.constexpr = False, XPF: tl.constexpr = False):
    r0 = tl.program_id(0) * RB
    b = tl.program_id(1)
    mg = tl.program_id(2)
    KB: tl.constexpr = WIDE // NB
    m = mg * MB + tl.arange(0, MB)
    k = tl.arange(0, SUB)
    ws = ()                                  # the mixing weights first: they may load while the kernel before runs
    for t in tl.static_range(KB // SUB):
        ws = ws + (tl.load(FN + m[:, None] * WIDE + b * KB + t * SUB + k[None, :]),)
    if XPF:
        # (switch "hc_xpf", posted mixes) the old streams' lines this program reads, and its rows' post / comb, into
        # L2 beside the weights' loads: the kernel before (the gather) writes neither, so their (often DRAM) latency
        # runs with the weights' instead of after the wait (a prefetch moves no data the kernel reads)
        xs_ = (b * KB) // D
        xc0 = b * KB - xs_ * D
        li = tl.arange(0, 4 * (KB // 64))                    # 4 streams x the block's 128-byte lines
        for i in tl.static_range(RB):
            rx = tl.minimum(r0 + i, rows - 1)
            _prefetch_l2(X + rx * WIDE + (li // (KB // 64)) * D + xc0 + (li % (KB // 64)) * 64)
            _prefetch_l2(COMB + rx * 16 + tl.arange(0, 1))
            _prefetch_l2(POST + rx * 4 + tl.arange(0, 1))
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    if RB == 1:
        _hc_mix_rows(X, XO, G, RS, POST, COMB, PART, ws, r0, b, mg, m, k, WIDE, D, NB, KB, SUB, MB, WORLD, POSTED)
    else:
        for i in tl.static_range(RB):
            if r0 + i < rows:
                _hc_mix_rows(X, XO, G, RS, POST, COMB, PART, ws, r0 + i, b, mg, m, k, WIDE, D, NB, KB, SUB, MB,
                             WORLD, POSTED)


@triton.jit
def _prefetch_l2(ptrs):
    """prefetch.global.L2 of every address in ptrs (no data moved into registers, nothing computed)."""

    return tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.b32 $0, 0;", "=r,l", [ptrs], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _discard_l2(ptrs):
    """discard.global.L2 of the 128-byte line at every address in ptrs: the line leaves L2 without a write-back (its
    data is dead: nothing reads it again before it is rewritten)."""

    return tl.inline_asm_elementwise("discard.global.L2 [$1], 128; mov.b32 $0, 0;", "=r,l", [ptrs], dtype=tl.int32,
                                     is_pure=False, pack=1)


# _hc_finish's source with only the 40 partial sums' loop unrolled (adds alone: no contraction to change), so their
# loads are in flight together, and the norm weight prefetched into L2 first (its loads sit in the output loop, after
# stores they cannot pass); everything else is _hc_finish's code (its compiled collapse keeps its own fma pattern)
@triton.jit
def _hc_finish_u(X, PART, BASE, SCALE, PRE_IN, NW, OUT, PRE_OUT, POST, COMB, eps, hc_eps,
                 D: tl.constexpr, NB: tl.constexpr, ITERS: tl.constexpr, BLOCK: tl.constexpr,
                 PDL: tl.constexpr = False):
    pl = tl.arange(0, BLOCK // 8)                               # one address a 128-byte line of the bf16 weight
    _prefetch_l2(NW + tl.minimum(pl * 64, D - 1))
    _prefetch_l2(BASE + tl.arange(0, 1))
    _prefetch_l2(SCALE + tl.arange(0, 1))
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    m = tl.arange(0, 32)
    mix = tl.zeros((32,), dtype=tl.float32)
    ss = 0.0
    for b in tl.static_range(NB):
        mix += tl.load(PART + (r * NB + b) * 32 + m)
        ss += tl.load(PART + (r * NB + b) * 32 + 24)
    mix = mix * (1.0 / tl.sqrt(ss / (4 * D) + eps))
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    base = tl.load(BASE + m, mask=m < 24, other=0.0)
    sv = tl.arange(0, 4)
    pre_l = tl.sum(tl.where(m[None, :] == sv[:, None], (mix * s0 + base)[None, :], 0.0), axis=1)
    post_l = tl.sum(tl.where(m[None, :] == (sv[:, None] + 4), (mix * s1 + base)[None, :], 0.0), axis=1)
    pre = 1.0 / (1.0 + tl.exp(-pre_l)) + hc_eps
    post = 2.0 * (1.0 / (1.0 + tl.exp(-post_l)))
    ii = tl.arange(0, 4)[:, None]
    jj = tl.arange(0, 4)[None, :]
    flat = 8 + ii * 4 + jj
    cl = tl.sum(tl.where(m[None, None, :] == flat[:, :, None], (mix * s2 + base)[None, None, :], 0.0), axis=2)
    cmax = tl.max(cl, axis=1)
    ce = tl.exp(cl - cmax[:, None])
    comb = ce / tl.sum(ce, axis=1)[:, None] + hc_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    for _ in range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    tl.store(PRE_OUT + r * 4 + sv, pre)
    tl.store(POST + r * 4 + sv, post)
    tl.store(COMB + r * 16 + ii * 4 + jj, comb)
    # collapse with the carried pre-mix (the previous sublayer's), then RMSNorm, BLOCK columns at a time
    p0 = tl.load(PRE_IN + r * 4 + 0)
    p1 = tl.load(PRE_IN + r * 4 + 1)
    p2 = tl.load(PRE_IN + r * 4 + 2)
    p3 = tl.load(PRE_IN + r * 4 + 3)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c0 in range(0, D, BLOCK):
        d = c0 + tl.arange(0, BLOCK)
        x0 = tl.load(X + r * (4 * D) + d).to(tl.float32)
        x1 = tl.load(X + r * (4 * D) + D + d).to(tl.float32)
        x2 = tl.load(X + r * (4 * D) + 2 * D + d).to(tl.float32)
        x3 = tl.load(X + r * (4 * D) + 3 * D + d).to(tl.float32)
        c = (((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3).to(tl.bfloat16).to(tl.float32)
        acc += c * c
    rinv = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / D + eps)
    for c0 in range(0, D, BLOCK):
        d = c0 + tl.arange(0, BLOCK)
        x0 = tl.load(X + r * (4 * D) + d).to(tl.float32)
        x1 = tl.load(X + r * (4 * D) + D + d).to(tl.float32)
        x2 = tl.load(X + r * (4 * D) + 2 * D + d).to(tl.float32)
        x3 = tl.load(X + r * (4 * D) + 3 * D + d).to(tl.float32)
        c = (((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3).to(tl.bfloat16).to(tl.float32)
        w = tl.load(NW + d).to(tl.float32)
        tl.store(OUT + r * D + d, (w * (c * rinv)).to(tl.bfloat16))


# -- the verify window's mHC finish, faster with the same bits (switch "hc_split") -----------------------------
# _hc_finish_u's two independent halves as two programs a row (switch "hc_split"): (r, 0) the mixes' sum, the
# Sinkhorn and pre / post / comb (read by the next sublayers only), (r, 1) the collapse with the carried pre-mix and
# the RMSNorm (what the sublayer reads next), side by side instead of one after the other. Each half is _hc_finish_u's
# code with its layouts (8 warps, the same shapes: the same reduction trees); the collapse's 20 loads of the row are
# issued together up front and its collapsed rows kept for the output pass (_hc_finish_u's second pass reloads the
# same bits and repeats the same arithmetic), the same expression a column (the compiler contracts it as before).
@triton.jit
def _rot_out(OUT, r, cs, rinv, NW, S, H, part, D: tl.constexpr, BLOCK: tl.constexpr, ROT: tl.constexpr,
             RSPLIT: tl.constexpr = 1):
    """The finish's output row (w * (c * rinv) -> bf16, BLOCK columns at a time): stored (ROT False), or rotated by
    rot_many's arithmetic for an EXL3 layer reading it (suh S, rows H [R, D] fp16; ROT True), every block (RSPLIT 1)
    or block ``part`` alone (RSPLIT = D / BLOCK)."""

    NBK: tl.constexpr = BLOCK // 128
    for c in tl.static_range(D // BLOCK):
        if RSPLIT == 1 or c == part:
            d = c * BLOCK + tl.arange(0, BLOCK)
            w = tl.load(NW + d).to(tl.float32)
            y = (w * (cs[c] * rinv)).to(tl.bfloat16)
            if ROT:
                v = tl.reshape(y.to(tl.float32), (NBK, 128))
                dd = tl.reshape(d, (NBK, 128))
                sv = tl.load(S + dd).to(tl.float32)
                tl.store(H + r * D + dd, _rot128(v, sv, NBK).to(tl.float16))
            else:
                tl.store(OUT + r * D + d, y)


@triton.jit
def _hc_finish_s(X, PART, BASE, SCALE, PRE_IN, NW, OUT, PRE_OUT, POST, COMB, eps, hc_eps, S0, H0, S1, H1, S2, H2,
                 S3, H3, D: tl.constexpr, NB: tl.constexpr, ITERS: tl.constexpr, BLOCK: tl.constexpr,
                 NROT: tl.constexpr = 0, RSPLIT: tl.constexpr = 1, PDL: tl.constexpr = False,
                 SINK: tl.constexpr = True):
    """(r, 0): the Sinkhorn half; (r, 1): the collapse + RMSNorm half; (r, 2 + j), j < NROT * RSPLIT: the collapse +
    RMSNorm again (the same code, the same bits) and the output row rotated for EXL3 layer j // RSPLIT (rot_many's
    bits: what its group launch would make of the stored row; all of it, or with RSPLIT = D / BLOCK its BLOCK columns
    j % RSPLIT), side by side. SINK False (switch "hc_defer"): no Sinkhorn programs (_hc_sinkhorn runs that half), the
    grid's programs start at role 1."""

    r = tl.program_id(0)
    role = tl.program_id(1) if SINK else tl.program_id(1) + 1
    if role == 0:
        # the weights (no kernel writes them) before the PDL wait: they load while the mixes run
        m = tl.arange(0, 32)
        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        s2 = tl.load(SCALE + 2)
        base = tl.load(BASE + m, mask=m < 24, other=0.0)
        if PDL:
            gdc_wait()
            gdc_launch_dependents()
        mix = tl.zeros((32,), dtype=tl.float32)
        ss = 0.0
        for b in tl.static_range(NB):
            mix += tl.load(PART + (r * NB + b) * 32 + m)
            ss += tl.load(PART + (r * NB + b) * 32 + 24)
        mix = mix * (1.0 / tl.sqrt(ss / (4 * D) + eps))
        sv = tl.arange(0, 4)
        pre_l = tl.sum(tl.where(m[None, :] == sv[:, None], (mix * s0 + base)[None, :], 0.0), axis=1)
        post_l = tl.sum(tl.where(m[None, :] == (sv[:, None] + 4), (mix * s1 + base)[None, :], 0.0), axis=1)
        pre = 1.0 / (1.0 + tl.exp(-pre_l)) + hc_eps
        post = 2.0 * (1.0 / (1.0 + tl.exp(-post_l)))
        ii = tl.arange(0, 4)[:, None]
        jj = tl.arange(0, 4)[None, :]
        flat = 8 + ii * 4 + jj
        cl = tl.sum(tl.where(m[None, None, :] == flat[:, :, None], (mix * s2 + base)[None, None, :], 0.0), axis=2)
        cmax = tl.max(cl, axis=1)
        ce = tl.exp(cl - cmax[:, None])
        comb = ce / tl.sum(ce, axis=1)[:, None] + hc_eps
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
        for _ in range(ITERS - 1):
            comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
            comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
        tl.store(PRE_OUT + r * 4 + sv, pre)
        tl.store(POST + r * 4 + sv, post)
        tl.store(COMB + r * 16 + ii * 4 + jj, comb)
    else:
        pl = tl.arange(0, BLOCK // 8)                           # one address a 128-byte line of the bf16 weight
        _prefetch_l2(NW + tl.minimum(pl * 64, D - 1))
        ti = (role - 2) // RSPLIT                               # a rotation program's layer and block
        part = (role - 2) % RSPLIT
        if NROT > 0 and role >= 2:                              # its suh (fp16: 64 a line)
            sl = tl.minimum(pl * 64, D - 1)
            if ti == 0:
                _prefetch_l2(S0 + sl)
            elif ti == 1:
                _prefetch_l2(S1 + sl)
            elif ti == 2:
                _prefetch_l2(S2 + sl)
            elif ti == 3:
                _prefetch_l2(S3 + sl)
        # the carried pre-mix (written by the previous finish, before the mixes this kernel waits for) before the
        # PDL wait too: its load (often from DRAM) runs while the mixes do
        p0 = tl.load(PRE_IN + r * 4 + 0)
        p1 = tl.load(PRE_IN + r * 4 + 1)
        p2 = tl.load(PRE_IN + r * 4 + 2)
        p3 = tl.load(PRE_IN + r * 4 + 3)
        if PDL:
            gdc_wait()
            gdc_launch_dependents()
        xs = ()
        for c in tl.static_range(D // BLOCK):
            d = c * BLOCK + tl.arange(0, BLOCK)
            xs = xs + (tl.load(X + r * (4 * D) + d), tl.load(X + r * (4 * D) + D + d),
                       tl.load(X + r * (4 * D) + 2 * D + d), tl.load(X + r * (4 * D) + 3 * D + d))
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        cs = ()
        for c in tl.static_range(D // BLOCK):
            x0 = xs[4 * c].to(tl.float32)
            x1 = xs[4 * c + 1].to(tl.float32)
            x2 = xs[4 * c + 2].to(tl.float32)
            x3 = xs[4 * c + 3].to(tl.float32)
            cc = (((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3).to(tl.bfloat16).to(tl.float32)
            acc += cc * cc
            cs = cs + (cc,)
        rinv = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / D + eps)
        if role == 1:
            _rot_out(OUT, r, cs, rinv, NW, S0, H0, 0, D, BLOCK, False)
        elif NROT > 0 and ti == 0:
            _rot_out(OUT, r, cs, rinv, NW, S0, H0, part, D, BLOCK, True, RSPLIT)
        elif NROT > 1 and ti == 1:
            _rot_out(OUT, r, cs, rinv, NW, S1, H1, part, D, BLOCK, True, RSPLIT)
        elif NROT > 2 and ti == 2:
            _rot_out(OUT, r, cs, rinv, NW, S2, H2, part, D, BLOCK, True, RSPLIT)
        elif NROT > 3 and ti == 3:
            _rot_out(OUT, r, cs, rinv, NW, S3, H3, part, D, BLOCK, True, RSPLIT)


# (switch "hc_defer") _hc_finish_s's Sinkhorn half (its role 0) as a kernel of its own: the same code with the same
# layouts (8 warps, the same shapes: the same reduction trees, so the same bits), a program a row, launched on a side
# stream once the mixes are done (their partial sums), beside the collapse + RMSNorm on the main stream. Its outputs
# (pre, post, comb) are read sublayers later (the next finish's collapse, the next posted mixes), so the kernels after
# the finish no longer wait for it; rounds.py joins the side stream before the sublayer's gather.
@triton.jit
def _hc_sinkhorn(PART, BASE, SCALE, PRE_OUT, POST, COMB, eps, hc_eps, D: tl.constexpr, NB: tl.constexpr,
                 ITERS: tl.constexpr):
    r = tl.program_id(0)
    m = tl.arange(0, 32)
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    base = tl.load(BASE + m, mask=m < 24, other=0.0)
    mix = tl.zeros((32,), dtype=tl.float32)
    ss = 0.0
    for b in tl.static_range(NB):
        mix += tl.load(PART + (r * NB + b) * 32 + m)
        ss += tl.load(PART + (r * NB + b) * 32 + 24)
    mix = mix * (1.0 / tl.sqrt(ss / (4 * D) + eps))
    sv = tl.arange(0, 4)
    pre_l = tl.sum(tl.where(m[None, :] == sv[:, None], (mix * s0 + base)[None, :], 0.0), axis=1)
    post_l = tl.sum(tl.where(m[None, :] == (sv[:, None] + 4), (mix * s1 + base)[None, :], 0.0), axis=1)
    pre = 1.0 / (1.0 + tl.exp(-pre_l)) + hc_eps
    post = 2.0 * (1.0 / (1.0 + tl.exp(-post_l)))
    ii = tl.arange(0, 4)[:, None]
    jj = tl.arange(0, 4)[None, :]
    flat = 8 + ii * 4 + jj
    cl = tl.sum(tl.where(m[None, None, :] == flat[:, :, None], (mix * s2 + base)[None, None, :], 0.0), axis=2)
    cmax = tl.max(cl, axis=1)
    ce = tl.exp(cl - cmax[:, None])
    comb = ce / tl.sum(ce, axis=1)[:, None] + hc_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    for _ in range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    tl.store(PRE_OUT + r * 4 + sv, pre)
    tl.store(POST + r * 4 + sv, post)
    tl.store(COMB + r * 16 + ii * 4 + jj, comb)


# (switch "hc_dots") _hc_mix_rows' posted streams alone: out_s = post_s y + sum_j comb[j, s] x_j (its fma chain, the
# same bf16 rounding), a program a (row, K block of one stream), stored; elementwise, so any tiling has the same bits.
@triton.jit
def _hc_post_only(X, XO, G, RS, POST, COMB, WIDE: tl.constexpr, D: tl.constexpr, KB: tl.constexpr,
                  WORLD: tl.constexpr, PDL: tl.constexpr = False):
    r = tl.program_id(0)
    b = tl.program_id(1)
    s = (b * KB) // D
    d = b * KB - s * D + tl.arange(0, KB)
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    c0 = tl.load(COMB + r * 16 + 0 * 4 + s)
    c1 = tl.load(COMB + r * 16 + 1 * 4 + s)
    c2 = tl.load(COMB + r * 16 + 2 * 4 + s)
    c3 = tl.load(COMB + r * 16 + 3 * 4 + s)
    ps = tl.load(POST + r * 4 + s)
    ya = tl.load(G + r * D + d)
    for q in tl.static_range(1, WORLD):
        ya = ya + tl.load(G + q * RS + r * D + d)
    y = ya.to(tl.bfloat16).to(tl.float32)
    x0 = tl.load(X + r * WIDE + d).to(tl.float32)
    x1 = tl.load(X + r * WIDE + D + d).to(tl.float32)
    x2 = tl.load(X + r * WIDE + 2 * D + d).to(tl.float32)
    x3 = tl.load(X + r * WIDE + 3 * D + d).to(tl.float32)
    v = tl.fma(ps, y, tl.fma(c3, x3, tl.fma(c2, x2, tl.fma(c0, x0, c1 * x1))))
    tl.store(XO + r * WIDE + s * D + d, v.to(tl.bfloat16))


def hc_defer_on(d: int) -> bool:
    """Whether hc_pre2 takes ``sink`` (the Sinkhorn half on that side stream: switch "hc_defer", with "hc_split")."""

    return SMALL_SWITCHES["hc_defer"] and SMALL_SWITCHES["hc_split"] and d % 1024 == 0


def hc_dots_on(d: int) -> bool:
    """Whether the decode bodies' posted mHC steps take their mixes' dots off the main stream (switch "hc_dots")."""

    return SMALL_SWITCHES["hc_dots"] and hc_defer_on(d)


# rows a program of _hc_mix_part by window rows (switch "hc_rb"; one-GPU sweep: fewer weight-tile reads from L2
# against rows in series in a program); off: b768469's 1 up to 3 rows, else 2. A row's arithmetic is the same in any.
HC_RB = {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 2, 7: 4, 8: 4}


def _hc_rb(rows: int) -> int:
    env = os.environ.get("TF_DS_HC_RB")                      # (benchmarks: one value for every row count)
    if env:
        return min(rows, int(env))
    if not SMALL_SWITCHES["hc_rb"]:
        return 1 if rows <= 3 else 2
    return HC_RB.get(rows, 4 if rows <= 12 else 8)


HC_ROT_SPLIT_ROWS = int(os.environ.get("TF_DS_HC_ROT_SPLIT_ROWS") or 8)   # rotation programs a 1024 columns up to


def hc_rot_on(d: int) -> bool:
    """Whether hc_pre2 takes ``rot`` (its output rows rotated for the EXL3 layers that read them)."""

    return SMALL_SWITCHES["hc_split"] and SMALL_SWITCHES["hc_rot"] and d % 1024 == 0


def hc_pre2(h: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, pre_in: torch.Tensor,
            norm_w: torch.Tensor, eps: float, hc_eps: float, iters: int, out: torch.Tensor, pre_out: torch.Tensor,
            post: torch.Tensor, comb: torch.Tensor, part: torch.Tensor, gathered: torch.Tensor | None = None,
            h_out: torch.Tensor | None = None, rot: list | None = None,
            sink: torch.cuda.Stream | None = None) -> torch.Tensor:
    """hc_pre's outputs (the same bits), with the previous sublayer's hc_post fused in when ``gathered`` is given: the
    post (reading h, post, comb) is written into ``h_out`` (not h: other programs still read h) and the mixes are those
    of h_out. Returns the streams the mixes were taken of (h_out, or h). ``rot`` (up to four (suh, rows [R, D] fp16)
    pairs; only when hc_rot_on(D)): ``out``'s rows also written rotated for the EXL3 layers that read them (an
    Exl3Group's buffers, then its rotated(): rot_many's bits). ``sink`` (a side stream; only when hc_defer_on(D)): the
    Sinkhorn half (pre_out, post, comb) runs there (_hc_sinkhorn, once the mixes are done), the finish on the current
    stream is the collapse + RMSNorm alone; the caller makes the current stream wait for ``sink`` before anything reads
    pre_out, post or comb (or writes post / comb, or the mixes' partial sums ``part``)."""

    rows = h.shape[0]
    d = h.shape[-1]
    wide = 4 * d
    rb = _hc_rb(rows)                                       # rows a program (sharing its weight tiles)
    grid = (triton.cdiv(rows, rb), HC_BLOCKS, 3)
    dots = sink is not None and gathered is not None and hc_dots_on(d)
    if dots:
        # (switch "hc_dots") the post alone on this stream; the mixes of the stored new streams (the same partial
        # sums: _hc_mix_part's non-posted programs read the bf16 values the posted ones keep in registers) and the
        # Sinkhorn on the side stream, once the post is done (it reads post / comb, which the Sinkhorn rewrites)
        assert h_out is not None and h_out.data_ptr() != h.data_ptr()
        rot = rot or []
        assert not rot or (hc_rot_on(d) and len(rot) <= 4)
        st = [t for pair in rot for t in pair] + [out, out] * (4 - len(rot))
        rs = (d // 1024) if rot and rows <= HC_ROT_SPLIT_ROWS else 1
        _hc_post_only[(rows, HC_BLOCKS)](h, h_out, gathered, rows * d, post, comb, WIDE=wide, D=d,
                                         KB=wide // HC_BLOCKS, WORLD=gathered.shape[0], num_warps=2, **_pdl())
        _hc_finish_s[(rows, 1 + len(rot) * rs)](h_out, part, base, scale, pre_in, norm_w, out, pre_out, post, comb, eps,
                                                hc_eps, *st, D=d, NB=HC_BLOCKS, ITERS=iters, BLOCK=1024,
                                                NROT=len(rot), RSPLIT=rs, num_warps=8, SINK=False, **_pdl())
        sink.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sink):
            _hc_mix_part[grid](h_out, h_out, h_out, 0, post, comb, fn, part, rows, WIDE=wide, D=d, NB=HC_BLOCKS,
                               SUB=128, MB=8, WORLD=1, POSTED=False, RB=rb, num_warps=2)
            _hc_sinkhorn[(rows,)](part, base, scale, pre_out, post, comb, eps, hc_eps, D=d, NB=HC_BLOCKS,
                                  ITERS=iters, num_warps=8)
        return h_out
    if gathered is not None:
        assert h_out is not None and h_out.data_ptr() != h.data_ptr()
        xpf = {"XPF": True} if SMALL_SWITCHES["hc_xpf"] else {}
        _hc_mix_part[grid](h, h_out, gathered, rows * d, post, comb, fn, part, rows, WIDE=wide, D=d, NB=HC_BLOCKS,
                           SUB=128, MB=8, WORLD=gathered.shape[0], POSTED=True, RB=rb, num_warps=2, **xpf, **_pdl())
        src = h_out
    else:
        _hc_mix_part[grid](h, h, h, 0, post, comb, fn, part, rows, WIDE=wide, D=d, NB=HC_BLOCKS, SUB=128, MB=8,
                           WORLD=1, POSTED=False, RB=rb, num_warps=2, **_pdl())
        src = h
    if SMALL_SWITCHES["hc_split"] and d % 1024 == 0:
        rot = rot or []
        assert not rot or (hc_rot_on(d) and len(rot) <= 4)
        st = [t for pair in rot for t in pair] + [out, out] * (4 - len(rot))
        rs = (d // 1024) if rot and rows <= HC_ROT_SPLIT_ROWS else 1     # a rotation program a (layer, 1024 columns)
        defer = sink is not None
        if defer:
            assert hc_defer_on(d)
            # the Sinkhorn half on the side stream, after the mixes (their partial sums), beside the finish below
            sink.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(sink):
                _hc_sinkhorn[(rows,)](part, base, scale, pre_out, post, comb, eps, hc_eps, D=d, NB=HC_BLOCKS,
                                      ITERS=iters, num_warps=8)
        _hc_finish_s[(rows, (1 if defer else 2) + len(rot) * rs)](src, part, base, scale, pre_in, norm_w, out,
                                                                  pre_out, post, comb, eps, hc_eps, *st,
                                                                  D=d, NB=HC_BLOCKS, ITERS=iters, BLOCK=1024,
                                                                  NROT=len(rot), RSPLIT=rs, num_warps=8,
                                                                  SINK=not defer, **_pdl())
    else:
        _hc_finish_u[(rows,)](src, part, base, scale, pre_in, norm_w, out, pre_out, post, comb, eps, hc_eps, D=d,
                              NB=HC_BLOCKS, ITERS=iters, BLOCK=1024, num_warps=8, **_pdl())
    return src


# router / indexer weights: _rowmm's per-output arithmetic (an output's K chunks in order, each chunk one warp's: 8
# columns a lane, the same butterfly) with 4 outputs a program (one warp each) and the program's whole weight slice
# loaded up front (before the PDL wait), so the loads are all in flight together
@triton.jit
def _rowmm2(X, xs, W, OUT, scale, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            PDL: tl.constexpr = False, WTS: tl.constexpr = False):
    r = tl.program_id(0)
    nb = tl.program_id(1)
    nn = nb * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    ws = ()
    for t in tl.static_range(K // BK):
        ws = ws + (tl.load(W + nn[:, None] * K + (t * BK + kk)[None, :], mask=(nn < N)[:, None], other=0.0),)
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    acc = tl.zeros((BN,), dtype=tl.float32)
    for t in tl.static_range(K // BK):
        x = tl.load(X + r * xs + t * BK + kk).to(tl.float32)
        acc += tl.sum(ws[t].to(tl.float32) * x[None, :], axis=1)
    if WTS:                         # the indexer's weights: (rowmm(x, w).to(bf16) * scale) -> bf16, torch's roundings
        tl.store(OUT + r * N + nn, (acc.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16), mask=nn < N)
    else:
        tl.store(OUT + r * N + nn, acc, mask=nn < N)


ROWMM2_ROWS = 2          # rowmm2 up to this many rows (beyond, rowmm's tiles share the weights better in L2)
# TF_DS_ROWMM_RB=n (default 16; 0: rowmm): windows of more than ROWMM2_ROWS rows take _rowmm2r, up to n rows a
# program sharing the program's weight slice (loaded once, before the PDL wait), each row rowmm2's arithmetic (rowmm's
# bits: checked on the router's and the indexer's shapes at 1-16 rows)
ROWMM_RB = int(os.environ.get("TF_DS_ROWMM_RB") or 16)


@triton.jit
def _rowmm2r(X, xs, W, OUT, scale, rows, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             RB: tl.constexpr, PDL: tl.constexpr = False, WTS: tl.constexpr = False):
    """_rowmm2 for RB rows a program: the weight slice loaded once (before the PDL wait), then each row's dot as
    _rowmm2 makes it (the same K chunks in order, the same per-output reduction), rows past ``rows`` skipped."""

    r0 = tl.program_id(0) * RB
    nb = tl.program_id(1)
    nn = nb * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    ws = ()
    for t in tl.static_range(K // BK):
        ws = ws + (tl.load(W + nn[:, None] * K + (t * BK + kk)[None, :], mask=(nn < N)[:, None], other=0.0),)
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    for i in tl.static_range(RB):
        r = r0 + i
        if r < rows:
            acc = tl.zeros((BN,), dtype=tl.float32)
            for t in tl.static_range(K // BK):
                x = tl.load(X + r * xs + t * BK + kk).to(tl.float32)
                acc += tl.sum(ws[t].to(tl.float32) * x[None, :], axis=1)
            if WTS:
                tl.store(OUT + r * N + nn, (acc.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16), mask=nn < N)
            else:
                tl.store(OUT + r * N + nn, acc, mask=nn < N)


def rowmm2(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """rowmm's bits (x [R, K] @ w[N, K]^T -> fp32 [R, N], a row alone), faster at 1-2 rows (and with ROWMM_RB, at
    more: _rowmm2r)."""

    rows, k = x.shape
    n = w.shape[0]
    if rows > ROWMM2_ROWS and ROWMM_RB > 0 and k % 256 == 0 and rows <= DECODE_ROWS:
        if out is None:
            out = torch.empty((rows, n), dtype=torch.float32, device=x.device)
        bn = 4 if n > 64 else 1
        rb = min(ROWMM_RB, rows)
        _rowmm2r[(triton.cdiv(rows, rb), triton.cdiv(n, bn))](x, x.stride(0), w, out, 1.0, rows, K=k, N=n, BN=bn,
                                                             BK=256, RB=rb, num_warps=bn, **_pdl())
        return out
    if rows > ROWMM2_ROWS or k % 256:
        return rowmm(x, w, out)
    if out is None:
        out = torch.empty((rows, n), dtype=torch.float32, device=x.device)
    bn = 4 if n > 64 else 1                      # a narrow layer: one output (one warp) a program
    _rowmm2[(rows, triton.cdiv(n, bn))](x, x.stride(0), w, out, 1.0, K=k, N=n, BN=bn, BK=256, num_warps=bn,
                                        **_pdl())
    return out


def rowmm_wts(x: torch.Tensor, w: torch.Tensor, scale: float) -> torch.Tensor:
    """(rowmm(x, w).to(bf16) * scale) as bf16 [R, N] in one launch (decode windows), the same bits."""

    rows, k = x.shape
    n = w.shape[0]
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
    bn = 4 if n > 64 else 1
    _rowmm2[(rows, triton.cdiv(n, bn))](x, x.stride(0), w, out, scale, K=k, N=n, BN=bn, BK=256, num_warps=bn,
                                        WTS=True, **_pdl())
    return out


# the indexer's / attention's q RMSNorm (_rmsnorm's arithmetic) and, from its bf16 rows, the rotated input rows of up
# to two EXL3 linears that read them (wq_b, the indexer's wq_b): rot_many's bits
@triton.jit
def _rmsnorm_rot(X, xs, W, OUT, os_, eps, S0, H0, S1, H1, D: tl.constexpr, BLOCK: tl.constexpr, NROT: tl.constexpr,
                 PDL: tl.constexpr = False):
    pl = tl.minimum(tl.arange(0, BLOCK // 64) * 64, D - 1)       # the suh lines (read after the row's stores)
    if NROT > 0:
        _prefetch_l2(S0 + pl)
    if NROT > 1:
        _prefetch_l2(S1 + pl)
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    ok = d < D
    x = tl.load(X + r * xs + d, mask=ok, other=0.0).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(W + d, mask=ok, other=0.0).to(tl.float32)
    y = (w * (x * rinv)).to(tl.bfloat16)
    tl.store(OUT + r * os_ + d, y, mask=ok)
    NBK: tl.constexpr = BLOCK // 128
    v = tl.reshape(y.to(tl.float32), (NBK, 128))
    dd = tl.reshape(d, (NBK, 128))
    okk = dd < D
    if NROT > 0:
        sv = tl.load(S0 + dd, mask=okk, other=0.0).to(tl.float32)
        tl.store(H0 + r * D + dd, _rot128(v, sv, NBK).to(tl.float16), mask=okk)
    if NROT > 1:
        sv = tl.load(S1 + dd, mask=okk, other=0.0).to(tl.float32)
        tl.store(H1 + r * D + dd, _rot128(v, sv, NBK).to(tl.float16), mask=okk)


@triton.jit
def _q_kv_norm(X, xs, W, OUT, os_, eps, S0, H0, S1, H1, Y, WK, COS, SIN, POS, KOUT, RING, SLOT_OF, ring_size,
               QUANT: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, NROT: tl.constexpr, DK: tl.constexpr,
               RD: tl.constexpr, PDL: tl.constexpr = False, SPLIT: tl.constexpr = False):
    """_rmsnorm_rot (programs (r, 0)) and _kv_norm_rope (programs (r, 1)) of the same rows in one launch: the two
    independent norms after attn_in, each program the code of the kernel it replaces."""

    role = tl.program_id(1)
    if role != 1:
        pl = tl.minimum(tl.arange(0, BLOCK // 64) * 64, D - 1)   # the suh lines (read after the row's stores)
        if NROT > 0 and (role == 0 or role == 2):
            _prefetch_l2(S0 + pl)
        if NROT > 1 and (role == 0 or role == 3):
            _prefetch_l2(S1 + pl)
        _prefetch_l2(W + pl)                                     # and the norm weights (weights: before the wait)
    else:
        _prefetch_l2(WK + tl.arange(0, DK // 64) * 64)
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    if role != 1:
        # programs (r, 0): the norm, its rows and (SPLIT False) the rotations; (r, 2 + i) with SPLIT: the norm again
        # (the same code, the same bits) and rotation i alone, beside the others
        d = tl.arange(0, BLOCK)
        ok = d < D
        x = tl.load(X + r * xs + d, mask=ok, other=0.0).to(tl.float32)
        rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
        w = tl.load(W + d, mask=ok, other=0.0).to(tl.float32)
        y = (w * (x * rinv)).to(tl.bfloat16)
        if role == 0:
            tl.store(OUT + r * os_ + d, y, mask=ok)
        NBK: tl.constexpr = BLOCK // 128
        v = tl.reshape(y.to(tl.float32), (NBK, 128))
        dd = tl.reshape(d, (NBK, 128))
        okk = dd < D
        if NROT > 0 and ((not SPLIT and role == 0) or role == 2):
            sv = tl.load(S0 + dd, mask=okk, other=0.0).to(tl.float32)
            tl.store(H0 + r * D + dd, _rot128(v, sv, NBK).to(tl.float16), mask=okk)
        if NROT > 1 and ((not SPLIT and role == 0) or role == 3):
            sv = tl.load(S1 + dd, mask=okk, other=0.0).to(tl.float32)
            tl.store(H1 + r * D + dd, _rot128(v, sv, NBK).to(tl.float16), mask=okk)
    else:
        HALF: tl.constexpr = DK // 2
        i = tl.arange(0, HALF)
        xe = tl.load(Y + r * DK + 2 * i).to(tl.float32)
        xo = tl.load(Y + r * DK + 2 * i + 1).to(tl.float32)
        rinv = 1.0 / tl.sqrt((tl.sum(xe * xe, axis=0) + tl.sum(xo * xo, axis=0)) / DK + eps)
        we = tl.load(WK + 2 * i).to(tl.float32)
        wo = tl.load(WK + 2 * i + 1).to(tl.float32)
        ne = (we * (xe * rinv)).to(tl.bfloat16).to(tl.float32)
        no = (wo * (xo * rinv)).to(tl.bfloat16).to(tl.float32)
        p = tl.load(POS + r)
        PAIRS0: tl.constexpr = HALF - RD // 2
        j = tl.maximum(i - PAIRS0, 0)
        cs = tl.load(COS + p * (RD // 2) + j)
        sn = tl.load(SIN + p * (RD // 2) + j)
        rot = i >= PAIRS0
        re = tl.where(rot, (ne * cs - no * sn), ne).to(tl.bfloat16).to(tl.float32)
        im = tl.where(rot, (ne * sn + no * cs), no).to(tl.bfloat16).to(tl.float32)
        if QUANT:
            # 32-element blocks = 16 pairs
            a = tl.maximum(tl.abs(re), tl.abs(im))
            amax = tl.max(tl.reshape(a, (HALF // 16, 16)), axis=1)
            s = _pow2_ceil(tl.maximum(amax, 1e-4) / 448.0)
            sb = tl.reshape(tl.broadcast_to(s[:, None], (HALF // 16, 16)), (HALF,))
            re = (tl.minimum(tl.maximum(re / sb, -448.0), 448.0)).to(tl.float8e4nv).to(tl.float32) * sb
            im = (tl.minimum(tl.maximum(im / sb, -448.0), 448.0)).to(tl.float8e4nv).to(tl.float32) * sb
        tl.store(KOUT + r * DK + 2 * i, re.to(tl.bfloat16))
        tl.store(KOUT + r * DK + 2 * i + 1, im.to(tl.bfloat16))
        slot = tl.load(SLOT_OF + r)
        if slot >= 0:
            tl.store(RING + slot * DK + 2 * i, re.to(tl.bfloat16))
            tl.store(RING + slot * DK + 2 * i + 1, im.to(tl.bfloat16))


def q_kv_norm(x: torch.Tensor, w: torch.Tensor, eps: float, rot: list, y: torch.Tensor, wk: torch.Tensor,
              cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor, ring: torch.Tensor, slots: torch.Tensor,
              quant: bool, rd: int) -> torch.Tensor:
    """rmsnorm_rot(x, w, eps, rot) and kv_norm_rope(y, wk, cos, sin, pos, ring, slots, eps, quant, rd) in one launch;
    returns rmsnorm_rot's rows."""

    rows, d = x.shape
    dk = y.shape[1]
    assert d <= 2048
    out = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
    kout = torch.empty_like(y)
    st = [t for pair in rot for t in pair] + [out, out] * (2 - len(rot))
    split = SMALL_SWITCHES["qkv_split"] and len(rot) > 0
    _q_kv_norm[(rows, 2 + (len(rot) if split else 0))](x, x.stride(0), w, out, out.stride(0), eps, *st, y, wk, cos,
                                                       sin, pos, kout, ring, slots, ring.shape[0], QUANT=quant, D=d,
                                                       BLOCK=triton.next_power_of_2(d), NROT=len(rot), DK=dk, RD=rd,
                                                       num_warps=4, SPLIT=split, **_pdl())
    return out


def rmsnorm_rot(x: torch.Tensor, w: torch.Tensor, eps: float, rot: list,
                out: torch.Tensor | None = None) -> torch.Tensor:
    rows, d = x.shape
    if out is None:
        out = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
    st = [t for pair in rot for t in pair] + [out, out] * (2 - len(rot))
    _rmsnorm_rot[(rows,)](x, x.stride(0), w, out, out.stride(0), eps, *st, D=d, BLOCK=triton.next_power_of_2(d),
                          NROT=len(rot), num_warps=4 if d <= 2048 else 8, **_pdl())
    return out


# -- RMSNorm (DeepSeek: bf16(w * (x * rsqrt(mean(x^2) + eps)))) ----------------------------------------------------
@triton.jit
def _rmsnorm(X, xs, W, OUT, os_, eps, D: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    ok = d < D
    x = tl.load(X + r * xs + d, mask=ok, other=0.0).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(W + d, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * os_ + d, (w * (x * rinv)).to(tl.bfloat16), mask=ok)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    rows, d = x.shape
    if out is None:
        out = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
    _rmsnorm[(rows,)](x, x.stride(0), w, out, out.stride(0), eps, D=d, BLOCK=triton.next_power_of_2(d),
                      num_warps=4 if d <= 2048 else 8, **_pdl())
    return out


# -- quantization helpers -------------------------------------------------------------------------------------------
@triton.jit
def _pow2_ceil(v):
    bits = v.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    return tl.exp2(e.to(tl.float32))


@triton.jit
def _e2m1(code):
    """E2M1 nibble (sign bit 3) -> fp32 by building the float's bits: magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6."""

    m = code & 7
    bits = tl.where(m >= 2, (((m >> 1) + 126) << 23) | ((m & 1) << 22), tl.where(m == 1, 0x3F000000, 0))
    bits = bits | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


# -- the window KV: RMSNorm, RoPE on the last 64 (adjacent pairs), FP8 quant-dequant per 32, into the ring ------
@triton.jit
def _kv_norm_rope(Y, W, COS, SIN, POS, OUT, RING, SLOT_OF, ring_size, eps, QUANT: tl.constexpr,
                  D: tl.constexpr, RD: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    xe = tl.load(Y + r * D + 2 * i).to(tl.float32)
    xo = tl.load(Y + r * D + 2 * i + 1).to(tl.float32)
    rinv = 1.0 / tl.sqrt((tl.sum(xe * xe, axis=0) + tl.sum(xo * xo, axis=0)) / D + eps)
    we = tl.load(W + 2 * i).to(tl.float32)
    wo = tl.load(W + 2 * i + 1).to(tl.float32)
    ne = (we * (xe * rinv)).to(tl.bfloat16).to(tl.float32)
    no = (wo * (xo * rinv)).to(tl.bfloat16).to(tl.float32)
    p = tl.load(POS + r)
    PAIRS0: tl.constexpr = HALF - RD // 2
    j = tl.maximum(i - PAIRS0, 0)
    cs = tl.load(COS + p * (RD // 2) + j)
    sn = tl.load(SIN + p * (RD // 2) + j)
    rot = i >= PAIRS0
    re = tl.where(rot, (ne * cs - no * sn), ne).to(tl.bfloat16).to(tl.float32)
    im = tl.where(rot, (ne * sn + no * cs), no).to(tl.bfloat16).to(tl.float32)
    if QUANT:
        # 32-element blocks = 16 pairs
        a = tl.maximum(tl.abs(re), tl.abs(im))
        amax = tl.max(tl.reshape(a, (HALF // 16, 16)), axis=1)
        s = _pow2_ceil(tl.maximum(amax, 1e-4) / 448.0)
        sb = tl.reshape(tl.broadcast_to(s[:, None], (HALF // 16, 16)), (HALF,))
        re = (tl.minimum(tl.maximum(re / sb, -448.0), 448.0)).to(tl.float8e4nv).to(tl.float32) * sb
        im = (tl.minimum(tl.maximum(im / sb, -448.0), 448.0)).to(tl.float8e4nv).to(tl.float32) * sb
    tl.store(OUT + r * D + 2 * i, re.to(tl.bfloat16))
    tl.store(OUT + r * D + 2 * i + 1, im.to(tl.bfloat16))
    slot = tl.load(SLOT_OF + r)
    if slot >= 0:
        tl.store(RING + slot * D + 2 * i, re.to(tl.bfloat16))
        tl.store(RING + slot * D + 2 * i + 1, im.to(tl.bfloat16))


def kv_norm_rope(y: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor,
                 ring: torch.Tensor, slots: torch.Tensor, eps: float, quant: bool, rd: int,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    rows, d = y.shape
    if out is None:
        out = torch.empty_like(y)
    _kv_norm_rope[(rows,)](y, w, cos, sin, pos, out, ring, slots, ring.shape[0], eps, QUANT=quant, D=d, RD=rd,
                           num_warps=4, **_pdl())
    return out


# -- RoPE on the last RD dims of every head (q forward, attention output inverse) -----------------------------------
@triton.jit
def _rope_heads(X, COS, SIN, POS, H: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr, INV: tl.constexpr,
                HB: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb * HB + tl.arange(0, HB)
    j = tl.arange(0, RD // 2)
    p = tl.load(POS + r)
    cs = tl.load(COS + p * (RD // 2) + j)
    sn = tl.load(SIN + p * (RD // 2) + j)
    if INV:
        sn = -sn
    base = X + r * (H * HD) + h[:, None] * HD + (HD - RD) + 2 * j[None, :]
    xe = tl.load(base).to(tl.float32)
    xo = tl.load(base + 1).to(tl.float32)
    tl.store(base, (xe * cs[None, :] - xo * sn[None, :]).to(tl.bfloat16))
    tl.store(base + 1, (xe * sn[None, :] + xo * cs[None, :]).to(tl.bfloat16))


def rope_heads(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor, rd: int,
               inverse: bool = False) -> torch.Tensor:
    """x [R, H, HD] bf16 in place."""

    rows, h, hd = x.shape
    hb = min(h, 16)
    _rope_heads[(rows, h // hb)](x, cos, sin, pos, H=h, HD=hd, RD=rd, INV=inverse, HB=hb, num_warps=4, **_pdl())
    return x


# -- sparse attention with a sink: window (ring or a linear source) + selected compressed entries ------------------
# Rows are handled as two 256-wide halves (q, keys, accumulators), which is also how the FP4 cache packs a row. The
# keys (window then picks, in blocks of BN) are split over SPLITS programs a head group; a second kernel merges the
# partial softmax states in split order, so a row's bits never depend on how many rows share the call.
@triton.jit
def _attn_step(q_lo, q_hi, k_lo, k_hi, ok, m_i, l_i, acc_lo, acc_hi, scale):
    s = (tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))) * scale
    s = tl.where(ok[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.exp(m_i - m_new)
    pr = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(pr, axis=1)
    p16 = pr.to(tl.bfloat16)
    acc_lo = acc_lo * alpha[:, None] + tl.dot(p16, k_lo)
    acc_hi = acc_hi * alpha[:, None] + tl.dot(p16, k_hi)
    return m_new, l_i, acc_lo, acc_hi


@triton.jit
def _comp_keys(COMP, CSC, row, ok, hc, HD: tl.constexpr, BN: tl.constexpr, PACKED: tl.constexpr):
    HALF: tl.constexpr = HD // 2
    if PACKED:
        cb = tl.load(COMP + row[:, None] * HALF + hc[None, :], mask=ok[:, None], other=0).to(tl.int32)
        g = tl.arange(0, HALF // 16)
        sl = tl.load(CSC + row[:, None] * (HD // 16) + g[None, :], mask=ok[:, None], other=0)
        sh = tl.load(CSC + row[:, None] * (HD // 16) + HALF // 16 + g[None, :], mask=ok[:, None], other=0)
        sl = sl.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sh = sh.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        lo = tl.reshape(tl.reshape(_e2m1(cb & 15), (BN, HALF // 16, 16)) * sl[:, :, None], (BN, HALF))
        hi = tl.reshape(tl.reshape(_e2m1(cb >> 4), (BN, HALF // 16, 16)) * sh[:, :, None], (BN, HALF))
        return lo.to(tl.bfloat16), hi.to(tl.bfloat16)
    else:
        kb = COMP + row[:, None] * HD
        return (tl.load(kb + hc[None, :], mask=ok[:, None], other=0.0),
                tl.load(kb + HALF + hc[None, :], mask=ok[:, None], other=0.0))


@triton.jit
def _sparse_attn_part(Q, WSRC, WLO, COMP, CSC, IDX, POS, PM, PL, PO, SINK, OUT, scale, ring_size, n_idx, RBASE, CBASE,
                      H: tl.constexpr, HD: tl.constexpr, HB: tl.constexpr, WIN: tl.constexpr, BN: tl.constexpr,
                      RING: tl.constexpr, HAS_COMP: tl.constexpr, PACKED: tl.constexpr, SPLITS: tl.constexpr,
                      NBLK: tl.constexpr, FINAL: tl.constexpr, HAS_BASE: tl.constexpr, PDL: tl.constexpr = False,
                      WPF: tl.constexpr = False):
    if WPF:
        # (switch "attn_pf") the split's window block into L2 before the PDL wait: ring rows written kernels ago (the
        # window KV norm), at slots of round inputs (positions, ring bases); a prefetch moves no data a kernel reads
        spw = tl.program_id(2)
        if spw < WIN // BN:
            nw = tl.arange(0, BN)
            pw = tl.load(POS + tl.program_id(0))
            wpw = pw - (WIN - 1) + spw * BN + nw
            okw = wpw >= 0
            sw = tl.where(okw, wpw % ring_size, 0)
            if HAS_BASE:
                sw = sw + tl.load(RBASE + tl.program_id(0))
            _prefetch_l2(WSRC + sw[:, None] * HD + (tl.arange(0, HD // 64) * 64)[None, :])
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    hb = tl.program_id(1)
    sp = tl.program_id(2)
    HALF: tl.constexpr = HD // 2
    h = hb * HB + tl.arange(0, HB)
    hc = tl.arange(0, HALF)
    qb = Q + r * (H * HD) + h[:, None] * HD
    q_lo = tl.load(qb + hc[None, :])
    q_hi = tl.load(qb + HALF + hc[None, :])
    p = tl.load(POS + r)
    wlo = tl.load(WLO)
    m_i = tl.full((HB,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((HB,), dtype=tl.float32)
    acc_lo = tl.zeros((HB, HALF), dtype=tl.float32)
    acc_hi = tl.zeros((HB, HALF), dtype=tl.float32)
    n = tl.arange(0, BN)
    WB: tl.constexpr = WIN // BN
    # key blocks 0..WB-1 are the window, WB.. the picks; this program takes blocks sp, sp + SPLITS, ...
    for b in range(sp, NBLK, SPLITS):
        if b < WB:
            wp = p - (WIN - 1) + b * BN + n
            ok = wp >= 0
            if RING:
                slot = wp % ring_size
            else:
                slot = wp - wlo
                ok = ok & (slot >= 0)
            slot = tl.where(ok, slot, 0)
            if HAS_BASE:                   # a concurrent round: the row's stream's ring starts at pool row RBASE[r]
                slot = slot + tl.load(RBASE + r)
            kb = WSRC + slot[:, None] * HD
            k_lo = tl.load(kb + hc[None, :], mask=ok[:, None], other=0.0)
            k_hi = tl.load(kb + HALF + hc[None, :], mask=ok[:, None], other=0.0)
        else:
            t = (b - WB) * BN
            ii = tl.load(IDX + r * n_idx + t + n, mask=(t + n) < n_idx, other=-1)
            ok = ii >= 0
            row = tl.where(ok, ii, 0)
            if HAS_BASE:                   # ... and its compressed rows at pool row CBASE[r]
                row = row + tl.load(CBASE + r)
            k_lo, k_hi = _comp_keys(COMP, CSC, row, ok, hc, HD, BN, PACKED)
        m_i, l_i, acc_lo, acc_hi = _attn_step(q_lo, q_hi, k_lo, k_hi, ok, m_i, l_i, acc_lo, acc_hi, scale)
    if FINAL:                         # one split (prompt chunks): the sink and the division here, no merge pass
        l_i = l_i + tl.exp(tl.load(SINK + h) - m_i)
        ob = OUT + r * (H * HD) + h[:, None] * HD
        tl.store(ob + hc[None, :], (acc_lo / l_i[:, None]).to(tl.bfloat16))
        tl.store(ob + HALF + hc[None, :], (acc_hi / l_i[:, None]).to(tl.bfloat16))
        return
    base = (r * (H // HB) + hb) * SPLITS + sp
    tl.store(PM + base * HB + tl.arange(0, HB), m_i)
    tl.store(PL + base * HB + tl.arange(0, HB), l_i)
    ob = PO + base * HB * HD + tl.arange(0, HB)[:, None] * HD
    tl.store(ob + hc[None, :], acc_lo)
    tl.store(ob + HALF + hc[None, :], acc_hi)


@triton.jit
def _sparse_attn_merge(PM, PL, PO, SINK, OUT, H: tl.constexpr, HD: tl.constexpr, HB: tl.constexpr,
                       SPLITS: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    hb = tl.program_id(1)
    hh = tl.arange(0, HB)
    d = tl.arange(0, HD)
    base0 = (r * (H // HB) + hb) * SPLITS
    m = tl.full((HB,), -1e30, dtype=tl.float32)
    for sp in range(SPLITS):
        m = tl.maximum(m, tl.load(PM + (base0 + sp) * HB + hh))
    l = tl.zeros((HB,), dtype=tl.float32)
    acc = tl.zeros((HB, HD), dtype=tl.float32)
    for sp in range(SPLITS):
        ms = tl.load(PM + (base0 + sp) * HB + hh)
        a = tl.exp(ms - m)
        l += tl.load(PL + (base0 + sp) * HB + hh) * a
        acc += tl.load(PO + (base0 + sp) * HB * HD + hh[:, None] * HD + d[None, :]) * a[:, None]
    h = hb * HB + hh
    l += tl.exp(tl.load(SINK + h) - m)
    tl.store(OUT + r * (H * HD) + h[:, None] * HD + d[None, :], (acc / l[:, None]).to(tl.bfloat16))


@triton.jit
def _sparse_attn_merge_rot(PM, PL, PO, SINK, OUT, COS, SIN, POS, SUH, XH, rows, H: tl.constexpr, HD: tl.constexpr,
                           HB: tl.constexpr, SPLITS: tl.constexpr, RD: tl.constexpr, GH: tl.constexpr,
                           PDL: tl.constexpr = False, DISCARD: tl.constexpr = False):
    """_sparse_attn_merge for one head a program (its arithmetic is per head: the same bits as the HB-head program),
    then, through OUT, rope_heads' inverse RoPE of the head and wo_a's input rotation of it (GH heads a slice, the
    slices' suh concatenated in SUH, their rotated rows in XH: slice g's [rows, GH * HD] block at
    g * rows * GH * HD)."""

    h = tl.program_id(1)
    _prefetch_l2(SUH + h * HD + tl.arange(0, HD // 64) * 64)    # wo_a's suh of this head (read last, after barriers)
    _prefetch_l2(SINK + h + tl.arange(0, 1))
    pp = tl.load(POS + tl.program_id(0))                       # (positions: a round input, no kernel writes them)
    _prefetch_l2(COS + pp * (RD // 2) + tl.arange(0, 1))
    _prefetch_l2(SIN + pp * (RD // 2) + tl.arange(0, 1))
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    d = tl.arange(0, HD)
    base0 = (r * (H // HB) + h // HB) * SPLITS
    hi = h % HB
    m = -1e30
    for sp in range(SPLITS):
        m = tl.maximum(m, tl.load(PM + (base0 + sp) * HB + hi))
    l = 0.0
    acc = tl.zeros((HD,), dtype=tl.float32)
    for sp in range(SPLITS):
        ms = tl.load(PM + (base0 + sp) * HB + hi)
        a = tl.exp(ms - m)
        l += tl.load(PL + (base0 + sp) * HB + hi) * a
        acc += tl.load(PO + (base0 + sp) * HB * HD + hi * HD + d) * a
    l += tl.exp(tl.load(SINK + h) - m)
    ob = OUT + r * (H * HD) + h * HD
    tl.store(ob + d, (acc / l).to(tl.bfloat16))
    tl.debug_barrier()
    if DISCARD:
        # this head's split outputs (SPLITS x HD fp32, 128-byte aligned rows of 2 KB) are read by this program alone
        # and dead now: their L2 lines go without a write-back (~3 MB a 6-row layer otherwise reaches DRAM)
        q = tl.arange(0, SPLITS * (HD // 32))
        _discard_l2(PO + (base0 + q // (HD // 32)) * HB * HD + hi * HD + (q % (HD // 32)) * 32)
    # inverse RoPE on the last RD dims (rope_heads, INV), its compiled contraction spelled out: re = fma(xe, cs,
    # -(xo sn)), im = fma(xe, sn, xo cs) (checked against rope_heads on 16M random pairs: no other order matches)
    j = tl.arange(0, RD // 2)
    p = tl.load(POS + r)
    cs = tl.load(COS + p * (RD // 2) + j)
    sn = tl.load(SIN + p * (RD // 2) + j)
    sn = -sn
    rb = ob + (HD - RD) + 2 * j
    xe = tl.load(rb).to(tl.float32)
    xo = tl.load(rb + 1).to(tl.float32)
    tl.store(rb, tl.fma(xe, cs, -(xo * sn)).to(tl.bfloat16))
    tl.store(rb + 1, tl.fma(xe, sn, xo * cs).to(tl.bfloat16))
    tl.debug_barrier()
    # wo_a's input rotation of the head: its HD / 128 blocks (slice g = h // GH, columns (h % GH) * HD ..)
    NBK: tl.constexpr = HD // 128
    bi = tl.arange(0, NBK)
    c = tl.arange(0, 128)
    v = tl.load(ob + bi[:, None] * 128 + c[None, :]).to(tl.float32)
    s = tl.load(SUH + h * HD + bi[:, None] * 128 + c[None, :]).to(tl.float32)
    y = _rot128(v, s, NBK)
    g = h // GH
    col = (h % GH) * HD + bi * 128
    tl.store(XH + g * (rows * GH * HD) + r * (GH * HD) + col[:, None] + c[None, :], y.to(tl.float16))


# (switch "merge_reg") _sparse_attn_merge_rot with the head's row kept in registers: the merge's bf16 row, the inverse
# RoPE of its last RD values (pairs split out of the row: the same fma pattern on the same bf16 inputs, rounded to bf16)
# and wo_a's rotation of the roped bf16 row (_rot128, defined element by element), then one store of o (its final,
# roped content: what the two stores of _sparse_attn_merge_rot leave) and of the rotated rows. The same bits.
@triton.jit
def _sparse_attn_merge_rot2(PM, PL, PO, SINK, OUT, COS, SIN, POS, SUH, XH, rows, H: tl.constexpr, HD: tl.constexpr,
                            HB: tl.constexpr, SPLITS: tl.constexpr, RD: tl.constexpr, GH: tl.constexpr,
                            PDL: tl.constexpr = False, DISCARD: tl.constexpr = False):
    h = tl.program_id(1)
    _prefetch_l2(SUH + h * HD + tl.arange(0, HD // 64) * 64)    # wo_a's suh of this head
    _prefetch_l2(SINK + h + tl.arange(0, 1))
    pp = tl.load(POS + tl.program_id(0))                       # (positions: a round input, no kernel writes them)
    _prefetch_l2(COS + pp * (RD // 2) + tl.arange(0, 1))
    _prefetch_l2(SIN + pp * (RD // 2) + tl.arange(0, 1))
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    d = tl.arange(0, HD)
    base0 = (r * (H // HB) + h // HB) * SPLITS
    hi = h % HB
    m = -1e30
    for sp in range(SPLITS):
        m = tl.maximum(m, tl.load(PM + (base0 + sp) * HB + hi))
    l = 0.0
    acc = tl.zeros((HD,), dtype=tl.float32)
    for sp in range(SPLITS):
        ms = tl.load(PM + (base0 + sp) * HB + hi)
        a = tl.exp(ms - m)
        l += tl.load(PL + (base0 + sp) * HB + hi) * a
        acc += tl.load(PO + (base0 + sp) * HB * HD + hi * HD + d) * a
    l += tl.exp(tl.load(SINK + h) - m)
    y = (acc / l).to(tl.bfloat16).to(tl.float32)               # the merge's bf16 row
    if DISCARD:
        tl.debug_barrier()                                     # every thread's split loads are consumed
        q = tl.arange(0, SPLITS * (HD // 32))
        _discard_l2(PO + (base0 + q // (HD // 32)) * HB * HD + hi * HD + (q % (HD // 32)) * 32)
    # inverse RoPE on the last RD values (pairs j = HD / 2 - RD / 2 ..): rope_heads' contraction as in
    # _sparse_attn_merge_rot, on the same bf16 values
    ye, yo = tl.split(tl.reshape(y, (HD // 2, 2)))
    jj = tl.arange(0, HD // 2)
    J0: tl.constexpr = (HD - RD) // 2
    rot = jj >= J0
    jr = tl.maximum(jj - J0, 0)
    p = tl.load(POS + r)
    cs = tl.load(COS + p * (RD // 2) + jr)
    sn = tl.load(SIN + p * (RD // 2) + jr)
    sn = -sn
    re = tl.where(rot, tl.fma(ye, cs, -(yo * sn)).to(tl.bfloat16).to(tl.float32), ye)
    im = tl.where(rot, tl.fma(ye, sn, yo * cs).to(tl.bfloat16).to(tl.float32), yo)
    o = tl.reshape(tl.join(re, im), (HD,))                     # the roped bf16 row (as fp32)
    tl.store(OUT + r * (H * HD) + h * HD + d, o.to(tl.bfloat16))
    # wo_a's input rotation of the head: its HD / 128 blocks (slice g = h // GH, columns (h % GH) * HD ..)
    NBK: tl.constexpr = HD // 128
    bi = tl.arange(0, NBK)
    c = tl.arange(0, 128)
    v = tl.reshape(o, (NBK, 128))
    sv = tl.load(SUH + h * HD + bi[:, None] * 128 + c[None, :]).to(tl.float32)
    yr = _rot128(v, sv, NBK)
    g = h // GH
    col = (h % GH) * HD + bi * 128
    tl.store(XH + g * (rows * GH * HD) + r * (GH * HD) + col[:, None] + c[None, :], yr.to(tl.float16))


ATTN_SPLITS = 8


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0

# TF_EXL3_L2_DISCARD=1 (default): the decode attention merge drops the split outputs it has read from L2 (no
# write-back to DRAM of the dead fp32 partials; the same bits)
L2_DISCARD = os.environ.get("TF_EXL3_L2_DISCARD", "1") != "0"


def sparse_attn(q: torch.Tensor, sink: torch.Tensor, wsrc: torch.Tensor, wlo: torch.Tensor, ring: bool,
                comp, idx: torch.Tensor | None, pos: torch.Tensor, scale: float, window: int,
                out: torch.Tensor | None = None, wbase: torch.Tensor | None = None,
                cbase: torch.Tensor | None = None, ring_rows: int | None = None,
                rot: tuple | None = None) -> torch.Tensor:
    """q [R, H, HD] bf16 -> o [R, H, HD]; window keys from ``wsrc`` (a ring: slot = position % size; else linear from
    position wlo[0]); compressed keys comp[idx[r, j]] (idx -1 = none). ``comp`` is a bf16 [N, HD] tensor or a packed
    FP4 pair (codes uint8 [N, HD/2], E4M3 scales uint8 [N, HD/16])."""

    rows, h, hd = q.shape
    if out is None:
        out = torch.empty_like(q)
    hb, bn = 16, 32
    has = comp is not None and idx is not None and idx.shape[1] > 0
    packed = has and isinstance(comp, tuple)
    codes, scales = (comp if packed else (comp, None)) if has else (wsrc, None)
    n_idx = idx.shape[1] if has else 0
    # decode / verify windows split the keys (parallelism for a few rows); prompt chunks have rows enough
    sp = ATTN_SPLITS if rows <= DECODE_ROWS else 1
    groups = h // hb
    final = sp == 1
    picks = triton.cdiv(n_idx, bn) if has else 0
    if final and picks:
        # prompt chunks: so that prompts of any length share five compiled variants, the pick list is padded with -1
        # to a multiple of 16 entries and the pick blocks are rounded up to a power of two; a masked key or block
        # leaves m, l and acc unchanged (alpha 1, p 0), so the bits are those of the exact count
        if n_idx % 16:
            idx = torch.nn.functional.pad(idx, (0, 16 - n_idx % 16), value=-1)
            n_idx = idx.shape[1]
        picks = triton.next_power_of_2(picks)
    nblk = window // bn + picks
    # only ring windows read it (a fixed value otherwise: one variant); a pool of rings: one stream's ring rows
    ring_size = (ring_rows or wsrc.shape[0]) if ring else 16
    if final:
        pm = pl = po = out
    else:
        pm = torch.empty((rows * groups * sp * hb,), dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        po = torch.empty((rows * groups * sp * hb * hd,), dtype=torch.float32, device=q.device)
    based = wbase is not None
    if based:
        assert ring, "per-row bases address rings (concurrent rounds and batched drafts)"
    _sparse_attn_part[(rows, groups, sp)](q, wsrc, wlo, codes, scales if packed else wsrc, idx if has else pos, pos,
                                          pm, pl, po, sink, out, scale, ring_size, n_idx,
                                          wbase if based else pos, (cbase if cbase is not None else wbase) if based else pos,
                                          H=h, HD=hd, HB=hb, WIN=window, BN=bn, RING=ring, HAS_COMP=has, PACKED=packed,
                                          SPLITS=sp, NBLK=nblk, FINAL=final, HAS_BASE=based, num_warps=4, num_stages=1,
                                          WPF=SMALL_SWITCHES["attn_pf"] and ring and not final, **_pdl())
    if not final and rot is not None:
        # rot = (cos, sin, rope dim, wo_a's suh concatenated, its rotated rows' buffer, heads a slice): the merge, the
        # inverse RoPE and wo_a's input rotation in one launch (o is still written, roped)
        cos, sin, rd, suh, xh, gh = rot
        merge = _sparse_attn_merge_rot2 if SMALL_SWITCHES["merge_reg"] and _pow2(hd) else _sparse_attn_merge_rot
        merge[(rows, h)](pm, pl, po, sink, out, cos, sin, pos, suh, xh, rows, H=h, HD=hd, HB=hb,
                         SPLITS=sp, RD=rd, GH=gh, num_warps=4,
                         DISCARD=L2_DISCARD and hd % 32 == 0 and _pow2(sp * (hd // 32)), **_pdl())
    elif not final:
        _sparse_attn_merge[(rows, groups)](pm, pl, po, sink, out, H=h, HD=hd, HB=hb, SPLITS=sp, num_warps=8,
                                           **_pdl())
    return out


# -- indexer scores: sum_h relu(q_h . k_t) w_h over t < n, masked past each row's visible count -------------------
@triton.jit
def _index_score(Q, K, KS, Wt, VIS, OUT, n, KB, CAND, cs, IH: tl.constexpr, ID: tl.constexpr, BN: tl.constexpr,
                 PACKED: tl.constexpr, HAS_BASE: tl.constexpr, PDL: tl.constexpr = False,
                 HAS_CAND: tl.constexpr = False, CB: tl.constexpr = 8, KEYS: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    b = tl.program_id(1)
    HALF: tl.constexpr = ID // 2
    hh = tl.arange(0, IH)
    hc = tl.arange(0, HALF)
    t = b * BN + tl.arange(0, BN)
    ok = t < n                         # the scores written (every t < n: -inf past the row's visible keys)
    if HAS_BASE:                       # a concurrent round: the row's stream's keys start at pool row KB[r]
        kt = tl.load(KB + r) + t
        ld = ok & (t < tl.load(VIS + r))   # keys read: none past its own visible ones (its extent may end the pool)
    else:
        kt = t
        ld = ok
    q_lo = tl.load(Q + r * (IH * ID) + hh[:, None] * ID + hc[None, :])
    q_hi = tl.load(Q + r * (IH * ID) + hh[:, None] * ID + HALF + hc[None, :])
    if PACKED:
        # FP4 rows: byte j = element j and j + ID/2; a power-of-two (E8M0) scale per 32 elements
        cb = tl.load(K + kt[:, None] * HALF + hc[None, :], mask=ld[:, None], other=0).to(tl.int32)
        g = tl.arange(0, HALF // 32)
        el = tl.load(KS + kt[:, None] * (ID // 32) + g[None, :], mask=ld[:, None], other=127).to(tl.int32)
        eh = tl.load(KS + kt[:, None] * (ID // 32) + HALF // 32 + g[None, :], mask=ld[:, None], other=127).to(tl.int32)
        sl = (el << 23).to(tl.float32, bitcast=True)                    # E8M0 byte -> 2^(byte - 127)
        sh = (eh << 23).to(tl.float32, bitcast=True)
        k_lo = tl.reshape(tl.reshape(_e2m1(cb & 15), (BN, HALF // 32, 32)) * sl[:, :, None], (BN, HALF)).to(tl.bfloat16)
        k_hi = tl.reshape(tl.reshape(_e2m1(cb >> 4), (BN, HALF // 32, 32)) * sh[:, :, None], (BN, HALF)).to(tl.bfloat16)
    else:
        k_lo = tl.load(K + kt[:, None] * ID + hc[None, :], mask=ld[:, None], other=0.0)
        k_hi = tl.load(K + kt[:, None] * ID + HALF + hc[None, :], mask=ld[:, None], other=0.0)
    s = tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))      # [IH, BN] fp32
    w = tl.load(Wt + r * IH + hh).to(tl.float32)
    sc = tl.sum(tl.maximum(s, 0.0) * w[:, None], axis=0)
    vis = tl.load(VIS + r)
    sc = tl.where(t < vis, sc, float("-inf"))
    if HAS_CAND:                       # apply_candidates: -inf outside the candidate pool's blocks
        keep = tl.load(CAND + r * cs + t // CB, mask=ok, other=0)
        sc = tl.where(keep != 0, sc, float("-inf"))
    if KEYS:                           # topk_indices' keys: the score's bits in total order above, ~index below
        bits = tl.where(sc == 0.0, 0, sc.to(tl.int32, bitcast=True))    # (score + 0.0: -0 as +0)
        ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
        key = (ordered << 32) | (0xFFFFFFFF - t.to(tl.int64))
        tl.store(OUT + r * n + t, key, mask=ok)
    else:
        tl.store(OUT + r * n + t, sc, mask=ok)


@triton.jit
def _index_score_rows(Q, K, KS, Wt, VIS, OUT, n, rows, IH: tl.constexpr, ID: tl.constexpr, BN: tl.constexpr,
                      RB: tl.constexpr, PACKED: tl.constexpr):
    """Prompt chunks: RB rows a program share each key tile (dequantized once for all of them)."""

    rb = tl.program_id(0)
    b = tl.program_id(1)
    HALF: tl.constexpr = ID // 2
    rr = rb * RB + tl.arange(0, RB)
    rok = rr < rows
    hh = tl.arange(0, IH)
    hc = tl.arange(0, HALF)
    t = b * BN + tl.arange(0, BN)
    ok = t < n
    qrow = rr[:, None] * IH + hh[None, :]                               # [RB, IH]
    qflat = tl.reshape(qrow, (RB * IH,))
    q_lo = tl.load(Q + qflat[:, None] * ID + hc[None, :], mask=tl.reshape(tl.broadcast_to(rok[:, None], (RB, IH)), (RB * IH,))[:, None], other=0.0)
    q_hi = tl.load(Q + qflat[:, None] * ID + HALF + hc[None, :], mask=tl.reshape(tl.broadcast_to(rok[:, None], (RB, IH)), (RB * IH,))[:, None], other=0.0)
    if PACKED:
        cb = tl.load(K + t[:, None] * HALF + hc[None, :], mask=ok[:, None], other=0).to(tl.int32)
        g = tl.arange(0, HALF // 32)
        el = tl.load(KS + t[:, None] * (ID // 32) + g[None, :], mask=ok[:, None], other=127).to(tl.int32)
        eh = tl.load(KS + t[:, None] * (ID // 32) + HALF // 32 + g[None, :], mask=ok[:, None], other=127).to(tl.int32)
        sl = (el << 23).to(tl.float32, bitcast=True)
        sh = (eh << 23).to(tl.float32, bitcast=True)
        k_lo = tl.reshape(tl.reshape(_e2m1(cb & 15), (BN, HALF // 32, 32)) * sl[:, :, None], (BN, HALF)).to(tl.bfloat16)
        k_hi = tl.reshape(tl.reshape(_e2m1(cb >> 4), (BN, HALF // 32, 32)) * sh[:, :, None], (BN, HALF)).to(tl.bfloat16)
    else:
        k_lo = tl.load(K + t[:, None] * ID + hc[None, :], mask=ok[:, None], other=0.0)
        k_hi = tl.load(K + t[:, None] * ID + HALF + hc[None, :], mask=ok[:, None], other=0.0)
    s = tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))      # [RB * IH, BN]
    w = tl.load(Wt + qflat, mask=tl.reshape(tl.broadcast_to(rok[:, None], (RB, IH)), (RB * IH,)), other=0.0).to(tl.float32)
    sc = tl.sum(tl.reshape(tl.maximum(s, 0.0) * w[:, None], (RB, IH, BN)), axis=1)    # [RB, BN]
    vis = tl.load(VIS + rr, mask=rok, other=0)
    sc = tl.where(t[None, :] < vis[:, None], sc, float("-inf"))
    tl.store(OUT + rr[:, None] * n + t[None, :], sc, mask=rok[:, None] & ok[None, :])


def index_score(q: torch.Tensor, k, w: torch.Tensor, vis: torch.Tensor, n: int,
                out: torch.Tensor | None = None, base: torch.Tensor | None = None) -> torch.Tensor:
    """q [R, IH, ID] bf16, k bf16 [>= n, ID] or a packed FP4 pair (codes [N, ID/2], E8M0 [N, ID/32]),
    w [R, IH] -> score [R, n] fp32 (-inf at t >= vis[r]). Decode / verify windows (R <= DECODE_ROWS) take one row a program
    (row-invariant); prompt chunks share each key tile among 8 rows."""

    rows, ih, idim = q.shape
    if out is None:
        out = torch.empty((rows, n), dtype=torch.float32, device=q.device)
    packed = isinstance(k, tuple)
    codes, scales = k if packed else (k, k)
    bn = 64
    if rows <= DECODE_ROWS:
        _index_score[(rows, triton.cdiv(n, bn))](q, codes, scales, w, vis, out, n, vis if base is None else base,
                                                 vis, 0, IH=ih, ID=idim, BN=bn, PACKED=packed,
                                                 HAS_BASE=base is not None, num_warps=4, **_pdl())
    else:
        assert base is None, "concurrent rounds are decode windows (DECODE_ROWS rows at most)"
        rbs = 8
        _index_score_rows[(triton.cdiv(rows, rbs), triton.cdiv(n, bn))](q, codes, scales, w, vis, out, n, rows,
                                                                          IH=ih, ID=idim, BN=bn, RB=rbs,
                                                                          PACKED=packed, num_warps=8)
    return out


def index_keys(q: torch.Tensor, k, w: torch.Tensor, vis: torch.Tensor, n: int, base: torch.Tensor | None = None,
               cand: torch.Tensor | None = None, cand_block: int = 8) -> torch.Tensor:
    """topk_indices' int64 keys [R, n] of index_score's scores (decode windows), with apply_candidates' mask (``cand``:
    the pool's block mask [R, >= ceil(n / cand_block)]) folded in: one launch for index_score + apply_candidates +
    the keys' six torch ops, the same keys."""

    rows, ih, idim = q.shape
    assert rows <= DECODE_ROWS
    out = torch.empty((rows, n), dtype=torch.int64, device=q.device)
    packed = isinstance(k, tuple)
    codes, scales = k if packed else (k, k)
    bn = 64
    has_cand = cand is not None
    cv = cand.view(torch.uint8) if has_cand else vis
    _index_score[(rows, triton.cdiv(n, bn))](q, codes, scales, w, vis, out, n, vis if base is None else base,
                                             cv, cv.stride(0) if has_cand else 0, IH=ih, ID=idim, BN=bn,
                                             PACKED=packed, HAS_BASE=base is not None, num_warps=4,
                                             HAS_CAND=has_cand, CB=cand_block, KEYS=True, **_pdl())
    return out


@triton.jit
def _topk_finish(TOP, VIS, OUT, KK: tl.constexpr, PDL: tl.constexpr = False):
    """topk_indices' tail and the visible mask: the selected keys' indices sorted ascending, -1 at or past vis[r]."""

    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    j = tl.arange(0, KK)
    key = tl.load(TOP + r * KK + j)
    idx = 0xFFFFFFFF - (key & 0xFFFFFFFF)
    idx = tl.sort(idx)
    vis = tl.load(VIS + r)
    tl.store(OUT + r * KK + j, tl.where(idx < vis, idx, -1))


# (switch "topk_fused") topk_select in one launch a row: the row's n keys (unique: the score's ordered bits above, the
# inverted index below) sorted ascending, the k largest (the last k) kept, then _topk_finish's tail (their indices sorted
# ascending, -1 at or past vis[r]). The keys are a total order, so the k largest are the set torch's top-k returns.
@triton.jit
def _topk_sel(KEYS, VIS, OUT, N: tl.constexpr, KK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    key = tl.load(KEYS + r * N + tl.arange(0, N))
    srt = tl.reshape(tl.sort(key), (N // KK, KK))
    last = tl.arange(0, N // KK)[:, None] == (N // KK - 1)
    top = tl.sum(tl.where(last, srt, 0), axis=0)                 # the k largest keys (one term a column: exact)
    idx = 0xFFFFFFFF - (top & 0xFFFFFFFF)
    idx = tl.sort(idx)
    vis = tl.load(VIS + r)
    tl.store(OUT + r * KK + tl.arange(0, KK), tl.where(idx < vis, idx, -1))


TOPK_FUSED_MAX = int(os.environ.get("TF_DS_TOPK_FUSED_MAX") or 4096)   # keys a row the fused selection takes at most


def topk_select(keys: torch.Tensor, k: int, vis: torch.Tensor) -> torch.Tensor:
    """torch.where(topk_indices(score, k) < vis[:, None], ..., -1) from index_keys' keys: torch's top-k of the keys,
    then one launch for the indices, their sort and the mask (k a power of two). The same int64 [R, k]."""

    rows, n = keys.shape
    out = torch.empty((rows, k), dtype=torch.int64, device=keys.device)
    if (SMALL_SWITCHES["topk_fused"] and _pow2(n) and _pow2(k) and k <= n <= TOPK_FUSED_MAX and keys.is_contiguous()
            and keys.dtype == torch.int64):
        _topk_sel[(rows,)](keys, vis, out, N=n, KK=k, num_warps=8 if n > 1024 else 4, **_pdl())
        return out
    top = keys.topk(k, dim=-1, sorted=False).values.contiguous()
    _topk_finish[(rows,)](top, vis, out, KK=k, num_warps=4, **_pdl())
    return out


@triton.jit
def _score_keys(S, OUT, n, BN: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    t = tl.program_id(1) * BN + tl.arange(0, BN)
    ok = t < n
    sc = tl.load(S + r * n + t, mask=ok, other=0.0)
    bits = tl.where(sc == 0.0, 0, sc.to(tl.int32, bitcast=True))
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    tl.store(OUT + r * n + t, (ordered << 32) | (0xFFFFFFFF - t.to(tl.int64)), mask=ok)


def score_keys(score: torch.Tensor) -> torch.Tensor:
    """topk_indices' int64 keys of fp32 scores [R, n] (contiguous), one launch."""

    rows, n = score.shape
    out = torch.empty((rows, n), dtype=torch.int64, device=score.device)
    _score_keys[(rows, triton.cdiv(n, 1024))](score, out, n, BN=1024, num_warps=4, **_pdl())
    return out


# fp4_qd(x, 32, e4m3_scale=False) (the indexer's q): 32-element blocks, a power-of-two scale (ops._pow2_ceil of
# amax / 6, amax floored at 6 * 2^-126), values rounded to E2M1 (ties to the even code), in one launch with the bits of
# the torch ops (IEEE divisions, the scale built from its exponent bits as ldexp makes it)
@triton.jit
def _fp4qd_p2(X, OUT, nblk, BLKS: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    pb = tl.program_id(0)
    bi = pb * BLKS + tl.arange(0, BLKS)
    e = tl.arange(0, 32)
    okb = bi < nblk
    v = tl.load(X + bi[:, None] * 32 + e[None, :], mask=okb[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(v), axis=1)
    sv = tl.math.div_rn(tl.maximum(amax, 6 * 2.0 ** -126), 6.0)
    bits = sv.to(tl.int32, bitcast=True)
    ex = ((bits >> 23) & 0xFF) + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    sc = (ex << 23).to(tl.float32, bitcast=True)              # 2^(exp - 127 + (mantissa != 0))
    t = tl.minimum(tl.maximum(tl.math.div_rn(v, sc[:, None]), -6.0), 6.0)
    a = tl.abs(t)
    idx = ((a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a > 1.75).to(tl.int32)
           + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    tie = (a == 0.75) | (a == 1.75) | (a == 3.5)              # a tie at an odd code's lower midpoint: up to the even
    idx = tl.where(tie, idx + 1, idx)
    g = tl.where(idx < 4, idx.to(tl.float32) * 0.5, tl.where(idx == 4, 2.0, tl.where(idx == 5, 3.0,
                                                                                         tl.where(idx == 6, 4.0, 6.0))))
    q = (g.to(tl.int32, bitcast=True) | (t.to(tl.int32, bitcast=True) & -2147483648)).to(tl.float32, bitcast=True)
    tl.store(OUT + bi[:, None] * 32 + e[None, :], (q * sc[:, None]).to(tl.bfloat16), mask=okb[:, None])


def fp4_qd_p2(x: torch.Tensor) -> torch.Tensor:
    """ops.fp4_qd(x, 32, e4m3_scale=False) of a bf16 tensor (numel a multiple of 32), one launch."""

    out = torch.empty_like(x)
    nblk = x.numel() // 32
    blks = 32
    _fp4qd_p2[(triton.cdiv(nblk, blks),)](x, out, nblk, BLKS=blks, num_warps=4, **_pdl())
    return out


@triton.jit
def _e2m1_round(t):
    """ops._e2m1_code's magnitude index of t (already clamped to [-6, 6]): round to nearest, ties to the even code."""

    a = tl.abs(t)
    idx = ((a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a > 1.75).to(tl.int32)
           + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    return tl.where((a == 0.75) | (a == 1.75) | (a == 3.5), idx + 1, idx)


# store_rows of a packed FP4 cache (ops.fp4_pack + the two row writes) in one launch: per row, BLOCK-element blocks
# with an E4M3 scale (torch's float -> float8_e4m3fn rounding, spelled out in integer ops) or a power-of-two (E8M0)
# scale, E2M1 codes (byte j: element j low, element j + D / 2 high), written to the cache rows ROWS[r]
@triton.jit
def _fp4_store(X, xs, CODES, SCALES, ROWS, D: tl.constexpr, BLOCK: tl.constexpr, E4M3: tl.constexpr,
               PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    NB: tl.constexpr = D // BLOCK
    HALF: tl.constexpr = D // 2
    blk = tl.arange(0, NB)
    e = tl.arange(0, BLOCK)
    v = tl.load(X + r * xs + blk[:, None] * BLOCK + e[None, :]).to(tl.float32)
    amax = tl.max(tl.abs(v), axis=1)
    if E4M3:
        f = tl.math.div_rn(tl.maximum(amax, 6 * 2.0 ** -9), 6.0)
        fb = f.to(tl.int32, bitcast=True)
        sub = (f + 16384.0).to(tl.int32, bitcast=True) - (141 << 23)          # below 2^-6: the denormal add
        nor = (fb + (-120 << 23) + 0x7FFFF + ((fb >> 20) & 1)) >> 20           # RNE of the normal range
        nor = tl.where(nor == 0x7F, 0x7E, nor)
        sb = tl.where(fb >= (1087 << 20), 0x7E, tl.where(fb < (121 << 23), sub, nor))
        sc = sb.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        f = tl.math.div_rn(tl.maximum(amax, 6 * 2.0 ** -126), 6.0)
        bits = f.to(tl.int32, bitcast=True)
        sb = ((bits >> 23) & 0xFF) + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
        sc = (sb << 23).to(tl.float32, bitcast=True)
    t = tl.minimum(tl.maximum(tl.math.div_rn(v, sc[:, None]), -6.0), 6.0)
    idx = _e2m1_round(t)
    code = idx | tl.where((t < 0) & (idx > 0), 8, 0)
    pair = tl.permute(tl.reshape(code, (2, HALF)), (1, 0))                 # [HALF, 2]: element j, element j + D/2
    lo, hi = tl.split(pair)
    j = tl.arange(0, HALF)
    row = tl.load(ROWS + r)
    tl.store(CODES + row * HALF + j, (lo | (hi << 4)).to(tl.uint8))
    tl.store(SCALES + row * NB + blk, sb.to(tl.uint8))


def fp4_store(x: torch.Tensor, cache: tuple, rows: torch.Tensor, block: int, e4m3: bool) -> None:
    """model.store_rows(cache, rows, x, block, e4m3) for a packed cache, one launch, the same bytes."""

    n, d = x.shape
    _fp4_store[(n,)](x, x.stride(0), cache[0], cache[1], rows, D=d, BLOCK=block, E4M3=e4m3, num_warps=4, **_pdl())


# the round's first streams: every row's embedding copied into its hc streams, the carried pre-mix [1, 0, 0, 0]
# (embed[ids][:, None].expand(-1, hc, -1).contiguous() and pre = zeros; pre[:, 0] = 1, in one launch)
@triton.jit
def _embed_init(EMB, IDS, H, PRE, D: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    tok = tl.load(IDS + r)
    v = tl.load(EMB + tok * D + d)
    for s in tl.static_range(HC):
        tl.store(H + (r * HC + s) * D + d, v)
    if cb == 0:
        j = tl.arange(0, HC)
        tl.store(PRE + r * HC + j, tl.where(j == 0, 1.0, 0.0))


def embed_init(embed: torch.Tensor, ids: torch.Tensor, hc: int) -> tuple:
    rows = ids.shape[0]
    d = embed.shape[1]
    h = torch.empty((rows, hc, d), dtype=embed.dtype, device=ids.device)
    pre = torch.empty((rows, hc), dtype=torch.float32, device=ids.device)
    _embed_init[(rows, d // 1024)](embed, ids, h, pre, D=d, HC=hc, BLOCK=1024, num_warps=4, **_pdl())
    return h, pre


# a DSpark tap: h.to(fp32).mean(1).to(bf16) (torch's mean: ((x0 + x1) + x2) + x3, then / 4, checked bit for bit on
# streams of mixed magnitudes) written into its column block of the taps buffer (no cat)
@triton.jit
def _tap(H, OUT, os_, D: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x0 = tl.load(H + r * 4 * D + d).to(tl.float32)
    x1 = tl.load(H + r * 4 * D + D + d).to(tl.float32)
    x2 = tl.load(H + r * 4 * D + 2 * D + d).to(tl.float32)
    x3 = tl.load(H + r * 4 * D + 3 * D + d).to(tl.float32)
    tl.store(OUT + r * os_ + d, ((((x0 + x1) + x2) + x3) / 4.0).to(tl.bfloat16))


def tap(h: torch.Tensor, out: torch.Tensor) -> None:
    """out [R, D] (a column block of the taps buffer) = h.to(fp32).mean(1).to(bf16) for h [R, 4, D] bf16."""

    rows, hc, d = h.shape
    assert hc == 4
    _tap[(rows, d // 1024)](h, out, out.stride(0), D=d, BLOCK=1024, num_warps=4, **_pdl())


# -- MoE gate: sqrt(softplus(logits)), top-k by logits + bias, weights normalized and scaled; shared expert last ----
@triton.jit
def _route_row(L, BIAS, PICK, WTS, scale, shared_id, NE: tl.constexpr, NB: tl.constexpr, TOPK: tl.constexpr,
               SLOTS: tl.constexpr, SP: tl.constexpr, CG: tl.constexpr):
    e = tl.arange(0, NB)
    ok = e < NE
    if CG:                             # logits other programs just wrote: past this SM's L1
        lg = tl.load(L + e, mask=ok, other=0.0, cache_modifier=".cg")
    else:
        lg = tl.load(L + e, mask=ok, other=0.0)
    sp = tl.where(lg > 20.0, lg, tl.log(1.0 + tl.exp(lg)))
    sc = tl.sqrt(sp)
    b = tl.load(BIAS + e, mask=ok, other=0.0)
    sel = tl.where(ok, sc + b, float("-inf"))
    tot = 0.0
    s = tl.arange(0, SP)
    picks = tl.full((SP,), shared_id, dtype=tl.int32)
    ws = tl.zeros((SP,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(sel, axis=0)
        idx = tl.min(tl.where(sel == best, e, NB), axis=0)          # lowest index among ties
        v = tl.sum(tl.where(e == idx, sc, 0.0), axis=0)
        picks = tl.where(s == k, idx, picks)
        ws = tl.where(s == k, v, ws)
        tot += v
        sel = tl.where(e == idx, float("-inf"), sel)
    ws = tl.where(s < TOPK, ws / (tot + 1e-20) * scale, 1.0)
    tl.store(PICK + s, picks, mask=s < SLOTS)
    tl.store(WTS + s, ws, mask=s < SLOTS)


@triton.jit
def _route(L, BIAS, PICK, WTS, scale, shared_id, NE: tl.constexpr, NB: tl.constexpr, TOPK: tl.constexpr,
           SLOTS: tl.constexpr, SP: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        _prefetch_l2(BIAS + tl.minimum(tl.arange(0, NB // 32) * 32, NE - 1))   # the bias (a weight) while waiting
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    _route_row(L + r * NE, BIAS, PICK + r * SLOTS, WTS + r * SLOTS, scale, shared_id, NE, NB, TOPK, SLOTS, SP, False)


# the router matmul as chunk sums (switch "rowmm_parts"): S[r, t, n] = the dot of chunk t (BK columns) of x[r] and
# w[n], a tile [BN, BK] over NW warps (BN / NW outputs a warp, each output's chunk 8 columns a lane and the same
# butterfly as in rowmm2, so each chunk's sum has the bits it has inside rowmm2; x's chunk loaded as a tile of that
# layout, no layout conversion), programs over (output blocks, groups of CPG chunks), every row of the window in a
# program (RB == rows: its weight chunks loaded once, before the PDL wait; every store after every load); the route
# kernel adds a row's chunk sums in K order from zero (rowmm2's acc: the same adds in the same order, so the same
# logits) before routing
@triton.jit
def _rowmm_parts(X, xs, W, S, rows, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 CPG: tl.constexpr, RB: tl.constexpr, PDL: tl.constexpr = False):
    nb = tl.program_id(0)
    g = tl.program_id(1)
    nn = nb * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    KC: tl.constexpr = K // BK
    ws = ()
    for c in tl.static_range(CPG):
        ws = ws + (tl.load(W + nn[:, None] * K + ((g * CPG + c) * BK + kk)[None, :], mask=(nn < N)[:, None],
                           other=0.0),)
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    out = ()
    for i in tl.static_range(RB):                      # RB == rows (a variant a row count)
        for c in tl.static_range(CPG):
            # x's chunk as a [BN, BK] tile (every warp the same row: the tile's own layout, no conversion)
            x = tl.load(X + i * xs + ((g * CPG + c) * BK + kk)[None, :] + 0 * nn[:, None]).to(tl.float32)
            out = out + (tl.sum(ws[c].to(tl.float32) * x, axis=1),)
    for i in tl.static_range(RB):
        for c in tl.static_range(CPG):
            tl.store(S + (i * KC + g * CPG + c) * N + nn, out[i * CPG + c], mask=nn < N)


@triton.jit
def _route_parts(S, BIAS, PICK, WTS, scale, shared_id, NE: tl.constexpr, NB: tl.constexpr, TOPK: tl.constexpr,
                 SLOTS: tl.constexpr, SP: tl.constexpr, KC: tl.constexpr, PDL: tl.constexpr = False):
    """_route of a row whose logits are rowmm_parts' chunk sums: the logits are summed first (from zero, chunk 0
    first: rowmm2's accumulation), then _route_row's code."""

    if PDL:
        _prefetch_l2(BIAS + tl.minimum(tl.arange(0, NB // 32) * 32, NE - 1))   # the bias (a weight) while waiting
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    e = tl.arange(0, NB)
    ok = e < NE
    lg = tl.zeros((NB,), dtype=tl.float32)
    for t in tl.static_range(KC):
        lg += tl.load(S + (r * KC + t) * NE + e, mask=ok, other=0.0)
    PICK = PICK + r * SLOTS
    WTS = WTS + r * SLOTS
    sp = tl.where(lg > 20.0, lg, tl.log(1.0 + tl.exp(lg)))
    sc = tl.sqrt(sp)
    b = tl.load(BIAS + e, mask=ok, other=0.0)
    sel = tl.where(ok, sc + b, float("-inf"))
    tot = 0.0
    s = tl.arange(0, SP)
    picks = tl.full((SP,), shared_id, dtype=tl.int32)
    ws = tl.zeros((SP,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(sel, axis=0)
        idx = tl.min(tl.where(sel == best, e, NB), axis=0)          # lowest index among ties
        v = tl.sum(tl.where(e == idx, sc, 0.0), axis=0)
        picks = tl.where(s == k, idx, picks)
        ws = tl.where(s == k, v, ws)
        tot += v
        sel = tl.where(e == idx, float("-inf"), sel)
    ws = tl.where(s < TOPK, ws / (tot + 1e-20) * scale, 1.0)
    tl.store(PICK + s, picks, mask=s < SLOTS)
    tl.store(WTS + s, ws, mask=s < SLOTS)


# rowmm_parts' (outputs, chunks, warps) a program by rows (a one-GPU sweep of tilings at 1-16 rows);
# TF_DS_ROWMM_PARTS="BN,CPG,NW" sets one for every row count (benchmarks)
_RP_ENV = os.environ.get("TF_DS_ROWMM_PARTS")
_RP = [int(v) for v in _RP_ENV.split(",")] if _RP_ENV else None
# rowmm_parts serves windows of ROWMM_PARTS_MIN rows or more (TF_DS_ROWMM_PARTS_MIN, default 1)
ROWMM_PARTS_MIN = int(os.environ.get("TF_DS_ROWMM_PARTS_MIN") or 1)


def _rp_tile(rows: int) -> tuple:
    if _RP is not None:
        return tuple(_RP)
    return (16, 1, 2) if rows <= 6 else (8, 1, 2)


def rowmm_gate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The MoE gate's logits for route(): rowmm2(x, w) [R, N], or (switch "rowmm_parts", decode windows) its chunk
    sums [R, K / 256, N] (route() adds them in K order: the same logits)."""

    rows, k = x.shape
    n = w.shape[0]
    bn, cpg, nw = _rp_tile(rows)
    kc = k // 256
    if (SMALL_SWITCHES["rowmm_parts"] and ROWMM_PARTS_MIN <= rows <= DECODE_ROWS and k % 256 == 0 and kc % cpg == 0
            and n >= 32 and x.dtype == torch.bfloat16 and w.dtype == torch.float16):
        out = torch.empty((rows, kc, n), dtype=torch.float32, device=x.device)
        _rowmm_parts[(triton.cdiv(n, bn), kc // cpg)](x, x.stride(0), w, out, rows, K=k, N=n, BN=bn, BK=256, CPG=cpg,
                                                      RB=rows, num_warps=nw, **_pdl())
        return out
    return rowmm2(x, w)


def route(logits: torch.Tensor, bias: torch.Tensor, topk: int, scale: float, shared_id: int,
          pick: torch.Tensor, wts: torch.Tensor) -> None:
    if logits.dim() == 3:                                   # rowmm_gate's chunk sums
        rows, kc, ne = logits.shape
        _route_parts[(rows,)](logits, bias, pick, wts, scale, shared_id, NE=ne, NB=triton.next_power_of_2(ne),
                              TOPK=topk, SLOTS=pick.shape[1], SP=triton.next_power_of_2(pick.shape[1]), KC=kc,
                              num_warps=1, **_pdl())
        return
    rows, ne = logits.shape
    # one warp a row (switch "rowmm"): its reductions are a max, a min index and a one-element sum, exact in any
    # layout, so the picks and weights are the 4-warp launch's; no cross-warp barriers in its six rounds
    nw = 1 if SMALL_SWITCHES["rowmm"] else 4
    _route[(rows,)](logits, bias, pick, wts, scale, shared_id, NE=ne, NB=triton.next_power_of_2(ne), TOPK=topk,
                    SLOTS=pick.shape[1], SP=triton.next_power_of_2(pick.shape[1]), num_warps=nw, **_pdl())


# -- row-invariant small matmul y[r] = x[r] @ W^T (fp32 accumulate in fixed K order): router, hc, indexer weights ---
@triton.jit
def _rowmm(X, xs, W, OUT, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    nb = tl.program_id(1)
    nn = nb * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + r * xs + k0 + kk).to(tl.float32)
        w = tl.load(W + nn[:, None] * K + (k0 + kk)[None, :], mask=(nn < N)[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + r * N + nn, acc, mask=nn < N)


def rowmm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [R, K] @ w[N, K]^T -> fp32 [R, N], a row alone."""

    rows, k = x.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty((rows, n), dtype=torch.float32, device=x.device)
    bn = 8
    _rowmm[(rows, triton.cdiv(n, bn))](x, x.stride(0), w, out, K=k, N=n, BN=bn, BK=256, num_warps=4, **_pdl())
    return out


# -- the final collapse: RMSNorm(sum_j pre[j] h_j) (no mixes) ---------------------------------------------------------
@triton.jit
def _collapse_norm(X, PRE, NW, OUT, eps, D: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    p0 = tl.load(PRE + r * 4 + 0)
    p1 = tl.load(PRE + r * 4 + 1)
    p2 = tl.load(PRE + r * 4 + 2)
    p3 = tl.load(PRE + r * 4 + 3)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c0 in range(0, D, BLOCK):
        d = c0 + tl.arange(0, BLOCK)
        c = (((p0 * tl.load(X + r * (4 * D) + d).to(tl.float32) + p1 * tl.load(X + r * (4 * D) + D + d).to(tl.float32))
              + p2 * tl.load(X + r * (4 * D) + 2 * D + d).to(tl.float32))
             + p3 * tl.load(X + r * (4 * D) + 3 * D + d).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        acc += c * c
    rinv = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / D + eps)
    for c0 in range(0, D, BLOCK):
        d = c0 + tl.arange(0, BLOCK)
        c = (((p0 * tl.load(X + r * (4 * D) + d).to(tl.float32) + p1 * tl.load(X + r * (4 * D) + D + d).to(tl.float32))
              + p2 * tl.load(X + r * (4 * D) + 2 * D + d).to(tl.float32))
             + p3 * tl.load(X + r * (4 * D) + 3 * D + d).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + r * D + d, (tl.load(NW + d).to(tl.float32) * (c * rinv)).to(tl.bfloat16))


def collapse_norm(h: torch.Tensor, pre: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    rows, _, d = h.shape
    out = torch.empty((rows, d), dtype=torch.bfloat16, device=h.device)
    _collapse_norm[(rows,)](h, pre, w, out, eps, D=d, BLOCK=1024, num_warps=8, **_pdl())
    return out


@triton.jit
def _collapse(X, PRE, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    p0 = tl.load(PRE + r * 4 + 0)
    p1 = tl.load(PRE + r * 4 + 1)
    p2 = tl.load(PRE + r * 4 + 2)
    p3 = tl.load(PRE + r * 4 + 3)
    c = (((p0 * tl.load(X + r * (4 * D) + d).to(tl.float32) + p1 * tl.load(X + r * (4 * D) + D + d).to(tl.float32))
          + p2 * tl.load(X + r * (4 * D) + 2 * D + d).to(tl.float32))
         + p3 * tl.load(X + r * (4 * D) + 3 * D + d).to(tl.float32))
    tl.store(OUT + r * D + d, c.to(tl.bfloat16))


def collapse(h: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    rows, _, d = h.shape
    out = torch.empty((rows, d), dtype=torch.bfloat16, device=h.device)
    _collapse[(rows, d // 1024)](h, pre, out, D=d, BLOCK=1024, num_warps=4)
    return out


# -- Engram gate: per (row, stream) normalized dot of the stream with its key, signed sqrt, sigmoid; h + gate * v ----
@triton.jit
def _engram_gate(H, KV, QK, OUT, eps, D: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    s = tl.program_id(1)
    sh = tl.zeros((BLOCK,), dtype=tl.float32)
    sk = tl.zeros((BLOCK,), dtype=tl.float32)
    sd = tl.zeros((BLOCK,), dtype=tl.float32)
    for c0 in range(0, D, BLOCK):
        d = c0 + tl.arange(0, BLOCK)
        h = tl.load(H + r * (4 * D) + s * D + d).to(tl.float32)
        k = tl.load(KV + r * (5 * D) + s * D + d).to(tl.float32)
        w = tl.load(QK + s * D + d)
        sh += h * h
        sk += k * k
        sd += h * w * k
    rstd = (1.0 / tl.sqrt(tl.sum(sh, axis=0) / D + eps)) * (1.0 / tl.sqrt(tl.sum(sk, axis=0) / D + eps))
    dot = tl.sum(sd, axis=0) * rstd * (1.0 / tl.sqrt(D * 1.0))
    mag = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    sg = tl.where(dot < 0, -mag, mag)
    gate = 1.0 / (1.0 + tl.exp(-sg))
    for c0 in range(0, D, BLOCK):
        d = c0 + tl.arange(0, BLOCK)
        h = tl.load(H + r * (4 * D) + s * D + d).to(tl.float32)
        v = tl.load(KV + r * (5 * D) + 4 * D + d).to(tl.float32)
        tl.store(OUT + r * (4 * D) + s * D + d, (h + gate * v).to(tl.bfloat16))


def engram_gate(h: torch.Tensor, kv: torch.Tensor, qk: torch.Tensor, eps: float) -> torch.Tensor:
    """h [R, 4, D] bf16, kv [R, 5 * D] bf16 (4 keys then the value), qk [4, D] f32 -> new h."""

    rows, _, d = h.shape
    out = torch.empty_like(h)
    _engram_gate[(rows, 4)](h, kv, qk, out, eps, D=d, BLOCK=1024, num_warps=4, **_pdl())
    return out


def topk_indices(score: torch.Tensor, k: int) -> torch.Tensor:
    """Each row's k highest scores' indices, ascending; ties go to the lower index. A total order (the score's bits
    above, the inverted index below, one int64 key), so the set is the same whatever the row's width, the rows beside
    it or torch's algorithm: a stream's selection in a concurrent round is its solo selection."""

    bits = (score + 0.0).view(torch.int32)                       # + 0.0: -0 becomes +0
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
    keys = (ordered << 32) | (0xFFFFFFFF - torch.arange(score.shape[-1], device=score.device, dtype=torch.int64))
    top = keys.topk(k, dim=-1, sorted=False).values
    return (0xFFFFFFFF - (top & 0xFFFFFFFF)).sort(dim=-1).values

