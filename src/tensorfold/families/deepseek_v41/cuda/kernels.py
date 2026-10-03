"""Row-independent Triton kernels for DeepSeek-V4.1 decode and prompt chunks.

Every kernel works on one row at a time (or a fixed tile of heads within a row), with fixed-order reductions, so a
row's bits never depend on how many rows share the call: a verify window reproduces serial decode exactly.
Rounding follows DeepSeek's reference (fp32 math, bf16 where its tensors are bf16).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

HC_BLOCKS = 40          # fixed K split of the mHC mixing dots (a function of the shape only)


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
def _hc_post(G, RS, X, XOUT, POST, COMB, D: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    """y = bf16(rank-ordered sum of partials); out_k = post_k y + sum_j comb[j, k] x_j (fp32) -> bf16."""

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
                                 num_warps=4)


# -- RMSNorm (DeepSeek: bf16(w * (x * rsqrt(mean(x^2) + eps)))) ----------------------------------------------------
@triton.jit
def _rmsnorm(X, xs, W, OUT, os_, eps, D: tl.constexpr, BLOCK: tl.constexpr):
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
                      num_warps=4 if d <= 2048 else 8)
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
                  D: tl.constexpr, RD: tl.constexpr):
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
                           num_warps=4)
    return out


# -- RoPE on the last RD dims of every head (q forward, attention output inverse) -----------------------------------
@triton.jit
def _rope_heads(X, COS, SIN, POS, H: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr, INV: tl.constexpr,
                HB: tl.constexpr):
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
    _rope_heads[(rows, h // hb)](x, cos, sin, pos, H=h, HD=hd, RD=rd, INV=inverse, HB=hb, num_warps=4)
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
def _sparse_attn_part(Q, WSRC, WLO, COMP, CSC, IDX, POS, PM, PL, PO, SINK, OUT, scale, ring_size, n_idx,
                      H: tl.constexpr, HD: tl.constexpr, HB: tl.constexpr, WIN: tl.constexpr, BN: tl.constexpr,
                      RING: tl.constexpr, HAS_COMP: tl.constexpr, PACKED: tl.constexpr, SPLITS: tl.constexpr,
                      NBLK: tl.constexpr, FINAL: tl.constexpr):
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
            kb = WSRC + slot[:, None] * HD
            k_lo = tl.load(kb + hc[None, :], mask=ok[:, None], other=0.0)
            k_hi = tl.load(kb + HALF + hc[None, :], mask=ok[:, None], other=0.0)
        else:
            t = (b - WB) * BN
            ii = tl.load(IDX + r * n_idx + t + n, mask=(t + n) < n_idx, other=-1)
            ok = ii >= 0
            k_lo, k_hi = _comp_keys(COMP, CSC, tl.where(ok, ii, 0), ok, hc, HD, BN, PACKED)
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
                       SPLITS: tl.constexpr):
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


ATTN_SPLITS = 8


def sparse_attn(q: torch.Tensor, sink: torch.Tensor, wsrc: torch.Tensor, wlo: torch.Tensor, ring: bool,
                comp, idx: torch.Tensor | None, pos: torch.Tensor, scale: float, window: int,
                out: torch.Tensor | None = None) -> torch.Tensor:
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
    nblk = window // bn + (triton.cdiv(n_idx, bn) if has else 0)
    # decode / verify windows split the keys (parallelism for a few rows); prompt chunks have rows enough
    sp = ATTN_SPLITS if rows <= 16 else 1
    groups = h // hb
    final = sp == 1
    if final:
        pm = pl = po = out
    else:
        pm = torch.empty((rows * groups * sp * hb,), dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        po = torch.empty((rows * groups * sp * hb * hd,), dtype=torch.float32, device=q.device)
    _sparse_attn_part[(rows, groups, sp)](q, wsrc, wlo, codes, scales if packed else wsrc, idx if has else pos, pos,
                                          pm, pl, po, sink, out, scale, wsrc.shape[0], n_idx, H=h, HD=hd, HB=hb,
                                          WIN=window, BN=bn, RING=ring, HAS_COMP=has, PACKED=packed, SPLITS=sp,
                                          NBLK=nblk, FINAL=final, num_warps=4, num_stages=1)
    if not final:
        _sparse_attn_merge[(rows, groups)](pm, pl, po, sink, out, H=h, HD=hd, HB=hb, SPLITS=sp, num_warps=8)
    return out


# -- indexer scores: sum_h relu(q_h . k_t) w_h over t < n, masked past each row's visible count -------------------
@triton.jit
def _index_score(Q, K, KS, Wt, VIS, OUT, n, IH: tl.constexpr, ID: tl.constexpr, BN: tl.constexpr,
                 PACKED: tl.constexpr):
    r = tl.program_id(0)
    b = tl.program_id(1)
    HALF: tl.constexpr = ID // 2
    hh = tl.arange(0, IH)
    hc = tl.arange(0, HALF)
    t = b * BN + tl.arange(0, BN)
    ok = t < n
    q_lo = tl.load(Q + r * (IH * ID) + hh[:, None] * ID + hc[None, :])
    q_hi = tl.load(Q + r * (IH * ID) + hh[:, None] * ID + HALF + hc[None, :])
    if PACKED:
        # FP4 rows: byte j = element j and j + ID/2; a power-of-two (E8M0) scale per 32 elements
        cb = tl.load(K + t[:, None] * HALF + hc[None, :], mask=ok[:, None], other=0).to(tl.int32)
        g = tl.arange(0, HALF // 32)
        el = tl.load(KS + t[:, None] * (ID // 32) + g[None, :], mask=ok[:, None], other=127).to(tl.int32)
        eh = tl.load(KS + t[:, None] * (ID // 32) + HALF // 32 + g[None, :], mask=ok[:, None], other=127).to(tl.int32)
        sl = (el << 23).to(tl.float32, bitcast=True)                    # E8M0 byte -> 2^(byte - 127)
        sh = (eh << 23).to(tl.float32, bitcast=True)
        k_lo = tl.reshape(tl.reshape(_e2m1(cb & 15), (BN, HALF // 32, 32)) * sl[:, :, None], (BN, HALF)).to(tl.bfloat16)
        k_hi = tl.reshape(tl.reshape(_e2m1(cb >> 4), (BN, HALF // 32, 32)) * sh[:, :, None], (BN, HALF)).to(tl.bfloat16)
    else:
        k_lo = tl.load(K + t[:, None] * ID + hc[None, :], mask=ok[:, None], other=0.0)
        k_hi = tl.load(K + t[:, None] * ID + HALF + hc[None, :], mask=ok[:, None], other=0.0)
    s = tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))      # [IH, BN] fp32
    w = tl.load(Wt + r * IH + hh).to(tl.float32)
    sc = tl.sum(tl.maximum(s, 0.0) * w[:, None], axis=0)
    vis = tl.load(VIS + r)
    sc = tl.where(t < vis, sc, float("-inf"))
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
                out: torch.Tensor | None = None) -> torch.Tensor:
    """q [R, IH, ID] bf16, k bf16 [>= n, ID] or a packed FP4 pair (codes [N, ID/2], E8M0 [N, ID/32]),
    w [R, IH] -> score [R, n] fp32 (-inf at t >= vis[r]). Decode / verify windows (R <= 16) take one row a program
    (row-invariant); prompt chunks share each key tile among 8 rows."""

    rows, ih, idim = q.shape
    if out is None:
        out = torch.empty((rows, n), dtype=torch.float32, device=q.device)
    packed = isinstance(k, tuple)
    codes, scales = k if packed else (k, k)
    bn = 64
    if rows <= 16:
        _index_score[(rows, triton.cdiv(n, bn))](q, codes, scales, w, vis, out, n, IH=ih, ID=idim, BN=bn,
                                                 PACKED=packed, num_warps=4)
    else:
        rbs = 8
        _index_score_rows[(triton.cdiv(rows, rbs), triton.cdiv(n, bn))](q, codes, scales, w, vis, out, n, rows,
                                                                          IH=ih, ID=idim, BN=bn, RB=rbs,
                                                                          PACKED=packed, num_warps=8)
    return out


# -- MoE gate: sqrt(softplus(logits)), top-k by logits + bias, weights normalized and scaled; shared expert last ----
@triton.jit
def _route(L, BIAS, PICK, WTS, scale, shared_id, NE: tl.constexpr, NB: tl.constexpr, TOPK: tl.constexpr,
           SLOTS: tl.constexpr, SP: tl.constexpr):
    r = tl.program_id(0)
    e = tl.arange(0, NB)
    ok = e < NE
    lg = tl.load(L + r * NE + e, mask=ok, other=0.0)
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
    tl.store(PICK + r * SLOTS + s, picks, mask=s < SLOTS)
    tl.store(WTS + r * SLOTS + s, ws, mask=s < SLOTS)


def route(logits: torch.Tensor, bias: torch.Tensor, topk: int, scale: float, shared_id: int,
          pick: torch.Tensor, wts: torch.Tensor) -> None:
    rows, ne = logits.shape
    _route[(rows,)](logits, bias, pick, wts, scale, shared_id, NE=ne, NB=triton.next_power_of_2(ne), TOPK=topk,
                    SLOTS=pick.shape[1], SP=triton.next_power_of_2(pick.shape[1]), num_warps=4)


# -- row-invariant small matmul y[r] = x[r] @ W^T (fp32 accumulate in fixed K order): router, hc, indexer weights ---
@triton.jit
def _rowmm(X, xs, W, OUT, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    _rowmm[(rows, triton.cdiv(n, bn))](x, x.stride(0), w, out, K=k, N=n, BN=bn, BK=256, num_warps=4)
    return out


# -- the final collapse: RMSNorm(sum_j pre[j] h_j) (no mixes) ---------------------------------------------------------
@triton.jit
def _collapse_norm(X, PRE, NW, OUT, eps, D: tl.constexpr, BLOCK: tl.constexpr):
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
    _collapse_norm[(rows,)](h, pre, w, out, eps, D=d, BLOCK=1024, num_warps=8)
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
def _engram_gate(H, KV, QK, OUT, eps, D: tl.constexpr, BLOCK: tl.constexpr):
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
    _engram_gate[(rows, 4)](h, kv, qk, out, eps, D=d, BLOCK=1024, num_warps=4)
    return out
