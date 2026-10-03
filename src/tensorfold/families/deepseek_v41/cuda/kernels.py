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

HC_BLOCKS = 16          # fixed K split of the mHC mixing dots (a function of the shape only)


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
    _hc_partial[(rows, HC_BLOCKS)](h, fn, part, WIDE=wide, NB=HC_BLOCKS, SUB=128, num_warps=4)
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
@triton.jit
def _sparse_attn(Q, SINK, WSRC, WLO, COMP, IDX, POS, OUT, scale, ring_size, n_idx,
                 H: tl.constexpr, HD: tl.constexpr, HB: tl.constexpr, WIN: tl.constexpr, BN: tl.constexpr,
                 RING: tl.constexpr, HAS_COMP: tl.constexpr):
    r = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb * HB + tl.arange(0, HB)
    dcol = tl.arange(0, HD)
    q = tl.load(Q + r * (H * HD) + h[:, None] * HD + dcol[None, :])
    p = tl.load(POS + r)
    wlo = tl.load(WLO)
    m_i = tl.full((HB,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((HB,), dtype=tl.float32)
    acc = tl.zeros((HB, HD), dtype=tl.float32)
    n = tl.arange(0, BN)
    # window: positions p - WIN + 1 .. p
    for t in range(0, WIN, BN):
        wp = p - (WIN - 1) + t + n
        ok = wp >= 0
        if RING:
            slot = wp % ring_size
        else:
            slot = wp - wlo
            ok = ok & (slot >= 0)
        slot = tl.where(ok, slot, 0)
        k = tl.load(WSRC + slot[:, None] * HD + dcol[None, :], mask=ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        pr = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pr, axis=1)
        acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), k)
        m_i = m_new
    if HAS_COMP:
        for t in range(0, n_idx, BN):
            ii = tl.load(IDX + r * n_idx + t + n, mask=(t + n) < n_idx, other=-1)
            ok = ii >= 0
            k = tl.load(COMP + tl.where(ok, ii, 0)[:, None] * HD + dcol[None, :], mask=ok[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            s = tl.where(ok[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            pr = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(pr, axis=1)
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), k)
            m_i = m_new
    sink = tl.load(SINK + h)
    l_i = l_i + tl.exp(sink - m_i)
    o = acc / l_i[:, None]
    tl.store(OUT + r * (H * HD) + h[:, None] * HD + dcol[None, :], o.to(tl.bfloat16))


def sparse_attn(q: torch.Tensor, sink: torch.Tensor, wsrc: torch.Tensor, wlo: torch.Tensor, ring: bool,
                comp: torch.Tensor | None, idx: torch.Tensor | None, pos: torch.Tensor, scale: float, window: int,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """q [R, H, HD] bf16 -> o [R, H, HD]; window keys from ``wsrc`` (a ring: slot = position % size; else linear from
    position wlo[0]); compressed keys comp[idx[r, j]] (idx -1 = none)."""

    rows, h, hd = q.shape
    if out is None:
        out = torch.empty_like(q)
    hb = 16
    has = comp is not None and idx is not None and idx.shape[1] > 0
    n_idx = idx.shape[1] if has else 0
    _sparse_attn[(rows, h // hb)](q, sink, wsrc, wlo, comp if has else wsrc, idx if has else pos, pos, out, scale,
                                  wsrc.shape[0], n_idx, H=h, HD=hd, HB=hb, WIN=window, BN=32, RING=ring,
                                  HAS_COMP=has, num_warps=8, num_stages=1)
    return out


# -- indexer scores: sum_h relu(q_h . k_t) w_h over t < n, masked past each row's visible count -------------------
@triton.jit
def _index_score(Q, K, Wt, VIS, OUT, n, IH: tl.constexpr, ID: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0)
    b = tl.program_id(1)
    hh = tl.arange(0, IH)
    dd = tl.arange(0, ID)
    t = b * BN + tl.arange(0, BN)
    q = tl.load(Q + r * (IH * ID) + hh[:, None] * ID + dd[None, :])
    k = tl.load(K + t[:, None] * ID + dd[None, :], mask=(t < n)[:, None], other=0.0)
    s = tl.dot(q, tl.trans(k))                                   # [IH, BN] fp32
    w = tl.load(Wt + r * IH + hh).to(tl.float32)
    sc = tl.sum(tl.maximum(s, 0.0) * w[:, None], axis=0)
    vis = tl.load(VIS + r)
    sc = tl.where(t < vis, sc, float("-inf"))
    tl.store(OUT + r * n + t, sc, mask=t < n)


def index_score(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, vis: torch.Tensor, n: int,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """q [R, IH, ID] bf16, k [>= n, ID] bf16, w [R, IH] -> score [R, n] fp32 (-inf at t >= vis[r])."""

    rows, ih, idim = q.shape
    if out is None:
        out = torch.empty((rows, n), dtype=torch.float32, device=q.device)
    bn = 64
    _index_score[(rows, triton.cdiv(n, bn))](q, k, w, vis, out, n, IH=ih, ID=idim, BN=bn, num_warps=4)
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
    bn = 32
    _rowmm[(rows, triton.cdiv(n, bn))](x, x.stride(0), w, out, K=k, N=n, BN=bn, BK=128, num_warps=4)
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
