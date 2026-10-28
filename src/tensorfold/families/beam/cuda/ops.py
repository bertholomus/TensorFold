"""Beam layer kernels around the projections: per-head q/k RMSNorm with rotary (window layers) or SSMax (full layers)
plus the KV write, the sigmoid output gate, the sandwich-norm residual boundary, and the fp32 router.

Rounding follows Reflection's Beam model (Apache-2.0): values the reference rounds to bf16 are rounded here at the same
points; normalization, rotary, SSMax and routing run in fp32.
"""
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ROPE, SSMAX = 1, 2


@triton.jit
def _qk_prep(Q, K, V, QW, KW, POS, INV, OQ, KD, VD, DST, R,
             sq, sk, sv, soq, skd, svd,
             HQ: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr, MODE: tl.constexpr, TOK: tl.constexpr):
    t = tl.program_id(0) * TOK + tl.arange(0, TOK)
    h = tl.program_id(1)
    c = tl.arange(0, D)
    half: tl.constexpr = D // 2
    pc = tl.where(c < half, c + half, c - half)                     # rotary partner column (NeoX halves)
    tm = t < R
    t64 = t.to(tl.int64)
    if h < HQ:
        src = Q + t64[:, None] * sq + h * D
        w = tl.load(QW + c).to(tl.float32)
        wp = tl.load(QW + pc).to(tl.float32)
    else:
        src = K + t64[:, None] * sk + (h - HQ) * D
        w = tl.load(KW + c).to(tl.float32)
        wp = tl.load(KW + pc).to(tl.float32)
    x = tl.load(src + c[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    y = (x * rstd[:, None] * w[None, :]).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(POS + t, mask=tm, other=0)
    if MODE == 1:
        xp = tl.load(src + pc[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        yp = (xp * rstd[:, None] * wp[None, :]).to(tl.bfloat16).to(tl.float32)
        f = pos.to(tl.float32)[:, None] * tl.load(INV + c % half)[None, :]
        cs = libdevice.cos(f)
        sn = libdevice.sin(f)
        y = tl.where(c[None, :] >= half, tl.fma(yp, sn, y * cs), tl.fma(y, cs, -yp * sn))
    elif MODE == 2:
        if h < HQ:
            p = pos.to(tl.float32) + 1.0
            y = y * libdevice.exp(0.1 * libdevice.log1p(p / 2048.0))[:, None]
    if h < HQ:
        tl.store(OQ + t64[:, None] * soq + h * D + c[None, :], y.to(OQ.dtype.element_ty), mask=tm[:, None])
    else:
        slot = tl.load(DST + t, mask=tm, other=-1).to(tl.int64)
        ok = tm[:, None] & (slot[:, None] >= 0)
        tl.store(KD + slot[:, None] * skd + (h - HQ) * D + c[None, :], y.to(KD.dtype.element_ty), mask=ok)
        v = tl.load(V + t64[:, None] * sv + (h - HQ) * D + c[None, :], mask=ok, other=0.0)
        tl.store(VD + slot[:, None] * svd + (h - HQ) * D + c[None, :], v.to(VD.dtype.element_ty), mask=ok)


def inv_freq(theta: float = 1e4, dim: int = 128, device="cuda") -> torch.Tensor:
    """fp32 rotary frequencies, computed the way vLLM's rotary cache computes them."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))


def qk_prep(q, k, v, qw, kw, pos, inv, q_out, k_dst, v_dst, dst, *, mode: int, eps: float):
    """q [R, HQ*128], k/v [R, HKV*128] (row-strided bf16 views). Writes q_out [R, HQ, 128] (normed, rotated or
    SSMax-scaled) and k (normed, rotated) / v into k_dst/v_dst [slots, HKV, 128] at slot dst[i] (dst < 0 skips)."""
    R = q.shape[0]
    D = 128
    HQ = q.shape[1] // D
    HKV = k.shape[1] // D
    if R == 0:
        return q_out
    tok = 4
    _qk_prep[(triton.cdiv(R, tok), HQ + HKV)](
        q, k, v, qw, kw, pos, inv, q_out, k_dst, v_dst, dst, R,
        q.stride(0), k.stride(0), v.stride(0), q_out.stride(0), k_dst.stride(0), v_dst.stride(0),
        HQ=HQ, D=D, EPS=eps, MODE=mode, TOK=tok, num_warps=4)
    return q_out


@triton.jit
def _gate(O, G, B, Y, N, so, sg, sy, HQ: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    g = tl.load(G + r * sg + i, mask=m, other=0.0).to(tl.float32)
    b = tl.load(B + (i // D) % HQ, mask=m, other=0.0).to(tl.float32)
    g = (g + b).to(tl.bfloat16).to(tl.float32)
    s = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    o = tl.load(O + r * so + i, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + r * sy + i, (o * s).to(Y.dtype.element_ty), mask=m)


def gate(o, g, b, out=None):
    """out = bf16(o * bf16(sigmoid(bf16(g + b[head])))) for o, g [R, HQ*128]."""
    R, N = o.shape
    out = torch.empty_like(o) if out is None else out
    if R:
        _gate[(R, triton.cdiv(N, 1024))](o, g, b, out, N, o.stride(0), g.stride(0), out.stride(0),
                                        HQ=b.shape[0], D=128, BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _boundary(X, X1, R_, WP, WN, H, Y, YF, N, sx, sr, sp, xs, rs, EPS: tl.constexpr, NEXT: tl.constexpr,
              F32: tl.constexpr, KEEP_F32: tl.constexpr, PARTS: tl.constexpr, APART: tl.constexpr,
              BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    m = c < N
    x = tl.load(X + row * sx + c, mask=m, other=0.0).to(tl.float32)
    if APART:                                     # two ranks' partials in two tensors, X's first (rank order)
        x = x + tl.load(X1 + row * sx + c, mask=m, other=0.0).to(tl.float32)
    for p in tl.static_range(1, PARTS):           # the ranks' fp32 partials added in rank order (Work.reduce's sum)
        x = x + tl.load(X + p * sp + row * sx + c, mask=m, other=0.0).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 0) / N + EPS) * tl.load(WP + c, mask=m, other=0.0).to(tl.float32)
    r = tl.load(R_ + row * sr + c, mask=m, other=0.0).to(tl.float32)
    if F32:                                       # an fp32 residual stream: no roundings before the next norm
        h = x * xs + r * rs
    else:                                         # Reflection's bf16 stream: each piece rounded as the model does
        x = x.to(tl.bfloat16).to(tl.float32)
        x = (x * xs).to(tl.bfloat16).to(tl.float32)
        r = (r * rs).to(tl.bfloat16).to(tl.float32)
        h = (x + r).to(tl.bfloat16)
    tl.store(H + row * N + c, h.to(H.dtype.element_ty), mask=m)
    if NEXT:
        hf = tl.where(m, h.to(tl.float32), 0.0)
        y = hf * tl.rsqrt(tl.sum(hf * hf, 0) / N + EPS) * tl.load(WN + c, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + row * N + c, y.to(tl.bfloat16), mask=m)
        if KEEP_F32:                              # the router's input, unrounded
            tl.store(YF + row * N + c, y, mask=m)


def scales(layer: int, attention: bool) -> tuple[float, float]:
    """Depth-scaled residual coefficients: n counts residual adds through this one (2L, 2L+1 with L = layer + 1)."""
    n = 2 * (layer + 1) + (0 if attention else 1)
    return 1.0 / math.sqrt(n), math.sqrt(n - 1) / math.sqrt(n)


def boundary(x, r, w_post, w_next, sc, eps, h_out=None, y_out=None, yf_out=None):
    """h = bf16(bf16(bf16(rms(x) * w_post) * xs) + bf16(r * rs)); y = bf16(rms(h) * w_next) (w_next None: no y).
    x: the sublayer output (fp32 sum of the rank partials, or bf16), or the ranks' fp32 partials [world, T, N]
    (Work.parts), added here in rank order as Work.reduce adds them (the same bits, no summed copy), or two ranks'
    partials as a pair of [T, N] tensors in rank order (the reduce-scatter path; the same sum); r: the residual,
    bf16 (the model's roundings) or fp32 (h = rms(x) * w_post * xs + r * rs unrounded). yf_out: y also unrounded
    (fp32); it may share memory with the partials of rank 0 (each row reads its partials before it writes). In-place
    safe for h_out=r."""
    x1 = None
    if isinstance(x, tuple):
        x, x1 = x
        assert x1.shape == x.shape and x1.stride(0) == x.stride(0) and x1.dtype == x.dtype
    parts, sp = (x.shape[0], x.stride(0)) if x.dim() == 3 else (1, 0)
    if x.dim() == 3:
        x = x[0]
    T, N = x.shape
    h_out = torch.empty(T, N, device=x.device, dtype=r.dtype) if h_out is None else h_out
    if w_next is not None and y_out is None:
        y_out = torch.empty(T, N, device=x.device, dtype=torch.bfloat16)
    if T:
        _boundary[(T,)](x, x1 if x1 is not None else x, r, w_post, w_next if w_next is not None else w_post, h_out,
                        y_out if y_out is not None else h_out, yf_out if yf_out is not None else h_out, N,
                        x.stride(0), r.stride(0), sp, sc[0], sc[1], EPS=eps, NEXT=w_next is not None,
                        F32=r.dtype == torch.float32, KEEP_F32=yf_out is not None and w_next is not None,
                        PARTS=parts, APART=x1 is not None, BLOCK=triton.next_power_of_2(N), num_warps=8)
    return h_out, y_out


@triton.jit
def _rms(X, W, Y, N, sx, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    m = c < N
    x = tl.load(X + row * sx + c, mask=m, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / N + EPS) * tl.load(W + c, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * N + c, y.to(Y.dtype.element_ty), mask=m)


def rms_norm(x, w, eps, out=None):
    T, N = x.shape
    out = torch.empty(T, N, device=x.device, dtype=torch.bfloat16) if out is None else out
    if T:
        _rms[(T,)](x, w, out, N, x.stride(0), EPS=eps, BLOCK=triton.next_power_of_2(N), num_warps=8)
    return out


@triton.jit
def _router(X, W, O, T, K, sx, E: tl.constexpr, BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    pt = tl.program_id(0) * BT + tl.arange(0, BT)
    pe = tl.program_id(1) * BE + tl.arange(0, BE)
    acc = tl.zeros([BT, BE], tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        x = tl.load(X + pt[:, None].to(tl.int64) * sx + kk[None, :], mask=(pt[:, None] < T) & (kk[None, :] < K), other=0.0)
        w = tl.load(W + pe[None, :].to(tl.int64) * K + kk[:, None], mask=(kk[:, None] < K) & (pe[None, :] < E), other=0.0)
        acc = tl.dot(x.to(tl.float32), w.to(tl.float32), acc, input_precision="ieee")
    tl.store(O + pt[:, None].to(tl.int64) * E + pe[None, :], acc, mask=(pt[:, None] < T) & (pe[None, :] < E))


def router_logits(x, w, out=None):
    """fp32 logits [T, E] = x (bf16 or fp32 [T, K]) @ w.T (bf16 or fp32 [E, K]), both read as fp32 (exact), in true
    fp32 (no TF32)."""
    T, K = x.shape
    E = w.shape[0]
    out = torch.empty(T, E, device=x.device, dtype=torch.float32) if out is None else out
    if T:
        # tiles by the rows (tools/beam_route_check.py, tools/beam_router_bench.py: every setting gives the same bits,
        # each output one FMA chain over K in order; 16-wide expert tiles spread a decode row's 5.2 MB of weights over
        # 16 programs: 210 -> 90 us; prompt rows take their own tiles)
        if T > 64:
            bt, be, bk, warps = ROUTER_PROMPT if T <= 2048 else ROUTER_BIG
        else:
            (bt, warps), be, bk = (16, ROUTER_WARPS[0]) if T <= 16 else (32, ROUTER_WARPS[1]), ROUTER_BE, ROUTER_BK
        _router[(triton.cdiv(T, bt), triton.cdiv(E, be))](x, w, out, T, K, x.stride(0), E=E, BT=bt, BE=be, BK=bk,
                                                          num_warps=warps)
    return out


ROUTER_BE = int(os.environ.get("TF_BEAM_ROUTER_BE", "16"))
ROUTER_BK = int(os.environ.get("TF_BEAM_ROUTER_BK", "64"))
# warps at <= 16 rows, <= 64 rows
ROUTER_WARPS = tuple(int(v) for v in os.environ.get("TF_BEAM_ROUTER_WARPS", "4,2").split(","))
# more rows (prompt chunks): rows a program, experts a program, K step, warps. Wider expert tiles read each fp32 row
# fewer times (16-wide: 16 passes over 84 MB at 4,096 rows); tools/beam_router_bench.py on GB10, the same bits for
# every setting: 32,16,64,1 (before) 1,461 us at 2,048 rows and 4,360 at 4,096; 64,64,16,2 711 / 1,748; 64,128,16,2
# 897 / 1,389. Up to 2,048 rows ROUTER_PROMPT, more ROUTER_BIG (TF_BEAM_ROUTER_PROMPT / TF_BEAM_ROUTER_BIG).
ROUTER_PROMPT = tuple(int(v) for v in os.environ.get("TF_BEAM_ROUTER_PROMPT", "64,64,16,2").split(","))
ROUTER_BIG = tuple(int(v) for v in os.environ.get("TF_BEAM_ROUTER_BIG", "64,128,16,2").split(","))


@triton.jit
def _route_pick(L, BIAS, S, C, E: tl.constexpr, G: tl.constexpr, TG: tl.constexpr):
    """One row: scores = sigmoid(logits) in torch's fp32 arithmetic (1 / (1 + exp(-x)), libm exp, IEEE division),
    choice = scores + bias, the groups ranked by the sum of their two best choices (ties: the lower group first), and
    choice kept in the top TG groups, -inf elsewhere."""
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, E)
    x = tl.load(L + t * E + e)
    s = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))
    c = s + tl.load(BIAS + e)
    c2 = tl.reshape(c, (G, E // G))
    m1 = tl.max(c2, axis=1)
    j = tl.arange(0, E // G)
    first = tl.min(tl.where(c2 == m1[:, None], j[None, :], E), axis=1)          # the best's first column
    m2 = tl.max(tl.where(j[None, :] == first[:, None], float("-inf"), c2), axis=1)
    gs = m1 + m2
    g = tl.arange(0, G)
    beats = (gs[None, :] > gs[:, None]) | ((gs[None, :] == gs[:, None]) & (g[None, :] < g[:, None]))
    keep = tl.sum(beats.to(tl.int32), axis=1) < TG
    keep_e = tl.reshape(tl.broadcast_to(keep[:, None], (G, E // G)), (E,))
    tl.store(S + t * E + e, s)
    tl.store(C + t * E + e, tl.where(keep_e, c, float("-inf")))


ROUTE_FUSED = os.environ.get("TF_BEAM_ROUTE_FUSED", "1") == "1"


def route(logits, bias, *, groups: int, top_groups: int, top_k: int):
    """Grouped sigmoid routing with a score-correction bias (noaux_tc): pick top_groups groups by the sum of their
    two best biased scores, then top_k experts inside them; weights are the unbiased scores renormalized.
    By default the scores and the group pick run in one kernel (tools/beam_route_check.py: the same bits), then
    torch's top_k (34 us a call at decode, 81 us all in torch); TF_BEAM_ROUTE_FUSED=0 runs torch's ops throughout."""
    T, E = logits.shape
    if ROUTE_FUSED and T:
        scores, choice = torch.empty_like(logits), torch.empty_like(logits)
        _route_pick[(T,)](logits, bias, scores, choice, E=E, G=groups, TG=top_groups, num_warps=4)
        idx = choice.topk(top_k, dim=-1)[1]
        wt = scores.gather(1, idx)
        return idx.to(torch.int32), wt / wt.sum(-1, keepdim=True)
    scores = logits.sigmoid()
    choice = scores + bias
    gs = choice.view(T, groups, E // groups).topk(2, dim=-1)[0].sum(-1)
    gidx = gs.topk(top_groups, dim=-1)[1]
    gmask = torch.zeros_like(gs).scatter_(1, gidx, 1.0).bool()
    choice = choice.masked_fill(~gmask.repeat_interleave(E // groups, dim=1), float("-inf"))
    idx = choice.topk(top_k, dim=-1)[1]
    wt = scores.gather(1, idx)
    return idx.to(torch.int32), wt / wt.sum(-1, keepdim=True)


def softcap(logits, cap: float):
    return torch.tanh(logits.float() / cap) * cap


@triton.jit
def _silu_mul(G, U, Y, N, sg, su, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    g = tl.load(G + r * sg + i, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + r * su + i, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + r * N + i, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16), mask=m)


def silu_mul(g, u, out=None):
    """bf16(silu(g) * u) in fp32 for the dense layers' SwiGLU."""
    R, N = g.shape
    out = torch.empty((R, N), device=g.device, dtype=torch.bfloat16) if out is None else out
    if R:
        _silu_mul[(R, triton.cdiv(N, 2048))](g, u, out, N, g.stride(0), u.stride(0), BLOCK=2048, num_warps=4)
    return out
