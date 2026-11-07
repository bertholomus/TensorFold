"""dspark.py's draft on the GPU: about 90 launches where the plain torch ops take 477, the feed-forward and the attention's
projections in FP8 blocks (half the bytes a draft reads, .4 GiB less a rank). A draft only has to be right often: a row's numbers
may differ from ``Drafter``'s, so long as all ranks agree on them, and they do: x stays the same on every rank.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .cp4_cuda import Fp8Block
from .cp4_kernels import fp8_linear
from .dspark import DEPTH, KEYS, ROWS, WINDOW, Drafter, Io
from .tp_kernels import add_norm

BF16, F16, F32 = torch.bfloat16, torch.float16, torch.float32
TOP_BLOCK = 2048                      # vocabulary columns a program of ``_add_top``
FP8 = set((os.environ.get("TF_ML4_DRAFT_FP8", "wq,wo,w1,w2,mk")).split(","))


def fp8_blocks(w: torch.Tensor) -> Fp8Block:
    """w [N, K] as e4m3 and the fp32 scale of each 128 x 128 block."""

    n, k = w.shape
    b = w.float().view(n // 128, 128, k // 128, 128)
    s = b.abs().amax((1, 3)).clamp_min(1e-20) / 448.0
    return Fp8Block((b / s[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k).contiguous(), s.contiguous())


def narrow(w: torch.Tensor, name: str) -> Fp8Block | torch.Tensor:
    return fp8_blocks(w) if name in FP8 else w.to(F16)


def linear8(x: torch.Tensor, m: Fp8Block | torch.Tensor) -> torch.Tensor:
    return fp8_linear(x, m.w, m.scale) if isinstance(m, Fp8Block) else F.linear(x, m)


@triton.jit
def _tap_rows(TAPS, TAP, OUT, tap_stride, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    """Row r of OUT: the taps of step row TAP[r], side by side."""

    r = tl.program_id(0)
    t = tl.program_id(1)
    i = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    src = tl.load(TAP + r)
    tl.store(OUT + r * (T * D) + t * D + i, tl.load(TAPS + t * tap_stride + src * D + i))


@triton.jit
def _cols_norm(E, NW, OUT, R, DL: tl.constexpr, D: tl.constexpr, EPS, BLOCK: tl.constexpr):
    """Row r of every rank's columns E [W, R, DL], RMS-normed."""

    r = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    ok = i < D
    x = tl.load(E + (i // DL) * R * DL + r * DL + i % DL, mask=ok, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
    y = x * inv * tl.load(NW + i, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * D + i, y.to(tl.bfloat16), mask=ok)


@triton.jit
def _latent_rows(KV, kv_stride, NW, POS, DST, INV, RING, ring_stride, first, C: tl.constexpr, RD: tl.constexpr, EPS):
    """What a draft layer keeps of a row: its latent, normed, and its rotated key, to row DST[r] of the layer's ring. KV's rows hold
    one such part for each layer of the launch."""

    r = tl.program_id(0)
    at = tl.program_id(1)
    kv = KV + r * kv_stride + at * (C + RD)
    i = tl.arange(0, C)
    lat = tl.load(kv + i).to(tl.float32)
    inv = tl.rsqrt(tl.sum(lat * lat, axis=0) / C + EPS)
    y = lat * inv * tl.load(NW + (first + at) * C + i).to(tl.float32)
    row = RING + (first + at).to(tl.int64) * ring_stride + tl.load(DST + r) * (C + RD)
    tl.store(row + i, y.to(tl.bfloat16))
    h = tl.arange(0, RD // 2)
    x1 = tl.load(kv + C + 2 * h).to(tl.float32)
    x2 = tl.load(kv + C + 2 * h + 1).to(tl.float32)
    ang = tl.load(POS + r).to(tl.float32) * tl.load(INV + h)
    co, si = tl.cos(ang), tl.sin(ang)
    tl.store(row + C + 2 * h, (x1 * co - x2 * si).to(tl.bfloat16))
    tl.store(row + C + 2 * h + 1, (x2 * co + x1 * si).to(tl.bfloat16))


@triton.jit
def _attend_rows(Q, QL, POS, RING, SLOT, SEEN, INV, OUT, n, sm, HL: tl.constexpr, QK: tl.constexpr, NOPE: tl.constexpr,
                 RD: tl.constexpr, C: tl.constexpr, SLOT_ROWS: tl.constexpr, WIN: tl.constexpr, NKEYS: tl.constexpr,
                 DEEP: tl.constexpr, M: tl.constexpr, BN: tl.constexpr):
    """A stream's block rows for one head over its ring and the block itself: OUT [HL, n, C] the normalized latent output. QL
    [HL, n, C] holds the rows' absorbed queries, Q the rows whose last RD columns a head rotates here."""

    b = tl.program_id(0)
    h = tl.program_id(1)
    m = tl.arange(0, M)
    ok = m < DEEP
    rows = b * DEEP + m
    cc = tl.arange(0, C)
    ql = tl.load(QL + (h * n + rows[:, None]) * C + cc[None, :], mask=ok[:, None], other=0.0).to(tl.bfloat16)
    p2 = tl.arange(0, RD // 2)
    qa = Q + rows[:, None] * (HL * QK) + h * QK + NOPE + 2 * p2[None, :]
    x1 = tl.load(qa, mask=ok[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(qa + 1, mask=ok[:, None], other=0.0).to(tl.float32)
    ang = tl.load(POS + rows, mask=ok, other=0).to(tl.float32)[:, None] * tl.load(INV + p2)[None, :]
    co, si = tl.cos(ang), tl.sin(ang)
    q1 = (x1 * co - x2 * si).to(tl.bfloat16)
    q2 = (x2 * co + x1 * si).to(tl.bfloat16)
    ring = RING + tl.load(SLOT + b) * SLOT_ROWS * (C + RD)
    seen = tl.load(SEEN + b)
    best = tl.full((M,), -1e30, dtype=tl.float32)
    total = tl.zeros((M,), dtype=tl.float32)
    acc = tl.zeros((M, C), dtype=tl.float32)
    for j0 in range(0, NKEYS, BN):
        j = j0 + tl.arange(0, BN)
        vis = ((j < seen) | (j >= WIN)) & (j < NKEYS)
        at = ring + j.to(tl.int64) * (C + RD)
        lat = tl.load(at[:, None] + cc[None, :], mask=vis[:, None], other=0.0)
        k1 = tl.load(at[None, :] + C + 2 * p2[:, None], mask=vis[None, :], other=0.0)
        k2 = tl.load(at[None, :] + C + 2 * p2[:, None] + 1, mask=vis[None, :], other=0.0)
        s = tl.dot(ql, tl.trans(lat))
        s = tl.dot(q1, k1, s)
        s = tl.dot(q2, k2, s)
        s = tl.where(vis[None, :], s * sm, -1e30)
        new = tl.maximum(best, tl.max(s, axis=1))
        p = tl.where(vis[None, :], tl.exp(s - new[:, None]), 0.0)
        fix = tl.exp(best - new)
        total = total * fix + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), lat, acc * fix[:, None])
        best = new
    o = acc / tl.maximum(total, 1e-30)[:, None]
    tl.store(OUT + (h * n + rows[:, None]) * C + cc[None, :], o.to(tl.float16), mask=ok[:, None])


@triton.jit
def _swiglu(GU, OUT, H: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GU + r * (2 * H) + i).to(tl.float32)
    u = tl.load(GU + r * (2 * H) + H + i).to(tl.float32)
    tl.store(OUT + r * H + i, (g / (1.0 + tl.exp(-g)) * u).to(tl.float16))


@triton.jit
def _won(E, b, B: tl.constexpr, W: tl.constexpr):
    """The token every rank's best key names for stream b: E [W, B, 2], a key the score over 0x7FFFFFFF - id."""

    best = tl.load(E + b * 2)
    for w in tl.static_range(1, W):
        best = tl.maximum(best, tl.load(E + (w * B + b) * 2))
    return 0x7FFFFFFF - (best & 0x7FFFFFFF)


@triton.jit
def _tanh(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit
def _mk_step(STATE, NEXT, ANCHOR, E, MK_IN, WS, JH, T, B: tl.constexpr, NR: tl.constexpr, C: tl.constexpr, DEEP: tl.constexpr,
             AT: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr):
    """One token of the recurrent head for stream b: state and token through ``WS`` [3 C, 2 C], with JH the hidden row's share and
    the bias; NEXT the new state, T what the head adds to the logits, before its output matrix."""

    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    b = tl.program_id(1)
    if AT == 0:
        prev = tl.load(ANCHOR + b)
    else:
        prev = _won(E, b, B, NR)
    g = tl.zeros((BJ,), dtype=tl.float32)
    c = tl.zeros((BJ,), dtype=tl.float32)
    o = tl.zeros((BJ,), dtype=tl.float32)
    for k0 in range(0, C, BK):
        k = k0 + tl.arange(0, BK)
        e = tl.load(MK_IN + prev * C + k).to(tl.float32)[None, :]
        w = WS + j[:, None] * (2 * C) + C + k[None, :]
        g += tl.sum(tl.load(w).to(tl.float32) * e, axis=1)
        c += tl.sum(tl.load(w + C * 2 * C).to(tl.float32) * e, axis=1)
        o += tl.sum(tl.load(w + 2 * C * 2 * C).to(tl.float32) * e, axis=1)
        if AT > 0:
            st = tl.load(STATE + b * C + k)[None, :]
            w = WS + j[:, None] * (2 * C) + k[None, :]
            g += tl.sum(tl.load(w).to(tl.float32) * st, axis=1)
            c += tl.sum(tl.load(w + C * 2 * C).to(tl.float32) * st, axis=1)
            o += tl.sum(tl.load(w + 2 * C * 2 * C).to(tl.float32) * st, axis=1)
    jh = JH + (b * DEEP + AT) * (3 * C) + j
    g += tl.load(jh).to(tl.float32)
    c += tl.load(jh + C).to(tl.float32)
    o += tl.load(jh + 2 * C).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-g))
    old = tl.zeros((BJ,), dtype=tl.float32)
    if AT > 0:
        old = tl.load(STATE + b * C + j)
    tl.store(NEXT + b * C + j, gate * old + (1.0 - gate) * _tanh(c))
    tl.store(T + b * C + j, _tanh(o).to(tl.float16))


@triton.jit
def _add_top(BASE, BIAS, TOP, V, first, cap, DEEP: tl.constexpr, AT: tl.constexpr, BLOCK: tl.constexpr):
    """A stream's logits for its token AT: the target's head, softcapped, plus the recurrent head's bias; a program's best into TOP
    as a key: the score's ordered bits over 0x7FFFFFFF - id, so the lowest id wins a tie, as an argmax does."""

    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.program_id(1)
    ok = c < V
    x = tl.load(BASE + (b * DEEP + AT).to(tl.int64) * V + c, mask=ok, other=0.0)
    y = tl.load(BIAS + b.to(tl.int64) * V + c, mask=ok, other=0.0).to(tl.float32)
    v = tl.where(ok, _tanh(x / cap) * cap + y, -3.0e38)
    best = tl.max(v, axis=0)
    at = tl.argmax(v, axis=0)
    bits = best.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    ordered = tl.where(bits >= 0x80000000, bits ^ 0xFFFFFFFF, bits | 0x80000000)
    tl.atomic_max(TOP + b * 2, (ordered << 31) | (0x7FFFFFFF - (first + tl.program_id(0) * BLOCK + at).to(tl.int64)))


@triton.jit
def _take(E, OUT, B: tl.constexpr, NR: tl.constexpr, DEEP: tl.constexpr):
    b = tl.program_id(0)
    at = tl.program_id(1)
    tl.store(OUT + b * DEEP + at, _won(E + at * (NR * B * 2), b, B, NR))


class FusedDrafter(Drafter):
    def __init__(self, lane, src, rows: int, graphs=None, **kw) -> None:
        super().__init__(lane, src, rows, graphs, **kw)
        c = self.cfg
        self.kv_all = torch.cat([lay.kv_a for lay in self.layers]).contiguous()
        self.kv_norms = torch.stack([lay.kv_norm for lay in self.layers]).contiguous()
        for lay in self.layers:
            lay.kv_a = lay.kv_a.to(F16)
            lay.wuk, lay.wuv = lay.wuk.to(F16), lay.wuv.to(F16)
            lay.wq, lay.wo, lay.w2 = narrow(lay.wq, "wq"), narrow(lay.wo, "wo"), narrow(lay.w2, "w2")
            lay.w1, lay.w3 = narrow(torch.cat([lay.w1, lay.w3]), "w1"), None
        self.mk_out = narrow(self.mk_out, "mk")
        C = self.rank_mk
        self.mk_ws = self.mk_w[:, :2 * C].contiguous()
        self.mk_wh = self.mk_w[:, 2 * C:].to(F16).contiguous()
        self.mk_b, self.mk_w = self.mk_b.to(F16), None
        self.tmax = 1 << (c.dim - 1).bit_length()
        torch.cuda.empty_cache()

    def absorb(self, rows: torch.Tensor, pos: torch.Tensor, dst: torch.Tensor) -> None:
        c, lane = self.cfg, self.lane
        every = lane._gather(F.linear(rows, self.proj))
        main = torch.empty((rows.shape[0], c.dim), dtype=BF16, device=rows.device)
        _cols_norm[(rows.shape[0],)](every, self.target_norm, main, rows.shape[0], DL=every.shape[2], D=c.dim, EPS=c.eps,
                                     BLOCK=self.tmax, num_warps=8)
        self._keep(F.linear(main, self.kv_all), pos, dst, 0, len(self.layers))

    def _keep(self, kv: torch.Tensor, pos: torch.Tensor, dst: torch.Tensor, first: int, layers: int) -> None:
        c = self.cfg
        _latent_rows[(kv.shape[0], layers)](kv, kv.stride(0), self.kv_norms, pos, dst, self.inv, self.ring, self.ring.stride(0), first,
                                            C=c.kv_lora, RD=c.rd, EPS=c.eps, num_warps=4)

    def _attend(self, i: int, lay, q: torch.Tensor, pos: torch.Tensor, slot: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        c, hl = self.cfg, self.hl
        n, B = q.shape[0], slot.numel()
        ql = torch.bmm(q.view(n, hl, c.qk_head)[..., :c.nope].transpose(0, 1), lay.wuk)
        o = torch.empty((hl, n, c.kv_lora), dtype=F16, device=q.device)
        _attend_rows[(B, hl)](q, ql, pos, self.ring[i], slot, seen, self.inv, o, n, c.sm_scale, HL=hl, QK=c.qk_head, NOPE=c.nope,
                              RD=c.rd, C=c.kv_lora, SLOT_ROWS=ROWS, WIN=WINDOW, NKEYS=KEYS, DEEP=DEPTH, M=16, BN=32, num_warps=4,
                              num_stages=2)
        return torch.bmm(o, lay.wuv.transpose(1, 2)).transpose(0, 1).reshape(n, hl * c.v_head)

    def block(self, x: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, slot: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        c, lane = self.cfg, self.lane
        n, every, x = x.shape[0], None, x.clone()
        for i, lay in enumerate(self.layers):
            _, _, x16 = add_norm(every, x, lay.attn_norm, c.eps)
            self._keep(F.linear(x16, lay.kv_a), pos, rows, i, 1)
            o = self._attend(i, lay, linear8(x16, lay.wq), pos, slot, seen)
            every = lane._gather(linear8(o, lay.wo))
            _, _, x16 = add_norm(every, x, lay.ffn_norm, c.eps)
            gu = linear8(x16, lay.w1)
            hid = gu.shape[1] // 2
            h = torch.empty((n, hid), dtype=F16, device=x.device)
            _swiglu[(n, hid // 1024)](gu, h, H=hid, BLOCK=1024, num_warps=4)
            every = lane._gather(linear8(h, lay.w2))
        return add_norm(every, x, self.final_norm, c.eps)[2]

    def pick(self, h: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        c, lane, B, C = self.cfg, self.lane, anchor.numel(), self.rank_mk
        dev, V = h.device, lane.vl
        base = lane.head(h, out_dtype=F32)
        jh = torch.addmm(self.mk_b, h, self.mk_wh.t())
        top = torch.zeros((DEPTH, B, 2), dtype=torch.long, device=dev)
        every = torch.empty((DEPTH, lane.world, B, 2), dtype=torch.long, device=dev)
        state = [torch.empty((B, C), dtype=F32, device=dev) for _ in range(2)]
        t = torch.empty((B, C), dtype=F16, device=dev)
        for i in range(DEPTH):
            _mk_step[(C // 16, B)](state[i % 2], state[1 - i % 2], anchor, every[max(i - 1, 0)], self.mk_in, self.mk_ws, jh, t, B=B,
                                   NR=lane.world, C=C, DEEP=DEPTH, AT=i, BJ=16, BK=128, num_warps=4)
            bias = linear8(t, self.mk_out)
            _add_top[(triton.cdiv(V, TOP_BLOCK), B)](base, bias, top[i], V, lane.rank * V, float(c.softcap), DEEP=DEPTH, AT=i,
                                                     BLOCK=TOP_BLOCK, num_warps=8)
            lane.comm.all_gather(top[i].view(-1).view(F32), every[i].view(-1).view(F32))
        out = torch.empty((B, DEPTH), dtype=torch.long, device=dev)
        _take[(B, DEPTH)](every, out, B=B, NR=lane.world, DEEP=DEPTH, num_warps=1)
        return out

    def _first(self, io: Io, tokens: list) -> None:
        pass                                                     # the graph takes the rows

    def _body(self, io: Io, drafts: bool) -> torch.Tensor:
        taps, d = self.taps, self.cfg.dim
        rows = torch.empty((io.A, taps.shape[0] * d), dtype=BF16, device=self.dev)
        _tap_rows[(io.A, taps.shape[0], d // 1024)](taps, io.tap, rows, taps.stride(0), T=taps.shape[0], D=d, BLOCK=1024, num_warps=4)
        self.absorb(rows, io.pos, io.dst)
        if drafts:
            io.x.view(io.B, DEPTH, -1)[:, 0] = self.lane.rows(io.anchor)
            return self.pick(self.block(io.x, io.bpos, io.brow, io.slot, io.seen), io.anchor)
        return torch.zeros((io.B, 0), dtype=torch.long, device=self.dev)
