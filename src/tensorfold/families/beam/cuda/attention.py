"""Beam attention: grouped-query attention of several sequences' rows over their KV stores, with a per-layer key window
and the sigmoid output gate in the epilogue.

- A sequence's rows attend to the keys at positions [max(0, p - window + 1), p] of their own position p.
- Keys sit in one of three stores (``mode``): LINEAR, slot = position - base (one sequence: a prompt chunk's window
  buffer); RING, slot = ring[s] + position % ring_size (a window layer's last positions per sequence); PAGED,
  slot = table[s, position // page] * page + position % page (a full layer's pool).
- The query heads sharing a key head run as one block of (row, head) pairs, so each key block is read once for all.
- Long key ranges split into parts (flash decoding) that ``_merge`` combines. Decode rows (``part``) take 16-row tiles and
  a full layer's keys in parts of ``part`` positions counted from 0, so a row's bits do not depend on the rows beside it
  (other sequences, or a sequence's draft rows under verification); a prompt chunk's parts follow its own span.
- Softmax in fp32 with exp2; P rounded to bf16 for P @ V; K/V may be stored in fp8 (read back to bf16).
- The gate: out = bf16(bf16(o) * bf16(sigmoid(bf16(g + bias[head])))), Reflection's Beam rounding.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

LINEAR, RING, PAGED = 0, 1, 2
PART = 4096                 # a decode row's key part in a full layer (a multiple of BLOCK_N)


@triton.jit
def _gated(o, GT, GB, r, h, d, m_ok, sg_r, HAS_GATE: tl.constexpr, D: tl.constexpr):
    o = o.to(tl.bfloat16).to(tl.float32)
    if HAS_GATE:
        g = tl.load(GT + r[:, None].to(tl.int64) * sg_r + h[:, None] * D + d[None, :], mask=m_ok[:, None], other=0.0)
        b = tl.load(GB + h, mask=m_ok, other=0.0).to(tl.float32)
        g = (g.to(tl.float32) + b[:, None]).to(tl.bfloat16).to(tl.float32)
        s = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        o = o * s
    return o.to(tl.bfloat16)


@triton.jit
def _attn(Q, K, V, O, POS, ROW0, TAB, GT, GB, PO, PML,
          RT, W, BASE, RING_N, PAGE, scale, SPLITS,
          sq_r, sq_h, sk_s, sk_h, sv_s, sv_h, so_r, so_h, s_tab, sg_r,
          G: tl.constexpr, D: tl.constexpr, MODE: tl.constexpr, ONE: tl.constexpr, HAS_GATE: tl.constexpr,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PART: tl.constexpr = 0):
    pid_m = tl.program_id(0)
    g = tl.program_id(1)
    z = tl.program_id(2)
    s = z // SPLITS
    sid = z % SPLITS
    r0 = tl.load(ROW0 + s)
    r1 = tl.load(ROW0 + s + 1)
    MT = (r1 - r0) * G
    if pid_m * BLOCK_M >= MT:
        return
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_ok = m < MT
    r = r0 + m // G
    h = g * G + m % G
    pos = tl.load(POS + r, mask=m_ok, other=0)
    lo = tl.maximum(pos - W + 1, 0)
    d = tl.arange(0, D)
    q = tl.load(Q + r[:, None].to(tl.int64) * sq_r + h[:, None] * sq_h + d[None, :], mask=m_ok[:, None], other=0.0)
    hi_blk = tl.max(tl.where(m_ok, pos, -1), 0) + 1
    lo_blk = tl.min(tl.where(m_ok, lo, 2147483647), 0)
    lo_blk = (lo_blk // BLOCK_N) * BLOCK_N
    if PART > 0:                                                           # absolute parts: [sid, sid + 1) * PART
        s_lo = tl.maximum(lo_blk, sid * PART)
        s_hi = tl.minimum(hi_blk, (sid + 1) * PART)
    else:
        per = tl.cdiv(tl.cdiv(tl.maximum(hi_blk - lo_blk, 0), SPLITS), BLOCK_N) * BLOCK_N
        s_lo = lo_blk + sid * per
        s_hi = tl.minimum(s_lo + per, hi_blk)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    qk = scale * 1.4426950408889634
    if MODE == 1:
        rbase = tl.load(TAB + s).to(tl.int64)
    for start in range(s_lo, s_hi, BLOCK_N):
        j = start + tl.arange(0, BLOCK_N)
        j_ok = j < s_hi
        if MODE == 0:
            j_ok = j_ok & (j >= BASE)                                      # the buffer starts at BASE
            slot = (j - BASE).to(tl.int64)
        elif MODE == 1:
            slot = rbase + (j % RING_N).to(tl.int64)
        else:
            page = tl.load(TAB + s * s_tab + start // PAGE).to(tl.int64)
            slot = page * PAGE + (j % PAGE).to(tl.int64)
        k = tl.load(K + slot[None, :] * sk_s + g * sk_h + d[:, None], mask=j_ok[None, :], other=0.0).to(tl.bfloat16)
        sc = tl.dot(q, k) * qk
        ok = (j[None, :] <= pos[:, None]) & (j[None, :] >= lo[:, None]) & j_ok[None, :] & m_ok[:, None]
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(sc - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + slot[:, None] * sv_s + g * sv_h + d[None, :], mask=j_ok[:, None], other=0.0).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    if ONE:
        o = _gated(acc / tl.where(l_i > 0, l_i, 1.0)[:, None], GT, GB, r, h, d, m_ok, sg_r, HAS_GATE, D)
        tl.store(O + r[:, None].to(tl.int64) * so_r + h[:, None] * so_h + d[None, :], o, mask=m_ok[:, None])
    else:
        mg = (r0 * G + m).to(tl.int64)                                     # the (row, head-in-group) index of all rows
        base = (sid * tl.num_programs(1) + g).to(tl.int64) * (RT * G)
        tl.store(PO + (base + mg[:, None]) * D + d[None, :], acc, mask=m_ok[:, None])
        tl.store(PML + (base + mg) * 2, m_i, mask=m_ok)
        tl.store(PML + (base + mg) * 2 + 1, l_i, mask=m_ok)


@triton.jit
def _merge(PO, PML, O, GT, GB, RT, HKV, SPLITS, so_r, so_h, sg_r,
           G: tl.constexpr, D: tl.constexpr, HAS_GATE: tl.constexpr, BLOCK_M: tl.constexpr):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    g = tl.program_id(1)
    MT = RT * G
    m_ok = m < MT
    r = m // G
    h = g * G + m % G
    d = tl.arange(0, D)
    mx = tl.full([BLOCK_M], float("-inf"), tl.float32)
    for sp in range(SPLITS):
        base = (sp * HKV + g).to(tl.int64) * MT
        mx = tl.maximum(mx, tl.load(PML + (base + m) * 2, mask=m_ok, other=float("-inf")))
    mx_use = tl.where(mx == float("-inf"), 0.0, mx)
    l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for sp in range(SPLITS):
        base = (sp * HKV + g).to(tl.int64) * MT
        w = tl.exp2(tl.load(PML + (base + m) * 2, mask=m_ok, other=float("-inf")) - mx_use)
        l += tl.load(PML + (base + m) * 2 + 1, mask=m_ok, other=0.0) * w
        acc += tl.load(PO + (base + m[:, None]) * D + d[None, :], mask=m_ok[:, None], other=0.0) * w[:, None]
    o = _gated(acc / tl.where(l > 0, l, 1.0)[:, None], GT, GB, r, h, d, m_ok, sg_r, HAS_GATE, D)
    tl.store(O + r[:, None].to(tl.int64) * so_r + h[:, None] * so_h + d[None, :], o, mask=m_ok[:, None])


def splits_for(span: int, programs: int, sms: int = 48, min_keys: int = 1024, most: int = 64) -> int:
    """Key-range parts so that ~2 programs run per SM, each part at least min_keys long."""
    if span <= min_keys or programs >= 2 * sms:
        return 1
    return max(1, min(most, (2 * sms) // max(programs, 1), span // min_keys))


class Scratch:
    """Split partials, grown to the largest call."""

    def __init__(self) -> None:
        self.po = self.pml = None

    def get(self, n: int, device):
        if self.po is None or self.po.numel() < n * 128:
            self.po = torch.empty(n * 128, device=device, dtype=torch.float32)
            self.pml = torch.empty(n * 2, device=device, dtype=torch.float32)
        return self.po, self.pml


def attention(q, k, v, pos, row0, *, window: int, mode: int, max_rows: int, span: int, table=None, base: int = 0,
              ring_n: int = 0, page: int = 0, gate=None, gate_bias=None, out=None, splits: int | None = None,
              scratch: Scratch | None = None, part: int = 0):
    """q [R, HQ, 128] bf16 (row-strided view ok); k, v [slots, HKV, 128] stores (bf16 or fp8); pos [R] int32; row0
    [S + 1] int32 row starts of the S sequences (ascending positions within each). max_rows: the most rows of one
    sequence; span: the widest key range of a row (picks the splits). table: RING [S] ring starts (slots) / PAGED
    [S, pages] page ids. gate [R, HQ * 128] bf16 and gate_bias [HQ]: the output gate. part: decode rows (see the
    module notes); a PAGED call then takes span // part + 1 parts at most, counted from position 0."""
    R, HQ, D = q.shape
    HKV = k.shape[1]
    G = HQ // HKV
    S = row0.shape[0] - 1
    assert HQ == G * HKV and D == 128 and q.dtype == torch.bfloat16
    out = torch.empty((R, HQ, D), device=q.device, dtype=torch.bfloat16) if out is None else out
    if R == 0:
        return out
    MT = max_rows * G
    if part:
        assert part % 64 == 0
        bm = 16
        if splits is None:
            splits = triton.cdiv(span, part) if mode == PAGED else 1
    else:
        bm = 16 if MT <= 16 else 32 if MT <= 32 else 64
    nm = triton.cdiv(MT, bm)
    if splits is None:
        splits = splits_for(span, nm * HKV * S)
    has_gate = gate is not None
    gt = gate if has_gate else q
    gb = gate_bias if has_gate else q
    sg = gate.stride(0) if has_gate else 0
    tab = table if table is not None else row0
    s_tab = table.stride(0) if (table is not None and table.dim() == 2) else 0
    if splits > 1:
        po, pml = (scratch or Scratch()).get(splits * HKV * R * G, q.device)
    else:
        po = pml = out
    # the split count is a runtime value (one compiled kernel for every span), ONE its single-part path
    _attn[(nm, HKV, S * splits)](q, k, v, out, pos, row0, tab, gt, gb, po, pml, R, window, base, max(ring_n, 1),
                                 max(page, 1), D ** -0.5, splits, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
                                 v.stride(0), v.stride(1), out.stride(0), out.stride(1), s_tab, sg,
                                 G=G, D=D, MODE=mode, ONE=splits == 1, HAS_GATE=has_gate,
                                 BLOCK_M=bm, BLOCK_N=64, PART=part if (part and splits > 1) else 0,
                                 num_warps=4, num_stages=2)
    if splits > 1:
        _merge[(triton.cdiv(R * G, 16), HKV)](po, pml, out, gt, gb, R, HKV, splits, out.stride(0), out.stride(1), sg,
                                              G=G, D=D, HAS_GATE=has_gate, BLOCK_M=16, num_warps=4)
    return out


def reference(q, k, v, pos, row0, *, window: int, slot_of, gate=None, gate_bias=None) -> torch.Tensor:
    """fp32 torch transcription for the tests: slot_of(seq, positions) -> slots; softmax fp32, P in bf16, gate."""
    R, HQ, D = q.shape
    HKV = k.shape[1]
    G = HQ // HKV
    out = torch.empty_like(q)
    S = row0.shape[0] - 1
    for s in range(S):
        for i in range(int(row0[s]), int(row0[s + 1])):
            p = int(pos[i])
            js = torch.arange(max(0, p - window + 1), p + 1, device=q.device)
            slots = slot_of(s, js)
            kk = k[slots].to(torch.bfloat16).float().repeat_interleave(G, dim=1)
            vv = v[slots].to(torch.bfloat16).float().repeat_interleave(G, dim=1)
            sc = torch.einsum("hd,nhd->hn", q[i].float(), kk) * D ** -0.5
            pr = torch.softmax(sc, -1)
            o = torch.einsum("hn,nhd->hd", pr.to(torch.bfloat16).float(), vv).to(torch.bfloat16)
            if gate is not None:
                gg = (gate[i].view(HQ, D).float() + gate_bias.float()[:, None]).to(torch.bfloat16)
                o = (o.float() * torch.sigmoid(gg.float()).to(torch.bfloat16).float()).to(torch.bfloat16)
            out[i] = o
    return out
