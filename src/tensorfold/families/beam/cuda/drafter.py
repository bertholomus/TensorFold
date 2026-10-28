"""The DFlash2 drafter (Reflection's DFlash2BeamDraftModel) on one tensor-parallel rank.

Five non-causal sliding-window layers draft k tokens from the target's residual stream: the target's states after
layers 4, 16, 28, 40 and 46 of every kept position, concatenated, go through ``fc`` (applied one state at a time as
the target's forward passes each layer, into an fp32 sum) and ``hidden_norm`` into each layer's keys and values (the context: a slot's latest ``window`` positions in a ring); the block (the last kept token,
then k mask tokens) attends to its context and to the whole block. Each layer's attention and MLP pass a grouped
causal convolution over the block, before the projection and after it (then the residual add and the next norm), with
coefficients made from the layer's input. The target's head scores the mask rows; the candidate selector walks the
top candidates of consecutive rows: each row's pick is the best of its own logit plus <predecessor[last pick] * h,
successor[candidate]> (h the row's hidden state projected to the codebook rank), drawn by the target's keyed rule
(exact_sampling.choose at the pick's position) when the request samples, so the drafts follow the draws they are
checked against. The walk runs on the host: it reads a few codebook rows a step, and the codebooks stay in the page
cache.

Split over the ranks like the target: q by heads, k and v by key heads, o by its input columns (a partial sum), the
MLP by width (down a partial sum), ``fc`` by output columns (gathered); the convolutions, norms, the selector's
projection and codebooks stay whole. Every rank computes the same drafts from the same gathered bits (the host walk
is the same numpy code on the same inputs). The drafts only choose which rows the target checks: the kept tokens are
the target's own draws either way."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.engine.exact_sampling import choose
from tensorfold.families.beam.cuda import forward as F
from tensorfold.families.beam.cuda import ops
from tensorfold.families.beam.cuda.weights import Pack, span

# TF_BEAM_DRAFT_BITS=4: the layers' q/k/v, o, gate/up and down projections as affine 4-bit groups of 64 (the GLM
# drafter's quantize4 and lane matmul), about 0.3 of their bytes to read a step; the context's k/v projection keeps bf16
# (its rows can be a whole prompt chunk). The drafts change, the replies do not (the target checks every draft).
DRAFT_BITS = int(os.environ.get("TF_BEAM_DRAFT_BITS", "16") or 16)
# TF_BEAM_DRAFT_HEAD_BITS=4: the candidates' logits from a 4-bit copy of the rank's head rows (0.28 of the bytes; the
# target's own draws keep the bf16 head)
DRAFT_HEAD_BITS = int(os.environ.get("TF_BEAM_DRAFT_HEAD_BITS", "16") or 16)


def f32(words: np.ndarray) -> np.ndarray:
    """bf16 words (uint16) as float32 values (exact)."""
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16(x: np.ndarray) -> np.ndarray:
    """float32 values rounded to bf16 (to nearest, ties to even), kept as float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)).view(np.float32)


@triton.jit
def _conv(X, DL, BASE, OUT, RES, NW, sx, sd, so, sr, D: tl.constexpr, GS: tl.constexpr, BLK: tl.constexpr,
          TAPS: tl.constexpr, NORM: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    """One row of the grouped causal convolution over its block (fp32 coefficients: the base kernel plus the row's
    input-made delta of the channel's group; no taps before the block's first row), rounded to bf16. NORM adds it to
    the residual (rounded, stored back) and writes the RMSNorm of the sum instead."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    m = c < D
    G: tl.constexpr = D // GS
    g = c // GS
    p = row % BLK
    acc = (tl.load(BASE + c, mask=m, other=0.0).to(tl.float32)
           + tl.load(DL + row * sd + g, mask=m, other=0.0).to(tl.float32)) * \
        tl.load(X + row * sx + c, mask=m, other=0.0).to(tl.float32)
    for t in tl.static_range(1, TAPS):
        tm = m & (p >= t)
        coef = (tl.load(BASE + t * D + c, mask=tm, other=0.0).to(tl.float32)
                + tl.load(DL + row * sd + t * G + g, mask=tm, other=0.0).to(tl.float32))
        acc += coef * tl.load(X + (row - t) * sx + c, mask=tm, other=0.0).to(tl.float32)
    y = acc.to(tl.bfloat16)
    if NORM:
        r = (tl.load(RES + row * sr + c, mask=m, other=0.0).to(tl.float32) + y.to(tl.float32)).to(tl.bfloat16)
        tl.store(RES + row * sr + c, r, mask=m)
        x = r.to(tl.float32)
        y = (x * tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.load(NW + c, mask=m, other=0.0).to(tl.float32)
             ).to(tl.bfloat16)
    tl.store(OUT + row * so + c, y, mask=m)


@triton.jit
def _dattn(Q, K, V, O, POS, SLOT, sq_r, sq_h, so_r, ring, window, scale,
           HKV: tl.constexpr, G: tl.constexpr, B: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    """One sequence's block, one key head: its G query heads x B rows attend to the ring's keys from each row's
    position - window to the block's end (the whole block: non-causal)."""
    s = tl.program_id(0)
    kh = tl.program_id(1)
    m = tl.arange(0, BM)
    g = m // B
    r = m % B
    mv = g < G
    d = tl.arange(0, D)
    p0 = tl.load(POS + s)
    slot = tl.load(SLOT + s).to(tl.int64)
    qrow = s * B + r
    head = kh * G + g
    q = tl.load(Q + qrow[:, None] * sq_r + head[:, None] * sq_h + d[None, :], mask=mv[:, None], other=0.0)
    qpos = p0 + r
    lo = tl.maximum(p0 - window, 0)
    end = p0 + B
    m_i = tl.full([BM], -1.0e30, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for k0 in range(lo // BK * BK, end, BK):
        kp = k0 + tl.arange(0, BK)
        kin = (kp >= lo) & (kp < end)
        at = ((slot * ring + kp % ring) * HKV + kh) * D
        k = tl.load(K + at[:, None] + d[None, :], mask=kin[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        sc = tl.where(kin[None, :] & (kp[None, :] >= qpos[:, None] - window), sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        pe = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pe, 1)
        v = tl.load(V + at[:, None] + d[None, :], mask=kin[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(pe.to(tl.bfloat16), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(O + qrow[:, None] * so_r + head[:, None] * D + d[None, :], out.to(tl.bfloat16), mask=mv[:, None])


@dataclass
class DLayer:
    qkv: torch.Tensor             # bf16 [rank's (heads + 2 * key heads) * 128, hidden]: q, then k, then v
    o: torch.Tensor               # bf16 [hidden, rank's heads * 128], a partial sum
    q_norm: torch.Tensor          # bf16 [128]
    k_norm: torch.Tensor
    ln_in: torch.Tensor           # bf16 [hidden]
    ln_post: torch.Tensor
    gu: torch.Tensor              # bf16 [2 * rank's width, hidden]: gate, then up
    down: torch.Tensor            # bf16 [hidden, rank's width], a partial sum
    attn_base: torch.Tensor       # bf16 [2, taps, hidden]: attention_conv's base kernels (before, after)
    attn_proj: torch.Tensor       # bf16 [2 * taps * groups, hidden]: their input-made deltas
    mlp_base: torch.Tensor
    mlp_proj: torch.Tensor
    kv: torch.Tensor | None = None  # bf16 [rank's 2 * key heads * 128, hidden]: the context's k, v rows of a 4-bit qkv


@dataclass
class _Meta:
    pos: torch.Tensor             # int64 [rows]: each block row's position
    dst: torch.Tensor             # int64 [rows]: its ring row (slot * ring + position % ring)
    p0: torch.Tensor              # int64 [seqs]: each block's first position
    slot: torch.Tensor            # int64 [seqs]


class DFlash2:
    def __init__(self, model_dir: Path, w, comm, *, slots: int, k: int, device, entries: int = 0) -> None:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        dc = cfg["dflash_config"]
        self.cfg, self.dc = cfg, dc
        self.comm, self.rank, self.world, self.target = comm, w.rank, w.world, w
        self.k = int(k)
        self.rows = 1 + self.k                     # the block: the last kept token, k mask tokens (the reference's)
        self._b = self.rows                        # this draft's block rows: 1 + the most drafts a sequence wants
        self.layers = tuple(int(L) + 1 for L in dc["target_layer_ids"])  # forward() keeps a layer's incoming residual
        self.window = int(cfg["sliding_window"]) - 1                 # keys at most this far behind a row
        self.mask_id = int(dc["mask_token_id"])
        self.top_k = int(dc["selector_top_k"])
        self.group = int(dc["conv_group_size"])
        self.taps = int(dc["conv_kernel_size"])
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg["rope_theta"])
        D, hd = int(cfg["hidden_size"]), int(cfg["head_dim"])
        H, HKV, I = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]), int(cfg["intermediate_size"])
        self.hidden, self.head_dim = D, hd
        self.heads, self.kv_heads = H // self.world, HKV // self.world
        self.slots, self.device = slots, device
        qlo, qhi = span(H * hd, self.world, self.rank, unit=hd)
        klo, khi = span(HKV * hd, self.world, self.rank, unit=hd)
        mlo, mhi = span(I, self.world, self.rank)
        flo, fhi = span(D, self.world, self.rank)
        pack = Pack(Path(model_dir))
        try:
            def get(name, rows=None):
                return pack.get(name, device, rows=rows).contiguous()

            def cols(name, lo, hi):
                return pack.get(name, device)[:, lo:hi].contiguous()

            self.dl: list[DLayer] = []
            for i in range(int(cfg["num_hidden_layers"])):
                p = f"layers.{i}."
                self.dl.append(DLayer(
                    qkv=torch.cat([get(f"{p}self_attn.q_proj.weight", (qlo, qhi)),
                                   get(f"{p}self_attn.k_proj.weight", (klo, khi)),
                                   get(f"{p}self_attn.v_proj.weight", (klo, khi))]),
                    o=cols(f"{p}self_attn.o_proj.weight", qlo, qhi),
                    q_norm=get(f"{p}self_attn.q_norm.weight"), k_norm=get(f"{p}self_attn.k_norm.weight"),
                    ln_in=get(f"{p}input_layernorm.weight"), ln_post=get(f"{p}post_attention_layernorm.weight"),
                    gu=torch.cat([get(f"{p}mlp.gate_proj.weight", (mlo, mhi)), get(f"{p}mlp.up_proj.weight", (mlo, mhi))]),
                    down=cols(f"{p}mlp.down_proj.weight", mlo, mhi),
                    attn_base=get(f"{p}attention_conv.base_kernel"),
                    attn_proj=get(f"{p}attention_conv.kernel_projection.weight", self._dyn_rows(pack, p, "attention")),
                    mlp_base=get(f"{p}mlp_conv.base_kernel"),
                    mlp_proj=get(f"{p}mlp_conv.kernel_projection.weight", self._dyn_rows(pack, p, "mlp"))))
            self.fc = get("fc.weight", (flo, fhi))                     # [rank's hidden columns, 5 * hidden]
            self.hidden_norm = get("hidden_norm.weight")
            self.norm = get("norm.weight")
            self.sel_proj = get("candidate_selector.hidden_projection.weight")       # [rank 256, hidden]
            # bf16 words [vocab, 256] in the page cache: the walk reads a few rows a step
            self.pred_book = pack.map("candidate_selector.predecessor_codebook")
            self.succ_book = pack.map("candidate_selector.successor_codebook")
        finally:
            pack.close()
        self.qmm, self.head4 = None, None
        if DRAFT_BITS == 4 or DRAFT_HEAD_BITS == 4:
            from tensorfold.families.glm5_next.cuda import qmm

            self.qmm = qmm
        if DRAFT_HEAD_BITS == 4:
            self.head4 = qmm.quantize4(w.head)
        elif DRAFT_HEAD_BITS != 16:
            raise ValueError(f"TF_BEAM_DRAFT_HEAD_BITS is 16 or 4, not {DRAFT_HEAD_BITS}")
        if DRAFT_BITS == 4:
            for lw in self.dl:
                lw.kv = lw.qkv[self.heads * hd:].contiguous()
                lw.qkv, lw.o, lw.gu, lw.down = (qmm.quantize4(t) for t in (lw.qkv, lw.o, lw.gu, lw.down))
        elif DRAFT_BITS != 16:
            raise ValueError(f"TF_BEAM_DRAFT_BITS is 16 or 4, not {DRAFT_BITS}")
        if self.qmm is not None:
            torch.cuda.empty_cache()                 # the bf16 copies and the quantizer's scratch go back to the system
        self.inv = ops.inv_freq(self.theta, hd, device)
        # the context: each layer's keys and values of a slot's latest positions (and the block's), at position % ring;
        # the prompt cache's entries keep theirs after the slots (slots + e)
        self.ring = -(-(self.window + self.rows + 63) // 64) * 64
        shape = (len(self.dl), slots + entries, self.ring, self.kv_heads, hd)
        self.ck = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.cv = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.acc = None               # fp32 [rows, rank's fc rows]: fc of the last step's target states
        self.rounds = 0

    def nbytes(self) -> int:
        total = 0
        for t in [self.fc, self.hidden_norm, self.norm, self.sel_proj, self.ck, self.cv, self.head4] + [
                t for lw in self.dl for t in vars(lw).values()]:
            if t is not None:
                total += t.nbytes() if callable(getattr(t, "nbytes", None)) else t.numel() * t.element_size()
        return total

    def _mm(self, x: torch.Tensor, w, f32: bool = False) -> torch.Tensor:
        """x @ w.T: bf16, or fp32 with f32; w bf16 or a 4-bit qmm.Q4."""
        if isinstance(w, torch.Tensor):
            return torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
        return self.qmm.matmul(x, w, f32=f32)

    # -- the context -----------------------------------------------------------------------------------------------

    def start(self, rows: int) -> None:
        """A step's target states start: the fc sum for its rows (grown to the widest step), zeroed."""
        if self.acc is None or self.acc.shape[0] < rows:
            self.acc = None
            self.acc = torch.empty((rows, self.fc.shape[0]), dtype=torch.float32, device=self.device)
        self.acc[:rows].zero_()

    def tap(self, i: int, x: torch.Tensor) -> None:
        """State i of the step's rows (the target's incoming residual of that layer), rounded to bf16, through its
        columns of fc, added to the step's sum."""
        D = self.hidden
        self.acc[:x.shape[0]].add_(torch.mm(x.to(torch.bfloat16), self.fc[:, i * D:(i + 1) * D].t(),
                                            out_dtype=torch.float32))

    def copy_context(self, src: int, dst: int) -> None:
        """Slot (or prompt cache entry, ``slots + e``) ``src``'s context into ``dst``'s."""
        self.ck[:, dst].copy_(self.ck[:, src])
        self.cv[:, dst].copy_(self.cv[:, src])

    @torch.no_grad()
    def context(self, entries, work) -> None:
        """Write entries' (slot, first position, row of the step that tapped them, rows, which) context from the fc
        sum (rounded to bf16, gathered): hidden_norm, then each layer's normed, rotated keys and its values."""
        dev, hd, HQ, HKV = self.device, self.head_dim, self.heads, self.kv_heads
        if len(entries) == 1:
            _, _, at, n, _ = entries[0]
            part = self.acc[at:at + n].to(torch.bfloat16)
        else:
            idx = torch.cat([torch.arange(at, at + n) for _, _, at, n, _ in entries]).to(dev)
            part = self.acc.index_select(0, idx).to(torch.bfloat16)
        pos = torch.cat([torch.arange(first, first + n) for _, first, _, n, _ in entries])
        slot = torch.cat([torch.full((n,), s, dtype=torch.int64) for s, _, _, n, _ in entries])
        dst = (slot * self.ring + pos % self.ring).to(dev)
        pos = pos.to(dev)
        N = part.shape[0]
        if self.world > 1:
            got = torch.empty((self.world,) + tuple(part.shape), dtype=part.dtype, device=dev)
            work.comm.all_gather(part, got)
            part = got.permute(1, 0, 2).reshape(N, -1)
        ctx = ops.rms_norm(part, self.hidden_norm, self.eps)
        for L, lw in enumerate(self.dl):
            kv = torch.mm(ctx, (lw.qkv[HQ * hd:] if lw.kv is None else lw.kv).t())
            k, v = kv[:, :HKV * hd], kv[:, HKV * hd:]
            ops.qk_prep(k[:, :0], k, v, lw.q_norm, lw.k_norm, pos, self.inv, kv.view(N, -1, hd)[:, :0],
                        self.ck[L].view(-1, HKV, hd), self.cv[L].view(-1, HKV, hd), dst, mode=1, eps=self.eps)

    # -- a draft ---------------------------------------------------------------------------------------------------

    def _dyn_rows(self, pack, p: str, which: str) -> tuple[int, int]:
        """This rank's rows of a conv's kernel projection (an even split of its 2 * taps * groups outputs)."""
        n = pack.shape(f"{p}{which}_conv.kernel_projection.weight")[0]
        return span(n, self.world, self.rank, unit=1)

    def _dyn(self, h, proj, work) -> torch.Tensor:
        """The convs' input-made kernel deltas [rows, 2 * taps * groups]: each rank makes its share of the outputs and
        one all-gather brings the rest (a split by output columns: every value keeps its bits)."""
        mine = torch.mm(h, proj.t())
        if self.world == 1:
            return mine
        got = torch.empty((self.world,) + tuple(mine.shape), dtype=mine.dtype, device=mine.device)
        work.comm.all_gather(mine, got)
        return got.permute(1, 0, 2).reshape(mine.shape[0], -1)

    def _conv(self, x, dyn, base, side: int, res=None, nw=None) -> torch.Tensor:
        R, D = x.shape
        out = torch.empty_like(x)
        norm = res is not None
        _conv[(R,)](x, dyn[:, side * self.taps * (D // self.group):], base[side], out, res if norm else x,
                    nw if norm else x, x.stride(0), dyn.stride(0), out.stride(0), (res if norm else x).stride(0),
                    D=D, GS=self.group, BLK=self._b, TAPS=self.taps, NORM=norm, EPS=self.eps,
                    BLOCK=triton.next_power_of_2(D), num_warps=8)
        return out

    def _layer(self, L: int, lw: DLayer, h, res, nxt, m: _Meta, S: int, work) -> torch.Tensor:
        """One layer on the block rows: h the normed input (bf16 [rows, hidden]), res the residual (updated in place);
        returns the next norm of the residual."""
        R, hd, HQ, HKV = h.shape[0], self.head_dim, self.heads, self.kv_heads
        dyn = self._dyn(h, lw.attn_proj, work)
        qkv = self._mm(self._conv(h, dyn, lw.attn_base, 0), lw.qkv)
        q, k, v = qkv[:, :HQ * hd], qkv[:, HQ * hd:(HQ + HKV) * hd], qkv[:, (HQ + HKV) * hd:]
        qo = torch.empty((R, HQ, hd), dtype=torch.bfloat16, device=h.device)
        ck, cv = self.ck[L], self.cv[L]
        ops.qk_prep(q, k, v, lw.q_norm, lw.k_norm, m.pos, self.inv, qo, ck.view(-1, HKV, hd), cv.view(-1, HKV, hd),
                    m.dst, mode=1, eps=self.eps)
        att = torch.empty((R, HQ * hd), dtype=torch.bfloat16, device=h.device)
        G = HQ // HKV
        _dattn[(S, HKV)](qo, ck, cv, att, m.p0, m.slot, qo.stride(0), qo.stride(1), att.stride(0), self.ring,
                         self.window, hd ** -0.5, HKV=HKV, G=G, B=self._b, D=hd,
                         BM=max(16, triton.next_power_of_2(G * self._b)), BK=64, num_warps=4)
        o = work.reduce(self._mm(att, lw.o, f32=True)).to(torch.bfloat16)
        h = self._conv(o, dyn, lw.attn_base, 1, res, lw.ln_post)
        dyn = self._dyn(h, lw.mlp_proj, work)
        gu = self._mm(self._conv(h, dyn, lw.mlp_base, 0), lw.gu)
        n = gu.shape[1] // 2
        d = work.reduce(self._mm(ops.silu_mul(gu[:, :n], gu[:, n:]), lw.down, f32=True))
        return self._conv(d.to(torch.bfloat16), dyn, lw.mlp_base, 1, res, nxt)

    def _candidates(self, h, work):
        """The rows' top candidates over the whole vocabulary: ids int64 and soft-capped logits fp32 [rows, top_k]."""
        if self.head4 is None:
            lg = F.logits(self.target, h)
        else:
            lg = self.qmm.matmul(h, self.head4, f32=True)
            if self.target.cfg.softcap:
                lg = ops.softcap(lg, self.target.cfg.softcap)
        v, i = torch.topk(lg, self.top_k, dim=-1)
        i = i + self.target.head_lo
        if self.world == 1:
            return i, v
        packed = torch.cat([v.view(torch.int32), i.to(torch.int32)], 1).contiguous()
        got = torch.empty((self.world,) + tuple(packed.shape), dtype=torch.int32, device=h.device)
        work.comm.all_gather(packed, got)
        n = h.shape[0]
        vals = got[:, :, :self.top_k].view(torch.float32).permute(1, 0, 2).reshape(n, -1)
        ids = got[:, :, self.top_k:].permute(1, 0, 2).reshape(n, -1)
        best, at = torch.topk(vals, self.top_k, dim=-1)
        return ids.gather(1, at).long(), best

    def walk(self, anchors: np.ndarray, cand: np.ndarray, unary: np.ndarray, hp: np.ndarray, starts) -> np.ndarray:
        """Each sequence's k picks int64 [S, k] on the host: a row's candidates (ids [S, k, top_k], logits fp32)
        scored against the last pick (the anchor first) through the codebooks (hp: the rows' projections, fp32
        [S, k, 256]); the best taken, or drawn by the target's keyed rule at the pick's position."""
        S, K, n = cand.shape
        out = np.empty((S, K), dtype=np.int64)
        prev = anchors
        for t in range(K):
            wgt = bf16(f32(self.pred_book[prev]) * hp[:, t])                      # [S, 256]
            succ = f32(self.succ_book[cand[:, t].reshape(-1)]).reshape(S, n, -1)  # [S, top_k, 256]
            score = unary[:, t] + bf16((succ * wgt[:, None, :]).sum(-1, dtype=np.float32))
            for j, s in enumerate(starts):
                out[j, t] = (cand[j, t, int(np.argmax(score[j]))] if s[4] is None else
                             choose(score[j], cand[j, t], s[1] + 1 + t, s[4]))
            prev = out[:, t]
        return out

    @torch.no_grad()
    def draft(self, ids: torch.Tensor, starts, work) -> None:
        """Fill each sequence's draft rows of ids (host int64). starts: (slot, its last token's position, that
        token's row, drafts wanted <= k, sampling or None)."""
        dev, S, K = self.device, len(starts), max(s[3] for s in starts)
        B = self._b = 1 + K                        # a shorter block when no sequence wants all k drafts
        anchors = ids[[s[2] for s in starts]]
        blk = torch.full((S, B), self.mask_id, dtype=torch.int64)
        blk[:, 0] = anchors
        p0 = torch.tensor([s[1] for s in starts], dtype=torch.int64)
        slot = torch.tensor([s[0] for s in starts], dtype=torch.int64)
        pos = p0[:, None] + torch.arange(B)
        m = _Meta(pos=pos.reshape(-1).to(dev), dst=(slot[:, None] * self.ring + pos % self.ring).reshape(-1).to(dev),
                  p0=p0.to(dev), slot=slot.to(dev))
        res, _ = F.embed(self.target, blk.view(-1) if self.target.embed_host is not None else blk.view(-1).to(dev),
                         work, res_dtype=torch.bfloat16)
        h = ops.rms_norm(res, self.dl[0].ln_in, self.eps)
        for L, lw in enumerate(self.dl):
            h = self._layer(L, lw, h, res, self.dl[L + 1].ln_in if L + 1 < len(self.dl) else self.norm, m, S, work)
        sel = h.view(S, B, -1)[:, 1:].reshape(S * K, -1)
        cand, unary = self._candidates(sel, work)
        hp = torch.mm(sel, self.sel_proj.t()).float()
        picks = self.walk(anchors.numpy(), cand.cpu().numpy().reshape(S, K, -1), unary.cpu().numpy().reshape(S, K, -1),
                          hp.cpu().numpy().reshape(S, K, -1), starts)
        for j, s in enumerate(starts):
            ids[s[2] + 1:s[2] + 1 + s[3]] = torch.from_numpy(picks[j, :s[3]])
        self.rounds += 1
