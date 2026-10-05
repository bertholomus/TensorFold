"""DSpark drafting (DeepSeek's in-checkpoint draft blocks, ``mtp.*``) and exact verification.

The drafter only proposes. Its three blocks attend a window of the target's projected taps (``main_x``: main_proj
and main_norm of the mean-over-streams inputs of layers 37/38/39) and a block of [last token, noise x 4]; one pass
gives five base logits, the Markov head adds its bias row by row. The target then runs the pending token and k drafts
as one window and keeps the matching prefix plus its own next token, so every emitted token is the target's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ..ops import BF16, F32, rms_norm
from . import kernels as K
from . import markov as _markov
from . import model as _model
from . import rounds as _rounds
from .model import KV_QUANT, Model, attn_in, l2_fork, l2_join, mm, q_proj, wo_a_out, wo_ab

# TF_DS_DRAFT_FUSED=1 (default): the batched drafter graph's stages take the decode round's fused kernels (each
# sublayer's hc_post fused into the next hc_pre, q's RMSNorm writing wq_b's rotated rows with q's RoPE in wq_b's
# epilogue, wo_a's epilogue writing wo_b's rotated rows; each under its kernels.SMALL_SWITCHES switch) and its paced L2
# prefetch forks (model.L2_PREFETCH: wo_a / wo_b after wq_b, the MoE's mix, gate and shared expert after wo_b, the next
# stage's first weights or the head's after the experts). The same arithmetic in the same order: the same drafts.
FUSED = os.environ.get("TF_DS_DRAFT_FUSED", "1") != "0"


@dataclass
class DraftCache:
    rings: list               # per stage [RING, head_dim] bf16: main_x window keys by position
    ring_size: int
    absorbed: int = 0         # positions < absorbed are in the rings


class Drafter:
    def __init__(self, model: Model):
        self.m = model
        self.dw = model.w.dspark
        c = model.cfg
        self.size = c.dspark_block
        self.noise = c.dspark_noise
        self.topk = c.dspark_topk
        self.ring_size = c.window + 16
        self._block_idx: dict[int, torch.Tensor] = {}   # per row count: a graph keeps the tensor it captured
        self._markov: dict = {}
        self.markov()                                    # the cached bias rows: made at load, before the pools

    def markov(self) -> "_markov.Markov | None":
        """The Markov loop's kernels and cached rows (markov.py) under the current flags, or None (3b9818d's loop)."""

        if not _markov.ON:
            return None
        key = (_markov.SPLIT, _markov.CACHE_ROWS, _markov.SUB)
        mk = self._markov.get(key)
        if mk is None:
            mk = self._markov[key] = _markov.Markov(self.m)
        return mk

    def new_cache(self) -> DraftCache:
        c = self.m.cfg
        return DraftCache([torch.zeros((self.ring_size, c.head_dim), dtype=BF16, device="cuda")
                           for _ in self.dw.blocks], self.ring_size)

    def _cs(self, lay, sc):
        return self.m._cs(lay.idx, sc)

    def stage_kv(self, main_x: torch.Tensor) -> list:
        """Every stage's wkv of main_x: one grouped launch at decode sizes (model.GROUPED), else one mm a stage; the
        same bits either way."""

        if not _model.GROUPED or _model.EXACT_MM or main_x.shape[0] > 128:
            return [mm(lay.wkv, main_x) for lay in self.dw.blocks]
        group = getattr(self, "_kv_group", None)
        if group is None:
            from tensorfold.cuda.exl3.linear import Exl3Group

            group = self._kv_group = Exl3Group([lay.wkv for lay in self.dw.blocks])
        return group([main_x] * len(self.dw.blocks), out_dtypes=[BF16] * len(self.dw.blocks))

    @torch.inference_mode()
    def absorb(self, dc: DraftCache, sc, taps: torch.Tensor, start: int) -> None:
        """Target taps [n, 3 * d] of positions start .. start + n - 1 into every stage's window ring."""

        c = self.m.cfg
        n = taps.shape[0]
        keep = min(n, self.ring_size)
        taps = taps[-keep:]
        first = start + n - keep
        pos = torch.arange(first, first + keep, device=taps.device)
        main_x = K.rmsnorm(mm(self.dw.main_proj, taps.contiguous()), self.dw.main_norm, c.eps)
        for lay, ring, y in zip(self.dw.blocks, dc.rings, self.stage_kv(main_x)):
            cos, sin = self._cs(lay, sc)
            K.kv_norm_rope(y, lay.kv_norm, cos, sin, pos, ring, pos % self.ring_size, c.eps, KV_QUANT, c.rope_dim)
        dc.absorbed = start + n

    @torch.inference_mode()
    def absorb_many(self, dpool: "DraftPool", sc, items: list, eager: bool = False) -> None:
        """``absorb`` for several streams in one pass over a ring pool: ``items`` (slot, taps [n, 3 d], start) each
        (fewer than ring rows a stream). The same per-row arithmetic as one absorb a stream. ``eager``: launched on a
        side stream before the round's tokens are known (multi.EAGER_ABSORB): the indices go as one pinned copy (a
        pageable one would wait for the forward) and the caller sets each view's ``absorbed``."""

        c = self.m.cfg
        taps = torch.cat([t for _, t, _ in items], 0).contiguous()
        pos_l = [p for _, t, st in items for p in range(st, st + t.shape[0])]
        slot_l = [sl for sl, t, _ in items for _ in range(t.shape[0])]
        if eager:
            idx = torch.tensor([pos_l, slot_l], dtype=torch.long).pin_memory().to(taps.device, non_blocking=True)
            pos, slot = idx[0], idx[1]
        else:
            pos = torch.tensor(pos_l, dtype=torch.long, device=taps.device)
            slot = torch.tensor(slot_l, dtype=torch.long, device=taps.device)
        main_x = K.rmsnorm(mm(self.dw.main_proj, taps), self.dw.main_norm, c.eps)
        rows = slot * dpool.ring_size + pos % dpool.ring_size
        for lay, ring, y in zip(self.dw.blocks, dpool.rings, self.stage_kv(main_x)):
            cos, sin = self._cs(lay, sc)
            K.kv_norm_rope(y, lay.kv_norm, cos, sin, pos, ring, rows, c.eps, KV_QUANT, c.rope_dim)
        if not eager:
            for sl, t, st in items:
                dpool.views[sl].absorbed = st + t.shape[0]

    def _attention(self, lay, x, sc, ring, q0, pos=None, wpos=None):
        c = self.m.cfg
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        cos, sin = self._cs(lay, sc)
        if pos is None:
            pos = torch.arange(q0, q0 + n, device=x.device)
        qa, ykv, _, _ = attn_in(lay, x, comp=False)                       # wq_a and wkv: one launch
        qr = K.rmsnorm(qa, lay.q_norm, c.eps)
        q = mm(lay.wq_b, qr).view(n, self.m.Hl, hd)
        K.rope_heads(q, cos, sin, pos, rd)
        kvb = K.kv_norm_rope(ykv, lay.kv_norm, cos, sin, pos, ring, self.m._neg(n), c.eps, KV_QUANT, rd)
        bidx = self._block_idx.get(n)
        if bidx is None:
            bidx = self._block_idx[n] = torch.arange(n, device=x.device).repeat(n, 1).contiguous()
        if wpos is None:
            wpos = torch.full((n,), q0 - 1, dtype=torch.int64, device=x.device)
        # every row sees the 128 newest absorbed positions and every block row (no mask inside the block)
        o = K.sparse_attn(q, lay.sink, ring, self.m._zero, True, kvb, bidx, wpos, hd ** -0.5, c.window)
        K.rope_heads(o, cos, sin, pos, rd, inverse=True)
        return mm(lay.wo_b, wo_a_out(lay, o), F32)

    @torch.inference_mode()
    def draft(self, dc: DraftCache, sc, token: int, q0: int) -> tuple[list[int], torch.Tensor]:
        """Greedy drafts d1..d5 after ``token`` (which sits at position q0) and their confidences."""

        m, c = self.m, self.m.cfg
        assert dc.absorbed == q0, (dc.absorbed, q0)
        n = self.size
        dev = "cuda"
        ids = torch.full((n,), self.noise, dtype=torch.long, device=dev)
        ids[0] = token
        h = m.w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device=dev)
        pre[:, 0] = 1.0
        x = torch.empty((n, c.dim), dtype=BF16, device=dev)
        part = torch.empty((n * K.HC_BLOCKS * 32,), dtype=F32, device=dev)
        pre_a = torch.empty((n, c.hc), dtype=F32, device=dev)
        pre_f = torch.empty((n, c.hc), dtype=F32, device=dev)
        post = torch.empty((n, c.hc), dtype=F32, device=dev)
        comb = torch.empty((n, c.hc, c.hc), dtype=F32, device=dev)
        for lay, ring in zip(self.dw.blocks, dc.rings):
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            K.hc_post(m.comm.gather(self._attention(lay, x, sc, ring, q0)), h, post, comb, h)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            K.hc_post(m.comm.gather(m.moe(lay, x, topk=self.topk)), h, post, comb, h)
            pre, pre_f = pre_f, pre
        xc = K.collapse(h, pre)                                           # [n, d] (pre-norm, for the confidence)
        local = mm(m.w.head, K.rmsnorm(xc, self.dw.norm, c.eps), F32)
        logits = m.comm.gather(local).permute(1, 0, 2).reshape(n, -1)
        out, embs = [token], []
        for i in range(n):
            e = self.dw.markov_embed[out[-1]]                             # [rank] bf16
            bias = (self.dw.markov_head.float() @ e.float())             # [V] fp32
            embs.append(e)
            out.append(int((logits[i] + bias).argmax()))
        conf_in = torch.cat([xc.float(), torch.stack(embs).float()], -1)
        conf = conf_in @ self.dw.conf.float().t()
        return out[1:], conf[:, 0]


class DraftGraph:
    """The drafter's pass and its Markov loop as one CUDA graph: token and position from device buffers, the drafts
    (and confidences) left on the device; one read per round."""

    def __init__(self, drafter: Drafter, sc, dc: DraftCache):
        self.d, self.sc, self.dc = drafter, sc, dc
        n = drafter.size
        self.token = torch.zeros((1,), dtype=torch.long, device="cuda")
        self.q0 = torch.zeros((1,), dtype=torch.long, device="cuda")
        self.graph = None
        self.out = None
        self.conf = None
        self.mk = drafter.markov()

    def _body(self):
        d, m, c = self.d, self.d.m, self.d.m.cfg
        n = d.size
        dev = "cuda"
        ids = torch.full((n,), d.noise, dtype=torch.long, device=dev)
        ids[0:1] = self.token
        pos = self.q0 + torch.arange(n, device=dev)
        wpos = (self.q0 - 1).expand(n).contiguous()
        h = m.w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device=dev)
        pre[:, 0] = 1.0
        x = torch.empty((n, c.dim), dtype=BF16, device=dev)
        part = torch.empty((n * K.HC_BLOCKS * 32,), dtype=F32, device=dev)
        pre_a = torch.empty((n, c.hc), dtype=F32, device=dev)
        pre_f = torch.empty((n, c.hc), dtype=F32, device=dev)
        post = torch.empty((n, c.hc), dtype=F32, device=dev)
        comb = torch.empty((n, c.hc, c.hc), dtype=F32, device=dev)
        for lay, ring in zip(d.dw.blocks, self.dc.rings):
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            K.hc_post(m.comm.gather(d._attention(lay, x, self.sc, ring, None, pos=pos, wpos=wpos)), h, post, comb, h)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            K.hc_post(m.comm.gather(m.moe(lay, x, topk=d.topk)), h, post, comb, h)
            pre, pre_f = pre_f, pre
        xc = K.collapse(h, pre)
        local = mm(m.w.head, K.rmsnorm(xc, d.dw.norm, c.eps), F32)
        out = torch.empty((n + 1,), dtype=torch.long, device=dev)
        out[0:1] = self.token
        if self.mk is not None:                                           # markov.py: the same drafts
            self.mk.steps(self.mk.logits_of(local), out.view(1, n + 1), n, n)
            embs = d.dw.markov_embed[out[:n]]
        else:
            logits = m.comm.gather(local).permute(1, 0, 2).reshape(n, -1)
            head = d.dw.markov_head                                       # [V, rank] fp16
            embs = []
            for i in range(n):
                e = d.dw.markov_embed[out[i:i + 1]][0]
                embs.append(e)
                out[i + 1:i + 2] = (logits[i] + (head @ e.to(torch.float16)).float()).argmax().view(1)
            embs = torch.stack(embs)
        conf_in = torch.cat([xc.float(), embs.float()], -1)
        self.out = out[1:]
        self.conf = (conf_in @ d.dw.conf.float().t())[:, 0]
        self.packed = torch.cat([self.out.to(F32), self.conf])       # one host read a round (ids < 2^24)

    def capture(self, pool=None):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self._body()
        torch.cuda.synchronize()

    def run(self, token: int, q0: int):
        self.token.fill_(token)
        self.q0.fill_(q0)
        self.graph.replay()
        v = self.packed.tolist()
        n = self.d.size
        return [int(x) for x in v[:n]], v[n:]


class DraftPool:
    """``slots`` drafter caches in one ring plane a stage (slot s: rows [s * ring, (s + 1) * ring)); ``views[s]`` is
    slot s's DraftCache over its rows (absorb and the single-stream graph use it unchanged)."""

    def __init__(self, drafter: Drafter, slots: int):
        c = drafter.m.cfg
        self.ring_size = drafter.ring_size
        self.rings = [torch.zeros((slots * self.ring_size, c.head_dim), dtype=BF16, device="cuda")
                      for _ in drafter.dw.blocks]
        self.views = [DraftCache([r[s * self.ring_size:(s + 1) * self.ring_size] for r in self.rings], self.ring_size)
                      for s in range(slots)]


class BatchDraftGraph:
    """Drafts for N streams in one pass of the drafter (5 N rows: each stream's [token, noise x 4] at its own
    positions, attending its own ring through per-row bases and its own five block rows) and one batched Markov loop.
    Drafts only propose: they may differ in the last bits from a stream's solo drafts, never a reply."""

    def __init__(self, drafter: Drafter, sc, dpool: DraftPool, streams: int, steps: int | None = None):
        self.d, self.sc, self.dp, self.N = drafter, sc, dpool, streams
        # the Markov loop runs only as far as a round at this many streams verifies (its first drafts are the same)
        self.steps = max(1, min(drafter.size, steps or drafter.size))
        dev = "cuda"
        n = drafter.size
        self.inputs = torch.zeros((3, streams), dtype=torch.long, device=dev)   # one buffer, so one copy fills it
        self.tokens, self.q0, self.slots = self.inputs.unbind(0)
        self.bidx = (torch.arange(streams, device=dev)[:, None, None] * n +
                     torch.arange(n, device=dev)[None, None, :]).expand(streams, n, n).reshape(streams * n, n).contiguous()
        self.zero = torch.zeros((streams * n,), dtype=torch.long, device=dev)
        self.graph = None
        self.mk = drafter.markov() if streams <= _markov.NP else None

    def _attention(self, lay, x, ring, pos, wpos, rbase, fused=False):
        d, m, c = self.d, self.d.m, self.d.m.cfg
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        cos, sin = d._cs(lay, self.sc)
        qa, ykv, _, _ = attn_in(lay, x, comp=False)                       # wq_a and wkv: one launch
        if fused and K.on("rot_q"):
            # q's RMSNorm writes wq_b's rotated rows, wq_b's epilogue applies q's RoPE (rounds.py's q path)
            q = q_proj(lay, qa, c.eps, idx=False, rope=(cos, sin, pos, hd, rd))[1].view(n, m.Hl, hd)
        else:
            qr = K.rmsnorm(qa, lay.q_norm, c.eps)
            q = mm(lay.wq_b, qr).view(n, m.Hl, hd)
            K.rope_heads(q, cos, sin, pos, rd)
        if fused:
            l2_fork(m, lay, "attn")                                       # wo_a, wo_b into L2 after wq_b
        kvb = K.kv_norm_rope(ykv, lay.kv_norm, cos, sin, pos, ring, m._neg(n), c.eps, KV_QUANT, rd)
        o = K.sparse_attn(q, lay.sink, ring, m._zero, True, kvb, self.bidx, wpos, hd ** -0.5, c.window,
                          wbase=rbase, cbase=self.zero, ring_rows=self.dp.ring_size)
        K.rope_heads(o, cos, sin, pos, rd, inverse=True)
        if fused and K.on("rot_wob"):
            out = wo_ab(lay, o)                                           # wo_a's epilogue rotates for wo_b
        else:
            out = mm(lay.wo_b, wo_a_out(lay, o), F32)
        if fused:
            l2_fork(m, lay, "moe")                                        # the MoE's mHC mix, gate, shared expert
        return out

    def _stages_fused(self, h, pre, x, part, pre_a, pre_f, post, comb, pos, wpos, rbase):
        """The stages with each sublayer's hc_post fused into the next hc_pre (kernels.hc_pre2, as rounds.py's
        _layers_fused) and the L2 forks; returns (h, pre)."""

        d, m, c = self.d, self.d.m, self.d.m.cfg
        spare = torch.empty_like(h)
        pending = None                                   # the last sublayer's gathered partials, not yet posted

        def pre_mix(h, params, pre_in, norm, pre_out):
            nonlocal spare, pending
            fn, scale, base = params
            if pending is None:
                K.hc_pre2(h, fn, scale, base, pre_in, norm, c.eps, c.hc_eps, c.hc_iters, x, pre_out, post, comb, part)
                return h
            out = K.hc_pre2(h, fn, scale, base, pre_in, norm, c.eps, c.hc_eps, c.hc_iters, x, pre_out, post, comb,
                            part, gathered=pending, h_out=spare)
            pending = None
            spare = h
            return out

        blocks = d.dw.blocks
        for j, (lay, ring) in enumerate(zip(blocks, self.dp.rings)):
            h = pre_mix(h, lay.hc_attn, pre, lay.attn_norm, pre_a)
            pending = m.comm.gather(self._attention(lay, x, ring, pos, wpos, rbase, fused=True))
            h = pre_mix(h, lay.hc_ffn, pre_a, lay.ffn_norm, pre_f)
            y = m.moe(lay, x, topk=d.topk)
            l2_fork(m, lay, "next", blocks[j + 1] if j + 1 < len(blocks) else m.w.head)
            pending = m.comm.gather(y)
            pre, pre_f = pre_f, pre
        K.hc_post(pending, h, post, comb, h)
        l2_join(m)
        return h, pre

    def _body(self):
        d, m, c = self.d, self.d.m, self.d.m.cfg
        N, n = self.N, d.size
        R = N * n
        dev = "cuda"
        ids = torch.full((N, n), d.noise, dtype=torch.long, device=dev)
        ids[:, 0] = self.tokens
        ids = ids.view(R)
        pos = (self.q0[:, None] + torch.arange(n, device=dev)[None]).reshape(R)
        wpos = (self.q0 - 1)[:, None].expand(N, n).reshape(R).contiguous()
        rbase = (self.slots * self.dp.ring_size)[:, None].expand(N, n).reshape(R).contiguous()
        h = m.w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((R, c.hc), dtype=F32, device=dev)
        pre[:, 0] = 1.0
        x = torch.empty((R, c.dim), dtype=BF16, device=dev)
        part = torch.empty((R * K.HC_BLOCKS * 32,), dtype=F32, device=dev)
        pre_a = torch.empty((R, c.hc), dtype=F32, device=dev)
        pre_f = torch.empty((R, c.hc), dtype=F32, device=dev)
        post = torch.empty((R, c.hc), dtype=F32, device=dev)
        comb = torch.empty((R, c.hc, c.hc), dtype=F32, device=dev)
        if FUSED and K.on("hc"):
            h, pre = self._stages_fused(h, pre, x, part, pre_a, pre_f, post, comb, pos, wpos, rbase)
        else:
            for lay, ring in zip(d.dw.blocks, self.dp.rings):
                fn, scale, base = lay.hc_attn
                K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb,
                         part)
                K.hc_post(m.comm.gather(self._attention(lay, x, ring, pos, wpos, rbase)), h, post, comb, h)
                fn, scale, base = lay.hc_ffn
                K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb,
                         part)
                K.hc_post(m.comm.gather(m.moe(lay, x, topk=d.topk)), h, post, comb, h)
                pre, pre_f = pre_f, pre
        xc = K.collapse(h, pre)
        local = mm(m.w.head, K.rmsnorm(xc, d.dw.norm, c.eps), F32)
        steps = self.steps
        out = torch.empty((N, steps + 1), dtype=torch.long, device=dev)
        out[:, 0] = self.tokens
        if self.mk is not None:
            # markov.py: fixed-order kernels a step (cached bias rows, the vocabulary split over the ranks), every
            # bias the cuBLAS loop's bits: the same drafts
            self.mk.steps(self.mk.logits_of(local), out, n, steps)
            embs = d.dw.markov_embed[out[:, :steps]]                      # [N, steps, rank]: the rows the steps read
        else:
            logits = m.comm.gather(local).permute(1, 0, 2).reshape(R, -1).view(N, n, -1)
            head = d.dw.markov_head                                       # [V, rank] fp16
            embs = []
            for i in range(steps):
                e = d.dw.markov_embed[out[:, i]]                          # [N, rank]
                embs.append(e)
                out[:, i + 1] = (logits[:, i] + (e.to(torch.float16) @ head.t()).float()).argmax(-1)
            embs = torch.stack(embs, 1)
        conf_in = torch.cat([xc.float().view(N, n, -1)[:, :steps], embs.float()], -1)
        conf = (conf_in @ d.dw.conf.float().t())[..., 0]                 # [N, steps]
        self.packed = torch.cat([out[:, 1:].to(F32), conf], 1)           # [N, 2 steps]: one host read a round

    def capture(self, pool=None):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self._body()
        torch.cuda.synchronize()

    def run(self, tokens: list[int], q0: list[int], slots: list[int]) -> list[list[int]]:
        if _rounds.ONE_COPY:
            self.inputs.copy_(torch.tensor([tokens, q0, slots], dtype=torch.long).pin_memory(), non_blocking=True)
        else:
            self.tokens.copy_(torch.tensor(tokens, dtype=torch.long))
            self.q0.copy_(torch.tensor(q0, dtype=torch.long))
            self.slots.copy_(torch.tensor(slots, dtype=torch.long))
        if self.graph is None:
            self._body()
        else:
            self.graph.replay()
        n = self.steps
        rows = self.packed.tolist()
        self.last_conf = [row[n:] for row in rows]       # the confidence head's logits, a list a stream
        return [[int(x) for x in row[:n]] for row in rows]


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0


def spec_decode(model: Model, drafter: Drafter, sc, dc, first: int, max_new: int, k: int, eos: tuple,
                on_tokens=None, stats: SpecStats | None = None) -> list[int]:
    """Greedy decode with DSpark drafts verified k at a time; returns the new tokens (``first`` included).

    ``first`` is the token at position sc.length (sampled, not yet run by the target); the drafter has absorbed every
    position before it.
    """

    out = [first]
    if on_tokens:
        on_tokens([first])
    stats = stats if stats is not None else SpecStats()
    tok = first
    while len(out) < max_new and tok not in eos:
        P = sc.length
        drafts, _conf = drafter.draft(dc, sc, tok, P)
        kk = min(k, max_new - len(out), len(drafts))
        window = [tok] + drafts[:kk]
        taps: list = []
        logits = model.forward(sc, torch.tensor(window, dtype=torch.long, device="cuda"), P, all_logits=True,
                               taps=taps)
        best = logits.argmax(-1).tolist()
        m = 0
        while m < kk and drafts[m] == best[m]:
            m += 1
        new = drafts[:m] + [best[m]]
        stats.rounds += 1
        stats.drafted += kk
        stats.accepted += m
        # keep positions P .. P + m (the pending token and the accepted drafts); later rows are rolled back
        sc.length = P + m + 1
        tap = torch.cat(taps, -1)[:m + 1]
        drafter.absorb(dc, sc, tap, P)
        stop = len(new)
        for i, t in enumerate(new):
            if t in eos:
                stop = i + 1
                break
        new = new[:stop][:max_new - len(out)]
        out += new
        if on_tokens:
            on_tokens(new)
        tok = out[-1]
        if tok in eos:
            break
    return out
