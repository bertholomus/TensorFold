"""DSpark drafting (DeepSeek's in-checkpoint draft blocks, ``mtp.*``) and exact verification.

The drafter only proposes. Its three blocks attend a window of the target's projected taps (``main_x``: main_proj
and main_norm of the mean-over-streams inputs of layers 37/38/39) and a block of [last token, noise x 4]; one pass
gives five base logits, the Markov head adds its bias row by row. The target then runs the pending token and k drafts
as one window and keeps the matching prefix plus its own next token, so every emitted token is the target's.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ops import BF16, F32, rms_norm
from . import kernels as K
from .model import KV_QUANT, Model, mm


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

    def new_cache(self) -> DraftCache:
        c = self.m.cfg
        return DraftCache([torch.zeros((self.ring_size, c.head_dim), dtype=BF16, device="cuda")
                           for _ in self.dw.blocks], self.ring_size)

    def _cs(self, lay, sc):
        return self.m._cs(lay.idx, sc)

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
        for lay, ring in zip(self.dw.blocks, dc.rings):
            cos, sin = self._cs(lay, sc)
            y = mm(lay.wkv, main_x)
            K.kv_norm_rope(y, lay.kv_norm, cos, sin, pos, ring, pos % self.ring_size, c.eps, KV_QUANT, c.rope_dim)
        dc.absorbed = start + n

    def _attention(self, lay, x, sc, ring, q0, pos=None, wpos=None):
        c = self.m.cfg
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        cos, sin = self._cs(lay, sc)
        if pos is None:
            pos = torch.arange(q0, q0 + n, device=x.device)
        qr = K.rmsnorm(mm(lay.wq_a, x), lay.q_norm, c.eps)
        q = mm(lay.wq_b, qr).view(n, self.m.Hl, hd)
        K.rope_heads(q, cos, sin, pos, rd)
        kvb = K.kv_norm_rope(mm(lay.wkv, x), lay.kv_norm, cos, sin, pos, ring, self.m._neg(n), c.eps, KV_QUANT, rd)
        bidx = self._block_idx.get(n)
        if bidx is None:
            bidx = self._block_idx[n] = torch.arange(n, device=x.device).repeat(n, 1).contiguous()
        if wpos is None:
            wpos = torch.full((n,), q0 - 1, dtype=torch.int64, device=x.device)
        # every row sees the 128 newest absorbed positions and every block row (no mask inside the block)
        o = K.sparse_attn(q, lay.sink, ring, self.m._zero, True, kvb, bidx, wpos, hd ** -0.5, c.window)
        K.rope_heads(o, cos, sin, pos, rd, inverse=True)
        og = o.view(n, len(lay.wo_a), -1)
        u = torch.cat([mm(wo, og[:, g].contiguous()) for g, wo in enumerate(lay.wo_a)], -1)
        return mm(lay.wo_b, u, F32)

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
        logits = m.comm.gather(local).permute(1, 0, 2).reshape(n, -1)
        out = torch.empty((n + 1,), dtype=torch.long, device=dev)
        out[0:1] = self.token
        head = d.dw.markov_head                                           # [V, rank] fp16
        embs = []
        for i in range(n):
            e = d.dw.markov_embed[out[i:i + 1]][0]
            embs.append(e)
            out[i + 1:i + 2] = (logits[i] + (head @ e.to(torch.float16)).float()).argmax().view(1)
        conf_in = torch.cat([xc.float(), torch.stack(embs).float()], -1)
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

    def __init__(self, drafter: Drafter, sc, dpool: DraftPool, streams: int):
        self.d, self.sc, self.dp, self.N = drafter, sc, dpool, streams
        dev = "cuda"
        n = drafter.size
        self.tokens = torch.zeros((streams,), dtype=torch.long, device=dev)
        self.q0 = torch.zeros((streams,), dtype=torch.long, device=dev)
        self.slots = torch.zeros((streams,), dtype=torch.long, device=dev)
        self.bidx = (torch.arange(streams, device=dev)[:, None, None] * n +
                     torch.arange(n, device=dev)[None, None, :]).expand(streams, n, n).reshape(streams * n, n).contiguous()
        self.zero = torch.zeros((streams * n,), dtype=torch.long, device=dev)
        self.graph = None

    def _attention(self, lay, x, ring, pos, wpos, rbase):
        d, c = self.d, self.d.m.cfg
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        cos, sin = d._cs(lay, self.sc)
        qr = K.rmsnorm(mm(lay.wq_a, x), lay.q_norm, c.eps)
        q = mm(lay.wq_b, qr).view(n, d.m.Hl, hd)
        K.rope_heads(q, cos, sin, pos, rd)
        kvb = K.kv_norm_rope(mm(lay.wkv, x), lay.kv_norm, cos, sin, pos, ring, d.m._neg(n), c.eps, KV_QUANT, rd)
        o = K.sparse_attn(q, lay.sink, ring, d.m._zero, True, kvb, self.bidx, wpos, hd ** -0.5, c.window,
                          wbase=rbase, cbase=self.zero, ring_rows=self.dp.ring_size)
        K.rope_heads(o, cos, sin, pos, rd, inverse=True)
        og = o.view(n, len(lay.wo_a), -1)
        u = torch.cat([mm(wo, og[:, g].contiguous()) for g, wo in enumerate(lay.wo_a)], -1)
        return mm(lay.wo_b, u, F32)

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
        for lay, ring in zip(d.dw.blocks, self.dp.rings):
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            K.hc_post(m.comm.gather(self._attention(lay, x, ring, pos, wpos, rbase)), h, post, comb, h)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            K.hc_post(m.comm.gather(m.moe(lay, x, topk=d.topk)), h, post, comb, h)
            pre, pre_f = pre_f, pre
        xc = K.collapse(h, pre)
        local = mm(m.w.head, K.rmsnorm(xc, d.dw.norm, c.eps), F32)
        logits = m.comm.gather(local).permute(1, 0, 2).reshape(R, -1).view(N, n, -1)
        out = torch.empty((N, n + 1), dtype=torch.long, device=dev)
        out[:, 0] = self.tokens
        head = d.dw.markov_head                                           # [V, rank] fp16
        embs = []
        for i in range(n):
            e = d.dw.markov_embed[out[:, i]]                              # [N, rank]
            embs.append(e)
            out[:, i + 1] = (logits[:, i] + (e.to(torch.float16) @ head.t()).float()).argmax(-1)
        conf_in = torch.cat([xc.float().view(N, n, -1), torch.stack(embs, 1).float()], -1)
        conf = (conf_in @ d.dw.conf.float().t())[..., 0]                 # [N, n]
        self.packed = torch.cat([out[:, 1:].to(F32), conf], 1)           # [N, 2 n]: one host read a round

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
        self.tokens.copy_(torch.tensor(tokens, dtype=torch.long))
        self.q0.copy_(torch.tensor(q0, dtype=torch.long))
        self.slots.copy_(torch.tensor(slots, dtype=torch.long))
        if self.graph is None:
            self._body()
        else:
            self.graph.replay()
        n = self.d.size
        return [[int(x) for x in row[:n]] for row in self.packed.tolist()]


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
