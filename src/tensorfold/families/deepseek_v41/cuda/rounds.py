"""Concurrent decode rounds (``--parallel``): one forward over every live stream's window, each row at its own
stream's position and in its own stream's cache rows (a pool slot), read from device tables.

A row's arithmetic is its solo window's: every kernel is row-invariant, the cache kernels take the row's slot base
(the ring, the compressed rows, the indexer keys, the compressor's raw inputs), and selection is a total order
(``kernels.topk_indices``), so a row's top-k does not depend on the round's bucket or the rows beside it. Rounds keep
the decode windows' kernels by staying at 16 rows or fewer (the row thresholds the solo windows sit under). Prompts
never come here: a stream's prompt fills its slot through the single-stream path on the slot's views (model.py).
"""

from __future__ import annotations

import torch

from ..ops import BF16, F32, fp4_qd
from . import kernels as K
from .graph import BUCKET_MIN, bucket_for
from .model import KV_QUANT, RAW, Model, PoolCache, SeqCache, _candidates, apply_candidates, mm, store_rows

MAX_ROWS = 16


class RoundDecoder:
    """Graph-captured forward of R rows from several streams over a pool; inputs in device buffers: token ids,
    positions, slots, Engram rows."""

    def __init__(self, model: Model, pool: PoolCache, rows: int, bucket: int, taps: bool):
        if rows > MAX_ROWS:
            raise ValueError(f"a concurrent round holds {MAX_ROWS} rows at most, not {rows}")
        self.m, self.pool, self.R, self.bucket, self.want_taps = model, pool, rows, bucket, taps
        c = model.cfg
        dev = "cuda"
        self.ids = torch.zeros((rows,), dtype=torch.long, device=dev)
        self.pos = torch.zeros((rows,), dtype=torch.long, device=dev)
        self.slot = torch.zeros((rows,), dtype=torch.long, device=dev)
        self.base = torch.zeros((rows,), dtype=torch.long, device=dev)     # the row's stream's extent, positions
        self.end = torch.ones((rows,), dtype=torch.long, device=dev)
        self.table = SeqCache(cap=pool.cap)                                # RoPE tables for any pool position
        self.e_in = {}
        if model.engram is not None:
            lo, hi = model.engram.cols
            for i in c.engram_layers:
                if i < len(model.w.layers):
                    self.e_in[i] = torch.zeros((rows, (hi - lo) * c.engram_dim), dtype=BF16, device=dev)
        self.graph = None
        self.logits = None
        self.taps = None

    def _attention(self, lay, x, shared, cos, sin):
        m, c, pool = self.m, self.m.cfg, self.pool
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        ratio = lay.ratio
        pos, slot = self.pos, self.slot
        RS = pool.ring_size
        qr = K.rmsnorm(mm(lay.wq_a, x), lay.q_norm, c.eps)
        q = mm(lay.wq_b, qr).view(n, m.Hl, hd)
        K.rope_heads(q, cos, sin, pos, rd)
        ring = pool.ring[lay.idx]
        wbase = slot * RS
        K.kv_norm_rope(mm(lay.wkv, x), lay.kv_norm, cos, sin, pos, ring, wbase + pos % RS, c.eps, KV_QUANT, rd)
        comp, cidx, cbase = None, None, None
        if ratio:
            cbase = self.base // ratio
            if lay.comp_wkv is not None:
                scratch = self.end // ratio - 1 - cbase        # the last row of the stream's extent
                if ratio == 1:
                    lat = K.rmsnorm(mm(lay.comp_wkv, x), lay.comp_norm, c.eps)
                    groups = pos
                    target = pos
                else:
                    kvr = mm(lay.comp_wkv, x, F32)
                    scr = mm(lay.comp_wgate, x, F32)
                    rk, rs = pool.comp_raw[lay.idx]
                    rbase = slot * RAW
                    rk[rbase + pos % RAW] = kvr
                    rs[rbase + pos % RAW] = scr
                    groups = pos // ratio
                    first = groups * ratio
                    gi = rbase[:, None] + (first[:, None] + torch.arange(ratio, device=x.device)[None]) % RAW
                    kvg, sg = rk[gi], rs[gi]
                    lat = K.rmsnorm((kvg * sg.softmax(dim=1)).sum(1).to(BF16), lay.comp_norm, c.eps)
                    target = torch.where((pos + 1) % ratio == 0, groups, scratch)
                shared["kv_layer"] = lay.idx
                if lay.idx_wk is not None:
                    k = K.rmsnorm(mm(lay.idx_wk, lat), lay.idx_k_norm, c.eps).view(n, 1, c.idx_dim)
                    K.rope_heads(k, cos, sin, groups * ratio, rd)
                    k = k.view(n, c.idx_dim)
                    store_rows(pool.index_k[lay.idx], cbase + target, k, 32, False)
                lat = lat.clone().view(n, 1, hd)
                K.rope_heads(lat, cos, sin, groups * ratio, rd)
                lat = lat.view(n, hd)
                store_rows(pool.comp[lay.idx], cbase + target, lat, 16, True)
            src = shared["kv_layer"]
            nb = self.bucket // ratio
            vis = (pos + 1) // ratio
            if lay.idx_wq_b is not None:
                iq = mm(lay.idx_wq_b, qr).view(n, c.idx_heads, c.idx_dim)
                K.rope_heads(iq, cos, sin, pos, rd)
                if KV_QUANT:
                    iq = fp4_qd(iq, 32, e4m3_scale=False)
                wts = K.rowmm(x, lay.idx_proj_h).to(BF16) * (c.idx_dim ** -0.5 * c.idx_heads ** -0.5)
                score = K.index_score(iq, pool.index_k[src], wts, vis, nb, base=cbase)
                if lay.idx == c.cand_source:
                    shared["cand"] = _candidates(score, vis[:, None], c.cand_blocks, c.cand_block)
                elif 0 <= c.cand_source < lay.idx:
                    apply_candidates(score, shared["cand"], c.cand_block)
                kk = min(c.idx_topk, nb)
                top = K.topk_indices(score, kk)
                shared["topk"] = torch.where(top < vis[:, None], top, -1).contiguous()
            cidx = shared["topk"]
            comp = pool.comp[src]
        o = K.sparse_attn(q, lay.sink, ring, m._zero, True, comp, cidx, pos, hd ** -0.5, c.window,
                          wbase=wbase, cbase=cbase, ring_rows=RS)
        K.rope_heads(o, cos, sin, pos, rd, inverse=True)
        og = o.view(n, len(lay.wo_a), -1)
        u = torch.cat([mm(wo, og[:, g].contiguous()) for g, wo in enumerate(lay.wo_a)], -1)
        return mm(lay.wo_b, u, F32)

    def _body(self):
        m, c, w = self.m, self.m.cfg, self.m.w
        n = self.R
        dev = "cuda"
        view = self.table
        h = w.embed[self.ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device=dev)
        pre[:, 0] = 1.0
        x = torch.empty((n, c.dim), dtype=BF16, device=dev)
        part = torch.empty((n * K.HC_BLOCKS * 32,), dtype=F32, device=dev)
        pre_a = torch.empty((n, c.hc), dtype=F32, device=dev)
        pre_f = torch.empty((n, c.hc), dtype=F32, device=dev)
        post = torch.empty((n, c.hc), dtype=F32, device=dev)
        comb = torch.empty((n, c.hc, c.hc), dtype=F32, device=dev)
        shared: dict = {}
        taps = []
        for lay in w.layers:
            if lay.engram_wkv is not None and lay.idx in self.e_in:
                kv = m.comm.sum(mm(lay.engram_wkv, self.e_in[lay.idx], F32)).to(BF16)
                h = K.engram_gate(h, kv.contiguous(), lay.engram_qk, c.eps)
            if self.want_taps and lay.idx in c.dspark_taps:
                taps.append(h.to(F32).mean(1).to(BF16))
            cos, sin = m._cs(lay.idx, view)
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            K.hc_post(m.comm.gather(self._attention(lay, x, shared, cos, sin)), h, post, comb, h)
            fn, scale, base = lay.hc_ffn
            K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb, part)
            K.hc_post(m.comm.gather(m.moe(lay, x)), h, post, comb, h)
            pre, pre_f = pre_f, pre
        xc = K.collapse_norm(h, pre.contiguous(), w.norm, c.eps)
        local = mm(w.head, xc, F32)
        g = m.comm.gather(local)
        self.logits = g.permute(1, 0, 2).reshape(n, -1)
        self.taps = torch.cat(taps, -1) if taps else None

    def capture(self, pool=None) -> None:
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

    def set(self, ids: list[int], pos: list[int], slots: list[int], base: list[int], end: list[int]) -> None:
        self.ids.copy_(torch.tensor(ids, dtype=torch.long), non_blocking=False)
        self.pos.copy_(torch.tensor(pos, dtype=torch.long), non_blocking=False)
        self.slot.copy_(torch.tensor(slots, dtype=torch.long), non_blocking=False)
        self.base.copy_(torch.tensor(base, dtype=torch.long), non_blocking=False)
        self.end.copy_(torch.tensor(end, dtype=torch.long), non_blocking=False)

    def run(self, ids: list[int], pos: list[int], slots: list[int], base: list[int], end: list[int],
            e_rows: dict | None) -> torch.Tensor:
        self.set(ids, pos, slots, base, end)
        if e_rows:
            for i, t in e_rows.items():
                self.e_in[i].copy_(t)
        if self.graph is None:
            self._body()
        else:
            self.graph.replay()
        return self.logits


class RoundRunner:
    """Round graphs per (rows, bucket) over one pool, captured on first use (or ahead, ``warm``)."""

    def __init__(self, model: Model, pool: PoolCache, graphs: bool = True, graph_pool=None):
        self.m, self.pool = model, pool
        self.graphs: dict = {} if graphs else None
        self.graph_pool = graph_pool
        self.captures = 0

    def bucket(self, deepest: int) -> int:
        return bucket_for(deepest, self.pool.cap)

    def forward(self, windows: list[tuple]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``windows``: each stream's (slot, extent base, extent size, first position, token ids, host ids so far):
        rows in that order. Returns logits [R, V] and taps [R, 3 d] (the rows in window order)."""

        m = self.m
        ids, pos, slots, base, end = [], [], [], [], []
        hashes = []
        for slot, b0, size, p0, toks, host in windows:
            n = len(toks)
            ids += toks
            pos += range(p0, p0 + n)
            slots += [slot] * n
            base += [b0] * n
            end += [b0 + size] * n
            if m.engram is not None:
                hashes.append(m.engram.hashes(host, p0, n))
        R = len(ids)
        b = self.bucket(max(p + 1 for p in pos))
        e_rows = None
        if m.engram is not None:
            import numpy as np

            hs = np.concatenate(hashes, 0)
            lo, hi = m.engram.cols
            e_rows = {i: m.engram.rows(i, hs[:, m.cfg.engram_layers.index(i), lo:hi])
                      for i in m.cfg.engram_layers if i < len(m.w.layers)}
        key = (R, b)
        g = self.graphs.get(key) if self.graphs is not None else None
        if g is None:
            g = RoundDecoder(m, self.pool, R, b, True)
            g.set(ids, pos, slots, base, end)
            for i, t in (e_rows or {}).items():
                g.e_in[i].copy_(t)
            if self.graphs is not None:
                if self.graph_pool is None:
                    self.graph_pool = torch.cuda.graph_pool_handle()
                g.capture(self.graph_pool)
                self.graphs[key] = g
                self.captures += 1
        out = g.run(ids, pos, slots, base, end, e_rows)
        return out, g.taps
