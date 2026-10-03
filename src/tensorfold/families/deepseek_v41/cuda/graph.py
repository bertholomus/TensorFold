"""Decode and verify windows as CUDA graphs: one capture per (rows, context bucket), replayed at any position.

Everything position-dependent is read from device buffers (positions, token ids, Engram rows), so a graph captured
once runs every step: the window ring slots, RoPE rows, compressor groups (written to a scratch row when a group is
not complete yet), and the indexer's visible counts. The indexer scans a fixed number of compressed entries per
context bucket (masked past each row's visible count), so the arithmetic of a row is the same as the eager path.
"""

from __future__ import annotations

import os

import torch

from ..ops import BF16, F32, fp4_qd
from . import kernels as K
from .model import KV_QUANT, RAW, Model, SeqCache, _candidates, mm

BUCKET_MIN = int(os.environ.get("TF_DS_BUCKET_MIN") or 1024)


def bucket_for(n_tokens: int, cap: int) -> int:
    b = BUCKET_MIN
    while b < n_tokens:
        b *= 2
    return min(b, max(cap, BUCKET_MIN))


class StaticDecoder:
    """Graph-captured forward of R rows for one sequence cache layout (capacity fixed at creation)."""

    def __init__(self, model: Model, sc: SeqCache, rows: int, bucket: int, taps: bool):
        self.m, self.sc, self.R, self.bucket, self.want_taps = model, sc, rows, bucket, taps
        c = model.cfg
        dev = "cuda"
        self.ids = torch.zeros((rows,), dtype=torch.long, device=dev)
        self.pos = torch.zeros((rows,), dtype=torch.long, device=dev)
        self.e_in = {}
        if model.engram is not None:
            lo, hi = model.engram.cols
            for i in c.engram_layers:
                if i < len(model.w.layers):
                    self.e_in[i] = torch.zeros((rows, (hi - lo) * c.engram_dim), dtype=BF16, device=dev)
        self.graph = None
        self.logits = None
        self.taps = None

    # -- the captured body ---------------------------------------------------------------------------------------
    def _attention(self, lay, x, shared):
        m, c, sc = self.m, self.m.cfg, self.sc
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        ratio = lay.ratio
        pos = self.pos
        cos, sin = m._cs(lay.idx, sc)
        qr = K.rmsnorm(mm(lay.wq_a, x), lay.q_norm, c.eps)
        q = mm(lay.wq_b, qr).view(n, m.Hl, hd)
        K.rope_heads(q, cos, sin, pos, rd)
        ring = sc.ring[lay.idx]
        R = sc.ring_size
        K.kv_norm_rope(mm(lay.wkv, x), lay.kv_norm, cos, sin, pos, ring, pos % R, c.eps, KV_QUANT, rd)
        comp, cidx = None, None
        if ratio:
            if lay.comp_wkv is not None:
                scratch = sc.comp[lay.idx].shape[0] - 1
                if ratio == 1:
                    lat = K.rmsnorm(mm(lay.comp_wkv, x), lay.comp_norm, c.eps)
                    groups = pos
                    target = pos
                else:
                    kvr = mm(lay.comp_wkv, x, F32)
                    scr = mm(lay.comp_wgate, x, F32)
                    rk, rs = sc.comp_raw[lay.idx]
                    rk[pos % RAW] = kvr
                    rs[pos % RAW] = scr
                    groups = pos // ratio
                    first = groups * ratio
                    gi = (first[:, None] + torch.arange(ratio, device=x.device)[None]) % RAW     # [n, ratio]
                    kvg, sg = rk[gi], rs[gi]
                    lat = K.rmsnorm((kvg * sg.softmax(dim=1)).sum(1).to(BF16), lay.comp_norm, c.eps)
                    target = torch.where((pos + 1) % ratio == 0, groups, scratch)
                shared["kv_layer"] = lay.idx
                if lay.idx_wk is not None:
                    k = K.rmsnorm(mm(lay.idx_wk, lat), lay.idx_k_norm, c.eps).view(n, 1, c.idx_dim)
                    K.rope_heads(k, cos, sin, groups * ratio, rd)
                    k = k.view(n, c.idx_dim)
                    if KV_QUANT:
                        k = fp4_qd(k, 32, e4m3_scale=False)
                    sc.index_k[lay.idx][target] = k
                lat = lat.clone().view(n, 1, hd)
                K.rope_heads(lat, cos, sin, groups * ratio, rd)
                lat = lat.view(n, hd)
                if KV_QUANT:
                    lat = fp4_qd(lat, 16, e4m3_scale=True)
                sc.comp[lay.idx][target] = lat
            src = shared["kv_layer"]
            nb = self.bucket // ratio
            vis = (pos + 1) // ratio
            if lay.idx_wq_b is not None:
                iq = mm(lay.idx_wq_b, qr).view(n, c.idx_heads, c.idx_dim)
                K.rope_heads(iq, cos, sin, pos, rd)
                if KV_QUANT:
                    iq = fp4_qd(iq, 32, e4m3_scale=False)
                wts = K.rowmm(x, lay.idx_proj_h).to(BF16) * (c.idx_dim ** -0.5 * c.idx_heads ** -0.5)
                score = K.index_score(iq, sc.index_k[src], wts, vis, nb)
                if lay.idx == c.cand_source:
                    shared["cand"] = _candidates(score, vis[:, None], c.cand_blocks, c.cand_block)
                elif 0 <= c.cand_source < lay.idx:
                    score.masked_fill_(~shared["cand"], float("-inf"))
                kk = min(c.idx_topk, nb)
                top = score.topk(kk, dim=-1, sorted=False).indices.sort(dim=-1).values
                shared["topk"] = torch.where(top < vis[:, None], top, -1).contiguous()
            cidx = shared["topk"]
            comp = sc.comp[src]
        o = K.sparse_attn(q, lay.sink, ring, m._zero, True, comp, cidx, pos, hd ** -0.5, c.window)
        K.rope_heads(o, cos, sin, pos, rd, inverse=True)
        og = o.view(n, len(lay.wo_a), -1)
        u = torch.cat([mm(wo, og[:, g].contiguous()) for g, wo in enumerate(lay.wo_a)], -1)
        return mm(lay.wo_b, u, F32)

    def _body(self):
        m, c, w = self.m, self.m.cfg, self.m.w
        n = self.R
        dev = "cuda"
        self.sc.tokens[self.pos] = self.ids
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
            fn, scale, base = lay.hc_attn
            K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb, part)
            K.hc_post(m.comm.gather(self._attention(lay, x, shared)), h, post, comb, h)
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
            self._body()                       # warm-up: builds kernels and the allocator's blocks
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self._body()
        torch.cuda.synchronize()

    def run(self, ids: list[int], start: int, e_rows: dict | None) -> torch.Tensor:
        n = self.R
        self.ids.copy_(torch.tensor(ids, dtype=torch.long), non_blocking=False)
        self.pos.copy_(torch.arange(start, start + n, dtype=torch.long), non_blocking=False)
        if e_rows:
            for i, t in e_rows.items():
                self.e_in[i].copy_(t)
        self.graph.replay()
        return self.logits


class GraphRunner:
    """Graphs per (rows, bucket) for one cache (capacity); captured lazily, shared memory pool."""

    def __init__(self, model: Model, max_rows: int):
        self.m = model
        self.max_rows = max_rows
        self.graphs: dict = {}
        self.pool = None
        self.sc = None

    def bind(self, sc: SeqCache) -> None:
        if self.sc is not sc:
            self.graphs.clear()
            self.sc = sc

    def forward(self, sc: SeqCache, ids: list[int], start: int, taps: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        m = self.m
        self.bind(sc)
        n = len(ids)
        b = bucket_for(start + n, sc.cap)
        key = (n, b, taps)
        g = self.graphs.get(key)
        e_rows = None
        if m.engram is not None:
            del sc.host[start:]
            sc.host.extend(int(t) for t in ids)
            hashes = m.engram.hashes(sc.host, start, n)
            lo, hi = m.engram.cols
            e_rows = {i: m.engram.rows(i, hashes[:, m.cfg.engram_layers.index(i), lo:hi])
                      for i in m.cfg.engram_layers if i < len(m.w.layers)}
        if g is None:
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            g = StaticDecoder(m, sc, n, b, taps)
            # capture with this step's inputs in place: the warm-up and capture write the caches for these rows,
            # which the replay below rewrites with the same values
            g.ids.copy_(torch.tensor(ids, dtype=torch.long))
            g.pos.copy_(torch.arange(start, start + n, dtype=torch.long))
            for i, t in (e_rows or {}).items():
                g.e_in[i].copy_(t)
            g.capture(self.pool)
            self.graphs[key] = g
        out = g.run(ids, start, e_rows)
        sc.length = start + n
        return out, g.taps
