"""Concurrent decode rounds (``--parallel``): one forward over every live stream's window, each row at its own
stream's position and in its own stream's cache rows (a pool slot), read from device tables.

A row's arithmetic is its solo window's: every kernel is row-invariant, the cache kernels take the row's slot base
(the ring, the compressed rows, the indexer keys, the compressor's raw inputs), and selection is a total order
(``kernels.topk_indices``), so a row's top-k does not depend on the round's bucket or the rows beside it. Rounds keep
the decode windows' kernels by staying at 16 rows or fewer (the row thresholds the solo windows sit under). Prompts
never come here: a stream's prompt fills its slot through the single-stream path on the slot's views (model.py).
"""

from __future__ import annotations

import os

import torch

from ..ops import BF16, F32, fp4_qd
from . import kernels as K
from .graph import BUCKET_MIN, bucket_for
from .model import (KV_QUANT, RAW, Model, PoolCache, SeqCache, _candidates, apply_candidates, attn_in, mm, q_proj,
                    store_rows, wo_a_out, wo_a_rot, wo_ab)

MAX_ROWS = K.DECODE_ROWS
# TF_DS_ENGRAM_SPLIT=1 (default): a round is one graph a stretch of layers, cut before each Engram layer, so a layer's
# Engram rows are read while the layers before it run (the arithmetic is the one-graph round's)
ENGRAM_SPLIT = os.environ.get("TF_DS_ENGRAM_SPLIT", "1") == "1"


def _candidates_fast(score: torch.Tensor, vis: torch.Tensor, nblocks: int, bsize: int) -> torch.Tensor:
    """model._candidates' block mask; when the pool takes every block (nblocks >= the blocks there are, contexts up to
    nblocks * bsize entries) it is just "the block holds a finite score, or it is the newest" (topk of every block
    then scatter of s > -inf), without the top-k."""

    width = score.shape[-1]
    nb = -(-width // bsize)
    if nblocks < nb or width % bsize:
        return _candidates(score, vis[:, None], nblocks, bsize)
    s = score.view(score.shape[0], nb, bsize).amax(-1)
    last = (vis - 1) // bsize
    return (s > float("-inf")) | (torch.arange(nb, device=score.device)[None] == last[:, None])


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
        cuts = sorted(i for i in self.e_in if 0 < i < len(model.w.layers)) if ENGRAM_SPLIT else []
        edges = [0, *cuts, len(model.w.layers)]
        self.stretches = list(zip(edges[:-1], edges[1:]))       # [first layer, end layer) a graph
        self.graphs = None
        self.graph = None
        self.logits = None
        self.taps = None
        self._idx = {}

    def _ix(self, key):
        """A round's index arithmetic (kernels switch "glue"): computed once a stretch, on first use, from the round's
        device inputs, instead of once a layer; the same integer ops, so the same values."""

        t = self._idx.get(key)
        if t is None:
            pos, RS = self.pos, self.pool.ring_size
            kind = key[0]
            if kind == "wbase":
                t = self.slot * RS
            elif kind == "wslot":
                t = self._ix(("wbase",)) + pos % RS
            elif kind == "cbase":
                t = self.base // key[1]
            elif kind == "scratch":
                t = self.end // key[1] - 1 - self._ix(("cbase", key[1]))
            elif kind == "rbase":
                t = self.slot * RAW
            elif kind == "rslot":
                t = self._ix(("rbase",)) + pos % RAW
            elif kind == "groups":
                t = pos // key[1]
            elif kind == "gi":
                ratio = key[1]
                first = self._ix(("groups", ratio)) * ratio
                ar = torch.arange(ratio, device=pos.device)[None]
                t = self._ix(("rbase",))[:, None] + (first[:, None] + ar) % RAW
            elif kind == "target":
                ratio = key[1]
                t = torch.where((pos + 1) % ratio == 0, self._ix(("groups", ratio)), self._ix(("scratch", ratio)))
            elif kind == "ctarget":
                ratio = key[1]
                t = self._ix(("cbase", ratio)) + (pos if ratio == 1 else self._ix(("target", ratio)))
            elif kind == "gpos":
                t = (pos if key[1] == 1 else self._ix(("groups", key[1]))) * key[1]
            elif kind == "vis":
                t = (pos + 1) // key[1]
            elif kind == "alltop":
                # the indexer's selection when its top-k takes every scanned entry (bucket / ratio <= idx_topk):
                # topk_indices of all nb keys is 0 .. nb - 1 whatever the scores, then the visible mask
                ar = torch.arange(self.bucket // key[1], device=pos.device)[None]
                t = torch.where(ar < self._ix(("vis", key[1]))[:, None], ar, -1)
            else:
                raise KeyError(key)
            self._idx[key] = t
        return t

    def _attention(self, lay, x, shared, cos, sin):
        if any(K.on(k) for k in ("glue", "rot_q", "rot_wob", "rot_attn", "idx", "comp")):
            return self._attention2(lay, x, shared, cos, sin)
        m, c, pool = self.m, self.m.cfg, self.pool
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        ratio = lay.ratio
        pos, slot = self.pos, self.slot
        RS = pool.ring_size
        qa, ykv, ckv, cgate = attn_in(lay, x, comp=bool(ratio))         # wq_a, wkv, the compressor's: one launch
        qr = K.rmsnorm(qa, lay.q_norm, c.eps)
        q = mm(lay.wq_b, qr).view(n, m.Hl, hd)
        K.rope_heads(q, cos, sin, pos, rd)
        ring = pool.ring[lay.idx]
        wbase = slot * RS
        K.kv_norm_rope(ykv, lay.kv_norm, cos, sin, pos, ring, wbase + pos % RS, c.eps, KV_QUANT, rd)
        comp, cidx, cbase = None, None, None
        if ratio:
            cbase = self.base // ratio
            if lay.comp_wkv is not None:
                scratch = self.end // ratio - 1 - cbase        # the last row of the stream's extent
                if ratio == 1:
                    lat = K.rmsnorm(ckv, lay.comp_norm, c.eps)
                    groups = pos
                    target = pos
                else:
                    kvr, scr = ckv, cgate
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
        return mm(lay.wo_b, wo_a_out(lay, o), F32)

    def _attention2(self, lay, x, shared, cos, sin):
        """_attention with the small-kernel switches (kernels.SMALL_SWITCHES): "glue" takes the round's index tensors
        from _ix; "rot_q" folds wq_b's input rotation into the q RMSNorm and q's RoPE into wq_b's epilogue, "rot_attn"
        the inverse RoPE and wo_a's input rotation into the attention merge, "rot_wob" wo_b's input rotation into
        wo_a's epilogue; "idx" / "comp" fuse the indexer's and the compressor's glue. The same arithmetic in the same
        order, so the same bits."""

        m, c, pool = self.m, self.m.cfg, self.pool
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        ratio = lay.ratio
        pos, slot = self.pos, self.slot
        RS = pool.ring_size
        glue, rot = K.on("glue"), K.on("rot_q")
        ix = self._ix if glue else self._ix_now
        qa, ykv, ckv, cgate = attn_in(lay, x, comp=bool(ratio))         # wq_a, wkv, the compressor's: one launch
        iq = None
        has_idx = bool(ratio) and lay.idx_wq_b is not None
        # the indexer's top-k takes every entry it scans (short contexts): its selection needs no scores at all
        idx_all = (has_idx and K.on("idx") and lay.idx != c.cand_source and self.bucket // ratio <= c.idx_topk)
        need_iq = has_idx and not idx_all
        if rot:
            # q's RMSNorm writes wq_b's (and the indexer wq_b's) rotated rows, in the same launch as the window KV's
            # norm + RoPE into the ring; wq_b's epilogue applies q's RoPE
            kv = (ykv, cos, sin, pos, pool.ring[lay.idx], ix(("wslot",)), KV_QUANT, rd)
            qr, q, iq = q_proj(lay, qa, c.eps, idx=need_iq, rope=(cos, sin, pos, hd, rd), kv=kv)
            q = q.view(n, m.Hl, hd)
        else:
            qr = K.rmsnorm(qa, lay.q_norm, c.eps)
            q = mm(lay.wq_b, qr).view(n, m.Hl, hd)
            K.rope_heads(q, cos, sin, pos, rd)
        ring, wbase, comp, cidx, cbase = self._kv_idx(lay, x, ykv, ckv, cgate, shared, cos, sin, ix, iq, qr, idx_all,
                                                      kv_done=rot)
        if K.on("rot_attn"):
            # the merge, the inverse RoPE and wo_a's input rotation in one launch
            suh, xh, gh = wo_a_rot(lay, n, x.device, hd)
            K.sparse_attn(q, lay.sink, ring, m._zero, True, comp, cidx, pos, hd ** -0.5, c.window,
                          wbase=wbase, cbase=cbase, ring_rows=RS, rot=(cos, sin, rd, suh, xh[0], gh))
            return wo_ab(lay, xh=xh, fold=K.on("rot_wob"))
        o = K.sparse_attn(q, lay.sink, ring, m._zero, True, comp, cidx, pos, hd ** -0.5, c.window,
                          wbase=wbase, cbase=cbase, ring_rows=RS)
        K.rope_heads(o, cos, sin, pos, rd, inverse=True)
        if K.on("rot_wob"):
            return wo_ab(lay, o)                             # wo_a's epilogue rotates for wo_b
        return mm(lay.wo_b, wo_a_out(lay, o), F32)

    def _kv_idx(self, lay, x, ykv, ckv, cgate, shared, cos, sin, ix, iq, qr, idx_all, kv_done=False):
        """The window KV into the ring, the compressor's caches (a kv-source layer) and the indexer's selection:
        (ring, ring base, compressed cache, selected indices, compressed base)."""

        m, c, pool = self.m, self.m.cfg, self.pool
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        ratio = lay.ratio
        pos = self.pos
        rowmm = K.rowmm2 if K.on("rowmm") else K.rowmm
        ring = pool.ring[lay.idx]
        wbase = ix(("wbase",))
        if not kv_done:
            K.kv_norm_rope(ykv, lay.kv_norm, cos, sin, pos, ring, ix(("wslot",)), c.eps, KV_QUANT, rd)
        comp, cidx, cbase = None, None, None
        if ratio:
            cbase = ix(("cbase", ratio))
            if lay.comp_wkv is not None:
                if ratio == 1:
                    lat = K.rmsnorm(ckv, lay.comp_norm, c.eps)
                else:
                    kvr, scr = ckv, cgate
                    rk, rs = pool.comp_raw[lay.idx]
                    rslot = ix(("rslot",))
                    rk[rslot] = kvr
                    rs[rslot] = scr
                    gi = ix(("gi", ratio))
                    kvg, sg = rk[gi], rs[gi]
                    lat = K.rmsnorm((kvg * sg.softmax(dim=1)).sum(1).to(BF16), lay.comp_norm, c.eps)
                shared["kv_layer"] = lay.idx
                ctarget = ix(("ctarget", ratio))
                gpos = ix(("gpos", ratio))
                packed = K.on("comp") and isinstance(pool.comp[lay.idx], tuple)
                if lay.idx_wk is not None:
                    k = K.rmsnorm(mm(lay.idx_wk, lat), lay.idx_k_norm, c.eps).view(n, 1, c.idx_dim)
                    K.rope_heads(k, cos, sin, gpos, rd)
                    k = k.view(n, c.idx_dim)
                    if packed:
                        K.fp4_store(k, pool.index_k[lay.idx], ctarget, 32, False)
                    else:
                        store_rows(pool.index_k[lay.idx], ctarget, k, 32, False)
                lat = (lat if packed else lat.clone()).view(n, 1, hd)     # (lat is not read again: rotate in place)
                K.rope_heads(lat, cos, sin, gpos, rd)
                lat = lat.view(n, hd)
                if packed:
                    K.fp4_store(lat, pool.comp[lay.idx], ctarget, 16, True)
                else:
                    store_rows(pool.comp[lay.idx], ctarget, lat, 16, True)
            src = shared["kv_layer"]
            nb = self.bucket // ratio
            if idx_all:
                shared["topk"] = ix(("alltop", ratio))
            elif lay.idx_wq_b is not None:
                vis = ix(("vis", ratio))
                fused = K.on("idx")
                if iq is None:
                    iq = mm(lay.idx_wq_b, qr)
                iq = iq.view(n, c.idx_heads, c.idx_dim)
                K.rope_heads(iq, cos, sin, pos, rd)
                if KV_QUANT:
                    iq = K.fp4_qd_p2(iq) if fused else fp4_qd(iq, 32, e4m3_scale=False)
                wscale = c.idx_dim ** -0.5 * c.idx_heads ** -0.5
                wts = K.rowmm_wts(x, lay.idx_proj_h, wscale) if fused else rowmm(x, lay.idx_proj_h).to(BF16) * wscale
                kk = min(c.idx_topk, nb)
                if fused and lay.idx != c.cand_source and kk & (kk - 1) == 0:
                    # scores -> (candidate mask) -> top-k keys in one launch, the top-k's indices sorted and masked
                    # in one more (the cand-source layer keeps the scores for _candidates)
                    cand = shared["cand"] if 0 <= c.cand_source < lay.idx else None
                    keys = K.index_keys(iq, pool.index_k[src], wts, vis, nb, base=cbase, cand=cand,
                                        cand_block=c.cand_block)
                    shared["topk"] = K.topk_select(keys, kk, vis)
                elif fused and lay.idx == c.cand_source and kk & (kk - 1) == 0:
                    # the cand-source layer: the scores for the candidate pool, then their keys' top-k as above
                    score = K.index_score(iq, pool.index_k[src], wts, vis, nb, base=cbase)
                    shared["cand"] = _candidates_fast(score, vis, c.cand_blocks, c.cand_block)
                    shared["topk"] = K.topk_select(K.score_keys(score), kk, vis)
                else:
                    score = K.index_score(iq, pool.index_k[src], wts, vis, nb, base=cbase)
                    if lay.idx == c.cand_source:
                        shared["cand"] = _candidates(score, vis[:, None], c.cand_blocks, c.cand_block)
                    elif 0 <= c.cand_source < lay.idx:
                        apply_candidates(score, shared["cand"], c.cand_block)
                    top = K.topk_indices(score, kk)
                    shared["topk"] = torch.where(top < vis[:, None], top, -1).contiguous()
            cidx = shared["topk"]
            comp = pool.comp[src]
        return ring, wbase, comp, cidx, cbase

    def _ix_now(self, key):
        """_ix without keeping the result (the "glue" switch off: every layer computes its own)."""

        saved = self._idx
        self._idx = {}
        try:
            return self._ix(key)
        finally:
            self._idx = saved

    def _body(self):
        for k in range(len(self.stretches)):
            self._stretch(k)

    def _stretch(self, k: int):
        """Layers [first, end) of the round (the embedding before the first, the head after the last); the state
        between stretches lives on the decoder."""

        m, c, w = self.m, self.m.cfg, self.m.w
        n = self.R
        dev = "cuda"
        view = self.table
        first, end = self.stretches[k]
        self._idx = {}                                          # this stretch's index tensors (_ix)
        if first == 0:
            if K.on("glue") and w.embed.dtype == BF16 and c.dim % 1024 == 0:
                h, pre = K.embed_init(w.embed, self.ids, c.hc)          # the streams and pre-mix in one launch
            else:
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
        else:
            h, pre, x, part, pre_a, pre_f, post, comb, shared, taps = self._state
        if K.on("hc"):
            h, pre, pre_f = self._layers_fused(w.layers[first:end], h, pre, x, part, pre_a, pre_f, post, comb, shared,
                                               taps)
        else:
            for lay in w.layers[first:end]:
                if lay.engram_wkv is not None and lay.idx in self.e_in:
                    kv = m.comm.sum(mm(lay.engram_wkv, self.e_in[lay.idx], F32)).to(BF16)
                    h = K.engram_gate(h, kv.contiguous(), lay.engram_qk, c.eps)
                if self.want_taps and lay.idx in c.dspark_taps:
                    taps.append(h.to(F32).mean(1).to(BF16))
                cos, sin = m._cs(lay.idx, view)
                fn, scale, base = lay.hc_attn
                K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb,
                         part)
                K.hc_post(m.comm.gather(self._attention(lay, x, shared, cos, sin)), h, post, comb, h)
                fn, scale, base = lay.hc_ffn
                K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb,
                         part)
                K.hc_post(m.comm.gather(m.moe(lay, x)), h, post, comb, h)
                pre, pre_f = pre_f, pre
        self._idx = {}
        if end < len(w.layers):
            self._state = (h, pre, x, part, pre_a, pre_f, post, comb, shared, taps)
            return
        self._state = None
        xc = K.collapse_norm(h, pre.contiguous(), w.norm, c.eps)
        local = mm(w.head, xc, F32)
        g = m.comm.gather(local)
        self.logits = g.permute(1, 0, 2).reshape(n, -1)
        if len(taps) == 2 and isinstance(taps[1], int):     # the taps buffer (every tap layer written)
            assert taps[1] * c.dim == taps[0].shape[1]
            self.taps = taps[0]
        else:
            self.taps = torch.cat(taps, -1) if taps else None

    def _layers_fused(self, layers, h, pre, x, part, pre_a, pre_f, post, comb, shared, taps):
        """The layer loop with each sublayer's hc_post fused into the next hc_pre (kernels.hc_pre2: the post written
        into a second stream buffer, the mixes taken of it in the same programs). A post the next sublayer cannot take
        (an Engram layer's gate reads the streams first, the stretch's end) runs alone (hc_post in place). The same
        bits; a tap is taken of the same streams (after the fused kernel wrote them)."""

        m, c = self.m, self.m.cfg
        view = self.table
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

        for lay in layers:
            if lay.engram_wkv is not None and lay.idx in self.e_in:
                if pending is not None:
                    K.hc_post(pending, h, post, comb, h)
                    pending = None
                kv = m.comm.sum(mm(lay.engram_wkv, self.e_in[lay.idx], F32)).to(BF16)
                h = K.engram_gate(h, kv.contiguous(), lay.engram_qk, c.eps)
            cos, sin = m._cs(lay.idx, view)
            h = pre_mix(h, lay.hc_attn, pre, lay.attn_norm, pre_a)
            if self.want_taps and lay.idx in c.dspark_taps:
                if K.on("glue") and c.dim % 1024 == 0:
                    # one launch a tap, straight into its block of the taps buffer (made at the first tap)
                    if not taps:
                        ntap = sum(1 for i in c.dspark_taps if i < len(m.w.layers))
                        taps.append(torch.empty((h.shape[0], ntap * c.dim), dtype=BF16, device=h.device))
                        taps.append(0)
                    j = taps[1]
                    K.tap(h, taps[0][:, j * c.dim:(j + 1) * c.dim])
                    taps[1] = j + 1
                else:
                    taps.append(h.to(F32).mean(1).to(BF16))
            pending = m.comm.gather(self._attention(lay, x, shared, cos, sin))
            h = pre_mix(h, lay.hc_ffn, pre_a, lay.ffn_norm, pre_f)
            pending = m.comm.gather(m.moe(lay, x))
            pre, pre_f = pre_f, pre
        if pending is not None:
            K.hc_post(pending, h, post, comb, h)
        return h, pre, pre_f

    def capture(self, pool=None) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graphs = []
        for k in range(len(self.stretches)):                  # in order, one pool: a stretch's state feeds the next
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._stretch(k)
            graphs.append(g)
        torch.cuda.synchronize()
        self.graphs = graphs
        self.graph = graphs[0]

    def set(self, ids: list[int], pos: list[int], slots: list[int], base: list[int], end: list[int]) -> None:
        self.ids.copy_(torch.tensor(ids, dtype=torch.long), non_blocking=False)
        self.pos.copy_(torch.tensor(pos, dtype=torch.long), non_blocking=False)
        self.slot.copy_(torch.tensor(slots, dtype=torch.long), non_blocking=False)
        self.base.copy_(torch.tensor(base, dtype=torch.long), non_blocking=False)
        self.end.copy_(torch.tensor(end, dtype=torch.long), non_blocking=False)

    def run(self, ids: list[int], pos: list[int], slots: list[int], base: list[int], end: list[int],
            e_rows) -> torch.Tensor:
        """``e_rows``: layer -> its Engram rows on the GPU, or a callable giving them (called just before the stretch
        that starts at that layer, so the read overlaps the stretches before it)."""

        self.set(ids, pos, slots, base, end)
        for k, (first, _) in enumerate(self.stretches):
            for i in self.e_in:
                if e_rows and (i == first or (k == 0 and i < self.stretches[0][1])):
                    t = e_rows[i]
                    self.e_in[i].copy_(t() if callable(t) else t)
            if self.graphs is None:
                self._stretch(k)
            else:
                self.graphs[k].replay()
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

    def forward(self, windows: list[tuple], replay: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``windows``: each stream's (slot, extent base, extent size, first position, token ids, host ids so far):
        rows in that order. Returns logits [R, V] and taps [R, 3 d] (the rows in window order); ``replay=False``
        only captures a missing graph (warm-up) and returns (None, None)."""

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
        if m.engram is not None and replay:
            import numpy as np

            hs = np.concatenate(hashes, 0)
            lo, hi = m.engram.cols
            idx = {i: hs[:, m.cfg.engram_layers.index(i), lo:hi] for i in m.cfg.engram_layers if i < len(m.w.layers)}
            if ENGRAM_SPLIT:
                for i, ix in idx.items():                     # every layer's read starts now, in layer order
                    m.engram.prefetch(i, ix, lane=2)
                e_rows = {i: (lambda i=i, ix=ix: m.engram.rows(i, ix)) for i, ix in idx.items()}
            else:
                e_rows = {i: m.engram.rows(i, ix) for i, ix in idx.items()}
        key = (R, b)
        g = self.graphs.get(key) if self.graphs is not None else None
        if g is None:
            g = RoundDecoder(m, self.pool, R, b, True)
            g.set(ids, pos, slots, base, end)
            if self.graphs is not None:
                if self.graph_pool is None:
                    self.graph_pool = torch.cuda.graph_pool_handle()
                g.capture(self.graph_pool)
                self.graphs[key] = g
                self.captures += 1
        if not replay:
            return None, None
        out = g.run(ids, pos, slots, base, end, e_rows)
        return out, g.taps
