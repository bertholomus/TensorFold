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
import torch.nn.functional as F

from ..ops import BF16, F32, fp4_qd
from . import kernels as K
from . import model as _model
from .graph import BUCKET_MIN, bucket_for
from .model import (KV_QUANT, RAW, Model, PoolCache, SeqCache, _candidates, apply_candidates, attn_in, attn_in_rot,
                    l2_fork, l2_join, mm, moe_side, q_proj, store_rows, wo_a_out, wo_a_rot, wo_ab)

MAX_ROWS = K.DECODE_ROWS
# TF_DS_ENGRAM_SPLIT=1 (default): a round is one graph a stretch of layers, cut before each Engram layer, so a layer's
# Engram rows are read while the layers before it run (the arithmetic is the one-graph round's)
ENGRAM_SPLIT = os.environ.get("TF_DS_ENGRAM_SPLIT", "1") == "1"
# TF_DS_ENGRAM_TOUCH=1 (default): a round reads its streams' first rows' Engram rows before the drafts
# (RoundRunner.engram_touch: the drive awake when the round's read starts; no bit changes)
ENGRAM_TOUCH = os.environ.get("TF_DS_ENGRAM_TOUCH", "1") == "1"
# TF_DS_ONE_COPY=1 (default): a round's row inputs (ids, positions, slots, extents) go to the GPU as one pinned
# non-blocking copy, not five synchronous ones (each waited for the stream: GPU idle between the drafts and the forward);
# also the drafter's inputs (dspark.py; the absorb's indices stay synchronous: pinned was slower there). 0: as before
ONE_COPY = os.environ.get("TF_DS_ONE_COPY", "1") == "1"
# TF_DS_ROUND_SHARED_OUT=1 (default): a RoundRunner's graphs write their logits and taps into one pair of buffers the
# runner makes before any capture ([16, vocab] fp32, [16, taps] bf16), not a pair a graph in the graph pool (64 graphs
# at ready held ~284 MiB a rank; long contexts capture more). The same copies of the same values: a round's logits and
# taps are read (sampled, absorbed) before the runner's next round, so one pair serves every graph. forward() returns
# views of the pair, valid until the runner's next forward (clone to keep them). 0: a pair a graph, as before
SHARED_OUT = os.environ.get("TF_DS_ROUND_SHARED_OUT", "1") == "1"


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


def _candidate_blocks(score: torch.Tensor, vis: torch.Tensor, nblocks: int, bsize: int) -> torch.Tensor:
    """model._candidates' pool as block ids [rows, nblocks] int32 (-1: no block), for index_keys_cand: the blocks
    _candidates' mask sets (the same top-k of the block maxima, the newest block pinned in, finite maxima only)."""

    width = score.shape[-1]
    s = F.pad(score, (0, -width % bsize), value=float("-inf")).unflatten(-1, (-1, bsize)).amax(-1)
    nb = s.shape[-1]
    last = (vis - 1) // bsize
    s = s.masked_fill(torch.arange(nb, device=score.device)[None] == last[:, None], float("inf"))
    idx = K.topk_indices(s, min(nblocks, nb))
    return torch.where(s.gather(-1, idx) > float("-inf"), idx, -1).to(torch.int32)


class RoundDecoder:
    """Graph-captured forward of R rows from several streams over a pool; inputs in device buffers: token ids,
    positions, slots, Engram rows."""

    def __init__(self, model: Model, pool: PoolCache, rows: int, bucket: int, taps: bool, out: tuple | None = None):
        if rows > MAX_ROWS:
            raise ValueError(f"a concurrent round holds {MAX_ROWS} rows at most, not {rows}")
        self.m, self.pool, self.R, self.bucket, self.want_taps = model, pool, rows, bucket, taps
        self.out = out                                  # (logits, taps) buffers the runner's graphs share (SHARED_OUT)
        c = model.cfg
        dev = "cuda"
        self.inputs = torch.zeros((5, rows), dtype=torch.long, device=dev)  # one buffer, so one copy fills it
        # ids, positions, slots, and the row's stream's extent (base, end) in positions
        self.ids, self.pos, self.slot, self.base, self.end = self.inputs.unbind(0)
        self.end.fill_(1)
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

    def _attention(self, lay, x, shared, cos, sin, xh=None):
        if xh is not None or any(K.on(k) for k in ("glue", "rot_q", "rot_wob", "rot_attn", "idx", "comp")):
            return self._attention2(lay, x, shared, cos, sin, xh=xh)
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
        l2_fork(m, lay, "attn")                                         # wo_a (and wo_b) into L2 after wq_b
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
        out = mm(lay.wo_b, wo_a_out(lay, o), F32)
        l2_fork(m, lay, "moe")                                          # the MoE's mHC mix, gate, shared expert
        return out

    def _attention2(self, lay, x, shared, cos, sin, xh=None):
        """_attention with the small-kernel switches (kernels.SMALL_SWITCHES): "glue" takes the round's index tensors
        from _ix; "rot_q" folds wq_b's input rotation into the q RMSNorm and q's RoPE into wq_b's epilogue, "rot_attn"
        the inverse RoPE and wo_a's input rotation into the attention merge, "rot_wob" wo_b's input rotation into
        wo_a's epilogue; "idx" / "comp" fuse the indexer's and the compressor's glue. The same arithmetic in the same
        order, so the same bits. ``xh``: attn_in's rotated input rows, written by the mHC finish (switch "hc_rot")."""

        m, c, pool = self.m, self.m.cfg, self.pool
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        ratio = lay.ratio
        pos, slot = self.pos, self.slot
        RS = pool.ring_size
        glue, rot = K.on("glue"), K.on("rot_q")
        ix = self._ix if glue else self._ix_now
        qa, ykv, ckv, cgate = attn_in(lay, x, comp=bool(ratio), xh=xh)  # wq_a, wkv, the compressor's: one launch
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
        l2_fork(m, lay, "attn")                                         # wo_a (and wo_b) into L2 after wq_b
        ring, wbase, comp, cidx, cbase = self._kv_idx(lay, x, ykv, ckv, cgate, shared, cos, sin, ix, iq, qr, idx_all,
                                                      kv_done=rot)
        if K.on("rot_attn"):
            # the merge, the inverse RoPE and wo_a's input rotation in one launch
            suh, xh, gh = wo_a_rot(lay, n, x.device, hd)
            K.sparse_attn(q, lay.sink, ring, m._zero, True, comp, cidx, pos, hd ** -0.5, c.window,
                          wbase=wbase, cbase=cbase, ring_rows=RS, rot=(cos, sin, rd, suh, xh[0], gh))
            out = wo_ab(lay, xh=xh, fold=K.on("rot_wob"))
        else:
            o = K.sparse_attn(q, lay.sink, ring, m._zero, True, comp, cidx, pos, hd ** -0.5, c.window,
                              wbase=wbase, cbase=cbase, ring_rows=RS)
            K.rope_heads(o, cos, sin, pos, rd, inverse=True)
            if K.on("rot_wob"):
                out = wo_ab(lay, o)                          # wo_a's epilogue rotates for wo_b
            else:
                out = mm(lay.wo_b, wo_a_out(lay, o), F32)
        l2_fork(m, lay, "moe")                                          # the MoE's mHC mix, gate, shared expert
        return out

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
                if fused and 0 <= c.cand_source < lay.idx and kk & (kk - 1) == 0 and "cblk" in shared:
                    # past the pool's width: score the pool's blocks only (the same keys there, -inf keys elsewhere
                    # left out: the same top-k)
                    keys = K.index_keys_cand(iq, pool.index_k[src], wts, vis, nb, shared["cblk"], c.cand_block,
                                             base=cbase)
                    shared["topk"] = K.topk_select(keys, kk, vis)
                elif fused and lay.idx != c.cand_source and kk & (kk - 1) == 0:
                    # scores -> (candidate mask) -> top-k keys in one launch, the top-k's indices sorted and masked
                    # in one more (the cand-source layer keeps the scores for _candidates)
                    cand = shared["cand"] if 0 <= c.cand_source < lay.idx else None
                    if nb > K.TOPK_FUSED_MAX and K.on("topk_prune"):
                        # long buckets: each 64-key tile's maximum too; the top-k from the k best tiles (exact)
                        keys, tm = K.index_keys(iq, pool.index_k[src], wts, vis, nb, base=cbase, cand=cand,
                                                cand_block=c.cand_block, tmax=True, pruned_k=kk)
                        shared["topk"] = K.topk_select_pruned(keys, tm, kk, vis)
                    else:
                        keys = K.index_keys(iq, pool.index_k[src], wts, vis, nb, base=cbase, cand=cand,
                                            cand_block=c.cand_block)
                        shared["topk"] = K.topk_select(keys, kk, vis)
                elif fused and lay.idx == c.cand_source and kk & (kk - 1) == 0:
                    # the cand-source layer: the scores for the candidate pool, then their keys' top-k as above
                    score = K.index_score(iq, pool.index_k[src], wts, vis, nb, base=cbase)
                    if K.on("cand_only") and nb > c.cand_blocks * c.cand_block:
                        shared["cblk"] = _candidate_blocks(score, vis, c.cand_blocks, c.cand_block)
                    else:
                        shared["cand"] = _candidates_fast(score, vis, c.cand_blocks, c.cand_block)
                    if nb > K.TOPK_FUSED_MAX and K.on("topk_prune"):
                        keys, tm = K.score_keys(score, tmax=True)
                        shared["topk"] = K.topk_select_pruned(keys, tm, kk, vis)
                    else:
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
            h, pre, pre_f = self._layers_fused(first, end, h, pre, x, part, pre_a, pre_f, post, comb, shared, taps)
        else:
            for li in range(first, end):
                lay = w.layers[li]
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
                y = m.moe(lay, x, shared_side=moe_side())           # the gate is in L2 by then
                self._l2_next(li, end)
                K.hc_post(m.comm.gather(y), h, post, comb, h)
                pre, pre_f = pre_f, pre
        l2_join(m)                                              # a captured stretch ends with its prefetches done
        self._idx = {}
        if end < len(w.layers):
            self._state = (h, pre, x, part, pre_a, pre_f, post, comb, shared, taps)
            return
        self._state = None
        xc = K.collapse_norm(h, pre.contiguous(), w.norm, c.eps)
        local = mm(w.head, xc, F32)
        g = m.comm.gather(local)
        out = self.out
        if out is not None and out[0].shape[1] == g.shape[0] * g.shape[2]:
            lg = out[0][:n]
            lg.view(n, g.shape[0], g.shape[2]).copy_(g.permute(1, 0, 2))   # reshape's copy, into the shared rows
            self.logits = lg
        else:
            self.logits = g.permute(1, 0, 2).reshape(n, -1)
        if len(taps) == 2 and isinstance(taps[1], int):     # the taps buffer (every tap layer written)
            assert taps[1] * c.dim == taps[0].shape[1]
            self.taps = taps[0]
        elif taps and out is not None and out[1].shape[1] == sum(t.shape[1] for t in taps):
            self.taps = torch.cat(taps, -1, out=out[1][:n])
        else:
            self.taps = torch.cat(taps, -1) if taps else None

    def _l2_next(self, li: int, end: int) -> None:
        """After layer li's experts: the next layer's first weights into L2 (the head's after the last layer); a
        stretch's graph waits for its last fork at its end, so that one is smaller."""

        w = self.m.w
        l2_fork(self.m, w.layers[li], "next", w.layers[li + 1] if li + 1 < len(w.layers) else w.head,
                mb=_model.L2_STRETCH_END_MB if li + 1 == end < len(w.layers) else None)

    def _layers_fused(self, first, end, h, pre, x, part, pre_a, pre_f, post, comb, shared, taps):
        """Layers [first, end) with each sublayer's hc_post fused into the next hc_pre (kernels.hc_pre2: the post
        written into a second stream buffer, the mixes taken of it in the same programs). A post the next sublayer
        cannot take (an Engram layer's gate reads the streams first, the stretch's end) runs alone (hc_post in place).
        The same bits; a tap is taken of the same streams (after the fused kernel wrote them)."""

        m, c = self.m, self.m.cfg
        view = self.table
        spare = torch.empty_like(h)
        pending = None                                   # the last sublayer's gathered partials, not yet posted
        # (switch "hc_defer") each finish's Sinkhorn half on this side stream, joined before the sublayer's gather
        # (the next posted mixes read post / comb, the next finish pre; the next mixes write part)
        sink = None
        if K.hc_defer_on(c.dim):
            sink = m.__dict__.get("_tf_hc_sink")
            if sink is None:
                sink = m._tf_hc_sink = torch.cuda.Stream()

        def join():
            if sink is not None:
                torch.cuda.current_stream().wait_stream(sink)

        def pre_mix(h, params, pre_in, norm, pre_out, rot=None):
            nonlocal spare, pending
            fn, scale, base = params
            if pending is None:
                K.hc_pre2(h, fn, scale, base, pre_in, norm, c.eps, c.hc_eps, c.hc_iters, x, pre_out, post, comb, part,
                          rot=rot, sink=sink)
                return h
            out = K.hc_pre2(h, fn, scale, base, pre_in, norm, c.eps, c.hc_eps, c.hc_iters, x, pre_out, post, comb,
                            part, gathered=pending, h_out=spare, rot=rot, sink=sink)
            pending = None
            spare = h
            return out

        for li in range(first, end):
            lay = m.w.layers[li]
            if lay.engram_wkv is not None and lay.idx in self.e_in:
                if pending is not None:
                    K.hc_post(pending, h, post, comb, h)
                    pending = None
                kv = m.comm.sum(mm(lay.engram_wkv, self.e_in[lay.idx], F32)).to(BF16)
                h = K.engram_gate(h, kv.contiguous(), lay.engram_qk, c.eps)
            cos, sin = m._cs(lay.idx, view)
            # (switch "hc_rot") the finish also writes attn_in's rotated input rows
            ar = attn_in_rot(lay, h.shape[0], h.device, comp=bool(lay.ratio)) if K.hc_rot_on(c.dim) else None
            h = pre_mix(h, lay.hc_attn, pre, lay.attn_norm, pre_a, rot=ar[1] if ar else None)
            if self.want_taps and lay.idx in c.dspark_taps:
                if K.on("glue") and c.dim % 1024 == 0:
                    # one launch a tap, straight into its block of the taps buffer (made at the first tap)
                    if not taps:
                        ntap = sum(1 for i in c.dspark_taps if i < len(m.w.layers))
                        out = self.out
                        if out is not None and out[1].shape[1] == ntap * c.dim:
                            taps.append(out[1][:h.shape[0]])            # the runner's shared rows (SHARED_OUT)
                        else:
                            taps.append(torch.empty((h.shape[0], ntap * c.dim), dtype=BF16, device=h.device))
                        taps.append(0)
                    j = taps[1]
                    K.tap(h, taps[0][:, j * c.dim:(j + 1) * c.dim])
                    taps[1] = j + 1
                else:
                    taps.append(h.to(F32).mean(1).to(BF16))
            y = self._attention(lay, x, shared, cos, sin, xh=ar[0] if ar else None)
            join()
            pending = m.comm.gather(y)
            h = pre_mix(h, lay.hc_ffn, pre_a, lay.ffn_norm, pre_f)
            y = m.moe(lay, x, shared_side=moe_side())               # the gate is in L2 by then
            self._l2_next(li, end)
            join()
            pending = m.comm.gather(y)
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
        if ONE_COPY:                                    # pinned: the copy runs in stream order, nothing waits for it
            self.inputs.copy_(torch.tensor([ids, list(pos), slots, base, end], dtype=torch.long).pin_memory(),
                              non_blocking=True)
            return
        self.ids.copy_(torch.tensor(ids, dtype=torch.long), non_blocking=False)
        self.pos.copy_(torch.tensor(pos, dtype=torch.long), non_blocking=False)
        self.slot.copy_(torch.tensor(slots, dtype=torch.long), non_blocking=False)
        self.base.copy_(torch.tensor(base, dtype=torch.long), non_blocking=False)
        self.end.copy_(torch.tensor(end, dtype=torch.long), non_blocking=False)

    def run(self, ids: list[int], pos: list[int], slots: list[int], base: list[int], end: list[int],
            e_rows, e_start: dict | None = None) -> torch.Tensor:
        """``e_rows``: layer -> its Engram rows on the GPU, or a callable giving them (called just before the stretch
        that starts at that layer, so the read overlaps the stretches before it). ``e_start``: layer -> a callable that
        starts that layer's read, called once the stretch before the one that needs the rows is enqueued (before the
        first stretch for rows it needs)."""

        def start(k: int) -> None:                       # the reads the stretch after stretch k needs
            if e_start:
                nxt = self.stretches[k + 1][0] if k + 1 < len(self.stretches) else None
                for i, fn in e_start.items():
                    if (k < 0 and i < self.stretches[0][1]) or (k >= 0 and i == nxt):
                        fn()

        from .multi import _ht

        _ht("fwd prep")
        self.set(ids, pos, slots, base, end)
        _ht("fwd set")
        start(-1)
        for k, (first, _) in enumerate(self.stretches):
            for i in self.e_in:
                if e_rows and (i == first or (k == 0 and i < self.stretches[0][1])):
                    t = e_rows[i]
                    self.e_in[i].copy_(t() if callable(t) else t)
                    _ht(f"fwd rows {i} in")
            if self.graphs is None:
                self._stretch(k)
            else:
                self.graphs[k].replay()
            _ht(f"fwd stretch {k} launched")
            start(k)
            _ht(f"fwd reads after {k} submitted")
        return self.logits


class RoundRunner:
    """Round graphs per (rows, bucket) over one pool, captured on first use (or ahead, ``warm``)."""

    def __init__(self, model: Model, pool: PoolCache, graphs: bool = True, graph_pool=None):
        self.m, self.pool = model, pool
        self.graphs: dict = {} if graphs else None
        self.graph_pool = graph_pool
        self.captures = 0
        self.out = None
        if SHARED_OUT:                                  # outside every capture: each graph writes these same rows
            c, w = model.cfg, model.w
            ntap = sum(1 for i in c.dspark_taps if i < len(w.layers))
            self.out = (torch.empty((MAX_ROWS, w.head.n * model.comm.world), dtype=F32, device="cuda"),
                        torch.empty((MAX_ROWS, max(ntap, 1) * c.dim), dtype=BF16, device="cuda"))

    def bucket(self, deepest: int) -> int:
        return bucket_for(deepest, self.pool.cap)

    def engram_touch(self, tails: list[list[int]]) -> None:
        """``tails``: each stream's host ids ending with its pending token, the n-gram's tokens before it included (or
        from the sequence's start): that row's Engram rows read now on the decode lane (Engram.touch), ahead of the
        round's own read (ENGRAM_TOUCH). (A row's hash depends only on its n-gram and whether it reaches before
        position 0, so the tail's last row hashes as the whole sequence's.)"""

        m = self.m
        if not ENGRAM_TOUCH or m.engram is None or not tails:
            return
        import numpy as np

        from .multi import _ht

        _ht("touch start")
        hs = np.concatenate([m.engram.hashes(t, len(t) - 1, 1) for t in tails], 0)
        _ht("touch hashed")
        lo, hi = m.engram.cols
        for i in m.cfg.engram_layers:
            if i < len(m.w.layers):
                m.engram.touch(i, hs[:, m.cfg.engram_layers.index(i), lo:hi])
                _ht(f"touch {i} submitted")

    def forward(self, windows: list[tuple], replay: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``windows``: each stream's (slot, extent base, extent size, first position, token ids, host ids so far):
        rows in that order. Returns logits [R, V] and taps [R, 3 d] (the rows in window order; with SHARED_OUT views of
        the runner's pair, valid until its next forward); ``replay=False``
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
        e_rows = e_start = None
        if m.engram is not None and replay:
            import numpy as np

            hs = np.concatenate(hashes, 0)
            lo, hi = m.engram.cols
            idx = {i: hs[:, m.cfg.engram_layers.index(i), lo:hi] for i in m.cfg.engram_layers if i < len(m.w.layers)}
            if ENGRAM_SPLIT:
                if m.engram.decode_aio():
                    # AIO reads: their submission takes the calling thread ~3 us a row read (O_DIRECT block mapping),
                    # so each layer's is submitted once the stretch before the one that needs it is enqueued
                    e_start = {i: (lambda i=i, ix=ix: m.engram.prefetch(i, ix, lane=2)) for i, ix in idx.items()}
                else:
                    for i, ix in idx.items():                 # every layer's read starts now, in layer order
                        m.engram.prefetch(i, ix, lane=2)
                e_rows = {i: (lambda i=i, ix=ix: m.engram.rows(i, ix)) for i, ix in idx.items()}
            else:
                e_rows = {i: m.engram.rows(i, ix) for i, ix in idx.items()}
        key = (R, b)
        g = self.graphs.get(key) if self.graphs is not None else None
        if g is None:
            g = RoundDecoder(m, self.pool, R, b, True, out=self.out)
            g.set(ids, pos, slots, base, end)
            if self.graphs is not None:
                if self.graph_pool is None:
                    self.graph_pool = torch.cuda.graph_pool_handle()
                g.capture(self.graph_pool)
                self.graphs[key] = g
                self.captures += 1
        if not replay:
            return None, None
        out = g.run(ids, pos, slots, base, end, e_rows, e_start)
        return out, g.taps
