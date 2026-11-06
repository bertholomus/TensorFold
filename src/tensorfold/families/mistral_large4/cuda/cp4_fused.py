"""cp4.Ml4Cp's round on the GPU with its glue fused, as tp_cuda is to tp.py: the same sums, about a third of the launches, no
sort, and the residual added where the next norm reads it. ``Ml4Cp`` stays the reference; numbers differ from it only by
the order of reductions. Needs experts split over the ranks and every stream striped."""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F

from tensorfold.cuda.comm import little_cores

from . import cp4
from .cp4_kernels import SCORE4_ROW, SCORE8, index_scores, score4, score4_ready, score4_rows, score8, score8_query, sparse_mla
from .cp4_cuda import SCORE_BYTES
from .model import route_softmax
from .tp_kernels import (add_norm, key_row, keep_rows, kth_rows, kv_row, layer_norm16, local_top, merge_rows, q_rope, q_scale, qi_prep,
                         select_ready, top8)

F16, F32 = torch.float16, torch.float32
SLICE_ROWS = int(os.environ.get("TF_ML4_CP_SLICE_ROWS") or 1024)          # a round's queries a slice: its exchange runs beside the next's kernels; 0: none
ONE = cp4.Stripe(1, 1)                                                    # a prompt's copy: a row number is a position
PAIR = int(os.environ.get("TF_ML4_CP_PAIR") or 0)                         # a longer round goes as two, this many rows and the rest; 0: never
SLOTS = os.environ.get("TF_ML4_CP_SLOTS", "1") == "1"                     # a deep prompt's rounds run over its rows gathered a layer ahead
FOUR = os.environ.get("TF_ML4_CP_SCORE4", "1") == "1"                     # and pick them in FP4
TF32 = os.environ.get("TF_ML4_CP_TF32", "1") == "1"                       # and take their rows' small matrices in TF32
PACE_LITTLE = os.environ.get("TF_ML4_CP_PACE_LITTLE", "1") == "1"         # the wait for the GPU spins on a little core
# layer:earlier: on a deep prompt these layers take that layer's selection, as the model's own layers do from 17 on: the first layers'
# hardly matter there, and scores are most of what heats a node
REUSE = dict(tuple(int(v) for v in x.split(":")) for x in os.environ.get("TF_ML4_CP_REUSE", "2:1,5:4,6:4,8:4,9:4,10:4,12:4,13:4,14:4,7:3,11:3,15:3").split(",") if x)


def mark(stream) -> torch.cuda.Event:
    e = torch.cuda.Event()
    e.record(stream)
    return e


@dataclass
class Part:
    """A slice of a round's queries: a share of every rank's rows, so an exchange of it is an even all-to-all."""

    lo: int                           # this rank's rows of it
    hi: int
    counts: list                      # rows per rank
    qpos: torch.Tensor                # [Q] positions, rank-major
    vis: torch.Tensor                 # [Q] rows of this rank each query sees
    vis_max: int
    at: torch.Tensor | None = None    # a round's own part: where all its rows go in the prompt's copy, rank-major


@dataclass
class Pair:
    """Two rounds of a prompt half a layer apart: each one's exchanges run beside the other's kernels, and the second's queries
    see the first's rows of a layer because the first is ahead."""

    a: cp4.Flight
    b: cp4.Flight
    at: int = 0                       # the next layer
    old: int = 0                      # rows a rank of earlier rounds

    @property
    def rnd(self) -> cp4.Round:
        return self.a.rnd


class Copy:
    """A prompt's rows on every rank for the length of the prompt, in position order: an extent of the pool."""

    def __init__(self, pool: cp4.KvPool, base: int, rows: int) -> None:
        self.pool, self.base, self.rows = pool, base, rows

    def keys(self, L: int) -> torch.Tensor:
        return self.pool.keys[L][self.base:self.base + self.rows]

    def lat(self, L: int) -> torch.Tensor:
        return self.pool.lat[L][self.base:self.base + self.rows]

    def wait(self, L: int) -> None:
        pass


class Slots:
    """A prompt too long for a copy: all its rows of one layer at a time, the next layer's gathered behind the experts of this one
    on a communicator of their own (a reader layer's are its source's, again). The rows of earlier rounds lie rank-major, then the
    flight's own in position order: a query sees a prefix either way, and with one row per earlier position a row of the flight is
    numbered by its position."""

    def __init__(self, pool: cp4.KvPool, comm, rows: int) -> None:
        c, dev = pool.cfg, pool.device
        self.pool, self.comm, self.rows, self.stream = pool, comm, rows, torch.cuda.Stream()
        self.key = torch.empty((rows, SCORE4_ROW), dtype=torch.uint8, device=dev) if FOUR else torch.empty((rows, c.idx_dim), dtype=F16, device=dev)
        self.mine = torch.empty((pool.rows, SCORE4_ROW), dtype=torch.uint8, device=dev) if FOUR else None    # this rank's, on their way
        self.raw = torch.empty(rows * pool.fmt_rope.row_bytes, dtype=torch.uint8, device=dev)
        self.got: torch.cuda.Event | None = None     # when the rows are in
        self.free: torch.cuda.Event | None = None    # when the last layer's readers are done
        self.busy = False

    def keys(self, L: int) -> torch.Tensor:
        return self.key

    def lat(self, L: int) -> torch.Tensor:
        return self.raw[:self.rows * self.pool.fmt(L).row_bytes].view(self.rows, -1)

    def wait(self, L: int) -> None:
        if self.got is not None:
            torch.cuda.current_stream().wait_event(self.got)

    def fetch(self, L: int, lay, seq: cp4.Seq, old: int) -> None:
        """Layer ``L``'s ``old`` rows a rank, once the last layer's readers are done."""

        pool, comm, at = self.pool, self.comm, slice(seq.base, seq.base + old)
        self.got = None
        if not old:
            return
        with torch.cuda.stream(self.stream):
            if self.free is not None:
                self.stream.wait_event(self.free)
            comm.all_gather(pool.lat[lay.kv_source][at].view(-1), self.lat(lay.kv_source)[:comm.world * old].view(-1))
            if lay.indexer == "full" and L not in REUSE:
                keys = pool.keys[L][at]
                comm.all_gather((keys if self.mine is None else score4_rows(keys, out=self.mine[:old])).view(-1), self.key[:comm.world * old].view(-1))
            self.got = mark(self.stream)

    def done(self) -> None:
        self.free = mark(torch.cuda.current_stream())


def forward(w, comm, pool: cp4.KvPool, ops: cp4.Ops) -> cp4.Ml4Cp:
    """The lane's rounds: fused when the experts are split, the reference else (``TF_ML4_FUSED_ROUND=0`` for an A/B)."""

    fused = w.split and os.environ.get("TF_ML4_FUSED_ROUND", "1") != "0"
    return (FusedCp if fused else cp4.Ml4Cp)(w, comm, pool, ops)


class FusedCp(cp4.Ml4Cp):
    def __init__(self, w, comm, pool: cp4.KvPool, ops: cp4.Ops | None = None) -> None:
        super().__init__(w, comm, pool, ops)
        select_ready()
        self.side = torch.cuda.Stream(priority=-1) if SLICE_ROWS else None
        self.slots: Slots | None = None
        self.lag: torch.cuda.Event | None = None
        self.scores = torch.empty(SCORE_BYTES // 32 * 9, dtype=F32, device=pool.device) if SCORE8 else None   # a tile's scores and largests

    def open_slots(self, rows: int) -> None:
        """Room for a prompt of ``rows`` tokens to run over all its rows without a copy of them; every rank calls this at the same point."""

        if SLOTS and self.side is not None and self.world > 1:
            kinds = self.cfg.indexer_types
            assert all(src < L < len(kinds) and src not in REUSE and kinds[L] == kinds[src] == "full" for L, src in REUSE.items()), REUSE
            if FOUR:
                score4_ready()
            self.slots = Slots(self.pool, self.comm.fork("tf_nccl_uid_rows"), rows)

    def _copy(self, seq: cp4.Seq, start: int, n: int) -> Copy | Slots | None:
        if seq.rep is not None:
            return Copy(self.pool, seq.rep, seq.rep_rows)
        st, free = seq.stripe, self.slots is not None and not self.slots.busy
        return self.slots if free and n >= 2 * SLICE_ROWS and start % (st.world * st.block) == 0 else None

    def begin(self, seq: cp4.Seq, start: int, n: int, tokens: torch.Tensor, last: int | None = None) -> cp4.Flight | Pair:
        copy = self._copy(seq, start, n)
        if PAIR and n > PAIR and self.side is not None:
            p = torch.arange(start, start + PAIR)
            cut = int((seq.stripe.rank(p) == self.rank).sum())
            f = Pair(self._one(seq, start, PAIR, tokens[:cut], None, copy), self._one(seq, start + PAIR, n - PAIR, tokens[cut:], last, copy))
        else:
            f = self._one(seq, start, n, tokens, last, copy)
        if copy is self.slots and copy is not None:
            copy.busy = True
            f.old = start // seq.stripe.world
            copy.fetch(0, self.w.layers[0], seq, f.old)
        return f

    def _one(self, seq: cp4.Seq, start: int, n: int, tokens: torch.Tensor, last: int | None, copy) -> cp4.Flight:
        f = super().begin(seq, start, n, tokens, last)
        if copy is not None:
            p = torch.arange(start, start + n)
            own = p[seq.stripe.rank(p) == self.rank]
            f.rnd.copy = copy
            f.rnd.mine = Part(0, own.numel(), f.rnd.counts, f.rnd.own, (own + 1).to(torch.int32).to(tokens.device),
                              int(own[-1]) + 1 if own.numel() else 0, f.rnd.qpos)
        elif self.side is not None and n >= 2 * SLICE_ROWS and seq.stripe.world > 1:
            f.rnd.parts = self._parts(seq, start, n, n // SLICE_ROWS, tokens.device)
        return f

    def _parts(self, seq: cp4.Seq, start: int, n: int, count: int, dev) -> list:
        st, out = seq.stripe, []
        p = torch.arange(start, start + n)
        who = st.rank(p)
        per = [p[who == k] for k in range(st.world)]
        for i in range(count):
            cuts = [(t.numel() * i // count, t.numel() * (i + 1) // count) for t in per]
            qpos = torch.cat([t[a:b] for t, (a, b) in zip(per, cuts)])
            vis = st.vis(self.rank, qpos)
            out.append(Part(*cuts[self.rank], [b - a for a, b in cuts], qpos.to(dev), vis.to(torch.int32).to(dev),
                            int(vis.max()) if vis.numel() else 0))
        return out

    def _split(self, t: torch.Tensor, counts: list) -> list:
        at, out = 0, []
        for c in counts:
            out.append(t[at:at + c])
            at += c
        return out

    def _top(self, L: int, lay, qi: torch.Tensor, wts: torch.Tensor, rnd: cp4.Round, part) -> tuple:
        """This rank's best k of every query of ``part`` (a Part or the Round): keys [Q, k], rows [Q, k]."""

        c, st, seq, rank = self.cfg, rnd.seq.stripe, rnd.seq, self.rank
        Q, k, dev = part.qpos.numel(), c.idx_topk, qi.device
        q = torch.empty((Q, c.idx_heads * c.idx_dim), dtype=F16, device=dev)
        qi_prep(qi, part.qpos, lay.idx_ssmax_s, lay.idx_ssmax_b, self.inv_freq, c.idx_heads, c.idx_dim, c.rd if lay.rope else 0, q)
        q = q.view(Q, c.idx_heads, c.idx_dim)
        w32 = wts.float() * ((c.idx_heads ** -0.5) * (c.idx_dim ** -0.5))
        keys = self.pool.keys[L][seq.base:seq.base + seq.rows]
        if rnd.mine is not None:
            keys, st, rank = rnd.copy.keys(L), ONE, 0
        cand = torch.empty((Q, k), dtype=torch.int64, device=dev)
        rows = torch.empty((Q, k), dtype=torch.int32, device=dev)
        n = part.vis_max
        tile = max(64, SCORE_BYTES // (max(n, 1) * 4) // 64 * 64)
        four = keys.dtype == torch.uint8
        narrow = SCORE8 and (Q >= SCORE8 or four) and n
        if narrow:
            q8, qw = (score4_rows(q, w32), None) if four else score8_query(q, w32)
            wide = (n + 7) // 8 * 8
        for c0 in range(0, Q, tile):
            c1 = min(Q, c0 + tile)
            if narrow:
                s = self.scores[:(c1 - c0) * wide].view(c1 - c0, wide)
                top = self.scores[(c1 - c0) * wide:(c1 - c0) * wide * 9 // 8].view(c1 - c0, wide // 8)
                if four:
                    score4(q8, c0, keys, part.vis[c0:c1], n, s, top)
                else:
                    score8(q8, qw, c0, keys, part.vis[c0:c1], n, s, top)
                if n >= 16 * k:
                    top8(s, top, part.vis[c0:c1], k, st, rank, cand[c0:c1], rows[c0:c1])
                else:                                            # too few eights to bound k scores
                    local_top(s, part.vis[c0:c1], k, st, rank, cand[c0:c1], rows[c0:c1])
                continue
            s = torch.empty((c1 - c0, max(n, 1)), dtype=F32, device=dev)
            if n:
                index_scores(q[c0:c1], w32[c0:c1], keys, part.vis[c0:c1], n, s)
            local_top(s, part.vis[c0:c1], k, st, rank, cand[c0:c1], rows[c0:c1])
        return cand, rows

    def _select(self, L: int, lay, qi: torch.Tensor, wts: torch.Tensor, rnd: cp4.Round) -> tuple:
        comm, k = self.comm, self.cfg.idx_topk
        cand, rows = self._top(L, lay, qi, wts, rnd, rnd)
        got = torch.empty((self.world, rnd.counts[self.rank], k), dtype=torch.int64, device=qi.device)
        cp4.alltoall(comm, self._split(cand, rnd.counts), list(got))
        thresh = cp4.gather_rows(comm, kth_rows(got), rnd.counts)
        return keep_rows(cand, rows, thresh, rnd.seq.base)

    def _attend(self, L: int, lay, q_raw: torch.Tensor, qpos: torch.Tensor, rows: torch.Tensor, cnt: torch.Tensor,
                plane: torch.Tensor | None = None) -> tuple:
        """This rank's share of every query's attention: o [Q, H, V] and lse [Q, H], normalized over its rows of ``plane`` (the pool's)."""

        c = self.cfg
        rope_w = c.rd if lay.rope else 0
        nope = c.qk_head - rope_w
        Q, H, C, V, dev = qpos.numel(), c.n_heads, c.kv_lora, c.v_head, q_raw.device
        wuk, wuv = self._heads(lay)
        o = torch.empty((Q, H, V), dtype=F16, device=dev)
        lse = torch.empty((Q, H), dtype=F32, device=dev)
        for a in range(0, Q, cp4.ATTN_ROWS):
            m = min(cp4.ATTN_ROWS, Q - a)
            at = slice(a, a + m)
            pos = qpos[at]
            qn = torch.empty((H, m, nope), dtype=F16, device=dev)
            q_scale(q_raw[at], pos, lay.ssmax_s, lay.ssmax_b, c.qk_head, qn)
            pe = None
            if rope_w:
                pe = torch.empty((m, H * rope_w), dtype=F16, device=dev)
                q_rope(q_raw[at], pos, lay.ssmax_s, lay.ssmax_b, self.inv_freq, H, c.qk_head, rope_w, pe)
                pe = pe.view(m, H, rope_w)
            q_lat = torch.bmm(qn, wuk)
            part = torch.empty((H, m, C), dtype=F16, device=dev)
            sparse_mla(q_lat.transpose(0, 1), pe, rows[at], cnt[at], self.pool.lat[lay.kv_source] if plane is None else plane, c.sm_scale,
                       part.transpose(0, 1), lse[at])
            torch.bmm(part, wuv.transpose(1, 2), out=o[at].transpose(0, 1))
        return o, lse

    def _send(self, o: torch.Tensor, lse: torch.Tensor, counts: list, n: int) -> tuple:
        """Every rank's share of this rank's ``n`` rows: [W, n, H, V] and [W, n, H]."""

        comm = self.comm
        r_o = torch.empty((self.world, n, *o.shape[1:]), dtype=F16, device=o.device)
        r_l = torch.empty((self.world, n, lse.shape[1]), dtype=F32, device=o.device)
        cp4.alltoall(comm, self._split(o, counts), list(r_o))
        cp4.alltoall(comm, self._split(lse, counts), list(r_l))
        return r_o, r_l

    def _whole(self, L: int, lay, q: torch.Tensor, qi: torch.Tensor | None, wts: torch.Tensor | None, rnd: cp4.Round) -> torch.Tensor:
        comm, counts = self.comm, rnd.counts
        if qi is not None:
            qi, wts = cp4.gather_rows(comm, qi, counts), cp4.gather_rows(comm, wts, counts)
        q_all = cp4.gather_rows(comm, q, counts)
        if qi is not None:
            rnd.sel = self._select(L, lay, qi, wts, rnd)
        o, lse = self._attend(L, lay, q_all, rnd.qpos, *rnd.sel)
        return merge_rows(*self._send(o, lse, counts, q.shape[0]))

    def _sliced(self, L: int, lay, q: torch.Tensor, qi: torch.Tensor | None, wts: torch.Tensor | None, rnd: cp4.Round) -> torch.Tensor:
        """``_whole`` a slice at a time. Exchanges run on the side stream, in the same order on every rank, kernels here: a
        slice's top-k beside the last one's keys, its attention beside the last one's rows. A tensor the side stream reads
        stays referenced until this stream has waited for that read; the side stream waits for this one before any write,
        so what its pool hands out again is free."""

        comm, side, parts, seq = self.comm, self.side, rnd.parts, rnd.seq
        main, k, last = torch.cuda.current_stream(), self.cfg.idx_topk, len(rnd.parts) - 1
        hold: list = []

        side.wait_stream(main)
        asked, q_all = [], []
        with torch.cuda.stream(side):
            if qi is not None:
                for p in parts:
                    asked.append((cp4.gather_rows(comm, qi[p.lo:p.hi], p.counts), cp4.gather_rows(comm, wts[p.lo:p.hi], p.counts),
                                  mark(side)))
            for p in parts:
                q_all.append((cp4.gather_rows(comm, q[p.lo:p.hi], p.counts), mark(side)))

        tops, kept, limits = [], [], [None] * len(parts)

        def limit(i: int) -> None:
            got, e = kept[i]
            main.wait_event(e)
            t = kth_rows(got)
            hold.append(t)
            e = mark(main)
            with torch.cuda.stream(side):
                side.wait_event(e)
                limits[i] = (cp4.gather_rows(comm, t, parts[i].counts), mark(side))

        if qi is not None:
            rnd.sel = [None] * len(parts)
            for i, p in enumerate(parts):
                g, w, e = asked[i]
                main.wait_event(e)
                tops.append(self._top(L, lay, g, w, rnd, p))
                e = mark(main)
                with torch.cuda.stream(side):
                    side.wait_event(e)
                    got = torch.empty((self.world, p.hi - p.lo, k), dtype=torch.int64, device=q.device)
                    cp4.alltoall(comm, self._split(tops[i][0], p.counts), list(got))
                    kept.append((got, mark(side)))
                if i:
                    limit(i - 1)
        outs = []
        for i, p in enumerate(parts):
            if qi is not None:
                t, e = limits[i]
                main.wait_event(e)
                rnd.sel[i] = keep_rows(*tops[i], t, seq.base)
            g, e = q_all[i]
            main.wait_event(e)
            o, lse = self._attend(L, lay, g, p.qpos, *rnd.sel[i])
            hold += [o, lse]
            e = mark(main)
            with torch.cuda.stream(side):
                side.wait_event(e)
                outs.append((*self._send(o, lse, p.counts, p.hi - p.lo), mark(side)))
            if qi is not None and i == 0:
                limit(last)
        att = torch.empty((q.shape[0], self.cfg.n_heads * self.cfg.v_head), dtype=F16, device=q.device)
        for p, (r_o, r_l, e) in zip(parts, outs):
            main.wait_event(e)
            merge_rows(r_o, r_l, att[p.lo:p.hi])
        return att

    def _share(self, L: int, lay, rnd: cp4.Round) -> torch.cuda.Event | None:
        """Every rank's new rows of this layer into this rank's copy of the prompt, on the side: when they are in."""

        pool, main, copy = self.pool, torch.cuda.current_stream(), rnd.copy
        src = lay.kv_source
        planes = ([(pool.lat[src], copy.lat(src))] if src == L or copy is self.slots else []) + ([(pool.keys[L], copy.keys(L))] if lay.indexer == "full" and not (copy is self.slots and L in REUSE) else [])
        if not planes:
            return None
        side = self.side or main
        side.wait_stream(main)
        with torch.cuda.stream(side):
            for plane, rows in planes:
                new = cp4.gather_rows(self.comm, plane[rnd.own_rows], rnd.counts)
                rows.index_copy_(0, rnd.mine.at, new if rows.dtype == plane.dtype else score4_rows(new))
            return mark(side)

    def _copied(self, L: int, lay, q: torch.Tensor, qi: torch.Tensor | None, wts: torch.Tensor | None, rnd: cp4.Round) -> torch.Tensor:
        """``_whole`` over the prompt's copy: this rank's queries over all the rows, so nothing to exchange or merge."""

        if not q.shape[0]:
            return q.new_empty((0, self.cfg.n_heads * self.cfg.v_head))
        me = rnd.mine
        if qi is not None:
            rnd.sel = self._top(L, lay, qi, wts, rnd, me)[1], me.vis.clamp(max=self.cfg.idx_topk)
            if L in REUSE.values():
                rnd.picks[L] = rnd.sel
        elif lay.indexer == "full":
            rnd.sel = rnd.picks[REUSE[L]]
            if L == max(REUSE):
                rnd.picks.clear()
        return self._attend(L, lay, q, me.qpos, *rnd.sel, rnd.copy.lat(lay.kv_source))[0].view(q.shape[0], -1)

    def _landed(self, rnd: cp4.Round) -> None:
        if rnd.ready is not None:
            torch.cuda.current_stream().wait_event(rnd.ready)
        rnd.ready = rnd.held = None

    def _layer(self, L: int, x: torch.Tensor, rnd: cp4.Round) -> torch.Tensor:
        x = self._front(L, x, rnd)
        self._back(rnd)
        return x

    def _front(self, L: int, x: torch.Tensor, rnd: cp4.Round) -> torch.Tensor:
        """A layer up to the experts: their rows leave here."""

        c, lay, ops, pool = self.cfg, self.w.layers[L], self.ops, self.pool
        n, inv = x.shape[0], self.inv_freq
        rope_w = c.rd if lay.rope else 0
        deep = rnd.mine is not None and rnd.copy is self.slots
        small = ops.mat32 if deep and TF32 else ops.mat
        self._landed(rnd)
        x, xn, x16 = add_norm(rnd.pending, x, lay.attn_norm, c.eps)
        rnd.pending = None
        if lay.kv_source == L and n:
            kv_row(small(lay.kv_a, xn), lay.kv_a_norm, rnd.own, rnd.own_rows, inv, pool.lat[L], c.eps, c.kv_lora, rope_w)
        if lay.indexer == "full" and n:
            key_row(small(lay.idx_wk, xn), lay.idx_k_norm, lay.idx_k_norm_b, rnd.own, rnd.own_rows, inv, pool.keys[L],
                    c.index_norm_eps, c.idx_dim, rope_w)
        shared = self._share(L, lay, rnd) if rnd.mine is not None else None
        q = ops.lin(lay.q, x16)
        qi = wts = None
        if lay.indexer == "full" and not (deep and L in REUSE):
            qi = ops.mat(lay.idx_wq_b, layer_norm16(xn, lay.idx_q_norm, lay.idx_q_norm_b, c.index_norm_eps))
            wts = small(lay.idx_proj, xn)
        if rnd.mine is not None:
            rnd.copy.wait(L)
        if shared is not None:
            torch.cuda.current_stream().wait_event(shared)
        y = ops.lin(lay.o, (self._copied if rnd.mine is not None else self._sliced if rnd.parts else self._whole)(L, lay, q, qi, wts, rnd))
        x, xn, x16 = add_norm(y[None], x, lay.ffn_norm, c.eps)
        if lay.dense:
            act = F.silu(ops.lin(lay.dense_gate, x16).float()) * ops.lin(lay.dense_up, x16).float()
            rnd.pending = ops.lin(lay.dense_down, act)[None]
        else:
            w, idx = route_softmax(ops.router(lay.gate_w, xn), c.topk)
            picks = torch.cat([idx.float(), w.float()], 1)
            side = self.side or torch.cuda.current_stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                rnd.moe = (lay, cp4.gather_rows(self.comm, x16, rnd.counts), cp4.gather_rows(self.comm, picks, rnd.counts), mark(side),
                           x16, picks, idx.shape[1])
        return x

    def _back(self, rnd: cp4.Round) -> None:
        """A layer's experts, when it has any: every rank's quarter of all the rows, a row's owner gets the four partials [W, n, d],
        to add in rank order."""

        if rnd.moe is None:
            return
        comm, counts, main = self.comm, rnd.counts, torch.cuda.current_stream()
        side = self.side or main
        lay, every, picks, e, x16, _, k = rnd.moe
        rnd.moe = None
        main.wait_event(e)
        part = self.ops.split_moe(self, lay, every, picks[:, :k].to(torch.int32), picks[:, k:]).to(x16.dtype)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            recvs = torch.empty((comm.world, *x16.shape), dtype=x16.dtype, device=x16.device)
            cp4.alltoall(comm, self._split(part, counts), list(recvs))
            rnd.ready = mark(side)
        rnd.pending, rnd.held = recvs, part

    def advance(self, f: cp4.Flight | Pair, layers: int, collect: list | None = None, sel_out: list | None = None) -> None:
        slots = f.rnd.copy if f.rnd.copy is self.slots else None
        if not isinstance(f, Pair) and slots is None:
            super().advance(f, layers, collect, sel_out)
            return
        a, b = (f.a, f.b) if isinstance(f, Pair) else (f, None)
        end = self.cfg.n_layers
        for L in range(f.at, min(f.at + layers, end)):
            a.x = self._front(L, a.x, a.rnd)
            if b is not None:
                b.x = self._front(L, b.x, b.rnd)
            if slots is not None:
                slots.done()
                if L + 1 < end:
                    slots.fetch(L + 1, self.w.layers[L + 1], f.rnd.seq, f.old)
            self._back(a.rnd)
            if b is not None:
                self._back(b.rnd)
                b.at = L + 1
            a.at = f.at = L + 1
            self._pace()

    def _pace(self) -> None:
        """This thread stays a layer ahead of the GPU: never idle between two slices, and what a new request queues behind is short."""

        at, self.lag = self.lag, mark(torch.cuda.current_stream())
        if at is None:
            return
        keep, little = os.sched_getaffinity(0), little_cores() if PACE_LITTLE else None
        if little:
            os.sched_setaffinity(0, little)
        at.synchronize()
        if little:
            os.sched_setaffinity(0, keep)

    def let_go(self, f: cp4.Flight | Pair) -> None:
        """A round that stops between its layers: its stream is gone."""

        if f.rnd.copy is self.slots and f.rnd.copy is not None:
            f.rnd.copy.busy = False

    def finish(self, f: cp4.Flight | Pair) -> torch.Tensor | None:
        if isinstance(f, Pair):
            self.finish(f.a)
            return self.finish(f.b)
        self.let_go(f)
        self._landed(f.rnd)
        if f.rnd.pending is not None:
            f.x = add_norm(f.rnd.pending, f.x, self.w.norm, self.cfg.eps)[0]
            f.rnd.pending = None
        return super().finish(f)
