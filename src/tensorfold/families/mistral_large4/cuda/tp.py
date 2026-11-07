"""The decode step: every rank works on every decode row, so a token reads each replicated weight once
across the lane instead of once on its owner.

``cp4.Ml4Cp.round`` puts a row's per-token work on its owner: 18 GB read by one rank for one token, 66 ms at GB10's 273 GB/s
before anything else runs. Here x is the same on all ranks and the projections are split: q and the indexer's wq_b by head,
o and the shared expert's down by input columns, the head by vocabulary. Five collectives a full layer, four a shared one:

    all-gather  absorbed q, k_pe query and indexer q of this rank's heads
    all-gather  this rank's top-K candidate keys (every rank then finds the same K-th key, one exchange less than a round)
    all-to-all  partial o_lat by head: rank r merges its own heads, so it expands them with its slice of W_UV
    all-gather  o_proj partials, added in rank order
    all-gather  expert and shared-expert partials, added in rank order

Ranks add the same partials in the same order, so x stays bit-identical on all of them, and with it the router's picks: no
dispatch, a rank just runs its own experts on the row. A row's KV goes where ``Stripe`` says, as it does in a round.

Everything a step reads that moves is in ``Rows``' device tensors, so ``Graphs`` can replay a capture at any position.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from . import cp4
from .cp4 import alltoall, filter_selection, kth_key, merge_partials
from .model import gptj_rotate, layer_norm_bias, rms_norm, rotate_leading, route_softmax, softcap, ssmax_apply

F32 = torch.float32
HELD_ROWS = 64                        # past any step's rows or a draft's tokens


@dataclass
class Rows:
    """One step's rows. ``pos`` is the same on every rank; ``wr`` is this rank's pool row for the row's new KV (the pool's
    scratch row when another rank owns it), ``base`` the stream's first local row, ``vis`` how many of its rows this rank
    sees, the new one included."""

    pos: torch.Tensor                 # int64 [R]
    wr: torch.Tensor                  # int64 [R]
    base: torch.Tensor                # int64 [R]
    vis: torch.Tensor                 # int32 [R]
    n: int                            # score width: no row sees more rows on this rank
    indexed: bool                     # False: every row's context fits the budget, so all its rows are selected


def plan_rows(seqs: list, positions: list, rank: int, n: int | None = None, indexed: bool | None = None,
              k: int = 0, scratch: int = 0, pad: int = 0) -> tuple:
    """``seqs[i]`` at ``positions[i]``, host-side: (pos, wr, base, vis, n, indexed). Rows past the real ones, up to ``pad``, repeat
    the first and write nowhere: a capture has a fixed row count, and a repeated row picks the same experts."""

    wr, base, vis = [], [], []
    for s, p in zip(seqs, positions):
        st = s.stripe
        t = torch.tensor(p)
        wr.append(s.base + int(st.row(t)) if int(st.rank(t)) == rank else scratch)
        base.append(s.base)
        vis.append(int(st.vis(rank, t)))
    n = max(vis) if n is None else n
    indexed = max(positions) + 1 > k if indexed is None else indexed
    extra = max(0, pad - len(positions))
    return positions + positions[:1] * extra, wr + [scratch] * extra, base + base[:1] * extra, vis + vis[:1] * extra, n, indexed


def rows_on(plan: tuple, device) -> Rows:
    pos, wr, base, vis, n, indexed = plan
    return Rows(torch.tensor(pos, device=device), torch.tensor(wr, device=device), torch.tensor(base, device=device),
                torch.tensor(vis, dtype=torch.int32, device=device), n, indexed)


def make_rows(seqs: list, positions: list, rank: int, device, n: int | None = None, indexed: bool | None = None,
              k: int = 0, scratch: int = 0, pad: int = 0) -> Rows:
    return rows_on(plan_rows(seqs, positions, rank, n, indexed, k, scratch, pad), device)


def width(st: cp4.Stripe, total: int, world: int, k: int) -> int:
    """The score width for a row of ``total`` tokens: a power of two from the budget up, past what any rank holds."""

    need, n = int(st.vis(0, torch.tensor(total - 1))), k
    while n < need:
        n *= 2
    return n


@dataclass
class TpLayer:
    """This rank's slices of one layer (views: nothing here takes memory)."""

    q: object
    o: object
    wuk: torch.Tensor
    wuv: torch.Tensor
    ssmax_s: torch.Tensor
    ssmax_b: torch.Tensor
    idx_wq_b: object = None
    idx_ssmax_s: torch.Tensor | None = None
    idx_ssmax_b: torch.Tensor | None = None
    gate: object = None               # the shared expert's, or the dense layer's
    up: object = None
    down: object = None


class Ml4Tp:
    def __init__(self, fwd: cp4.Ml4Cp) -> None:
        self.fwd, self.cfg, self.w, self.ops, self.pool = fwd, fwd.cfg, fwd.w, fwd.ops, fwd.pool
        self.comm, self.rank, self.world = fwd.comm, fwd.rank, fwd.world
        self.inv_freq = fwd.inv_freq
        c, W, r = self.cfg, self.world, self.rank
        self.hl, self.il = c.n_heads // W, c.idx_heads // W
        self.layers = [self._slices(lay) for lay in self.w.layers]
        head = self.w.head if self.w.head is not None else self.w.head_bf16
        self.vl = c.vocab // W
        self.head = head if getattr(head, "n", 0) == self.vl else self.ops.split_mat(head, r * self.vl, (r + 1) * self.vl)
        self.table = None
        self.taps, self.tap_at = None, {}                        # dspark.py's: rows of the residual stream its context is made of

    def _slices(self, lay) -> TpLayer:
        c, W, r, ops = self.cfg, self.world, self.rank, self.ops
        hs = slice(r * self.hl, (r + 1) * self.hl)
        wuk, wuv = self.fwd._heads(lay)
        qw, ov = self.hl * c.qk_head, self.hl * c.v_head
        t = TpLayer(q=ops.split_mat(lay.q, r * qw, (r + 1) * qw), o=ops.split_in(lay.o, r * ov, (r + 1) * ov),
                    wuk=wuk[hs], wuv=wuv[hs], ssmax_s=lay.ssmax_s[hs], ssmax_b=lay.ssmax_b[hs])
        if lay.indexer == "full":
            iw = self.il * c.idx_dim
            t.idx_wq_b = ops.split_mat(lay.idx_wq_b, r * iw, (r + 1) * iw)
            t.idx_ssmax_s = lay.idx_ssmax_s[r * self.il:(r + 1) * self.il]
            t.idx_ssmax_b = lay.idx_ssmax_b[r * self.il:(r + 1) * self.il]
        t.gate, t.up, t.down = ops.split_ffn(lay, r, W)
        return t

    # -- pieces --------------------------------------------------------------------------------------------------------

    def tap(self, at: int, x: torch.Tensor) -> None:
        """The residual stream entering layer ``at`` (past the last: the last layer's output), when the draft head reads it."""

        t = self.tap_at.get(at)
        if t is not None:
            self.taps[t, :x.shape[0]].copy_(x)

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Tokens ids [R] on the GPU: their rows [R, d], each from the rank whose part of the table holds it. The whole table stays
        in the pack's map: an unseen row from there is a read of the disk, about .4 ms."""

        at = ids - self.rank * self.vl
        mine = (at >= 0) & (at < self.vl)
        return self._total(torch.where(mine[:, None], self.part_of_table()[at.clamp(0, self.vl - 1)].float(), 0.0))

    def part_of_table(self) -> torch.Tensor:
        if self.table is None:                                   # a mapped file's rows reach the GPU only through a host copy
            first = self.rank * self.vl
            self.table = self.w.embed[first:first + self.vl].clone().to(self.pool.device)
        return self.table

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Normed rows h: this rank's softcapped slice of their logits."""

        return softcap(self.ops.logits(self.head, h), self.cfg.softcap)

    def last_logits(self, h: torch.Tensor | None) -> torch.Tensor | None:
        """A round's last row, h [1, d] on its owner and None on the others, through every rank's slice of the head: logits [V] on
        the owner. The others send zeros, so the sum is the row."""

        row = self._total(torch.zeros((1, self.cfg.dim), dtype=F32, device=self.pool.device) if h is None else h.float())
        part = self.head_logits(row).contiguous()
        every = torch.empty((self.world, *part.shape), dtype=part.dtype, device=part.device)
        self.comm.all_gather(part.view(-1), every.view(-1))
        return None if h is None else every.view(-1)

    def _total(self, part: torch.Tensor) -> torch.Tensor:
        """All ranks' fp32 partials added in rank order, the same bits on every rank."""

        if self.comm is None:
            return part
        every = torch.empty((self.world, *part.shape), dtype=part.dtype, device=part.device)
        self.comm.all_gather(part.reshape(-1), every.view(-1))
        out = every[0]
        for k in range(1, self.world):
            out = out + every[k]
        return out

    def _everyone(self, parts: list) -> list:
        """Each of ``parts`` [R, h, *] of this rank's heads as [R, W * h, *]: one all-gather of the bytes."""

        if self.comm is None:
            return parts
        R = parts[0].shape[0]
        flat = [p.reshape(R, -1) for p in parts]
        sizes = [f.shape[1] for f in flat]
        send = torch.cat(flat, 1)
        every = torch.empty((self.world, *send.shape), dtype=send.dtype, device=send.device)
        self.comm.all_gather(send.view(-1), every.view(-1))
        out, at = [], 0
        for p, n in zip(parts, sizes):
            got = every[:, :, at:at + n].reshape(self.world, R, p.shape[1], -1)
            out.append(got.permute(1, 0, 2, 3).reshape(R, self.world * p.shape[1], *p.shape[2:]))
            at += n
        return out

    def _select(self, L: int, q: torch.Tensor, wts: torch.Tensor, s: Rows) -> tuple:
        c, st = self.cfg, self.stripe
        cand, rows = self.ops.local_topk_rows(q, wts, self.pool.keys[L], s.base, s.vis, s.n, c.idx_topk,
                                              lambda r: st.pos(self.rank, r))
        if self.comm is None:
            return filter_selection(cand, rows, None)
        every = torch.empty((self.world, *cand.shape), dtype=cand.dtype, device=cand.device)
        self.comm.all_gather(cand.view(-1), every.view(-1))
        thresh = kth_key(every.permute(1, 0, 2).reshape(cand.shape[0], -1), c.idx_topk)
        return filter_selection(cand, rows, thresh)

    def _whole(self, s: Rows) -> tuple:
        """The selection of a context inside the budget: every row this rank holds of it."""

        r = torch.arange(s.n, dtype=torch.int32, device=s.pos.device)[None, :].expand(s.pos.numel(), -1)
        return torch.where(r < s.vis[:, None], r, cp4.NO_ROW).contiguous(), s.vis

    def _attention(self, L: int, lay, t: TpLayer, q_lat: torch.Tensor, q_pe, sel: tuple, s: Rows) -> torch.Tensor:
        """This rank's heads of the merged, expanded attention output [R, hl * v]."""

        c = self.cfg
        rows, cnt = sel
        rows = torch.where(rows == cp4.NO_ROW, rows, rows + s.base[:, None].to(rows.dtype))
        o, lse = self.ops.attend(q_lat, q_pe, self.pool, lay.kv_source, rows, cnt, c.sm_scale)
        R, hl = o.shape[0], self.hl
        if self.comm is None:
            merged = merge_partials(o[None], lse[None])
        else:
            s_o = [o[:, k * hl:(k + 1) * hl].contiguous() for k in range(self.world)]
            s_l = [lse[:, k * hl:(k + 1) * hl].contiguous() for k in range(self.world)]
            r_o = [torch.empty_like(s_o[0]) for _ in range(self.world)]
            r_l = [torch.empty_like(s_l[0]) for _ in range(self.world)]
            alltoall(self.comm, s_o, r_o)
            alltoall(self.comm, s_l, r_l)
            merged = merge_partials(torch.stack(r_o), torch.stack(r_l))
        return self.ops.expand_o(merged.to(self.ops.wire), t.wuv).reshape(R, hl * c.v_head)

    def _layer(self, L: int, x: torch.Tensor, s: Rows, sel: tuple | None) -> tuple:
        c, lay, t, ops = self.cfg, self.w.layers[L], self.layers[L], self.ops
        R = x.shape[0]
        rope_w = c.rd if lay.rope else 0
        nope_w = c.qk_head - rope_w
        xn = rms_norm(x, lay.attn_norm, c.eps)
        if lay.kv_source == L:
            ckv = ops.mat(lay.kv_a, xn)
            lat = rms_norm(ckv[:, :c.kv_lora], lay.kv_a_norm, c.eps)
            kpe = gptj_rotate(ckv[:, c.kv_lora:].float(), self.inv_freq, s.pos) if rope_w else None
            self.pool.store(L, s.wr, lat, kpe)
        q = ops.lin(t.q, xn).to(ops.wire)
        q = ssmax_apply(q.float().view(R, self.hl, c.qk_head), t.ssmax_s, t.ssmax_b, s.pos)
        q_pe = rotate_leading(q[..., nope_w:], self.inv_freq, s.pos, rope_w) if rope_w else None
        q_lat = ops.absorb_q(q[..., :nope_w], t.wuk)
        parts = [q_lat.to(ops.wire)] + ([q_pe.to(ops.wire)] if rope_w else [])
        full = lay.indexer == "full"
        if full:
            k = layer_norm_bias(ops.mat(lay.idx_wk, xn), lay.idx_k_norm, lay.idx_k_norm_b, c.index_norm_eps)
            k = rotate_leading(k.view(R, 1, 1, -1), self.inv_freq, s.pos, rope_w).view(R, -1)
            self.pool.keys[L][s.wr] = k.to(self.pool.keys[L].dtype)
            if s.indexed:
                qi = ops.mat(t.idx_wq_b, layer_norm_bias(xn, lay.idx_q_norm, lay.idx_q_norm_b,
                                                         c.index_norm_eps)).to(ops.wire)
                qi = ssmax_apply(qi.float().view(R, self.il, c.idx_dim), t.idx_ssmax_s, t.idx_ssmax_b, s.pos)
                parts.append(rotate_leading(qi, self.inv_freq, s.pos, rope_w).to(ops.wire))
        got = self._everyone(parts)
        q_lat, q_pe = got[0], (got[1] if rope_w else None)
        if full:
            if s.indexed:
                # relu(a c) = c relu(a) for c > 0: the eager D^-0.5 and H^-0.5 fold into the head weights
                wts = ops.mat(lay.idx_proj, xn).float() * (c.idx_heads ** -0.5) * (c.idx_dim ** -0.5)
                sel = self._select(L, got[-1], wts, s)
            else:
                sel = self._whole(s)
        x = x + self._total(ops.lin32(t.o, self._attention(L, lay, t, q_lat, q_pe, sel, s)))
        xn2 = rms_norm(x, lay.ffn_norm, c.eps)
        if lay.dense:
            return x + self._total(ops.ffn_part(t, xn2)), sel
        w, idx = route_softmax(ops.router(lay.gate_w, xn2), c.topk)
        return x + self._total(ops.moe_part(self.fwd, lay, t, xn2, idx, w)), sel

    def forward(self, x: torch.Tensor, s: Rows) -> torch.Tensor:
        """Rows' residual stream x [R, d] fp32 through every layer: this rank's slice of the logits [R, V / W]."""

        sel = None
        for L in range(self.cfg.n_layers):
            self.tap(L, x)
            x, sel = self._layer(L, x, s, sel)
        self.tap(self.cfg.n_layers, x)
        return self.head_logits(rms_norm(x, self.w.norm, self.cfg.eps))

    def best(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy ids [R] from every rank's slice, the lowest id on a tie (argmax over the whole vocabulary)."""

        v, i = logits.float().max(1)
        i = i + self.rank * self.vl
        if self.comm is None:
            return i
        mine = torch.stack([v, i.to(F32)], 1)                    # ids < 2^24: exact in fp32
        every = torch.empty((self.world, *mine.shape), dtype=F32, device=mine.device)
        self.comm.all_gather(mine.view(-1), every.view(-1))
        return every[every[:, :, 0].argmax(0), torch.arange(mine.shape[0], device=mine.device), 1].long()

    @property
    def stripe(self) -> cp4.Stripe:
        return self._stripe

    def use(self, stripe: cp4.Stripe) -> None:
        """Every stream of a step shares one stripe map (a key's position comes from it)."""

        self._stripe = stripe

    def step(self, tokens: torch.Tensor, s: Rows) -> torch.Tensor:
        """The rows' tokens through the lane: this rank's logits slice."""

        return self.forward(self.ops.embed(self.w, tokens), s)


class Held:
    """Pinned memory of a step behind its draft: its tokens and row numbers on their way in, the tokens with the drafts in them on
    their way out. A step ends before the next one writes any of these."""

    def __init__(self) -> None:
        self.ints = torch.zeros((4, HELD_ROWS), dtype=torch.long, pin_memory=True)
        self.vis = torch.zeros((HELD_ROWS,), dtype=torch.int32, pin_memory=True)
        self.back = torch.zeros((HELD_ROWS,), dtype=torch.long, pin_memory=True)


class Graphs:
    """One capture per (rows, score width, indexed), the ids out of it. A capture's collectives run when it is replayed, so
    every rank must replay the same captures in the same order: the serving loop's ops already are that order."""

    def __init__(self, lane: Ml4Tp) -> None:
        self.lane, self.caps, self.pool, self.held = lane, {}, None, None
        self.mark = lambda name: None

    def _capture(self, key: tuple, like: Rows):
        R, n, indexed = key
        lane, dev = self.lane, like.pos.device
        tok = torch.zeros((R,), dtype=torch.long, device=dev)
        s = Rows(torch.zeros_like(like.pos), torch.zeros_like(like.wr), torch.zeros_like(like.base),
                 torch.zeros_like(like.vis), n, indexed)
        for b, v in zip((s.pos, s.wr, s.base, s.vis), (like.pos, like.wr, like.base, like.vis)):
            b.copy_(v)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                            # the Triton kernels compile and the buffers exist
            lane.best(lane.forward(lane.rows(tok), s))
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            logits = lane.forward(lane.rows(tok), s)
            ids = lane.best(logits)
        self.pool = self.pool or g.pool()
        self.caps[key] = (g, tok, s, ids, logits)
        from .cp4_cuda import trim_host

        trim_host()                                              # a capture's host side, about .1 GiB of the floor
        self.note(key)
        return self.caps[key]

    def note(self, key: tuple) -> None:
        """What a capture cost (TF_ML4_STEP_LOG): the host's MemAvailable and what the allocator holds, after it."""

        if os.environ.get("TF_ML4_STEP_LOG") and self.lane.rank == 0:
            with open("/proc/meminfo") as f:
                free = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable")) / 2 ** 20
            print(f"[ml4-cap] {key}: MemAvailable {free:.2f} GiB, reserved {torch.cuda.memory_reserved() / 2 ** 30:.2f}, allocated "
                  f"{torch.cuda.memory_allocated() / 2 ** 30:.2f}, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}", flush=True)

    def run(self, tokens: torch.Tensor, rows: Rows) -> tuple:
        """(ids [R], this rank's logits slice [R, V / W]) of a replay; the two are the capture's own buffers."""

        key = (tokens.numel(), rows.n, rows.indexed)
        g, tok, s, ids, logits = self.caps.get(key) or self._capture(key, rows)
        tok.copy_(tokens)
        s.pos.copy_(rows.pos)
        s.wr.copy_(rows.wr)
        s.base.copy_(rows.base)
        s.vis.copy_(rows.vis)
        g.replay()
        return ids, logits

    def ready(self, tokens: list, plan: tuple) -> bool:
        return (len(tokens), plan[4], plan[5]) in self.caps

    def run_behind(self, tokens: list, plan: tuple, drafted: torch.Tensor | None) -> tuple:
        """``run`` behind whatever is on the GPU, nothing here waiting for it. A row can be a draft the host has not seen, -1 - its
        place in ``drafted``. (ids, logits, the tokens once the ids are in)."""

        from .tp_kernels import fill_rows

        pos, wr, base, vis, n, indexed = plan
        R = len(tokens)
        g, tok, s, ids, logits = self.caps[R, n, indexed]
        h = self.held = self.held or Held()
        h.ints[:, :R] = torch.tensor([tokens, pos, wr, base])
        h.vis[:R] = torch.tensor(vis, dtype=torch.int32)
        for b, v in zip((s.pos, s.wr, s.base, s.vis), (h.ints[1, :R], h.ints[2, :R], h.ints[3, :R], h.vis[:R])):
            b.copy_(v, non_blocking=True)
        if min(tokens) < 0:
            fill_rows(h.ints[0, :R].to(tok.device, non_blocking=True), drafted, tok)
        else:
            tok.copy_(h.ints[0, :R], non_blocking=True)
        h.back[:R].copy_(tok, non_blocking=True)
        g.replay()
        self.mark("slaunch")
        return ids, logits, h.back[:R]
