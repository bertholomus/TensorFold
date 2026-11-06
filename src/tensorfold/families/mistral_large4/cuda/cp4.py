"""CP4: one stream striped by position blocks across the ranks (DESIGN-CP4, D2 §6.4 option 1), and the rank-local
forward that serves DP rows with the same code as a group of one.

A round is rows at consecutive positions; a row's owner (the rank that stores its KV) does the per-token work, every
rank does the per-key work over its own quarter, and three small merges put the pieces together: unique int64 candidate
keys so the top-K cut over the union is exact (§3.2-4), a shipped (partial, lse) so softmax over the union is exact
up to rounding (§3.5-6), and the expert all-to-all. A decode step is a round of one row, so there is no replicated row
anywhere: every number exists on one rank and is shipped from there.

``Ops`` is the torch reference (the CPU gates in tests/test_ml4_cp4.py); ``cp4_cuda.CudaOps`` overrides the same calls
with the Triton kernels. Everything here takes the ``Ml4Weights`` slots ``model.py`` takes, and ``model.py`` stays the
reference semantics; the functions it owns (norms, rope, ssmax, router) are imported, not restated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import fp8kv
from .model import (gptj_rotate, layer_norm_bias, lerope_inv_freq, rms_norm, rotate_leading, route_softmax,
                    silu_expert, softcap, ssmax_apply)

F32 = torch.float32
BLOCK = int(os.environ.get("TF_ML4_CP_BLOCK") or 1024)    # §2: rows per rank per round, a power of two
ATTN_ROWS = int(os.environ.get("TF_ML4_CP_ATTN_ROWS") or 1024)   # queries a pass of attention holds latents for (§10)
PAD_KEY = -(2 ** 63)                                      # sorts below every real key
NO_ROW = 2 ** 31 - 1                                      # a selection list's padding: sorts after every real row


# -- striping (§2) ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Stripe:
    """Position blocks of ``block`` rows dealt to ``world`` ranks in turn. Local rows are in position order, so a
    query sees a prefix of every rank's rows. world 1 is the DP case: one rank holds everything in order."""

    world: int
    block: int

    def rank(self, p):
        return (p // self.block) % self.world

    def row(self, p):
        return (p // (self.world * self.block)) * self.block + p % self.block

    def pos(self, rank: int, row):
        """Inverse of ``row`` on ``rank``."""

        return (row // self.block) * (self.world * self.block) + rank * self.block + row % self.block

    def vis(self, rank: int, p: torch.Tensor) -> torch.Tensor:
        """Rows of ``rank`` a query at position ``p`` sees (itself included)."""

        wb = self.world * self.block
        n = p + 1
        return (n // wb) * self.block + ((n % wb) - rank * self.block).clamp(0, self.block)

    def rows_for(self, total: int, rank: int) -> int:
        """Rows ``rank`` holds of the first ``total`` positions."""

        return int(self.vis(rank, torch.tensor(total - 1))) if total else 0


# -- KV planes (§6, §8) -------------------------------------------------------------------------------------------------

class KvPool:
    """One rank's planes: a row array per writer layer, a key array per full-indexer layer, the same local row numbers
    in all of them. ``kind`` "fp8" is the serving row (fp8kv); "fp16" is gate 2's baseline; "raw" keeps fp32 rows, so
    the merge gates measure the merge alone. Streams hold extents of it: ``base`` is added to their local rows."""

    def __init__(self, cfg, rows: int, device="cpu", kind: str = "fp8", fmt: str = "f32x128") -> None:
        self.cfg, self.rows, self.kind, self.device = cfg, int(rows), kind, device
        self.dtype = F32 if kind == "raw" else torch.float16
        self.fmt_rope = fp8kv.row_format(fmt, cfg.kv_lora, cfg.rd)
        self.fmt_nope = self.fmt_rope.nope()
        self.lat: dict[int, torch.Tensor] = {}
        self.kpe: dict[int, torch.Tensor] = {}
        self.keys: dict[int, torch.Tensor] = {}
        for L in range(cfg.n_layers):
            if cfg.kv_source_layers[L] == L:
                if kind == "fp8":
                    self.lat[L] = torch.empty((self.rows, self.fmt(L).row_bytes), dtype=torch.uint8, device=device)
                else:
                    self.lat[L] = torch.zeros((self.rows, cfg.kv_lora), dtype=self.dtype, device=device)
                    if cfg.is_rope(L):
                        self.kpe[L] = torch.zeros((self.rows, cfg.rd), dtype=self.dtype, device=device)
            if cfg.is_full_indexer(L):
                self.keys[L] = torch.zeros((self.rows, cfg.idx_dim), device=device,
                                           dtype=F32 if kind == "raw" else torch.float16)

    def fmt(self, L: int) -> fp8kv.RowFormat:
        return self.fmt_rope if self.cfg.is_rope(L) else self.fmt_nope

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in (self.lat, self.kpe, self.keys) for t in d.values())

    @staticmethod
    def row_bytes(cfg, fmt: str = "f32x128") -> int:
        """fp8 pool bytes per position on one rank's planes: 51,440 on the real dims (§6)."""

        rope = fp8kv.row_format(fmt, cfg.kv_lora, cfg.rd)
        total = 2 * cfg.idx_dim * sum(1 for L in range(cfg.n_layers) if cfg.is_full_indexer(L))
        for L in range(cfg.n_layers):
            if cfg.kv_source_layers[L] == L:
                total += (rope if cfg.is_rope(L) else rope.nope()).row_bytes
        return total

    def store(self, L: int, rows: torch.Tensor, lat: torch.Tensor, kpe: torch.Tensor | None) -> None:
        if self.kind == "fp8":
            self.lat[L][rows] = fp8kv.encode(lat, kpe, self.fmt(L))
            return
        self.lat[L][rows] = lat.to(self.dtype)
        if kpe is not None:
            self.kpe[L][rows] = kpe.to(self.dtype)

    def gather(self, L: int, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """fp32 (latent, k_pe) of ``rows`` (any shape) on writer layer ``L`` — what the attention kernel reads."""

        flat = rows.reshape(-1)
        if self.kind == "fp8":
            lat, kpe = fp8kv.decode(self.lat[L][flat], self.fmt(L))
        else:
            lat, kpe = self.lat[L][flat].float(), (self.kpe[L][flat].float() if L in self.kpe else None)
        lat = lat.view(*rows.shape, -1)
        return lat, (None if kpe is None else kpe.view(*rows.shape, -1))


@dataclass
class Seq:
    """A stream's place in the pool: an extent of ``rows`` local rows from ``base`` on each rank of its group."""

    stripe: Stripe
    base: int = 0
    rows: int = 0
    rep: int | None = None            # a prompt's copy: all its rows from here, in position order, on every rank
    rep_rows: int = 0


@dataclass
class Round:
    """One round's rows as every rank sees them (rank-major query order: rank 0's rows, then rank 1's ...)."""

    seq: Seq
    qpos: torch.Tensor                # [Q] all rows' positions
    counts: list                      # rows per rank
    own: torch.Tensor                 # this rank's positions
    own_rows: torch.Tensor            # ... and their pool rows
    vis: torch.Tensor                 # [Q] rows of this rank each query sees
    vis_max: int                      # the largest of them
    sel: tuple | None = None          # (rows [Q, K], counts [Q]) of the nearest full layer
    picks: dict = field(default_factory=dict)   # a deep prompt's selections by layer, for the layers that take an earlier one
    pending: torch.Tensor | None = None   # FusedCp: [W, n, d] partials the residual has yet to take
    parts: list | None = None         # FusedCp: the round's queries in slices, ``sel`` then a list of a slice's each
    mine: object = None               # FusedCp: this rank's queries, when they run over all the rows
    copy: object = None               # FusedCp: ... and where those rows are
    moe: tuple | None = None          # FusedCp: a layer's experts, their rows on the way
    ready: object = None              # FusedCp: when ``pending`` has landed, and what its exchange reads till then
    held: object = None

    @property
    def nmax(self) -> int:
        return max(self.counts)


@dataclass
class Flight:
    """A round between its layers: decode steps and a shorter prompt's round can run while it waits."""

    rnd: Round
    x: torch.Tensor
    last: int | None
    at: int = 0                       # the next layer
    old: int = 0                      # FusedCp: rows a rank of earlier rounds


def make_round(seq: Seq, start: int, n: int, rank: int, device) -> Round:
    """Built on the host: everything here is known from (start, n), so a round costs no device sync."""

    st = seq.stripe
    p = torch.arange(start, start + n)
    who = st.rank(p)
    per = [p[who == k] for k in range(st.world)]
    counts = [int(t.numel()) for t in per]
    if st.world == 1:                                         # DP: this rank's own rows, nobody else's
        rank = 0
    qpos, own = torch.cat(per), per[rank]
    vis = st.vis(rank, qpos)
    return Round(seq=seq, qpos=qpos.to(device), counts=counts, own=own.to(device),
                 own_rows=(seq.base + st.row(own)).to(device), vis=vis.to(torch.int32).to(device),
                 vis_max=int(vis.max()) if vis.numel() else 0)


# -- comm helpers ---------------------------------------------------------------------------------------------------------

def alltoall(comm, sends: list, recvs: list) -> None:
    """``sends[j]`` lands in rank j's ``recvs[me]``; slot ``comm.rank`` is copied here. One NCCL group when the
    communicator has ``alltoall``, else the fixed pair order every rank agrees on (the CPU lane's ``exchange``)."""

    me = comm.rank
    if recvs[me].numel():
        recvs[me].copy_(sends[me])
    fn = getattr(comm, "alltoall", None)
    if fn is not None:
        fn(sends, recvs)
        return
    for i in range(comm.world):
        for j in range(i + 1, comm.world):
            if me in (i, j):
                peer = j if me == i else i
                comm.exchange([sends[peer]], [recvs[peer]], peer)


def gather_rows(comm, t: torch.Tensor, counts: list) -> torch.Tensor:
    """All ranks' rows of ``t`` in rank order, from a padded all-gather (ranks may hold unequal rows, or none)."""

    world, nmax = comm.world, max(counts)
    send = t
    if t.shape[0] != nmax:
        send = torch.zeros((nmax, *t.shape[1:]), dtype=t.dtype, device=t.device)
        send[:t.shape[0]] = t
    out = torch.empty((world * nmax, *t.shape[1:]), dtype=t.dtype, device=t.device)
    comm.all_gather(send.contiguous().view(-1), out.view(-1))
    if all(c == nmax for c in counts):
        return out
    return torch.cat([out[k * nmax:k * nmax + c] for k, c in enumerate(counts)])


# -- the merges (§3) --------------------------------------------------------------------------------------------------------

def score_keys(score: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """int64 keys, larger = better: the fp32 score's ordered bits above ``0xFFFFFFFF - pos``, so ties go to the lower
    position (``select_topk``'s stable-sort rule) and no two keys of a query are equal. ``+ 0.0`` folds -0 into +0."""

    bits = (score + 0.0).contiguous().view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    return (ordered << 32) | (0xFFFFFFFF - pos.to(torch.int64))


def kth_key(cands: torch.Tensor, k: int) -> torch.Tensor:
    """Per row of ``cands`` [n, m] the k-th largest key: everything >= it is the global top-k. Fewer than k real keys
    gives PAD_KEY, so every real key passes: the eager rule for a row that sees less than its budget."""

    if cands.shape[1] <= k:
        return torch.full((cands.shape[0],), PAD_KEY, dtype=torch.int64, device=cands.device)
    return torch.topk(cands, k, dim=1, sorted=True).values[:, -1]


def filter_selection(keys: torch.Tensor, rows: torch.Tensor, thresh: torch.Tensor | None) -> tuple:
    """This rank's share of the global selection: (rows [Q, K] ascending, NO_ROW padded; count [Q])."""

    keep = keys != PAD_KEY
    if thresh is not None:
        keep &= keys >= thresh[:, None]
    packed = torch.where(keep, rows, torch.full_like(rows, NO_ROW)).sort(dim=1).values
    return packed, keep.sum(1).to(torch.int32)


def merge_partials(o: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """Softmax over a union of disjoint row sets from its parts: o [W, n, H, v] normalized partials, lse [W, n, H]
    (a rank with no rows: lse -inf, o 0). fp32, adds in rank order."""

    m = lse.max(0).values
    a = torch.exp(lse.float() - m.float())
    a = torch.where(torch.isfinite(lse), a, torch.zeros_like(a))
    den = a.sum(0)
    acc = torch.zeros(o.shape[1:], dtype=F32, device=o.device)
    for k in range(o.shape[0]):
        acc += a[k][..., None] * o[k].float()
    return acc / den[..., None]


# -- the expert all-to-all, batched (§3; cuda/a2a.py is its row-at-a-time reference) ----------------------------------------

def ep_dispatch(comm, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, n_routed: int, run) -> torch.Tensor:
    """out [R, d] fp32 = this rank's rows' routed sum, for ``x`` [R, d] rows with picks ``idx``/``w`` [R, k]. A row goes
    once to each rank that owns one of its picks and comes back as one partial; partials add in destination-rank order
    (cuda/a2a.py's order). ``run(rows, eids, wts) -> [n, d]`` runs this rank's share: ``eids`` -1 marks another rank's
    slot. One small all-gather carries the counts, every other byte rides two all-to-alls."""

    world, rank, dev = comm.world, comm.rank, x.device
    R, k = idx.shape
    dest = idx // (n_routed // world)
    hit = torch.zeros((R, world), dtype=torch.bool, device=dev)
    if R:
        hit.scatter_(1, dest.long(), True)
    counts = torch.empty(world * world, dtype=torch.int32, device=dev)
    comm.all_gather(hit.sum(0).to(torch.int32), counts)
    C = counts.view(world, world).tolist()                    # C[src][dst]: the one host read of a layer

    rows_to = [hit[:, d].nonzero().squeeze(1) for d in range(world)]
    s_x, s_e, s_w = [], [], []
    for d in range(world):
        r = rows_to[d]
        mine = dest[r] == d
        s_x.append(x[r].contiguous())
        s_e.append(torch.where(mine, idx[r], torch.full_like(idx[r], -1)).to(torch.int32).contiguous())
        s_w.append(torch.where(mine, w[r], torch.zeros_like(w[r])).float().contiguous())
    r_x = [torch.empty((C[s][rank], x.shape[1]), dtype=x.dtype, device=dev) for s in range(world)]
    r_e = [torch.empty((C[s][rank], k), dtype=torch.int32, device=dev) for s in range(world)]
    r_w = [torch.empty((C[s][rank], k), dtype=F32, device=dev) for s in range(world)]
    alltoall(comm, s_x, r_x)
    alltoall(comm, s_e, r_e)
    alltoall(comm, s_w, r_w)

    rows, eids, wts = torch.cat(r_x), torch.cat(r_e), torch.cat(r_w)
    vec = run(rows, eids, wts).to(x.dtype)
    sizes = [C[s][rank] for s in range(world)]
    back = list(vec.split(sizes)) if vec.shape[0] else [vec[:0] for _ in range(world)]
    got = [torch.empty((C[rank][d], x.shape[1]), dtype=x.dtype, device=dev) for d in range(world)]
    alltoall(comm, [b.contiguous() for b in back], got)
    out = torch.zeros((R, x.shape[1]), dtype=F32, device=dev)
    for d in range(world):
        if got[d].shape[0]:
            out.index_add_(0, rows_to[d], got[d].float())
    return out


# -- the torch ops (the CPU reference; CudaOps overrides these) ----------------------------------------------------------------

class Ops:
    """Reference arithmetic: fp32, plain torch, the eager model's math in absorbed form."""

    wire = F32                    # q / o wire dtype

    def lin(self, mod, x: torch.Tensor) -> torch.Tensor:
        """x @ W.T for a linear slot (a plain tensor on CPU, Exl3Linear on CUDA)."""

        return x.float() @ mod.float().t()

    def mat(self, mat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ mat.float().t()

    def router(self, gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ gate.float().t()

    def embed(self, w, tokens: torch.Tensor) -> torch.Tensor:
        return w.embed[tokens].float()

    def logits(self, head, h: torch.Tensor) -> torch.Tensor:
        return self.lin(head, h)

    def local_topk(self, qi: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, vis: torch.Tensor, n: int, k: int,
                   pos_of) -> tuple[torch.Tensor, torch.Tensor]:
        """Per query the top-k of its visible rows of ``keys``: (int64 keys [Q, k], int32 rows [Q, k]), PAD_KEY padded.
        ``qi`` [Q, ih, id] and ``wts`` [Q, ih] carry the indexer's scales, ``n`` is the largest ``vis``, ``pos_of(rows)``
        gives positions."""

        Q = qi.shape[0]
        out_k = torch.full((Q, k), PAD_KEY, dtype=torch.int64, device=qi.device)
        out_r = torch.zeros((Q, k), dtype=torch.int32, device=qi.device)
        if n == 0:
            return out_k, out_r
        rows = torch.arange(n, device=qi.device)
        pos = pos_of(rows)
        per_head = F.relu(torch.einsum("qhd,td->qht", qi.float(), keys[:n].float()))
        score = torch.einsum("qh,qht->qt", wts.float(), per_head)
        key = score_keys(score, pos[None, :].expand(Q, n))
        key = key.masked_fill(rows[None, :] >= vis[:, None], PAD_KEY)
        m = min(k, n)
        top, at = key.topk(m, dim=1)
        out_k[:, :m], out_r[:, :m] = top, at.to(torch.int32)
        return out_k, out_r

    def attend(self, q_lat: torch.Tensor, q_pe: torch.Tensor | None, pool: KvPool, src: int, rows: torch.Tensor,
               cnt: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse MLA over each query's rows: (normalized o_lat [Q, H, kv_lora] fp32, lse [Q, H] fp32)."""

        Q, H, C = q_lat.shape
        o = torch.zeros((Q, H, C), dtype=F32, device=q_lat.device)
        lse = torch.full((Q, H), float("-inf"), dtype=F32, device=q_lat.device)
        for i in range(Q):
            c = int(cnt[i])
            if not c:
                continue
            lat, kpe = pool.gather(src, rows[i, :c].long())
            s = q_lat[i].float() @ lat.t()
            if kpe is not None:
                s = s + q_pe[i].float() @ kpe.t()
            s = s * sm_scale
            lse[i] = torch.logsumexp(s, -1)
            o[i] = torch.softmax(s, -1) @ lat
        return o, lse

    def local_topk_rows(self, qi: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, base: torch.Tensor,
                        vis: torch.Tensor, n: int, k: int, pos_of) -> tuple[torch.Tensor, torch.Tensor]:
        """``local_topk`` for rows of different streams: row r's keys start at local row ``base[r]``."""

        got = [self.local_topk(qi[r:r + 1], wts[r:r + 1], keys[int(base[r]):], vis[r:r + 1], n, k, pos_of)
               for r in range(qi.shape[0])]
        return torch.cat([g[0] for g in got]), torch.cat([g[1] for g in got])

    # -- tp.py's slices ------------------------------------------------------------------------------------------------

    def split_mat(self, mat, c0: int, c1: int):
        """Output columns c0..c1 of a weight (rows of a [N, K] tensor)."""

        return mat[c0:c1]

    def split_in(self, mat, k0: int, k1: int):
        """Input columns k0..k1 of a weight."""

        return mat[:, k0:k1]

    def split_ffn(self, lay, rank: int, world: int) -> tuple:
        """This rank's (gate, up, down) of the shared expert (the dense MLP on a dense layer): a quarter of its width."""

        gate, up, down = (lay.dense_gate, lay.dense_up, lay.dense_down) if lay.dense else lay.experts[-1]
        w = gate.shape[0] // world
        return gate[rank * w:(rank + 1) * w], up[rank * w:(rank + 1) * w], down[:, rank * w:(rank + 1) * w]

    def lin32(self, mod, x: torch.Tensor) -> torch.Tensor:
        return self.lin(mod, x)

    def ffn_part(self, t, x: torch.Tensor) -> torch.Tensor:
        return silu_expert(x.float(), t.gate, t.up, t.down)

    def moe_part(self, cp: "Ml4Cp", lay, t, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """This rank's share of the rows' experts: its routed picks and its slice of the shared expert."""

        if cp.w.split:
            return self.split_moe(cp, lay, x, idx, w)
        first = cp.rank * (cp.cfg.n_routed // cp.world)
        mine = (idx >= first) & (idx < first + cp.cfg.n_routed // cp.world)
        out = self.experts(cp, lay, x.to(self.wire), torch.where(mine, idx, torch.full_like(idx, -1)).to(torch.int32),
                           torch.where(mine, w, torch.zeros_like(w)), None)
        return out + self.ffn_part(t, x)

    def split_moe(self, cp: "Ml4Cp", lay, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """This rank's quarter of every pick and of the shared expert, for all rows: fp32 partial sums [Q, d]. Here the weights
        are whole, so the quarter is cut per call; ``CudaOps`` holds only its quarter."""

        pos = cp._expert_pos.get(lay.idx)
        if pos is None:
            pos = cp._expert_pos[lay.idx] = {int(e): i for i, e in enumerate(lay.expert_ids)}

        def quarter(row: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
            n = gate.shape[0] // cp.world
            a = cp.rank * n
            return silu_expert(row, gate[a:a + n], up[a:a + n], down[:, a:a + n])[0]

        out = torch.zeros((x.shape[0], cp.cfg.dim), dtype=F32, device=x.device)
        for r in range(x.shape[0]):
            row = x[r:r + 1].float()
            for s in range(idx.shape[1]):
                out[r] = out[r] + w[r, s] * quarter(row, *lay.experts[pos[int(idx[r, s])]])
            out[r] = out[r] + quarter(row, *lay.experts[-1])
        return out

    def absorb_q(self, q_nope: torch.Tensor, wuk: torch.Tensor) -> torch.Tensor:
        return torch.einsum("qhn,hnc->qhc", q_nope.float(), wuk.float())

    def expand_o(self, o_lat: torch.Tensor, wuv: torch.Tensor) -> torch.Tensor:
        return torch.einsum("qhc,hvc->qhv", o_lat.float(), wuv.float())

    def experts(self, cp: "Ml4Cp", lay, rows: torch.Tensor, eids: torch.Tensor, wts: torch.Tensor,
                shared_from: int | None) -> torch.Tensor:
        """Rows' routed share (their ``eids``, -1 skipped) and, from row ``shared_from`` on, the shared expert."""

        out = torch.zeros((rows.shape[0], cp.cfg.dim), dtype=F32, device=rows.device)
        trip = lay.experts
        pos = cp._expert_pos.get(lay.idx)
        if pos is None:
            pos = cp._expert_pos[lay.idx] = {int(e): i for i, e in enumerate(lay.expert_ids)}
        for r in range(rows.shape[0]):
            row = rows[r:r + 1].float()
            for s in range(eids.shape[1]):
                e = int(eids[r, s])
                if e >= 0:
                    out[r] = out[r] + wts[r, s] * silu_expert(row, *trip[pos[e]])[0]
            if shared_from is not None and r >= shared_from:
                out[r] = out[r] + silu_expert(row, *trip[-1])[0]
        return out


# -- the round forward (§3) -----------------------------------------------------------------------------------------------------

class Ml4Cp:
    """One rank's forward over ``KvPool`` rows: ``round`` runs a CP round (all ranks) or a DP round (this rank's rows;
    the group of one) through every layer and returns this rank's last-row logits when it owns the row asked for."""

    def __init__(self, w, comm, pool: KvPool, ops: Ops | None = None) -> None:
        self.w, self.cfg, self.comm, self.pool = w, w.cfg, comm, pool
        self.ops = ops or Ops()
        self.rank = comm.rank if comm is not None else 0
        self.world = comm.world if comm is not None else 1
        self.inv_freq = lerope_inv_freq(self.cfg, w.alpha.float()) if self.cfg.rd else None
        self._expert_pos: dict = {}

    # -- attention -----------------------------------------------------------------------------------------------------

    def _heads(self, lay) -> tuple[torch.Tensor, torch.Tensor]:
        """(W_UK [H, nope_w, kv_lora], W_UV [H, v, kv_lora]): views of kv_b, no folded copies (§10)."""

        c = self.cfg
        nope_w = c.qk_head - (c.rd if lay.rope else 0)
        w = lay.kv_b.view(c.n_heads, nope_w + c.v_head, c.kv_lora)
        return w[:, :nope_w], w[:, nope_w:]

    def _select(self, L: int, lay, qi_raw: torch.Tensor, wts: torch.Tensor, rnd: Round) -> tuple:
        """§3 steps 2-4: this rank's rows of the global top-K for every query of the round."""

        c, st, comm = self.cfg, rnd.seq.stripe, self.comm
        rope_w = c.rd if lay.rope else 0
        Q = rnd.qpos.numel()
        q = qi_raw.float().view(Q, c.idx_heads, c.idx_dim)
        q = ssmax_apply(q, lay.idx_ssmax_s, lay.idx_ssmax_b, rnd.qpos)
        q = rotate_leading(q, self.inv_freq, rnd.qpos, rope_w)
        # relu(a c) = c relu(a) for c > 0: the eager D^-0.5 and H^-0.5 fold into the head weights
        wts = wts.float() * (c.idx_heads ** -0.5) * (c.idx_dim ** -0.5)
        keys = self.pool.keys[L][rnd.seq.base:rnd.seq.base + rnd.seq.rows]
        rank = self.rank if st.world > 1 else 0
        cand, rows = self.ops.local_topk(q, wts, keys, rnd.vis, rnd.vis_max, c.idx_topk, lambda r: st.pos(rank, r))
        if st.world == 1:
            return filter_selection(cand, rows, None)
        offs = [sum(rnd.counts[:k]) for k in range(st.world)]
        sends = [cand[offs[k]:offs[k] + rnd.counts[k]].contiguous() for k in range(st.world)]
        mine = rnd.counts[self.rank]
        recvs = [torch.empty((mine, c.idx_topk), dtype=torch.int64, device=cand.device) for _ in range(st.world)]
        alltoall(comm, sends, recvs)
        thresh = kth_key(torch.cat(recvs, 1), c.idx_topk)
        return filter_selection(cand, rows, gather_rows(comm, thresh, rnd.counts))

    def _attention(self, L: int, lay, q_raw: torch.Tensor, rnd: Round) -> torch.Tensor:
        """§3 steps 5-6: this rank's rows of attention output [n_own, H * v] from all ranks' partials."""

        c, st, comm = self.cfg, rnd.seq.stripe, self.comm
        rope_w = c.rd if lay.rope else 0
        nope_w = c.qk_head - rope_w
        Q = rnd.qpos.numel()
        wuk, wuv = self._heads(lay)
        rows, cnt = rnd.sel
        rows = torch.where(rows == NO_ROW, rows, rows + rnd.seq.base)
        o, lse = [], []
        for a in range(0, Q, ATTN_ROWS):                      # a query's latents are 64 KB: a round's would be 256 MB
            at = slice(a, a + ATTN_ROWS)
            pos = rnd.qpos[at]
            q = ssmax_apply(q_raw[at].float().view(pos.numel(), c.n_heads, c.qk_head), lay.ssmax_s, lay.ssmax_b, pos)
            q_pe = rotate_leading(q[..., nope_w:], self.inv_freq, pos, rope_w) if rope_w else None
            q_lat = self.ops.absorb_q(q[..., :nope_w], wuk)
            part, part_lse = self.ops.attend(q_lat, q_pe, self.pool, lay.kv_source, rows[at], cnt[at], c.sm_scale)
            o.append(self.ops.expand_o(part, wuv).to(self.ops.wire))
            lse.append(part_lse)
        o, lse = torch.cat(o), torch.cat(lse)
        if st.world == 1:
            return o.reshape(Q, c.n_heads * c.v_head)
        n = rnd.counts[self.rank]
        offs = [sum(rnd.counts[:k]) for k in range(st.world)]
        s_o = [o[offs[k]:offs[k] + rnd.counts[k]].contiguous() for k in range(st.world)]
        s_l = [lse[offs[k]:offs[k] + rnd.counts[k]].contiguous() for k in range(st.world)]
        r_o = [torch.empty((n, c.n_heads, c.v_head), dtype=o.dtype, device=o.device) for _ in range(st.world)]
        r_l = [torch.empty((n, c.n_heads), dtype=F32, device=o.device) for _ in range(st.world)]
        alltoall(comm, s_o, r_o)
        alltoall(comm, s_l, r_l)
        return merge_partials(torch.stack(r_o), torch.stack(r_l)).reshape(n, c.n_heads * c.v_head)

    # -- one layer -----------------------------------------------------------------------------------------------------

    def _layer(self, L: int, x: torch.Tensor, rnd: Round) -> torch.Tensor:
        c, lay, ops, st = self.cfg, self.w.layers[L], self.ops, rnd.seq.stripe
        n = x.shape[0]
        active = st.world > 1 or n > 0                        # a DP rank without rows skips to the experts
        if active:
            rope_w = c.rd if lay.rope else 0
            xn = rms_norm(x, lay.attn_norm, c.eps)
            if lay.kv_source == L and n:
                ckv = ops.mat(lay.kv_a, xn)
                lat = rms_norm(ckv[:, :c.kv_lora], lay.kv_a_norm, c.eps)
                kpe = gptj_rotate(ckv[:, c.kv_lora:].float(), self.inv_freq, rnd.own) if rope_w else None
                self.pool.store(L, rnd.own_rows, lat, kpe)
            q = ops.lin(lay.q, xn).to(ops.wire)
            if lay.indexer == "full":
                if n:
                    k = layer_norm_bias(ops.mat(lay.idx_wk, xn), lay.idx_k_norm, lay.idx_k_norm_b, c.index_norm_eps)
                    k = rotate_leading(k.view(n, 1, 1, -1), self.inv_freq, rnd.own, rope_w).view(n, -1)
                    self.pool.keys[L][rnd.own_rows] = k.to(self.pool.keys[L].dtype)
                qi = ops.mat(lay.idx_wq_b, layer_norm_bias(xn, lay.idx_q_norm, lay.idx_q_norm_b,
                                                           c.index_norm_eps)).to(ops.wire)
                wts = ops.mat(lay.idx_proj, xn)
                if st.world > 1:
                    qi, wts = gather_rows(self.comm, qi, rnd.counts), gather_rows(self.comm, wts, rnd.counts)
                q_all = gather_rows(self.comm, q, rnd.counts) if st.world > 1 else q
                rnd.sel = self._select(L, lay, qi, wts, rnd)
            else:
                q_all = gather_rows(self.comm, q, rnd.counts) if st.world > 1 else q
            o = self._attention(L, lay, q_all, rnd)
            x = x + ops.lin(lay.o, o)
        xn2 = rms_norm(x, lay.ffn_norm, c.eps)
        if lay.dense:
            act = F.silu(ops.lin(lay.dense_gate, xn2).float()) * ops.lin(lay.dense_up, xn2).float()
            return x + ops.lin(lay.dense_down, act)
        return x + self._moe(lay, xn2, rnd)

    def _moe_split(self, lay, wire: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, rnd: Round) -> torch.Tensor:
        """Every rank runs its quarter of every expert over all the round's rows; a row's owner gets the four partials
        [W, n, d], to add in rank order. Sizes come from the round, so nothing here reads the device."""

        comm, counts, me = self.comm, rnd.counts, self.rank
        assert rnd.seq.stripe.world == comm.world, "a stream on one rank cannot use experts split over four"
        every = gather_rows(comm, wire, counts)
        picks = gather_rows(comm, torch.cat([idx.float(), w.float()], 1), counts)
        k = idx.shape[1]
        part = self.ops.split_moe(self, lay, every, picks[:, :k].to(torch.int32), picks[:, k:]).to(wire.dtype)
        offs = [sum(counts[:j]) for j in range(comm.world)]
        sends = [part[offs[j]:offs[j] + counts[j]].contiguous() for j in range(comm.world)]
        recvs = torch.empty((comm.world, *wire.shape), dtype=wire.dtype, device=wire.device)
        alltoall(comm, sends, list(recvs))
        return recvs

    def _moe(self, lay, xr: torch.Tensor, rnd: Round) -> torch.Tensor:
        c, ops, n = self.cfg, self.ops, xr.shape[0]
        w, idx = route_softmax(ops.router(lay.gate_w, xr), c.topk)
        wire = xr.to(ops.wire)
        if self.comm is None:
            return ops.experts(self, lay, wire, idx.to(torch.int32), w, 0)
        if self.w.split:
            parts = self._moe_split(lay, wire, idx, w, rnd)
            out = parts[0].float()
            for p in parts[1:]:
                out += p.float()
            return out
        routed = ep_dispatch(self.comm, wire, idx, w, c.n_routed,
                             lambda rows, eids, wts: ops.experts(self, lay, rows, eids, wts, None))
        return routed + ops.experts(self, lay, wire, idx[:, :0].to(torch.int32), w[:, :0], 0)

    # -- a round ------------------------------------------------------------------------------------------------------

    def begin(self, seq: Seq, start: int, n: int, tokens: torch.Tensor, last: int | None = None) -> Flight:
        """``tokens``: this rank's rows' ids."""

        rnd = make_round(seq, start, n, self.rank, tokens.device)
        assert tokens.numel() == rnd.own.numel(), f"{tokens.numel()} tokens for {rnd.own.numel()} rows"
        return Flight(rnd, self.ops.embed(self.w, tokens), last)

    def advance(self, f: Flight, layers: int, collect: list | None = None, sel_out: list | None = None) -> None:
        rnd = f.rnd
        for L in range(f.at, min(f.at + layers, self.cfg.n_layers)):
            f.x = self._layer(L, f.x, rnd)
            f.at = L + 1
            if collect is not None:
                collect.append((L, rnd.own.clone(), f.x.clone()))
            if sel_out is not None and self.w.layers[L].indexer == "full":
                sel_out.append((L, rnd.qpos.clone(), rnd.sel[0].clone(), rnd.sel[1].clone()))

    def open_slots(self, rows: int) -> None:
        pass

    def let_go(self, f: Flight) -> None:
        pass

    def finish(self, f: Flight) -> torch.Tensor | None:
        """Logits [V] of position ``last`` on its owner, else None."""

        if f.last is None:
            return None
        at = (f.rnd.own == f.last).nonzero()
        if at.numel() == 0:
            return None
        h = rms_norm(f.x[int(at[0, 0]):int(at[0, 0]) + 1], self.w.norm, self.cfg.eps)
        head = self.w.head if self.w.head is not None else self.w.head_bf16
        return softcap(self.ops.logits(head, h), self.cfg.softcap)[0]

    def round(self, seq: Seq, start: int, n: int, tokens: torch.Tensor, last: int | None = None,
              collect: list | None = None, sel_out: list | None = None) -> torch.Tensor | None:
        """A whole round: ``begin``, every layer, ``finish``."""

        f = self.begin(seq, start, n, tokens, last)
        self.advance(f, self.cfg.n_layers, collect, sel_out)
        return self.finish(f)
