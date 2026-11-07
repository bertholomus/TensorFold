"""tp.py's step on the GPU with its glue fused (tp_kernels.py): the same sum, about a quarter of the launches, the buffers
of the collectives written and read in place. ``Ml4Tp`` stays the reference; a rank's numbers differ from it only by the
order of reductions."""
from __future__ import annotations

import os

import torch

from tensorfold.cuda.exl3 import prefetch
from tensorfold.cuda.exl3.linear import Exl3Group

from . import cp4
from .cp4_cuda import Exl3Part, Fp8Block
from .cp4_kernels import fp8_linear, small_mat
from .model import softcap
from .tp import Ml4Tp, Rows
from .tp_kernels import (add_norm, attend_step, key_row, kv_row, layer_norm16, local_top, merge_expand, own_rows, pick_rows, q_absorb,
                         q_rope, qi_prep, route, rows_of, score_step, select_ready, silu_mul)

F16, F32 = torch.float16, torch.float32
RDMA_KB = int(os.environ.get("TF_ML4_RDMA_KB") or 1024)               # a step's largest gather: 8 rows of fp32 partials, 192 KB
# weights the next kernels read, pulled into L2 on a side stream while gathers and attention leave DRAM idle: MiB across
# attention's span, across the gather after o, across the gather after the experts (L2 is 24 MB). 0,0,0: no side stream
L2_MB = [int(x) for x in (os.environ.get("TF_ML4_L2_MB") or "0,0,0").split(",")]


def fast_comm(comm, log=print):
    """A step's gathers over RDMA writes when every rank can open them (tensorfold.cuda.rdma), the rest over NCCL."""

    if comm is None or os.environ.get("TF_ML4_RDMA", "1") == "0":
        return comm
    from tensorfold.cuda.rdma import Hybrid, RdmaGather

    try:
        return Hybrid(comm, RdmaGather(comm.store, comm.rank, comm.world, max_bytes=RDMA_KB << 10))
    except Exception as exc:                                     # noqa: BLE001  (every rank sees the same refusal)
        log(f"[ml4-engine] rank {comm.rank}: gathers stay on NCCL: {exc}")
        return comm


def spans(m) -> list:
    """(address, bytes) of what a step reads of ``m``: an input slice of a layer is one run a strip of 128 columns."""

    if m is None:
        return []
    if isinstance(m, Fp8Block):
        return spans(m.w)
    if isinstance(m, torch.Tensor):
        if m.is_contiguous():
            return [(m.data_ptr(), m.numel() * m.element_size())]
        return [r for h in m for r in spans(h)] if m.dim() > 2 else []            # W_UK, W_UV: a run a head
    if not isinstance(m, Exl3Part):
        return [(m.words.data_ptr(), m.words.numel() * 4)]
    if m.layout != "strips":
        return []
    ks, nbs = m.strides
    run, strips, at = m.k // 16 * ks * 4, m.n // 128, m.words.data_ptr()
    return [(at, run * strips)] if nbs * 4 == run else [(at + i * nbs * 4, run) for i in range(strips)]


def cut(ranges: list, mb: int) -> tuple:
    """The first ``mb`` MiB of ``ranges`` as launches of the prefetch kernel, and the ranges left."""

    left, took = mb << 20, []
    ranges = list(ranges)
    while ranges and left >= 16:
        at, n = ranges.pop(0)
        if n > left:
            ranges.insert(0, (at + left // 16 * 16, n - left // 16 * 16))
            n = left
        if at % 16 == 0 and n >= 16:
            took.append((at, n // 16 * 16))
            left -= n // 16 * 16
    cap = prefetch._ext().l2_prefetch_max()
    return [([a for a, _ in took[i:i + cap]], [n for _, n in took[i:i + cap]]) for i in range(0, len(took), cap)], ranges


class FusedTp(Ml4Tp):
    def __init__(self, fwd: cp4.Ml4Cp) -> None:
        super().__init__(fwd)
        self.pairs = [Exl3Group([t.gate, t.up]) for t in self.layers]
        self.part_of_table()                                     # before memory is planned
        select_ready()
        self.side = prefetch.SideStream() if any(L2_MB) else None
        self.beside = [torch.cuda.Stream(priority=-1) for _ in range(2)]
        self.ahead = [self._ahead(L) for L in range(self.cfg.n_layers)] if self.side else []

    def _ahead(self, L: int) -> list:
        """Layer L's three prefetches: in the order the kernels after each one read them."""

        lay, t = self.w.layers[L], self.layers[L]
        after = [r for m in (t.wuv, t.o, lay.gate_w, t.gate, t.up, t.down) for r in spans(m)]
        attn, after = cut(after, L2_MB[0])
        ffn, _ = cut(after, L2_MB[1])
        if L + 1 == self.cfg.n_layers:
            nxt = [self.head]
        else:
            n, tn = self.w.layers[L + 1], self.layers[L + 1]
            nxt = [tn.idx_wq_b, tn.q, tn.wuk, n.idx_wk, n.idx_proj, n.kv_a if n.kv_source == L + 1 else None]
        return [attn, ffn, cut([r for m in nxt for r in spans(m)], L2_MB[2])[0]]

    def _fetch(self, L: int, at: int) -> None:
        if self.side is not None:
            for r in self.ahead[L][at]:
                self.side.fork(r)

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        return rows_of(self._gather(own_rows(ids, self.table, self.rank * self.vl)), ids, self.vl)

    def _gather(self, send: torch.Tensor) -> torch.Tensor:
        every = torch.empty((self.world, *send.shape), dtype=send.dtype, device=send.device)
        self.comm.all_gather(send.view(-1).view(F32), every.view(-1).view(F32))      # any dtype: the RDMA gather takes fp32
        return every

    def _ffn(self, L: int, x16: torch.Tensor) -> torch.Tensor:
        t = self.layers[L]
        g, u = self.pairs[L]([x16, x16], None, [F32, F32])
        return t.down(silu_mul(g, u), out_dtype=F32)

    def _select(self, L: int, every: torch.Tensor, at: int, wts: torch.Tensor, s: Rows) -> tuple:
        c, st = self.cfg, self.stripe
        score = torch.empty((s.pos.numel(), s.n), dtype=F32, device=wts.device)
        score_step(every, at, wts, (c.idx_heads ** -0.5) * (c.idx_dim ** -0.5), self.il, c.idx_dim, self.pool.keys[L], s.base,
                   s.vis, s.n, score)
        key, rows = local_top(score, s.vis, c.idx_topk, st, self.rank)
        return pick_rows(self._gather(key), rows, self.rank)

    def forward(self, x: torch.Tensor, s: Rows) -> torch.Tensor:
        c, pool, ops, inv = self.cfg, self.pool, self.ops, self.inv_freq
        R, dev, hl, il = x.shape[0], x.device, self.hl, self.il
        C = c.kv_lora
        x, sel, every = x.clone(), None, None
        whole = self._whole(s) if not s.indexed else None
        stride = R * hl * C + 2 * R * hl
        for L in range(c.n_layers):
            lay, t = self.w.layers[L], self.layers[L]
            rope_w = c.rd if lay.rope else 0
            full = lay.indexer == "full"
            idx = full and s.indexed
            x, xn, x16 = add_norm(every, x, lay.attn_norm, c.eps)
            self.tap(L, x)
            q_at = hl * (C + rope_w)
            send = torch.empty((R, q_at + (il * c.idx_dim if idx else 0)), dtype=F16, device=dev)
            here, rows, index = torch.cuda.current_stream(), self.beside[0], self.beside[1]
            rows.wait_stream(here)
            with torch.cuda.stream(rows):                        # three short chains, none reading another's
                if lay.kv_source == L:
                    kv_row(small_mat(xn, lay.kv_a), lay.kv_a_norm, s.pos, s.wr, inv, pool.lat[L], c.eps, C, rope_w)
                if full:
                    key_row(small_mat(xn, lay.idx_wk), lay.idx_k_norm, lay.idx_k_norm_b, s.pos, s.wr, inv, pool.keys[L],
                            c.index_norm_eps, c.idx_dim, rope_w)
                wts = small_mat(xn, lay.idx_proj) if idx else None
            if idx:
                index.wait_stream(here)
                with torch.cuda.stream(index):
                    qi = fp8_linear(layer_norm16(xn, lay.idx_q_norm, lay.idx_q_norm_b, c.index_norm_eps), t.idx_wq_b.w,
                                    t.idx_wq_b.scale)
                    qi_prep(qi, s.pos, t.idx_ssmax_s, t.idx_ssmax_b, inv, il, c.idx_dim, rope_w, send[:, q_at:])
            q = t.q(x16, out_dtype=F16)
            q_absorb(q, s.pos, t.ssmax_s, t.ssmax_b, t.wuk, c.qk_head, send)
            if rope_w:
                q_rope(q, s.pos, t.ssmax_s, t.ssmax_b, inv, hl, c.qk_head, rope_w, send[:, hl * C:])
            here.wait_stream(rows)
            if idx:
                here.wait_stream(index)
            self._fetch(L, 0)
            queries = self._gather(send)
            if idx:
                sel = self._select(L, queries, q_at, wts, s)
            elif full:
                sel = whole
            blocks = torch.empty((self.world, stride), dtype=F16, device=dev)
            attend_step(queries, hl, rope_w, sel[0], sel[1], s.base, pool.lat[lay.kv_source], c.sm_scale, blocks)
            got = torch.empty_like(blocks)
            cp4.alltoall(self.comm, list(blocks), list(got))
            o = merge_expand(got, t.wuv, R, torch.empty((R, hl * c.v_head), dtype=F16, device=dev))
            part = t.o(o, out_dtype=F32)
            self._fetch(L, 1)
            every = self._gather(part)
            x, xn, x16 = add_norm(every, x, lay.ffn_norm, c.eps)
            if lay.dense:
                part = self._ffn(L, x16)
            else:
                ex, shared = lay.experts, torch.cuda.Event()
                logits = small_mat(xn, lay.gate_w)
                rows.wait_stream(here)
                with torch.cuda.stream(rows):                    # the shared expert beside the routed ones' reads
                    part = self._ffn(L, x16)
                    shared.record(rows)
                pick, wts = route(logits, ops._lut(self.fwd, lay, dev), c.topk, ex.count)
                part = ops._routed_add(ex, x16, pick, wts, c.topk, part, shared)
            self._fetch(L, 2)
            every = self._gather(part)
        x, _, x16 = add_norm(every, x, self.w.norm, c.eps)
        self.tap(c.n_layers, x)
        if self.side is not None:
            self.side.join()
        return softcap(self.head(x16, out_dtype=F32), c.softcap)

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        return softcap(self.head(h.to(F16), out_dtype=F32), self.cfg.softcap)
