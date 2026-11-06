"""tp.py's step on the GPU with its glue fused (tp_kernels.py): the same sum, about a quarter of the launches, the buffers
of the collectives written and read in place. ``Ml4Tp`` stays the reference; a rank's numbers differ from it only by the
order of reductions."""
from __future__ import annotations

import os

import torch

from tensorfold.cuda.exl3.linear import Exl3Group

from . import cp4
from .cp4_kernels import fp8_linear, small_mat
from .model import softcap
from .tp import Ml4Tp, Rows
from .tp_kernels import (add_norm, attend_step, key_row, kv_row, layer_norm16, local_top, merge_expand, pick_rows, q_absorb, q_rope,
                         qi_prep, route, score_step, select_ready, silu_mul)

F16, F32 = torch.float16, torch.float32
RDMA_KB = int(os.environ.get("TF_ML4_RDMA_KB") or 1024)               # a step's largest gather: 8 rows of fp32 partials, 192 KB


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


class FusedTp(Ml4Tp):
    def __init__(self, fwd: cp4.Ml4Cp) -> None:
        super().__init__(fwd)
        self.pairs = [Exl3Group([t.gate, t.up]) for t in self.layers]
        select_ready()

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
            if lay.kv_source == L:
                kv_row(small_mat(xn, lay.kv_a), lay.kv_a_norm, s.pos, s.wr, inv, pool.lat[L], c.eps, C, rope_w)
            q_at = hl * (C + rope_w)
            send = torch.empty((R, q_at + (il * c.idx_dim if idx else 0)), dtype=F16, device=dev)
            q = t.q(x16, out_dtype=F16)
            q_absorb(q, s.pos, t.ssmax_s, t.ssmax_b, t.wuk, c.qk_head, send)
            if rope_w:
                q_rope(q, s.pos, t.ssmax_s, t.ssmax_b, inv, hl, c.qk_head, rope_w, send[:, hl * C:])
            if full:
                key_row(small_mat(xn, lay.idx_wk), lay.idx_k_norm, lay.idx_k_norm_b, s.pos, s.wr, inv, pool.keys[L],
                        c.index_norm_eps, c.idx_dim, rope_w)
            if idx:
                qi = fp8_linear(layer_norm16(xn, lay.idx_q_norm, lay.idx_q_norm_b, c.index_norm_eps), t.idx_wq_b.w,
                                t.idx_wq_b.scale)
                qi_prep(qi, s.pos, t.idx_ssmax_s, t.idx_ssmax_b, inv, il, c.idx_dim, rope_w, send[:, q_at:])
            queries = self._gather(send)
            if idx:
                sel = self._select(L, queries, q_at, small_mat(xn, lay.idx_proj), s)
            elif full:
                sel = whole
            blocks = torch.empty((self.world, stride), dtype=F16, device=dev)
            attend_step(queries, hl, rope_w, sel[0], sel[1], s.base, pool.lat[lay.kv_source], c.sm_scale, blocks)
            got = torch.empty_like(blocks)
            cp4.alltoall(self.comm, list(blocks), list(got))
            o = merge_expand(got, t.wuv, R, torch.empty((R, hl * c.v_head), dtype=F16, device=dev))
            every = self._gather(t.o(o, out_dtype=F32))
            x, xn, x16 = add_norm(every, x, lay.ffn_norm, c.eps)
            part = self._ffn(L, x16)
            if not lay.dense:
                ex = lay.experts
                pick, wts = route(small_mat(xn, lay.gate_w), ops._lut(self.fwd, lay, dev), c.topk, ex.count)
                part = ops._routed_add(ex, x16, pick, wts, c.topk, part)
            every = self._gather(part)
        x, _, x16 = add_norm(every, x, self.w.norm, c.eps)
        return softcap(self.head(x16, out_dtype=F32), c.softcap)
