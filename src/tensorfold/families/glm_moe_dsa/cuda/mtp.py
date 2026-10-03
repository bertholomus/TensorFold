"""The MTP head: reads final-normed main rows, chains its own shared_head.norm output, zeros position 0's embedding."""

from __future__ import annotations

import os
from typing import Sequence

import torch

from tensorfold.families.glm5_next.cuda.weights import Weights as FlashWeights  # noqa: F401  (type docs)

from . import glue, rope as rope_mod
from .forward import Buffers, State, check_room, dsa_block, embed, mm, moe_block, residual
from .weights import Weights


# GLM-5.3's MTP layer is a full indexer layer, and the checkpoint asks for its selection to be shared over a draft
# chain (index_share_for_mtp_iteration): a chain's first step (the absorbed rows) scores and selects, the later
# one-row steps attend the first step's last row's tokens. TF_GLM_MTP_REUSE: 1 = reuse plus the row's own token in the
# list's spare slot, 2 = reuse alone, 0 = every step selects (drafts are proposals: the reply is the same either way)
MTP_REUSE = os.environ.get("TF_GLM_MTP_REUSE") or "0"


def reuse_rows(st: State, b: Buffers) -> None:
    """Row 0 of b.tokens / b.counts takes the last MTP step's last row's selection; mode 1 puts the row's own position
    (st.mtp_pos_dev) in the list's spare slot when that selection is a full top-k (past the dense limit)."""

    src = getattr(b, "sel_row", 0)
    if src:
        b.tokens[0].copy_(b.tokens[src])
        b.counts[0].copy_(b.counts[src])
    if MTP_REUSE == "1" and getattr(st, "dcp", 1) == 1:      # (dcp: the reused list alone, no own token)
        W = b.tokens.shape[1]
        full = b.counts[0:1] >= W - 1
        b.tokens[0, W - 1:W].copy_(torch.where(full, st.mtp_pos_dev, b.tokens[0, W - 1:W]))
        b.counts[0:1].copy_(torch.where(full, torch.full_like(b.counts[0:1], W), b.counts[0:1]))


def mtp_stage(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], hidden: torch.Tensor) -> int:
    """Host work before an MTP step: the next tokens and the input hidden rows into the static buffers."""

    n = len(next_tokens)
    b.zero_first = st.mtp_len == 0
    check_room(w, st, n, pos=st.mtp_len)
    b.staged.synchronize()
    b.ids_host[:n].numpy()[:] = list(next_tokens)
    b.ids[:n].copy_(b.ids_host[:n], non_blocking=True)
    if hidden.data_ptr() != b.hin.data_ptr():
        b.hin[:n].copy_(hidden)
    b.staged.record()
    return n


def mtp_compute(w: Weights, st: State, b: Buffers, n: int, *, last_only: bool = True,
                nch: int | None = None, host_pos: int | None = None, sparse_np: int | None = None,
                reuse: bool = False) -> torch.Tensor:
    """The MTP head's GPU work on staged rows (capturable); ``reuse``: a chained one-row draft step (MTP_REUSE)."""

    c = w.cfg
    m = w.mtp
    D = c.hidden
    embed(w, b, b.ids[:n], b.me[:n])
    if b.zero_first:
        b.me[0].zero_()
    glue.rmsnorm(b.me[:n], m.enorm, c.eps, b.mcat[:n, :D])
    glue.rmsnorm(b.hin[:n], m.hnorm, c.eps, b.mcat[:n, D:])
    mm(b, b.mcat[:n], m.eh, None if b.prefill else _group_sums(b, b.mcat[:n]), b.mx[:n])
    rope_mod.table(b.cos[:n], b.sin[:n], st.mtp_pos_dev, n, c.rope_theta, c.qk_rope)
    layer = m.layer
    glue.rmsnorm(b.mx[:n], layer.in_norm, c.eps, b.normed[:n], b.xs[:n])
    if reuse:
        reuse_rows(st, b)
    g = dsa_block(layer, w, st.mtp_kc, st.mtp_pos_dev, b, n, nch,
                  st.index[-1] if getattr(st, "index", None) is not None else None, host_pos, sparse_np,
                  pc=st.mtp_pc, reuse=reuse)
    b.sel_row = 0 if reuse else n - 1
    residual(b.mx[:n], b.mx[:n], g)
    glue.rmsnorm(b.mx[:n], layer.post_norm, c.eps, b.normed[:n], b.xs[:n])
    g = moe_block(layer, w, b, n)
    residual(b.mx[:n], b.mx[:n], g)
    lo = n - 1 if last_only else 0
    k = n - lo
    glue.rmsnorm(b.mx[lo:n], m.norm, c.eps, b.fnormed[:k], b.fxs[:k])
    head = w.head if w.draft_head is None else w.draft_head         # TF_GLM_DRAFT_VOCAB: the draft ids only
    return mm(b, b.fnormed[:k], head, b.fxs[:k], b.logits[:k, :head.n])


def _group_sums(b: Buffers, x: torch.Tensor):
    from tensorfold.families.glm5_next.cuda import qmm

    return None if b.prefill else qmm.group_sums(x, b.mxs[:x.shape[0]])


@torch.no_grad()
def mtp_forward(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], hidden: torch.Tensor,
                *, last_only: bool = True, reuse: bool = False) -> torch.Tensor:
    """Write hidden/token rows into cache slots mtp_len onward and expose logits and b.mx; the caller advances st.mtp_len."""

    n = mtp_stage(w, st, b, next_tokens, hidden)
    from tensorfold.families.glm5_next.cuda.attention import CHUNK

    return mtp_compute(w, st, b, n, last_only=last_only, nch=-(-(st.mtp_len + n) // CHUNK), host_pos=st.mtp_len,
                       reuse=reuse)
