"""Ling-3.1-flash's MTP drafter: layer 72 (an MLA block and the full MoE) on final-normed main rows, drafting
through the shared head (modeling 925-983, 1413-1448).

Row j of a step (MTP slot ``st.mtp_len + j``, RoPE position the same) reads the main model's final-normed hidden
row and the next token: ``x = eh_proj(cat[enorm(embed(t)), hnorm(h)])`` (the embedding first), then the MLA block
on its own latent and rope-key caches, the MoE, and ``final_layernorm``, whose output gives the draft logits through
the main head and is the next chained step's hidden row (draft steps 2 and 3 read it at slots i + 1 and i + 2).
Nothing is zeroed at position 0 (the reference zeroes nothing) and nothing is clamped (the reference clamps nothing).
"""

from __future__ import annotations

from typing import Sequence

import torch

from tensorfold.families.glm5_next.cuda import glue, latent

from .forward import Buffers, State, check_room, embed, mla_block, mm, moe_block, residual
from .weights import Weights


def mtp_stage(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], hidden: torch.Tensor) -> int:
    """Host work before an MTP step: the next tokens and the input hidden rows into the static buffers."""

    n = len(next_tokens)
    if n > b.rows:
        raise ValueError(f"MTP step of {n} rows, buffers hold {b.rows}")
    check_room(w, st, n, pos=st.mtp_len)
    b.staged.synchronize()
    b.ids_host[:n].numpy()[:] = list(next_tokens)
    b.ids[:n].copy_(b.ids_host[:n], non_blocking=True)
    if hidden.data_ptr() != b.hin.data_ptr():
        b.hin[:n].copy_(hidden)
    b.staged.record()
    return n


def mtp_compute(w: Weights, st: State, b: Buffers, n: int, *, last_only: bool = True, nch: int | None = None,
                host_pos: int | None = None) -> torch.Tensor:
    """The MTP layer's GPU work on staged rows (capturable): draft logits of the last row (or of every row), and
    b.fnormed[:k] the final_layernorm rows they came from (the next chained step's hidden rows)."""

    c = w.cfg
    m = w.mtp
    D = c.hidden
    embed(w, b, b.ids[:n], b.me[:n])
    glue.rmsnorm(b.me[:n], m.enorm, c.eps, b.mcat[:n, :D])
    glue.rmsnorm(b.hin[:n], m.hnorm, c.eps, b.mcat[:n, D:])
    mm(b, b.mcat[:n], m.eh, None, b.mx[:n])
    b.rope.fill(st.mtp_pos_dev, n)                 # slot j's position: the main position j (device: graph-safe)
    layer = m.layer
    glue.rmsnorm(b.mx[:n], layer.in_norm, c.eps, b.normed[:n])
    g = mla_block(layer.mla, w, st.mtp_kc, st.mtp_pc, st.mtp_pos_dev, b, n, nch, host_pos)
    residual(b.mx[:n], b.mx[:n], g)
    glue.rmsnorm(b.mx[:n], layer.post_norm, c.eps, b.normed[:n])
    g = moe_block(layer, w, b, n)
    residual(b.mx[:n], b.mx[:n], g)
    lo = n - 1 if last_only else 0
    k = n - lo
    glue.rmsnorm(b.mx[lo:n], m.norm, c.eps, b.fnormed[:k])            # final_layernorm
    if k > b.logits.shape[0]:
        raise ValueError(f"{k} MTP logit rows, buffers hold {b.logits.shape[0]} (prompt buffers: last_only)")
    return mm(b, b.fnormed[:k], w.head, None, b.logits[:k])


@torch.no_grad()
def mtp_forward(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], hidden: torch.Tensor,
                *, last_only: bool = True) -> torch.Tensor:
    """Write hidden / token rows into MTP cache slots mtp_len onward and return their draft logits (b.fnormed keeps
    the final_layernorm rows); the caller advances st.mtp_len."""

    n = mtp_stage(w, st, b, next_tokens, hidden)
    return mtp_compute(w, st, b, n, last_only=last_only, nch=latent.chunks_for(st.mtp_len + n),
                       host_pos=st.mtp_len)
