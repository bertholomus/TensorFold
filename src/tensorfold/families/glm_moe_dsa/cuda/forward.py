"""GLM-5.3's tensor-parallel forward: Flash's row-invariant kernels, plain residuals, interleaved rope, raw-token DSA."""

from __future__ import annotations

from typing import Sequence

import torch
import triton

from tensorfold.families.glm5_next.cuda.qmm import B16

from tensorfold.families.glm5_next.cuda import latent as latent_mod, prof

from . import glue, rope as rope_mod, select as select_mod
from .weights import LayerW, Weights

# the Flash buffers' DSA block is reused where the shapes match; the indexer differs (no k-pool)
from tensorfold.families.glm5_next.cuda.forward import (  # noqa: F401
    Buffers as FlashBuffers, State as FlashState, gather, mm, out_proj,
)


class Buffers(FlashBuffers):
    """Flash's buffers plus GLM-5.3's rope cos/sin rows and plain-residual state (no stream copies)."""

    def __init__(self, w: Weights, rows: int, capacity: int = 2560, *, prefill: bool = False) -> None:
        super().__init__(w, rows, capacity, prefill=prefill)
        c = w.cfg
        dev = w.device
        bf = torch.bfloat16
        # raw-token selection list: index_topk entries plus the current row's tail
        self.tokens = torch.empty((rows, c.index_topk + 1), dtype=torch.int32, device=dev)
        self.counts = torch.empty((rows,), dtype=torch.int32, device=dev)
        self.cos = torch.empty((rows, c.qk_rope // 2), dtype=torch.float32, device=dev)
        self.sin = torch.empty((rows, c.qk_rope // 2), dtype=torch.float32, device=dev)
        self.ik = torch.empty((rows, c.index_dim), dtype=bf, device=dev)          # the window's indexer keys
        # no hyper-connections: x holds one stream
        self.x = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        del self.taps, self.tap_at


class State(FlashState):
    """Latent caches per DSA layer (plus the MTP layer's); no KDA state, indexer planes per full layer."""

    def __init__(self, w: Weights, capacity: int, rows: int) -> None:
        c = w.cfg
        dev = w.device
        self.capacity = capacity
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.mtp_pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        dsa_layers = [l for l in w.layers if l.kind == "dsa"]
        self.dsa_index = {l.index: i for i, l in enumerate(dsa_layers)}
        self.latent = latent_mod.ENABLED
        self.kc = [torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
        self.vc = [None for _ in dsa_layers]
        self.mtp_len = 0
        self.mtp_drafted = 0
        if w.mtp is not None:
            self.mtp_kc = torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev)
            self.mtp_vc = None
        # one indexer key plane per full-indexer group (and the MTP layer, last): a "shared" layer
        # reads the nearest preceding full layer's keys, so a group shares one plane
        self.index = None
        self.index_slot = {}
        if w.meta.get("long_context"):
            slot = -1
            for i in range(c.layers):
                if c.indexer_types[i] == "full":
                    slot += 1
                self.index_slot[i] = slot
            n_idx = slot + 1 + (1 if w.mtp is not None else 0)
            self.index = [torch.zeros((capacity, c.index_dim), dtype=torch.bfloat16, device=dev)
                          for _ in range(n_idx)]

    def reset(self) -> None:
        self.set_pos(0)
        self.set_mtp_len(0)
        self.mtp_drafted = 0

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        self.pos_dev.fill_(pos)

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n
        self.mtp_pos_dev.fill_(n)

    @property
    def parity(self) -> int:
        return 0

    def clone(self) -> "State":
        import copy

        other = copy.copy(self)
        other.pos_dev = self.pos_dev.clone()
        other.mtp_pos_dev = self.mtp_pos_dev.clone()
        other.kc = [x.clone() for x in self.kc]
        other.vc = [None for _ in self.kc]
        if self.index is not None:
            other.index = [x.clone() for x in self.index]
        if hasattr(self, "mtp_kc"):
            other.mtp_kc = self.mtp_kc.clone()
            other.mtp_vc = None
        return other


def dsa_block(layer: LayerW, w: Weights, lc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers, R: int,
              nch: int | None, index: torch.Tensor | None, host_pos: int | None,
              sparse_np: int | None = None) -> torch.Tensor:
    """MLA over the latent cache with GLM-5.3's rope, indexer update on full layers, raw-token sparse selection.

    A "shared" indexer layer passes ``index=None``: its selection reuse is expressed by scoring and
    selecting in the owning full layer (which runs the same rows either just before or within the
    same window), so the shared layer reads that layer's tokens from the buffers below.
    """

    c = w.cfg
    a = layer.dsa
    HL = a.heads
    s = b.lat_s
    with prof.timed("dsa: projections"):
        mm(b, b.normed[:R], a.q_a, b.xs[:R], b.qr[:R])
        glue.rmsnorm(b.qr[:R], a.q_norm, c.eps, b.qr[:R], b.xs_qr[:R])
        mm(b, b.normed[:R], a.kv_a, b.xs[:R], b.lat[:R])
        glue.rmsnorm(b.lat[:R], a.kv_norm, c.eps, b.lat[:R], b.xs_lat[:R])
    with prof.timed("dsa: rope"):
        rope_mod.apply(b.qr[:R], b.lat[:R, c.kv_lora:], b.cos[:R], b.sin[:R], c)
    with prof.timed("dsa: q_b"):
        mm(b, b.qr[:R], a.q_b, b.xs_qr[:R], b.q[:R].view(R, HL * c.qk_dim))
    with prof.timed("dsa: latent write"):
        latent_mod.latent_write(b.lat[:R, :c.kv_lora], lc, pos_dev)
    all_sparse = sparse_np is not None or (host_pos is not None and host_pos >= c.dense_limit)
    sparse_rows = index is not None and (all_sparse or (host_pos is not None and host_pos + R - 1 >= c.dense_limit))
    if index is not None:
        with prof.timed("dsa: indexer update"):
            ix = a.index
            mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ik[:R])
            glue.layernorm(b.ik[:R], ix.ln_w, ix.ln_b, c.eps, b.ik[:R])
            at = host_pos if host_pos is not None else int(pos_dev.item())
            index[at:at + R].copy_(b.ik[:R])
    with prof.timed("dsa: absorb"):
        qa = latent_mod.absorb_q(b.q[:R], a.absorb, s.qa[:R])
    ol = s.ol[:R]
    scale = c.qk_dim ** -0.5
    if not all_sparse:
        with prof.timed("dsa: dense attention"):
            latent_mod.attention(qa, lc, pos_dev, s, scale=scale, nch=min(nch or s.nch, s.nch), out=ol)
    if sparse_rows:
        with prof.timed("dsa: select tokens"):
            mm(b, b.qr[:R], a.index.qb, b.xs_qr[:R], b.qi[:R])
            select_mod.select_tokens(b.qi[:R], a.index.weights, index, host_pos, R, c.index_topk, pos_dev,
                                     tokens=b.tokens[:R], counts=b.counts[:R], bucket=sparse_np)
        with prof.timed("dsa: sparse attention"):
            latent_mod.sparse_attention(qa, lc, b.tokens[:R], b.counts[:R], ol, scale)
    with prof.timed("dsa: expand"):
        o = latent_mod.expand_v(ol, a.absorb, b.vn[:R]).view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, qmm_sums(b, o, R), R)


def qmm_sums(b: Buffers, o: torch.Tensor, R: int):
    from tensorfold.families.glm5_next.cuda import qmm

    return None if b.prefill else qmm.group_sums(o, b.xs_ao[:R])


def mlp_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    m = layer.mlp
    mm(b, b.normed[:R], m.gu, b.xs[:R], b.gu[:R])
    glue.silu_mul(b.gu[:R], b.act[:R], b.xs_act[:R])
    return out_proj(w, b, b.act[:R], m.down, b.xs_act[:R], R)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    from tensorfold.cuda import experts as grouped
    from tensorfold.families.glm5_next.cuda import exl3_mm

    c = w.cfg
    m = layer.moe
    with prof.timed("moe: route"):
        glue.router(b.normed[:R], m.router, b.mlog[:R])
        glue.select(b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], c.top_k, c.experts, c.routed_scale, c.norm_topk)
        grouped.route(b.pick[:R], b.plan)
    # GLM-5.3 clamps nothing (no swiglu_limit): the Flash EXL3 kernel's clamp passes with a bound nothing reaches
    exl3_mm.routed(b.normed[:R], b.pick, b.plan, m.experts, b.exl3, b.ey.view(-1, c.hidden), R, 1e30)
    s = m.shared
    mm(b, b.normed[:R], s.gu, b.xs[:R], b.sgu[:R])
    glue.silu_mul(b.sgu[:R], b.sact[:R], b.sxs[:R])
    mm(b, b.sact[:R], s.down, b.sxs[:R], b.sy[:R], f32=True)
    b.ey[:R, c.top_k].copy_(b.sy[:R])
    glue.combine(b.ey[:R], b.wts[:R], b.part[:R])
    with prof.timed("moe: all-gather"):
        return gather(w, b, R)


def layer_forward(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None = None,
                  host_pos: int | None = None, sparse_np: int | None = None) -> None:
    di = st.dsa_index[layer.index]
    with prof.timed("dsa (total)"):
        # a "shared" indexer layer scores and selects nothing of its own: it reuses the owning full
        # layer's selection, which the same window computed one block earlier into the buffers
        idx = st.index[st.index_slot[layer.index]] if getattr(st, "index", None) is not None \
            and layer.dsa.index is not None else None
        g = dsa_block(layer, w, st.kc[di], st.pos_dev, b, R, nch, idx, host_pos, sparse_np)
    glue.residual_add(b.x[:R], b.x[:R], g)
    with prof.timed("moe (total)" if layer.mlp is None else "mlp"):
        g = mlp_block(layer, w, b, R) if layer.mlp is not None else moe_block(layer, w, b, R)
    glue.residual_add(b.x[:R], b.x[:R], g)


def check_room(w: Weights, st: State, R: int, pos: int | None = None) -> None:
    pos = st.pos if pos is None else pos
    if pos + R > w.cfg.dense_limit and st.index is None:
        raise ValueError(f"context {pos + R} past {w.cfg.dense_limit} tokens: this engine was started without long "
                         "contexts (DSA's sparse top-k)")
    if pos + R > st.capacity:
        raise ValueError("context past the cache capacity")


def stage(w: Weights, st: State, b: Buffers, tokens: Sequence[int]) -> int:
    """Host work before a forward: the token ids into the static device buffer (pinned copy)."""

    R = len(tokens)
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    check_room(w, st, R)
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = list(tokens)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()
    return R


def compute(w: Weights, st: State, b: Buffers, R: int, *, logits: bool = True, nch: int | None = None,
            host_pos: int | None = None, sparse_np: int | None = None):
    """Run capturable GPU work on static buffers and device positions."""

    from tensorfold.families.glm5_next.cuda import glue as fglue

    c = w.cfg
    fglue.embed(b.ids[:R], w.embed, c.hidden, 1, b.x[:R])
    rope_mod.table(b.cos[:R], b.sin[:R], st.pos, R, c.rope_theta, c.qk_rope)
    for layer in w.layers:
        layer_forward(layer, w, st, b, R, nch, host_pos, sparse_np)
    if not logits:
        return None
    glue.rmsnorm(b.x[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    if b.prefill:                        # the head reads the last row only (fnormed keeps every row for the MTP)
        return mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])
    return mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])


def chunks_for(st: State, R: int) -> int:
    from tensorfold.families.glm5_next.cuda.attention import CHUNK

    return -(-(st.pos + R) // CHUNK)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True) -> torch.Tensor | None:
    """Return logits for token rows, leaving committed state unchanged until commit."""

    R = stage(w, st, b, tokens)
    return compute(w, st, b, R, logits=logits, nch=chunks_for(st, R), host_pos=st.pos)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the last forward's first ``keep`` rows; a prompt chunk keeps all."""

    if not 1 <= keep <= R or (b.prefill and keep != R):
        raise ValueError("keep must be in 1..R, and all of a prompt chunk")
    st.set_pos(st.pos + keep)
