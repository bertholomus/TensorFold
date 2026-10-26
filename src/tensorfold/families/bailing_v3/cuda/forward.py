"""Ling-3.1-flash's tensor-parallel forward and commit: KDA and RoPE-MLA layers, grouped top-k MoE, plain residuals.

Per layer: ``h = rmsnorm(x)``; a KDA block (glm5_next's chain, beta kept fp32) or an MLA
block (latent cache plus a shared rope key, dense at every length, sigmoid head gate); ``x = bf16(x + sum of the
ranks' fp32 partials)``; ``h2 = rmsnorm(x)``; a dense SwiGLU (layers 0-3) or the routed experts plus the shared
expert; the same residual. Every kernel keeps rows apart, so a window's row r has the serial step's bits; the ranks
sum their partials in rank order, so both ranks hold the same bits.

The kernels are the GLM families' (one source a kernel): the KDA chain, replay and conv shift from glm5_next, the
EXL3 linear, the rope-key latent attention and the rank collectives from glm_moe_dsa, the routed experts from
tensorfold.cuda.exl3. Ling's own pieces are the grouped router (route.py), its rope arithmetic (rope.py), the
shared expert's clamped SwiGLU (act.py), the head gate and the q_nope-only absorb (Ling's qk_dim 192 is not a power
of two; the rope term is mla_pe's separate q_pe . k_pe score).
"""

from __future__ import annotations

import os
from typing import Sequence

import torch

from tensorfold.cuda.geometry import share

from tensorfold.families.glm5_next.cuda import glue, kda as kda_mod, latent, prof, qmm
from tensorfold.families.glm5_next.cuda.forward import Cut, _shift_conv  # noqa: F401  (Cut: the engine's prompt cut)

from tensorfold.families.glm_moe_dsa.cuda import glue as dsa_glue, mla_pe
from tensorfold.families.glm_moe_dsa.cuda.forward import embed, gather, mm, out_proj, residual
from tensorfold.families.glm_moe_dsa.cuda.x3 import X3, X3Pair, X3Scratch

from .. import memory
from . import act, rope as rope_mod, route as route_mod
from .weights import LayerW, MLAW, Weights

# prompt chunks' routed experts (tensorfold.cuda.exl3.experts.routed's kernel): "mma3" (gate/up rotating its rows from
# the layer input; its own summation order, other bits than the decode windows') or "mma" (the decode windows' bits)
PROMPT_EXPERTS = os.environ.get("TF_LING_PROMPT_EXPERTS") or "mma3"
# the dense latent attention's chunk partials: at most TF_LING_LATENT_MB a Buffers (memory.latent_rows; the startup
# estimate prices the same rows)
_latent_rows = memory.latent_rows
# TF_LING_FILLED=1: each row's latent attention walks only the chunks up to its last key's (mla_pe.attention's
# filled), so a decode graph captured for the whole cache reads and writes what eager does at this depth, with eager's
# bits; off, a graph walks every chunk of the cache (2,056 B of partials a row, head, chunk and layer, written then
# merged). Off until the node gates hold it (graph == eager across chunk boundaries, verify costs A/B)
FILLED = os.environ.get("TF_LING_FILLED", "0") == "1"


def _b16_part(w: Weights, rows: int) -> int:
    """fp32 words of split-K partials the largest BF16 matmul of ``rows`` rows needs (qmm's _matmul_b16)."""

    # the matrices the forward multiplies through ``mm`` (kv_b's rows are read by the absorb kernels instead)
    mats: list = [w.head, w.mtp.eh if w.mtp is not None else None]
    for layer in list(w.layers) + ([w.mtp.layer] if w.mtp is not None else []):
        if layer.kda is not None:
            k = layer.kda
            mats += [k.q, k.k, k.v, k.qkv, k.f, k.g, k.b, k.o]
        if layer.mla is not None:
            a = layer.mla
            mats += [a.q_a, a.q_b, a.kv_a, a.gate, a.dense]
        for m in (layer.mlp, layer.moe.shared if layer.moe is not None else None):
            if m is not None:
                mats += [m.gu, m.down]
    need = [qmm.b16_split_k(q.n, q.k) * q.n for q in mats
            if isinstance(q, qmm.B16) and qmm.b16_split_k(q.n, q.k) > 1]
    return max(need, default=1) * rows


class Buffers:
    """Scratch for windows of up to ``rows`` rows, sliced [:R] for smaller ones; ``prefill`` for prompt chunks."""

    def __init__(self, w: Weights, rows: int, capacity: int = 2560, *, prefill: bool = False) -> None:
        c = w.cfg
        dev = w.device
        bf, f32 = torch.bfloat16, torch.float32
        D = c.hidden
        HL = share(c.heads, w.world, w.rank)
        LL = share(c.lin_heads, w.world, w.rank)
        if c.lin_dim != kda_mod.DK:
            raise ValueError(f"bailing_v3: KDA head_dim {c.lin_dim}; the chain kernel is written for {kda_mod.DK}")
        self.rows, self.prefill, self.world = rows, prefill, w.world
        head_rows = 1 if prefill else rows
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        # cos / sin rows of the window's positions (filled from the device position: graph-safe)
        self.rope = rope_mod.Table(rows, c.rope_theta, c.qk_rope, dev)
        # the residual stream and the block inputs
        self.x = torch.empty((rows, D), dtype=bf, device=dev)
        self.normed = torch.empty((rows, D), dtype=bf, device=dev)
        # a BF16 projection's rows written into a column slice (qmm's BF16 matmul writes contiguous rows only)
        self.stage = torch.empty((rows * max(LL * kda_mod.DV, 1),), dtype=bf, device=dev)
        # KDA: P = q | k | v | b (one row a token; the decode window's rows live in State.proj for commit's conv
        # shift), the decay input f and the output gate g
        width = 3 * LL * kda_mod.DK + LL
        if prefill:
            self.kproj = torch.zeros((1, rows, width), dtype=bf, device=dev)
            self.kscratch = kda_mod.KDAScratch(rows, LL, dev)
        self.ka = torch.empty((rows, LL * kda_mod.DK), dtype=bf, device=dev)
        self.kg = torch.empty((rows, LL * kda_mod.DV), dtype=bf, device=dev)
        # MLA
        self.qr = torch.empty((rows, c.q_lora), dtype=bf, device=dev)
        self.q = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.qn = torch.empty((rows, HL, c.qk_nope), dtype=bf, device=dev)         # q_nope, contiguous (absorb)
        self.qp = torch.empty((rows, HL, c.qk_rope), dtype=bf, device=dev)         # the rotated rope slices
        self.lat = torch.empty((rows, c.kv_lora + c.qk_rope), dtype=bf, device=dev)   # latent | rope key
        self.vn = torch.empty((rows, HL, c.v_dim), dtype=bf, device=dev)
        self.gl = torch.empty((rows, HL), dtype=bf, device=dev)      # the head gate's g_proj rows (bf16)
        self.gf = torch.empty((rows, HL), dtype=f32, device=dev)     # sigmoid of them in fp32
        self.gb = torch.empty((rows, HL), dtype=bf, device=dev)      # the gate, rounded to bf16
        chunks = latent.chunks_for(capacity)
        self.lat_s = latent.LatentScratch(rows, HL, chunks, dev, lw=c.kv_lora,
                                          part_rows=_latent_rows(rows, HL, chunks, c.kv_lora))
        # dense MLP and shared-expert widths as loaded: the split moves EXL3 widths in whole 128-column blocks, so a
        # rank's width is not share(width, world) where the blocks do not divide (Ling's 1280 at TP4: 384 or 256)
        blocks = list(w.layers) + ([w.mtp.layer] if w.mtp is not None else [])
        dl = max((l.mlp.width for l in blocks if l.mlp is not None), default=share(c.dense_width, w.world, w.rank))
        self.gu = torch.empty((rows, 2 * dl), dtype=bf, device=dev)
        self.act = torch.empty((rows, dl), dtype=bf, device=dev)
        self.xs_act = torch.empty((rows, max(dl // 64, 1)), dtype=f32, device=dev)
        # MoE: fp32 router logits, their sigmoid scores, the picks (top_k routed, then the shared slot E)
        slots = c.top_k + 1
        self.mlog = torch.empty((rows, c.experts), dtype=f32, device=dev)
        self.msc = torch.empty((rows, c.experts), dtype=f32, device=dev)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=dev)
        self.wts = torch.empty((rows, slots), dtype=f32, device=dev)
        sl = max((l.moe.shared.width for l in blocks if l.moe is not None and l.moe.shared is not None),
                 default=share(c.shared_width, w.world, w.rank))
        self.sgu = torch.empty((rows, 2 * sl), dtype=bf, device=dev)
        self.sact = torch.empty((rows, sl), dtype=bf, device=dev)
        self.sxs = torch.empty((rows, max(sl // 64, 1)), dtype=f32, device=dev)
        self.sy = torch.empty((rows, D), dtype=f32, device=dev)
        self.moe = None
        first = next((l.moe.experts for l in w.layers if l.moe is not None), None)
        if first is not None and first.ex is not None:
            from tensorfold.cuda.exl3 import experts as x3experts

            self.moe = x3experts.Scratch(first.ex, rows, slots, device=dev, prompt=prefill)
        # rank partials and their gather
        self.part = torch.empty((rows, D), dtype=f32, device=dev)
        self.gath = torch.empty((w.world * rows * D,), dtype=f32, device=dev)
        self.sk = torch.empty((_b16_part(w, rows),), dtype=f32, device=dev)
        # EXL3 groups (attention, MLPs, MTP, head) share one scratch
        self.x3 = X3Scratch(w.meta.get("x3", []), dev, prefill=prefill)
        self.side = None                     # (glm_moe_dsa's second stream: not used here)
        # final rows: model.norm's output feeds the head and the MTP layer
        self.fnormed = torch.empty((rows, D), dtype=bf, device=dev)
        self.logits = torch.empty((head_rows, w.head.n), dtype=bf, device=dev)
        # MTP
        self.hin = torch.empty((rows, D), dtype=bf, device=dev)       # its input rows (final-normed main rows)
        self.me = torch.empty((rows, D), dtype=bf, device=dev)
        self.mcat = torch.empty((rows, 2 * D), dtype=bf, device=dev)
        self.mx = torch.empty((rows, D), dtype=bf, device=dev)
        # a block drafter's taps: the residual stream after chosen layers (``set_taps``), copied by every forward
        self.taps: list[torch.Tensor] = []
        self.tap_at: dict[int, list[int]] = {}
        self.layers = c.layers
        self.experts = c.experts
        self.top_k = c.top_k

    def set_taps(self, layers: Sequence[int]) -> None:
        """Keep the residual stream after each of ``layers`` (HF's ``hidden_states[layer + 1]``, bf16): set before any
        graph capture, since the copies are captured work. The last layer is refused: HF's last hidden state is after
        model.norm, which is ``fnormed``, not a tap."""

        bad = [i for i in layers if not 0 <= i < self.layers - 1]
        if bad:
            raise ValueError(f"taps after layers {bad}: a tap follows one of layers 0..{self.layers - 2}")
        self.tap_at = {}
        for slot, layer in enumerate(layers):
            self.tap_at.setdefault(int(layer), []).append(slot)
        self.taps = [torch.empty_like(self.x) for _ in layers]


class State:
    """Committed caches of one sequence: KDA recurrent states and conv windows, MLA latents and rope keys (and the
    MTP layer's)."""

    def __init__(self, w: Weights, capacity: int, rows: int) -> None:
        c = w.cfg
        dev = w.device
        LL = share(c.lin_heads, w.world, w.rank)
        self.capacity = capacity
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.mtp_pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        mla_layers = [l for l in w.layers if l.kind == "mla"]
        self.kda_index = {l.index: i for i, l in enumerate(kda_layers)}
        self.mla_index = {l.index: i for i, l in enumerate(mla_layers)}
        n = len(kda_layers)
        width = 3 * LL * kda_mod.DK + LL
        # conv: the last conv - 1 pre-conv q | k | v rows; rec: two fp32 [H, 128 (v), 128 (k)] states a layer
        # (cur and 1 - cur: a window runs cur -> 1 - cur, commit replays the kept rows when it keeps fewer)
        self.conv = torch.zeros((n, c.conv - 1, 3 * LL * kda_mod.DK), dtype=torch.bfloat16, device=dev)
        self.rec = torch.zeros((2, n, LL, kda_mod.DV, kda_mod.DK), dtype=torch.float32, device=dev)
        self.cur = [0] * n
        self.proj = torch.zeros((n, rows, width), dtype=torch.bfloat16, device=dev)
        self.scratch_set = kda_mod.KDAScratchSet(n, rows, LL, dev) if n else None
        self.scratch = self.scratch_set.views if n else []
        self.latent = True
        self.kc = [torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev) for _ in mla_layers]
        self.pc = [torch.zeros((capacity, c.qk_rope), dtype=torch.bfloat16, device=dev) for _ in mla_layers]
        self.vc = [None for _ in mla_layers]
        self.index = None                    # no lightning indexer: MLA is dense at every length
        self.mtp_len = 0
        self.mtp_drafted = 0
        if w.mtp is not None:
            self.mtp_kc = torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev)
            self.mtp_pc = torch.zeros((capacity, c.qk_rope), dtype=torch.bfloat16, device=dev)
            self.mtp_vc = None

    def reset(self) -> None:
        self.conv.zero_()
        self.rec.zero_()
        self.cur = [0] * len(self.cur)
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
        return self.cur[0] if self.cur else 0

    def clone(self) -> "State":
        import copy

        other = copy.copy(self)
        other.conv = self.conv.clone()
        other.rec = self.rec.clone()
        other.cur = list(self.cur)
        other.pos_dev = self.pos_dev.clone()
        other.mtp_pos_dev = self.mtp_pos_dev.clone()
        other.kc = [x.clone() for x in self.kc]
        other.pc = [x.clone() for x in self.pc]
        other.vc = [None for _ in self.kc]
        if hasattr(self, "mtp_kc"):
            other.mtp_kc = self.mtp_kc.clone()
            other.mtp_pc = self.mtp_pc.clone()
            other.mtp_vc = None
        return other


# -- blocks ---------------------------------------------------------------------------------------------------
def into(b: Buffers, x: torch.Tensor, q, out: torch.Tensor) -> torch.Tensor:
    """A bf16 projection of x into ``out`` (rows with unit column stride; a column slice of a wider buffer is fine):
    an EXL3 group writes it directly (the linear stages what it cannot write in place), a BF16 matrix through a
    contiguous staging block (the same bits: the matmul's rows do not depend on where they land)."""

    if isinstance(q, (X3, X3Pair)) or out.is_contiguous():
        return mm(b, x, q, None, out)
    R = x.shape[0]
    y = b.stage[:R * q.n].view(R, q.n)
    mm(b, x, q, None, y)
    out.copy_(y)
    return out


def kda_block(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, cut: Cut | None = None):
    """Kimi Delta Attention: five full-rank projections and the beta logits, glm5_next's chain with beta in fp32."""

    c = w.cfg
    k = layer.kda
    li = st.kda_index[layer.index]
    pre = b.prefill
    p = b.kproj[0, :R] if pre else st.proj[li, :R]
    C = k.heads * kda_mod.DK
    x = b.normed[:R]
    with prof.timed("kda: projections"):
        if k.qkv is not None:            # the conversion's fused group: the rank's q | k | v rows in one call
            into(b, x, k.qkv, p[:, :3 * C])
        else:
            into(b, x, k.q, p[:, :C])
            into(b, x, k.k, p[:, C:2 * C])
            into(b, x, k.v, p[:, 2 * C:3 * C])
        into(b, x, k.b, p[:, k.b_off:k.b_off + k.heads])
        mm(b, x, k.f, None, b.ka[:R])
        mm(b, x, k.g, None, b.kg[:R])
    cur = st.cur[li]
    with prof.timed("kda: chain"):
        if cut is None:
            out = kda_mod.chain(p, k.b_off, b.ka[:R], b.kg[:R], st.conv[li], k.conv, st.rec[cur, li], k.a_log,
                                k.dt_bias, k.norm, c.eps, c.lower, R, b.kscratch if pre else st.scratch[li],
                                st.rec[1 - cur, li], beta_bf16=False)
        else:
            n = cut.point
            first = kda_mod.chain(p[:n], k.b_off, b.ka[:n], b.kg[:n], st.conv[li], k.conv, st.rec[cur, li],
                                  k.a_log, k.dt_bias, k.norm, c.eps, c.lower, n, b.kscratch, cut.rec[li],
                                  beta_bf16=False).clone()
            _shift_conv(cut.conv[li:li + 1], b.kproj[:, :n], n)
            rest = kda_mod.chain(p[n:], k.b_off, b.ka[n:R], b.kg[n:R], cut.conv[li], k.conv, cut.rec[li],
                                 k.a_log, k.dt_bias, k.norm, c.eps, c.lower, R - n, b.kscratch,
                                 st.rec[1 - cur, li], beta_bf16=False)
            out = torch.cat((first, rest))
    if pre:                              # a prompt chunk keeps every row: the layer commits now
        st.cur[li] = 1 - cur
        _shift_conv(st.conv[li:li + 1], b.kproj[:, :R], R)
    return out_proj(w, b, out, k.o, None, R)


def mla_block(a: MLAW, w: Weights, lc: torch.Tensor, pc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers, R: int,
              nch: int | None = None, host_pos: int | None = None):
    """RoPE MLA over the latent and rope-key caches, dense at every length, then the sigmoid head gate.

    Scores are ``(q_nope W_UK . c + q_pe . k_pe) * qk_dim ** -0.5`` with q_nope absorbed into the latent space and
    the rope term scored separately (mla_pe); values are the latent, expanded by W_UV per head. The head gate
    (modeling 709-718): ``o = bf16(o * bf16(sigmoid(fp32(g_proj(h)))))`` per head, before ``dense``. b.normed holds
    the block input h; b.rope the window's cos / sin rows.
    """

    c = w.cfg
    HL = a.heads
    s = b.lat_s
    x = b.normed[:R]
    with prof.timed("mla: projections"):
        # the latent and the shared rope key: kv_a_layernorm covers the latent only, the rope key rotates as projected
        mm(b, x, a.kv_a, None, b.lat[:R])
        glue.rmsnorm(b.lat[:R, :c.kv_lora], a.kv_norm, c.eps, b.lat[:R, :c.kv_lora])
        rope_mod.apply_key(b.lat[:R, c.kv_lora:], b.rope, R, c)
        latent.latent_write(b.lat[:R, :c.kv_lora], lc, pos_dev)
        latent.latent_write(b.lat[:R, c.kv_lora:], pc, pos_dev)
        # queries: q_a, its norm, q_b per head [qk_nope | qk_rope], the rope slices rotated
        mm(b, x, a.q_a, None, b.qr[:R])
        glue.rmsnorm(b.qr[:R], a.q_norm, c.eps, b.qr[:R])
        q2 = b.q[:R].view(R, HL * c.qk_dim)
        mm(b, b.qr[:R], a.q_b, None, q2)
        rope_mod.apply_q(q2, b.rope, R, c, HL)
        mla_pe.gather_pe(b.q[:R], 0, b.qn[:R])                     # q_nope [R, HL, qk_nope], contiguous
        qp = mla_pe.gather_pe(b.q[:R], c.qk_nope, b.qp[:R])        # q_pe [R, HL, qk_rope]
    with prof.timed("mla: absorb"):
        qa = mla_pe.absorb_q(b.qn[:R], a.absorb, s.qa[:R])
    n = s.nch if nch is None else max(1, min(nch, s.nch))
    if host_pos is not None:             # the chunks these rows can see
        n = max(1, min(n, latent.chunks_for(host_pos + R)))
    with prof.timed("mla: attention"):
        ol = mla_pe.attention(qa, qp, lc, pc, pos_dev, s, scale=c.qk_dim ** -0.5, nch=n, out=s.ol[:R], filled=FILLED)
    with prof.timed("mla: expand and gate"):
        o = mla_pe.expand_v(ol, a.absorb, b.vn[:R])                # [R, HL, v_dim] bf16
        mm(b, x, a.gate, None, b.gl[:R])
        b.gf[:R].copy_(b.gl[:R])                                   # fp32(g_proj(h)): exact
        torch.sigmoid(b.gf[:R], out=b.gf[:R])
        b.gb[:R].copy_(b.gf[:R])                                   # .type_as(hidden_states): one bf16 rounding
        o.mul_(b.gb[:R, :, None])                                  # bf16 * bf16, rounded once (torch's mul)
    return out_proj(w, b, o.view(R, HL * c.v_dim), a.dense, None, R)


def mlp_block(layer: LayerW, w: Weights, b: Buffers, R: int):
    """Dense SwiGLU: down(bf16(bf16(silu(gate)) * up)), the rank's width share, fp32 partials."""

    m = layer.mlp
    mm(b, b.normed[:R], m.gu, None, b.gu[:R])
    dsa_glue.silu_mul(b.gu[:R], b.act[:R], b.xs_act[:R])
    return out_proj(w, b, b.act[:R], m.down, None, R)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int):
    """Grouped top-k routing (route.py), the routed experts' weighted sum in slot order, then the shared expert.

    The activation is the reference's bf16(bf16(silu(gate)) * up) with the config's clamp after silu where the layer
    has one (min(silu(g), L) * clip(u, -L, L), L = 10 on the routed experts of layers 65-71 and the shared
    expert of 64-71; never the MTP layer; TF_LING_SWIGLU_CLAMP=0 drops it): ACT_BF16_POST in the routed experts'
    epilogue, act.silu_mul for the shared expert, both ACT_BF16's and glue.silu_mul's bits at L = inf. A prompt
    chunk computes the shared expert first and the routed launch adds it last; a decode window adds it after (the
    same add).
    """

    from tensorfold.cuda.exl3 import experts as x3experts

    c = w.cfg
    m = layer.moe
    s = m.shared
    limit, shared_limit = c.swiglu_limit(layer.index), c.swiglu_limit(layer.index, shared=True)

    def shared() -> None:
        mm(b, b.normed[:R], s.gu, None, b.sgu[:R])
        act.silu_mul(b.sgu[:R], b.sact[:R], b.sxs[:R], shared_limit)
        mm(b, b.sact[:R], s.down, None, b.sy[:R], f32=True)

    inline = b.prefill and s is not None
    if inline:
        shared()
    with prof.timed("moe: route"):
        glue.router(b.normed[:R], m.router, b.mlog[:R])           # fp32 logits (bf16 inputs, fp32 sums)
        route_mod.route(b.mlog[:R], m.bias, b.msc[:R], b.pick[:R], b.wts[:R], n_group=c.n_group,
                        topk_group=c.topk_group, top_k=c.top_k, scale=c.routed_scale)
    with prof.timed("moe: routed"):
        x3experts.routed(b.normed[:R], b.pick[:R], b.wts[:R], m.experts.ex, b.moe, b.part[:R], R, limit=limit,
                         act_mode=x3experts.ACT_BF16_POST, add=b.sy[:R] if inline else None,
                         kernel=PROMPT_EXPERTS, y_unused=True)
    if s is not None and not inline:
        shared()
        b.part[:R].add_(b.sy[:R])
    with prof.timed("moe: all-gather"):
        return gather(w, b, R)


def layer_forward(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None = None,
                  host_pos: int | None = None, cut: Cut | None = None) -> None:
    c = w.cfg
    glue.rmsnorm(b.x[:R], layer.in_norm, c.eps, b.normed[:R])
    if layer.kind == "kda":
        with prof.timed("kda (total)"):
            g = kda_block(layer, w, st, b, R, cut)
    else:
        di = st.mla_index[layer.index]
        with prof.timed("mla (total)"):
            g = mla_block(layer.mla, w, st.kc[di], st.pc[di], st.pos_dev, b, R, nch, host_pos)
    residual(b.x[:R], b.x[:R], g)
    glue.rmsnorm(b.x[:R], layer.post_norm, c.eps, b.normed[:R])
    with prof.timed("moe (total)" if layer.mlp is None else "mlp"):
        g = mlp_block(layer, w, b, R) if layer.mlp is not None else moe_block(layer, w, b, R)
    residual(b.x[:R], b.x[:R], g)


def check_room(w: Weights, st: State, R: int, pos: int | None = None) -> None:
    """MLA attends densely at any length: only the caches' capacity limits a window."""

    pos = st.pos if pos is None else pos
    if pos + R > st.capacity:
        raise ValueError(f"context {pos + R} past the cache capacity {st.capacity}")


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
            host_pos: int | None = None, cut: Cut | None = None):
    """Run capturable GPU work on static buffers and device positions. b.fnormed keeps model.norm's output of every
    row (the MTP layer's input) whether or not logits are asked for."""

    if cut is not None and (not b.prefill or not 0 < cut.point < R):
        raise ValueError("a prompt cut must lie inside a prefill chunk")
    c = w.cfg
    embed(w, b, b.ids[:R], b.x[:R])
    b.rope.fill(st.pos_dev, R)                     # device position: graph-safe
    for layer in w.layers:
        layer_forward(layer, w, st, b, R, nch, host_pos, cut)
        for slot in b.tap_at.get(layer.index, ()):
            b.taps[slot][:R].copy_(b.x[:R])
    glue.rmsnorm(b.x[:R], w.norm, c.eps, b.fnormed[:R])
    if not logits:
        return None
    if b.prefill:                        # the head reads the last row only
        return mm(b, b.fnormed[R - 1:R], w.head, None, b.logits[:1])
    return mm(b, b.fnormed[:R], w.head, None, b.logits[:R])


def chunks_for(st: State, R: int) -> int:
    return latent.chunks_for(st.pos + R)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True) -> torch.Tensor | None:
    """Return logits for token rows, leaving committed state unchanged until commit."""

    R = stage(w, st, b, tokens)
    return compute(w, st, b, R, logits=logits, nch=chunks_for(st, R), host_pos=st.pos)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the last forward's first ``keep`` rows; a prompt chunk keeps all, its KDA layers already committed.

    A decode window ran every KDA layer from rec[cur] into rec[1 - cur]: keeping fewer rows replays the kept ones
    from rec[cur] (the saved k, v, decay and fp32 beta through the chain's own update routine: the serial bits), then
    cur flips and the conv windows shift to the last kept rows. The latent caches need nothing: rows past the kept
    ones are overwritten before any later row can see them."""

    if not 1 <= keep <= R or (b.prefill and keep != R):
        raise ValueError("keep must be in 1..R, and all of a prompt chunk")
    n = 0 if b.prefill else len(st.cur)
    if n:
        cur = st.cur[0]
        if keep < R:
            kda_mod.replay_layers(st.rec[cur], st.scratch_set, keep, st.rec[1 - cur])
        st.cur = [1 - cur] * n
        _shift_conv(st.conv, st.proj, keep)
    st.set_pos(st.pos + keep)
