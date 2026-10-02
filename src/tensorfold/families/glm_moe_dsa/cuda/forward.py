"""GLM-5.3's tensor-parallel forward: Flash's row-invariant kernels, plain residuals, interleaved rope, raw-token DSA."""

from __future__ import annotations

import os
from typing import Sequence

import torch
import triton

from tensorfold.families.glm5_next.cuda.qmm import B16

from tensorfold.families.glm5_next.cuda import latent as latent_mod, prof, qmm as flash_qmm

from tensorfold.cuda.geometry import share

from . import glue, mla_pe, rope as rope_mod, select as select_mod
from .weights import LayerW, Weights

# the Flash buffers' DSA block is reused where the shapes match; the indexer differs (no k-pool)
from tensorfold.families.glm5_next.cuda.forward import (  # noqa: F401
    Buffers as FlashBuffers, State as FlashState, gather as flash_gather, mm as flash_mm,
)

from .x3 import X3, X3Pair, X3Scratch

# decode windows run independent work on a second stream: the latent / rope-key / indexer-key path beside the query
# path, the shared expert beside the router and routed experts (TF_GLM_SIDE=0: one stream, as before)
SIDE = os.environ.get("TF_GLM_SIDE", "1") != "0"
# a prompt chunk's rank partials: "rows" sends each rank the fp32 partials of its share of the rows (contiguous, rank r
# sums rows R r / world ..), sums them in rank order (residual_add's arithmetic) and gathers the bf16 sums back in row
# order: 2.7x fewer bytes than "gather" (every fp32 partial to every rank, summed by residual_add), the same bits
PROMPT_REDUCE = os.environ.get("TF_GLM_PROMPT_REDUCE") or "rows"


# prompt chunks of this many rows or more run as two micro-batches, each one's collectives (rows mode) on a second
# stream beside the other's compute (TF_GLM_PROMPT_OVERLAP=0: one batch); every row keeps its bits
OVERLAP_ROWS = 256 if os.environ.get("TF_GLM_PROMPT_OVERLAP", "1") != "0" else 1 << 30
COMM_PRIORITY = int(os.environ.get("TF_GLM_COMM_PRIORITY", "-1"))   # the comm stream's priority (lower is higher)
# prompt chunks run the shared expert before the routed experts and the routed combine adds its output (one pass over
# the fp32 partial less; the same add); TF_GLM_SHARED_INLINE=0: after them, then part += sy. (On a stream of its own
# beside the routed experts it was slower: 41.1 vs 40.4-40.9 s at 26k tokens on TP4.)
SHARED_INLINE = os.environ.get("TF_GLM_SHARED_INLINE", "1") != "0"


class Rows:
    """A prompt chunk's reduced partials as the bf16 branch residual_add would add, in row order (``done``: the comm
    stream's event when the reduction runs beside compute)."""

    def __init__(self, bg: torch.Tensor, done: torch.cuda.Event | None = None) -> None:
        self.bg = bg                                    # [R, D] bf16
        self.done = done


def gather(w: Weights, b, R: int):
    """Every rank's fp32 partial b.part[:R] in rank order ([world, R, D], summed by residual), or for a prompt chunk in
    PROMPT_REDUCE "rows" mode the bf16 branch, reduced a share of the rows on each rank (Rows): the same bits after
    residual."""

    world = w.world
    if not (b.prefill and PROMPT_REDUCE == "rows" and w.comm is not None and world > 2):
        return flash_gather(w, b, R)                     # (two ranks would save a quarter of the bytes)
    D = b.part.shape[1]
    cut = [R * k // world for k in range(world + 1)]
    n = cut[w.rank + 1] - cut[w.rank]
    # the gather buffer (world x rows x D fp32) holds every rank's partial of this rank's rows (world n D words), their
    # bf16 sum (n D / 2) and every rank's sums in row order (R D / 2)
    qr = b.gath[:world * n * D].view(world, n, D)
    off = world * n * D
    br = b.gath[off:off + -(-n * D // 2)].view(torch.bfloat16)[:n * D].view(n, D)
    off += -(-n * D // 2)
    bg = b.gath[off:off + -(-R * D // 2)].view(torch.bfloat16)[:R * D].view(R, D)

    def reduce() -> None:
        part = b.part[:R]
        w.comm.grouped([(part[cut[k]:cut[k + 1]], k) for k in range(world)], [(qr[k], k) for k in range(world)])
        glue.rank_sum(qr.view(world, -1), br.view(-1))           # rank order, then bf16 (residual_add's branch)
        w.comm.grouped([(br, k) for k in range(world)], [(bg[cut[k]:cut[k + 1]], k) for k in range(world)])

    cs = getattr(b, "comm_stream", None)
    if cs is None or not getattr(b, "micro", False):
        reduce()
        return Rows(bg)
    ready = torch.cuda.Event()                          # this micro-batch's partial is written
    ready.record()
    cs.wait_event(ready)
    with torch.cuda.stream(cs):
        reduce()
        done = torch.cuda.Event()
        done.record()
    return Rows(bg, done)


def residual(x: torch.Tensor, xout: torch.Tensor, g) -> None:
    """x + the gathered partials' bf16 sum (residual_add), from fp32 partials or Rows."""

    if isinstance(g, Rows):
        if g.done is not None:
            torch.cuda.current_stream().wait_event(g.done)
        glue.residual_add(x, xout, g.bg.view(1, *g.bg.shape))   # one bf16 "partial": x + the branch, as residual_add
    else:
        glue.residual_add(x, xout, g)


def micro_batch(b, lo: int, hi: int, half: int, R: int):
    """Rows lo..hi of a prompt chunk's R rows as buffers of their own (a shallow copy, every row-indexed tensor sliced),
    with half ``half`` of the gather buffer for their collectives and ``full`` the whole chunk's buffers."""

    import copy

    def rows(obj):
        view = copy.copy(obj)
        for name, value in vars(obj).items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == b.rows:
                setattr(view, name, value[lo:hi])
        return view

    v = rows(b)
    v.lat_s = rows(b.lat_s)
    g = b.gath.numel() // 2
    v.gath = b.gath[half * g:(half + 1) * g]
    v.micro, v.full, v.chunk = True, b, R
    return v


def mm(b, x: torch.Tensor, q, xs: torch.Tensor | None, out: torch.Tensor, f32: bool = False,
       sc: X3Scratch | None = None) -> torch.Tensor:
    """A projection: an EXL3 group on the row-invariant EXL3 linear (prompt GEMM in chunks), else Flash's matmuls."""

    if isinstance(q, (X3, X3Pair)):
        if f32 and out.dtype != torch.float32:
            raise ValueError("mm: an f32 projection needs an fp32 output buffer")
        return q(x, out, b.x3 if sc is None else sc, prefill=b.prefill)
    return flash_mm(b, x, q, xs, out, f32=f32)


def fork(b):
    """The side stream after everything the current stream has queued so far (a context to queue the side work in)."""

    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    b.side.wait_event(ready)
    return torch.cuda.stream(b.side)


def join(b) -> None:
    """The current stream waits for the side stream's work so far."""

    done = torch.cuda.Event()
    done.record(b.side)
    torch.cuda.current_stream().wait_event(done)


def out_proj(w, b, x: torch.Tensor, q, xs, R: int) -> torch.Tensor:
    """A rank's fp32 partial of a row-split input (o_proj, MLP down) gathered over the ranks."""

    mm(b, x, q, xs, b.part[:R], f32=True)
    return gather(w, b, R)


class Buffers(FlashBuffers):
    """Flash's buffers plus GLM-5.3's rope cos/sin rows and plain-residual state (no stream copies)."""

    def __init__(self, w: Weights, rows: int, capacity: int = 2560, *, prefill: bool = False) -> None:
        import copy
        import dataclasses

        # Flash's constructor sizes its 4-bit-only EXL3 expert scratch when cfg.quant is "exl3" (GBs at prompt-chunk
        # rows); this family runs routed experts on the universal kernel, so build Flash's part from a plain view
        view = copy.copy(w)
        view.cfg = dataclasses.replace(w.cfg, quant="plain")
        super().__init__(view, rows, capacity, prefill=prefill)
        c = w.cfg
        dev = w.device
        bf = torch.bfloat16
        f32 = torch.float32
        del self.ey, self.plan, self.eact
        self.exl3 = None
        # the shared expert's MLP rows (Flash allocates these only on its EXL3 path)
        sl = share(c.shared_width, w.world)
        self.sgu = torch.empty((rows, 2 * sl), dtype=bf, device=dev)
        self.sact = torch.empty((rows, sl), dtype=bf, device=dev)
        self.sxs = torch.empty((rows, sl // 64), dtype=f32, device=dev)
        self.sy = torch.empty((rows, c.hidden), dtype=f32, device=dev)
        # raw-token selection list: index_topk entries plus the current row's tail
        self.tokens = torch.empty((rows, c.index_topk + 1), dtype=torch.int32, device=dev)
        self.counts = torch.empty((rows,), dtype=torch.int32, device=dev)
        self.cos = torch.empty((rows, c.qk_rope // 2), dtype=torch.float32, device=dev)
        self.sin = torch.empty((rows, c.qk_rope // 2), dtype=torch.float32, device=dev)
        self.ik = torch.empty((rows, c.index_dim), dtype=bf, device=dev)          # the window's indexer keys
        self.qp = torch.empty((rows, share(c.heads, w.world, w.rank), c.qk_rope), dtype=bf, device=dev)  # rope slices
        self.iw = torch.empty((rows, c.index_heads), dtype=bf, device=dev)        # indexer head weights per token
        # no hyper-connections: x holds one stream
        self.x = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        del self.taps, self.tap_at
        # kv_a's output is the latent then the shared rope key (kv_lora + qk_rope): Flash's buffer has no rope part
        self.lat = torch.empty((rows, c.kv_lora + c.qk_rope), dtype=bf, device=dev)
        # EXL3 groups (attention, MLPs, MTP, head) share one scratch; the routed experts run on the universal kernel
        self.x3 = X3Scratch(w.meta.get("x3", []), dev, prefill=prefill)
        self.moe = None
        self.side = None                     # decode windows: the second stream and its own EXL3 scratch
        # prompt chunks: the stream a micro-batch's collectives run on beside the other's compute (high priority: its
        # few blocks go ahead of the compute kernels' queued ones), and the device position of the second micro-batch's
        # first row
        self.comm_stream = (torch.cuda.Stream(device=dev, priority=COMM_PRIORITY) if prefill and dev.type == "cuda"
                            else None)
        self.pos_mb = torch.zeros((1,), dtype=torch.int32, device=dev)
        if SIDE and not prefill and dev.type == "cuda":
            self.side = torch.cuda.Stream(device=dev)
            self.x3s = X3Scratch(w.meta.get("x3", []), dev, prefill=False)
        first = next((l.moe.experts for l in w.layers if l.moe is not None), None)
        if first is not None and first.ex is not None:
            from tensorfold.cuda.exl3 import experts as x3experts

            self.moe = x3experts.Scratch(first.ex, rows, c.top_k + 1, device=dev, prompt=prefill)


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
        # the shared rope key per token (q_pe . k_pe is the second score term; Flash has qk_rope 0 and no such cache)
        self.pc = [torch.zeros((capacity, c.qk_rope), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
        self.vc = [None for _ in dsa_layers]
        self.mtp_len = 0
        self.mtp_drafted = 0
        if w.mtp is not None:
            self.mtp_kc = torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev)
            self.mtp_pc = torch.zeros((capacity, c.qk_rope), dtype=torch.bfloat16, device=dev)
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
        other.pc = [x.clone() for x in self.pc]
        other.vc = [None for _ in self.kc]
        if self.index is not None:
            other.index = [x.clone() for x in self.index]
        if hasattr(self, "mtp_kc"):
            other.mtp_kc = self.mtp_kc.clone()
            other.mtp_pc = self.mtp_pc.clone()
            other.mtp_vc = None
        return other


def dsa_block(layer: LayerW, w: Weights, lc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers, R: int,
              nch: int | None, index: torch.Tensor | None, host_pos: int | None,
              sparse_np: int | None = None, pc: torch.Tensor | None = None) -> torch.Tensor:
    """MLA over the latent cache with GLM-5.3's rope, indexer update on full layers, raw-token sparse selection.

    A "shared" indexer layer passes ``index=None``: its selection reuse is expressed by scoring and
    selecting in the owning full layer (which runs the same rows either just before or within the
    same window), so the shared layer reads that layer's tokens from the buffers below.
    """

    c = w.cfg
    a = layer.dsa
    HL = a.heads
    s = b.lat_s

    def keys(sc: X3Scratch | None) -> None:
        """This window's latent, rope key and (full layers) indexer key, into their caches at the window's slots."""

        with prof.timed("dsa: projections"):
            mm(b, b.normed[:R], a.kv_a, b.xs[:R], b.lat[:R], sc=sc)
            # kv_a_layernorm covers the latent only; the rope key after it stays as projected
            glue.rmsnorm(b.lat[:R, :c.kv_lora], a.kv_norm, c.eps, b.lat[:R, :c.kv_lora], b.xs_lat[:R])
        with prof.timed("dsa: rope"):
            rope_mod.apply_key(b.lat[:R, c.kv_lora:], b.cos[:R], b.sin[:R], c)
        with prof.timed("dsa: latent write"):
            latent_mod.latent_write(b.lat[:R, :c.kv_lora], lc, pos_dev)
            latent_mod.latent_write(b.lat[:R, c.kv_lora:], pc, pos_dev)      # the rope key: its own [cap, qk_rope] cache
        if index is not None:
            with prof.timed("dsa: indexer update"):
                ix = a.index
                mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ik[:R])
                glue.layernorm(b.ik[:R], ix.ln_w, ix.ln_b, c.eps, b.ik[:R])
                rope_mod.apply_index(b.ik[:R], b.cos[:R], b.sin[:R], c, 1)
                if host_pos is not None:
                    index[host_pos:host_pos + R].copy_(b.ik[:R])
                else:
                    latent_mod.latent_write(b.ik[:R], index, pos_dev)

    if b.side is not None:                   # the keys on the side stream while the queries project
        with fork(b):
            keys(b.x3s)
    with prof.timed("dsa: projections"):
        mm(b, b.normed[:R], a.q_a, b.xs[:R], b.qr[:R])
        glue.rmsnorm(b.qr[:R], a.q_norm, c.eps, b.qr[:R], b.xs_qr[:R])
    with prof.timed("dsa: q_b"):
        q2 = b.q[:R].view(R, HL * c.qk_dim)
        mm(b, b.qr[:R], a.q_b, b.xs_qr[:R], q2)
    with prof.timed("dsa: rope"):
        # heads are [qk_nope | qk_rope]: rotate each head's rope slice (the shared rope key rotates in keys())
        rope_mod.apply_q(q2, b.cos[:R], b.sin[:R], c, HL)
    if b.side is not None:
        join(b)
    else:
        keys(None)
    long_ctx = bool(w.meta.get("long_context"))
    all_sparse = sparse_np is not None or (host_pos is not None and host_pos >= c.dense_limit)
    sparse_rows = long_ctx and (all_sparse or (host_pos is not None and host_pos + R - 1 >= c.dense_limit))
    with prof.timed("dsa: absorb"):
        qa = mla_pe.absorb_q(b.q[:R], a.absorb, s.qa[:R])              # rope columns of wk are zero: q_nope . W_UK
        qp = mla_pe.gather_pe(b.q[:R], c.qk_nope, b.qp[:R])
    ol = s.ol[:R]
    scale = c.qk_dim ** -0.5
    if not all_sparse:
        with prof.timed("dsa: dense attention"):
            mla_pe.attention(qa, qp, lc, pc, pos_dev, s, scale=scale, nch=min(nch or s.nch, s.nch), out=ol)
    if sparse_rows:
        if a.index is not None:
            with prof.timed("dsa: select tokens"):
                ix = a.index
                mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
                rope_mod.apply_index(b.qi[:R], b.cos[:R], b.sin[:R], c, c.index_heads)
                # per-token head weights: weights_proj applied to the layer input (the scorer folds the scales)
                if getattr(b, "micro", False):
                    # cuBLAS picks its kernel by the row count and a row's bits move with it: a micro-batch multiplies
                    # the whole chunk's rows (the other's are recomputed when it gets here) and keeps its own
                    f = b.full
                    torch.mm(f.normed[:b.chunk], ix.weights.t(), out=f.iw[:b.chunk])
                elif b.prefill:
                    torch.mm(b.normed[:R], ix.weights.t(), out=b.iw[:R])
                else:
                    # decode windows: the row-invariant BF16 matmul, so a verify window's row gets the 1-row step's
                    # weights (cuBLAS by row count gave them different bits, near-tie selections moved, and past the
                    # dense limit MTP drafts' output parted from serial decoding: tools/check_window_rows.py)
                    flash_qmm.matmul(b.normed[:R], ix.w16, None, out=b.iw[:R], part=b.sk)
                select_mod.select_tokens(b.qi[:R], b.iw[:R], index, host_pos, R, c.index_topk, pos_dev,
                                         tokens=b.tokens[:R], counts=b.counts[:R], bucket=sparse_np)
        # a "shared" layer attends the tokens its group's full layer selected for these rows (still in b.tokens)
        with prof.timed("dsa: sparse attention"):
            mla_pe.sparse_attention(qa, qp, lc, pc, b.tokens[:R], b.counts[:R], ol, scale)
    with prof.timed("dsa: expand"):
        o = mla_pe.expand_v(ol, a.absorb, b.vn[:R]).view(R, HL * c.v_dim)
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
    """Routed experts on ``tensorfold.cuda.exl3.experts`` (any width and codebook), then the shared expert, in fp32.

    The routed slots' weighted sum comes out of one fused launch in slot order (the shared expert's slot, id E,
    is skipped there); the shared expert's output is added last, the same order as Flash's combine (a prompt chunk
    computes it first and the fused launch adds it: the same add, without another pass over the partial).
    """

    import math

    from tensorfold.cuda.exl3 import experts as x3experts

    c = w.cfg
    m = layer.moe
    s = m.shared

    def shared(sc: X3Scratch | None) -> None:
        mm(b, b.normed[:R], s.gu, b.xs[:R], b.sgu[:R], sc=sc)
        glue.silu_mul(b.sgu[:R], b.sact[:R], b.sxs[:R])
        mm(b, b.sact[:R], s.down, b.sxs[:R], b.sy[:R], f32=True, sc=sc)

    inline = SHARED_INLINE and b.prefill and b.side is None   # a prompt chunk: the shared expert first
    if b.side is not None:                   # the shared expert on the side stream while the routed experts run
        with fork(b):
            shared(b.x3s)
    elif inline:
        shared(None)
    with prof.timed("moe: route"):
        glue.router(b.normed[:R], m.router, b.mlog[:R])
        glue.select(b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], c.top_k, c.experts, c.routed_scale, c.norm_topk)
    with prof.timed("moe: routed"):
        # GLM-5.3 has no swiglu_limit; the bf16 activation roundings are the GLM family's
        x3experts.routed(b.normed[:R], b.pick[:R], b.wts[:R], m.experts.ex, b.moe, b.part[:R], R,
                         limit=math.inf, act_mode=x3experts.ACT_BF16, add=b.sy[:R] if inline else None)
    if b.side is not None:
        join(b)
    elif not inline:
        shared(None)
    if not inline:
        b.part[:R].add_(b.sy[:R])
    with prof.timed("moe: all-gather"):
        return gather(w, b, R)


def attn_part(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None = None,
              host_pos: int | None = None, sparse_np: int | None = None, pos_dev: torch.Tensor | None = None):
    """A layer's attention half up to its gathered partials (residual() adds them); rows from pos_dev (default st's)."""

    di = st.dsa_index[layer.index]
    c = w.cfg
    with prof.timed("dsa (total)"):
        # pre-attention RMSNorm (input_layernorm): the block reads b.normed / b.xs
        glue.rmsnorm(b.x[:R], layer.in_norm, c.eps, b.normed[:R], b.xs[:R])
        # a "shared" indexer layer scores and selects nothing of its own: it reuses the owning full
        # layer's selection, which the same window computed one block earlier into the buffers
        idx = st.index[st.index_slot[layer.index]] if getattr(st, "index", None) is not None \
            and layer.dsa.index is not None else None
        return dsa_block(layer, w, st.kc[di], st.pos_dev if pos_dev is None else pos_dev, b, R, nch, idx, host_pos,
                         sparse_np, pc=st.pc[di])


def ffn_part(layer: LayerW, w: Weights, b: Buffers, R: int):
    """A layer's MLP / MoE half up to its gathered partials."""

    with prof.timed("moe (total)" if layer.mlp is None else "mlp"):
        # post-attention RMSNorm before the MLP / MoE
        glue.rmsnorm(b.x[:R], layer.post_norm, w.cfg.eps, b.normed[:R], b.xs[:R])
        return mlp_block(layer, w, b, R) if layer.mlp is not None else moe_block(layer, w, b, R)


def layer_forward(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None = None,
                  host_pos: int | None = None, sparse_np: int | None = None) -> None:
    residual(b.x[:R], b.x[:R], attn_part(layer, w, st, b, R, nch, host_pos, sparse_np))
    residual(b.x[:R], b.x[:R], ffn_part(layer, w, b, R))


def prompt_layers(w: Weights, st: State, b: Buffers, R: int, nch: int | None, host_pos: int | None,
                  sparse_np: int | None) -> None:
    """A prompt chunk's layers as two micro-batches of rows: each half-layer's collectives run on the comm stream
    while the other micro-batch computes. Rows never depend on their window, so every row keeps its bits (the
    second micro-batch attends the first's keys, written one half-layer earlier)."""

    h = -(-R // 32) * 16                                  # about half the rows, a multiple of 16
    b.pos_mb.copy_(st.pos_dev + h)
    mbs = [(micro_batch(b, 0, h, 0, R), h, st.pos_dev, host_pos),
           (micro_batch(b, h, R, 1, R), R - h, b.pos_mb, None if host_pos is None else host_pos + h)]
    pend = [None, None]
    for layer in w.layers:
        for i, (v, r, pos, hp) in enumerate(mbs):
            if pend[i] is not None:
                residual(v.x[:r], v.x[:r], pend[i])
            pend[i] = attn_part(layer, w, st, v, r, nch, hp, sparse_np, pos)
        for i, (v, r, pos, hp) in enumerate(mbs):
            residual(v.x[:r], v.x[:r], pend[i])
            pend[i] = ffn_part(layer, w, v, r)
    for i, (v, r, pos, hp) in enumerate(mbs):
        residual(v.x[:r], v.x[:r], pend[i])


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


def embed(w: Weights, b, ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The tokens' embedding rows into ``out`` ([R, D] bf16; capturable). Each rank holds one vocabulary span of the
    table (weights.embed_span): it writes the rows it holds and zeros elsewhere, every rank gathers all ranks' rows
    (bf16 pairs as fp32 words, so a decode window's rows take the RDMA gather) and keeps each token's row from the rank
    that holds it. Only copies: the rows of the whole table, bit for bit."""

    R, D = out.shape
    if w.embed.shape[0] == w.cfg.vocab:                     # the whole table: one rank, or TF_GLM_EMBED_SPLIT=0
        from tensorfold.families.glm5_next.cuda import glue as fglue

        return fglue.embed(ids, w.embed, D, 1, out)
    if w.embed.shape[0]:
        glue.embed_span(ids, w.embed, w.embed_lo, out)
    else:                                                  # a vocabulary smaller than a block a rank (tiny configs)
        out.zero_()
    got = b.gath[:w.world * R * D // 2]
    w.comm.all_gather(out.view(torch.float32).view(-1), got)
    return glue.embed_pick(ids, got.view(torch.bfloat16).view(w.world, R, D), out, w.cfg.vocab)


def compute(w: Weights, st: State, b: Buffers, R: int, *, logits: bool = True, nch: int | None = None,
            host_pos: int | None = None, sparse_np: int | None = None):
    """Run capturable GPU work on static buffers and device positions."""

    c = w.cfg
    embed(w, b, b.ids[:R], b.x[:R])
    rope_mod.table(b.cos[:R], b.sin[:R], st.pos_dev, R, c.rope_theta, c.qk_rope)   # device position: graph-safe
    if b.prefill and R >= OVERLAP_ROWS and w.comm is not None and w.world > 2 and b.comm_stream is not None:
        prompt_layers(w, st, b, R, nch, host_pos, sparse_np)
    else:
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
