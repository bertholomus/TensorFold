"""Beam-1.0's forward on one tensor-parallel rank.

A layer: the attention block (q, k, v and gate projections; per-head q/k RMSNorm with RoPE on window layers or SSMax
on full ones, the keys and values written to the layer's store; windowed GQA with the sigmoid output gate; o_proj),
the sandwich-norm boundary (post-attention norm, depth-scaled residual add, pre-feedforward norm), the dense or routed
SwiGLU, and the second boundary (post-feedforward norm, residual add, the next layer's input norm). o_proj's and the
MLP's partial sums are gathered across ranks and added in rank order, so every rank holds the same bits."""

from __future__ import annotations

import os

import torch

from tensorfold.cuda.exl3 import experts as exl3_experts
from tensorfold.cuda.exl3 import prefill as exl3_prefill
from tensorfold.families.beam.cuda import attention as A
from tensorfold.families.beam.cuda import ops
from tensorfold.families.beam.cuda.cache import Batch, Cache
from tensorfold.families.beam.cuda.weights import DenseW, LayerW, MoEW, Weights

DECODE_ROWS = 128             # Exl3Linear's rows; larger windows take the prompt matmul
SMALL_EXPERT_ROWS = 64        # windows under this take the experts' decode path (exl3.experts.EXACT_ROWS)
FULL = 1 << 30                # a full layer's window
# prompt chunks of this many rows or more on two ranks or more (TF_BEAM_SPLIT; 0: off) gather each rank partial in
# TF_BEAM_SPLIT_PIECES row pieces on a high-priority side stream, each piece as soon as it is made: o_proj's pieces
# travel while the compute stream makes the next ones, and the MLP's while it runs the boundaries, the router and the
# next layer's q and gate projections of the pieces already gathered (each decoded W_q kept for its layer's pieces:
# Workspace.keep). Every kernel runs on the same rows with the same arithmetic (the prompt matmul, the boundary
# and the router are row-invariant from 129 rows; attention and the experts take the whole chunk): the same bits.
SPLIT = int(os.environ.get("TF_BEAM_SPLIT") or 0)
PIECES = int(os.environ.get("TF_BEAM_SPLIT_PIECES") or 4)
# split chunks of an even number of rows on two ranks (TF_BEAM_RS): a reduce-scatter at each boundary instead of the
# gather (forward_rs): each rank adds and normalizes half of every piece's rows and sends the other rank its rows. 1: the
# partials swapped point to point and added by the boundary; 2: NCCL's reduce-scatter adds them (two ranks: one add)
RS = int(os.environ.get("TF_BEAM_RS") or 0)
KEEP_MB = 128                 # the pieces' decoded W_q: a layer's q, gate and o_proj (94 MB at TP2)


class Stage:
    """Rows of a host table on their way to the device: gathered into pinned memory, one async copy (its event keeps
    the next gather off rows still in flight)."""

    def __init__(self) -> None:
        self.pin = None
        self.done = None

    def rows(self, table: torch.Tensor, ids: torch.Tensor, device) -> torch.Tensor:
        n = ids.numel()
        if self.done is not None:
            self.done.synchronize()
        if self.pin is None or self.pin.shape[0] < n:
            self.pin = torch.empty((max(n, 64), table.shape[1]), dtype=table.dtype, pin_memory=True)
        torch.index_select(table, 0, ids, out=self.pin[:n])
        out = self.pin[:n].to(device, non_blocking=True)
        self.done = torch.cuda.Event()
        self.done.record()
        return out


class Work:
    """One rank's scratch: the prompt matmul's workspace, the experts' decode and prompt scratch, attention partials,
    the rotary frequencies, the gather buffer, the host rows' staging. The experts' prompt scratch, the router's fp32
    input and the gather buffer share one buffer (``share``)."""

    def __init__(self, w: Weights, comm=None, *, rows: int, small_rows: int = 32, device="cuda") -> None:
        self.cfg = w.cfg
        self.comm = comm
        self.world = w.world
        self.ws = exl3_prefill.Workspace()
        self.att = A.Scratch()
        self.inv = ops.inv_freq(w.cfg.rope_theta, w.cfg.head_dim, device)
        ex = next((lw.mlp.ex for lw in w.layers.values() if isinstance(lw.mlp, MoEW)), None)
        self.ex_small = exl3_experts.Scratch(ex, small_rows, w.cfg.top_k, device=device) if ex is not None else None
        self.ex_big = (exl3_experts.Scratch(ex, rows, w.cfg.top_k, device=device, prompt=True)
                       if ex is not None and rows >= SMALL_EXPERT_ROWS else None)
        self.fold = fold_z(self.ex_big, ex, rows, w.cfg.top_k) if self.ex_big is not None else None
        self.small_rows = small_rows
        self.gather = None
        self.yf = None
        self.picks = None                 # debug (TF_BEAM_TIMES): a list that collects each MoE layer's expert picks
        self.stage = Stage()
        self.side = self.ws_keep = self.split_buf = self.split_outs = None
        if SPLIT and self.world > 1:
            self.side = torch.cuda.Stream(device=device, priority=-1)
            self.ws_keep = exl3_prefill.Workspace(keep=exl3_prefill.UnpackCache(KEEP_MB << 20))
        if self.ex_big is not None:
            self.share(ex, rows, w.cfg.top_k)

    def share(self, ex, rows: int, top_k: int) -> None:
        """One buffer for scratch that is never live at once, in stream order (NCCL runs on the compute stream): the
        experts' z (gate/up's sums from 0, the rotated rows xg and xu behind them, which down's sums later overwrite;
        with mma3 no rotated rows, and down3's bf16 slot outputs over gate/up's sums), and from its start the gather
        buffer (read by the rank-order sum before anything else writes) and the router's fp32 input (written by the
        attention boundary, read by the router before the experts). The same kernels write the same bits; 0.3 GB
        fewer at 2,048 rows (mma3: 0.5 GB fewer than mma's arena, 168 MB at 2,048 rows)."""
        s, X = self.ex_big, exl3_experts
        D, I, P = ex.dims, ex.width, rows * top_k
        zgu = max(2 * I * P, max(2 * s.cfg_gu[2] * I, s.cfg_d[2] * D) * (X.EXACT_ROWS - 1) * top_k)
        rot = 0 if self.fold == "mma3" else D * P           # xg and xu (P x D halves each)
        n = max(zgu + rot, s.z.numel(), self.world * max(rows, 64) * D)
        s.z = s.xg = s.xu = None
        torch.cuda.empty_cache()
        arena = torch.zeros((n,), dtype=torch.float32, device=self.inv.device)
        s.z = arena
        if rot:
            half = D * P // 2
            s.xg = arena[zgu:zgu + half].view(torch.float16).view(P, D)
            s.xu = arena[zgu + half:zgu + 2 * half].view(torch.float16).view(P, D)
        self.gather = arena
        self.yf = arena[:rows * D].view(rows, D)

    def router_in(self, rows: int) -> torch.Tensor:
        """fp32 [rows, hidden] for the router's unrounded input (grown to the largest call)."""
        if self.yf is None or self.yf.shape[0] < rows:
            self.yf = torch.empty((rows, self.cfg.hidden), dtype=torch.float32, device=self.inv.device)
        return self.yf[:rows]

    def parts(self, part: torch.Tensor) -> torch.Tensor:
        """The ranks' fp32 partial sums gathered in rank order, [world, R, D] (ops.boundary adds them as reduce does);
        one rank: part itself."""
        if self.world == 1:
            return part
        R, D = part.shape
        if self.gather is None or self.gather.numel() < self.world * R * D:
            self.gather = torch.empty((self.world * max(R, 64) * D,), dtype=torch.float32, device=part.device)
        recv = self.gather[:self.world * R * D].view(self.world, R, D)
        self.comm.all_gather(part.contiguous(), recv)
        return recv

    def pieces(self, R: int) -> list[tuple[slice, torch.Tensor]]:
        """A split chunk's row pieces (PIECES, each a multiple of 128 rows but the last, 256 rows or more) and their
        gather buffers [world, rows, hidden]: behind the router's fp32 input in the shared buffer (the experts' scratch
        is not live while a gather writes or a boundary reads them), else in a buffer of their own."""
        D, W = self.cfg.hidden, self.world
        k = max(1, min(PIECES, R // 256))
        step = R // k // 128 * 128
        bounds = [i * step for i in range(k)] + [R]
        n = W * R * D
        if self.gather is not None and self.gather.numel() >= R * D + n:
            base, off = self.gather, R * D
        else:
            if self.split_buf is None or self.split_buf.numel() < n:
                self.split_buf = torch.empty((n,), dtype=torch.float32, device=self.inv.device)
            base, off = self.split_buf, 0
        return [(slice(a, b), base[off + W * a * D:off + W * b * D].view(W, b - a, D))
                for a, b in zip(bounds, bounds[1:])]

    def outs(self, R: int, shapes: dict) -> dict:
        """A split chunk's outputs that the side stream reads or the next layer takes, one buffer each for every layer
        (tensors made a layer and freed behind the side stream's reads kept torch's allocator growing after the
        warm-up: +1.25 GiB of reserved memory over a few 8,192-row prompts): o_proj's partial "attn" (behind the
        gather buffers in the shared buffer when it has room: the experts' scratch is not live while it is), the
        MLP's partial, the router's logits, the next layer's q and gate. ``shapes``: name -> (columns, dtype); views
        of R rows."""
        D, at = self.cfg.hidden, (self.world + 1) * R * self.cfg.hidden
        out = {}
        if self.gather is not None and self.gather.numel() >= at + R * D:
            out["attn"] = self.gather[at:at + R * D].view(R, D)
        bufs = self.split_outs = self.split_outs or {}
        for name, (cols, dtype) in shapes.items():
            if name in out:
                continue
            t = bufs.get(name)
            if t is None or t.shape[0] < R or t.shape[1] != cols or t.dtype != dtype:
                t = bufs[name] = torch.empty((R, cols), dtype=dtype, device=self.inv.device)
            out[name] = t[:R]
        return out

    def gather_side(self, part: torch.Tensor, recv: torch.Tensor) -> torch.cuda.Event:
        """recv [world, rows, hidden] <- every rank's ``part`` (contiguous rows) on the side stream, after the compute
        stream's work so far; the event marks its end (the compute stream waits on it before it reads recv). Every
        NCCL call of a split chunk runs on the side stream, so none runs beside another."""
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            self.comm.all_gather(part, recv)
            done = torch.cuda.Event()
            done.record()
        part.record_stream(self.side)
        return done

    def halves(self, R: int) -> list[tuple[slice, slice, slice, torch.Tensor]]:
        """forward_rs' pieces (pieces()' rows; R even, so every piece is): a piece's rows, this rank's half of them, the
        other rank's half, and a buffer [half, hidden] fp32 for the other rank's partial of this rank's half."""
        out = []
        for rows, buf in self.pieces(R):
            a, h = rows.start, (rows.stop - rows.start) // 2
            lo, hi = slice(a, a + h), slice(a + h, rows.stop)
            mine, other = (lo, hi) if self.comm.rank == 0 else (hi, lo)
            out.append((rows, mine, other, buf[0, :h]))
        return out

    def swap_side(self, send: torch.Tensor, recv: torch.Tensor, after: torch.cuda.Event | None = None
                  ) -> torch.cuda.Event:
        """Two ranks: ``send`` (this rank's partial of the other rank's half of a piece) to the other rank and its
        partial of this rank's half into ``recv``, on the side stream after the compute stream's work so far (after:
        only after that event of it)."""
        if after is None:
            self.side.wait_stream(torch.cuda.current_stream())
        else:
            self.side.wait_event(after)
        peer = 1 - self.comm.rank
        with torch.cuda.stream(self.side):
            self.comm.grouped([(send, peer)], [(recv, peer)])
            done = torch.cuda.Event()
            done.record()
        send.record_stream(self.side)
        recv.record_stream(self.side)
        return done

    def scatter_side(self, part: torch.Tensor, mine: torch.Tensor, after: torch.cuda.Event | None = None
                     ) -> torch.cuda.Event:
        """Two ranks (RS 2): a piece's partials ``part`` [rows, hidden] summed into this rank's half ``mine`` (its rows
        of part: in place) by NCCL's reduce-scatter, on the side stream as swap_side runs."""
        if after is None:
            self.side.wait_stream(torch.cuda.current_stream())
        else:
            self.side.wait_event(after)
        with torch.cuda.stream(self.side):
            self.comm.reduce_scatter(part, mine)
            done = torch.cuda.Event()
            done.record()
        part.record_stream(self.side)
        return done

    def gather_half(self, ts, rows: slice, mine: slice) -> torch.cuda.Event:
        """Each of ``ts`` [R, columns]: this rank's half of a piece's rows to the other rank and the other half back,
        in place (an all-gather of the piece's rows), on the side stream after the compute stream's work so far."""
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            for t in ts:
                self.comm.all_gather(t[mine], t[rows])
            done = torch.cuda.Event()
            done.record()
        return done

    def reduce(self, part: torch.Tensor) -> torch.Tensor:
        """The ranks' fp32 partial sums added in rank order (every rank the same bits)."""
        if self.world == 1:
            return part
        recv = self.parts(part)
        out = recv[0] + recv[1]
        for r in range(2, self.world):
            out += recv[r]
        return out


def fold_z(s, ex, rows: int, top_k: int) -> str | None:
    """Size the prompt experts' partial sums for the kernel that runs them. Prompt chunks (EXACT_ROWS rows or more)
    take grouped_mma, which adds gate/up's K splits inside its programs and writes one split (2 P I floats, then
    down's P D), or grouped_mma3 (TF_EXL3_PROMPT_KERNEL=mma3: gate/up's 2 P I floats, then down3's bf16 slot outputs
    over them, P D / 2 floats' bytes); the Scratch holds every split's (640 MiB at 2,048 rows; mma needs 320, mma3
    160). Calls of fewer rows keep the split layout, which fits in the smaller buffer; routed() checks each call's
    need. Nothing else changes: the same kernels write the same bits. Returns the kernel folded for (None: as is)."""

    X = exl3_experts
    D, I, P = ex.dims, ex.width, rows * top_k
    gu, dn = s.cfg_gu, s.cfg_d
    small = max(2 * gu[2] * I, dn[2] * D) * (X.EXACT_ROWS - 1) * top_k
    fast = X.PROMPT and all(n % (16 * t[0]) == 0 for n, t in ((I, X.PROMPT_TILES["gateup"]),
                                                               (D, X.PROMPT_TILES["down"])))
    if fast and X.PROMPT_KERNEL == "mma3" and I % 256 == 0 and D % 256 == 0 and max(ex.k2_gu[1], ex.k2_d[1]) <= 8:
        kind, need = "mma3", max(2 * I * P, D * P // 2, small)
    elif (fast and X.PROMPT_KERNEL == "mma" and I % 64 == 0 and D % 64 == 0
          and all((k // 16) % (c[1] * c[2] * X.MMA_KB) == 0 for k, c in ((D, gu), (I, dn)))):
        kind, need = "mma", max(max(2 * I, D) * P, small)
    else:
        return None
    if need < s.z.numel():
        dev = s.z.device
        s.z = None
        torch.cuda.empty_cache()              # the old buffer's segment goes back whole, not split under the new one
        s.z = torch.zeros((need,), dtype=torch.float32, device=dev)
    return kind


def mm(lin, x: torch.Tensor, work: Work, out_dtype=torch.bfloat16, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty((x.shape[0], lin.n), dtype=out_dtype, device=x.device)
    if x.shape[0] <= DECODE_ROWS:
        return lin(x, out=out)
    return exl3_prefill.matmul(lin, x.contiguous(), out, work.ws)


def attention(lw: LayerW, y: torch.Tensor, b: Batch, cache: Cache, work: Work, parts: bool = False) -> torch.Tensor:
    """The attention block's output, all ranks' partials added: fp32 [R, hidden] (parts: the gathered partials,
    [world, R, hidden], for ops.boundary to add)."""
    a = lw.attn
    o = attend(lw, (mm(a.q, y, work), mm(a.k, y, work), mm(a.v, y, work), mm(a.g, y, work)), b, cache, work)
    out = mm(a.o, o, work, torch.float32)
    return work.parts(out) if parts else work.reduce(out)


def attend(lw: LayerW, qkvg: tuple, b: Batch, cache: Cache, work: Work) -> torch.Tensor:
    """The attention block from its projections (q, k, v, gate) to o_proj's input: per-head q/k norms with RoPE or
    SSMax, the keys and values stored, windowed or full attention with the sigmoid gate; bf16 [R, rank heads x 128]."""
    cfg, a, R = work.cfg, lw.attn, b.rows
    q, k, v, g = qkvg
    hq = q.shape[1] // 128
    qo = torch.empty((R, hq, 128), dtype=torch.bfloat16, device=q.device)
    if lw.sliding:
        i = cache.win[lw.index]
        if b.chunk:
            kb, vb = cache.k_buf, cache.v_buf
            if b.buf_from.numel():
                kb[b.buf_to] = cache.k_ring[i][b.buf_from]
                vb[b.buf_to] = cache.v_ring[i][b.buf_from]
            dst = torch.arange(cache.back, cache.back + R, device=q.device, dtype=torch.int64)
            ops.qk_prep(q, k, v, a.q_norm, a.k_norm, b.pos, work.inv, qo, kb, vb, dst, mode=ops.ROPE, eps=cfg.eps)
            o = A.attention(qo, kb, vb, b.pos, b.row0, window=cfg.window, mode=A.LINEAR, max_rows=R,
                            span=min(cfg.window, b.span), base=b.p0 - cache.back, gate=g, gate_bias=a.gate_bias,
                            scratch=work.att)
            cache.k_ring[i][b.keep_dst] = kb[b.keep_rows]
            cache.v_ring[i][b.keep_dst] = vb[b.keep_rows]
        else:
            ops.qk_prep(q, k, v, a.q_norm, a.k_norm, b.pos, work.inv, qo, cache.k_ring[i], cache.v_ring[i],
                        b.ring_dst, mode=ops.ROPE, eps=cfg.eps)
            o = A.attention(qo, cache.k_ring[i], cache.v_ring[i], b.pos, b.row0, window=cfg.window, mode=A.RING,
                            max_rows=b.max_rows, span=min(cfg.window, b.span), table=b.ring_start,
                            ring_n=cache.ring, gate=g, gate_bias=a.gate_bias, scratch=work.att, part=A.PART)
    else:
        i = cache.full[lw.index]
        ops.qk_prep(q, k, v, a.q_norm, a.k_norm, b.pos, work.inv, qo, cache.k_full[i], cache.v_full[i], b.full_dst,
                    mode=ops.SSMAX, eps=cfg.eps)
        o = A.attention(qo, cache.k_full[i], cache.v_full[i], b.pos, b.row0, window=FULL, mode=A.PAGED,
                        max_rows=b.max_rows, span=b.span, table=b.table, page=cache.page, gate=g,
                        gate_bias=a.gate_bias, scratch=work.att, part=0 if b.chunk else A.PART)
    return o.view(R, hq * 128)


def mlp(lw: LayerW, y: torch.Tensor, work: Work, yf: torch.Tensor | None = None, parts: bool = False) -> torch.Tensor:
    """The dense or routed SwiGLU's output, all ranks' partials added: fp32 [R, hidden] (parts: the gathered
    partials, [world, R, hidden]). yf: the router's input unrounded (fp32), else y."""
    return (work.parts if parts else work.reduce)(mlp_part(lw, y, work, yf))


def mlp_part(lw: LayerW, y: torch.Tensor, work: Work, yf: torch.Tensor | None = None,
             logits: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """This rank's partial of the dense or routed SwiGLU, fp32 [R, hidden] (logits: the router's, made already; out:
    where it goes, else a new tensor)."""
    cfg, m, R = work.cfg, lw.mlp, y.shape[0]
    if isinstance(m, DenseW):
        act = ops.silu_mul(mm(m.gate, y, work), mm(m.up, y, work))
        return mm(m.down, act, work, torch.float32, out=out)
    if logits is None:
        logits = ops.router_logits(y if yf is None else yf, m.router)
    idx, wt = ops.route(logits, m.bias, groups=cfg.groups, top_groups=cfg.top_groups, top_k=cfg.top_k)
    wt = wt * cfg.routed_scale
    if work.picks is not None:
        work.picks.append(idx)
    s = work.ex_small if R <= work.small_rows else work.ex_big
    return exl3_experts.routed(y, idx, wt, m.ex, s, out, R, act_mode=exl3_experts.ACT_F32,
                               y_unused=True)        # only the weighted sum is read: decode skips the per-slot outputs


def layer(lw: LayerW, x: torch.Tensor, y: torch.Tensor, b: Batch, cache: Cache, work: Work,
          next_norm: torch.Tensor | None):
    """(residual, normed input) -> the next (residual, normed input); next_norm None: y not made."""
    L = lw.index
    eps = work.cfg.eps
    yf = work.router_in(y.shape[0]) if isinstance(lw.mlp, MoEW) else None
    # the boundaries add the ranks' gathered partials themselves, in rank order (no summed copy; the same bits)
    x, y = ops.boundary(attention(lw, y, b, cache, work, parts=True), x, lw.ln_post_attn, lw.ln_pre_ff,
                        ops.scales(L, attention=True), eps, yf_out=yf)
    x, y = ops.boundary(mlp(lw, y, work, yf, parts=True), x, lw.ln_post_ff, next_norm, ops.scales(L, attention=False),
                        eps)
    return x, y


def embed(w: Weights, ids: torch.Tensor, work: Work, res_dtype=torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows of ``ids`` and their normed input. A host table (host ids) gives each rank its rows from the page
    cache; else each rank looks the ids up in its vocabulary slice (even or not), one all-gather brings every rank's
    rows, and each row is taken whole from the rank whose slice holds its id (exact copies either way)."""
    if w.embed_host is not None:
        x = work.stage.rows(w.embed_host, ids.cpu(), work.inv.device)
        return x.to(res_dtype), ops.rms_norm(x, w.layers[0].ln_in, w.cfg.eps)
    ids = ids.to(work.inv.device).long()
    if w.world == 1:
        x = w.embed[ids]
    else:
        n = w.embed.shape[0]
        part = w.embed[(ids - w.head_lo).clamp(0, n - 1)]           # other ranks' ids give rows nobody takes
        recv = torch.empty((w.world,) + tuple(part.shape), dtype=part.dtype, device=part.device)
        work.comm.all_gather(part, recv)
        starts = torch.tensor(w.vocab_starts[1:], dtype=ids.dtype, device=ids.device)
        owner = (ids[:, None] >= starts[None, :]).sum(1)
        x = recv[owner, torch.arange(ids.numel(), device=ids.device)]
    x = x.to(res_dtype)
    return x, ops.rms_norm(x, w.layers[0].ln_in, w.cfg.eps)


def forward(w: Weights, ids: torch.Tensor, b: Batch, cache: Cache, work: Work,
            res_dtype=torch.float32, capture: tuple[int, ...] = (), tap=None) -> torch.Tensor:
    """Token ids of the batch's rows -> the final normed hidden state [R, hidden] bf16. The residual stream stays fp32
    by default (unrounded adds; layer checks vs a reference forward put KL at 0.0078, against 0.0115 with bf16 adds).
    capture: layer ids whose incoming residual (0: the embedding) goes to tap(i, x), i its index in capture (a
    drafter's target states)."""
    x, y = embed(w, ids, work, res_dtype)
    if work.side is not None and b.chunk and b.rows >= SPLIT:
        if RS and work.world == 2 and b.rows % 2 == 0:
            return forward_rs(w, x, y, b, cache, work, capture, tap)
        return forward_split(w, x, y, b, cache, work, capture, tap)
    n = w.cfg.layers
    for L in range(n):
        if L in capture:
            tap(capture.index(L), x)
        nxt = w.layers[L + 1].ln_in if L + 1 < n else w.norm
        x, y = layer(w.layers[L], x, y, b, cache, work, nxt)
    return y


def forward_split(w: Weights, x: torch.Tensor, y: torch.Tensor, b: Batch, cache: Cache, work: Work,
                  capture: tuple[int, ...], tap) -> torch.Tensor:
    """forward() of a prompt chunk with each rank gather in row pieces on the side stream (SPLIT): the same layers,
    kernels and rows, so the same bits; o_proj's pieces travel while the next ones are made, and the MLP's while the
    boundaries and the next layer's projections of the pieces already gathered run (the first layer's projections and
    the attention and MLP themselves take the whole chunk). The boundaries write the residual and the normed rows in
    place (each row read before it is written; nothing reads the old ones after), and the outputs the side stream
    reads or the next layer takes live in Work.outs' buffers, so a layer makes no tensor of the chunk's size."""
    cfg, R, n, ws = w.cfg, b.rows, w.cfg.layers, work.ws_keep
    D, eps = cfg.hidden, cfg.eps
    cur = torch.cuda.current_stream()
    pieces = work.pieces(R)
    a0 = w.layers[0].attn
    bufs = work.outs(R, {"attn": (D, torch.float32), "mlp": (D, torch.float32), "logit": (cfg.experts, torch.float32),
                         "q": (a0.q.n, torch.bfloat16), "g": (a0.g.n, torch.bfloat16)})
    proj = None
    for L in range(n):
        if L in capture:
            tap(capture.index(L), x)
        lw = w.layers[L]
        a = lw.attn
        if proj is None:
            proj = (mm(a.q, y, work), mm(a.k, y, work), mm(a.v, y, work), mm(a.g, y, work))
        o = attend(lw, proj, b, cache, work)
        part = bufs["attn"]
        done = []
        for rows, recv in pieces:
            exl3_prefill.matmul(a.o, o[rows], part[rows], ws)
            done.append(work.gather_side(part[rows], recv))
        moe = isinstance(lw.mlp, MoEW)
        yf = work.router_in(R) if moe else None
        logit = bufs["logit"] if moe else None
        sc = ops.scales(L, attention=True)
        for (rows, recv), e in zip(pieces, done):
            cur.wait_event(e)
            ops.boundary(recv, x[rows], lw.ln_post_attn, lw.ln_pre_ff, sc, eps, h_out=x[rows], y_out=y[rows],
                         yf_out=yf[rows] if moe else None)
            if moe:
                ops.router_logits(yf[rows], lw.mlp.router, out=logit[rows])
        part = mlp_part(lw, y, work, yf, logit, out=bufs["mlp"])
        done = [work.gather_side(part[rows], recv) for rows, recv in pieces]
        nxt = w.layers[L + 1] if L + 1 < n else None
        # the next layer's q and gate projections by pieces (k and v, 512 columns, fill the GPU only on whole chunks)
        lins = () if nxt is None else (nxt.attn.q, nxt.attn.g)
        qg = [bufs[k] if bufs[k].shape[1] == lin.n else torch.empty((R, lin.n), dtype=torch.bfloat16, device=y.device)
              for k, lin in zip("qg", lins)]
        sc = ops.scales(L, attention=False)
        for (rows, recv), e in zip(pieces, done):
            cur.wait_event(e)
            ops.boundary(recv, x[rows], lw.ln_post_ff, nxt.ln_in if nxt is not None else w.norm, sc, eps,
                         h_out=x[rows], y_out=y[rows])
            for lin, out in zip(lins, qg):
                exl3_prefill.matmul(lin, y[rows], out[rows], ws)
        proj = None if nxt is None else (qg[0], mm(nxt.attn.k, y, work), mm(nxt.attn.v, y, work), qg[1])
    return y


def forward_rs(w: Weights, x: torch.Tensor, y: torch.Tensor, b: Batch, cache: Cache, work: Work,
               capture: tuple[int, ...], tap) -> torch.Tensor:
    """forward_split() with a reduce-scatter at each boundary (RS: two ranks, an even chunk). For every piece each rank
    sends the other its partial of the other's half of the rows (half a gather's bytes) and gets the other's partial
    of its own half, which the boundary adds in rank order as before (RS 2: NCCL's reduce-scatter adds the pair: one
    add, the same bits); the boundary and the router run on that half only, and an all-gather in place brings the
    other half's normed rows (bf16) and router logits. The residual (fp32) stays with its half's rank; a layer the
    drafter taps gathers its incoming residual first, rounded to bf16 as the tap rounds it. Every row gets the
    arithmetic it gets in forward_split: the same bits."""
    cfg, R, n, ws = w.cfg, b.rows, w.cfg.layers, work.ws_keep
    D, eps = cfg.hidden, cfg.eps
    cur = torch.cuda.current_stream()
    first = work.comm.rank == 0
    pieces = work.halves(R)
    a0 = w.layers[0].attn
    bufs = work.outs(R, {"attn": (D, torch.float32), "mlp": (D, torch.float32), "logit": (cfg.experts, torch.float32),
                         "q": (a0.q.n, torch.bfloat16), "g": (a0.g.n, torch.bfloat16),
                         **({"xb": (D, torch.bfloat16)} if capture else {})})
    proj, xt = None, x                        # xt: every row's incoming residual of a tapped layer

    def exchange(i, after=None):              # piece i's partials of each half to the rank that adds them
        rows, mine, other, recv = pieces[i]
        if RS == 2:
            return work.scatter_side(part[rows], part[mine], after)
        return work.swap_side(part[other], recv, after)

    def summed(i):                            # the boundary's input for this rank's half of piece i
        rows, mine, other, recv = pieces[i]
        if RS == 2:
            return part[mine]                 # the sum (NCCL's one add)
        return (part[mine], recv) if first else (recv, part[mine])   # both partials in rank order

    for L in range(n):
        if L in capture:
            tap(capture.index(L), xt)
        lw = w.layers[L]
        a = lw.attn
        if proj is None:
            proj = (mm(a.q, y, work), mm(a.k, y, work), mm(a.v, y, work), mm(a.g, y, work))
        o = attend(lw, proj, b, cache, work)
        part = bufs["attn"]
        done = []
        for i, (rows, mine, other, recv) in enumerate(pieces):
            exl3_prefill.matmul(a.o, o[rows], part[rows], ws)
            done.append(exchange(i))
        moe = isinstance(lw.mlp, MoEW)
        yf = work.router_in(R) if moe else None
        logit = bufs["logit"] if moe else None
        sc = ops.scales(L, attention=True)
        back = []
        for i, ((rows, mine, other, recv), e) in enumerate(zip(pieces, done)):
            cur.wait_event(e)
            ops.boundary(summed(i), x[mine], lw.ln_post_attn, lw.ln_pre_ff, sc, eps, h_out=x[mine], y_out=y[mine],
                         yf_out=yf[mine] if moe else None)
            if moe:
                ops.router_logits(yf[mine], lw.mlp.router, out=logit[mine])
            back.append(work.gather_half((y, logit) if moe else (y,), rows, mine))
        for e in back:
            cur.wait_event(e)
        part = mlp_part(lw, y, work, yf, logit, out=bufs["mlp"])
        ready = torch.cuda.Event()
        ready.record()
        nxt = w.layers[L + 1] if L + 1 < n else None
        lins = () if nxt is None else (nxt.attn.q, nxt.attn.g)
        qg = [bufs[k] if bufs[k].shape[1] == lin.n else torch.empty((R, lin.n), dtype=torch.bfloat16, device=y.device)
              for k, lin in zip("qg", lins)]
        xb = bufs["xb"] if L + 1 in capture else None
        sc = ops.scales(L, attention=False)

        def project(i):                       # the next layer's q and gate of piece i, once its rows are back
            cur.wait_event(back[i])
            for lin, out in zip(lins, qg):
                exl3_prefill.matmul(lin, y[pieces[i][0]], out[pieces[i][0]], ws)

        # one swap ahead of the boundaries and each piece's gather right behind its boundary (the side stream runs
        # swap 0, swap 1, gather 0, swap 2, gather 1, ...), and a piece's projections behind the next boundary
        done, back = [exchange(0, ready)], []
        for i, (rows, mine, other, recv) in enumerate(pieces):
            if i + 1 < len(pieces):
                done.append(exchange(i + 1, ready))
            cur.wait_event(done[i])
            ops.boundary(summed(i), x[mine], lw.ln_post_ff, nxt.ln_in if nxt is not None else w.norm, sc, eps,
                         h_out=x[mine], y_out=y[mine])
            if xb is not None:
                xb[mine].copy_(x[mine])
            back.append(work.gather_half((y,) if xb is None else (y, xb), rows, mine))
            if i:
                project(i - 1)
        project(len(pieces) - 1)
        if xb is not None:
            xt = xb
        proj = None if nxt is None else (qg[0], mm(nxt.attn.k, y, work), mm(nxt.attn.v, y, work), qg[1])
    return y


def logits(w: Weights, y: torch.Tensor) -> torch.Tensor:
    """This rank's vocabulary slice of the soft-capped logits, fp32 [R, rank's vocab]."""
    out = torch.mm(y, w.head.t(), out_dtype=torch.float32)
    return ops.softcap(out, w.cfg.softcap) if w.cfg.softcap else out
