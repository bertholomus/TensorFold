"""The checkpoint's own draft head on the lane: DSpark, ``mtp_dspark.*`` of the source, kept in BF16 in a file beside the pack
(``TF_ML4_DRAFT``), so the pack's bits stay as they are.

The target's rows leave five taps of the residual stream. ``absorb`` projects a row's taps to one context row, and each of the
three draft layers keeps that row's latent and rotated key in a stream's ring: its last ``WINDOW`` rows. A block [anchor, mask 1,
mask 2] then goes through the layers, every row seeing the ring and the whole block, and its three hidden rows through the
target's head, where a small recurrent head adds what the drafts so far say about the next one. The three tokens are the next
step's rows 1 to 3: a stream keeps what the target itself says of them, so drafts change the speed and not the output.

It runs on every rank the way tp.py's step does: x is the same on all of them, attention split by head and the feed-forward by
column, partials added in rank order, so all ranks hold the same drafts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import gptj_rotate, rms_norm, yarn_inv_freq

BF16, F32 = torch.bfloat16, torch.float32
TAP_LAYERS = (2, 19, 37, 55)          # the residual entering these (the output of layers 1, 18, 36, 54); then the last layer's
DEPTH = 3                             # drafts a block
WINDOW = 2048                         # context rows a block sees
KEYS = WINDOW + DEPTH                 # a slot's rows a block reads: the ring, then the block
ROWS = KEYS + 1                       # ... and one nobody reads, for what no stream keeps
SLOTS = int(os.environ.get("TF_ML4_DRAFT_SLOTS") or 16)   # streams that can draft at once
CHUNK = 256                           # a prompt's rows a pass


@dataclass
class Layer:
    attn_norm: torch.Tensor
    wq: torch.Tensor                  # this rank's heads
    kv_a: torch.Tensor                # whole: every rank keeps every latent
    kv_norm: torch.Tensor
    wuk: torch.Tensor                 # [heads, nope, C]
    wuv: torch.Tensor                 # [heads, v, C]
    wo: torch.Tensor                  # this rank's input columns
    ffn_norm: torch.Tensor
    w1: torch.Tensor                  # this rank's hidden columns
    w3: torch.Tensor
    w2: torch.Tensor


@dataclass
class Keep:
    """What a prompt's round leaves of a window: its last ``n`` rows from ``src``, to ``at`` of ``buf``."""

    buf: torch.Tensor
    src: int
    at: int
    n: int
    index: dict

    def put(self, where: int, x: torch.Tensor) -> None:
        t = self.index.get(where)
        if t is not None:
            self.buf[t, self.at:self.at + self.n].copy_(x[self.src:])


class Window:
    """A prompt's last ``WINDOW`` rows on this rank, kept until its last round has run: they are the first block's context."""

    def __init__(self, stripe, rank: int, prompt: int, slot: int, drafter: Drafter) -> None:
        self.first, self.slot, self.rank, self.drafter, self.at, self.buf = max(0, prompt - WINDOW), slot, rank, drafter, 0, None
        p = torch.arange(self.first, prompt)
        who = stripe.rank(p)
        self.per = [p[who == k] for k in range(stripe.world)]

    def round(self, own: torch.Tensor) -> Keep | None:
        """``own``: a round's rows on this rank, in order, so the window's are the last of them."""

        src = int((own < self.first).sum())
        n = own.numel() - src
        if not n:
            return None
        keep = Keep(self.rows(), src, self.at, n, self.drafter.index)
        self.at += n
        return keep

    def rows(self) -> torch.Tensor:
        if self.buf is None:
            d = self.drafter
            self.buf = torch.zeros((len(d.index), max(t.numel() for t in self.per), d.cfg.dim), dtype=BF16, device=d.dev)
        return self.buf


class Io:
    """A capture's buffers for ``B`` streams, the integers one copy. Filled from pinned memory on the GPU: a copy from anything else
    waits on the host."""

    def __init__(self, B: int, d: int, dev, pinned: bool = False) -> None:
        A, n = B * (DEPTH + 1), B * DEPTH
        self.B, self.A = B, A
        self.meta = torch.zeros(3 * A + 2 * n + 3 * B, dtype=torch.long, device=dev)
        self.tap, self.pos, self.dst, self.bpos, self.brow, self.slot, self.seen, self.anchor = self.meta.split([A, A, A, n, n, B, B, B])
        self.x = torch.zeros((n, d), dtype=F32, device=dev)
        self.first = torch.zeros((B, d), dtype=F32, device=dev)                 # the anchors' rows
        self.host_meta = torch.zeros(self.meta.shape, dtype=torch.long, pin_memory=pinned)
        self.host_first = torch.zeros(self.first.shape, dtype=F32, pin_memory=pinned)
        self.out = None


class Source:
    """The head's tensors in their file: a rank's slices come off it without the rest."""

    def __init__(self, path: str) -> None:
        from safetensors import safe_open

        self.f = safe_open(path, "pt", device="cpu")

    def keys(self) -> list:
        return list(self.f.keys())

    def shape(self, name: str) -> list:
        return self.f.get_slice(name).get_shape()

    def get(self, name: str, rows: slice = slice(None), cols: slice = slice(None)) -> torch.Tensor:
        t = self.f.get_slice(name)
        return (t[rows, cols] if len(t.get_shape()) > 1 else t[rows]).clone()        # GB10: a GPU copy from a mapped file hangs


class Drafter:
    def __init__(self, lane, src, rows: int, graphs=None, taps: tuple = TAP_LAYERS) -> None:
        c, W, r = lane.cfg, lane.world, lane.rank
        dev = lane.pool.device
        self.lane, self.cfg, self.graphs, self.dev, self.caps = lane, c, graphs, dev, {}
        self.mark = lambda name: None
        self.hl = c.n_heads // W
        dl, hw = c.dim // W, self.hl * c.qk_head

        def whole(name: str) -> torch.Tensor:
            return src.get(f"mtp_dspark.{name}").to(dev)

        def rows_of(name: str, n: int) -> torch.Tensor:
            return src.get(f"mtp_dspark.{name}", slice(r * n, (r + 1) * n)).to(dev)

        def cols_of(name: str, n: int) -> torch.Tensor:
            return src.get(f"mtp_dspark.{name}", slice(None), slice(r * n, (r + 1) * n)).contiguous().to(dev)

        self.proj = rows_of("backbone.target_projection.weight", dl)
        self.target_norm, self.final_norm = whole("backbone.target_norm.weight"), whole("backbone.final_norm.weight")
        self.mask = whole("backbone.mask_embedding").float()
        self.layers = []
        for i in range(sum(1 for k in src.keys() if k.endswith(".attention_norm.weight"))):
            p, nv = f"backbone.layers.{i}.", c.nope + c.v_head
            kvb = rows_of(p + "attention.wkv_b.weight", self.hl * nv).view(self.hl, nv, c.kv_lora)
            hidden = src.shape(f"mtp_dspark.{p}feed_forward.w1.weight")[0] // W
            self.layers.append(Layer(whole(p + "attention_norm.weight"), rows_of(p + "attention.wq.weight", hw),
                                     whole(p + "attention.wkv_a_with_mqa.weight"), whole(p + "attention.kv_a_norm.weight"),
                                     kvb[:, :c.nope].contiguous(), kvb[:, c.nope:].contiguous(),
                                     cols_of(p + "attention.wo.weight", self.hl * c.v_head), whole(p + "ffn_norm.weight"),
                                     rows_of(p + "feed_forward.w1.weight", hidden), rows_of(p + "feed_forward.w3.weight", hidden),
                                     cols_of(p + "feed_forward.w2.weight", hidden)))
        self.mk_in = whole("autoregressive_head.embedding.weight")
        self.mk_w, self.mk_b = whole("autoregressive_head.joint_proj.weight"), whole("autoregressive_head.joint_proj.bias")
        self.mk_out = rows_of("autoregressive_head.output.weight", c.vocab // W)
        self.rank_mk = self.mk_in.shape[1]
        self.inv = yarn_inv_freq(c.rd, c.rope_theta, dev, factor=c.rope_factor, original_max=c.orig_len)   # no LeRoPE alpha here
        self.ring = torch.zeros((len(self.layers), SLOTS * ROWS, c.kv_lora + c.rd), dtype=BF16, device=dev)
        self.index = {L: i for i, L in enumerate(taps)} | {c.n_layers: len(taps)}
        self.taps = torch.zeros((len(self.index), rows, c.dim), dtype=BF16, device=dev)
        lane.taps, lane.tap_at = self.taps, self.index

    # -- the pieces ------------------------------------------------------------------------------------------------------

    def _latent(self, lay: Layer, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Rows x [n, d] bf16 at ``pos``: what a layer keeps of each, its latent and rotated key [n, C + rd]."""

        c = self.cfg
        kv = F.linear(x, lay.kv_a)
        lat = rms_norm(kv[:, :c.kv_lora], lay.kv_norm, c.eps)
        return torch.cat([lat, gptj_rotate(kv[:, c.kv_lora:].float(), self.inv, pos)], 1).to(BF16)

    def absorb(self, rows: torch.Tensor, pos: torch.Tensor, dst: torch.Tensor) -> None:
        """The target's rows [n, taps * d] at ``pos`` into every layer's rows ``dst``. Context rows skip a layer's input norm."""

        c, lane = self.cfg, self.lane
        part = F.linear(rows, self.proj).float()
        main = lane._everyone([part[:, None]])[0].reshape(rows.shape[0], c.dim)
        main = rms_norm(main, self.target_norm, c.eps).to(BF16)
        for i, lay in enumerate(self.layers):
            self.ring[i][dst] = self._latent(lay, main, pos)

    def _attend(self, i: int, lay: Layer, q: torch.Tensor, pos: torch.Tensor, slot: torch.Tensor, see: torch.Tensor) -> torch.Tensor:
        """A block's queries [n, hl, qk] over its stream's rows: this rank's heads of the output [n, hl * v]."""

        c, hl = self.cfg, self.hl
        n, B = q.shape[0], slot.numel()
        q_lat = torch.bmm(q[..., :c.nope].transpose(0, 1), lay.wuk).transpose(0, 1)
        q_pe = gptj_rotate(q[..., c.nope:].float().reshape(n * hl, c.rd), self.inv, pos.repeat_interleave(hl)).view(n, hl, c.rd)
        qq = torch.cat([q_lat.float(), q_pe], -1).view(B, -1, c.kv_lora + c.rd)
        keys = self.ring[i].view(SLOTS, ROWS, -1)[slot, :KEYS].float()
        s = torch.bmm(qq, keys.transpose(1, 2)) * c.sm_scale
        p = s.masked_fill(~see[:, None, :], float("-inf")).softmax(-1)
        o = torch.bmm(p, keys[..., :c.kv_lora]).view(n, hl, c.kv_lora).to(BF16)
        return torch.bmm(o.transpose(0, 1), lay.wuv.transpose(1, 2)).transpose(0, 1).reshape(n, hl * c.v_head)

    def block(self, x: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, slot: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        """x [n, d] the blocks' rows at ``pos``, kept at ``rows`` of a stream's ``slot``, ``seen`` [B] rows of its ring: their last
        hidden rows [n, d]."""

        c, lane = self.cfg, self.lane
        at = torch.arange(KEYS, device=x.device)
        see = (at[None, :] < seen[:, None]) | (at[None, :] >= WINDOW)
        n = x.shape[0]
        for i, lay in enumerate(self.layers):
            xn = rms_norm(x, lay.attn_norm, c.eps).to(BF16)
            self.ring[i][rows] = self._latent(lay, xn, pos)
            o = self._attend(i, lay, F.linear(xn, lay.wq).view(n, self.hl, c.qk_head), pos, slot, see)
            x = x + lane._total(F.linear(o, lay.wo).float())
            xn = rms_norm(x, lay.ffn_norm, c.eps).to(BF16)
            x = x + lane._total(F.linear(F.silu(F.linear(xn, lay.w1)) * F.linear(xn, lay.w3), lay.w2).float())
        return rms_norm(x, self.final_norm, c.eps)

    def pick(self, h: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """The drafts [B, DEPTH] of hidden rows h [B * DEPTH, d]: the target's head, softcapped, plus the recurrent head's bias."""

        lane, B = self.lane, anchor.numel()
        base = lane.head_logits(h).view(B, DEPTH, -1)
        h = h.to(BF16).view(B, DEPTH, -1)
        state = torch.zeros((B, self.rank_mk), dtype=BF16, device=h.device)
        prev, out = anchor, []
        for i in range(DEPTH):
            joint = F.linear(torch.cat([state, self.mk_in[prev], h[:, i]], 1), self.mk_w, self.mk_b).float()
            gate, cand, bias = joint.chunk(3, 1)
            gate = torch.sigmoid(gate)
            state = (gate * state.float() + (1.0 - gate) * torch.tanh(cand)).to(BF16)
            prev = lane.best(base[:, i] + F.linear(torch.tanh(bias).to(BF16), self.mk_out).float())
            out.append(prev)
        return torch.stack(out, 1)

    # -- a step ----------------------------------------------------------------------------------------------------------

    def _body(self, io: Io, drafts: bool) -> torch.Tensor:
        self.absorb(self.taps[:, io.tap].permute(1, 0, 2).reshape(io.A, -1), io.pos, io.dst)
        if drafts:
            io.x.view(io.B, DEPTH, -1)[:, 0] = io.first
            return self.pick(self.block(io.x, io.bpos, io.brow, io.slot, io.seen), io.anchor)
        return torch.zeros((io.B, 0), dtype=torch.long, device=self.dev)

    def _capture(self, B: int, drafts: bool) -> Io:
        io = Io(B, self.cfg.dim, self.dev, self.graphs is not None)
        io.x.view(B, DEPTH, -1)[:, 1:] = self.mask
        io.brow.fill_(ROWS - 1)
        io.dst.fill_(ROWS - 1)
        self.caps[B, drafts] = io
        if self.graphs is None:
            return io
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._body(io, drafts)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.graphs.pool):
            io.out = self._body(io, drafts)
        self.graphs.pool = self.graphs.pool or g.pool()
        io.graph = g
        self.graphs.note(("draft", B, drafts))
        return io

    def _first(self, io: Io, tokens: list) -> None:
        io.host_first.copy_(self.lane.w.embed[torch.tensor(tokens)])
        io.first.copy_(io.host_first, non_blocking=True)

    def step(self, plan: list, drafts: bool = True, wait: bool = True) -> list | torch.Tensor:
        """``plan``, a stream each: ``[slot, its first row in the step just run, rows it keeps of that step, the first one's
        position, anchor]``. The kept rows join its ring; with ``drafts`` a block starts at the row after them: each stream's
        drafts. Without, a full step has no room for them, but they may be wanted again. ``wait`` False: the drafts stay on the
        device, [B, DEPTH], for the step that follows."""

        B = 1 << (len(plan) - 1).bit_length()
        io = self.caps.get((B, drafts)) or self._capture(B, drafts)
        host = [[] for _ in range(8)]
        for k in range(B):
            slot, first, kept, at, tok = plan[k] if k < len(plan) else plan[0]
            base = slot * ROWS
            real = k < len(plan)
            for j in range(DEPTH + 1):
                keep = real and j < kept
                host[0].append(first + j if keep else 0)
                host[1].append(at + j if keep else 0)
                host[2].append(base + (at + j) % WINDOW if keep else base + ROWS - 1)
            for j in range(DEPTH):
                host[3].append(at + kept + j)
                host[4].append(base + WINDOW + j if real else base + ROWS - 1)
            host[5].append(slot)
            host[6].append(at + kept)
            host[7].append(tok)
        io.host_meta.copy_(torch.tensor([v for h in host for v in h]))
        self._first(io, host[7])
        self.mark("dfill")
        io.meta.copy_(io.host_meta, non_blocking=True)
        if self.graphs is None:
            out = self._body(io, drafts)
        else:
            io.graph.replay()
            out = io.out
        return out[:len(plan)].tolist() if wait else out

    def close(self, w: Window) -> None:
        """A prompt's last round is done on every rank: its window goes into the stream's ring, a chunk at a time."""

        lane, rows = self.lane, w.rows()
        every = torch.empty((lane.world, *rows.shape), dtype=BF16, device=self.dev)
        lane.comm.all_gather(rows.view(-1).view(F32), every.view(-1).view(F32))
        for k, own in enumerate(w.per):
            for a in range(0, own.numel(), CHUNK):
                p = own[a:a + CHUNK].to(self.dev)
                rows = every[k, :, a:a + p.numel()].permute(1, 0, 2).reshape(p.numel(), -1)
                self.absorb(rows, p, w.slot * ROWS + p % WINDOW)
        w.buf = None
