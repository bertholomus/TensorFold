"""Ling-3.1-flash's decode loops: serial and MTP-drafted rounds on one sequence's state, TP2 rank-mirrored sampling.

Accepts drafts only when they match the serial keyed sample; all ranks sample identical gathered candidates without
a broadcast. The MTP head drafts through the shared head (bailing_v3.cuda.mtp): a draft step 1 reads the main
forward's final-normed rows and steps 2-3 the previous MTP call's; nothing is zeroed at position 0. No DFlash2 and
no sparse path: Ling's MLA is dense at every length (bailing_v3 State.index is None).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.sampling import comm_gather, nucleus_rows, one_rank
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows
from tensorfold.families.glm5_next.cuda import latent, prof

from .forward import Buffers, State, chunks_for, commit, compute, stage
from .mtp import mtp_compute, mtp_forward, mtp_stage
from .weights import Weights


def sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                offset: int | None = None, probs: list[float] | None = None) -> list[int]:
    """Rows of (this rank's vocabulary slice of) logits at their absolute positions -> tokens, same on all ranks."""

    R = logits.shape[0]
    greedy = sampling is None or sampling.temperature <= 0
    if not greedy and not sampling.top_k:           # top_k off: the shared nucleus rule over every rank's shard
        return nucleus_rows(logits, positions, sampling, offset=w.vocab_offset if offset is None else offset,
                            gather=one_rank if w.comm is None else comm_gather(w.comm), probs=probs)
    k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
    if probs is not None and greedy:
        k = min(logits.shape[1], 20 + MARGIN)       # the draft's confidence needs its competitors too
    vals, ids = torch.topk(logits.float(), k, dim=-1)
    ids = (ids + (w.vocab_offset if offset is None else offset)).to(torch.int32)
    if w.comm is None:
        values = vals.cpu().numpy().astype(np.float32)
        tokens = ids.cpu().numpy().astype(np.int64)
    else:
        packed = torch.cat([vals, ids.view(torch.float32)], dim=1).contiguous()
        got = torch.empty((w.world * packed.numel(),), dtype=torch.float32, device=logits.device)
        w.comm.all_gather(packed.view(-1), got)
        g = got.view(w.world, R, 2 * k).cpu()
        values = torch.cat([g[r, :, :k] for r in range(w.world)], dim=1).numpy().astype(np.float32)
        tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(w.world)], dim=1).numpy()
        tokens = tokens.astype(np.int64)
    if greedy:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if probs is not None:
        probs.extend(_probability(values, tokens, chosen, sampling))
    return chosen


def _probability(values: np.ndarray, tokens: np.ndarray, chosen: list[int], sampling: Sampling | None) -> list[float]:
    """Return each chosen token's top-k/top-p probability, using temperature 1 for greedy draft confidence."""

    temp = sampling.temperature if sampling is not None and sampling.temperature > 0 else 1.0
    top_p = sampling.top_p if sampling is not None else 1.0
    top_k = sampling.top_k if sampling is not None and sampling.top_k else values.shape[1]
    out = []
    for i, tok in enumerate(chosen):
        order = np.lexsort((tokens[i], -values[i]))[:top_k]
        v = values[i][order].astype(np.float64) / temp
        p = np.exp(v - v.max())
        p /= p.sum()
        if 0.0 < top_p < 1.0:
            keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
            p = p[:keep] / p[:keep].sum()
            order = order[:keep]
        ids = tokens[i][order]
        hit = np.nonzero(ids == tok)[0]
        out.append(float(p[hit[0]]) if len(hit) else 0.0)
    return out


PREFILL_ROWS = 2048      # rows of a prompt chunk


class Engine:
    """Weights, one sequence's state, buffers for decode windows (main model and MTP head) and for prompt chunks."""

    def __init__(self, w: Weights, *, capacity: int = 2560, max_rows: int = 8, prefill_rows: int = PREFILL_ROWS,
                 graphs: bool = False, graph_rows: tuple[int, ...] = (1, 2, 3, 4), taps: tuple[int, ...] = ()) -> None:
        """``taps``: layers whose residual stream a block drafter reads (its target_layer_ids), kept by every forward."""

        self.w = w
        w.meta["long_context"] = False            # dense at every length: no indexer, no long-context switch
        self.rows, self.prefill_rows = max_rows, prefill_rows
        self.buf = Buffers(w, max_rows, capacity)
        self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True)
        if taps:
            self.buf.set_taps(tuple(taps))         # before any graph capture
            self.pbuf.set_taps(tuple(taps))
        self.mbuf = Buffers(w, max_rows, capacity) if w.mtp is not None else None
        self.st = State(w, capacity, max_rows)
        self.last_hidden: torch.Tensor | None = None
        self.constraint = self.window = None            # a request's grammar, and the next sample's rows under it
        self.draft_n = w.head.n
        self.graphs = None
        self.replays = {"main": 0, "mtp": 0, "eager": 0}   # steps by path
        self.replayed = False                               # a replay with collectives not yet synchronized (settle)
        if graphs:
            from .graphs import Graphs

            self.graphs = Graphs(self, graph_rows, graph_rows)
            self.reset()

    def reset(self) -> None:
        self.st.reset()

    def settle(self) -> None:
        """Finish an outstanding graph replay before an uncaptured collective. The engine opens NCCL without its
        graph-mixing support (NCCL_GRAPH_MIXING_SUPPORT=0, engine.py), which does not cover an uncaptured collective
        launched while a graph launch on the same communicator is outstanding, whatever the stream order: a draft's
        sample follows an MTP replay. Request-level gathers (engine.py) run after a sample, so after a settle."""

        if self.replayed:
            torch.cuda.synchronize()
            self.replayed = False

    def synchronize(self) -> None:
        """torch.cuda.synchronize(), which settles any replay too."""

        torch.cuda.synchronize()
        self.replayed = False

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A step's forward (a CUDA graph when one was captured for its shape): logits [R, V/world]."""

        R = stage(self.w, self.st, self.buf, tokens)
        dense = self.st.pos + R <= self.w.cfg.dense_limit
        g = self.graphs.main.get((R, self.st.parity)) if self.graphs is not None and dense else None
        if g is not None:
            self.replays["main"] += 1
            g.replay()
            self.replayed = self.w.comm is not None
            return self.buf.logits[:R]
        self.replays["eager"] += 1
        self.settle()
        return compute(self.w, self.st, self.buf, R, nch=chunks_for(self.st, R), host_pos=self.st.pos)

    def mtp(self, next_tokens: Sequence[int], hidden: torch.Tensor) -> torch.Tensor:
        """The MTP head on rows (hidden, next token): logits of the last row [1, V/world]."""

        n = mtp_stage(self.w, self.st, self.mbuf, next_tokens, hidden)
        dense = self.st.mtp_len + n <= self.w.cfg.dense_limit
        g = self.graphs.mtp.get(n) if self.graphs is not None and dense else None
        if g is not None:
            self.replays["mtp"] += 1
            g.replay()
            self.replayed = self.w.comm is not None
            return self.mbuf.logits[:1, :self.draft_n]
        self.settle()
        return mtp_compute(self.w, self.st, self.mbuf, n, nch=-(-(self.st.mtp_len + n) // latent.CHUNK),
                           host_pos=self.st.mtp_len)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False, probs: list[float] | None = None) -> list[int]:
        if not draft and self.constraint is not None and self.window is not None:
            self.constraint.mask(logits, self.window, self.w.vocab_offset)   # this rank's vocabulary columns
            self.window = None
        self.settle()                                   # the candidates' all-gather is uncaptured
        return sample_rows(self.w, logits, positions, sampling, None, probs)

    def verify_window(self, tokens: list[int]) -> list[int]:
        """The window a reply's grammar keeps (a chain cut at its first rejected draft), masked at the next sample."""

        if self.constraint is None:
            return tokens
        self.window = self.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
        return self.window.tokens

    def follow(self, tokens: Sequence[int]) -> None:
        if self.constraint is not None:
            self.constraint.advance(tokens)

    def main_hidden(self, rows: slice) -> torch.Tensor:
        """Return the final-normed main-model rows that the MTP head reads after a forward with logits."""

        return self.buf.fnormed[rows]

    def draft_hidden(self, row: int) -> torch.Tensor:
        """The MTP head's own output row a chained draft reads (after an MTP step): its final_layernorm output."""

        return self.mbuf.fnormed[0:1]

    def tap_rows(self, n: int, b: Buffers | None = None) -> torch.Tensor:
        """The last forward's first n rows of the block drafter's taps, in layer order: [n, taps * hidden] (a prompt
        chunk's are in ``self.pbuf``)."""

        return torch.cat([t[:n] for t in (b or self.buf).taps], dim=1)


# -- MTP drafts ---------------------------------------------------------------------------------------------------
def absorb(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """Absorb hidden rows and their next tokens into the MTP cache in independent chunks; return the last logits."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    step = e.mbuf.rows
    logits = None
    for s0 in range(0, len(next_tokens), step):
        part = list(next_tokens[s0:s0 + step])
        logits = e.mtp(part, hidden[s0:s0 + step])
        st.set_mtp_len(st.mtp_len + len(part))
    return logits


def draft(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb kept positions and chain drafts until cumulative confidence fails, always keeping the first draft.

    Step 1's hidden rows are the main forward's b.fnormed rows (the caller's); steps 2-3 read the previous MTP
    call's b.fnormed row. Nothing is zeroed at position 0."""

    st = e.st
    logits = absorb(e, hidden, next_tokens)
    drafts: list[int] = []
    n = len(next_tokens)
    chain = 1.0
    for j in range(count):
        probs: list[float] = []
        d = e.sample(logits[:1], [position + j], sampling, draft=True, probs=probs if confidence > 0 else None)[0]
        if confidence > 0 and j > 0 and chain * probs[0] < confidence:
            break
        drafts.append(d)
        if confidence > 0:
            chain *= probs[0]
            if chain < confidence:            # a further draft could not pass either: skip its MTP step
                break
        if j + 1 < count:
            prev = e.draft_hidden(n - 1 if j == 0 else 0)
            logits = e.mtp([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
    return drafts


# -- prefix snapshots ---------------------------------------------------------------------------------------------
@dataclass
class Snapshot:
    """Copy committed KDA state and retain attention caches whose valid prefixes survive resume; pending MTP rows
    await next tokens from the new prompt. The MLA layers' rope-key rows (pc, and the MTP head's mtp_pc) are among
    the saved attention rows, beside the latents."""

    ids: list[int]
    rec: torch.Tensor
    conv: torch.Tensor
    pending: torch.Tensor | None
    mtp_len: int
    rows: list | None = None      # the attention rows of ids (latents and rope keys, main and MTP), saved when
    nbytes: int = 0               # another conversation takes the live caches


def take_snapshot(e: Engine, ids: Sequence[int], pending: torch.Tensor | None, *, mtp: bool) -> Snapshot:
    st = e.st
    rec = st.rec[st.cur[0]].clone() if st.cur else st.rec[0].clone()
    return Snapshot(list(ids), rec, st.conv.clone(), pending.clone() if pending is not None else None,
                    st.mtp_len - st.mtp_drafted if mtp and pending is not None else -1)


def _row_views(st, n: int, m: int) -> list[torch.Tensor]:
    """Views of the attention rows a snapshot of n tokens (m in the MTP head) depends on: the MLA layers' latents
    and rope keys, and the MTP head's own latent and rope-key rows."""

    views = [kc[:n] for kc in st.kc] + [pc[:n] for pc in st.pc]
    if m > 0 and hasattr(st, "mtp_kc"):
        views.append(st.mtp_kc[:m])
        views.append(st.mtp_pc[:m])
    return views


def save_rows(e: Engine, snap: Snapshot) -> None:
    """Copy a snapshot's attention rows out of the live caches before another conversation overwrites them."""

    views = _row_views(e.st, len(snap.ids), max(snap.mtp_len, 0))
    snap.rows = [v.clone() for v in views]
    snap.nbytes = sum(r.numel() * r.element_size() for r in snap.rows)


def row_bytes(e: Engine, snap: Snapshot) -> int:
    """What ``save_rows`` would copy for this snapshot."""
    return sum(v.numel() * v.element_size() for v in _row_views(e.st, len(snap.ids), max(snap.mtp_len, 0)))


def snapshot_bytes(snap: Snapshot) -> int:
    """A kept snapshot's device bytes: KDA states, conv windows, pending MTP rows, saved rows."""

    held = [snap.rec, snap.conv] + ([snap.pending] if snap.pending is not None else [])
    return sum(t.numel() * t.element_size() for t in held) + (snap.nbytes if snap.rows is not None else 0)


def load_rows(e: Engine, snap: Snapshot) -> None:
    """Put a saved snapshot's attention rows back into the live caches."""
    for dst, src in zip(_row_views(e.st, len(snap.ids), max(snap.mtp_len, 0)), snap.rows):
        dst.copy_(src)


def restore(e: Engine, snap: Snapshot) -> None:
    st = e.st
    if st.cur:
        st.rec[st.cur[0]].copy_(snap.rec)
    st.conv.copy_(snap.conv)
    st.set_pos(len(snap.ids))
    st.set_mtp_len(max(snap.mtp_len, 0))
    st.mtp_drafted = 0


# -- prefill ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True,
            resume: Snapshot | None = None, keep_at: int | None = None, keep=None) -> int:
    """Commit the prompt in chunks and sample its first token; a resumed prompt ends in a fresh prefill's state."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    e.settle()                                           # a prompt's chunks run uncaptured collectives
    w, st, b = e.w, e.st, e.pbuf
    use_mtp = mtp and w.mtp is not None
    begin = 0
    if resume is None:
        e.reset()
    else:
        begin = len(resume.ids)
        if begin >= len(prompt) or list(prompt[:begin]) != resume.ids:
            raise ValueError("a resumed prefill needs a snapshot of a strict prefix of the prompt")
        if use_mtp and resume.mtp_len < 0:
            raise ValueError("this snapshot's draft caches do not fit the request")
        restore(e, resume)
        if use_mtp:
            k = resume.pending.shape[0]
            _absorb_rows(e, resume.pending, list(prompt[begin - k + 1:begin + 1]))
    from .forward import Cut

    if keep_at is not None and (keep is None or not max(1, begin) <= keep_at <= len(prompt)):
        raise ValueError("a kept prefix needs a callback and a point in the prompt's prefill")
    kept = resume if keep_at == begin else None
    last = None
    prof.active = True
    for start in range(begin, len(prompt), e.prefill_rows):
        chunk = list(prompt[start:start + e.prefill_rows])
        R = len(chunk)
        point = keep_at - start if keep_at is not None else 0
        cut = Cut(point, torch.empty_like(st.rec[0]), st.conv.clone()) if 0 < point < R else None
        last = compute(w, st, b, stage(w, st, b, chunk), nch=chunks_for(st, R), host_pos=st.pos, cut=cut).clone()
        e.last_hidden = b.fnormed[R - 1:R].clone()
        if 0 < point <= R:
            rec = cut.rec if cut is not None else st.rec[st.cur[0] if st.cur else 0].clone()
            conv = cut.conv if cut is not None else st.conv.clone()
            kept = Snapshot(list(prompt[:keep_at]), rec, conv,
                            b.fnormed[point - 1:point].clone() if use_mtp else None,
                            keep_at - 1 if use_mtp else -1)
        if use_mtp:
            nxt = list(prompt[start + 1:start + R + 1])
            if nxt:
                with prof.timed("mtp absorb"):
                    _absorb_rows(e, b.fnormed[:len(nxt)], nxt)
        with prof.timed("commit"):
            commit(w, st, b, R, R)
    if kept is not None:
        keep(kept)
    prof.active = False
    prof.report(len(prompt) - begin)
    if e.constraint is not None:                         # the first token's row, under the reply's grammar
        e.window = e.constraint.window([0], [-1])
    first = e.sample(last, [len(prompt)], sampling)[0]
    e.follow([first])
    return first


def _absorb_rows(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> None:
    """A prompt's rows into the MTP cache through the prefill buffers (the prefill arithmetic, like the prompt)."""

    st = e.st
    mtp_forward(e.w, st, e.pbuf, next_tokens, hidden)
    st.set_mtp_len(st.mtp_len + len(next_tokens))


# -- decode loops -----------------------------------------------------------------------------------------------
@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    stages: dict[str, float] = field(default_factory=dict)
    depths: list[int] = field(default_factory=list)
    keeps: list[int] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


def _sync(w: Weights) -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *,
                  stop_eos: bool = False, on_tokens=None, think=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token. ``think`` (a
    call_gate.ThinkStop that has seen ``pending``) ends the loop where the server's thinking budget cuts the reply."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    stages = dict(forward=0.0, sample=0.0, commit=0.0)
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        logits = e.forward(e.verify_window([out[-1]]))
        e.synchronize()
        t1 = time.perf_counter()
        tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
        e.follow([tok])
        t2 = time.perf_counter()
        commit(w, st, b, 1, 1)
        t3 = time.perf_counter()
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        out.append(tok)
        if on_tokens is not None:
            on_tokens([tok])
        if think is not None and think.hit([tok]):
            break
    _sync(w)
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, stages=stages)


class DepthPolicy:
    """Choose a fixed draft count or adapt it to running acceptance."""

    def __init__(self, most: int = 3, fixed: bool = False, low: float = 0.8, high: float = 0.9,
                 confidence: float = 0.0) -> None:
        self.most, self.fixed, self.low, self.high = most, fixed, low, high
        self.confidence = confidence
        self.rate = 0.8

    def next(self, drafted: int, accepted: int) -> int:
        if self.fixed:
            return self.most
        if drafted:
            self.rate = 0.875 * self.rate + 0.125 * (accepted / drafted)
        return max(1, min(self.most, 1 if self.rate < self.low else 2 if self.rate < self.high else 3))


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, policy: DepthPolicy | None = None,
               stop_eos: bool = False, on_tokens=None, think=None) -> DecodeResult:
    """Verify pending and MTP draft rows through the first mismatch, starting with every prompt position except the last in the MTP cache."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy()
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    _sync(w)
    start = time.perf_counter()
    t0 = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    drafts = draft(e, e.last_hidden, [pending], st.pos + 1, depth, sampling, policy.confidence) if depth > 0 else []
    stages["draft"] += time.perf_counter() - t0
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        tokens = e.verify_window([out[-1]] + drafts)
        drafts = tokens[1:]
        R = len(tokens)
        logits = e.forward(tokens)
        e.synchronize()
        t1 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t2 = time.perf_counter()
        commit(w, st, b, R, keep)
        t3 = time.perf_counter()
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        e.follow(sampled[:keep])
        out.extend(sampled[:keep])
        fed = sampled[:keep][:max(0, count - (len(out) - keep))]
        if on_tokens is not None:
            on_tokens(fed)
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos) or (think is not None and think.hit(fed)):
            break
        t4 = time.perf_counter()
        depth = min(policy.next(len(drafts), keep - 1), count - len(out))
        drafts = (draft(e, e.main_hidden(slice(0, keep)), sampled[:keep], st.pos + 1, depth, sampling,
                        policy.confidence) if depth > 0 else [])
        stages["draft"] += time.perf_counter() - t4
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)
