"""GLM-5.3's decode loops: the Flash engine's exact verify/commit path over this family's buffers and state."""

from __future__ import annotations

import os
from typing import Sequence

import torch

from tensorfold.families.glm5_next.cuda.decode import (  # noqa: F401
    DecodeResult, DepthPolicy, absorb, sample_rows,
    snapshot_bytes, row_bytes, save_rows, load_rows, restore, take_snapshot, _row_views,
)

from .forward import Buffers, State, check_room, stage, commit  # noqa: F401
from .select import sparse_bucket  # noqa: F401

# draft depth by acceptance (TF_GLM_DEPTH_COST: what a draft position adds to a round, relative to a one-row round;
# empty or 0: the lane's fixed depth). Drafts only propose: the depth changes speed, never a reply.
DEPTH_COST = float(os.environ.get("TF_GLM_DEPTH_COST") or 0)
DEPTH_PROBE = 8                           # every this many rounds one position deeper, so deeper estimates stay current


class AcceptPolicy:
    """DepthPolicy's interface: each draft position's acceptance (given the ones before it) as a running mean of the
    rounds that reached it, and the depth (at least 1, at most ``most``) whose expected tokens a round per cost
    (1 + cost x depth) is highest, one deeper every DEPTH_PROBE rounds. Only the rounds' drafted / accepted counts
    decide it (the same on every rank), never a clock."""

    fixed = False
    confidence = 0.0

    def __init__(self, most: int, cost: float = DEPTH_COST, prior: float = 0.8, rate: float = 0.125) -> None:
        self.most, self.cost, self.rate = int(most), float(cost), float(rate)
        self.a = [float(prior)] * self.most
        self.rounds = 0

    def update(self, drafted: int, accepted: int) -> None:
        """A round that drafted ``drafted`` and kept the first ``accepted`` of them."""

        for j in range(min(drafted, accepted + 1)):
            self.a[j] += self.rate * ((1.0 if j < accepted else 0.0) - self.a[j])

    def best(self, cost: float | None = None) -> int:
        k = self.cost if cost is None else float(cost)
        depth, value, chain, gain = 1, 0.0, 1.0, 1.0
        for d in range(1, self.most + 1):
            chain *= self.a[d - 1]
            gain += chain
            v = gain / (1.0 + k * d)
            if v > value:
                depth, value = d, v
        self.rounds += 1
        if self.rounds % DEPTH_PROBE == 0:
            depth = min(self.most, depth + 1)
        return max(1, min(self.most, depth)) if self.most else 0

    def next(self, drafted: int, accepted: int) -> int:
        if drafted:
            self.update(drafted, accepted)
        return self.best()


def shared_cost(cost: float, streams: int) -> float:
    """A draft row's cost relative to a stream's share of a concurrent round (``cost``: relative to a one-row round of
    one stream). A round of n streams costs B + r x rows (r: mostly the routed experts' weight bytes its rows add); with
    cost = r / (B + r), a stream's share of the n-stream round of one row each is (B + n r) / n, so its draft row costs
    n r / (B + n r) = n cost / (1 + (n - 1) cost) of it: drafts pay off less as streams join."""

    n = max(1, int(streams))
    return n * cost / (1.0 + (n - 1) * cost)


def resume_cut(kept: Sequence[int] | None, prompt: Sequence[int], step: int, short: int = 0) -> int:
    """Where a prompt resumes from a kept prompt's rows (TF_GLM_KEEP_SLOTS; multi.resume_at's rule); 0: none."""

    from .multi import KEEP, resume_at

    return resume_at(kept, prompt, step, short) if KEEP else 0


@torch.no_grad()
def prefill(e: "Engine", prompt: Sequence[int], sampling, **_) -> int:
    """Commit the prompt in prompt chunks (this family's forward and state; no KDA, no taps) and sample its first token.

    MTP: the prompt's rows are absorbed into the MTP head's cache chunk by chunk, as Flash does, when the
    checkpoint has the head (``--no-drafts`` serves still absorb nothing: the MTP cache stays unused).
    A prompt that extends the last one resumes after the chunks they share (``resume_cut``; e.cached).
    """

    from . import forward as fwd, invariant
    from .multi import tiny_rows

    if not prompt:
        raise ValueError("prefill requires at least one token")
    w, st, b = e.w, e.st, e.pbuf
    short = tiny_rows(w) if invariant.INVARIANT else 0   # a whole prompt this short: the decode kernels (as batched)
    cut = resume_cut(getattr(e, "kept", None), prompt, e.prefill_rows, short)
    e.kept = None                                    # (set again once this prompt is whole)
    e.reset()
    if cut:
        st.set_pos(cut)
        st.set_mtp_len(cut)
    e.cached = cut
    last = None
    use_mtp = w.mtp is not None and not getattr(e, "serial_only", False)
    with invariant.suspended(cut == 0 and len(prompt) <= short):
        for start in range(cut, len(prompt), e.prefill_rows):
            chunk = list(prompt[start:start + e.prefill_rows])
            R = fwd.stage(w, st, b, chunk)
            last = fwd.compute(w, st, b, R, nch=fwd.chunks_for(st, R), host_pos=st.pos).clone()
            e.last_hidden = b.fnormed[R - 1:R].clone()
            if use_mtp:
                nxt = list(prompt[start + 1:start + R + 1])
                if nxt:
                    from .mtp import mtp_forward

                    mtp_forward(w, st, b, nxt, b.fnormed[:len(nxt)])
                    st.set_mtp_len(st.mtp_len + len(nxt))
            fwd.commit(w, st, b, R, R)
    e.kept = list(prompt)
    if e.constraint is not None:
        e.window = e.constraint.window([0], [-1])
    first = e.sample(last, [len(prompt)], sampling)[0]
    e.follow([first])
    return first


def draft(e: "Engine", hidden: torch.Tensor, next_tokens: Sequence[int], position: int, count: int, sampling,
          confidence: float = 0.0) -> list[int]:
    """Flash's draft chain (absorb the kept rows, then one-row MTP steps while the chain's confidence holds), with
    the MTP layer's selection shared by the chain's later steps when MTP_REUSE is on."""

    from .mtp import MTP_REUSE

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
            logits = e.mtp([d], prev, reuse=MTP_REUSE != "0")
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
    return drafts


@torch.no_grad()
def serial_decode(e: "Engine", pending: int, count: int, sampling, *, stop_eos: bool = False,
                  on_tokens=None) -> DecodeResult:
    """One token a step through this family's forward (a CUDA graph when captured) and the shared sampler."""

    import time

    from . import forward as fwd

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    stages = dict(forward=0.0, sample=0.0, commit=0.0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        logits = e.forward(e.verify_window([out[-1]]))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
        e.follow([tok])
        t2 = time.perf_counter()
        fwd.commit(w, st, b, 1, 1)
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += time.perf_counter() - t2
        out.append(tok)
        if on_tokens is not None:
            on_tokens([tok])
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, stages=stages)


@torch.no_grad()
def mtp_decode(e: "Engine", pending: int, count: int, sampling, *, policy: DepthPolicy | None = None,
               stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Flash's verify loop (pending + MTP drafts, kept through the first mismatch) over this family's commit.

    Flash's own ``mtp_decode`` commits through its KDA state (``st.cur`` ping-pong); GLM-5.3 has none, so the loop is
    restated here verbatim with this family's ``commit`` (advance the position; latent rows past ``keep`` are simply
    overwritten by the next window).
    """

    import time

    from . import forward as fwd

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy()
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    torch.cuda.synchronize()
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
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t2 = time.perf_counter()
        fwd.commit(w, st, b, R, keep)
        t3 = time.perf_counter()
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        e.follow(sampled[:keep])
        out.extend(sampled[:keep])
        if on_tokens is not None:
            on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        t4 = time.perf_counter()
        depth = min(policy.next(len(drafts), keep - 1), count - len(out))
        drafts = (draft(e, e.main_hidden(slice(0, keep)), sampled[:keep], st.pos + 1, depth, sampling,
                        policy.confidence) if depth > 0 else [])
        stages["draft"] += time.perf_counter() - t4
    torch.cuda.synchronize()
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)


class Engine:
    """Weights, one sequence's state, buffers for decode windows (main model and MTP head) and prompt chunks.

    The Flash Engine's verify/sample loops run unchanged over this family's forward (plain residuals,
    one latent cache per layer); only the buffer construction and graph capture differ, so this class
    wires the family modules into the same interface ``decode`` and the server expect.
    """

    def __init__(self, w, *, capacity: int = 2560, max_rows: int = 8, prefill_rows: int | None = None,
                 graphs: bool = False, graph_rows: tuple[int, ...] = (1, 2, 3, 4), long_context: bool = False,
                 taps: tuple[int, ...] = ()) -> None:
        from tensorfold.cuda.geometry import PREFILL_ROWS
        from tensorfold.families.glm5_next.cuda import latent

        from . import forward as fwd

        self.w = w
        w.meta["long_context"] = long_context
        self.rows, self.prefill_rows = max_rows, prefill_rows or PREFILL_ROWS
        self.buf = fwd.Buffers(w, max_rows, capacity)
        self.pbuf = fwd.Buffers(w, self.prefill_rows, capacity, prefill=True)
        self.mbuf = fwd.Buffers(w, max_rows, capacity) if w.mtp is not None else None
        self.st = fwd.State(w, capacity, max_rows)
        self.last_hidden: torch.Tensor | None = None
        self.constraint = self.window = None            # a request's grammar, and the next sample's rows under it
        self.draft_n = (w.head if w.draft_head is None else w.draft_head).n
        self.graphs = None
        self.replays = {"main": 0, "sparse": 0, "mtp": 0, "sparse_mtp": 0, "eager": 0}
        if graphs:
            self.graphs = Graphs(self, graph_rows, graph_rows)

    def reset(self) -> None:
        self.st.reset()

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A step's forward (a CUDA graph when one was captured for its shape): logits [R, V/world]."""

        from . import forward as fwd

        R = fwd.stage(self.w, self.st, self.buf, tokens)
        dense = self.st.pos + R <= self.w.cfg.dense_limit
        g, kind = None, "main"
        if self.graphs is not None and dense:
            g = self.graphs.main.get((R, 0))
        elif self.graphs is not None and self.st.pos >= self.w.cfg.dense_limit and self.st.index is not None:
            # every row past the dense limit: the sparse graph for this bucket (same kernels as eager)
            g, kind = self.graphs.sparse.get((R, 0, sparse_bucket(self.st.pos, R, self.w.meta.get("dcp", 1)))), \
                "sparse"
        if g is not None:
            self.replays[kind] += 1
            g.replay()
            return self.buf.logits[:R]
        self.replays["eager"] += 1
        return fwd.compute(self.w, self.st, self.buf, R, nch=fwd.chunks_for(self.st, R), host_pos=self.st.pos)

    def mtp(self, next_tokens: Sequence[int], hidden: torch.Tensor, reuse: bool = False) -> torch.Tensor:
        from .mtp import mtp_forward

        return mtp_forward(self.w, self.st, self.mbuf, next_tokens, hidden, reuse=reuse)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling, *, draft: bool = False,
               probs: list | None = None) -> list[int]:
        from tensorfold.families.glm5_next.cuda.decode import sample_rows

        # a draft's logits come from the draft head (TF_GLM_DRAFT_VOCAB) when there is one: its columns start at
        # this rank's first draft id
        offset = self.w.meta["draft_lo"] if draft and self.w.draft_head is not None else None
        return sample_rows(self.w, logits, positions, sampling, offset, probs=probs)

    def verify_window(self, tokens: list[int]) -> list[int]:
        """The window a reply's grammar keeps (a chain cut at its first rejected draft), masked at the next sample."""

        if self.constraint is None:
            return tokens
        self.window = self.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
        return self.window.tokens

    def follow(self, tokens: Sequence[int]) -> None:
        if self.constraint is not None:
            self.constraint.advance(tokens)

    def tap_rows(self, n: int, b=None) -> torch.Tensor:
        raise ValueError("GLM-5.3 has no hyper-connection taps: a DFlash2 drafter cannot attach")

    def main_hidden(self, rows: slice) -> torch.Tensor:
        """The final-normed main-model rows that the MTP head reads after a forward with logits."""

        return self.buf.fnormed[rows]

    def draft_hidden(self, row: int) -> torch.Tensor:
        """The MTP head's own output row a chained draft reads (after an MTP step): its shared_head.norm output."""

        return self.mbuf.fnormed[0:1]


class Graphs:
    """CUDA graphs of this family's forward and MTP compute (dense windows; sparse buckets for long contexts)."""

    def __init__(self, e: Engine, main_rows=(1, 2, 3, 4), mtp_rows=(1, 2, 3, 4)) -> None:
        from . import forward as fwd
        from .mtp import mtp_compute

        self.pool = torch.cuda.graph_pool_handle()
        self.main: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mtp: dict[int, torch.cuda.CUDAGraph] = {}
        self.sparse: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.sparse_mtp: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        w, st = e.w, e.st
        prof_active = False
        with torch.no_grad():
            for R in main_rows:
                for _ in range(2):
                    fwd.compute(w, st, e.buf, R)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    fwd.compute(w, st, e.buf, R)
                self.main[(R, 0)] = g
            if w.mtp is not None:
                e.mbuf.zero_first = False
                for n in mtp_rows:
                    for _ in range(2):
                        mtp_compute(w, st, e.mbuf, n)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        mtp_compute(w, st, e.mbuf, n)
                    self.mtp[n] = g
            if st.index is not None:
                # past the dense limit every row attends to its selected tokens: a graph a window and scored-token
                # bucket, the eager path's kernels and shapes (Engine.forward looks up sparse_bucket(pos, R))
                from .select import sparse_buckets

                for bucket in sparse_buckets(st.capacity, w.cfg.dense_limit, w.meta.get("dcp", 1)):
                    for R in main_rows:
                        for _ in range(2):
                            fwd.compute(w, st, e.buf, R, sparse_np=bucket)
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, pool=self.pool):
                            fwd.compute(w, st, e.buf, R, sparse_np=bucket)
                        self.sparse[(R, 0, bucket)] = g
        torch.cuda.synchronize()
