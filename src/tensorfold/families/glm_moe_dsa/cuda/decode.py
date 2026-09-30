"""GLM-5.3's decode loops: the Flash engine's exact verify/commit path over this family's buffers and state."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorfold.families.glm5_next.cuda.decode import (  # noqa: F401
    DecodeResult, DepthPolicy, absorb, draft, mtp_decode, prefill, serial_decode, sample_rows,
    snapshot_bytes, row_bytes, save_rows, load_rows, restore, take_snapshot, _row_views,
)

from .forward import Buffers, State, check_room, stage, commit  # noqa: F401
from .select import sparse_bucket  # noqa: F401


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
        self.draft_n = w.head.n
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
            g, kind = self.graphs.sparse.get((R, 0, sparse_bucket(self.st.pos, R))), "sparse"
        if g is not None:
            self.replays[kind] += 1
            g.replay()
            return self.buf.logits[:R]
        self.replays["eager"] += 1
        return fwd.compute(self.w, self.st, self.buf, R, nch=fwd.chunks_for(self.st, R), host_pos=self.st.pos)

    def mtp(self, next_tokens: Sequence[int], hidden: torch.Tensor) -> torch.Tensor:
        from .mtp import mtp_forward

        return mtp_forward(self.w, self.st, self.mbuf, next_tokens, hidden)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling, *, draft: bool = False,
               probs: list | None = None) -> list[int]:
        from tensorfold.families.glm5_next.cuda.decode import sample_rows

        return sample_rows(self.w, logits, positions, sampling, draft=draft, probs=probs)

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
        torch.cuda.synchronize()
