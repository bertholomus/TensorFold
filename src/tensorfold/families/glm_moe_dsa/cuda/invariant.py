"""GLM-5.3's chunk-invariant prompt path (TF_GLM_PROMPT_INVARIANT=1): a prompt row's bits depend on nothing but the row
itself and the cache it reads, not on the chunk it fills in, so a prompt resumed at any token (TF_GLM_KEEP_SLOTS) fills
exactly as its fresh prefill does.

Two things make today's prompt rows depend on their chunk: kernels picked by the chunk's row count (the EXL3 prompt GEMM
past 128 rows, the routed experts' prompt kernels from 64, the fused sparse attention and the Q scratch from 64, span
scores and the radix top-k from 16, absorb / expand's row-block kernels past 64), and cuBLAS products whose kernel
cuBLAS picks by M (absorb / expand, the indexer's per-token head weights, the dequantized GEMM). Here prompt chunks take
the prompt kernels at any row count, and the cuBLAS products run in blocks of exactly INVARIANT_ROWS rows (``blocked``):
the last block ends at the chunk's last row, overlapping the one before (the same rows, the same values), and a chunk
shorter than a block reaches into its buffers' spare rows. Other bits than the default path (a lane config's own
serial references); prompts shorter than a block pay a block's GEMM. Whole prompts of at most multi.tiny_rows tokens
are the exception: they fill with the decode kernels (``suspended``), as without it.
"""

from __future__ import annotations

import contextlib
import os

import torch

INVARIANT = os.environ.get("TF_GLM_PROMPT_INVARIANT", "0") == "1"
INVARIANT_ROWS = int(os.environ.get("TF_GLM_INVARIANT_ROWS") or 1024)


@contextlib.contextmanager
def suspended(on: bool = True):
    """With ``on``, INVARIANT off inside: a whole prompt of at most multi.tiny_rows tokens fills with the decode
    windows' row-invariant kernels, as the batched fill of short prompts does, so its rows are the same either way
    (and no resume starts from or into such a prompt: multi.resume_at)."""

    global INVARIANT
    was = INVARIANT
    if on:
        INVARIANT = False
    try:
        yield
    finally:
        INVARIANT = was


def ranges(R: int, M0: int) -> list[tuple[int, int]]:
    """Row blocks of exactly M0 rows covering [0, R): consecutive ones, the last ending at R when R >= M0; else one
    block [0, M0) reaching past R."""

    if R >= M0:
        out = [(a, a + M0) for a in range(0, R - M0 + 1, M0)]
        if out[-1][1] < R:
            out.append((R - M0, R))
        return out
    return [(0, M0)]


def rows(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor | None:
    """Rows [lo, hi) of t (dim 0), past t's own rows when its storage holds them (a buffer's spare rows), else None."""

    if hi <= t.shape[0]:
        return t[lo:hi]
    last = t.storage_offset() + (hi - 1) * t.stride(0) + sum((n - 1) * s for n, s in zip(t.shape[1:], t.stride()[1:]))
    if (last + 1) * t.element_size() > t.untyped_storage().nbytes():
        return None
    return t.as_strided((hi - lo, *t.shape[1:]), t.stride(), t.storage_offset() + lo * t.stride(0))


def blocked(fn, R: int, ins: list[torch.Tensor], outs: list[torch.Tensor], M0: int = 0) -> None:
    """fn(*in_views, *out_views) over INVARIANT_ROWS-row blocks of rows [0, R) of every tensor (rows on dim 0). A block
    past a tensor's storage goes through zero-padded copies (inputs) and copies back of its real rows (outputs)."""

    M0 = M0 or INVARIANT_ROWS
    for lo, hi in ranges(R, M0):
        iv = [rows(t, lo, hi) for t in ins]
        ov = [rows(t, lo, hi) for t in outs]
        if all(v is not None for v in iv + ov):
            fn(*iv, *ov)
            continue
        real = min(hi, R) - lo
        pads = []
        for t, v in zip(ins, iv):
            if v is None:
                v = torch.zeros((hi - lo, *t.shape[1:]), dtype=t.dtype, device=t.device)
                v[:real].copy_(t[lo:lo + real])
            pads.append(v)
        tmp = [torch.empty((hi - lo, *t.shape[1:]), dtype=t.dtype, device=t.device) if v is None else v
               for t, v in zip(outs, ov)]
        fn(*pads, *tmp)
        for t, v, o in zip(outs, ov, tmp):
            if v is None:
                t[lo:lo + real].copy_(o[:real])
