"""L2 prefetch on a side stream (``l2_prefetch.cu``): the weights the next kernels of a decode step read, pulled into L2
while the main stream runs kernels that leave DRAM idle (gathers, norms, attention), so the linears and experts that
follow read them from L2. It writes nothing and changes no bit of any output, only where the kernels after it find
their bytes. Forks are graph-safe: the side stream waits for the main stream at the fork, and ``join`` makes the main
stream wait for every prefetch issued so far (a captured body must join before its capture ends).

GB10 (LPDDR5X): a dependent load's latency grows several-fold while DRAM serves queued traffic, so a prefetch slows the
latency-bound kernels it runs beside; callers cap each fork's bytes and start it where DRAM would idle.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

CHUNK = 8 << 10                   # bytes a bulk prefetch
BLOCKS = 4                        # unpaced: blocks of 128 threads issuing every chunk at once
# ns a prefetch waits before issuing, so a gather's staging kernel launched right after the fork runs undisturbed
DELAY_NS = int(os.environ.get("TF_EXL3_L2_DELAY_NS") or 3000)
# > 0 (default 230; 200 before the decode scratch left L2 unwritten: TF_EXL3_L2_DISCARD): one block issues a wave of
# 128 chunks (1 MiB) every 128 * CHUNK / RATE_GBPS ns, a little under what DRAM serves (~245 GB/s), so DRAM's queue
# holds about a wave, the bytes arrive about in the order their readers take them, and the kernels running beside keep
# most of their load latency (GB10, one-GPU decode proxy: 1-row forward -1.0 ms more than issuing all at once); 0:
# every chunk issued at once from BLOCKS blocks
RATE_GBPS = float(os.environ.get("TF_EXL3_L2_RATE_GBPS") or 230)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_exl3_l2_prefetch_v6", sources=[str(here / "l2_prefetch.cpp"),
                                                                 str(here / "l2_prefetch.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def ranges(tensors, budget: int | None = None) -> tuple[list[int], list[int]]:
    """(pointers, sizes) of the tensors' bytes in order (None entries skipped), cut at ``budget`` bytes in all."""

    ptrs, sizes, left = [], [], (1 << 62) if budget is None else int(budget)
    for t in tensors:
        if t is None or left < 16:
            continue
        if not t.is_contiguous() or not t.is_cuda:
            raise ValueError("prefetch ranges are contiguous CUDA tensors")
        n = min(t.numel() * t.element_size(), left) // 16 * 16
        p = t.data_ptr()
        if n <= 0 or p % 16:
            continue
        ptrs.append(p)
        sizes.append(n)
        left -= n
    cap = int(_ext().l2_prefetch_max())
    return ptrs[:cap], sizes[:cap]


class SideStream:
    """Prefetch launches on one side stream: ``fork`` queues a prefetch that starts once everything issued on the
    current stream so far (and ``after``, an event of another stream, when given) is done; ``join`` makes the current
    stream wait for all of them."""

    def __init__(self) -> None:
        self.stream: torch.cuda.Stream | None = None
        self.live = False

    def fork(self, rng: tuple[list[int], list[int]], after: torch.cuda.Event | None = None) -> None:
        ptrs, sizes = rng
        if not ptrs:
            return
        cur = torch.cuda.current_stream()
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        self.stream.wait_stream(cur)
        if after is not None:
            self.stream.wait_event(after)
        with torch.cuda.stream(self.stream):
            wave_ns = int(128 * CHUNK / RATE_GBPS) if RATE_GBPS > 0 else 0
            _ext().l2_prefetch(ptrs, sizes, CHUNK, BLOCKS, DELAY_NS, wave_ns)
        self.live = True

    def join(self) -> None:
        if self.live:
            torch.cuda.current_stream().wait_stream(self.stream)
            self.live = False
