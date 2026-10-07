"""[boot] load-phase timing: one print per phase, no behavior change (perf_counter only, never synchronizes)."""

from __future__ import annotations

import time

_STARTED = time.perf_counter()         # this module's first import (the engine build), not process start
_PREV = _STARTED


def boot_mark(phase: str, rank: int | None = None) -> None:
    """Print "[boot] rank R: <phase> <s since first import> s (+<s since the last mark> s)"; prints only."""

    global _PREV
    now = time.perf_counter()
    who = f"rank {rank}: " if rank is not None else ""
    print(f"[boot] {who}{phase} {now - _STARTED:.1f} s (+{now - _PREV:.1f} s)", flush=True)
    _PREV = now
