"""The decode gathers' RDMA-write all-gather (tensorfold.cuda.rdma, rdma_gather.cu) on one box's RDMA devices: two ranks
in one process, their queue pairs looped through the box's own NIC (TF_RDMA_TEST_DEVICES names the devices, default the
box's first two active RoCE ones; skipped without them). Run with /dev/infiniband in the container and memlock
unlimited.

- Every rank gets every rank's partial in rank order, the same bits, for windows of 16 bytes to 128 KB, back to back.
- The idle back-off (v6): quiet past TF_RDMA_IDLE_S, both proxies nap (``idle_stats`` counts the spells), the next
  gather is whole and wakes them (the first one after the quiet spell printed); napping proxies cost the process a
  fraction of a core where spinning ones (TF_RDMA_IDLE_S=0) cost one each; back to back, the napping pair is as fast
  as the spinning one (printed: the loopback's, not a link's).
"""

import os
import resource
import statistics
import threading
import time
from pathlib import Path

import pytest
import torch

from tensorfold.cuda import rdma


def _devices() -> list:
    named = [d for d in (os.environ.get("TF_RDMA_TEST_DEVICES") or "").split(",") if d.strip()]
    if named:
        return named
    root, out = Path("/sys/class/infiniband"), []
    for d in sorted(p.name for p in root.iterdir()) if root.exists() else []:
        try:
            up = "ACTIVE" in (root / d / "ports" / "1" / "state").read_text()
            eth = (root / d / "ports" / "1" / "link_layer").read_text().strip() == "Ethernet"
        except OSError:
            continue
        if up and eth:
            out.append(d)
    return out[:2]


DEVICES = _devices()
GID = int(os.environ.get("NCCL_IB_GID_INDEX", "3"))
MAX_BYTES = 128 << 10
pytestmark = pytest.mark.skipif(not DEVICES or not os.path.exists("/dev/infiniband") or not torch.cuda.is_available(),
                                reason="no active RoCE devices (or no /dev/infiniband, or no GPU)")


class Store:
    """The ranks' rendezvous store: ``set``, and a ``get`` that waits for its key."""

    def __init__(self) -> None:
        self.d, self.cv = {}, threading.Condition()

    def set(self, key, value) -> None:
        with self.cv:
            self.d[key] = value
            self.cv.notify_all()

    def get(self, key):
        with self.cv:
            assert self.cv.wait_for(lambda: key in self.d, timeout=30), f"no {key}"
            return self.d[key]


def _pair(idle_s: float, prefix: str) -> list:
    """Two ranks' gathers over DEVICES, their proxies napping after ``idle_s`` quiet (0: never)."""

    keep, rdma.IDLE_S = rdma.IDLE_S, idle_s
    store, out, errs = Store(), {}, []

    def make(r: int) -> None:
        try:
            out[r] = rdma.RdmaGather(store, r, 2, max_bytes=MAX_BYTES, prefix=prefix, devices=DEVICES, gid_index=GID)
        except Exception as exc:                          # noqa: BLE001
            errs.append(exc)

    try:
        ts = [threading.Thread(target=make, args=(r,)) for r in (0, 1)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
    finally:
        rdma.IDLE_S = keep
    assert not errs, errs
    return [out[0], out[1]]


def _run(pair: list, n: int, reps: int, check: bool = True) -> float:
    """``reps`` gathers of ``n`` floats on both ranks at once (a thread and a stream each); the seconds they took. Rank
    r sends rank r's values for the rep; with ``check`` every rep's output is the ranks' sends in rank order."""

    base = torch.arange(n, device="cuda", dtype=torch.float32)
    errs, times = [], []
    go = threading.Barrier(2)

    def rank(r: int) -> None:
        s = torch.cuda.Stream()
        recv = torch.empty(2 * n, device="cuda", dtype=torch.float32)
        try:
            with torch.cuda.stream(s):
                sends = [base + (1e4 * r + 7.0 * k) for k in range(reps)]
                want = [torch.cat([base + 7.0 * k, base + (1e4 + 7.0 * k)]) for k in range(reps)] if check else None
                s.synchronize()
                go.wait()
                t0 = time.perf_counter()
                for k in range(reps):
                    pair[r].all_gather(sends[k], recv)
                    if check:
                        s.synchronize()
                        if not torch.equal(recv, want[k]):
                            errs.append(f"rank {r} rep {k}: not every rank's partial in rank order")
                            return
                s.synchronize()
                times.append(time.perf_counter() - t0)
        except Exception as exc:                          # noqa: BLE001
            errs.append(f"rank {r}: {exc}")

    ts = [threading.Thread(target=rank, args=(r,)) for r in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert not errs, errs
    assert len(times) == 2, "a rank's gathers did not finish"
    for g in pair:
        assert not g.failure(), g.failure()
    return max(times)


def _cpu_s() -> float:
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


def test_the_gather_gives_every_rank_every_partial_and_naps_when_quiet():
    naps = _pair(0.2, "tf_rdma_naps")
    for n in (4, 1024, 30720, MAX_BYTES // 4):            # (30720 floats: a 6-row window of 5120)
        _run(naps, n, 50)
    # quiet: both proxies go to napping, and the next gather wakes them, whole
    time.sleep(0.6)
    slept = [g.idle_stats() for g in naps]
    assert all(s[0] >= 1 for s in slept), slept
    t_first = _run(naps, 30720, 1)
    woke = [g.idle_stats() for g in naps]
    assert all(w[1] >= s[1] + 1 for w, s in zip(woke, slept)), (slept, woke)
    # napping proxies cost a fraction of a core; spinning ones (TF_RDMA_IDLE_S=0) a core each
    time.sleep(0.5)
    c0, t0 = _cpu_s(), time.monotonic()
    time.sleep(1.0)
    nap_cores = (_cpu_s() - c0) / (time.monotonic() - t0)
    spin = _pair(0.0, "tf_rdma_spin")
    try:
        _run(spin, 30720, 20)
        assert spin[0].idle_stats() == (0, 0)
        c0, t0 = _cpu_s(), time.monotonic()
        time.sleep(1.0)
        spin_cores = (_cpu_s() - c0) / (time.monotonic() - t0)
        # back to back the napping pair never naps: as fast as the spinning one
        per_nap, per_spin = [], []
        for _ in range(3):
            per_nap.append(_run(naps, 30720, 300, check=False) / 300 * 1e6)
            per_spin.append(_run(spin, 30720, 300, check=False) / 300 * 1e6)
    finally:
        for g in spin:
            g.ext.stop(g.h)
    nap_us, spin_us = statistics.median(per_nap), statistics.median(per_spin)
    print(f"\nloopback over {DEVICES}: the first 120 KB gather after a quiet spell {t_first * 1e6:.0f} us; napping "
          f"proxies {nap_cores:.2f} cores, spinning {spin_cores:.2f}; back to back {nap_us:.1f} us a gather napping, "
          f"{spin_us:.1f} spinning (the loopback's, not a link's); idle_stats {[g.idle_stats() for g in naps]}")
    assert nap_cores < 0.5 and spin_cores > 1.5, (nap_cores, spin_cores)
    assert nap_us <= spin_us * 1.25 + 5.0, (per_nap, per_spin)
    assert t_first < 0.02
