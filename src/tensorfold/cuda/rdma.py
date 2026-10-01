"""One-hop all-gather of small fp32 rank partials over RoCE RDMA writes (rdma_gather.cu), for TP decode on GB10.

NCCL moves a decode window's [R, hidden] fp32 partial through its proxy in ~35 us (R=1) inside a CUDA graph; a raw RoCE
write of the same 24 KB takes 6.4 us between two GB10 nodes (ib_write_lat). Here a staging kernel puts the partial in
pinned host memory and rings a doorbell, a CPU thread RDMA-writes it to every peer followed by a sequence flag on the same
queue pair, and a collecting kernel waits for the flags and copies the slots out in rank order: the same bytes as
``NCCL.all_gather``, the kernel pair captured in graphs like any other.
"""

from __future__ import annotations

import atexit
import os
import pickle
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rdma_gather_v2", sources=[str(here / "rdma_gather.cu")],
                extra_cuda_cflags=["-O3"], extra_ldflags=["-libverbs"], verbose=False)


def device_names() -> list[str]:
    """TF_RDMA_DEVICES (comma-separated), else every active RoCE port of this node, NCCL_IB_HCA's first; the ranks
    open the same number."""

    named = [d.strip() for d in (os.environ.get("TF_RDMA_DEVICES") or "").split(",") if d.strip()]
    if named:
        return named
    first = (os.environ.get("NCCL_IB_HCA") or "").strip().lstrip("=^").split(",")[0].split(":")[0].strip()
    root = Path("/sys/class/infiniband")
    active = []
    for d in sorted(p.name for p in root.iterdir()) if root.exists() else []:
        try:
            state = (root / d / "ports" / "1" / "state").read_text()
            layer = (root / d / "ports" / "1" / "link_layer").read_text().strip()
        except OSError:
            continue
        if "ACTIVE" in state and layer == "Ethernet":
            active.append(d)
    if first in active:
        active.remove(first)
        active.insert(0, first)
    return active or ([first] if first else [])


class RdmaGather:
    """``all_gather(send, recv)`` of fp32 tensors up to ``max_bytes`` a rank; every rank calls in the same order."""

    def __init__(self, store, rank: int, world: int, *, max_bytes: int, slots: int = 4, prefix: str = "tf_rdma",
                 devices: list[str] | None = None, gid_index: int | None = None) -> None:
        ext = _ext()
        gid = int(os.environ.get("NCCL_IB_GID_INDEX", "5")) if gid_index is None else int(gid_index)
        cpu = int(os.environ.get("TF_RDMA_CPU", "-1"))
        self.ext, self.rank, self.world, self.max_bytes = ext, rank, world, int(max_bytes)
        self.devices = device_names() if devices is None else list(devices)
        error = ""
        try:
            self.h = ext.create(rank, world, self.max_bytes, slots, self.devices, gid, cpu)
            info = pickle.dumps(ext.local_info(self.h))
        except Exception as exc:           # noqa: BLE001  (published, so every rank refuses together)
            error, info = f"{type(exc).__name__}: {exc}", b""
        store.set(f"{prefix}/info/{rank}", b"E" + error.encode() if error else b"I" + info)
        peers = [store.get(f"{prefix}/info/{r}") for r in range(world)]
        bad = [f"rank {r}: {p[1:].decode(errors='replace')}" for r, p in enumerate(peers) if p[:1] != b"I"]
        if bad:
            raise RuntimeError("RDMA gather unavailable (" + "; ".join(bad) + ")")
        try:
            ext.connect(self.h, [pickle.loads(peers[r][1:])[rank] for r in range(world)])
        except Exception as exc:           # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        # every QP is ready to receive before anyone writes, or every rank gives up
        store.set(f"{prefix}/up/{rank}", b"E" + error.encode() if error else b"U")
        ups = [store.get(f"{prefix}/up/{r}") for r in range(world)]
        bad = [f"rank {r}: {u[1:].decode(errors='replace')}" for r, u in enumerate(ups) if u[:1] != b"U"]
        if bad:
            raise RuntimeError("RDMA gather unavailable (" + "; ".join(bad) + ")")
        ext.start(self.h)
        atexit.register(ext.stop, self.h)

    def fits(self, send: torch.Tensor) -> bool:
        return send.dtype == torch.float32 and send.numel() % 4 == 0 and send.numel() * 4 <= self.max_bytes

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        self.ext.gather(self.h, send, recv)

    def failure(self) -> str:
        return self.ext.failure(self.h)


class Hybrid:
    """A communicator: the RDMA gather for the fp32 tensors it holds (decode partials, sampling), NCCL for the rest."""

    def __init__(self, nccl, rdma: RdmaGather) -> None:
        self.nccl, self.rdma = nccl, rdma
        self.rank, self.world = nccl.rank, nccl.world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if self.rdma.fits(send) and send.is_contiguous() and recv.is_contiguous():
            self.rdma.all_gather(send, recv)
        else:
            self.nccl.all_gather(send, recv)

    def __getattr__(self, name: str):
        return getattr(self.nccl, name)
