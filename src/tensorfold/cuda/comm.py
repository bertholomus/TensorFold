"""NCCL all-gather on the current stream so CUDA graphs capture it; a rank-order sum after it keeps ranks bit-equal.

``gather="p2p"`` moves the same bytes as grouped ncclSend/ncclRecv with every rank (itself included): one network hop
instead of the ring's world - 1 steps. Across four GB10 nodes a [1, 6144] fp32 partial took 35 us inside a CUDA graph
instead of 51 us (tools/nccl_bench.py), and the gathered tensor is the same.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os

import torch

_DTYPES = {torch.float32: 7, torch.bfloat16: 9, torch.int32: 2, torch.int64: 4}


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library() -> ctypes.CDLL:
    candidates = [os.environ.get("TF_NCCL_LIB", "")]
    found = ctypes.util.find_library("nccl")
    if found:
        candidates.append(found)
    candidates += glob.glob("/usr/lib/*/libnccl.so.2") + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/nccl/lib/libnccl.so.2")
    candidates += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libnccl*.so*"))
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError("libnccl not found (set TF_NCCL_LIB)")


class NCCL:
    def __init__(self, rank: int, world: int, master: str, port: int, *, gather: str = "ring") -> None:
        from datetime import timedelta

        from torch.distributed import TCPStore

        if gather not in ("ring", "p2p"):
            raise ValueError(f"gather must be ring (ncclAllGather) or p2p (grouped send/recv), not {gather!r}")
        self.rank, self.world, self.gather = rank, world, gather
        self.lib = _library()
        lib = self.lib
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p]
        for name in ("ncclSend", "ncclRecv"):
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p]
        self.store = TCPStore(master, port, world, rank == 0, timeout=timedelta(seconds=600))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set("tf_nccl_uid", bytes(uid.internal))
        else:
            raw = self.store.get("tf_nccl_uid")
            ctypes.memmove(ctypes.addressof(uid), raw, 128)
        self.comm = ctypes.c_void_p()
        torch.cuda.current_device()
        self._check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), world, uid, rank))

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NCCL error {code}: {self.lib.ncclGetErrorString(code).decode()}")

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order (contiguous tensors, same dtype)."""

        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        stream = torch.cuda.current_stream().cuda_stream
        if self.gather == "p2p":
            n, kind, lib = send.numel(), _DTYPES[send.dtype], self.lib
            step = n * send.element_size()
            self._check(lib.ncclGroupStart())
            try:
                for peer in range(self.world):
                    self._check(lib.ncclSend(send.data_ptr(), n, kind, peer, self.comm, stream))
                    self._check(lib.ncclRecv(recv.data_ptr() + peer * step, n, kind, peer, self.comm, stream))
            finally:
                self._check(lib.ncclGroupEnd())
            return
        self._check(self.lib.ncclAllGather(send.data_ptr(), recv.data_ptr(), send.numel(), _DTYPES[send.dtype],
                                           self.comm, stream))

    def grouped(self, sends, recvs) -> None:
        """One NCCL group of point-to-point transfers on the current stream: each (tensor, peer) of ``sends`` to that
        peer and each of ``recvs`` from it (contiguous tensors; a peer may be this rank; empty ones are skipped)."""

        stream = torch.cuda.current_stream().cuda_stream
        lib = self.lib
        self._check(lib.ncclGroupStart())
        try:
            for t, peer in sends:
                if t.numel():
                    self._check(lib.ncclSend(t.data_ptr(), t.numel(), _DTYPES[t.dtype], peer, self.comm, stream))
            for t, peer in recvs:
                if t.numel():
                    self._check(lib.ncclRecv(t.data_ptr(), t.numel(), _DTYPES[t.dtype], peer, self.comm, stream))
        finally:
            self._check(lib.ncclGroupEnd())

    def ready(self, label: str, *, every: float = 60.0, timeout: float = 3600.0) -> None:
        """Every rank finishes ``label`` before any goes on; a rank missing after ``timeout`` s is named."""

        import time
        from datetime import timedelta

        self.store.set(f"tf_ready/{label}/{self.rank}", "1")
        others = [r for r in range(self.world) if r != self.rank]
        started = time.monotonic()
        while True:
            try:
                self.store.wait([f"tf_ready/{label}/{r}" for r in others], timedelta(seconds=every))
                return
            except Exception as exc:                  # noqa: BLE001  (the store's timeout; anything else goes up)
                if "timeout" not in str(exc).lower():
                    raise
            waited = time.monotonic() - started
            missing = ", ".join(str(r) for r in others)
            if waited >= timeout:
                raise RuntimeError(f"rank {self.rank} finished {label} but rank {missing} has not after "
                                   f"{waited / 60:.0f} min: check that rank's log (a CUDA extension build waiting on "
                                   "a lock names the lock there)")
            print(f"[tensorfold] rank {self.rank} finished {label}; waiting for rank {missing} ({waited:.0f} s)",
                  flush=True)

    def abort(self) -> None:
        """A peer is gone: ncclCommAbort, so NCCL work in flight on this rank returns (callable from any thread; the
        communicator is unusable after it)."""

        comm, self.comm = self.comm, None
        if comm is not None and comm.value:
            self.lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
            self.lib.ncclCommAbort(comm)

    def async_error(self) -> int:
        """NCCL's asynchronous error code for this communicator (0: none; a network failure shows here)."""

        if self.comm is None:
            return -1
        code = ctypes.c_int(0)
        self.lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.ncclCommGetAsyncError(self.comm, ctypes.byref(code))
        return int(code.value)

    def barrier(self) -> None:
        x = torch.zeros((1,), dtype=torch.float32, device="cuda")
        y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
        self.all_gather(x, y)
        torch.cuda.synchronize()
