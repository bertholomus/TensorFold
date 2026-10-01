"""Latency of the decode forward's all-gathers on the TP ranks under one NCCL setting (the env this process gets).

usage: python3 nccl_bench.py RANK WORLD MASTER PORT LABEL   (every rank, same args; rank 0 prints one line a mode)
Modes, each 156 all-gathers of [R, 6144] fp32 (one decode forward's worth) with and without ~COMPUTE_US of work
between them: eager calls, one CUDA graph, and per-segment graphs with eager all-gathers between (the work alone is
subtracted from the interleaved modes, leaving what the all-gathers add).
"""

from __future__ import annotations

import sys
import time

import torch

RANK, WORLD, MASTER, PORT, LABEL = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
MODE = sys.argv[6] if len(sys.argv) > 6 else "allgather"     # or "p2p": grouped send/recv to every peer, one hop
N, D = 156, 6144


def p2p_gather(comm):
    """recv [world * n] <- every rank's send [n] through grouped ncclSend/ncclRecv (each peer directly, self included)."""

    import ctypes

    lib = comm.lib
    lib.ncclSend.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                             ctypes.c_void_p]
    lib.ncclRecv.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                             ctypes.c_void_p]

    def all_gather(send, recv):
        n = send.numel()
        stream = torch.cuda.current_stream().cuda_stream
        es = send.element_size()
        comm._check(lib.ncclGroupStart())
        for peer in range(comm.world):
            comm._check(lib.ncclSend(send.data_ptr(), n, 7, peer, comm.comm, stream))
            comm._check(lib.ncclRecv(recv.data_ptr() + peer * n * es, n, 7, peer, comm.comm, stream))
        comm._check(lib.ncclGroupEnd())

    return all_gather


def main() -> None:
    from tensorfold.cuda.comm import NCCL

    torch.cuda.set_device(0)
    comm = NCCL(RANK, WORLD, MASTER, PORT)
    comm.barrier()
    gather = comm.all_gather if MODE == "allgather" else p2p_gather(comm)
    if MODE != "allgather":                            # the p2p path moves the same bytes: check it once
        x = torch.full((D,), float(RANK + 1), device="cuda")
        y = torch.zeros((WORLD * D,), device="cuda")
        gather(x, y)
        torch.cuda.synchronize()
        assert all(float(y[r * D]) == r + 1 for r in range(WORLD)), y.view(WORLD, D)[:, 0]
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((1024, 1024), device="cuda", dtype=torch.bfloat16)

    def work() -> None:                                # ~100-200 us of GPU work between collectives
        for _ in range(4):
            torch.mm(a, b, out=c)

    work()                                             # cuBLAS's handle exists before any capture
    torch.cuda.synchronize()

    def timed(fn, reps=5, warm=2) -> float:
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        comm.barrier()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps * 1e3

    out = []
    for R in (1, 4):
        send = torch.zeros((R * D,), dtype=torch.float32, device="cuda")
        recv = torch.empty((WORLD * R * D,), dtype=torch.float32, device="cuda")

        def eager():
            for _ in range(N):
                gather(send, recv)

        def eager_work():
            for _ in range(N):
                work()
                gather(send, recv)

        def only_work():
            for _ in range(N):
                work()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            eager()
        gw = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gw):
            eager_work()
        segs = []
        pool = torch.cuda.graph_pool_handle()
        for _ in range(N):
            s = torch.cuda.CUDAGraph()
            with torch.cuda.graph(s, pool=pool):
                work()
            segs.append(s)

        def segmented():
            for s in segs:
                s.replay()
                gather(send, recv)

        w0 = timed(only_work)
        e = timed(eager)
        gr = timed(g.replay)
        ew = timed(eager_work) - w0
        gwt = timed(gw.replay) - w0
        sg = timed(segmented) - w0
        us = lambda ms: 1e3 * ms / N  # noqa: E731
        out.append(f"R={R}: bare eager {us(e):5.1f} graph {us(gr):5.1f} | with work: eager {us(ew):5.1f} "
                   f"graph {us(gwt):5.1f} segmented {us(sg):5.1f} us/all-gather (work {us(w0):.0f} us)")
        del g, gw, segs
    comm.barrier()
    if RANK == 0:
        print(f"[{LABEL} {MODE}] " + " || ".join(out), flush=True)


if __name__ == "__main__":
    main()
