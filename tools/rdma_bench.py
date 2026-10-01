"""The RDMA gather (tensorfold.cuda.rdma) against NCCL on the TP ranks: same bytes, and latency eager and in graphs.

usage: python3 rdma_bench.py RANK WORLD MASTER PORT   (every rank, same args; rank 0 prints)
"""

from __future__ import annotations

import sys
import time

import torch

RANK, WORLD, MASTER, PORT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
N, D = 156, 6144


def say(*a) -> None:
    if RANK == 0:
        print(*a, flush=True)


def main() -> None:
    import os

    from tensorfold.cuda.comm import NCCL
    from tensorfold.cuda.rdma import RdmaGather

    os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")
    torch.cuda.set_device(0)
    nccl = NCCL(RANK, WORLD, MASTER, PORT, gather="p2p")
    nccl.barrier()
    from tensorfold.cuda.rdma import device_names

    t = time.time()
    rdma = RdmaGather(nccl.store, RANK, WORLD, max_bytes=8 * D * 4)
    one = RdmaGather(nccl.store, RANK, WORLD, max_bytes=8 * D * 4, devices=device_names()[:1], prefix="tf_rdma1")
    say(f"rdma gathers up in {time.time() - t:.1f}s: {rdma.devices} and {one.devices}")

    # the same bytes as NCCL, eager and in a graph, every row count
    for R in (1, 2, 4, 8):
        send = torch.randn((R * D,), device="cuda") * (RANK + 1)
        a = torch.empty((WORLD * R * D,), device="cuda")
        b = torch.full((WORLD * R * D,), float("nan"), device="cuda")
        b1 = torch.full((WORLD * R * D,), float("nan"), device="cuda")
        nccl.all_gather(send, a)
        rdma.all_gather(send, b)
        one.all_gather(send, b1)
        torch.cuda.synchronize()
        same = torch.equal(a.view(torch.int32), b.view(torch.int32)) and torch.equal(a.view(torch.int32), b1.view(torch.int32))
        g = torch.cuda.CUDAGraph()
        c = torch.full_like(b, float("nan"))
        with torch.cuda.graph(g):
            rdma.all_gather(send, c)
        for k in range(3):
            send.copy_(torch.randn((R * D,), device="cuda") * (RANK + k + 2))
            nccl.all_gather(send, a)
            g.replay()
            torch.cuda.synchronize()
            same = same and torch.equal(a.view(torch.int32), c.view(torch.int32))
        say(f"R={R}: rdma bytes == nccl bytes (eager and graph replays): {same}")
        del g

    x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    z = torch.empty((1024, 1024), device="cuda", dtype=torch.bfloat16)

    def work():
        for _ in range(4):
            torch.mm(x, y, out=z)

    work()
    torch.cuda.synchronize()

    def timed(fn, reps=5) -> float:
        fn()
        torch.cuda.synchronize()
        nccl.barrier()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps * 1e3

    for R in (1, 4):
        send = torch.zeros((R * D,), device="cuda")
        recv = torch.empty((WORLD * R * D,), device="cuda")
        gw = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gw):
            for _ in range(N):
                work()
        w0 = timed(gw.replay)
        g0 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g0):
            for _ in range(N):
                nccl.all_gather(send, recv)
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            for _ in range(N):
                work()
                nccl.all_gather(send, recv)
        say(f"R={R}: nccl p2p: graph {timed(g0.replay) * 1e3 / N:.1f} us, with work {(timed(g1.replay) - w0) * 1e3 / N:.1f} us")
        del g0, g1
        for label, comm in (("rdma 2 ports", rdma), ("rdma 1 port", one)):
            comm.ext.configure(comm.h, False)
            g0 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g0):
                for _ in range(N):
                    comm.all_gather(send, recv)
            g1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1):
                for _ in range(N):
                    work()
                    comm.all_gather(send, recv)
            bare = timed(g0.replay) * 1e3 / N
            with_work = (timed(g1.replay) - w0) * 1e3 / N
            comm.ext.configure(comm.h, True)
            gp = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gp):
                for _ in range(N):
                    comm.all_gather(send, recv)
            comm.ext.probes(comm.h)
            timed(gp.replay)
            st, net, cp, n = comm.ext.probes(comm.h)
            comm.ext.configure(comm.h, False)
            say(f"   {label}: graph {bare:.1f} us, with work {with_work:.1f} us | probes (bare): stage {st / 1e3:.1f},"
                f" doorbell->flags {net / 1e3:.1f}, copy-out {cp / 1e3:.1f} us ({n:.0f})")
            del g0, g1, gp
    nccl.barrier()
    say(f"failure: {rdma.failure()!r} {one.failure()!r}")
    say("done")


if __name__ == "__main__":
    main()
