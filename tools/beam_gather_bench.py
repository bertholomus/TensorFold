"""Time Beam's rank all-gathers under one NCCL setting (the env this process gets): fp32 [rows, 5120] partials, the
sizes of a decode step (4, 16 and 32 rows) and of a prompt chunk (2,048 and 4,096 rows), eager calls on the current
stream as forward.Work.reduce makes them; checks the gathered rows once. Every rank runs it with the same args; rank 0
prints one line a size.
usage: python3 tools/beam_gather_bench.py RANK WORLD MASTER PORT LABEL"""

import sys

import torch

from tensorfold.cuda.comm import NCCL

D = 5120
SIZES = (4, 16, 32, 2048, 4096)


def main() -> None:
    rank, world, master, port, label = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    torch.cuda.set_device(0)
    comm = NCCL(rank, world, master, port)
    comm.barrier()
    out = []
    for rows in SIZES:
        send = torch.full((rows, D), float(rank + 1), device="cuda")
        recv = torch.empty((world, rows, D), device="cuda")
        comm.all_gather(send, recv)
        torch.cuda.synchronize()
        if not all(bool((recv[r] == r + 1).all()) for r in range(world)):
            raise SystemExit(f"{label}: the gathered rows differ at {rows} rows")
        n = 200 if rows <= 32 else 30
        for _ in range(5):
            comm.all_gather(send, recv)
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n):
            comm.all_gather(send, recv)
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / n
        out.append(f"{rows} rows {ms * 1e3:.0f} us ({send.numel() * 4 / ms / 1e6:.1f} GB/s)")
    comm.barrier()
    if rank == 0:
        print(f"GATHER {label}: " + ", ".join(out), flush=True)


if __name__ == "__main__":
    main()
