"""Whether Beam's rank gathers can hide behind compute on GB10: fp32 [rows, 5120] all-gathers (a prompt chunk's o_proj
or MLP partials, or half of them) on a high-priority side stream while the compute stream runs bf16 matmuls of a
q projection's shape (rows x 5120 -> 3072), against each alone. Times a step (CUDA events, 20 steps after 3): the
gather alone, the matmuls alone, both at once; hidden = the share of the shorter one that the overlap took off. Every
rank runs it with the same args; rank 0 prints one line a size.
usage: python3 tools/beam_overlap_bench.py RANK WORLD MASTER PORT"""

import sys

import torch

from tensorfold.cuda.comm import NCCL

D = 5120
N = 20


def timed(step) -> float:
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(N):
        step()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / N


def main() -> None:
    rank, world, master, port = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    torch.cuda.set_device(0)
    comm = NCCL(rank, world, master, port)
    comm.barrier()
    main_s, side = torch.cuda.current_stream(), torch.cuda.Stream(priority=-1)
    lines = []
    for rows, mms in ((1024, 1), (1024, 2), (2048, 2), (2048, 4)):
        send = torch.full((rows, D), float(rank + 1), device="cuda")
        recv = torch.empty((world, rows, D), device="cuda")
        a = torch.randn(rows, D, device="cuda").to(torch.bfloat16)
        b = torch.randn(D, 3072, device="cuda").to(torch.bfloat16)
        c = torch.empty(rows, 3072, device="cuda", dtype=torch.bfloat16)

        def gather():
            comm.all_gather(send, recv)

        def compute():
            for _ in range(mms):
                torch.mm(a, b, out=c)

        def both():
            side.wait_stream(main_s)
            with torch.cuda.stream(side):
                comm.all_gather(send, recv)
            compute()
            main_s.wait_stream(side)

        g, m, gm = timed(gather), timed(compute), timed(both)
        both()
        torch.cuda.synchronize()
        if not all(bool((recv[r] == r + 1).all()) for r in range(world)):
            raise SystemExit(f"the gathered rows differ at {rows} rows")
        hidden = (g + m - gm) / min(g, m) * 100
        lines.append(f"{rows} rows, {mms} matmuls: gather {g * 1e3:.0f} us, compute {m * 1e3:.0f} us, both "
                     f"{gm * 1e3:.0f} us (serial {(g + m) * 1e3:.0f}; {hidden:.0f} % of the shorter hidden)")
    comm.barrier()
    if rank == 0:
        for line in lines:
            print("OVERLAP " + line, flush=True)


if __name__ == "__main__":
    main()
