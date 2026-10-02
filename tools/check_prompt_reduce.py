"""A prompt chunk's rank partials over the real fabric, every way forward.gather can reduce them: every fp32 partial to
every rank ("gather", then residual_add), each rank's share of the rows to it, summed, bf16 sums gathered back in row
order ("rows"), and rows as two micro-batches whose reductions run on the comm stream ("overlap",
forward.micro_batch). Every rank checks the results are bit-equal and the ranks agree; ms a collective (reduce +
residual) for each way.

usage (one container a rank, TF_TP_WORLD and NCCL_* set as tp4_run.sh sets them):
  python3 tools/check_prompt_reduce.py RANK MASTER PORT [ROWS=2048,4096]
"""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import torch

RANK, MASTER, PORT = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ROWS = [int(r) for r in (sys.argv[4] if len(sys.argv) > 4 else "2048,4096").split(",")]
D = 6144


def main() -> None:
    from tensorfold.cuda.comm import NCCL
    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd

    torch.cuda.set_device(0)
    world = int(os.environ.get("TF_TP_WORLD", "4"))
    comm = NCCL(RANK, world, MASTER, PORT, gather=os.environ.get("TF_NCCL_GATHER") or "p2p")
    w = SimpleNamespace(world=world, comm=comm, rank=RANK)
    rows = max(ROWS)
    b = SimpleNamespace(prefill=True, world=world, rows=rows, lat_s=SimpleNamespace(),
                        part=torch.empty((rows, D), dtype=torch.float32, device="cuda"),
                        gath=torch.empty((world * rows * D,), dtype=torch.float32, device="cuda"),
                        comm_stream=torch.cuda.Stream())
    g = torch.Generator(device="cuda").manual_seed(1234 + RANK)          # each rank its own partial
    gx = torch.Generator(device="cuda").manual_seed(99)                  # one residual stream on every rank
    for R in ROWS:
        b.part[:R].copy_(torch.randn((R, D), device="cuda", generator=g) * 4)
        x = torch.randn((R, D), device="cuda", generator=gx).to(torch.bfloat16)
        outs, ms = {}, {}
        for mode in ("gather", "rows", "overlap", "gather", "rows", "overlap"):
            fwd.PROMPT_REDUCE = "gather" if mode == "gather" else "rows"
            out = torch.empty_like(x)
            h = -(-R // 32) * 16

            def step():
                if mode != "overlap":
                    fwd.residual(x, out, fwd.gather(w, b, R))
                    return
                halves = [(fwd.micro_batch(b, 0, h, 0, R), 0, h), (fwd.micro_batch(b, h, R, 1, R), h, R)]
                pend = [fwd.gather(w, v, hi - lo) for v, lo, hi in halves]       # both reductions queued
                for (v, lo, hi), g in zip(halves, pend):
                    fwd.residual(x[lo:hi], out[lo:hi], g)

            step()
            comm.barrier()
            reps = 10
            t = time.perf_counter()
            for _ in range(reps):
                step()
            torch.cuda.synchronize()
            ms[mode] = (time.perf_counter() - t) / reps * 1e3
            outs[mode] = out.clone()
        same = all(torch.equal(outs["gather"].view(torch.int16), outs[m].view(torch.int16)) for m in ("rows", "overlap"))
        sums = torch.tensor([float(outs["rows"].float().sum()), float(same)], device="cuda")
        every = torch.empty((world * 2,), device="cuda")
        comm.all_gather(sums, every)
        torch.cuda.synchronize()
        every = every.view(world, 2).tolist()
        if RANK == 0:
            agree = all(r[0] == every[0][0] for r in every)
            print(f"R={R}: gather {ms['gather']:.2f} ms, rows {ms['rows']:.2f} ms, rows as two micro-batches "
                  f"{ms['overlap']:.2f} ms a collective + residual; "
                  f"bit-equal on every rank {all(r[1] == 1.0 for r in every)}, ranks agree {agree}", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
