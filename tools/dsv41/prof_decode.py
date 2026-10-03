"""Decode-step profile of the engine on TP ranks: wall time a token and the top CUDA / CPU ops (rank 0 prints)."""

import argparse
import json
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--master", default="127.0.0.1", help="rank 0's address on the link between the machines")
    p.add_argument("--port", type=int, default=29662)
    p.add_argument("--model", required=True)
    p.add_argument("--engram")
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--profile-steps", type=int, default=4)
    p.add_argument("--out")
    a = p.parse_args()
    torch.cuda.set_device(0)
    from tensorfold.families.deepseek_v41.cuda.model import Comm, Engram, Model
    from tensorfold.families.deepseek_v41.cuda.weights import load
    from tensorfold.families.deepseek_v41.ops import compressed_token_map

    nccl = None
    if a.world > 1:
        from tensorfold.cuda.comm import NCCL

        nccl = NCCL(a.rank, a.world, a.master, a.port)
    w = load(a.model, a.rank, a.world, log=lambda m: None)
    eng = None
    if a.engram:
        tm, _ = compressed_token_map(f"{a.model}/tokenizer.json")
        eng = Engram(a.engram, w.cfg, tm, a.rank, a.world)
    model = Model(w, Comm(nccl, a.world, rdma_bytes=8 << 20), eng)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1000, 100000, (a.prompt_len,), generator=g).cuda()
    sc = model.new_cache(a.prompt_len + a.steps + a.profile_steps + 8)
    t0 = time.time()
    for s in range(0, a.prompt_len, 512):
        last = model.forward(sc, ids[s:s + 512], s)
    torch.cuda.synchronize()
    prefill_s = time.time() - t0
    pos = a.prompt_len
    tok = last[0].argmax().view(1)
    times = []
    for _ in range(a.steps):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        last = model.forward(sc, tok, pos)
        tok = last[0].argmax().view(1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)
        pos += 1
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(a.profile_steps):
            last = model.forward(sc, tok, pos)
            tok = last[0].argmax().view(1)
            pos += 1
        torch.cuda.synchronize()
    if a.rank == 0:
        times.sort()
        print(json.dumps({"prefill_s": prefill_s, "prompt_len": a.prompt_len,
                          "decode_ms_median": 1000 * times[len(times) // 2], "decode_ms_min": 1000 * times[0]}))
        ka = prof.key_averages()
        print(ka.table(sort_by="cuda_time_total", row_limit=40))
        print(ka.table(sort_by="self_cpu_time_total", row_limit=25))


if __name__ == "__main__":
    main()
