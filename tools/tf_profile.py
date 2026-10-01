"""Where GLM-5.3's decode time goes on TF_TP_WORLD ranks: every rank runs the same command, rank 0 prints the report.

usage (in the tf container, TF_TP_WORLD and NCCL_* set as tp4_start.sh sets them; see tp4_run.sh):
  python3 tools/tf_profile.py MODEL RANK MASTER PORT [CONTEXT] [TOKENS] [DEPTH_TOKENS]
Sections: end-to-end serial vs MTP decode of tf_greedy's prompts (stages, exactness), graph replay per window,
MTP step eager vs graph, sampling, NCCL all-gather latency, forward without collectives, kernel time by name, and
decode past the dense limit (DEPTH_TOKENS of filler, eager sparse path).
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

MODEL, RANK, MASTER, PORT = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
CONTEXT = int(sys.argv[5]) if len(sys.argv) > 5 else 32768
TOKENS = int(sys.argv[6]) if len(sys.argv) > 6 else 300
DEPTH = int(sys.argv[7]) if len(sys.argv) > 7 else 6000
SECTIONS = set((os.environ.get("TF_PROFILE_SECTIONS") or "e2e,graphs,mtp,sample,comm,local,kernels,depth").split(","))
PROMPTS = ["Write a detailed explanation of how a hash table works, including collisions, load factor and resizing.",
           "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
           "List the planets of the solar system with one interesting fact about each.",
           "Translate to French: The weather is nice today, so we will go for a walk in the park after lunch."]


def say(*args) -> None:
    if RANK == 0:
        print(*args, flush=True)


def timeit(fn, reps: int = 20, warm: int = 3) -> float:
    """Mean ms a call (every rank makes the same calls: collectives inside stay matched)."""

    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


class Local:
    """A communicator that gathers this rank's partial into every slot (no network): the forward's compute alone."""

    def __init__(self, world: int) -> None:
        self.world = world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        recv.view(self.world, -1).copy_(send.view(1, -1).expand(self.world, -1))


def kernel_table(prof, top: int = 28) -> None:
    times: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            name = ev.name if len(ev.name) < 90 else ev.name[:87] + "..."
            times[name] += ev.time_range.elapsed_us()
            counts[name] += 1
    total = sum(times.values())
    say(f"    kernel time {total / 1e3:.2f} ms over {sum(counts.values())} launches")
    for name, t in sorted(times.items(), key=lambda kv: -kv[1])[:top]:
        say(f"    {t / 1e3:7.3f} ms {100 * t / max(total, 1e-9):5.1f}% x{counts[name]:4d}  {name}")


def main() -> None:
    from tensorfold.families.glm_moe_dsa.cuda import decode as dec
    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine
    from tensorfold.families.glm_moe_dsa.cuda.mtp import mtp_compute
    from tensorfold.families.glm5_next.cuda.attention import CHUNK

    t = time.time()
    eng = GlmEngine(MODEL, rank=RANK, master=MASTER, port=PORT, policy="3", context=CONTEXT, context_explicit=True)
    e, w = eng.e, eng.w
    st = e.st
    say(f"== loaded in {time.time() - t:.0f}s: limit {eng.limit}, slots {eng.capacity_plan['cache_slots']}, "
        f"allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB, "
        f"NCCL_NET_PLUGIN={os.environ.get('NCCL_NET_PLUGIN')} NCCL_IB_HCA={os.environ.get('NCCL_IB_HCA')}")
    app = GlmApp(eng, MODEL, "GLM")
    prompts = [app._prepare({"messages": [{"role": "user", "content": p}], "max_tokens": TOKENS, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False}}, True).prompt for p in PROMPTS]
    say("   prompt tokens:", [len(p) for p in prompts])

    if "e2e" in SECTIONS:
        say("== end to end (no HTTP): serial vs MTP depth 3, greedy, stop at EOS")
        for i, p in enumerate(prompts):
            first = dec.prefill(e, p, None)
            s = dec.serial_decode(e, first, TOKENS, None, stop_eos=True)
            first2 = dec.prefill(e, p, None)
            m = dec.mtp_decode(e, first2, TOKENS, None, policy=dec.DepthPolicy(3, fixed=True), stop_eos=True)
            same = s.tokens == m.tokens
            sst = ", ".join(f"{k} {1e3 * v / max(1, len(s.tokens) - 1):.2f}" for k, v in s.stages.items())
            mst = ", ".join(f"{k} {1e3 * v / max(1, m.rounds):.2f}" for k, v in m.stages.items())
            say(f"   prompt {i}: serial {len(s.tokens)} tok {s.tokens_per_second:.2f} tok/s [ms/tok: {sst}]")
            say(f"             mtp3 {len(m.tokens)} tok {m.tokens_per_second:.2f} tok/s rounds {m.rounds} drafted "
                f"{m.drafted} accepted {m.accepted} [ms/round: {mst}] same={same}")
            keeps = {k: m.keeps.count(k) for k in sorted(set(m.keeps))}
            say(f"             tokens a round (keep: rounds) {keeps}")
        say(f"   replays {e.replays}")

    if "graphs" in SECTIONS:
        say(f"== main graph replay per window (pos {st.pos})")
        for R in sorted(e.graphs.main):
            g = e.graphs.main[R]
            say(f"   R={R[0]}: {timeit(g.replay):.2f} ms")

    if "mtp" in SECTIONS:
        say(f"== MTP step: eager vs graph (mtp_len {st.mtp_len})")
        mb = e.mbuf
        for n in sorted(e.graphs.mtp):
            eager = timeit(lambda: mtp_compute(w, st, mb, n, nch=-(-(st.mtp_len + n) // CHUNK), host_pos=st.mtp_len))
            graph = timeit(e.graphs.mtp[n].replay)
            say(f"   n={n}: eager {eager:.2f} ms, graph {graph:.2f} ms")
        hidden = e.buf.fnormed[0:1]
        tok = prompts[0][-1]
        dt = timeit(lambda: dec.draft(e, hidden, [tok], st.pos + 1, 3, None, 0.0), reps=10)
        say(f"   draft(depth 3) as mtp_decode calls it: {dt:.2f} ms")

    if "sample" in SECTIONS:
        say("== sampling (top-1 per rank, all-gather, host pick)")
        for R in (1, 4):
            logits = e.buf.logits[:R]
            say(f"   R={R}: {timeit(lambda: e.sample(logits, [st.pos + 1 + r for r in range(R)], None)):.3f} ms")

    if "comm" in SECTIONS:
        say("== NCCL all-gather of fp32 partials [R, hidden]")
        D = w.cfg.hidden
        for R in (1, 4):
            send = torch.zeros((R * D,), dtype=torch.float32, device="cuda")
            recv = torch.empty((w.world * R * D,), dtype=torch.float32, device="cuda")
            eager = timeit(lambda: w.comm.all_gather(send, recv), reps=200, warm=20)
            g = torch.cuda.CUDAGraph()
            n = 2 * len(w.layers)
            with torch.cuda.graph(g):
                for _ in range(n):
                    w.comm.all_gather(send, recv)
            graph = timeit(g.replay, reps=10, warm=2)
            say(f"   R={R}: eager {1e3 * eager:.1f} us a call; {n} in a graph {graph:.2f} ms ({1e3 * graph / n:.1f} us a call)")
            del g

    if "local" in SECTIONS:
        say("== forward without collectives (each rank alone; Local all-gather copies): compute-only graph time")
        comm = w.comm
        w.comm = Local(w.world)
        pool = torch.cuda.graph_pool_handle()
        local = {}
        try:
            mine = []
            for R in (1, 2, 4):
                for _ in range(2):
                    fwd.compute(w, st, e.buf, R)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    fwd.compute(w, st, e.buf, R)
                local[R] = g
                mine.append(int(1e3 * timeit(g.replay)))
            w.comm = comm
            ranks = eng._gather_ints(mine)
            w.comm = Local(w.world)
            for i, R in enumerate((1, 2, 4)):
                say(f"   R={R}: " + ", ".join(f"rank {r} {ranks[r][i] / 1e3:.2f}" for r in range(len(ranks))) + " ms")
            if "kernels" in SECTIONS:
                from torch.profiler import ProfilerActivity, profile

                for R in (1, 4):
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        local[R].replay()
                        torch.cuda.synchronize()
                    say(f"== kernels of one collective-free R={R} graph replay")
                    kernel_table(prof)
                mb = e.mbuf
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    mtp_compute(w, st, mb, 1, nch=-(-(st.mtp_len + 1) // CHUNK), host_pos=st.mtp_len)
                    torch.cuda.synchronize()
                say("== kernels of one eager MTP step (n=1, collective-free)")
                kernel_table(prof, top=16)
        finally:
            w.comm = comm
            local.clear()
        torch.cuda.synchronize()
        for R in (1, 2, 4):
            say(f"   with collectives R={R}: {timeit(e.graphs.main[(R, 0)].replay):.2f} ms")

    if "depth" in SECTIONS and DEPTH > 0:
        filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(DEPTH // 13))
        body = {"messages": [{"role": "user", "content": filler + "\n\nSummarize the list above in two sentences."}],
                "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        p = app._prepare(body, True).prompt
        say(f"== past the dense limit: {len(p)}-token prompt")
        torch.cuda.synchronize()
        t = time.perf_counter()
        first = dec.prefill(e, p, None)
        torch.cuda.synchronize()
        say(f"   prefill {time.perf_counter() - t:.1f}s ({len(p) / (time.perf_counter() - t):.0f} tok/s)")
        r0 = dict(e.replays)
        s = dec.serial_decode(e, first, 48, None, stop_eos=False)
        say(f"   serial 48 tok {s.tokens_per_second:.2f} tok/s; replays now {e.replays} (before {r0})")
        first = dec.prefill(e, p, None)
        m = dec.mtp_decode(e, first, 48, None, policy=dec.DepthPolicy(3, fixed=True), stop_eos=False)
        mst = ", ".join(f"{k} {1e3 * v / max(1, m.rounds):.2f}" for k, v in m.stages.items())
        say(f"   mtp3 48 tok {m.tokens_per_second:.2f} tok/s rounds {m.rounds} accepted {m.accepted} [ms/round: {mst}]"
            f" same={s.tokens == m.tokens}")
    eng.comm.barrier()
    say("== done")


if __name__ == "__main__":
    main()
