"""Where GLM-5.3's decode time goes on TF_TP_WORLD ranks: every rank runs the same command, rank 0 prints the report.

usage (in the tf container, TF_TP_WORLD and NCCL_* set as tp4_start.sh sets them; see tp4_run.sh):
  python3 tools/tf_profile.py MODEL RANK MASTER PORT [CONTEXT] [TOKENS] [DEPTH_TOKENS]
Sections: end-to-end serial vs MTP decode of tf_greedy's prompts (stages, exactness), graph replay per window,
MTP step eager vs graph, sampling, NCCL all-gather latency, forward without collectives, kernel time by name, and
decode past the dense limit (DEPTH_TOKENS of filler, eager sparse path); prefill_ab compares prompt-path variants
(TF_PROFILE_AB) by time, first token and cache checksums. TF_PROFILE_LOCAL=1 runs one rank on its own node (collectives
replaced by Alone): the compute of its share, no fabric.
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


class Alone:
    """TF_PROFILE_LOCAL=1: this rank on its own node, the others absent. fp32 partials gather as this rank's plus
    zeros (sane activations, so routing and selection stay realistic); int settings gather as copies (every rank
    agrees). The compute of this rank's share of each step, no fabric; prompt partials reduce in gather mode."""

    def __init__(self, rank: int, world: int) -> None:
        self.rank, self.world = rank, world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if send.dtype == torch.float32:
            recv.zero_()
            recv.view(self.world, -1)[self.rank].copy_(send.view(-1))
        else:
            recv.view(self.world, -1).copy_(send.view(1, -1).expand(self.world, -1))

    def barrier(self) -> None:
        torch.cuda.synchronize()


def plane_sum(x, n: int) -> torch.Tensor:
    """An int64 checksum of a cache plane's first n rows (bf16 bits, or an FP8 plane's codes and scales)."""

    from tensorfold.families.glm_moe_dsa.cuda.kv8 import Kv8
    from tensorfold.families.glm_moe_dsa.cuda.kvq import KvQ

    return x.checksum(n) if isinstance(x, (Kv8, KvQ)) else x[:n].view(torch.int16).to(torch.int64).sum()


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


def topk_of(w) -> int:
    return int(w.cfg.index_topk)


def main() -> None:
    from tensorfold.families.glm_moe_dsa.cuda import decode as dec
    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine
    from tensorfold.families.glm_moe_dsa.cuda.mtp import mtp_compute
    from tensorfold.families.glm5_next.cuda.attention import CHUNK

    t = time.time()
    alone = os.environ.get("TF_PROFILE_LOCAL") == "1"
    if alone:
        fwd.PROMPT_REDUCE = "gather"
        # alone, no rank holds the others' embedding rows: keep the whole table (real rows, real routing)
        os.environ["TF_GLM_EMBED_SPLIT"] = "0"
    rows = int(os.environ.get("TF_PROFILE_ROWS") or 0) or None     # a prompt chunk's rows (default 2,048)
    eng = GlmEngine(MODEL, rank=RANK, master=MASTER, port=PORT, policy="3", context=CONTEXT, context_explicit=True,
                    comm=Alone(RANK, int(os.environ.get("TF_TP_WORLD", "4"))) if alone else None, prefill_rows=rows)
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

    if "policy" in SECTIONS:
        say("== MTP depth policies on the same prompts (tok/s a prompt, all tokens / all seconds)")
        grid = [tuple(float(x) for x in item.split(":")) for item in
                (os.environ.get("TF_PROFILE_POLICIES") or "3:0,4:0,3:0.3,4:0.3,5:0.3,6:0.3,4:0.5,5:0.5,6:0.5").split(",")]
        for most, conf in grid:
            rates, toks, secs, rounds = [], 0, 0.0, 0
            for p in prompts:
                first = dec.prefill(e, p, None)
                m = dec.mtp_decode(e, first, TOKENS, None, policy=dec.DepthPolicy(int(most), fixed=True,
                                                                                 confidence=conf), stop_eos=True)
                rates.append(m.tokens_per_second)
                toks += len(m.tokens) - 1
                secs += m.seconds
                rounds += m.rounds
            say(f"   most {int(most)} confidence {conf:.2f}: " + " / ".join(f"{r:.2f}" for r in rates)
                + f"  all {toks / secs:.2f} tok/s, {toks / rounds:.2f} tok a round")

    if "graphs" in SECTIONS:
        say(f"== main graph replay per window (pos {st.pos})")
        for R in sorted(e.graphs.main):
            g = e.graphs.main[R]
            say(f"   R={R[0]}: {timeit(g.replay):.2f} ms")

    if "side" in SECTIONS and getattr(e.buf, "side", None) is not None:
        say(f"== side stream (keys beside queries, shared expert beside routed): shipped graphs vs one stream (pos {st.pos})")
        side, e.buf.side = e.buf.side, None
        pool = torch.cuda.graph_pool_handle()
        one = {}
        try:
            for R in (1, 2, 4):
                for _ in range(2):
                    fwd.compute(w, st, e.buf, R)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    fwd.compute(w, st, e.buf, R)
                one[R] = g
        finally:
            e.buf.side = side
        for R in (1, 2, 4):
            two = e.graphs.main[(R, 0)]
            two.replay()
            torch.cuda.synchronize()
            a = e.buf.logits[:R].clone()
            one[R].replay()
            torch.cuda.synchronize()
            b = e.buf.logits[:R].clone()
            say(f"   R={R}: two streams {timeit(two.replay):.2f} ms, one stream {timeit(one[R].replay):.2f} ms;"
                f" logits equal: {torch.equal(a.view(torch.int16), b.view(torch.int16))}")
        one.clear()

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

    if "prefill_ab" in SECTIONS:
        from tensorfold.cuda.exl3 import experts as x3experts
        from tensorfold.families.glm5_next.cuda import prof

        n = int(os.environ.get("TF_PROFILE_PREFILL", "24000"))
        filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(int(n / 13.6)))
        p = app._prepare({"messages": [{"role": "user", "content": filler + "\n\nSummarize."}], "max_tokens": 8,
                          "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True).prompt
        # a variant is settings joined by "+": experts0 / experts1 / experts2 (prompt chunks' experts: one-tile
        # launches, grouped_rows, grouped_mma), gather / rowred (prompt partials: fp32 all-gather vs row shares),
        # overlap0 / overlap1 (one batch vs two micro-batches whose collectives overlap compute), absorb0 /
        # absorb1 / absorb2 (absorb and expand: one program a column block, Triton row blocks, latent_rows.cu),
        # sparse0 / sparse1 (sparse attention as chunk programs + merge vs a program a row merging in registers),
        # select0 / select1 (prompt chunks score every bucket token vs up to their last visible one), shared0 /
        # shared1 (the shared expert after the routed experts and added, vs first and added by their combine), profile
        # (TF_GLM_PROFILE block times, with syncs: slower); every run is compared to the first. Before each timed run
        # the same settings prefill a TF_PROFILE_WARM-token prompt (default 6000; 0: none), so first-use compiles land
        # outside the timing. absorb3: absorb / expand as batched tensor-core matmuls; qscr0 / qscr1: a quantized latent's prompt
        # attention dequantizing in registers vs reading the fp16 scratch (same bits), qscr2: the bf16 latent-domain
        # scratch through the bf16 kernels (other bits); experts3: grouped_mma2
        # (each warp's weight tiles decoded into mma fragments, one chain over K: other bits), experts4: grouped_mma3
        # (gate/up rotating its rows from the layer input, down's epilogue inside, bf16 slot outputs: other bits);
        # gemm0 / gemm1: the EXL3 prompt GEMM's fixed tiles vs tiles by shape (TF_EXL3_PREFILL_TILES); skt16 / skt32:
        # prompt sparse attention's key tile (TF_GLM_PROMPT_KT); span0 / span1: prompt chunks' indexer scores a program
        # a 64-token tile vs a program a span with its rows' queries held (TF_GLM_SCORES_SPAN); fuse0 / fuse1: prompt
        # chunks' residual add and next RMSNorm as two launches vs one (TF_GLM_FUSE_NORM, same bits); deq0 / deqbf16 /
        # deqfp16 / deqauto: the EXL3 prompt GEMM on rotated rows vs the weight dequantized into the model's basis
        # (bf16, cuBLAS; fp16, Triton; auto: cuBLAS for bf16 outputs, Triton for fp32 partials) (TF_GLM_PROMPT_DEQ:
        # other bits); nw2 / nw4: mma3's gate/up columns a program, 256 or 512 (TF_EXL3_MMA3_NW, same bits); inv0 /
        # inv1: the chunk-invariant prompt path off / on (TF_GLM_PROMPT_INVARIANT).
        variants = (os.environ.get("TF_PROFILE_AB")
                    or "experts0+gather+overlap0+absorb0+sparse0+select0+shared0,"
                       "experts2+rowred+overlap1+absorb2+sparse1+select1+shared1").split(",")
        warm_n = int(os.environ.get("TF_PROFILE_WARM", "6000"))
        warm = None
        if warm_n > 0:
            wf = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(int(warm_n / 13.6)))
            warm = app._prepare({"messages": [{"role": "user", "content": wf + "\n\nSummarize."}], "max_tokens": 8,
                                 "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True).prompt
        say(f"== prefill A/B on a {len(p)}-token prompt{f' (each after a {len(warm)}-token warm-up)' if warm else ''}: "
            f"{', '.join(variants)}")
        from tensorfold.families.glm_moe_dsa.cuda import mla_pe

        from tensorfold.families.glm_moe_dsa.cuda import select as select_mod

        from tensorfold.families.glm_moe_dsa.cuda import x3 as x3mod

        def settings():
            return (x3experts.PROMPT, fwd.PROMPT_EXPERTS, fwd.PROMPT_REDUCE, fwd.OVERLAP_ROWS, mla_pe.PROMPT_RB,
                    mla_pe.ABSORB, mla_pe.FUSED_ROWS, select_mod.TRIM, fwd.SHARED_INLINE, mla_pe.QSCRATCH,
                    mla_pe.QSCRATCH_BF16, mla_pe.PROMPT_KT, x3mod.PROMPT_TILES, select_mod.SCORES_SPAN,
                    fwd.FUSE_NORM, x3mod.PROMPT_DEQ, x3experts.MMA3_NW, fwd.invariant.INVARIANT)

        def restore(v):
            (x3experts.PROMPT, fwd.PROMPT_EXPERTS, fwd.PROMPT_REDUCE, fwd.OVERLAP_ROWS, mla_pe.PROMPT_RB,
             mla_pe.ABSORB, mla_pe.FUSED_ROWS, select_mod.TRIM, fwd.SHARED_INLINE, mla_pe.QSCRATCH, mla_pe.QSCRATCH_BF16,
             mla_pe.PROMPT_KT, x3mod.PROMPT_TILES, select_mod.SCORES_SPAN, fwd.FUSE_NORM, x3mod.PROMPT_DEQ,
             x3experts.MMA3_NW, fwd.invariant.INVARIANT) = v

        shipped = settings()
        runs = []
        try:
            for v in variants:
                restore(shipped)
                timed = False
                for s in v.split("+"):
                    if s in ("experts0", "experts1", "experts2", "experts3", "experts4"):
                        x3experts.PROMPT = s != "experts0"
                        fwd.PROMPT_EXPERTS = {"experts2": "mma", "experts3": "mma2", "experts4": "mma3"}.get(s, "rows")
                    elif s in ("gather", "rowred"):
                        fwd.PROMPT_REDUCE = "rows" if s == "rowred" else s
                    elif s in ("overlap0", "overlap1"):
                        fwd.OVERLAP_ROWS = 256 if s == "overlap1" else 1 << 30
                    elif s in ("absorb0", "absorb1", "absorb2", "absorb3"):
                        mla_pe.PROMPT_RB = 1 << 30 if s == "absorb0" else 64
                        mla_pe.ABSORB = {"absorb2": "cuda", "absorb3": "bmm"}.get(s, "triton")
                    elif s in ("fuse0", "fuse1"):
                        fwd.FUSE_NORM = s == "fuse1"
                    elif s in ("span0", "span1"):
                        select_mod.SCORES_SPAN = s == "span1"
                    elif s in ("skt16", "skt32"):
                        mla_pe.PROMPT_KT = int(s[3:])
                    elif s in ("gemm0", "gemm1"):
                        x3mod.PROMPT_TILES = s == "gemm1"
                    elif s in ("deq0", "deqbf16", "deqfp16", "deqauto"):
                        x3mod.PROMPT_DEQ = s[3:]
                    elif s in ("nw2", "nw4"):
                        x3experts.MMA3_NW = int(s[2:])
                    elif s in ("inv0", "inv1"):
                        fwd.invariant.INVARIANT = s == "inv1"
                    elif s in ("qscr0", "qscr1", "qscr2"):
                        mla_pe.QSCRATCH, mla_pe.QSCRATCH_BF16 = s != "qscr0", s == "qscr2"
                    elif s in ("sparse0", "sparse1"):
                        mla_pe.FUSED_ROWS = 64 if s == "sparse1" else 1 << 30
                    elif s in ("select0", "select1"):
                        select_mod.TRIM = s == "select1"
                    elif s in ("shared0", "shared1"):
                        fwd.SHARED_INLINE = s == "shared1"
                    elif s == "profile":
                        timed = True
                    else:
                        raise ValueError(f"unknown A/B setting {s!r}")
                if warm is not None:
                    dec.prefill(e, warm, None)
                prof.active = timed and prof.ENABLED
                torch.cuda.synchronize()
                t = time.perf_counter()
                first = dec.prefill(e, p, None)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                prof.active = False
                n_ = len(p)
                sums = torch.stack([plane_sum(kc, n_) for kc in st.kc] + [plane_sum(pc, n_) for pc in st.pc]
                                   + [plane_sum(ix, n_) for ix in (st.index or [])] + [plane_sum(st.mtp_kc, n_)])
                runs.append((first, sums.cpu(), e.last_hidden.clone()))
                a, b_ = runs[0], runs[-1]
                say(f"   {v}: {dt:.1f} s ({n_ / dt:.0f} tok/s), first token {first}; vs the first: same first token "
                    f"{a[0] == first}, same cache checksums {torch.equal(a[1], b_[1])}, same last hidden "
                    f"{torch.equal(a[2].view(torch.int16), b_[2].view(torch.int16))}")
                if timed:
                    prof.report(n_)
        finally:
            restore(shipped)

    if "prefill" in SECTIONS:
        from tensorfold.families.glm5_next.cuda import prof

        n = int(os.environ.get("TF_PROFILE_PREFILL", "24000"))
        filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(int(n / 13.6)))
        p = app._prepare({"messages": [{"role": "user", "content": filler + "\n\nSummarize."}], "max_tokens": 8,
                          "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True).prompt
        say(f"== prefill of a {len(p)}-token prompt, chunk by chunk (block times with syncs: TF_GLM_PROFILE=1)")
        chunk_times = []
        orig = fwd.compute

        def timed_compute(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*a, **k)
            torch.cuda.synchronize()
            if a[2].prefill:
                chunk_times.append(time.perf_counter() - t0)
            return out

        fwd.compute = timed_compute
        prof.active = prof.ENABLED
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            dec.prefill(e, p, None)
        finally:
            fwd.compute = orig
            prof.active = False
        torch.cuda.synchronize()
        total = time.perf_counter() - t
        say(f"   {len(p)} tokens in {total:.1f} s ({len(p) / total:.0f} tok/s); peak allocated "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; chunks (s): "
            + " ".join(f"{x:.1f}" for x in chunk_times))
        if prof.ENABLED:
            prof.report(len(p))

    if "union" in SECTIONS:
        # how much consecutive rows' selections overlap (a sparse kernel over a block's union reads each gathered key
        # once for the block): over a TF_PROFILE_PREFILL-token prefill, the selections of TF_PROFILE_UNION_LAYERS (full
        # indexer layers) in every chunk, the union of blocks of 2 / 4 / 8 / 16 rows against 2,048 a row
        n = int(os.environ.get("TF_PROFILE_PREFILL", "32000"))
        layers = {int(x) for x in (os.environ.get("TF_PROFILE_UNION_LAYERS") or "2,20,40,60").split(",")}
        filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(int(n / 13.6)))
        text = os.environ.get("TF_PROFILE_UNION_TEXT")
        if text:                                          # a real document instead of the filler
            body = Path(text).read_text(errors="ignore")[:n * 4]
        else:
            body = filler
        p = app._prepare({"messages": [{"role": "user", "content": body + "\n\nSummarize."}], "max_tokens": 8,
                          "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True).prompt[:n]
        from tensorfold.families.glm_moe_dsa.cuda import select as select_mod

        seen: dict = {}
        calls = [0]
        current = [None]                                  # the layer whose dsa_block is running (MTP: None)
        real, real_block = select_mod.select_tokens, fwd.dsa_block

        def block(layer, *a, **k):
            current[0] = layer.index
            try:
                return real_block(layer, *a, **k)
            finally:
                current[0] = None

        def spy(qi, wts, keys, pos, R, topk, pos_dev, *, tokens, counts, **kw):
            real(qi, wts, keys, pos, R, topk, pos_dev, tokens=tokens, counts=counts, **kw)
            calls[0] += 1
            if current[0] in layers and pos is not None:
                seen.setdefault(current[0], []).append((pos, tokens[:R].clone(), counts[:R].clone()))

        select_mod.select_tokens, fwd.dsa_block = spy, block
        try:
            dec.prefill(e, p, None)
            torch.cuda.synchronize()
        finally:
            select_mod.select_tokens, fwd.dsa_block = real, real_block
        say(f"== selection overlap over a {len(p)}-token prompt ({calls[0]} selections, layers {sorted(seen)})")
        for layer, parts in sorted(seen.items()):
            for pos, tok, cnt in parts:
                keep = cnt >= topk_of(w)
                tok = tok[keep][:, :topk_of(w)].long()
                if tok.shape[0] < 16:
                    continue
                line = f"   layer {layer} rows {pos}..{pos + len(cnt) - 1} ({tok.shape[0]} full rows):"
                for B in (2, 4, 8, 16):
                    m = tok.shape[0] // B * B
                    blk = tok[:m].view(m // B, B * tok.shape[1])
                    srt = blk.sort(dim=1).values
                    distinct = 1 + (srt[:, 1:] != srt[:, :-1]).sum(1)
                    line += f" B{B} union {float(distinct.float().mean()) / tok.shape[1]:.2f}x"
                say(line)

    if "prefill_kernels" in SECTIONS:
        # kernel time by name over one prefill (torch profiler, CUDA activity only), then the same prefill untimed
        n = int(os.environ.get("TF_PROFILE_PREFILL", "24000"))
        filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(int(n / 13.6)))
        p = app._prepare({"messages": [{"role": "user", "content": filler + "\n\nSummarize."}], "max_tokens": 8,
                          "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True).prompt
        dec.prefill(e, p[:6000], None)                    # first-use compiles outside the timing
        torch.cuda.synchronize()
        t = time.perf_counter()
        dec.prefill(e, p, None)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        say(f"== prefill kernels, a {len(p)}-token prompt: {dt:.1f} s ({len(p) / dt:.0f} tok/s) without the profiler")
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as kp:
            dec.prefill(e, p, None)
            torch.cuda.synchronize()
        kernel_table(kp, top=int(os.environ.get("TF_PROFILE_TOP", "40")))

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
        from tensorfold.families.glm_moe_dsa.cuda.select import sparse_bucket

        g = e.graphs.sparse.get((1, 0, sparse_bucket(st.pos, 1))) if e.graphs is not None else None
        if g is not None:                      # one step two ways at the same state: graph logits vs eager logits
            fwd.stage(w, st, e.buf, [first])
            g.replay()
            torch.cuda.synchronize()
            a = e.buf.logits[:1].clone()
            fwd.compute(w, st, e.buf, 1, nch=fwd.chunks_for(st, 1), host_pos=st.pos)
            torch.cuda.synchronize()
            b = e.buf.logits[:1].clone()
            say(f"   sparse graph logits == eager logits: {torch.equal(a.view(torch.int16), b.view(torch.int16))}")
        r0 = dict(e.replays)
        s = dec.serial_decode(e, first, 48, None, stop_eos=False)
        say(f"   serial 48 tok {s.tokens_per_second:.2f} tok/s; replays now {e.replays} (before {r0})")
        graphs, e.graphs = e.graphs, None
        first = dec.prefill(e, p, None)
        s2 = dec.serial_decode(e, first, 48, None, stop_eos=False)
        e.graphs = graphs
        say(f"   serial eager 48 tok {s2.tokens_per_second:.2f} tok/s; same tokens as graphs: {s.tokens == s2.tokens}")
        for most, conf in ((3, 0.0), (4, 0.3)):
            first = dec.prefill(e, p, None)
            m = dec.mtp_decode(e, first, 48, None, policy=dec.DepthPolicy(most, fixed=True, confidence=conf),
                               stop_eos=False)
            mst = ", ".join(f"{k} {1e3 * v / max(1, m.rounds):.2f}" for k, v in m.stages.items())
            say(f"   mtp {most}/{conf} 48 tok {m.tokens_per_second:.2f} tok/s rounds {m.rounds} accepted {m.accepted}"
                f" [ms/round: {mst}] same={s.tokens == m.tokens}")
    eng.comm.barrier()
    say("== done")


if __name__ == "__main__":
    main()
