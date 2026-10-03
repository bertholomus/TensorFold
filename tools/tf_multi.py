"""Concurrent decoding on the lane's ranks (``--parallel``), no HTTP: every reply equals its solo run, token for token,
and the aggregate speed at 1, 2 and 4 streams.

Every rank builds the engine with ``parallel`` streams and a solo decoder beside it (same weights). First every
rank decodes each prompt alone (greedy, MTP-3: the reference, and c1 on today's path); then rank 0 submits the
prompts to the concurrent scheduler at once (burst) and staggered, in groups of 1, 2 and 4, while the followers
replay rank 0's steps (``multi.follow``). Rank 0 prints each group's aggregate tok/s and whether every reply equals
its solo reply.

usage (TP4, lane stopped; tp4_run.sh):
  python3 tools/tf_multi.py MODEL RANK MASTER PORT CONTEXT TEXT [STREAMS=4] [TOKENS=128] [PROMPTS=8]
  TEXT: a UTF-8 file; prompt i is a question over a slice of it, lengths from a few hundred tokens to CONTEXT / 4.
  TF_MULTI_PROMPTS=chat: one-line chat requests instead (about 20 tokens: the batched fill of short prompts); mixed:
  chat and text prompts alternating; resume: pairs of prompts sharing a ~20k-token body (run with TF_GLM_KEEP_SLOTS=1:
  the second of a pair resumes from the first one's warm slot; its reply must still equal its fresh solo run, and the
  rows it kept are logged); long: prompt 0 a TF_MULTI_LONG-token text (default CONTEXT minus 8,192) and chat
  requests after it (one long stream beside short ones, e.g. with TF_GLM_EXTENTS=1). Each group also logs its first
  tokens' latencies (from submit). TF_MULTI_GROUPS: the group sizes to run (default 1, 2, 4 and STREAMS).
  TF_MULTI_PROFILE=S: then for each group of TF_MULTI_PROFILE_GROUPS (default STREAMS) chat prompts at once: rounds,
  rows, tokens and rank 0's host stages a round over 3 s unprofiled (the profiler inflates the host's stages), then rank
  0's kernel time by name over S seconds of their rounds (torch profiler, CUDA activity: the collectives' kernels and
  their waits included) and a round's share of it; with several groups, each kernel's cost a row (smallest to largest
  group). TF_MULTI_PROFILE_EAGER=1 profiles eager rounds (no graph replays); TF_MULTI_ONLY_PROFILE=1 skips the solo
  references and the equality groups (the profile alone).
  TF_MULTI_LOCAL=1: one rank alone on one GPU (RANK 0's shard of TF_TP_WORLD, the collectives faked as tools/tf_rounds.py
  fakes them): the replies are not the lane's, but each must still equal its solo run bit for bit (batched fills,
  resumes, extents, draft depth), and the timings are this rank's compute without the fabric.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import torch


@torch.no_grad()
def main() -> None:
    from tensorfold.families.glm_moe_dsa.cuda import decode as dec
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp
    from tensorfold.families.glm_moe_dsa.cuda.engine import GRAPH_ROWS, MAX_ROWS, GlmEngine

    model, rank, master, port, context = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), \
        int(sys.argv[5])
    text_path = sys.argv[6]
    streams = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    tokens = int(sys.argv[8]) if len(sys.argv) > 8 else 128
    count = int(sys.argv[9]) if len(sys.argv) > 9 else 8

    def say(*a):
        if rank == 0:
            print(*a, flush=True)

    import os

    comm = None
    if os.environ.get("TF_MULTI_LOCAL") == "1":
        from tf_rounds import Alone, Quiet                # (tools/ is this script's directory)

        from tensorfold.families.glm_moe_dsa.cuda import engine as eng_mod, forward as fwd, multi as multi_mod

        os.environ["TF_GLM_EMBED_SPLIT"] = "0"            # alone, no rank holds the others' embedding rows
        fwd.PROMPT_REDUCE = "gather"
        multi_mod.Watchdog = multi_mod.Link = Quiet
        eng_mod._store = lambda c: None
        comm = Alone(rank, int(os.environ.get("TF_TP_WORLD", "4")))
    t0 = time.time()
    eng = GlmEngine(model, rank=rank, master=master, port=port, policy="3", context=context, context_explicit=True,
                    parallel=streams, comm=comm)
    w = eng.w
    solo = dec.Engine(w, capacity=eng.multi.slot_cap, max_rows=MAX_ROWS, prefill_rows=eng.multi.prefill_rows,
                      graphs=True, graph_rows=GRAPH_ROWS, long_context=bool(w.meta.get("long_context")))
    say(f"== loaded in {time.time() - t0:.0f} s: {streams} slots of {eng.multi.slot_cap} rows, window {eng.limit}")
    app = GlmApp(eng, model, "GLM") if rank == 0 else None

    kind = os.environ.get("TF_MULTI_PROMPTS") or "text"
    topics = ["how a hash table works", "the history of the printing press", "how vaccines train the immune system",
              "the rules of chess for a beginner", "how a CPU executes an instruction", "the water cycle",
              "how compilers optimize loops", "the causes of the French Revolution"]
    prompts = []
    for i in range(count):
        if rank == 0:
            if kind == "long" and i == 0:
                raw = Path(text_path).read_text(errors="ignore")
                n_long = int(os.environ.get("TF_MULTI_LONG") or context - 8192)
                ids = app.tok.encode(raw[:n_long * 8], add_special_tokens=False).ids
                while len(ids) < n_long:                 # (a short file repeats)
                    ids = ids + ids
                content = app.tok.decode(ids[:n_long]) + "\n\nSummarize the text above."
            elif kind in ("chat", "long") or (kind == "mixed" and i % 2 == 0):
                content = f"Write a detailed explanation of {topics[i % len(topics)]}."
            elif kind == "resume":                       # pair i // 2: one ~20k-token body, two questions
                raw = Path(text_path).read_text(errors="ignore")
                start = ((i // 2) * 7919 * 64) % max(1, len(raw) - 200000)
                body = app.tok.decode(app.tok.encode(raw[start:start + 160000], add_special_tokens=False).ids[:20000])
                content = body + ("\n\nList three key points of the text above." if i % 2 == 0 else
                                  "\n\nSummarize the text above.")
            else:
                raw = Path(text_path).read_text(errors="ignore")
                size = [300, 1500, 2600, 6000, 12000, 3000, 900, min(context // 4, 30000)][i % 8]
                start = (i * 7919 * 64) % max(1, len(raw) - size * 8)
                body = app.tok.decode(app.tok.encode(raw[start:start + size * 8], add_special_tokens=False).ids[:size])
                content = body + "\n\nSummarize the text above."
            p = app._prepare({"messages": [{"role": "user", "content": content}], "max_tokens": tokens,
                              "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True).prompt
        prompts.append(eng._share(p if rank == 0 else None))
    only = os.environ.get("TF_MULTI_ONLY_PROFILE") == "1"
    # solo references (every rank: the same steps)
    refs, solo_s = [], 0.0
    for p in ([] if only else prompts):
        solo.kept = None                                 # every reference is a fresh prefill
        first = dec.prefill(solo, p, None)
        torch.cuda.synchronize()
        m = dec.mtp_decode(solo, first, tokens, None, policy=dec.DepthPolicy(3, fixed=True), stop_eos=True)
        refs.append(m.tokens)
        solo_s += m.seconds
        say(f"   solo {len(p):6d}-token prompt: {len(m.tokens)} tokens, {m.tokens_per_second:.2f} tok/s, "
            f"accepted {m.accepted} of {m.drafted}")
    solo_tokens = sum(len(r) - 1 for r in refs)
    if refs:
        say(f"== solo: {solo_tokens} tokens in {solo_s:.1f} s of decode ({solo_tokens / solo_s:.2f} tok/s)")
    if rank != 0:
        eng.follow()
        eng.comm.barrier()
        return
    multi, sched = eng.multi, eng.scheduler
    sizes = [int(v) for v in (os.environ.get("TF_MULTI_GROUPS") or "").split(",") if v.strip()]
    for group in ([] if only else (sizes or sorted({1, 2, min(4, streams), streams}))):
        for staggered in (False, True):
            replies = [None] * len(prompts)
            done_at = [0.0] * len(prompts)
            first_at = [None] * len(prompts)

            def run(i: int) -> None:
                out: list[int] = []
                t_sub = time.perf_counter()

                def got(new) -> bool:
                    if first_at[i] is None:
                        first_at[i] = time.perf_counter() - t_sub
                    out.extend(new)
                    return False

                stats = sched.submit(list(prompts[i]), tokens, None, True, got, stop_eos=True)
                replies[i] = (out, stats)
                done_at[i] = time.perf_counter()

            before = dict(multi.rounds)
            logged = len(multi.round_log)
            t = time.perf_counter()
            for g0 in range(0, len(prompts), group):
                threads = []
                for i in range(g0, min(len(prompts), g0 + group)):
                    th = threading.Thread(target=run, args=(i,))
                    th.start()
                    threads.append(th)
                    if staggered:
                        time.sleep(0.5)
                for th in threads:
                    th.join()
            wall = time.perf_counter() - t
            same = sum(r is not None and r[0] == ref for r, ref in zip(replies, refs))
            kept = [r[1].get("cached", 0) for r in replies if r is not None]
            decode_s = sum(r[1]["decode_s"] for r in replies if r is not None)
            made = sum(len(r[0]) - 1 for r in replies if r is not None)
            used = {k: multi.rounds[k] - before[k] for k in before}
            by: dict[int, list[float]] = {}
            for live, rows, spent, got in multi.round_log[logged:]:
                acc = by.setdefault(live, [0.0, 0.0, 0.0, 0.0])
                acc[0] += spent
                acc[1] += got
                acc[2] += 1
                acc[3] += rows
            rates = "; ".join(f"{n} decoding: {a[1] / a[0]:.2f} tok/s over {int(a[2])} rounds ({a[3] / a[2]:.1f} rows, "
                              f"{1e3 * a[0] / a[2]:.1f} ms a round)" for n, a in sorted(by.items()) if a[0] > 0)
            firsts = [f for f in first_at if f is not None]
            say(f"== c{group} {'staggered' if staggered else 'burst'}: replies == solo {same} / {len(prompts)}; "
                f"{made} tokens in {wall:.1f} s wall (prefill included: {made / wall:.2f} tok/s); "
                f"first tokens {min(firsts, default=0):.2f}-{max(firsts, default=0):.2f} s; rows kept {kept}; "
                f"decode rounds by streams decoding: {rates}; graphs {used}")
            for i, (r, ref) in enumerate(zip(replies, refs)):
                if r is None or r[0] != ref:
                    first = next((k for k, (a, b) in enumerate(zip(r[0] if r else [], ref)) if a != b), None)
                    say(f"   prompt {i} ({len(prompts[i])} tokens) differs at token {first}: "
                        f"{(r[0] if r else [])[:12]} vs {ref[:12]}")
    prof_s = float(os.environ.get("TF_MULTI_PROFILE") or 0)
    if prof_s > 0:
        from collections import defaultdict

        from torch.profiler import ProfilerActivity, profile

        groups = [g for g in (int(x) for x in (os.environ.get("TF_MULTI_PROFILE_GROUPS") or str(streams)).split(","))
                  if 0 < g <= streams]
        chat = [app._prepare({"messages": [{"role": "user", "content": f"Write a detailed explanation of {topic}."}],
                              "max_tokens": 4096, "temperature": 0,
                              "chat_template_kwargs": {"enable_thinking": False}}, True).prompt for topic in topics]
        top = int(os.environ.get("TF_MULTI_TOP", "30"))
        seen = {}
        for g in groups:
            stop = [False]                               # the clients leave once the profile is taken
            threads = [threading.Thread(target=lambda q=q: sched.submit(list(q), 4096, None, True,
                                                                       lambda new: stop[0], stop_eos=False))
                       for q in chat[:g]]
            for th in threads:
                th.start()
            while len([s for s in list(multi.streams.values()) if not s.done]) < g or multi.filling:
                time.sleep(0.05)
            seen0 = len(multi.round_log)                 # three whole rounds of every stream (graphs captured)
            while sum(1 for x in multi.round_log[seen0:] if x[0] == g) < 3:
                time.sleep(0.05)
            # varying windows (draft cut, depth policy) bring new round shapes: wait for 20 rounds without a capture
            quiet_from, caps, t_wait = len(multi.round_log), multi.rounds["captured"], time.perf_counter()
            while len(multi.round_log) - quiet_from < 20 and time.perf_counter() - t_wait < 90:
                if multi.rounds["captured"] != caps:
                    quiet_from, caps = len(multi.round_log), multi.rounds["captured"]
                time.sleep(0.05)
            time.sleep(1.0)
            a, q0, r0 = len(multi.round_log), dict(multi.stage_s), dict(multi.rounds)
            routed = []                                  # distinct routed experts of a round's last MoE layer
            spans0 = multi._sample_spans

            def spy(logits, spans, *, draft=False, **kw):
                if not draft:
                    R = sum(sp[1] for sp in spans)
                    p = multi.buf.pick[:R]
                    routed.append(int(torch.unique(p[p < w.cfg.experts]).numel()))
                return spans0(logits, spans, draft=draft, **kw)

            multi._sample_spans = spy
            time.sleep(3.0)
            multi._sample_spans = spans0
            plain = multi.round_log[a:]
            n = max(1, len(plain))
            spent = sum(x[2] for x in plain)
            say(f"-- {g} chat stream{'s' if g > 1 else ''} at once, unprofiled: {len(plain)} rounds, "
                f"{sum(x[1] for x in plain) / n:.1f} rows, {1e3 * spent / n:.1f} ms a round, "
                f"{sum(x[3] for x in plain) / n:.2f} tokens a round ({sum(x[3] for x in plain) / max(spent, 1e-9):.1f} "
                f"tok/s over the rounds); graphs {({k: multi.rounds[k] - r0[k] for k in r0})}; distinct routed experts "
                f"in the last MoE layer {sum(routed) / max(1, len(routed)):.1f} a round")
            say("   host stages a round (ms): " + ", ".join(f"{k} {1e3 * (multi.stage_s[k] - q0[k]) / n:.2f}"
                                                         for k in multi.stage_s))
            if any(c[0] for c in multi.draft_stats):     # TF_GLM_DRAFT_STATS: kept drafts by chain probability
                say("   drafts kept by chain probability: " + ", ".join(
                    f"{t / 10:.1f}+ {c[1]}/{c[0]}" for t, c in enumerate(multi.draft_stats) if c[0]))
                multi.draft_stats = [[0, 0] for _ in range(10)]
            graphs = multi.graphs
            if os.environ.get("TF_MULTI_PROFILE_EAGER") == "1":
                multi.graphs = None                      # rounds run eager while profiled (rank 0 only: same kernels)
            logged = len(multi.round_log)
            with profile(activities=[ProfilerActivity.CUDA]) as kp:
                time.sleep(prof_s)
                torch.cuda.synchronize()
            log = multi.round_log[logged:]
            multi.graphs = graphs
            stop[0] = True
            for th in threads:
                th.join()
            rounds = max(1, len(log))
            times: dict = defaultdict(float)
            counts: dict = defaultdict(int)
            for ev in kp.events():
                if ev.device_type == torch.autograd.DeviceType.CUDA:
                    name = ev.name if len(ev.name) < 90 else ev.name[:87] + "..."
                    times[name] += ev.time_range.elapsed_us()
                    counts[name] += 1
            total = sum(times.values())
            say(f"   profiled: {len(log)} rounds in {prof_s:.0f} s, {sum(x[1] for x in log) / rounds:.1f} rows, "
                f"{1e3 * sum(x[2] for x in log) / rounds:.1f} ms a round; kernel time {total / 1e3 / rounds:.2f} ms a round")
            for name, t in sorted(times.items(), key=lambda kv: -kv[1])[:top]:
                say(f"    {t / 1e3 / rounds:7.3f} ms a round {100 * t / max(total, 1e-9):5.1f}% "
                    f"x{counts[name] / rounds:6.1f}  {name}")
            seen[g] = (sum(x[1] for x in log) / rounds, {k: v / 1e3 / rounds for k, v in times.items()})
        lo, hi = min(seen, default=0), max(seen, default=0)
        if hi > lo and seen[hi][0] > seen[lo][0]:
            (ra, ta), (rb, tb) = seen[lo], seen[hi]
            slope = {k: (tb.get(k, 0.0) - ta.get(k, 0.0)) / (rb - ra) for k in set(ta) | set(tb)}
            say(f"-- a row's kernel time ({lo} -> {hi} streams, {ra:.1f} -> {rb:.1f} rows): "
                f"{sum(slope.values()):.3f} ms a row on a {sum(ta.values()) - ra * sum(slope.values()):.2f} ms base")
            for name, v in sorted(slope.items(), key=lambda kv: -kv[1])[:top]:
                say(f"    {v:7.3f} ms a row  ({ta.get(name, 0.0):6.3f} -> {tb.get(name, 0.0):6.3f} ms)  {name}")
    multi.link.send(["stop"])
    eng.comm.barrier()
    say("== done")


if __name__ == "__main__":
    main()
