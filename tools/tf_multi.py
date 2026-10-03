"""Concurrent decoding on the lane's ranks (``--parallel``), no HTTP: every reply equals its solo run, token for token,
and the aggregate speed at 1, 2 and 4 streams.

Every rank builds the engine with ``parallel`` streams and a solo decoder beside it (same weights). First every
rank decodes each prompt alone (greedy, MTP-3: the reference, and c1 on today's path); then rank 0 submits the
prompts to the concurrent scheduler at once (burst) and staggered, in groups of 1, 2 and 4, while the followers
replay rank 0's steps (``multi.follow``). Rank 0 prints each group's aggregate tok/s and whether every reply equals
its solo reply.

usage (TP4, lane stopped; tp4_run.sh):
  python3 tools/tf_multi.py MODEL RANK MASTER PORT CONTEXT TEXT [STREAMS=4] [TOKENS=128] [PROMPTS=8]
  TEXT: a UTF-8 file; prompt i is a question over a slice of it, lengths from a few hundred tokens to CONTEXT / 4
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

    t0 = time.time()
    eng = GlmEngine(model, rank=rank, master=master, port=port, policy="3", context=context, context_explicit=True,
                    parallel=streams)
    w = eng.w
    solo = dec.Engine(w, capacity=eng.multi.slot_cap, max_rows=MAX_ROWS, prefill_rows=eng.multi.prefill_rows,
                      graphs=True, graph_rows=GRAPH_ROWS, long_context=bool(w.meta.get("long_context")))
    say(f"== loaded in {time.time() - t0:.0f} s: {streams} slots of {eng.multi.slot_cap} rows, window {eng.limit}")
    app = GlmApp(eng, model, "GLM") if rank == 0 else None
    prompts = []
    for i in range(count):
        if rank == 0:
            raw = Path(text_path).read_text(errors="ignore")
            size = [300, 1500, 2600, 6000, 12000, 3000, 900, min(context // 4, 30000)][i % 8]
            start = (i * 7919 * 64) % max(1, len(raw) - size * 8)
            body = app.tok.decode(app.tok.encode(raw[start:start + size * 8], add_special_tokens=False).ids[:size])
            p = app._prepare({"messages": [{"role": "user", "content": body + "\n\nSummarize the text above."}],
                              "max_tokens": tokens, "temperature": 0,
                              "chat_template_kwargs": {"enable_thinking": False}}, True).prompt
        prompts.append(eng._share(p if rank == 0 else None))
    # solo references (every rank: the same steps)
    refs, solo_s = [], 0.0
    for p in prompts:
        first = dec.prefill(solo, p, None)
        torch.cuda.synchronize()
        m = dec.mtp_decode(solo, first, tokens, None, policy=dec.DepthPolicy(3, fixed=True), stop_eos=True)
        refs.append(m.tokens)
        solo_s += m.seconds
        say(f"   solo {len(p):6d}-token prompt: {len(m.tokens)} tokens, {m.tokens_per_second:.2f} tok/s, "
            f"accepted {m.accepted} of {m.drafted}")
    solo_tokens = sum(len(r) - 1 for r in refs)
    say(f"== solo: {solo_tokens} tokens in {solo_s:.1f} s of decode ({solo_tokens / solo_s:.2f} tok/s)")
    if rank != 0:
        eng.follow()
        eng.comm.barrier()
        return
    multi, sched = eng.multi, eng.scheduler
    for group in sorted({1, 2, min(4, streams), streams}):
        for staggered in (False, True):
            replies = [None] * len(prompts)
            done_at = [0.0] * len(prompts)

            def run(i: int) -> None:
                out: list[int] = []
                stats = sched.submit(list(prompts[i]), tokens, None, True, lambda new: (out.extend(new), False)[1],
                                     stop_eos=True)
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
            say(f"== c{group} {'staggered' if staggered else 'burst'}: replies == solo {same} / {len(prompts)}; "
                f"{made} tokens in {wall:.1f} s wall (prefill included: {made / wall:.2f} tok/s); "
                f"decode rounds by streams decoding: {rates}; graphs {used}")
            for i, (r, ref) in enumerate(zip(replies, refs)):
                if r is None or r[0] != ref:
                    first = next((k for k, (a, b) in enumerate(zip(r[0] if r else [], ref)) if a != b), None)
                    say(f"   prompt {i} ({len(prompts[i])} tokens) differs at token {first}: "
                        f"{(r[0] if r else [])[:12]} vs {ref[:12]}")
    multi.link.send(["stop"])
    eng.comm.barrier()
    say("== done")


if __name__ == "__main__":
    main()
