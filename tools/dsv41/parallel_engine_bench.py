"""In-engine concurrent decode (no HTTP) on TP ranks: N streams of a fixed reply length at once through the engine's
own scheduler (rank 0 submits; the other ranks follow). Decode aggregate = every stream's tokens after its first one,
over the time from the first stream's first token to the last stream's last token; also the rounds with all N streams
live (steady state). Rank 0 prints and writes; kill the followers after it exits.

  python3 parallel_engine_bench.py --rank R --master <HEAD_IP> --model M --prompts decode_prompts.json \
      [--streams 1,2,4] [--tokens 384] [--reps 2] --out F
"""

import argparse
import json
import statistics
import threading
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--master", required=True)
    p.add_argument("--port", type=int, default=29670)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--set", default="b")
    p.add_argument("--streams", default="1,2,4")
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--out")
    a = p.parse_args()
    from pathlib import Path

    from tensorfold.families.deepseek_v41.cuda.engine import DsEngine

    eng = DsEngine(Path(a.model), rank=a.rank, world=a.world, master=a.master, port=a.port, drafts=3,
                   context=262144, parallel=a.parallel)
    if a.rank != 0:
        eng.follow()
        return
    prompts = [v for k, v in json.load(open(a.prompts)).items() if k.startswith(a.set + "/")]
    res = []
    for n in [int(x) for x in a.streams.split(",")]:
        for rep in range(a.reps):
            first, last, count = [None] * n, [None] * n, [0] * n
            log0 = len(eng.multi.round_log)

            def worker(i):
                eng.request.stop_eos = False

                def emit(new):
                    now = time.perf_counter()
                    if first[i] is None:
                        first[i] = now
                    last[i] = now
                    count[i] += len(new)
                    return False

                eng.generate(prompts[i % len(prompts)], a.tokens, None, emit, draft=True)

            th = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            for t in th:
                t.start()
            for t in th:
                t.join()
            span = max(last) - min(first)
            agg = (sum(count) - n) / span
            rounds = [r for r in eng.multi.round_log[log0:] if r[0] == n]
            steady = sum(r[3] for r in rounds) / max(sum(r[2] for r in rounds), 1e-9)
            row = {"streams": n, "rep": rep, "decode_aggregate_tps": round(agg, 2),
                   "steady_state_tps": round(steady, 2), "rounds_all_live": len(rounds),
                   "wall_s": round(time.perf_counter() - t0, 2), "tokens": count}
            res.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for n in sorted({r["streams"] for r in res}):
        rr = [r for r in res if r["streams"] == n]
        summary[n] = {"decode_aggregate_median": statistics.median(r["decode_aggregate_tps"] for r in rr),
                      "steady_state_median": statistics.median(r["steady_state_tps"] for r in rr)}
    print(json.dumps({"summary": summary}), flush=True)
    if a.out:
        json.dump({"rows": res, "summary": summary}, open(a.out, "w"), indent=1)
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
