"""In-engine decode speed (no HTTP) on TP ranks, same command on each: greedy replies of a fixed length for chat-
templated prompt ids (``decode_prompts.json``: {"set/name": [ids]}), DSpark and serial, median over reps.

  python3 decode_bench.py --rank R --master <HEAD_IP> --model M --prompts decode_prompts.json [--tokens 384]
"""

import argparse
import json
import statistics

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--master", required=True)
    p.add_argument("--port", type=int, default=29668)
    p.add_argument("--model", required=True)
    p.add_argument("--engram")
    p.add_argument("--prompts", required=True)
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--drafts", type=int, default=3)
    p.add_argument("--serial", action="store_true", help="also time serial decode")
    p.add_argument("--out")
    a = p.parse_args()
    from pathlib import Path

    from tensorfold.families.deepseek_v41.cuda.engine import DsEngine

    eng = DsEngine(Path(a.model), rank=a.rank, world=a.world, master=a.master, port=a.port, drafts=a.drafts,
                   context=8192, engram_dir=a.engram)
    eng.request.stop_eos = False
    prompts = json.load(open(a.prompts))
    rows = []
    for name, ids in prompts.items():
        for mode in (("draft", "serial") if a.serial else ("draft",)):
            tps, rounds, outs = [], [], []
            for _ in range(a.reps):
                got: list = []
                torch.cuda.synchronize()
                st = eng._run(ids, a.tokens, None, False, got.extend, mode == "draft")
                tps.append(st["tokens_per_second"])
                outs.append(tuple(got))
                if mode == "draft":
                    rounds.append(len(got) / max(st["rounds"], 1))
            row = {"prompt": name, "mode": mode, "tokens": a.tokens, "tps_median": round(statistics.median(tps), 2),
                   "tps_all": [round(x, 2) for x in tps], "identical": len(set(outs)) == 1}
            if rounds:
                row["tokens_per_round"] = round(statistics.median(rounds), 3)
                row.update({k: round(v, 2) for k, v in eng.last_spec_times.items()})
            rows.append(row)
            if a.rank == 0:
                print(json.dumps(row), flush=True)
    if a.rank == 0 and a.out:
        json.dump({"drafts": a.drafts, "tokens": a.tokens, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
