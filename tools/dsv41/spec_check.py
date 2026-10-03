"""Drafted == serial on the engine (TP ranks, same command on each): greedy serial vs DSpark replies, speeds.

  python3 spec_check.py --rank R --model M --oracle O --tokens 128 --drafts 3 [--limit N] [--out F]
"""

import argparse
import json
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--master", default="127.0.0.1", help="rank 0's address on the link between the machines")
    p.add_argument("--port", type=int, default=29664)
    p.add_argument("--model", required=True)
    p.add_argument("--engram")
    p.add_argument("--oracle", required=True)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--drafts", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("--temperature", type=float, default=0.0)
    a = p.parse_args()
    from pathlib import Path

    from tensorfold.families.deepseek_v41.cuda.engine import DsEngine

    eng = DsEngine(Path(a.model), rank=a.rank, world=a.world, master=a.master, port=a.port, drafts=a.drafts,
                   context=8192, engram_dir=a.engram)
    eng.request.stop_eos = False
    recs = [json.loads(line) for line in open(a.oracle)]
    recs = [r for r in recs if r["kind"] == "chat"] + [r for r in recs if r["kind"] == "raw"]
    if a.limit:
        recs = recs[:a.limit]
    sampling = None
    if a.temperature > 0:
        from tensorfold.engine.exact_sampling import Sampling

        sampling = Sampling(1234, a.temperature, 20, 0.95, 0.0)
    rows = []
    same = 0
    for r in recs:
        res = {}
        for mode in ("serial", "draft"):
            got: list = []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            st = eng._run(r["prompt_ids"], a.tokens, sampling, False, got.extend, mode == "draft")
            res[mode] = (got, st)
        s_tok, d_tok = res["serial"][0], res["draft"][0]
        ok = s_tok == d_tok
        same += int(ok)
        first_diff = next((i for i, (x, y) in enumerate(zip(s_tok, d_tok)) if x != y), None)
        row = {"index": r["index"], "kind": r["kind"], "equal": ok, "first_diff": first_diff,
               "serial_tps": round(res["serial"][1]["tokens_per_second"], 2),
               "draft_tps": round(res["draft"][1]["tokens_per_second"], 2),
               "accept_per_round": round(res["draft"][1]["accepted"] / max(res["draft"][1]["rounds"], 1), 3),
               "tokens_per_round": round(len(d_tok) / max(res["draft"][1]["rounds"], 1), 3),
               "round_ms": {k: round(res["draft"][1].get(k, 0), 2) for k in ("draft_ms", "verify_ms", "absorb_ms")},
               "matches_kit_greedy_prefix": next((i for i, (x, y) in enumerate(zip(s_tok, r["gen_ids"])) if x != y),
                                                 min(len(s_tok), len(r["gen_ids"])))}
        rows.append(row)
        if a.rank == 0:
            print(json.dumps(row), flush=True)
    if a.rank == 0:
        summ = {"prompts": len(rows), "drafted_equals_serial": same, "temperature": a.temperature,
                "serial_tps_median": sorted(x["serial_tps"] for x in rows)[len(rows) // 2],
                "draft_tps_median": sorted(x["draft_tps"] for x in rows)[len(rows) // 2],
                "tokens_per_round_mean": sum(x["tokens_per_round"] for x in rows) / len(rows)}
        print(json.dumps({"summary": summ}), flush=True)
        if a.out:
            json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
