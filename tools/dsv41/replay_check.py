"""Bounded replay vs exact prefill on long natural-text prompts (TP ranks, same command on each).

For each length: the first next-token distribution after the prompt (top-1, logprob gap) and a 64-token greedy
continuation in both modes (common prefix length).

  python3 replay_check.py --rank R --master <HEAD_IP> --model M [--engram E] --text T [--lengths 2048,8192,32768] \
      [--out F]

--master is rank 0's address on the link between the machines (default 127.0.0.1, which only suits --world 1).
"""

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--master", default="127.0.0.1", help="rank 0's address on the link between the machines")
    p.add_argument("--port", type=int, default=29671)
    p.add_argument("--model", required=True)
    p.add_argument("--engram")
    p.add_argument("--text", required=True)
    p.add_argument("--lengths", default="2048,8192,32768")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--out")
    a = p.parse_args()
    from tokenizers import Tokenizer

    from tensorfold.families.deepseek_v41.cuda.engine import DsEngine

    tok = Tokenizer.from_file(f"{a.model}/tokenizer.json")
    ids_all = tok.encode(open(a.text).read()).ids
    eng = DsEngine(Path(a.model), rank=a.rank, world=a.world, master=a.master, port=a.port, drafts=3,
                   context=max(int(x) for x in a.lengths.split(",")) + 1024, engram_dir=a.engram)
    eng.request.stop_eos = False
    rows = []
    for L in [int(x) for x in a.lengths.split(",")]:
        for off in (0, len(ids_all) // 3):
            prompt = [0] + ids_all[off:off + L - 1]
            if len(prompt) < L:
                prompt = [0] + (ids_all * 4)[off:off + L - 1]
            res = {}
            for mode in (False, True):
                eng.replay_mode = mode
                got: list = []
                t0 = time.time()
                st = eng._run(prompt, a.tokens, None, False, got.extend, True)
                lp = torch.log_softmax(eng.last_prefill_logits[0].float(), -1)
                res[mode] = (got, lp, st["prefill_s"])
            (ge, le, pe), (gr, lr, pr) = res[False], res[True]
            t_e = int(le.argmax())
            row = {"length": L, "offset": off, "top1_equal": int(lr.argmax()) == t_e,
                   "lp_gap_exact_top1": round(float((le[t_e] - lr[t_e]).abs()), 4),
                   "common_prefix": next((i for i, (x, y) in enumerate(zip(ge, gr)) if x != y), min(len(ge), len(gr))),
                   "prefill_s_exact": round(pe, 2), "prefill_s_replay": round(pr, 2)}
            rows.append(row)
            if a.rank == 0:
                print(json.dumps(row), flush=True)
    if a.rank == 0 and a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
