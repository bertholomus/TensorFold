"""Run the reference forward on an oracle file's sequences (prompt + greedy continuation) and compare.

  python3 ref_oracle.py --model M --engram E --oracle kit_oracle.jsonl --out ref_vs_kit.json [--no-kv-quant]

The oracle file is kit_bench.py's ``oracle`` output (or ours: same schema). For every sequence and every position j
>= 1 it compares the reference's next-token distribution after tokens < j with the oracle's teacher-forced top-k at j:
top-1 agreement, whether the oracle's top-1 is in our top-5, and the logprob gap on the oracle's top-1 token.
Reference top-20 per position is saved too (``--save-top``), so the reference itself becomes an oracle file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsv41_ref import Reference  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--engram")
    p.add_argument("--oracle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--save-top")
    p.add_argument("--no-kv-quant", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-tokens", type=int, default=4096)
    a = p.parse_args()
    recs = [json.loads(line) for line in open(a.oracle)]
    if a.limit:
        recs = recs[:a.limit]
    t0 = time.time()
    ref = Reference(a.model, a.engram, kv_quant=not a.no_kv_quant)
    print(f"[ref] loaded in {time.time() - t0:.1f} s", flush=True)
    seqs = [r["prompt_ids"] + r["gen_ids"] for r in recs]
    logits: list = []
    batch, total = [], 0
    for s in seqs + [None]:
        if s is None or (batch and total + len(s) > a.batch_tokens):
            t1 = time.time()
            logits += ref.forward(batch, log=lambda m: None)["logits"]
            print(f"[ref] batch of {len(batch)} sequences, {total} tokens in {time.time() - t1:.1f} s", flush=True)
            batch, total = [], 0
        if s is not None:
            batch.append(s)
            total += len(s)
    rows, top_out = [], []
    agree = n = in5 = 0
    gaps = []
    for r, lg in zip(recs, logits):
        lp = torch.log_softmax(lg.float(), -1)
        ids = r["prompt_ids"] + r["gen_ids"]
        tops = lp.topk(20, dim=-1)
        top_out.append({"index": r["index"], "ids": ids,
                        "top": [{str(int(t)): float(v) for t, v in zip(tops.indices[j], tops.values[j])}
                                for j in range(len(ids))]})
        a_seq = n_seq = 0
        for j, kit in enumerate(r["tf_top"]):
            if j == 0 or kit is None or j > lp.shape[0]:
                continue
            kit_best = max(kit.items(), key=lambda kv: kv[1])
            kit_tok = int(kit_best[0])
            ours = lp[j - 1]
            our_tok = int(ours.argmax())
            n_seq += 1
            a_seq += int(our_tok == kit_tok)
            in5 += int(kit_tok in tops.indices[j - 1][:5].tolist())
            gaps.append(abs(float(ours[kit_tok]) - float(kit_best[1])))
        agree += a_seq
        n += n_seq
        rows.append({"index": r["index"], "kind": r["kind"], "positions": n_seq, "top1_agree": a_seq / max(n_seq, 1)})
        print(json.dumps(rows[-1]), flush=True)
    gaps.sort()
    summary = {"sequences": len(recs), "positions": n, "top1_agreement": agree / max(n, 1),
               "kit_top1_in_our_top5": in5 / max(n, 1),
               "logprob_gap_median": gaps[len(gaps) // 2] if gaps else None,
               "logprob_gap_p95": gaps[int(len(gaps) * 0.95)] if gaps else None,
               "kv_quant": not a.no_kv_quant, "seconds": time.time() - t0}
    print(json.dumps(summary), flush=True)
    json.dump({"summary": summary, "per_sequence": rows}, open(a.out, "w"), indent=1)
    if a.save_top:
        with open(a.save_top, "w") as f:
            for t in top_out:
                f.write(json.dumps(t) + "\n")


if __name__ == "__main__":
    main()
