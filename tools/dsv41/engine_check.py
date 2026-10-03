"""Engine vs reference / kit oracle on TP ranks (run the same command on every rank).

  python3 engine_check.py --rank R --world 2 --master <HEAD_IP> --model M --engram E --oracle O \
      --ref-top ref_top.jsonl --out out.json [--decode N] [--chunk 512] [--limit K]

Prefill (teacher-forced) logits at every position of each oracle sequence (prompt + kit continuation), compared
with the reference's top-20 and the kit's; then, with --decode N, the first N positions after each prompt run again
as one-token decode steps and are compared with the prefill logits (decode-vs-prefill consistency of our engine).
"""

import argparse
import json
import time

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--master", default="127.0.0.1", help="rank 0's address on the link between the machines")
    p.add_argument("--port", type=int, default=29661)
    p.add_argument("--model", required=True)
    p.add_argument("--engram")
    p.add_argument("--oracle", required=True)
    p.add_argument("--ref-top")
    p.add_argument("--out")
    p.add_argument("--save-top")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--decode", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--layers", type=int, default=0)
    p.add_argument("--ref-inline", help="path of tools/ with dsv41_ref.py: compare logits with the reference directly")
    a = p.parse_args()
    torch.cuda.set_device(0)
    from tensorfold.families.deepseek_v41.cuda.model import Comm, Engram, Model
    from tensorfold.families.deepseek_v41.cuda.weights import load
    from tensorfold.families.deepseek_v41.ops import compressed_token_map

    nccl = None
    if a.world > 1:
        from tensorfold.cuda.comm import NCCL

        nccl = NCCL(a.rank, a.world, a.master, a.port)
    comm = Comm(nccl, a.world, rdma_bytes=8 << 20)
    t0 = time.time()
    w = load(a.model, a.rank, a.world, n_layers=a.layers or None)
    print(f"[check] rank {a.rank} loaded in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
          flush=True)
    eng = None
    if a.engram:
        tm, n = compressed_token_map(f"{a.model}/tokenizer.json")
        assert n == w.cfg.engram_cvocab
        eng = Engram(a.engram, w.cfg, tm, a.rank, a.world)
    model = Model(w, comm, eng)
    if nccl is not None:
        nccl.barrier()
    recs = [json.loads(line) for line in open(a.oracle)]
    refs = [json.loads(line) for line in open(a.ref_top)] if a.ref_top else [None] * len(recs)
    if a.limit:
        recs, refs = recs[:a.limit], refs[:a.limit]
    tot = {"ref": [0, 0], "kit": [0, 0], "dec": [0, 0], "dec_bits": [0, 0]}
    rows = []
    for rec, ref in zip(recs, refs):
        ids = rec["prompt_ids"] + rec["gen_ids"]
        sc = model.new_cache(len(ids) + 64)
        t1 = time.time()
        lg = []
        for s in range(0, len(ids), a.chunk):
            x = torch.tensor(ids[s:s + a.chunk], dtype=torch.long, device="cuda")
            lg.append(model.forward(sc, x, s, all_logits=True))
        lg = torch.cat(lg, 0)
        torch.cuda.synchronize()
        dt = time.time() - t1
        top1 = lg.argmax(-1).tolist()
        if a.save_top and a.rank == 0:
            lp = torch.log_softmax(lg.float(), -1)
            tv = lp.topk(20, dim=-1)
            with open(a.save_top, "a") as fh:
                fh.write(json.dumps({"index": rec["index"], "ids": ids, "top": [
                    {str(int(t)): float(v) for t, v in zip(tv.indices[j], tv.values[j])} for j in range(len(ids))]}) + "\n")
        if a.ref_inline and a.rank == 0:
            import sys
            sys.path.insert(0, a.ref_inline)
            from dsv41_ref import Reference
            if not hasattr(main, "_ref"):
                main._ref = Reference(a.model, a.engram)
            rl = main._ref.forward([ids], log=lambda m: None, n_layers=a.layers or None)["logits"][0].cuda()
            print(json.dumps({"inline_ref": rec["index"], "top1_agree": float((rl.argmax(-1) == lg.argmax(-1)).float().mean()),
                              "max_abs_diff": float((rl - lg).abs().max()), "mean_abs_diff": float((rl - lg).abs().mean()),
                              "logit_scale": float(rl.abs().mean())}), flush=True)
        r_ag = k_ag = n_pos = 0
        for j in range(1, len(ids)):
            n_pos += 1
            if ref is not None:
                rt = max(ref["top"][j - 1].items(), key=lambda kv: kv[1])[0]
                r_ag += int(int(rt) == top1[j - 1])
            kt = rec["tf_top"][j]
            if kt:
                k_ag += int(int(max(kt.items(), key=lambda kv: kv[1])[0]) == top1[j - 1])
        tot["ref"][0] += r_ag
        tot["ref"][1] += n_pos
        tot["kit"][0] += k_ag
        tot["kit"][1] += n_pos
        row = {"index": rec["index"], "len": len(ids), "prefill_s": round(dt, 2), "vs_ref": r_ag / n_pos,
               "vs_kit": k_ag / n_pos}
        if a.decode:
            P = len(rec["prompt_ids"])
            sc2 = model.new_cache(len(ids) + 64)
            for s in range(0, P, a.chunk):
                x = torch.tensor(ids[s:min(s + a.chunk, P)], dtype=torch.long, device="cuda")
                last = model.forward(sc2, x, s)
            d_ag = d_bits = 0
            steps = min(a.decode, len(ids) - P)
            t2 = time.time()
            for k in range(steps):
                d_ag += int(int(last[0].argmax()) == top1[P - 1 + k])
                d_bits += int(torch.equal(last[0], lg[P - 1 + k]))
                x = torch.tensor([ids[P + k]], dtype=torch.long, device="cuda")
                last = model.forward(sc2, x, P + k)
            torch.cuda.synchronize()
            row.update(decode_steps=steps, decode_vs_prefill_top1=d_ag / max(steps, 1),
                       decode_tok_s=steps / (time.time() - t2))
            tot["dec"][0] += d_ag
            tot["dec"][1] += steps
        rows.append(row)
        if a.rank == 0:
            print(json.dumps(row), flush=True)
    summ = {k: (v[0] / v[1] if v[1] else None) for k, v in tot.items()}
    summ["positions"] = tot["ref"][1]
    if a.rank == 0:
        print(json.dumps({"summary": summ}), flush=True)
        if a.out:
            json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
