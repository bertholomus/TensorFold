"""Bytes a rank reads for one serial decode token, by tensor class, from the checkpoint headers (no GPU): what the TP2
split leaves each rank of every tensor a token touches, routed experts at top-k of n, DSpark per round.

  python3 bytes_profile.py --model M [--engram E] [--world 2] [--context 8192] [--out F]
"""
import argparse, collections, json, re, struct
from pathlib import Path


def headers(folder):
    out = {}
    for p in sorted(Path(folder).glob("*.safetensors")):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, e in h.items():
            out[k] = e["data_offsets"][1] - e["data_offsets"][0]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--engram"); p.add_argument("--world", type=int, default=2)
    p.add_argument("--context", type=int, default=8192); p.add_argument("--out")
    a = p.parse_args()
    cfg = json.loads((Path(a.model) / "config.json").read_text())
    t = cfg.get("text_config", cfg)
    W = a.world
    L, E, K = t["num_hidden_layers"], t["n_routed_experts"], t["num_experts_per_tok"]
    sizes = headers(a.model)
    by = collections.defaultdict(float)
    # what each rank holds of a tensor: split classes read 1/W, replicated ones read whole
    split_rules = [(r"\.attn\.wq_b\.", "attn wq_b (heads split)", True), (r"\.attn\.wo_a\.", "attn wo_a (groups split)", True),
                   (r"\.attn\.wo_b\.", "attn wo_b (groups split)", True), (r"^head\.", "head (vocab split)", True),
                   (r"\.engram\.wkv\.", "engram wkv (hash cols split)", True)]
    repl_rules = [(r"\.attn\.wq_a\.", "attn wq_a (replicated)"), (r"\.attn\.wkv\.", "attn wkv (replicated)"),
                  (r"\.attn\.compressor\.", "compressor (replicated)"), (r"\.attn\.indexer\.", "indexer (replicated)"),
                  (r"\.ffn\.gate\.", "MoE gate (replicated)"), (r"\.hc_", "mHC mixes (replicated)"),
                  (r"norm\.weight$|attn_sink", "norms + sinks (replicated)"), (r"\.engram\.(q|k)_weight", "engram qk (replicated)")]
    expert_rx = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")
    shared_rx = re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.")
    for name, nbytes in sizes.items():
        if name.startswith(("mtp.", "vision.", "aligner.", "image_", "embed.")):
            continue
        if expert_rx.match(name):
            by["routed experts (top-%d of %d, halves)" % (K, E)] += nbytes / W * K / E
            continue
        if shared_rx.match(name):
            by["shared expert (halves)"] += nbytes / W
            continue
        for rx, label, _ in split_rules:
            if re.search(rx, name):
                by[label] += nbytes / W
                break
        else:
            for rx, label in repl_rules:
                if re.search(rx, name):
                    by[label] += nbytes
                    break
            else:
                by["other: " + re.sub(r"\d+", "N", name)[:60]] += nbytes
    by["embed row"] = 2 * t["hidden_size"]
    # Engram rows: per token, per engram layer, this rank's hash columns, FP8 rows of engram dim + E8M0 scale per 32
    if a.engram:
        es = headers(a.engram)
        rows = [k for k in es if k.endswith("engram.embed.weight")]
        n_cols = (t.get("engram_ngram", t.get("engram_max_ngram_size", 3)) - 1) * t.get("engram_heads", t.get("engram_n_heads", 8))
        dim = t.get("engram_dim", t.get("engram_head_dim", 256))
        by["Engram rows (host preads)"] = len(rows) * (n_cols // W) * (dim + dim // 32)
    # KV reads at this context: window ring + top-k compressed rows, plus the indexer's scan of compressed keys
    ratios = t["compress_ratios"]
    hd, win, topk = t["head_dim"], t["sliding_window"], t.get("index_topk", 512)
    kv = 0.0
    for i in range(L):
        kv += win * hd                                   # FP8 ring
        if ratios[i]:
            kv += min(topk, a.context // ratios[i]) * (hd // 2 + hd // 16)
    idx_layers = t.get("index_source_layer_ids") or []
    kv += sum(a.context // max(ratios[i], 1) * (t.get("index_head_dim", 128) // 2 + 4) for i in idx_layers if i < L)
    by["KV + indexer keys at %dK context" % (a.context // 1024)] = kv
    total = sum(by.values())
    rows = sorted(by.items(), key=lambda kv: -kv[1])
    # DSpark drafter, one round (a 5-row block, its own experts at top-k of its routed count)
    mtp = collections.defaultdict(float)
    dk, de = t.get("dspark_num_experts_per_tok", 3), t.get("dspark_n_routed_experts", 128)
    for name, nbytes in sizes.items():
        if not name.startswith("mtp."):
            continue
        if re.search(r"\.ffn\.experts\.", name):
            mtp["drafter routed experts (block of 5 rows)"] += nbytes / W * min(1.0, 5 * dk / de)
        elif re.search(r"markov_head|confidence", name):
            mtp["drafter Markov + confidence heads"] += nbytes
        else:
            mtp["drafter attention, shared expert, rest"] += nbytes / (W if re.search(r"wq_b|wo_a|wo_b|shared_experts", name) else 1)
    out = {"world": W, "context": a.context, "serial_token_bytes": round(total), "serial_token_gib": round(total / 2**30, 3),
           "by_class": [{"class": k, "bytes": round(v), "mib": round(v / 2**20, 1), "share": round(v / total, 4)} for k, v in rows],
           "dspark_round": [{"class": k, "mib": round(v / 2**20, 1)} for k, v in sorted(mtp.items(), key=lambda kv: -kv[1])]}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
