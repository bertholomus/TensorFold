"""The served engine's device weights for one rank, as sha256 digests by the Zig loader's names (tf-dsv41-load).

  python zrec_weights.py MODEL_DIR RANK WORLD OUT.jsonl

Loads with the served loader (weights.load, the lane's rank cache when TF_DS_RANK_CACHE is set) plus the one derived
tensor Model.__init__ makes from the weights (the indexer's weights_proj in fp16). Expert pointer tables hold addresses,
so they are left out; every other tensor is digested from its device bytes.
"""

import hashlib
import json
import sys
import time

import torch

from tensorfold.families.deepseek_v41.cuda.weights import load


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    model_dir, rank, world, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    torch.cuda.set_device(0)
    t0 = time.time()
    w = load(model_dir, rank, world, dspark=True)
    secs = time.time() - t0
    rows = []

    def put(name: str, t: torch.Tensor) -> None:
        rows.append({"name": name, "bytes": t.numel() * t.element_size(), "sha256": digest(t)})

    def lin(name: str, x) -> None:
        put(f"{name}.words", x.words)
        put(f"{name}.suh", x.suh)
        put(f"{name}.svh", x.svh)

    gl = w.cfg.o_groups // world

    def block(p: str, lay) -> None:
        for k, part in enumerate(("fn", "scale", "base")):
            put(f"{p}.hc_attn_{part}", lay.hc_attn[k])
        for k, part in enumerate(("fn", "scale", "base")):
            put(f"{p}.hc_ffn_{part}", lay.hc_ffn[k])
        put(f"{p}.attn_norm.weight", lay.attn_norm)
        put(f"{p}.ffn_norm.weight", lay.ffn_norm)
        lin(f"{p}.attn.wq_a", lay.wq_a)
        put(f"{p}.attn.q_norm.weight", lay.q_norm)
        lin(f"{p}.attn.wq_b", lay.wq_b)
        lin(f"{p}.attn.wkv", lay.wkv)
        put(f"{p}.attn.kv_norm.weight", lay.kv_norm)
        put(f"{p}.attn.attn_sink", lay.sink)
        for g, x in enumerate(lay.wo_a):
            lin(f"{p}.attn.wo_a.slice.{rank * gl + g}", x)
        lin(f"{p}.attn.wo_b", lay.wo_b)
        if lay.comp_wkv is not None:
            lin(f"{p}.attn.compressor.wkv", lay.comp_wkv)
            put(f"{p}.attn.compressor.norm.weight", lay.comp_norm)
            if lay.comp_wgate is not None:
                lin(f"{p}.attn.compressor.wgate", lay.comp_wgate)
        if lay.idx_wq_b is not None:
            lin(f"{p}.attn.indexer.wq_b", lay.idx_wq_b)
            put(f"{p}.attn.indexer.weights_proj.weight", lay.idx_proj)
            put(f"{p}.attn.indexer.weights_proj_h", lay.idx_proj.to(torch.float16).contiguous())
            if lay.idx_wk is not None:
                lin(f"{p}.attn.indexer.wk", lay.idx_wk)
                put(f"{p}.attn.indexer.k_norm.weight", lay.idx_k_norm)
        put(f"{p}.ffn.gate.weight", lay.gate_w)
        put(f"{p}.ffn.gate.bias", lay.gate_b)
        ex = lay.experts
        h = hashlib.sha256()
        n = 0
        for t in ex.keep:                                  # gate trellises, then up, then down: the packed buffer
            h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            n += t.numel() * t.element_size()
        rows.append({"name": f"{p}.ffn.experts.trellis", "bytes": n, "sha256": h.hexdigest()})
        for j, k2 in enumerate((ex.gate_k2, ex.up_k2, ex.down_k2)):
            put(f"{p}.ffn.experts.k2{j}", k2)
        for name in ("suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d"):
            put(f"{p}.ffn.experts.{name}", getattr(ex, name))
        if lay.engram_wkv is not None:
            lin(f"{p}.engram.wkv", lay.engram_wkv)
            put(f"{p}.engram.qk", lay.engram_qk)

    put("embed.weight", w.embed)
    put("norm.weight", w.norm)
    lin("head", w.head)
    for i, lay in enumerate(w.layers):
        block(f"layers.{i}", lay)
    if w.dspark is not None:
        n = len(w.dspark.blocks)
        for j, lay in enumerate(w.dspark.blocks):
            block(f"mtp.{j}", lay)
        last = f"mtp.{n - 1}"
        lin("mtp.0.main_proj", w.dspark.main_proj)
        put("mtp.0.main_norm.weight", w.dspark.main_norm)
        put(f"{last}.norm.weight", w.dspark.norm)
        put(f"{last}.markov_head.embed.weight", w.dspark.markov_embed)
        put(f"{last}.markov_head.head.weight", w.dspark.markov_head)
        put(f"{last}.confidence_head.proj.weight", w.dspark.conf)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"rank": rank, "tensors": len(rows), "bytes": sum(r["bytes"] for r in rows), "load_s": round(secs, 1)}))


if __name__ == "__main__":
    main()
