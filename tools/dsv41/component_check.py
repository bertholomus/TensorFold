"""Engine pieces vs reference pieces on identical inputs (one rank, first few layers)."""

import argparse
import json
import sys
from pathlib import Path

import torch


def rel(a, b):
    a, b = a.float(), b.float()
    return {"rel_l2": float((a - b).norm() / (b.norm() + 1e-30)), "max_abs": float((a - b).abs().max()),
            "scale": float(b.abs().mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--engram", required=True)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--tools", default=str(Path(__file__).resolve().parent / "ref"),
                   help="folder holding dsv41_ref.py (default: tools/dsv41/ref of this checkout)")
    p.add_argument("--len", type=int, default=85)
    a = p.parse_args()
    sys.path.insert(0, a.tools)
    from dsv41_ref import Reference
    from tensorfold.families.deepseek_v41.cuda.model import Comm, Engram, Model
    from tensorfold.families.deepseek_v41.cuda.weights import load
    from tensorfold.families.deepseek_v41.ops import compressed_token_map

    torch.manual_seed(0)
    tm, _ = compressed_token_map(f"{a.model}/tokenizer.json")
    ref = Reference(a.model, a.engram, token_map=tm)
    w = load(a.model, 0, 1, n_layers=a.layers)
    model = Model(w, Comm(None, 1), Engram(a.engram, w.cfg, tm, 0, 1))
    c = w.cfg
    ids = torch.randint(0, 120000, (a.len,), device="cuda")
    emb = w.embed[ids].to(torch.bfloat16)
    x = (emb * 3).to(torch.bfloat16)          # a plausible normed-input scale
    for li in range(a.layers):
        lay = w.layers[li]
        sc = model.new_cache(a.len + 8)
        ya = model.attention(lay, x, sc, 0, {"kv_layer": None} if lay.ratio and lay.comp_wkv is None else {})
        yr = ref.attention(li, x, [slice(0, a.len)], [dict()])
        print(json.dumps({"layer": li, "piece": "attention", **rel(ya, yr)}), flush=True)
        ym = model.moe(lay, x)
        yrm = ref.moe(li, x)
        print(json.dumps({"layer": li, "piece": "moe", **rel(ym, yrm)}), flush=True)
        if lay.engram_wkv is not None:
            h = (emb[:, None, :].expand(-1, c.hc, -1) * 1.0).contiguous()
            hashes = ref.hasher(ids)
            he = model.engram_apply(lay, h, hashes[:, c.engram_layers.index(li)])
            hr = ref.engram_apply(li, h, hashes[:, c.engram_layers.index(li)])
            print(json.dumps({"layer": li, "piece": "engram", **rel(he - h, hr - h)}), flush=True)
        mixe = model.hc_mixes(emb[:, None, :].expand(-1, c.hc, -1).contiguous(), lay.hc_attn)
        mixr = ref.hc_mixes(emb[:, None, :].expand(-1, c.hc, -1).contiguous(), f"layers.{li}", "attn")
        print(json.dumps({"layer": li, "piece": "hc_comb", **rel(mixe[2], mixr[2])}), flush=True)


if __name__ == "__main__":
    main()
