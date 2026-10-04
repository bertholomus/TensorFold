"""Prompt chunks and an eager decode step (model.forward, single-stream) with every small-kernel switch on against
every switch off (kernels.SMALL_SWITCHES): the logits and the KV caches (ring, compressed, indexer) bit for bit.

  python3 small_prompt_check.py --model M [--layers 22]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_bench import load_model  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import kernels as K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--layers", type=int, default=22)
args = ap.parse_args()
m = load_model(args.model, args.layers, [0, 20])
g = torch.Generator().manual_seed(3)
res = {}
for n in (700, 16, 2, 1):
    ids = torch.randint(1000, 100000, (n,), generator=g)
    outs = []
    for on in (False, True):
        for k in K.SMALL_SWITCHES:
            K.set_switch(k, on)
        sc = m.new_cache(2048)
        lg = m.forward(sc, ids.cuda(), 0, all_logits=True)
        # a decode step after the prompt (eager single-stream path)
        lg2 = m.forward(sc, ids[:1].cuda(), n, all_logits=True)
        torch.cuda.synchronize()
        flat = [lg, lg2] + [t for t in sc.ring] + [x for v in sc.comp.values() for x in (v if isinstance(v, tuple) else (v,))] + \
               [x for v in sc.index_k.values() for x in (v if isinstance(v, tuple) else (v,))]
        outs.append([t.clone() for t in flat])
    eq = all(torch.equal(a.view(torch.uint8) if a.dtype != torch.uint8 else a, b.view(torch.uint8) if b.dtype != torch.uint8 else b) for a, b in zip(*outs))
    res[n] = eq
    print(n, "prompt+step logits and caches equal (switches on vs off):", eq, flush=True)
print("ALL", all(res.values()))
