"""Speed of the sparse-attention kernel variants at decode shapes (1 and 4 rows, 32 heads, 128 window + 512 picks)."""

import json
import time

import torch
import triton

from tensorfold.families.deepseek_v41.cuda import kernels as K
from tensorfold.families.deepseek_v41.ops import fp4_pack, fp4_unpack

dev = "cuda"
BF = torch.bfloat16
torch.manual_seed(0)
res = {}
for R in (1, 4):
    q = (torch.randn(R, 32, 512, device=dev) * 0.05).to(BF)
    ring = torch.randn(144, 512, device=dev).to(BF)
    comp = torch.randn(100000, 512, device=dev).to(BF)
    pc = fp4_pack(comp, 16, True)
    cq = fp4_unpack(*pc, 16, True)
    sink = torch.randn(32, device=dev)
    idx = torch.randint(0, 100000, (R, 512), device=dev)
    pos = torch.arange(5000, 5000 + R, device=dev)
    z = torch.zeros(1, dtype=torch.int64, device=dev)
    for name, cc in (("bf16", cq), ("packed", pc)):
        f = lambda: K.sparse_attn(q, sink, ring, z, True, cc, idx, pos, 512 ** -0.5, 128)
        ms = triton.testing.do_bench(f, warmup=20, rep=200)
        res[f"R{R}_{name}_us"] = round(ms * 1000, 1)
print(json.dumps(res))
