"""What the TP4 startup admission says for several --context values on this node (nothing else on the GPU), and what
the KV state really costs a token. usage (in the tf container): TF_TP_WORLD=4 python3 tools/tf_admit.py MODEL"""

import os
import sys
from pathlib import Path

import torch

from tensorfold.cuda.capacity import admit
from tensorfold.cuda.geometry import split_weights
from tensorfold.families.glm5_next.cuda.split import rule
from tensorfold.families.glm_moe_dsa.cuda import kv8
from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine
from tensorfold.families.glm_moe_dsa.cuda.weights import Config, embed_transform

model, world = Path(sys.argv[1]), int(os.environ.get("TF_TP_WORLD", "4"))
torch.cuda.set_device(0)
cfg = Config.read(model)
full = sum(1 for t in cfg.indexer_types if t == "full")
per_token = kv8.slot_bytes(cfg.layers + 1, cfg.kv_lora, cfg.qk_rope, full + 1, cfg.index_dim, kv8.MODE)
print(f"KV state a token a rank as allocated (TF_GLM_KV={kv8.MODE}): {per_token} B ({cfg.layers + 1} latent+rope rows, "
      f"{full + 1} indexer keys)")
geometry = GlmEngine._geometry
for ctx in (32768, 65536, 131072, 196608, 262144, 327680, 393216, 425984, 458752, 491520, 524288, 1048576):
    try:
        r = admit(model, ctx, True, torch, lambda text: geometry(text, world),
                  embed_transform(split_weights(rule, world), cfg.vocab, world, 0), rank=0,
                  world=world, gather=lambda s: [s] * world)
        print(f"{ctx:8d}: admitted, estimate {r['total_bytes_estimate'] / 2**30:.2f} GiB of {r['budget_bytes'] / 2**30:.2f}"
              f" (weights {r['weight_bytes_estimate'] / 2**30:.2f}, cache+workspace "
              f"{r['cache_workspace_bytes_estimate'] / 2**30:.2f}); largest window {r['largest_window']}", flush=True)
    except ValueError as exc:
        print(f"{ctx:8d}: refused: {str(exc)[:240]}", flush=True)
