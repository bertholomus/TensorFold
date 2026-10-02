"""Layer-by-layer parity: TensorFold glm_moe_dsa (TP1) vs ExLlamaV3 reference on the same EXL3 checkpoint.

usage (container, both packages importable): python3 tf_parity.py MODEL_DIR
Prints per-module relative error of the residual stream and last-row logits agreement.
"""
import os, sys
import torch

os.environ.setdefault("TF_TP_WORLD", "1")
MODEL = sys.argv[1]
torch.manual_seed(0)

# ---------------- reference: ExLlamaV3 cache-less forward, module by module
from exllamav3 import Config, Model, Tokenizer

cfg = Config.from_directory(MODEL)
model = Model.from_config(cfg)
model.load("cuda:0")
tok = Tokenizer.from_config(cfg)
text = "The quick brown fox jumps over the lazy dog. In 1969, Neil Armstrong became the first person to walk on"
ids = tok.encode(text, add_bos=False)
if ids.dim() == 1:
    ids = ids.unsqueeze(0)
T = ids.shape[1]
print("tokens", T, ids[0, :12].tolist(), flush=True)

ref = []
with torch.inference_mode():
    params = {"attn_mode": "flash_attn_nc"}
    x = model.prepare_inputs(ids, params)
    for m in model.modules:
        x = m.prepare_for_device(x, params)
        x = m.forward(x, params)
        ref.append((m.key, x.float().clone()))
ref_logits = ref[-1][1][0]                       # [T, vocab]
print("ref modules", [k for k, _ in ref], flush=True)
model.unload()
del model
torch.cuda.empty_cache()

# ---------------- TensorFold: load rank 0 of world 1, run the prompt as one prefill chunk
from tensorfold.families.glm_moe_dsa.cuda import forward as fwd
from tensorfold.families.glm_moe_dsa.cuda.weights import load

w = load(MODEL, rank=0)
w.comm = None
w.meta["long_context"] = False
cap = 4096
b = fwd.Buffers(w, 512, cap, prefill=True)
st = fwd.State(w, cap, 512)
st.reset()
mine = []
with torch.no_grad():
    R = fwd.stage(w, st, b, ids[0].tolist())
    from tensorfold.families.glm_moe_dsa.cuda import glue, rope as rope_mod

    c = w.cfg
    fwd.embed(w, b, b.ids[:R], b.x[:R])
    mine.append(("embed", b.x[:R].float().clone()))
    rope_mod.table(b.cos[:R], b.sin[:R], st.pos_dev, R, c.rope_theta, c.qk_rope)
    nch = fwd.chunks_for(st, R)
    for layer in w.layers:
        fwd.layer_forward(layer, w, st, b, R, nch, st.pos, None)
        mine.append((f"layer{layer.index}", b.x[:R].float().clone()))
    glue.rmsnorm(b.x[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    full = torch.empty((R, w.head.n), dtype=torch.float32, device="cuda")
    b2 = fwd.Buffers(w, R, cap)                    # decode-style buffers: head over every row
    logits = fwd.mm(b2, b.fnormed[:R], w.head, b.fxs[:R], b2.logits[:R]).float()

# ---------------- compare
def rel(a, b):
    a = a.reshape(-1, a.shape[-1]).float().cuda(); b = b.reshape(-1, b.shape[-1]).float().cuda()
    return ((a - b).norm() / b.norm()).item(), torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()

refs = [v for k, v in ref if k == "model.embed_tokens" or ".layers." in k and k.count(".") == 2]
print(f"{'module':10s} {'rel_err':>10s} {'min_cos':>10s}")
for (name, a), r in zip(mine, refs):
    e, cmin = rel(a, r[0] if r.dim() == 3 else r)
    print(f"{name:10s} {e:10.5f} {cmin:10.5f}", flush=True)
V = min(logits.shape[1], ref_logits.shape[1])
lg, rl = logits[:, :V], ref_logits[:, :V]
e, cmin = rel(lg, rl)
agree = (lg.argmax(-1) == rl.argmax(-1)).float().mean().item()
print(f"logits     {e:10.5f} {cmin:10.5f}  argmax agreement {agree:.3f}  (vocab cmp {V})", flush=True)
print("ref argmax last 8:", rl.argmax(-1)[-8:].tolist())
print("tf  argmax last 8:", lg.argmax(-1)[-8:].tolist())
