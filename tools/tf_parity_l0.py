"""Layer-0 sub-step parity: TensorFold glm_moe_dsa vs ExLlamaV3 on the same rows (TP1).

Compares: attn_norm out, q_a+norm (q_resid), q_b out (pre-rope nope part), kv_a latent (normed), attention output
(after o_proj), the post-attention residual, the MLP output. usage: python3 tf_parity_l0.py MODEL_DIR
"""
import os, sys
import torch

os.environ.setdefault("TF_TP_WORLD", "1")
MODEL = sys.argv[1]

from exllamav3 import Config, Model, Tokenizer

cfg = Config.from_directory(MODEL)
model = Model.from_config(cfg)
model.load("cuda:0")
tok = Tokenizer.from_config(cfg)
ids = tok.encode("The quick brown fox jumps over the lazy dog. In 1969, Neil Armstrong", add_bos=False)
ids = ids if ids.dim() == 2 else ids.unsqueeze(0)
T = ids.shape[1]
R = {}
with torch.inference_mode():
    params = {"attn_mode": "flash_attn_nc"}
    x = model.prepare_inputs(ids, params)
    emb = model.modules[0]
    x = emb.forward(emb.prepare_for_device(x, params), params)
    blk = model.modules[1]
    x = blk.prepare_for_device(x, params)
    R["x0"] = x.float().clone()
    y = blk.attn_norm.forward(x, params, out_dtype=torch.half)
    R["attn_in"] = y.float().clone()
    at = blk.attn
    qr = at.q_a_proj.forward(y, params)
    R["q_a_raw"] = qr.float().clone()
    qr = at.q_a_layernorm.forward(qr, params, out_dtype=torch.half)
    R["q_resid"] = qr.float().clone()
    q = at.q_b_proj.forward(qr, params)
    R["q_b"] = q.float().clone()
    ckv = at.kv_a_proj_with_mqa.forward(y, params)
    R["kv_a_raw"] = ckv.float().clone()
    R["latent"] = at.kv_a_layernorm.forward(ckv[..., :at.kv_lora_rank].contiguous(), params,
                                           out_dtype=torch.half).float().clone()
    a = at.forward(y, params)
    R["attn_out"] = a.float().clone()
    x2 = x + a
    R["x1"] = x2.float().clone()
    y2 = blk.mlp_norm.forward(x2, params, out_dtype=torch.half)
    R["mlp_in"] = y2.float().clone()
    m = blk.mlp.forward(y2, params)
    R["mlp_out"] = m.float().clone()
model.unload()
del model
torch.cuda.empty_cache()

from tensorfold.families.glm_moe_dsa.cuda import forward as fwd, glue, rope as rope_mod, mla_pe
from tensorfold.families.glm5_next.cuda import latent as latent_mod
from tensorfold.families.glm_moe_dsa.cuda.weights import load

w = load(MODEL, rank=0)
w.comm = None
w.meta["long_context"] = False
c = w.cfg
cap = 1024
b = fwd.Buffers(w, 256, cap, prefill=True)
st = fwd.State(w, cap, 256)
st.reset()
M = {}
layer = w.layers[0]
a = layer.dsa
HL = a.heads
with torch.no_grad():
    n = fwd.stage(w, st, b, ids[0].tolist())
    fwd.embed(w, b, b.ids[:n], b.x[:n])
    M["x0"] = b.x[:n].float().clone()
    rope_mod.table(b.cos[:n], b.sin[:n], st.pos_dev, n, c.rope_theta, c.qk_rope)
    glue.rmsnorm(b.x[:n], layer.in_norm, c.eps, b.normed[:n], b.xs[:n])
    M["attn_in"] = b.normed[:n].float().clone()
    fwd.mm(b, b.normed[:n], a.q_a, b.xs[:n], b.qr[:n])
    M["q_a_raw"] = b.qr[:n].float().clone()
    glue.rmsnorm(b.qr[:n], a.q_norm, c.eps, b.qr[:n], b.xs_qr[:n])
    M["q_resid"] = b.qr[:n].float().clone()
    q2 = b.q[:n].view(n, HL * c.qk_dim)
    fwd.mm(b, b.qr[:n], a.q_b, b.xs_qr[:n], q2)
    M["q_b"] = q2.float().clone()
    fwd.mm(b, b.normed[:n], a.kv_a, b.xs[:n], b.lat[:n])
    M["kv_a_raw"] = b.lat[:n].float().clone()
    glue.rmsnorm(b.lat[:n, :c.kv_lora], a.kv_norm, c.eps, b.lat[:n, :c.kv_lora], b.xs_lat[:n])
    M["latent"] = b.lat[:n, :c.kv_lora].float().clone()
    # full block from the start, as the engine runs it
    st.reset()
    fwd.stage(w, st, b, ids[0].tolist())
    fwd.embed(w, b, b.ids[:n], b.x[:n])
    rope_mod.table(b.cos[:n], b.sin[:n], st.pos_dev, n, c.rope_theta, c.qk_rope)
    glue.rmsnorm(b.x[:n], layer.in_norm, c.eps, b.normed[:n], b.xs[:n])
    g = fwd.dsa_block(layer, w, st.kc[0], st.pos_dev, b, n, fwd.chunks_for(st, n), None, st.pos, None, pc=st.pc[0])
    M["attn_out"] = g.sum(0).float().clone()
    glue.residual_add(b.x[:n], b.x[:n], g)
    M["x1"] = b.x[:n].float().clone()
    glue.rmsnorm(b.x[:n], layer.post_norm, c.eps, b.normed[:n], b.xs[:n])
    M["mlp_in"] = b.normed[:n].float().clone()
    g = fwd.mlp_block(layer, w, b, n) if layer.mlp is not None else fwd.moe_block(layer, w, b, n)
    M["mlp_out"] = g.sum(0).float().clone()

print(f"{'step':10s} {'rel_err':>9s} {'min_cos':>9s} shapes")
for k in R:
    r = R[k].reshape(-1, R[k].shape[-1]).cuda()
    m = M[k].reshape(-1, M[k].shape[-1]).cuda()
    if r.shape != m.shape:
        cols = min(r.shape[1], m.shape[1])
        print(f"{k:10s} shape mismatch ref {tuple(r.shape)} tf {tuple(m.shape)}; comparing first {cols} cols")
        r, m = r[:, :cols], m[:, :cols]
    e = ((m - r).norm() / r.norm()).item()
    cs = torch.nn.functional.cosine_similarity(m, r, dim=-1).min().item()
    print(f"{k:10s} {e:9.5f} {cs:9.5f}")
print("attrs: in_norm", getattr(layer, "in_norm", None) is not None, "post_norm", getattr(layer, "post_norm", None) is not None)
