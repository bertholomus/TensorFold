"""Each Triton kernel of the engine vs the plain-torch definition (random inputs, one GPU), plus row invariance."""

import json

import torch
import torch.nn.functional as F

from tensorfold.families.deepseek_v41.cuda import kernels as K
from tensorfold.families.deepseek_v41.ops import (fp8_qd, freqs_cis, hc_split_sinkhorn, rms_norm, rope_,
                                                  sparse_attn)

torch.manual_seed(0)
dev = "cuda"
BF = torch.bfloat16
res = {}


def rel(a, b):
    a, b = a.float(), b.float()
    fin = torch.isfinite(b)
    return float(((a - b)[fin]).norm() / (b[fin].norm() + 1e-30))


D, R = 5120, 7
# hc_pre / hc_post
h = (torch.randn(R, 4, D, device=dev) * 0.5).to(BF)
fn = torch.randn(24, 4 * D, device=dev) * 0.01
scale = torch.rand(3, device=dev) + 0.5
base = torch.randn(24, device=dev) * 0.1
pre_in = torch.rand(R, 4, device=dev)
nw = (torch.rand(D, device=dev) + 0.5).to(BF)
out = torch.empty(R, D, dtype=BF, device=dev)
pre_o, post, comb = (torch.empty(R, 4, device=dev), torch.empty(R, 4, device=dev), torch.empty(R, 4, 4, device=dev))
part = torch.empty(R * K.HC_BLOCKS * 32, device=dev)
K.hc_pre(h, fn, scale, base, pre_in, nw, 1e-20, 1e-6, 20, out, pre_o, post, comb, part)
xf = h.flatten(1).float()
mixes = (xf @ fn.t()) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-20)
p_r, po_r, c_r = hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
x_r = rms_norm((pre_in[..., None] * h.float()).sum(1).to(BF), nw, 1e-20)
print("pre_o", pre_o[0].tolist(), "ref", p_r[0].tolist(), flush=True)
res["hc_pre"] = {"pre": rel(pre_o, p_r), "post": rel(post, po_r), "comb": rel(comb, c_r), "x": rel(out, x_r)}
g = torch.randn(2, R, D, device=dev)
ho = torch.empty_like(h)
K.hc_post(g, h, po_r, c_r, ho)
y = (g[0] + g[1]).to(BF)
ref = (po_r[..., None] * y.float()[:, None] + (c_r[..., None] * h.float()[:, :, None, :]).sum(1)).to(BF)
res["hc_post"] = rel(ho, ref)

# rope tables
f = freqs_cis(64, 4096, 65536, 160000.0, 16.0, 32, 1)
cos, sin = f.real.contiguous().float(), f.imag.contiguous().float()
pos = torch.tensor([5, 77, 300, 1000, 2047, 3000, 4095], device=dev)

# kv_norm_rope
y = (torch.randn(R, 512, device=dev) * 3).to(BF)
w = (torch.rand(512, device=dev) + 0.5).to(BF)
ring = torch.zeros(144, 512, dtype=BF, device=dev)
kv = K.kv_norm_rope(y, w, cos, sin, pos, ring, pos % 144, 1e-20, True, 64)
kr = rms_norm(y, w, 1e-20)
rope_(kr[..., -64:], f[pos])
kr = fp8_qd(kr, 32)
res["kv_norm_rope"] = {"kv": rel(kv, kr), "ring": rel(ring[pos % 144], kr), "bits_equal": bool(torch.equal(kv, kr))}

# rope_heads
q = (torch.randn(R, 32, 512, device=dev)).to(BF)
q2 = q.clone()
K.rope_heads(q, cos, sin, pos, 64)
rope_(q2[..., -64:], f[pos])
res["rope_heads"] = {"rel": rel(q, q2), "bits_equal": bool(torch.equal(q, q2))}
K.rope_heads(q, cos, sin, pos, 64, inverse=True)
rope_(q2[..., -64:], f[pos], inverse=True)
res["rope_inverse"] = rel(q, q2)

# sparse attention, ring mode + compressed entries
Hl = 32
q = (torch.randn(R, Hl, 512, device=dev) * 0.05).to(BF)
ring = (torch.randn(144, 512, device=dev)).to(BF)
comp = (torch.randn(3000, 512, device=dev)).to(BF)
sink = torch.randn(Hl, device=dev)
idx = torch.randint(0, 2000, (R, 512), device=dev)
idx[:, 400:] = -1
o = K.sparse_attn(q, sink, ring, torch.zeros(1, dtype=torch.int64, device=dev), True, comp, idx, pos, 512 ** -0.5, 128)
# torch reference: keys = window (positions p-127..p from ring) + comp[idx]
outs = []
for r in range(R):
    p = int(pos[r])
    wp = torch.arange(p - 127, p + 1, device=dev)
    wk = ring[wp.clamp_min(0) % 144]
    keys = torch.cat([wk, comp], 0)
    ii = torch.cat([torch.where(wp >= 0, torch.arange(128, device=dev), -1), torch.where(idx[r] >= 0, idx[r] + 128, -1)])
    outs.append(sparse_attn(q[r:r + 1], keys, sink, ii[None], 512 ** -0.5))
o_r = torch.cat(outs, 0)
res["sparse_attn"] = rel(o, o_r)
# row invariance: rows alone vs together
o1 = torch.cat([K.sparse_attn(q[r:r + 1], sink, ring, torch.zeros(1, dtype=torch.int64, device=dev), True, comp,
                              idx[r:r + 1].contiguous(), pos[r:r + 1], 512 ** -0.5, 128) for r in range(R)])
res["sparse_attn_row_invariant"] = bool(torch.equal(o1, o))

# index score
iq = torch.randn(R, 32, 128, device=dev).to(BF)
ik = torch.randn(1500, 128, device=dev).to(BF)
wts = torch.randn(R, 32, device=dev).to(BF)
vis = torch.tensor([10, 500, 1000, 1499, 1500, 3, 0], device=dev)
sc = K.index_score(iq, ik, wts, vis, 1500)
sr = torch.zeros(R, 1500, device=dev)
for hh in range(32):
    sr += (iq[:, hh].float() @ ik.float().t()).relu() * wts[:, hh:hh + 1].float()
sr = sr.masked_fill(torch.arange(1500, device=dev)[None] >= vis[:, None], float("-inf"))
res["index_score"] = rel(sc, sr)
res["index_score_row_invariant"] = bool(torch.equal(K.index_score(iq[2:3], ik, wts[2:3], vis[2:3], 1500), sc[2:3]))

# route
lg = torch.randn(R, 384, device=dev) * 2
bias = torch.randn(384, device=dev) * 0.1
pick = torch.empty(R, 7, dtype=torch.int32, device=dev)
wt = torch.empty(R, 7, device=dev)
K.route(lg, bias, 6, 1.5, 384, pick, wt)
scr = F.softplus(lg).sqrt()
ind = (scr + bias).topk(6, dim=-1).indices
ww = scr.gather(1, ind)
ww = ww / (ww.sum(-1, keepdim=True) + 1e-20) * 1.5
res["route"] = {"picks_equal": bool(torch.equal(pick[:, :6].long().sort(-1).values, ind.sort(-1).values)),
                "shared_slot": bool((pick[:, 6] == 384).all()), "wts": rel(wt[:, :6].sort(-1).values, ww.sort(-1).values)}

# rowmm, rmsnorm
x = torch.randn(R, D, device=dev).to(BF)
W = torch.randn(384, D, device=dev).to(torch.float16)
res["rowmm"] = rel(K.rowmm(x, W), x.float() @ W.float().t())
res["rowmm_row_invariant"] = bool(torch.equal(K.rowmm(x[3:4], W), K.rowmm(x, W)[3:4]))
res["rmsnorm"] = {"rel": rel(K.rmsnorm(x, nw, 1e-20), rms_norm(x, nw, 1e-20)),
                  "bits_equal": bool(torch.equal(K.rmsnorm(x, nw, 1e-20), rms_norm(x, nw, 1e-20)))}
# packed FP4 caches: pack/unpack == quant-dequant; kernels on packed rows == on the dequantized bf16 rows
from tensorfold.families.deepseek_v41.ops import fp4_pack, fp4_qd, fp4_unpack
xc = torch.randn(3000, 512, device=dev).to(BF) * 3
pc = fp4_pack(xc, 16, True)
res["fp4_pack_comp"] = bool(torch.equal(fp4_unpack(*pc, 16, True).float(), fp4_qd(xc, 16, True).float()))
xi = torch.randn(1500, 128, device=dev).to(BF) * 2
pi = fp4_pack(xi, 32, False)
res["fp4_pack_index"] = bool(torch.equal(fp4_unpack(*pi, 32, False).float(), fp4_qd(xi, 32, False).float()))
cq = fp4_unpack(*pc, 16, True)
o_b = K.sparse_attn(q, sink, ring, torch.zeros(1, dtype=torch.int64, device=dev), True, cq, idx, pos, 512 ** -0.5, 128)
o_p = K.sparse_attn(q, sink, ring, torch.zeros(1, dtype=torch.int64, device=dev), True, pc, idx, pos, 512 ** -0.5, 128)
res["sparse_attn_packed_equal"] = [bool(torch.equal(o_b, o_p)), rel(o_p, o_b)]
ki = fp4_unpack(*pi, 32, False)
s_b = K.index_score(iq, ki, wts, vis, 1500)
s_p = K.index_score(iq, pi, wts, vis, 1500)
res["index_score_packed_equal"] = [bool(torch.equal(s_b, s_p)), rel(s_p, s_b)]
# direct check of the kernel-side dequant: a 1-row comp, idx pointing at it, compare against bf16 rows

# prompt-chunk index scores (8 rows a program) vs the row kernel
iq40 = torch.randn(40, 32, 128, device=dev).to(BF)
w40 = torch.randn(40, 32, device=dev).to(BF)
v40 = torch.randint(0, 1500, (40,), device=dev)
a40 = K.index_score(iq40, pi, w40, v40, 1500)
b40 = torch.cat([K.index_score(iq40[i:i + 1], pi, w40[i:i + 1], v40[i:i + 1], 1500) for i in range(40)])
res["index_score_rows_vs_row"] = rel(a40, b40)
print(json.dumps(res, indent=1))
