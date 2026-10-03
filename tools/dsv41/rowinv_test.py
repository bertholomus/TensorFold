"""Which torch ops used in the engine's decode path change a row's bits with the number of rows (n = 1 vs 6)."""

import json

import torch
import torch.nn.functional as F

from tensorfold.families.deepseek_v41.ops import fp4_qd, fp8_qd, rms_norm

torch.manual_seed(0)
dev = "cuda"
BF = torch.bfloat16
out = {}


def check(name, fn, x):
    full = fn(x)
    rows = torch.cat([fn(x[i:i + 1]) for i in range(x.shape[0])], 0)
    out[name] = bool(torch.equal(full, rows))


x512 = (torch.randn(6, 512, device=dev) * 2).to(BF)
w512 = (torch.rand(512, device=dev) + 0.5).to(BF)
x5120 = (torch.randn(6, 5120, device=dev)).to(BF)
w5120 = (torch.rand(5120, device=dev) + 0.5).to(BF)
check("rms_norm_512", lambda t: rms_norm(t, w512, 1e-20), x512)
check("rms_norm_5120", lambda t: rms_norm(t, w5120, 1e-20), x5120)
check("fp4_qd_16_e4m3", lambda t: fp4_qd(t, 16, True), x512)
check("fp4_qd_32_pow2", lambda t: fp4_qd(t.view(-1, 32, 128).reshape(-1, 32 * 128) if False else t, 32, False), x512)
q = (torch.randn(6, 32, 128, device=dev)).to(BF)
check("fp4_qd_heads", lambda t: fp4_qd(t, 32, False), q)
h = (torch.randn(6, 4, 5120, device=dev)).to(BF)
check("stream_mean_sq", lambda t: t.float().square().mean(-1), h)
check("hc_pre_sum", lambda t: (torch.rand(1, 4, device=dev).expand(t.shape[0], 4)[..., None] * t.float()).sum(1), h)
key = torch.randn(6, 4, 5120, device=dev)
wq = torch.randn(4, 5120, device=dev)
check("engram_dot", lambda t: (t.float() * wq).sum(-1), h)
check("taps_mean", lambda t: t.float().mean(1).to(BF), h)
xf = torch.randn(6, 512, device=dev)
sf = torch.randn(6, 512, device=dev)

print(json.dumps(out, indent=1))
