"""DeepSeek-V4.1 forward on one TP rank: rows of one sequence at consecutive positions (a prompt chunk, a decode
token, a verify window), updating that sequence's caches.

Phase 1 (correctness first): EXL3 linears and the grouped expert kernel from TensorFold, attention / indexer /
compressor / mHC in plain torch following DeepSeek's definitions, partial sums gathered from every rank and added in
rank order so both ranks hold the same bits.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from tensorfold.cuda.exl3 import experts as exl3_experts
from tensorfold.cuda.exl3 import prefill as exl3_prefill

from ..config import Cfg
from ..ops import (BF16, F32, EngramHasher, fp4_qd, fp8_qd, freqs_cis, hc_split_sinkhorn, rms_norm, rope_,
                   sparse_attn)

RING_EXTRA = 16                     # window ring slots beyond the 128-token window (verify windows never clobber)
KV_QUANT = os.environ.get("TF_DS_KV", "native") != "bf16"


class Comm:
    """All-gather of fp32 partials then a rank-order sum (identity on one rank)."""

    def __init__(self, nccl, world: int):
        self.nccl, self.world = nccl, world
        self._buf: dict = {}

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """[world, *x.shape] of every rank's x."""

        if self.world == 1:
            return x[None]
        x = x.contiguous()
        out = torch.empty((self.world, *x.shape), dtype=x.dtype, device=x.device)
        self.nccl.all_gather(x.view(-1), out.view(-1))
        return out

    def sum(self, x: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return x
        g = self.gather(x)
        acc = g[0].clone()
        for r in range(1, self.world):
            acc += g[r]
        return acc


def mm(layer, x: torch.Tensor, out_dtype=BF16, ws: exl3_prefill.Workspace | None = None) -> torch.Tensor:
    """x [M, K] @ W: the row-invariant EXL3 linear up to 128 rows, the prompt GEMM beyond."""

    m = x.shape[0]
    if m <= 128:
        return layer(x.contiguous(), out_dtype=out_dtype)
    out = torch.empty((m, layer.n), dtype=out_dtype, device=x.device)
    return exl3_prefill.matmul(layer, x.contiguous(), out, ws or _WS)


_WS = exl3_prefill.Workspace()


@dataclass
class SeqCache:
    """One sequence's caches on this rank (replicated across ranks: KV is one latent shared by all heads)."""

    cap: int
    length: int = 0
    ring: list = field(default_factory=list)          # per layer [RING, head_dim] bf16 (fp8-rounded values)
    comp: dict = field(default_factory=dict)          # kv-source layer -> [cap // ratio, head_dim] bf16 (fp4-rounded)
    index_k: dict = field(default_factory=dict)       # kv-source layer -> [cap // ratio, idx_dim] bf16 (fp4-rounded)
    comp_state: dict = field(default_factory=dict)    # ratio>1 kv-source layer -> (kv [ratio, D] f32, score, count)
    tokens: torch.Tensor | None = None                # [cap] int64 token ids (Engram lookback)
    ring_size: int = 0


class Engram:
    """Hash rows of the original FP8 tables, read by offset from local NVMe; this rank's hash columns only."""

    def __init__(self, engram_dir: str, cfg: Cfg, token_map: list[int], rank: int, world: int):
        import json
        import struct
        from pathlib import Path

        self.cfg = cfg
        self.hasher = EngramHasher(cfg, token_map)
        n_cols = (cfg.engram_ngram - 1) * cfg.engram_heads
        self.cols = (rank * n_cols // world, (rank + 1) * n_cols // world)
        self.maps = {}
        for path in sorted(Path(engram_dir).glob("*.safetensors")):
            with open(path, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
            for name, e in header.items():
                if name.endswith("engram.embed.weight") or name.endswith("engram.embed.scale"):
                    lo, hi = e["data_offsets"]
                    mm_ = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size + lo, shape=(hi - lo,))
                    self.maps[name] = mm_.reshape(e["shape"])

    def rows(self, layer: int, idx: torch.Tensor) -> torch.Tensor:
        """idx [n, cols] (this rank's columns) -> bf16 [n, cols * head_dim]."""

        flat = idx.reshape(-1).cpu().numpy()
        wv = np.asarray(self.maps[f"layers.{layer}.engram.embed.weight"][flat])
        sv = np.asarray(self.maps[f"layers.{layer}.engram.embed.scale"][flat])
        v = torch.from_numpy(wv).view(torch.float8_e4m3fn).cuda().to(F32)
        e = torch.from_numpy(sv).cuda().to(torch.int32) - 127
        sc = torch.ldexp(torch.ones_like(e, dtype=F32), e)
        v = (v.view(-1, v.shape[-1] // 32, 32) * sc[..., None]).view(v.shape)
        return v.to(BF16).reshape(idx.shape[0], -1)


class Model:
    def __init__(self, w, comm: Comm, engram: Engram | None = None):
        self.w, self.cfg, self.comm, self.engram = w, w.cfg, comm, engram
        c = self.cfg
        self.Hl = c.n_heads // w.world
        self.scratch: dict = {}

    # -- caches ---------------------------------------------------------------------------------------------------
    def new_cache(self, cap: int) -> SeqCache:
        c = self.cfg
        ring = c.window + RING_EXTRA
        sc = SeqCache(cap=cap, ring_size=ring)
        sc.ring = [torch.zeros((ring, c.head_dim), dtype=BF16, device="cuda") for _ in self.w.layers]
        for i in c.kv_sources:
            if i >= len(self.w.layers):
                continue
            r = c.compress_ratios[i]
            sc.comp[i] = torch.zeros((cap // r + 1, c.head_dim), dtype=BF16, device="cuda")
            if i in c.index_sources:
                sc.index_k[i] = torch.zeros((cap // r + 1, c.idx_dim), dtype=BF16, device="cuda")
            if r > 1:
                sc.comp_state[i] = [torch.zeros((r, c.head_dim), dtype=F32, device="cuda"),
                                    torch.zeros((r, c.head_dim), dtype=F32, device="cuda"), 0]
        sc.tokens = torch.zeros((cap,), dtype=torch.int64, device="cuda")
        return sc

    def _freqs(self, layer: int, n: int) -> torch.Tensor:
        c = self.cfg
        if c.compress_ratios[layer]:
            return freqs_cis(c.rope_dim, n, c.orig_len, c.compress_theta, c.rope_factor, c.beta_fast, c.beta_slow)
        return freqs_cis(c.rope_dim, n, 0, c.rope_theta, c.rope_factor, c.beta_fast, c.beta_slow)

    def _f(self, layer: int, sc: SeqCache) -> torch.Tensor:
        # one table per (rope kind, capacity), rounded up so it is built once
        cap = 1 << max(12, (sc.cap - 1).bit_length())
        return self._freqs(layer, cap)

    # -- mHC -------------------------------------------------------------------------------------------------------
    def hc_mixes(self, h: torch.Tensor, params):
        c = self.cfg
        fn, scale, base = params
        xf = h.flatten(1).to(F32)
        rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + c.eps)
        mixes = (xf @ fn.t()) * rs
        return hc_split_sinkhorn(mixes, scale, base, c.hc, c.hc_iters, c.hc_eps)

    @staticmethod
    def hc_pre(h: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return (pre[..., None] * h.to(F32)).sum(1).to(h.dtype)

    @staticmethod
    def hc_post(y: torch.Tensor, res: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        out = post[..., None] * y.to(F32)[:, None, :] + (comb[..., None] * res.to(F32)[:, :, None, :]).sum(1)
        return out.to(y.dtype)

    # -- Engram ----------------------------------------------------------------------------------------------------
    def engram_apply(self, lay, h: torch.Tensor, hashes: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        lo, hi = self.engram.cols
        e = self.engram.rows(lay.idx, hashes[:, lo:hi])
        kv = self.comm.sum(mm(lay.engram_wkv, e, F32)).to(BF16)
        key, value = kv.split([c.hc * c.dim, c.dim], dim=-1)
        key = key.to(F32).unflatten(-1, (c.hc, c.dim))
        hf = h.to(F32)
        rstd = torch.rsqrt(hf.square().mean(-1) + c.eps) * torch.rsqrt(key.square().mean(-1) + c.eps)
        dot = (hf * lay.engram_qk * key).sum(-1) * rstd * c.dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (hf + gate[..., None] * value.to(F32)[:, None, :]).to(h.dtype)

    # -- attention -------------------------------------------------------------------------------------------------
    def _compress(self, lay, x: torch.Tensor, sc: SeqCache, start: int):
        """New compressed latents (pre-RoPE, bf16) from rows at start.., with their group indices."""

        c = self.cfg
        r = lay.ratio
        if r == 1:
            lat = rms_norm(mm(lay.comp_wkv, x, BF16), lay.comp_norm, c.eps)
            return lat, torch.arange(start, start + x.shape[0], device=x.device)
        kv = mm(lay.comp_wkv, x, F32)
        score = mm(lay.comp_wgate, x, F32)
        st = sc.comp_state[lay.idx]
        pend = st[2]
        assert pend == start % r, (pend, start, r)
        kv = torch.cat([st[0][:pend], kv], 0)
        score = torch.cat([st[1][:pend], score], 0)
        first = start - pend
        full = kv.shape[0] // r
        rem = kv.shape[0] - full * r
        st[0][:rem] = kv[full * r:]
        st[1][:rem] = score[full * r:]
        st[2] = rem
        if full == 0:
            return None, None
        kvg = kv[:full * r].unflatten(0, (full, r))
        sg = score[:full * r].unflatten(0, (full, r))
        lat = rms_norm((kvg * sg.softmax(dim=1)).sum(1).to(BF16), lay.comp_norm, c.eps)
        groups = torch.arange(first // r, first // r + full, device=x.device)
        return lat, groups

    def attention(self, lay, x: torch.Tensor, sc: SeqCache, start: int, shared: dict) -> torch.Tensor:
        c = self.cfg
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        ratio = lay.ratio
        f = self._f(lay.idx, sc)
        pos = torch.arange(start, start + n, device=x.device)
        qr = rms_norm(mm(lay.wq_a, x), lay.q_norm, c.eps)
        q = mm(lay.wq_b, qr).view(n, self.Hl, hd)
        rope_(q[..., -rd:], f[start:start + n])
        kv = rms_norm(mm(lay.wkv, x), lay.kv_norm, c.eps)
        rope_(kv[..., -rd:], f[start:start + n])
        if KV_QUANT:
            kv = fp8_qd(kv, 32)
        # window keys: the ring's last (window - 1) positions before start, then this block's rows
        ring = sc.ring[lay.idx]
        R = sc.ring_size
        lo = max(0, start - (c.window - 1))
        prev = torch.arange(lo, start, device=x.device)
        keys_w = torch.cat([ring[prev % R], kv], 0)                      # positions lo .. start + n - 1
        kpos = torch.cat([prev, pos])
        wpos = (pos[:, None] - c.window + 1).clamp_min(0) + torch.arange(c.window, device=x.device)[None]
        widx = torch.where(wpos <= pos[:, None], wpos - lo, -1)
        widx = torch.where(wpos < lo, -1, widx)
        keys, idx = keys_w, widx
        if ratio:
            if lay.comp_wkv is not None:
                lat, groups = self._compress(lay, x, sc, start)
                shared["kv_layer"] = lay.idx
                if lat is not None and lay.idx_wk is not None:
                    k = rms_norm(mm(lay.idx_wk, lat), lay.idx_k_norm, c.eps)
                    rope_(k[..., -rd:], f[groups * ratio])
                    if KV_QUANT:
                        k = fp4_qd(k, 32, e4m3_scale=False)
                    sc.index_k[lay.idx][groups] = k
                if lat is not None:
                    lat = lat.clone()
                    rope_(lat[..., -rd:], f[groups * ratio])
                    if KV_QUANT:
                        lat = fp4_qd(lat, 16, e4m3_scale=True)
                    sc.comp[lay.idx][groups] = lat
            src = shared["kv_layer"]
            n_comp_end = (start + n) // ratio
            vis = ((pos + 1) // ratio)[:, None]                          # compressed entries row i may see
            if lay.idx_wq_b is not None:
                if n_comp_end == 0:
                    cidx = torch.full((n, 0), -1, dtype=torch.long, device=x.device)
                else:
                    iq = mm(lay.idx_wq_b, qr).view(n, c.idx_heads, c.idx_dim)
                    rope_(iq[..., -rd:], f[start:start + n])
                    if KV_QUANT:
                        iq = fp4_qd(iq, 32, e4m3_scale=False)
                    wts = (x.to(F32) @ lay.idx_proj.t()).to(BF16) * (c.idx_dim ** -0.5 * c.idx_heads ** -0.5)
                    ik = sc.index_k[src][:n_comp_end].to(F32)
                    score = torch.zeros((n, n_comp_end), dtype=F32, device=x.device)
                    iqf, wf = iq.to(F32), wts.to(F32)
                    for h in range(c.idx_heads):
                        score += (iqf[:, h] @ ik.t()).relu_() * wf[:, h:h + 1]
                    tpos = torch.arange(n_comp_end, device=x.device)[None]
                    score.masked_fill_(tpos >= vis, float("-inf"))
                    if lay.idx == c.cand_source:
                        shared["cand"] = _candidates(score, vis, c.cand_blocks, c.cand_block)
                    elif 0 <= c.cand_source < lay.idx:
                        score.masked_fill_(~shared["cand"], float("-inf"))
                    kk = min(c.idx_topk, n_comp_end)
                    top = score.topk(kk, dim=-1, sorted=False).indices.sort(dim=-1).values
                    cidx = torch.where(top < vis, top, -1)
                shared["topk"] = cidx
            cidx = shared["topk"]
            ckv = sc.comp[src][:max(n_comp_end, 1)]
            keys = torch.cat([keys_w, ckv], 0)
            off = keys_w.shape[0]
            idx = torch.cat([widx, torch.where(cidx >= 0, cidx + off, -1)], -1)
        o = sparse_attn(q, keys, lay.sink, idx, hd ** -0.5)
        rope_(o[..., -rd:], f[start:start + n], inverse=True)
        # write this block's window keys into the ring (positions start .. start+n-1)
        keep = min(n, R)
        ring[pos[-keep:] % R] = kv[-keep:]
        og = o.view(n, len(lay.wo_a), -1)
        u = torch.cat([mm(wo, og[:, g].contiguous()) for g, wo in enumerate(lay.wo_a)], -1)
        return mm(lay.wo_b, u, F32)                                      # this rank's partial

    # -- MoE -------------------------------------------------------------------------------------------------------
    def moe(self, lay, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        n = x.shape[0]
        scores = F.softplus(x.to(F32) @ lay.gate_w.t()).sqrt()
        ind = (scores + lay.gate_b).topk(c.topk, dim=-1).indices
        wts = scores.gather(1, ind)
        if c.topk > 1:
            wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
        wts = wts * c.route_scale
        shared_id = lay.experts.count - 1
        pick = torch.cat([ind, torch.full((n, 1), shared_id, dtype=ind.dtype, device=x.device)], 1).to(torch.int32)
        wts = torch.cat([wts, torch.ones((n, 1), dtype=F32, device=x.device)], 1).contiguous()
        slots = c.topk + 1
        s = self.scratch.get("moe")
        if s is None or s.rows < n:
            self.scratch.pop("moe", None)
            s = exl3_experts.Scratch(lay.experts, rows=max(n, 8), slots=slots)
            self.scratch["moe"] = s
        out = exl3_experts.routed(x.contiguous(), pick.contiguous(), wts, lay.experts, s, None, n,
                                  limit=c.swiglu_limit, act_mode=exl3_experts.ACT_F32)
        return out                                                        # fp32 partial [n, d]

    # -- one block of rows ----------------------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, sc: SeqCache, ids: torch.Tensor, start: int, all_logits: bool = False,
                taps: list | None = None) -> torch.Tensor:
        """Rows ids [n] at positions start.. of one sequence -> fp32 logits [n or 1, V] (both ranks the same)."""

        c, w = self.cfg, self.w
        n = ids.shape[0]
        assert start == sc.length, (start, sc.length)
        sc.tokens[start:start + n] = ids
        h = w.embed[ids].to(BF16)[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        hashes = None
        if self.engram is not None:
            lb = c.engram_ngram - 1
            lo = max(0, start - lb)
            hashes = self.engram.hasher(sc.tokens[lo:start + n])[start - lo:]       # [n, L, cols]
        shared: dict = {}
        for lay in w.layers:
            if hashes is not None and lay.engram_wkv is not None:
                h = self.engram_apply(lay, h, hashes[:, c.engram_layers.index(lay.idx)])
            if taps is not None and lay.idx in c.dspark_taps:
                taps.append(h.to(F32).mean(1).to(BF16))
            res = h
            a_pre, a_post, a_comb = self.hc_mixes(h, lay.hc_attn)
            x = rms_norm(self.hc_pre(h, pre), lay.attn_norm, c.eps)
            y = self.comm.sum(self.attention(lay, x, sc, start, shared)).to(BF16)
            h = self.hc_post(y, res, a_post, a_comb)
            res = h
            f_pre, f_post, f_comb = self.hc_mixes(h, lay.hc_ffn)
            x = rms_norm(self.hc_pre(h, a_pre), lay.ffn_norm, c.eps)
            y = self.comm.sum(self.moe(lay, x)).to(BF16)
            h = self.hc_post(y, res, f_post, f_comb)
            pre = f_pre
        sc.length = start + n
        if not all_logits:
            h, pre = h[-1:], pre[-1:]
        x = rms_norm(self.hc_pre(h, pre), w.norm, c.eps)
        local = mm(w.head, x, F32)                                        # [n, V / world]
        g = self.comm.gather(local)                                       # [world, n, V / world]
        return g.permute(1, 0, 2).reshape(local.shape[0], -1)


def _candidates(score: torch.Tensor, vis: torch.Tensor, nblocks: int, bsize: int) -> torch.Tensor:
    width = score.shape[-1]
    s = F.pad(score, (0, -width % bsize), value=float("-inf")).unflatten(-1, (-1, bsize)).amax(-1)
    nb = s.shape[-1]
    last = (vis - 1) // bsize
    s = s.masked_fill(torch.arange(nb, device=score.device)[None] == last, float("inf"))
    top = s.topk(min(nblocks, nb), dim=-1)
    keep = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(bsize, dim=-1)[..., :width]
