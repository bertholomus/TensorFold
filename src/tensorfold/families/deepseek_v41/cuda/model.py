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
from . import kernels as K
from ..ops import (BF16, F32, EngramHasher, fp4_qd, fp8_qd, freqs_cis, hc_split_sinkhorn, rms_norm, rope_,
                   sparse_attn)

RAW = 64                            # per-position compressor inputs kept (a verify window rolls back by length alone)
RING_EXTRA = 16                     # window ring slots beyond the 128-token window (verify windows never clobber)
KV_QUANT = os.environ.get("TF_DS_KV", "native") != "bf16"
KERNELS = os.environ.get("TF_DS_KERNELS", "1") != "0"      # 0: the plain-torch phase-1 path (A/B and debugging)
TIMING = os.environ.get("TF_DS_TIMING", "0") == "1"        # per-section wall times (synchronizing; profiling only)
TIMES: dict = {}


class _T:
    """with _T("name"): ... adds the section's synchronized wall time to TIMES (no-op unless TF_DS_TIMING=1)."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if TIMING:
            import time
            torch.cuda.synchronize()
            self.t = time.perf_counter()

    def __exit__(self, *a):
        if TIMING:
            import time
            torch.cuda.synchronize()
            TIMES[self.name] = TIMES.get(self.name, 0.0) + time.perf_counter() - self.t


def store_rows(cache, rows: torch.Tensor, x: torch.Tensor, block: int, e4m3: bool) -> None:
    """Rows of a compressed cache: packed FP4 (a (codes, scales) pair) or bf16 (fp4-rounded unless TF_DS_KV=bf16)."""

    from ..ops import fp4_pack

    if isinstance(cache, tuple):
        codes, scales = fp4_pack(x, block, e4m3)
        cache[0][rows] = codes
        cache[1][rows] = scales
    else:
        cache[rows] = fp4_qd(x, block, e4m3_scale=e4m3) if KV_QUANT else x


class Comm:
    """All-gather of fp32 partials then a rank-order sum (identity on one rank). Decode-sized fp32 payloads go over
    our RDMA-write gather (tensorfold.cuda.rdma) when every rank opens it, the rest over NCCL."""

    def __init__(self, nccl, world: int, rdma_bytes: int = 0):
        self.nccl, self.world = nccl, world
        self._buf: dict = {}
        if world > 1 and rdma_bytes and os.environ.get("TF_DS_RDMA", "1") != "0":
            from tensorfold.cuda.rdma import Hybrid, RdmaGather

            try:
                self.nccl = Hybrid(nccl, RdmaGather(nccl.store, nccl.rank, world, max_bytes=rdma_bytes))
                if nccl.rank == 0:
                    print("[tensorfold] decode gathers over RDMA writes", flush=True)
            except Exception as exc:          # noqa: BLE001  (every rank sees the same refusal)
                print(f"[tensorfold] gathers stay on NCCL: {exc}", flush=True)

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


EXACT_MM = os.environ.get("TF_DS_EXACT_MM", "0") == "1"     # test mode: fp32 dequantized weights, fp32 matmuls


def _had(device) -> torch.Tensor:
    h = getattr(_had, "h", None)
    if h is None:
        i = torch.arange(128)
        b = i[:, None] & i[None, :]
        par = torch.zeros_like(b)
        for k in range(7):
            par ^= (b >> k) & 1
        h = _had.h = ((1 - 2 * par).float() / math.sqrt(128)).to(device)
    return h


def dequant_fp32(words_or_trellis, suh, svh, k: int, n: int, layer=None) -> torch.Tensor:
    """W [K, N] fp32 = diag(suh) H W_q H diag(svh) (the reference's dequantization)."""

    wq = layer.unpack().float() if layer is not None else words_or_trellis
    h = _had(wq.device)
    w = torch.einsum("bin,ij->bjn", wq.view(k // 128, 128, n), h).reshape(k, n) * suh.float()[:, None]
    return torch.einsum("kci,ij->kcj", w.view(k, n // 128, 128), h).reshape(k, n) * svh.float()[None, :]


def mm(layer, x: torch.Tensor, out_dtype=BF16, ws: exl3_prefill.Workspace | None = None) -> torch.Tensor:
    """x [M, K] @ W: the row-invariant EXL3 linear up to 128 rows, the prompt GEMM beyond."""

    m = x.shape[0]
    if EXACT_MM:
        w = dequant_fp32(None, layer.suh, layer.svh, layer.k, layer.n, layer=layer)
        return (x.float() @ w).to(out_dtype)
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
    comp_raw: dict = field(default_factory=dict)      # ratio>1 kv-source layer -> (kv, score) [RAW, D] f32 by position
    tokens: torch.Tensor | None = None                # [cap] int64 token ids
    host: list = field(default_factory=list)          # the same ids on the host (Engram hashes)
    ring_size: int = 0


@dataclass
class PoolCache:
    """A window of ``cap`` positions shared by up to ``slots`` streams (see ``Model.new_pool``)."""

    slots: int
    cap: int
    ring_size: int
    ring: list = field(default_factory=list)          # per layer [slots * ring_size, head_dim]
    comp: dict = field(default_factory=dict)          # one sequence's planes, by position
    index_k: dict = field(default_factory=dict)
    comp_raw: dict = field(default_factory=dict)      # per kv source [slots * RAW, head_dim] pairs
    tokens: torch.Tensor | None = None


def _engram_io():
    from pathlib import Path

    from tensorfold.cuda.build import load

    if not hasattr(_engram_io, "ext"):
        here = Path(__file__).parent
        from torch.utils import cpp_extension

        _engram_io.ext = cpp_extension.load(name="tf_ds_engram_io_v2", sources=[str(here / "engram_io.cpp")],
                                            extra_cflags=["-O3"], verbose=False)
    return _engram_io.ext


class Engram:
    """Hash rows of the original FP8 tables, read by offset from local NVMe; this rank's hash columns only.

    Hashes are computed on the host from the token ids (numpy, DeepSeek's rule), rows are read with parallel preads
    into a pinned buffer and copied to the GPU without a device sync.
    """

    def __init__(self, engram_dir: str, cfg: Cfg, token_map: list[int], rank: int, world: int):
        import json
        import struct
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        from ..ops import engram_multipliers, engram_primes

        self.cfg = cfg
        self.hasher = EngramHasher(cfg, token_map)
        n_cols = (cfg.engram_ngram - 1) * cfg.engram_heads
        self.cols = (rank * n_cols // world, (rank + 1) * n_cols // world)
        primes = engram_primes(cfg)
        flat = [[p for per in layer for p in per] for layer in primes]
        self.np_primes = np.array(flat, dtype=np.int64)                         # [L, cols]
        self.np_offsets = np.array([np.cumsum([0, *f[:-1]]) for f in flat], dtype=np.int64)
        self.np_mult = engram_multipliers(cfg).numpy().astype(np.int64)           # [L, ngram]
        self.np_map = np.asarray(token_map, dtype=np.int64)
        self.pad = int(token_map[cfg.engram_pad])
        self.files: dict = {}
        for path in sorted(Path(engram_dir).glob("*.safetensors")):
            with open(path, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
            fd = os.open(str(path), os.O_RDONLY)
            for name, e in header.items():
                if name.endswith("engram.embed.weight") or name.endswith("engram.embed.scale"):
                    lo, hi = e["data_offsets"]
                    self.files[name] = (fd, 8 + size + lo, int(np.prod(e["shape"][1:])))
        self.io = _engram_io()
        self.threads = int(os.environ.get("TF_DS_ENGRAM_THREADS") or 48)
        self.pinned: dict = {}

    def hashes(self, tokens: list[int], start: int, n: int) -> np.ndarray:
        """Row ids [n, L, cols] for positions start .. start + n - 1 of ``tokens`` (the whole sequence so far)."""

        c = self.cfg
        lb = c.engram_ngram - 1
        lo = max(0, start - lb)
        raw = np.asarray(tokens[lo:start + n], dtype=np.int64)
        # an image span's positions (negative in the host list) are DEAD: an n-gram never reaches into or past one
        dead = raw < 0
        comp = np.where(dead, -1, self.np_map[np.where(dead, 0, raw)])
        pos = np.arange(start, start + n)
        toks = []
        blocked = np.zeros(n, dtype=bool)
        for shift in range(c.engram_ngram):
            src = comp[np.clip(pos - shift - lo, 0, None)]
            blocked |= (pos < shift) | (src < 0)
            toks.append(np.where(blocked, self.pad, src))
        toks = np.stack(toks, -1)                                              # [n, ngram]
        prod = toks[:, None, :] * self.np_mult[None]                           # [n, L, ngram]
        rolling, out = prod[..., 0], []
        H = c.engram_heads
        for i in range(1, c.engram_ngram):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            out.append(rolling[..., None] % self.np_primes[None, :, (i - 1) * H:i * H])
        return np.concatenate(out, -1) + self.np_offsets[None]

    def rows(self, layer: int, idx: np.ndarray) -> torch.Tensor:
        """idx [n, cols] (this rank's columns, host) -> bf16 [n, cols * head_dim] on the GPU, no device sync."""

        fw, bw, rw = self.files[f"layers.{layer}.engram.embed.weight"]
        fs, bs, rs = self.files[f"layers.{layer}.engram.embed.scale"]
        flat = np.ascontiguousarray(idx.reshape(-1), dtype=np.int64)
        m = flat.shape[0]
        # one pinned staging pair a layer, grown (never shrunk); the copy out of it finishes before it is refilled
        buf = self.pinned.get(layer)
        if buf is not None:
            buf[2].synchronize()
        if buf is None or buf[0].shape[0] < m:
            cap = max(m, 64)
            buf = (torch.empty((cap, rw), dtype=torch.uint8, pin_memory=True),
                   torch.empty((cap, rs), dtype=torch.uint8, pin_memory=True), torch.cuda.Event())
            self.pinned[layer] = buf
        bw_v, bs_v = buf[0][:m], buf[1][:m]
        it = torch.from_numpy(flat)
        self.io.gather_rows2(fw, bw, rw, fs, bs, rs, it, bw_v, bs_v, self.threads)
        gw = bw_v.cuda(non_blocking=True)
        gs = bs_v.cuda(non_blocking=True)
        buf[2].record()
        v = gw.view(torch.float8_e4m3fn).to(F32)
        e = gs.to(torch.int32) - 127
        sc = torch.ldexp(torch.ones_like(e, dtype=F32), e)
        v = (v.view(m, rw // 32, 32) * sc[..., None]).view(m, rw)
        return v.to(BF16).reshape(idx.shape[0], -1)


class Model:
    def __init__(self, w, comm: Comm, engram: Engram | None = None):
        self.w, self.cfg, self.comm, self.engram = w, w.cfg, comm, engram
        c = self.cfg
        self.Hl = c.n_heads // w.world
        self.scratch: dict = {}
        self._zero = torch.zeros((1,), dtype=torch.int64, device="cuda")
        if not KERNELS and KV_QUANT:
            raise ValueError("TF_DS_KERNELS=0 (the torch path) reads bf16 caches only: also set TF_DS_KV=bf16")
        for lay in w.layers:
            if lay.idx_proj is not None:
                lay.idx_proj_h = lay.idx_proj.to(torch.float16).contiguous()

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
            rows = cap // r + 2                          # + a scratch row (graph-captured windows' incomplete groups)
            if KV_QUANT:                                 # packed FP4 (DeepSeek's cache formats), 288 + 68 B a row
                sc.comp[i] = (torch.zeros((rows, c.head_dim // 2), dtype=torch.uint8, device="cuda"),
                              torch.zeros((rows, c.head_dim // 16), dtype=torch.uint8, device="cuda"))
            else:
                sc.comp[i] = torch.zeros((rows, c.head_dim), dtype=BF16, device="cuda")
            if i in c.index_sources:
                if KV_QUANT:
                    sc.index_k[i] = (torch.zeros((rows, c.idx_dim // 2), dtype=torch.uint8, device="cuda"),
                                     torch.full((rows, c.idx_dim // 32), 127, dtype=torch.uint8, device="cuda"))
                else:
                    sc.index_k[i] = torch.zeros((rows, c.idx_dim), dtype=BF16, device="cuda")
            if r > 1:
                sc.comp_raw[i] = (torch.zeros((RAW, c.head_dim), dtype=F32, device="cuda"),
                                  torch.zeros((RAW, c.head_dim), dtype=F32, device="cuda"))
        sc.tokens = torch.zeros((cap,), dtype=torch.int64, device="cuda")
        return sc

    def new_pool(self, slots: int, cap: int) -> "PoolCache":
        """One window of ``cap`` positions that up to ``slots`` streams share by extents: the compressed and indexer
        planes are one sequence's (``new_cache(cap)``), a stream owning the rows of its extent [base, base + size) of
        positions (its last row its scratch row); each slot has its own window ring and compressor inputs.
        ``view(slot, base, size)`` is that stream's SeqCache over its rows, so the single-stream paths (prompt chunks,
        the drafter) run on it unchanged; a concurrent round addresses every stream's rows through per-row bases
        (rounds.py)."""

        one = self.new_cache(cap)
        pool = PoolCache(slots=slots, cap=cap, ring_size=one.ring_size)

        def plane(t: torch.Tensor) -> torch.Tensor:
            out = torch.empty((slots * t.shape[0], *t.shape[1:]), dtype=t.dtype, device=t.device)
            out.view(slots, *t.shape).copy_(t.expand(slots, *t.shape))
            return out

        pool.ring = [plane(t) for t in one.ring]
        pool.comp_raw = {i: tuple(plane(t) for t in x) for i, x in one.comp_raw.items()}
        pool.comp, pool.index_k, pool.tokens = one.comp, one.index_k, one.tokens
        return pool

    def pool_view(self, pool: "PoolCache", slot: int, base: int, size: int) -> SeqCache:
        """Slot ``slot``'s stream over positions [base, base + size) of the pool (both multiples of every compress
        ratio): its compressed rows [base / r, (base + size) / r), the last one its scratch row."""

        c = self.cfg

        def rows(x, lo: int, hi: int):
            return tuple(t[lo:hi] for t in x) if isinstance(x, tuple) else x[lo:hi]

        v = SeqCache(cap=size, ring_size=pool.ring_size)
        v.ring = [t[slot * pool.ring_size:(slot + 1) * pool.ring_size] for t in pool.ring]
        v.comp = {i: rows(x, base // c.compress_ratios[i], (base + size) // c.compress_ratios[i])
                  for i, x in pool.comp.items()}
        v.index_k = {i: rows(x, base // c.compress_ratios[i], (base + size) // c.compress_ratios[i])
                     for i, x in pool.index_k.items()}
        v.comp_raw = {i: rows(x, slot * RAW, (slot + 1) * RAW) for i, x in pool.comp_raw.items()}
        v.tokens = pool.tokens[base:base + size]
        return v

    def _freqs(self, layer: int, n: int) -> torch.Tensor:
        c = self.cfg
        if c.compress_ratios[layer]:
            return freqs_cis(c.rope_dim, n, c.orig_len, c.compress_theta, c.rope_factor, c.beta_fast, c.beta_slow)
        return freqs_cis(c.rope_dim, n, 0, c.rope_theta, c.rope_factor, c.beta_fast, c.beta_slow)

    def _cs(self, layer: int, sc: SeqCache):
        """(cos, sin) fp32 [cap, rope_dim / 2] for this layer's rope kind."""

        f = self._f(layer, sc)
        key = (self.cfg.compress_ratios[layer] > 0, f.shape[0])
        t = self.scratch.get(("cs", key))
        if t is None:
            t = (f.real.contiguous().float(), f.imag.contiguous().float())
            self.scratch[("cs", key)] = t
        return t

    def _f(self, layer: int, sc: SeqCache) -> torch.Tensor:
        # one table per rope kind, for the engine's whole window (``rope_cap``) whatever the cache: a table built at
        # another length can differ in a row's last bits (the CPU's vectorized sin/cos), and a stream's prompt and its
        # rounds must read the same rows
        cap = 1 << max(12, (max(sc.cap, getattr(self, "rope_cap", 0)) - 1).bit_length())
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
    def engram_apply(self, lay, h: torch.Tensor, hashes: torch.Tensor, img: torch.Tensor | None = None) -> torch.Tensor:
        if img is not None:
            out = self.engram_apply(lay, h, hashes)
            out[img] = h[img]                 # Engram's gate is shut inside an image span: those rows pass unchanged
            return out
        c = self.cfg
        lo, hi = self.engram.cols
        with _T("engram_rows"):
            e = self.engram.rows(lay.idx, hashes[:, lo:hi])
        kv = self.comm.sum(mm(lay.engram_wkv, e, F32)).to(BF16)
        if KERNELS:
            return K.engram_gate(h, kv.contiguous(), lay.engram_qk, c.eps)
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
        rk, rs = sc.comp_raw[lay.idx]
        pend = start % r
        if pend:                                   # the group's earlier rows come from the positional store
            prev = torch.arange(start - pend, start, device=x.device) % RAW
            kv = torch.cat([rk[prev], kv], 0)
            score = torch.cat([rs[prev], score], 0)
        keep = min(x.shape[0], RAW)
        pw = torch.arange(start + x.shape[0] - keep, start + x.shape[0], device=x.device) % RAW
        rk[pw] = kv[-keep:]
        rs[pw] = score[-keep:]
        first = start - pend
        full = kv.shape[0] // r
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
                        apply_candidates(score, shared["cand"], c.cand_block)
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

    def kv_source_update(self, lay, x: torch.Tensor, sc: SeqCache, start: int, shared: dict) -> None:
        """A kv-source layer's compressed latents (and index keys) for rows at start.., into the caches."""

        c = self.cfg
        rd, ratio = c.rope_dim, lay.ratio
        lat, groups = self._compress(lay, x, sc, start)
        shared["kv_layer"] = lay.idx
        if lat is not None and lay.idx_wk is not None:
            k = rms_norm(mm(lay.idx_wk, lat), lay.idx_k_norm, c.eps)
            rope_(k[..., -rd:], self._f(lay.idx, sc)[groups * ratio])
            store_rows(sc.index_k[lay.idx], groups, k, 32, False)
        if lat is not None:
            lat = lat.clone()
            rope_(lat[..., -rd:], self._f(lay.idx, sc)[groups * ratio])
            store_rows(sc.comp[lay.idx], groups, lat, 16, True)

    def attention_k(self, lay, x: torch.Tensor, sc: SeqCache, start: int, shared: dict, pos: torch.Tensor,
                    floor: int = 0, kv_done: bool = False):
        """The same math as ``attention`` with the row-independent Triton kernels. ``floor``: no window key before
        this position (bounded replay); ``kv_done``: the kv-source update already ran for these rows."""

        c = self.cfg
        n = x.shape[0]
        rd, hd = c.rope_dim, c.head_dim
        ratio = lay.ratio
        cos, sin = self._cs(lay.idx, sc)
        qr = K.rmsnorm(mm(lay.wq_a, x), lay.q_norm, c.eps)
        q = mm(lay.wq_b, qr).view(n, self.Hl, hd)
        K.rope_heads(q, cos, sin, pos, rd)
        y = mm(lay.wkv, x)
        ring = sc.ring[lay.idx]
        R = sc.ring_size
        ring_mode = n <= RING_EXTRA
        if ring_mode:
            kv = K.kv_norm_rope(y, lay.kv_norm, cos, sin, pos, ring, pos % R, c.eps, KV_QUANT, rd)
            wsrc, wlo = ring, self._zero
        else:
            lo = max(floor, start - (c.window - 1))
            wsrc = torch.empty((start - lo + n, hd), dtype=BF16, device=x.device)
            if start > lo:
                wsrc[:start - lo] = ring[torch.arange(lo, start, device=x.device) % R]
            K.kv_norm_rope(y, lay.kv_norm, cos, sin, pos, ring, self._neg(n), c.eps, KV_QUANT, rd,
                           out=wsrc[start - lo:])
            kv = wsrc[start - lo:]
            wlo = torch.tensor([lo], dtype=torch.int64, device=x.device)
        comp, cidx = None, None
        if ratio:
            if lay.comp_wkv is not None and not kv_done:
                self.kv_source_update(lay, x, sc, start, shared)
            src = shared["kv_layer"]
            n_comp_end = (start + n) // ratio
            vis = (pos + 1) // ratio
            if lay.idx_wq_b is not None:
                if n_comp_end == 0:
                    cidx = torch.full((n, 0), -1, dtype=torch.int64, device=x.device)
                else:
                    iq = mm(lay.idx_wq_b, qr).view(n, c.idx_heads, c.idx_dim)
                    K.rope_heads(iq, cos, sin, pos, rd)
                    if KV_QUANT:
                        iq = fp4_qd(iq, 32, e4m3_scale=False)
                    wl = K.rowmm(x, lay.idx_proj_h) if n <= 16 else x.float() @ lay.idx_proj.t()
                    wts = (wl.to(BF16) * (c.idx_dim ** -0.5 * c.idx_heads ** -0.5))
                    kk = min(c.idx_topk, n_comp_end)
                    cidx = torch.empty((n, kk), dtype=torch.int64, device=x.device)
                    # rows in blocks so the [rows, n_comp] score matrix stays bounded at long contexts
                    rb = max(16, min(n, (1 << 26) // (4 * max(n_comp_end, 1))))
                    cand_parts = []
                    for r0 in range(0, n, rb):
                        r1 = min(n, r0 + rb)
                        score = K.index_score(iq[r0:r1].contiguous(), sc.index_k[src], wts[r0:r1].contiguous(),
                                              vis[r0:r1].contiguous(), n_comp_end)
                        if lay.idx == c.cand_source:
                            cand_parts.append(_candidates(score, vis[r0:r1, None], c.cand_blocks, c.cand_block))
                        elif 0 <= c.cand_source < lay.idx:
                            apply_candidates(score, shared["cand"][r0:r1], c.cand_block)
                        top = K.topk_indices(score, kk)
                        cidx[r0:r1] = torch.where(top < vis[r0:r1, None], top, -1)
                        del score
                    if lay.idx == c.cand_source:
                        shared["cand"] = torch.cat(cand_parts, 0)
                shared["topk"] = cidx
            cidx = shared["topk"]
            comp = sc.comp[src]
        o = K.sparse_attn(q, lay.sink, wsrc, wlo, ring_mode, comp, cidx, pos, hd ** -0.5, c.window)
        K.rope_heads(o, cos, sin, pos, rd, inverse=True)
        if not ring_mode:
            keep = min(n, R)
            ring[pos[-keep:] % R] = kv[-keep:]
        og = o.view(n, len(lay.wo_a), -1)
        u = torch.cat([mm(wo, og[:, g].contiguous()) for g, wo in enumerate(lay.wo_a)], -1)
        return mm(lay.wo_b, u, F32)

    def _neg(self, n: int) -> torch.Tensor:
        t = self.scratch.get(("neg", n))
        if t is None:
            t = torch.full((n,), -1, dtype=torch.int64, device="cuda")
            self.scratch[("neg", n)] = t
        return t

    # -- MoE -------------------------------------------------------------------------------------------------------
    def moe(self, lay, x: torch.Tensor, topk: int | None = None, img: torch.Tensor | None = None) -> torch.Tensor:
        c = self.cfg
        n = x.shape[0]
        topk = topk or c.topk
        slots = topk + 1
        shared_id = lay.experts.count - 1
        vl = getattr(lay, "gate_b_vl", None)
        if img is not None and vl is None:
            vl = lay.gate_b                   # no VL bias in this pack: image rows route as text (warned at load)
        if KERNELS:
            # decode / verify windows: the row-invariant matmul; prompt chunks: one cuBLAS GEMM
            logits = K.rowmm(x, lay.gate_w) if n <= 16 else (x.float() @ lay.gate_w.float().t())
            pick = torch.empty((n, slots), dtype=torch.int32, device=x.device)
            wts = torch.empty((n, slots), dtype=F32, device=x.device)
            K.route(logits, lay.gate_b, topk, c.route_scale, shared_id, pick, wts)
            if img is not None:
                # inside an image span the gate picks with its VL bias (a row a program: the other rows are unchanged)
                ip = torch.empty((img.numel(), slots), dtype=torch.int32, device=x.device)
                iw = torch.empty((img.numel(), slots), dtype=F32, device=x.device)
                K.route(logits[img].contiguous(), vl, topk, c.route_scale, shared_id, ip, iw)
                pick[img], wts[img] = ip, iw
        else:
            scores = F.softplus(x.to(F32) @ lay.gate_w.float().t()).sqrt()
            bias = lay.gate_b if img is None else lay.gate_b.expand(n, -1).clone().index_copy_(
                0, img, vl.expand(img.numel(), -1))
            ind = (scores + bias).topk(topk, dim=-1).indices
            wts = scores.gather(1, ind)
            if topk > 1:
                wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
            wts = wts * c.route_scale
            pick = torch.cat([ind, torch.full((n, 1), shared_id, dtype=ind.dtype, device=x.device)], 1).to(torch.int32)
            wts = torch.cat([wts, torch.ones((n, 1), dtype=F32, device=x.device)], 1).contiguous()
        prompt = n >= 64
        skey = ("moe", slots, lay.experts.count, prompt)
        s = self.scratch.get(skey)
        if s is None or s.rows < n:
            # decode / verify windows (< 64 rows) take one scratch of 64 rows made once and never replaced: CUDA graphs
            # captured with it keep writing into its memory, so freeing it for a bigger one would corrupt whatever
            # reused that memory (an illegal access on the next replay). Prompt chunks (eager only) may grow theirs.
            assert prompt or s is None, "decode scratch must not be replaced"
            self.scratch.pop(skey, None)
            s = exl3_experts.Scratch(lay.experts, rows=max(n, 64) if not prompt else n, slots=slots, prompt=prompt)
            self.scratch[skey] = s
        if EXACT_MM:
            return self._moe_exact(lay, x, pick, wts)
        out = exl3_experts.routed(x.contiguous(), pick.contiguous(), wts, lay.experts, s, None, n,
                                  limit=c.swiglu_limit, act_mode=exl3_experts.ACT_F32)
        return out                                                        # fp32 partial [n, d]

    def _moe_exact(self, lay, x, pick, wts):
        """Test mode: each picked expert dequantized to fp32 (the reference's arithmetic, split by rank)."""

        c, ex = self.cfg, lay.experts
        D, I = ex.dims, ex.width
        y = torch.zeros((x.shape[0], D), dtype=F32, device=x.device)

        def mat(ptrs, k2s, suh, svh, e, k, n):
            words = getattr(ex, "_trellis_views", None)
            t = ex.keep[{"g": 0, "u": 1, "d": 2}[ptrs] * ex.count + e]
            wq = exl3_experts.dequant(t, ex.cb).float()
            return dequant_fp32(wq, suh[e], svh[e], k, n)

        for e in torch.unique(pick).tolist():
            rows, slot = torch.where(pick == e)
            xs = x[rows]
            g = (xs.float() @ mat("g", None, ex.suh_g, ex.svh_g, e, D, I)).to(BF16).float()
            u = (xs.float() @ mat("u", None, ex.suh_u, ex.svh_u, e, D, I)).to(BF16).float()
            u = u.clamp(-c.swiglu_limit, c.swiglu_limit)
            g = g.clamp(max=c.swiglu_limit)
            hh = F.silu(g) * u * wts[rows, slot, None]
            y[rows] += (hh.to(BF16).float() @ mat("d", None, ex.suh_d, ex.svh_d, e, I, D)).to(BF16).float()
        return y

    # -- one block of rows ----------------------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, sc: SeqCache, ids: torch.Tensor, start: int, all_logits: bool = False,
                taps: list | None = None, host_ids: list[int] | None = None, replay: int | None = None,
                image: tuple | None = None):
        """Rows ids [n] at positions start.. of one sequence -> fp32 logits [n or 1, V] (both ranks the same).
        ``image``: (rows [k] long, embeddings [k, dim] bf16), this block's image-span rows: their embeddings replace the
        token's, the MoE gate picks their experts with its VL bias and Engram leaves them untouched (DeepSeek's
        image_mask); ``host_ids`` then holds a negative id at each of them (no n-gram spans one)."""

        c, w = self.cfg, self.w
        n = ids.shape[0]
        assert start == sc.length, (start, sc.length)
        sc.tokens[start:start + n] = ids
        emb = w.embed[ids].to(BF16)
        img = None
        if image is not None and image[0].numel():
            img = image[0]
            emb[img] = image[1].to(BF16)
        h = emb[:, None, :].expand(-1, c.hc, -1).contiguous()
        pre = torch.zeros((n, c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        hashes = None
        if self.engram is not None:
            if host_ids is None:
                host_ids = ids.tolist()
            del sc.host[start:]
            sc.host.extend(int(t) for t in host_ids)
            hashes = self.engram.hashes(sc.host, start, n)                          # [n, L, cols] host
        shared: dict = {}
        if KERNELS:
            return self._forward_k(sc, ids, start, all_logits, taps, h, pre, hashes, shared, replay, img)
        for lay in w.layers:
            if hashes is not None and lay.engram_wkv is not None:
                h = self.engram_apply(lay, h, hashes[:, c.engram_layers.index(lay.idx)], img)
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
            y = self.comm.sum(self.moe(lay, x, img=img)).to(BF16)
            h = self.hc_post(y, res, f_post, f_comb)
            pre = f_pre
        sc.length = start + n
        if not all_logits:
            h, pre = h[-1:], pre[-1:]
        x = rms_norm(self.hc_pre(h, pre), w.norm, c.eps)
        local = mm(w.head, x, F32)                                        # [n, V / world]
        g = self.comm.gather(local)                                       # [world, n, V / world]
        return g.permute(1, 0, 2).reshape(local.shape[0], -1)

    def _forward_k(self, sc, ids, start, all_logits, taps, h, pre, hashes, shared, replay=None, img=None):
        """``replay`` (decoder SWA bounded replay, CED's prefill): the decoder layers run only for rows at positions
        >= replay, their window truncated there; the first decoder layer's kv source still covers every row."""

        c, w = self.cfg, self.w
        n = ids.shape[0]
        dev = ids.device
        pos = torch.arange(start, start + n, device=dev)
        x = torch.empty((n, c.dim), dtype=BF16, device=dev)
        part = torch.empty((n * K.HC_BLOCKS * 32,), dtype=F32, device=dev)
        pre_a = torch.empty((n, c.hc), dtype=F32, device=dev)
        pre_f = torch.empty((n, c.hc), dtype=F32, device=dev)
        post = torch.empty((n, c.hc), dtype=F32, device=dev)
        comb = torch.empty((n, c.hc, c.hc), dtype=F32, device=dev)
        floor, kv_done = 0, False
        self.taps_start = start
        for lay in w.layers:
            if replay is not None and lay.idx == c.n_layers // 2:
                fn, scale, base = lay.hc_attn
                K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb,
                         part)
                self.kv_source_update(lay, x, sc, start, shared)
                first = max(start, replay) - start
                if first >= n:
                    sc.length = start + n
                    return None                                  # an encoder-only chunk
                if first:
                    h, pre = h[first:].contiguous(), pre[first:].contiguous()
                    start, n = start + first, n - first
                    pos = pos[first:]
                    if img is not None:
                        img = img[img >= first] - first
                        img = img if img.numel() else None
                    x = torch.empty((n, c.dim), dtype=BF16, device=dev)
                    part = torch.empty((n * K.HC_BLOCKS * 32,), dtype=F32, device=dev)
                    pre_a = torch.empty((n, c.hc), dtype=F32, device=dev)
                    pre_f = torch.empty((n, c.hc), dtype=F32, device=dev)
                    post = torch.empty((n, c.hc), dtype=F32, device=dev)
                    comb = torch.empty((n, c.hc, c.hc), dtype=F32, device=dev)
                    self.taps_start = start
                floor, kv_done = replay, True
            if hashes is not None and lay.engram_wkv is not None:
                with _T("engram"):
                    rows = hashes[:, c.engram_layers.index(lay.idx)]
                    h = self.engram_apply(lay, h, rows[-n:] if rows.shape[0] != n else rows, img)
            if taps is not None and lay.idx in c.dspark_taps:
                taps.append(h.to(F32).mean(1).to(BF16))
            fn, scale, base = lay.hc_attn
            with _T("hc"):
                K.hc_pre(h, fn, scale, base, pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_a, post, comb,
                         part)
            with _T("attn_r%d" % (2 if lay.comp_wkv is not None else 1 if lay.idx_wq_b is not None else 0)):
                pa = self.attention_k(lay, x, sc, start, shared, pos, floor=floor,
                                      kv_done=kv_done and lay.idx == c.n_layers // 2)
            with _T("gather"):
                g = self.comm.gather(pa)
            with _T("hc"):
                K.hc_post(g, h, post, comb, h)
                fn, scale, base = lay.hc_ffn
                K.hc_pre(h, fn, scale, base, pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, x, pre_f, post, comb,
                         part)
            with _T("moe"):
                pm = self.moe(lay, x, img=img)
            with _T("gather"):
                g = self.comm.gather(pm)
            with _T("hc"):
                K.hc_post(g, h, post, comb, h)
            pre, pre_f = pre_f, pre
        sc.length = start + n
        if not all_logits:
            h, pre = h[-1:], pre[-1:]
        with _T("head"):
            xc = K.collapse_norm(h.contiguous(), pre.contiguous(), w.norm, c.eps)
            local = mm(w.head, xc, F32)
            g = self.comm.gather(local)
            out = g.permute(1, 0, 2).reshape(local.shape[0], -1)
        return out


def _candidates(score: torch.Tensor, vis: torch.Tensor, nblocks: int, bsize: int) -> torch.Tensor:
    """The candidate pool as a block mask [rows, ceil(width / bsize)] (the newest, partly filled block pinned in)."""

    width = score.shape[-1]
    s = F.pad(score, (0, -width % bsize), value=float("-inf")).unflatten(-1, (-1, bsize)).amax(-1)
    nb = s.shape[-1]
    last = (vis - 1) // bsize
    s = s.masked_fill(torch.arange(nb, device=score.device)[None] == last, float("inf"))
    idx = K.topk_indices(s, min(nblocks, nb))
    return torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, s.gather(-1, idx) > float("-inf"))


def apply_candidates(score: torch.Tensor, blocks: torch.Tensor, bsize: int) -> torch.Tensor:
    """-inf on every position outside the pool's blocks (score [rows, width], blocks [rows, >= ceil(width / bsize)])."""

    width = score.shape[-1]
    nb = -(-width // bsize)
    keep = blocks[:, :nb].repeat_interleave(bsize, dim=-1)[:, :width]
    return score.masked_fill_(~keep, float("-inf"))
