"""Reference forward of DeepSeek-V4.1-Flash on the EXL3 checkpoint: the oracle for every engine kernel.

Math follows DeepSeek's own MIT inference code (deepseek-ai/DeepSeek-V4.1-Flash, inference/model.py, engram.py,
kernel.py), re-implemented here in plain PyTorch; nothing of it is copied. Weights are the EXL3 checkpoint the kit and
our engine serve, each matrix dequantized on the GPU to fp32 (W = diag(suh) H W_q H diag(svh), TensorFold's EXL3
decoder for W_q) only when its layer runs, so one GB10 runs the whole 40-layer model on short prompts.

Scope: a full-sequence forward (prefill semantics, start_pos 0) for a batch of independent token sequences, run
layer by layer over all of them, so every routed expert is read once a layer for the whole batch. Returns fp32
logits for every position, and optionally the residual stream after every layer for kernel checks.

The quantization the trained model expects is simulated exactly as DeepSeek's kernels define it: SWA KV in FP8 E4M3
with a power-of-two scale per 32; compressed main KV in FP4 E2M1 with an E4M3 scale per 16; indexer Q and K in FP4
with a power-of-two scale per 32. ``kv_quant=False`` keeps those in bf16 instead (for A/B against the kit).
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
F32 = torch.float32

# fp32 matmuls must be real fp32: NGC containers turn TF32 on (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1), which also makes
# results depend on how many rows share a call. Run with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 as well.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


# ----------------------------------------------------------------------------------------------------------------
# config

@dataclass
class Cfg:
    vocab: int
    dim: int
    inter: int
    n_layers: int
    n_heads: int
    head_dim: int
    rope_dim: int
    q_rank: int
    o_rank: int
    o_groups: int
    window: int
    eps: float
    n_routed: int
    topk: int
    route_scale: float
    swiglu_limit: float
    compress_ratios: list[int]
    kv_sources: list[int]
    index_sources: list[int]
    idx_heads: int
    idx_dim: int
    idx_topk: int
    cand_source: int
    cand_blocks: int
    cand_block: int
    hc: int
    hc_iters: int
    hc_eps: float
    rope_theta: float
    compress_theta: float
    rope_factor: float
    orig_len: int
    beta_fast: float
    beta_slow: float
    engram_layers: list[int]
    engram_rows: list[int]
    engram_ngram: int
    engram_vocab: int
    engram_heads: int
    engram_dim: int
    engram_pad: int
    engram_cvocab: int
    dspark_block: int = 0
    dspark_noise: int = 0
    dspark_taps: list[int] = field(default_factory=list)
    dspark_rank: int = 0
    dspark_routed: int = 0
    dspark_topk: int = 0

    @classmethod
    def read(cls, model_dir: str | Path) -> "Cfg":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        t = raw.get("text_config", raw)
        rs = t.get("rope_scaling") or {}
        return cls(
            vocab=t["vocab_size"], dim=t["hidden_size"], inter=t["moe_intermediate_size"],
            n_layers=t["num_hidden_layers"], n_heads=t["num_attention_heads"], head_dim=t["head_dim"],
            rope_dim=t["qk_rope_head_dim"], q_rank=t["q_lora_rank"], o_rank=t["o_lora_rank"], o_groups=t["o_groups"],
            window=t["sliding_window"], eps=t["rms_norm_eps"], n_routed=t["n_routed_experts"],
            topk=t["num_experts_per_tok"], route_scale=t["routed_scaling_factor"], swiglu_limit=t["swiglu_limit"],
            compress_ratios=list(t["compress_ratios"]), kv_sources=list(t["kv_source_layer_ids"]),
            index_sources=list(t["index_source_layer_ids"]), idx_heads=t["index_n_heads"],
            idx_dim=t["index_head_dim"], idx_topk=t["index_topk"], cand_source=t.get("candidate_source_layer_id", -1),
            cand_blocks=t.get("candidate_topk_blocks", 0), cand_block=t.get("candidate_block_size", 0),
            hc=t["hc_mult"], hc_iters=t["hc_sinkhorn_iters"], hc_eps=t["hc_eps"], rope_theta=t["rope_theta"],
            compress_theta=t["compress_rope_theta"], rope_factor=rs.get("factor", 1.0),
            orig_len=rs.get("original_max_position_embeddings", 0), beta_fast=rs.get("beta_fast", 32),
            beta_slow=rs.get("beta_slow", 1), engram_layers=list(t.get("engram_layer_ids", [])),
            engram_rows=list(t.get("engram_num_embeddings", [])), engram_ngram=t.get("engram_max_ngram_size", 1),
            engram_vocab=t.get("engram_vocab_size", 0), engram_heads=t.get("engram_n_heads", 0),
            engram_dim=t.get("engram_head_dim", 0), engram_pad=t.get("engram_pad_token_id", 2),
            engram_cvocab=t.get("engram_compressed_vocab_size", 0), dspark_block=t.get("dspark_block_size", 0),
            dspark_noise=t.get("dspark_noise_token_id", 0), dspark_taps=list(t.get("dspark_target_layer_ids", [])),
            dspark_rank=t.get("dspark_markov_rank", 0), dspark_routed=t.get("dspark_n_routed_experts", 0),
            dspark_topk=t.get("dspark_num_experts_per_tok", 0))


# ----------------------------------------------------------------------------------------------------------------
# weights: plain tensors and EXL3 groups, read lazily from the safetensors shards

class Store:
    """Tensors of one checkpoint folder by name, read on demand (header offsets, no full-file mapping)."""

    DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I16": torch.int16,
          "I32": torch.int32, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "I64": torch.int64}

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.entries: dict[str, tuple[Path, int, dict]] = {}
        for path in sorted(self.root.glob("*.safetensors")):
            with open(path, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
            header.pop("__metadata__", None)
            for name, e in header.items():
                self.entries[name] = (path, 8 + size, e)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def get(self, name: str, device: str = "cuda") -> torch.Tensor:
        path, base, e = self.entries[name]
        lo, hi = e["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + lo)
            buf = bytearray(f.read(hi - lo))
        dt = e["dtype"]
        if dt == "F8_E8M0":
            t = torch.frombuffer(buf, dtype=torch.uint8) if buf else torch.empty(0, dtype=torch.uint8)
        else:
            t = torch.frombuffer(buf, dtype=self.DT[dt]) if buf else torch.empty(0, dtype=self.DT[dt])
        return t.reshape(e["shape"]).to(device)


@lru_cache(maxsize=1)
def _hadamard(device: str = "cuda") -> torch.Tensor:
    n = 128
    i = torch.arange(n)
    bits = (i[:, None] & i[None, :])
    parity = torch.zeros_like(bits)
    for b in range(7):
        parity ^= (bits >> b) & 1
    return ((1 - 2 * parity).to(F32) / math.sqrt(n)).to(device)


def _unpack_wq(trellis: torch.Tensor, codebook: str) -> torch.Tensor:
    from tensorfold.cuda.exl3.linear import unpack_cuda

    return unpack_cuda(trellis, codebook)


def exl3_weight(store: Store, prefix: str) -> torch.Tensor:
    """W [K, N] fp32 of EXL3 group ``prefix``: y = x @ W."""

    codebook = "mul1" if f"{prefix}.mul1" in store else "mcg" if f"{prefix}.mcg" in store else "3inst"
    wq = _unpack_wq(store.get(f"{prefix}.trellis"), codebook).to(F32)
    suh = store.get(f"{prefix}.suh").to(F32)
    svh = store.get(f"{prefix}.svh").to(F32)
    k, n = wq.shape
    h = _hadamard()
    w = torch.einsum("bin,ij->bjn", wq.view(k // 128, 128, n), h).reshape(k, n) * suh[:, None]
    del wq
    w = torch.einsum("kci,ij->kcj", w.view(k, n // 128, 128), h).reshape(k, n) * svh[None, :]
    return w


def lin(x: torch.Tensor, w: torch.Tensor, out: torch.dtype = BF16) -> torch.Tensor:
    """x @ w in fp32, output cast to ``out`` (bf16 for DeepSeek's default-dtype linears)."""

    return (x.to(F32) @ w).to(out)


# ----------------------------------------------------------------------------------------------------------------
# numeric pieces (DeepSeek's definitions)

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.to(F32)
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (w.to(F32) * xf).to(dt)


@lru_cache(maxsize=8)
def freqs_cis(dim: int, seqlen: int, orig_len: int, base: float, factor: float, beta_fast: float,
              beta_slow: float, device: str = "cuda") -> torch.Tensor:
    """Complex rotations [seqlen, dim/2]; YaRN ramp when ``orig_len`` > 0 (no attention mscale)."""

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=F32) / dim))
    if orig_len > 0:
        def corr(rot: float) -> float:
            return dim * math.log(orig_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=F32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    ang = torch.outer(torch.arange(seqlen, dtype=F32), freqs)
    return torch.polar(torch.ones_like(ang), ang).to(device)


def rope_(x: torch.Tensor, f: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent pairs of x's last dim in place by f [rows, dim/2]; x [rows, ..., dim] (rows first)."""

    xc = torch.view_as_complex(x.to(F32).unflatten(-1, (-1, 2)).contiguous())
    if inverse:
        f = f.conj()
    shape = [f.shape[0]] + [1] * (xc.dim() - 2) + [f.shape[1]]
    y = torch.view_as_real(xc * f.view(shape)).flatten(-2)
    x.copy_(y.to(x.dtype))
    return x


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(v)) for positive fp32 v, exactly as the bit trick in DeepSeek's kernels."""

    bits = v.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(v), exp)


def fp8_qd(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP8 E4M3 quantize-dequantize with a power-of-two scale per ``block`` (amax floored at 1e-4)."""

    shape, dt = x.shape, x.dtype
    v = x.to(F32).reshape(-1, block)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / 448.0)
    q = (v / s).clamp(-448, 448).to(torch.float8_e4m3fn).to(F32) * s
    return q.reshape(shape).to(dt)


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _to_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value, ties to the even code (0, 1, 2, 4), v already clamped to [-6, 6]."""

    a = v.abs()
    grid = _E2M1.to(v.device)
    mids = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(a, mids)                       # the lower neighbour on a tie
    tie = (idx < len(mids)) & (a == mids[idx.clamp(max=len(mids) - 1)])
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return torch.copysign(grid[idx], v)


def fp4_qd(x: torch.Tensor, block: int, e4m3_scale: bool) -> torch.Tensor:
    """FP4 E2M1 quantize-dequantize per ``block``: E4M3 scale (compressed KV) or power-of-two scale (indexer)."""

    shape, dt = x.shape, x.dtype
    v = x.to(F32).reshape(-1, block)
    amax = v.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        s = (amax.clamp_min(6 * 2.0 ** -9) / 6.0).to(torch.float8_e4m3fn).to(F32)
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2.0 ** -126) / 6.0)
    q = _to_e2m1((v / s).clamp(-6, 6)) * s
    return q.reshape(shape).to(dt)


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, idx: torch.Tensor, scale: float) -> torch.Tensor:
    """q [s, h, d], kv [n, d] (key = value), idx [s, k] (-1 = none), sink [h] -> o [s, h, d] (bf16 in, fp32 math)."""

    s, h, d = q.shape
    valid = idx >= 0
    g = kv[idx.clamp_min(0)].to(F32)                         # [s, k, d]
    sc = torch.einsum("shd,skd->shk", q.to(F32), g) * scale
    sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
    m = torch.maximum(sc.amax(-1), sink.to(F32)[None, :])     # include the sink in the max (no effect on the result)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp(sc - m[..., None])
    den = p.sum(-1) + torch.exp(sink.to(F32)[None, :] - m)
    o = torch.einsum("shk,skd->shd", p, g) / den[..., None]
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int, eps: float):
    """mixes [n, (2+hc)*hc] -> pre [n, hc], post [n, hc], comb [n, hc, hc] (comb[j, k]: stream j into stream k)."""

    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# ----------------------------------------------------------------------------------------------------------------
# Engram hashing (DeepSeek's engram.py math)

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def engram_primes(cfg: Cfg) -> list[list[list[int]]]:
    """[layer][ngram size - 2][head] bucket primes, drawn in order and never reused."""

    seen, out = set(), []
    for _ in cfg.engram_layers:
        per = []
        for _ in range(cfg.engram_ngram - 1):
            cur, sizes = cfg.engram_vocab - 1, []
            for _ in range(cfg.engram_heads):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per.append(sizes)
        out.append(per)
    return out


def engram_multipliers(cfg: Cfg) -> torch.Tensor:
    bound = max(1, (np.iinfo(np.int64).max // cfg.engram_cvocab) // 2)
    rows = []
    for layer in cfg.engram_layers:
        g = np.random.default_rng(10007 * layer)
        v = g.integers(low=0, high=bound, size=(cfg.engram_ngram,), dtype=np.int64)
        rows.append(torch.tensor(v * 2 + 1))
    return torch.stack(rows)


def compressed_token_map(tokenizer_json: str | Path) -> tuple[list[int], int]:
    """Token id -> compressed id (tokens that normalize alike share one), as DeepSeek's engram.py builds it."""

    from tokenizers import Regex, Tokenizer, normalizers

    tok = Tokenizer.from_file(str(tokenizer_json))
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * n
    for tid in range(n):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(tid)
        else:
            nt = norm.normalize_str(text)
            key = nt if nt else text
        new = key_to_new.get(key)
        if new is None:
            new = len(key_to_new)
            key_to_new[key] = new
        lookup[tid] = new
    return lookup, len(key_to_new)


class EngramHasher:
    def __init__(self, cfg: Cfg, token_map: list[int], device: str = "cuda"):
        primes = engram_primes(cfg)
        flat = [[p for per in layer for p in per] for layer in primes]
        self.cfg = cfg
        self.primes = torch.tensor(primes, dtype=torch.int64, device=device)          # [L, n-1, H]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *f[:-1]]) for f in flat]), dtype=torch.int64,
                                    device=device)                                       # [L, (n-1)*H]
        self.mult = engram_multipliers(cfg).to(device)                                   # [L, n]
        self.map = torch.tensor(token_map, dtype=torch.int64, device=device)
        self.pad = int(token_map[cfg.engram_pad])
        for li, rows in enumerate(cfg.engram_rows):
            assert sum(flat[li]) == rows, (li, sum(flat[li]), rows)

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [s] (one sequence from its start) -> row ids [s, L, (n-1)*H]."""

        s = ids.shape[0]
        comp = self.map[ids]
        pos = torch.arange(s, device=ids.device)
        toks = []
        for shift in range(self.cfg.engram_ngram):
            src = comp[(pos - shift).clamp_min(0)]
            toks.append(torch.where(pos < shift, self.pad, src))
        toks = torch.stack(toks, -1)                                   # [s, n]
        prod = toks[:, None, :] * self.mult[None]                      # [s, L, n]
        rolling, out = prod[..., 0], []
        for i in range(1, self.cfg.engram_ngram):
            rolling = torch.bitwise_xor(rolling, prod[..., i])
            out.append(rolling[..., None] % self.primes[None, :, i - 1])
        return torch.cat(out, -1) + self.offsets[None]


class EngramTables:
    """Rows of the FP8 Engram tables read straight from the original shards by offset."""

    def __init__(self, engram_dir: str | Path, cfg: Cfg):
        self.cfg = cfg
        self.maps = {}
        root = Path(engram_dir)
        for path in sorted(root.glob("*.safetensors")):
            with open(path, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
            for name, e in header.items():
                if name.endswith("engram.embed.weight") or name.endswith("engram.embed.scale"):
                    lo, hi = e["data_offsets"]
                    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size + lo, shape=(hi - lo,))
                    self.maps[name] = mm.reshape(e["shape"])

    def rows(self, layer: int, idx: torch.Tensor) -> torch.Tensor:
        """idx [...] -> bf16 [..., head_dim]: fp8 values times their power-of-two scale per 32."""

        flat = idx.reshape(-1).cpu().numpy()
        w = np.asarray(self.maps[f"layers.{layer}.engram.embed.weight"][flat])
        s = np.asarray(self.maps[f"layers.{layer}.engram.embed.scale"][flat])
        v = torch.from_numpy(w.copy()).view(torch.float8_e4m3fn).to("cuda").to(F32)
        e = torch.from_numpy(s.copy()).to("cuda").to(torch.int32) - 127           # E8M0: 2^(byte - 127)
        sc = torch.ldexp(torch.ones_like(e, dtype=F32), e)
        v = (v.view(-1, v.shape[-1] // 32, 32) * sc[..., None]).view(v.shape)
        return v.to(BF16).reshape(*idx.shape, -1)


# ----------------------------------------------------------------------------------------------------------------
# the model

class Reference:
    def __init__(self, model_dir: str | Path, engram_dir: str | Path | None, *, kv_quant: bool = True,
                 tokenizer_json: str | Path | None = None, token_map: list[int] | None = None):
        self.cfg = Cfg.read(model_dir)
        self.w = Store(model_dir)
        self.kv_quant = kv_quant
        c = self.cfg
        self.engram = None
        if c.engram_layers and engram_dir is not None:
            if token_map is None:
                token_map, n = compressed_token_map(tokenizer_json or Path(model_dir) / "tokenizer.json")
                assert n == c.engram_cvocab, (n, c.engram_cvocab)
            self.hasher = EngramHasher(c, token_map)
            self.engram = EngramTables(engram_dir, c)

    # -- per-layer pieces --------------------------------------------------------------------------------------
    def _plain(self, name: str) -> torch.Tensor:
        return self.w.get(name)

    def _freqs(self, layer: int, n: int) -> torch.Tensor:
        c = self.cfg
        if c.compress_ratios[layer]:
            return freqs_cis(c.rope_dim, n, c.orig_len, c.compress_theta, c.rope_factor, c.beta_fast, c.beta_slow)
        return freqs_cis(c.rope_dim, n, 0, c.rope_theta, c.rope_factor, c.beta_fast, c.beta_slow)

    def hc_mixes(self, x: torch.Tensor, pfx: str, kind: str):
        c = self.cfg
        fn = self._plain(f"{pfx}.hc_{kind}_fn").to(F32)
        scale = self._plain(f"{pfx}.hc_{kind}_scale").to(F32)
        base = self._plain(f"{pfx}.hc_{kind}_base").to(F32)
        xf = x.flatten(1).to(F32)                                          # [n, hc*d]
        rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + c.eps)
        mixes = (xf @ fn.t()) * rs
        return hc_split_sinkhorn(mixes, scale, base, c.hc, c.hc_iters, c.hc_eps)

    @staticmethod
    def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return (pre[..., None] * x.to(F32)).sum(1).to(x.dtype)

    @staticmethod
    def hc_post(y: torch.Tensor, res: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        out = post[..., None] * y.to(F32)[:, None, :] + (comb[..., None] * res.to(F32)[:, :, None, :]).sum(1)
        return out.to(y.dtype)

    def engram_apply(self, layer: int, h: torch.Tensor, hashes: torch.Tensor) -> torch.Tensor:
        """h [n, hc, d], hashes [n, cols] -> h + gate * value."""

        c = self.cfg
        pfx = f"layers.{layer}.engram"
        e = self.engram.rows(layer, hashes).flatten(1)                     # [n, cols*256] bf16
        wkv = exl3_weight(self.w, f"{pfx}.wkv")
        kv = lin(e, wkv)
        del wkv
        key, value = kv.split([c.hc * c.dim, c.dim], dim=-1)
        key = key.to(F32).unflatten(-1, (c.hc, c.dim))
        weight = self._plain(f"{pfx}.q_weight").to(F32) * self._plain(f"{pfx}.k_weight").to(F32)
        hf = h.to(F32)
        rstd = torch.rsqrt(hf.square().mean(-1) + c.eps) * torch.rsqrt(key.square().mean(-1) + c.eps)
        dot = (hf * weight * key).sum(-1) * rstd * c.dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (hf + gate[..., None] * value.to(F32)[:, None, :]).to(h.dtype)

    def attention(self, layer: int, x_all: torch.Tensor, seqs: list[slice], shared: list[dict]) -> torch.Tensor:
        """x_all [n, d] (normed attention input of every sequence's tokens) -> [n, d]."""

        c = self.cfg
        pfx = f"layers.{layer}.attn"
        ratio = c.compress_ratios[layer]
        rd = c.rope_dim
        wq_a = exl3_weight(self.w, f"{pfx}.wq_a")
        qr_all = rms_norm(lin(x_all, wq_a), self._plain(f"{pfx}.q_norm.weight"), c.eps)
        del wq_a
        wq_b = exl3_weight(self.w, f"{pfx}.wq_b")
        q_all = lin(qr_all, wq_b).unflatten(-1, (c.n_heads, c.head_dim))
        del wq_b
        wkv = exl3_weight(self.w, f"{pfx}.wkv")
        kv_all = rms_norm(lin(x_all, wkv), self._plain(f"{pfx}.kv_norm.weight"), c.eps)
        del wkv
        sink = self._plain(f"{pfx}.attn_sink").to(F32)
        is_kv_src = layer in c.kv_sources
        is_idx_src = layer in c.index_sources
        comp = None
        if is_kv_src:
            comp = {"wkv": exl3_weight(self.w, f"{pfx}.compressor.wkv"),
                    "norm": self._plain(f"{pfx}.compressor.norm.weight")}
            if ratio > 1:
                comp["wgate"] = exl3_weight(self.w, f"{pfx}.compressor.wgate")
        idx_w = None
        if is_idx_src:
            idx_w = {"wq_b": exl3_weight(self.w, f"{pfx}.indexer.wq_b"),
                     "proj": self._plain(f"{pfx}.indexer.weights_proj.weight").to(F32)}
            if is_kv_src:
                idx_w["wk"] = exl3_weight(self.w, f"{pfx}.indexer.wk")
                idx_w["k_norm"] = self._plain(f"{pfx}.indexer.k_norm.weight")
        outs = []
        for si, sl in enumerate(seqs):
            x, q, kv, qr = x_all[sl], q_all[sl].clone(), kv_all[sl].clone(), qr_all[sl]
            s = x.shape[0]
            f = self._freqs(layer, s)
            rope_(q[..., -rd:], f[:s])
            rope_(kv[..., -rd:], f[:s])
            if self.kv_quant:
                kv = fp8_qd(kv, 32)
            pos = torch.arange(s, device=x.device)
            widx = (pos[:, None] - c.window + 1).clamp_min(0) + torch.arange(min(s, c.window), device=x.device)
            widx = torch.where(widx > pos[:, None], -1, widx)
            keys, idx = kv, widx
            if ratio:
                sh = shared[si]
                n_comp = s // ratio
                latent = None
                if is_kv_src:
                    if ratio == 1:
                        latent = rms_norm(lin(x, comp["wkv"]), comp["norm"], c.eps)
                    else:
                        xf = x.to(F32)
                        kvc = xf @ comp["wkv"]
                        sc = xf @ comp["wgate"]
                        cut = n_comp * ratio
                        kvc = kvc[:cut].unflatten(0, (-1, ratio))
                        sc = sc[:cut].unflatten(0, (-1, ratio))
                        latent = rms_norm((kvc * sc.softmax(dim=1)).sum(1).to(BF16), comp["norm"], c.eps)
                if is_idx_src:
                    if is_kv_src and n_comp:
                        k = rms_norm(lin(latent, idx_w["wk"]), idx_w["k_norm"], c.eps)
                        rope_(k[..., -rd:], f[0:n_comp * ratio:ratio])
                        if self.kv_quant:
                            k = fp4_qd(k, 32, e4m3_scale=False)
                        sh["index_k"] = k
                    if n_comp == 0:
                        cidx = torch.empty((s, 0), dtype=torch.long, device=x.device)
                    else:
                        iq = lin(qr, idx_w["wq_b"]).unflatten(-1, (c.idx_heads, c.idx_dim))
                        rope_(iq[..., -rd:], f[:s])
                        if self.kv_quant:
                            iq = fp4_qd(iq, 32, e4m3_scale=False)
                        wts = lin(x, idx_w["proj"].t()) * (c.idx_dim ** -0.5 * c.idx_heads ** -0.5)
                        ik = sh["index_k"][:n_comp]
                        score = torch.einsum("shd,td->sht", iq.to(F32), ik.to(F32))
                        score = (score.relu() * wts.to(F32)[..., None]).sum(1)              # [s, n_comp]
                        lens = ((pos + 1) // ratio)[:, None]
                        score = score.masked_fill(torch.arange(n_comp, device=x.device)[None] >= lens, float("-inf"))
                        if layer == c.cand_source:
                            sh["cand"] = self._candidates(score, lens, c.cand_blocks, c.cand_block)
                        elif 0 <= c.cand_source < layer:
                            score = score.masked_fill(~sh["cand"], float("-inf"))
                        kk = min(c.idx_topk, n_comp)
                        top = score.topk(kk, dim=-1, sorted=False).indices.sort(dim=-1).values
                        cidx = torch.where(top < lens, top, -1)
                    sh["topk"] = cidx
                cidx = sh["topk"]
                if latent is not None and n_comp:
                    lat = latent.clone()
                    rope_(lat[..., -rd:], f[0:n_comp * ratio:ratio])
                    if self.kv_quant:
                        lat = fp4_qd(lat, 16, e4m3_scale=True)
                    sh["kv"] = lat
                ckv = sh["kv"][:n_comp] if n_comp else kv.new_zeros((0, c.head_dim))
                keys = torch.cat([kv, ckv], 0)
                idx = torch.cat([widx, torch.where(cidx >= 0, cidx + s, -1)], -1)
            o = sparse_attn(q, keys, sink, idx, c.head_dim ** -0.5)
            rope_(o[..., -rd:], f[:s], inverse=True)
            outs.append(o)
        o_all = torch.cat(outs, 0)                                            # [n, heads, 512]
        n = o_all.shape[0]
        og = o_all.view(n, c.o_groups, -1)
        parts = []
        for g in range(c.o_groups):
            wo = exl3_weight(self.w, f"{pfx}.wo_a.slice.{g}")
            parts.append(lin(og[:, g], wo))
            del wo
        u = torch.cat(parts, -1)                                               # [n, groups * o_rank]
        wo_b = exl3_weight(self.w, f"{pfx}.wo_b")
        out = lin(u, wo_b)
        return out

    @staticmethod
    def _candidates(score: torch.Tensor, lens: torch.Tensor, nblocks: int, bsize: int) -> torch.Tensor:
        width = score.shape[-1]
        sc = F.pad(score, (0, -width % bsize), value=float("-inf")).unflatten(-1, (-1, bsize)).amax(-1)
        nb = sc.shape[-1]
        last = (lens - 1) // bsize
        sc = sc.masked_fill(torch.arange(nb, device=score.device)[None] == last, float("inf"))
        top = sc.topk(min(nblocks, nb), dim=-1)
        keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
        return keep.repeat_interleave(bsize, dim=-1)[..., :width]

    def moe(self, layer: int, x: torch.Tensor, prefix: str | None = None, n_routed: int | None = None,
            topk: int | None = None) -> torch.Tensor:
        c = self.cfg
        pfx = prefix or f"layers.{layer}.ffn"
        k = topk or c.topk
        gw = self._plain(f"{pfx}.gate.weight").to(F32)
        gb = self._plain(f"{pfx}.gate.bias").to(F32)
        scores = F.softplus(x.to(F32) @ gw.t()).sqrt()
        ind = (scores + gb).topk(k, dim=-1).indices
        wts = scores.gather(1, ind)
        if k > 1:
            wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
        wts = wts * c.route_scale
        y = torch.zeros_like(x, dtype=F32)

        def expert(ep: str, xs: torch.Tensor, ws: torch.Tensor | None) -> torch.Tensor:
            w1 = exl3_weight(self.w, f"{ep}.w1")
            gate = lin(xs, w1).to(F32)
            del w1
            w3 = exl3_weight(self.w, f"{ep}.w3")
            up = lin(xs, w3).to(F32)
            del w3
            if c.swiglu_limit > 0:
                up = up.clamp(-c.swiglu_limit, c.swiglu_limit)
                gate = gate.clamp(max=c.swiglu_limit)
            h = F.silu(gate) * up
            if ws is not None:
                h = ws * h
            w2 = exl3_weight(self.w, f"{ep}.w2")
            out = lin(h.to(BF16), w2)
            del w2
            return out

        for e in torch.unique(ind).tolist():
            rows, slot = torch.where(ind == e)
            y[rows] += expert(f"{pfx}.experts.{e}", x[rows], wts[rows, slot, None]).to(F32)
        y += expert(f"{pfx}.shared_experts", x, None).to(F32)
        return y.to(x.dtype)

    # -- whole model -------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, seqs_ids: list[list[int]], dump_layers: bool = False, log=print, n_layers: int | None = None) -> dict:
        """Independent sequences (each from position 0) -> fp32 logits [n_i, vocab] each (and per-layer streams)."""

        c = self.cfg
        lens = [len(s) for s in seqs_ids]
        bounds, o = [], 0
        for n in lens:
            bounds.append(slice(o, o + n))
            o += n
        ids = torch.tensor([t for s in seqs_ids for t in s], dtype=torch.long, device="cuda")
        emb = self._plain("embed.weight")
        h = emb[ids].to(BF16)
        del emb
        h = h[:, None, :].expand(-1, c.hc, -1).contiguous()                  # [n, hc, d]
        pre = torch.zeros((h.shape[0], c.hc), dtype=F32, device="cuda")
        pre[:, 0] = 1.0
        hashes = None
        if self.engram is not None:
            hashes = torch.cat([self.hasher(ids[b]) for b in bounds], 0)      # [n, L, cols]
        shared = [dict() for _ in bounds]
        taps, dumps = [], []
        for layer in range(c.n_layers if n_layers is None else n_layers):
            if self.engram is not None and layer in c.engram_layers:
                h = self.engram_apply(layer, h, hashes[:, c.engram_layers.index(layer)])
            if layer in c.dspark_taps:
                taps.append(h.to(F32).mean(1).to(BF16))
            pfx = f"layers.{layer}"
            res = h
            a_pre, a_post, a_comb = self.hc_mixes(h, pfx, "attn")
            x = rms_norm(self.hc_pre(h, pre), self._plain(f"{pfx}.attn_norm.weight"), c.eps)
            x = self.attention(layer, x, bounds, shared)
            h = self.hc_post(x, res, a_post, a_comb)
            res = h
            f_pre, f_post, f_comb = self.hc_mixes(h, pfx, "ffn")
            x = rms_norm(self.hc_pre(h, a_pre), self._plain(f"{pfx}.ffn_norm.weight"), c.eps)
            x = self.moe(layer, x)
            h = self.hc_post(x, res, f_post, f_comb)
            pre = f_pre
            if dump_layers:
                dumps.append(h.cpu())
            log(f"[ref] layer {layer} done")
        x = rms_norm(self.hc_pre(h, pre), self._plain("norm.weight"), c.eps)
        head = exl3_weight(self.w, "head")
        logits = x.to(F32) @ head
        del head
        out = {"logits": [logits[b].cpu() for b in bounds]}
        if taps:
            out["taps"] = [torch.cat([t[b] for t in taps], -1).cpu() for b in bounds]
        if dump_layers:
            out["layers"] = dumps
        return out
