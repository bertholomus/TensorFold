"""Numeric definitions of DeepSeek-V4.1 (from DeepSeek's MIT inference code, re-implemented): RMSNorm, YaRN RoPE,
the FP8 / FP4 cache quantization, sparse attention with a sink, mHC Sinkhorn, Engram hashing."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Cfg

BF16 = torch.bfloat16
F32 = torch.float32


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


# The engine's RoPE tables: (cos, sin) fp32 [rows, dim/2] a rope kind, no complex table kept. Rows below ROPE_COMPAT are
# freqs_cis(dim, ROPE_COMPAT, ...)'s: every lane up to e5074cf ran --context 262144 on a 2^19-row table, and a CPU
# table's last bits may depend on its length (the reason the engine reads one table), so the rows an earlier lane served
# keep their bits. Rows from ROPE_COMPAT on come from the same formula in blocks of ROPE_BLOCK rows, each block one call
# of the same shape, so no row's bits depend on how long the table is. Rows are aligned up to ROPE_ALIGN (the old
# tables were the next power of two: 2^21 rows, 2 GiB for both kinds with their complex copies, at a 1M context).
ROPE_COMPAT = 1 << 19
ROPE_BLOCK = 1 << 16
ROPE_ALIGN = 4096


def rope_block(dim: int, start: int, n: int, orig_len: int, base: float, factor: float, beta_fast: float,
               beta_slow: float) -> torch.Tensor:
    """freqs_cis's rotations at positions start .. start + n - 1: complex64 [n, dim/2] on the CPU (its arithmetic)."""

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=F32) / dim))
    if orig_len > 0:
        def corr(rot: float) -> float:
            return dim * math.log(orig_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=F32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    ang = torch.outer(torch.arange(start, start + n, dtype=torch.int64).to(F32), freqs)     # exact below 2^24
    return torch.polar(torch.ones_like(ang), ang)


def rope_cs(dim: int, rows: int, orig_len: int, base: float, factor: float, beta_fast: float, beta_slow: float,
            device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) fp32 [rows, dim/2] on ``device``: freqs_cis's real and imaginary parts, rows below ROPE_COMPAT from
    freqs_cis(dim, ROPE_COMPAT, ...) itself, the rest from rope_block in ROPE_BLOCK steps (see ROPE_COMPAT)."""

    half = dim // 2
    cos = torch.empty((rows, half), dtype=F32, device=device)
    sin = torch.empty((rows, half), dtype=F32, device=device)
    f = freqs_cis.__wrapped__(dim, ROPE_COMPAT, orig_len, base, factor, beta_fast, beta_slow, device="cpu")
    n = min(rows, ROPE_COMPAT)
    cos[:n].copy_(f.real[:n])
    sin[:n].copy_(f.imag[:n])
    del f
    for start in range(ROPE_COMPAT, rows, ROPE_BLOCK):
        f = rope_block(dim, start, ROPE_BLOCK, orig_len, base, factor, beta_fast, beta_slow)
        n = min(rows - start, ROPE_BLOCK)
        cos[start:start + n].copy_(f.real[:n])
        sin[start:start + n].copy_(f.imag[:n])
    return cos, sin


class RopeTables:
    """The engine's (cos, sin) table a rope kind (rope_cs): the compressed layers' (YaRN, compress_rope_theta) and
    the others' (rope_theta), each built at its first use for a row count (aligned up to ROPE_ALIGN) and kept: graphs
    captured on a table hold its address, so none is ever replaced or freed."""

    def __init__(self, cfg: Cfg, device: str = "cuda"):
        self.cfg, self.device = cfg, device
        self.tables: dict = {}

    def args(self, compressed: bool) -> tuple:
        c = self.cfg
        if compressed:
            return (c.orig_len, c.compress_theta, c.rope_factor, c.beta_fast, c.beta_slow)
        return (0, c.rope_theta, c.rope_factor, c.beta_fast, c.beta_slow)

    @staticmethod
    def rows(need: int) -> int:
        return max(ROPE_ALIGN, -(-int(need) // ROPE_ALIGN) * ROPE_ALIGN)

    def cs(self, compressed: bool, need: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin) [rows(need), rope_dim / 2] of this kind (the longest table already made when it holds them:
        the same bits in every row)."""

        compressed = bool(compressed)
        rows = self.rows(need)
        have = [r for k, r in self.tables if k == compressed and r >= rows]
        if have:
            return self.tables[(compressed, max(have))]
        t = self.tables[(compressed, rows)] = rope_cs(self.cfg.rope_dim, rows, *self.args(compressed),
                                                      device=self.device)
        return t

    def nbytes(self) -> int:
        return sum(x.numel() * x.element_size() for pair in self.tables.values() for x in pair)


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


_E2M1_DEV: dict = {}


def _e2m1_on(device) -> torch.Tensor:
    t = _E2M1_DEV.get(str(device))
    if t is None:
        t = _E2M1_DEV[str(device)] = _E2M1.to(device)
    return t


def _to_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value, ties to the even code (0, 1, 2, 4), v already clamped to [-6, 6]."""

    a = v.abs()
    grid = _e2m1_on(v.device)
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


class HostIds:
    """A sequence's token ids on the host (Engram's n-grams; -1 at image positions) in a growable int32 buffer: 4 bytes
    an id where a Python list of ints held ~36 (a 1M prompt ~4 MiB instead of ~36 MiB, each kept prompt's copy alike).
    ``ids[a:b]`` is an int32 array view (valid until the next ``set``), ``len(ids)`` the ids held."""

    __slots__ = ("buf", "n")

    def __init__(self, ids=()) -> None:
        a = np.asarray(ids, dtype=np.int64).reshape(-1)
        assert a.size == 0 or (a.min() >= -2**31 and a.max() < 2**31), "ids fit int32"
        self.buf = np.array(a, dtype=np.int32)
        self.n = a.size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, s):
        return self.buf[:self.n][s]

    def view(self) -> np.ndarray:
        """Every id held (an int32 view)."""

        return self.buf[:self.n]

    def truncate(self, start: int) -> None:
        """Keep the ids before ``start`` (as ``del ids[start:]``)."""

        self.n = min(self.n, max(0, start))

    def set(self, start: int, ids) -> None:
        """The ids from ``start`` on are ``ids`` (as ``del l[start:]; l.extend(ids)``); ``start`` <= len."""

        assert 0 <= start <= self.n, (start, self.n)
        a = np.asarray(ids, dtype=np.int64).reshape(-1)
        assert a.size == 0 or (a.min() >= -2**31 and a.max() < 2**31), "ids fit int32"
        end = start + a.size
        if end > self.buf.size:
            grown = np.empty(max(end, self.buf.size + self.buf.size // 2, 1024), dtype=np.int32)
            grown[:start] = self.buf[:start]
            self.buf = grown
        self.buf[start:end] = a
        self.n = end

    def copy(self, top: int | None = None) -> "HostIds":
        """The first ``top`` ids (all of them when None) as a buffer of their own."""

        return HostIds(self.buf[:self.n if top is None else min(top, self.n)])


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


def _e2m1_code(v: torch.Tensor) -> torch.Tensor:
    """E2M1 code (sign bit 3, magnitude index 0..7 of 0, .5, 1, 1.5, 2, 3, 4, 6), round to nearest, ties to even."""

    a = v.abs()
    grid = _e2m1_on(v.device)
    mids = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(a, mids)
    tie = (idx < len(mids)) & (a == mids[idx.clamp(max=len(mids) - 1)])
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return (idx | torch.where((v < 0) & (idx > 0), 8, 0)).to(torch.uint8)


def fp4_pack(x: torch.Tensor, block: int, e4m3_scale: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """x [rows, D] -> (codes uint8 [rows, D / 2]: byte j holds element j (low nibble) and j + D / 2 (high nibble);
    scales uint8 [rows, D / block]: E4M3 bits, or the E8M0 exponent byte). Dequantized, the same values as fp4_qd."""

    rows, d = x.shape
    v = x.to(F32).reshape(-1, block)
    amax = v.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        s8 = (amax.clamp_min(6 * 2.0 ** -9) / 6.0).to(torch.float8_e4m3fn)
        s = s8.to(F32)
        sb = s8.view(torch.uint8)
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2.0 ** -126) / 6.0)
        e = ((s.view(torch.int32) >> 23) & 0xFF)                       # biased exponent = the E8M0 byte
        sb = e.to(torch.uint8)
    code = _e2m1_code((v / s).clamp(-6, 6)).reshape(rows, d)
    packed = code[:, :d // 2] | (code[:, d // 2:] << 4)
    return packed.contiguous(), sb.reshape(rows, d // block).contiguous()


def fp4_unpack(codes: torch.Tensor, scales: torch.Tensor, block: int, e4m3_scale: bool) -> torch.Tensor:
    """The inverse of fp4_pack, to bf16 (exact: an E2M1 value times its scale fits bf16)."""

    rows, h = codes.shape
    d = 2 * h
    code = torch.cat([codes & 15, codes >> 4], 1).long()
    mag = _e2m1_on(codes.device)[code & 7]
    val = torch.where((code & 8) != 0, -mag, mag)
    if e4m3_scale:
        s = scales.view(torch.float8_e4m3fn).to(F32)
    else:
        s = torch.ldexp(torch.ones(scales.shape, device=codes.device), scales.to(torch.int32) - 127)
    return (val.view(rows, d // block, block) * s[..., None]).view(rows, d).to(BF16)
