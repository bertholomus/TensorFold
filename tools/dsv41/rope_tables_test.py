"""The engine's RoPE tables (ops.RopeTables / rope_cs) against the tables they replace: the rows a 262,144-token lane
read keep their bits, no row's bits depend on the table's length, the complex rows the prompt path rotates with are
freqs_cis's, and the bytes. CPU by default (no GPU needed); exits 1 on a failed check.

  python3 tools/dsv41/rope_tables_test.py [--device cpu|cuda] [--out F]
"""

import argparse
import json
import math
import sys

import torch

from tensorfold.families.deepseek_v41.config import Cfg
from tensorfold.families.deepseek_v41.ops import (ROPE_ALIGN, ROPE_BLOCK, ROPE_COMPAT, RopeTables, freqs_cis, rope_,
                                                  rope_block, rope_cs)

# config.json's rope fields (DeepSeek-V4.1-Flash)
RD = 64
KINDS = {"compressed": (65536, 160000.0, 16.0, 32, 1), "plain": (0, 10000.0, 16.0, 32, 1)}
CONTEXTS = (262144, 524288, 1048576)
EXTRA = 14                                   # the engine's rope_cap: context + drafts + 1 + 8 (--mtp-drafts 5)


def bits(t: torch.Tensor) -> torch.Tensor:
    t = torch.view_as_real(t) if t.is_complex() else t
    return t.contiguous().view(torch.int32)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(bits(a.cpu()), bits(b.cpu())))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--out")
    a = p.parse_args()
    dev = a.device
    out: dict = {"checks": {}, "info": {}}
    ok = out["checks"]
    torch.manual_seed(0)
    old = {k: freqs_cis.__wrapped__(RD, ROPE_COMPAT, *args, device="cpu") for k, args in KINDS.items()}
    for kind, args in KINDS.items():
        f = old[kind]
        tables = {}
        for ctx in CONTEXTS:
            rows = RopeTables.rows(ctx + EXTRA)
            cos, sin = rope_cs(RD, rows, *args, device=dev)
            tables[ctx] = (cos, sin)
            n = min(rows, ROPE_COMPAT)
            # the rows the 262K lane's 2^19-row table held keep its bits (that table: freqs_cis(64, 2^19) .real/.imag)
            ok[f"{kind}/{ctx}/compat_rows_equal_old_table"] = same(cos[:n], f.real[:n]) and same(sin[:n], f.imag[:n])
            ok[f"{kind}/{ctx}/shape"] = (tuple(cos.shape) == (rows, RD // 2) == tuple(sin.shape)
                                         and rows % ROPE_ALIGN == 0)
            if rows > ROPE_COMPAT:
                # rows past the compat block: the rotations at their own positions (vs fp64 of the same fp32 angles)
                pos = torch.cat([torch.arange(ROPE_COMPAT, ROPE_COMPAT + 3), torch.randint(ROPE_COMPAT, rows, (509,)),
                                 torch.arange(rows - 3, rows)])
                ang32 = (pos.to(torch.float32)[:, None] * _freq_vector(args)[None]).double()
                err = max(float((cos[pos].cpu().double() - ang32.cos()).abs().max()),
                          float((sin[pos].cpu().double() - ang32.sin()).abs().max()))
                ok[f"{kind}/{ctx}/ext_rows_are_their_positions"] = err < 2e-6
                out["info"][f"{kind}/{ctx}/ext_max_abs_err_vs_fp64"] = err
        # no row's bits depend on the table's length
        c1, s1 = tables[1048576]
        for ctx in CONTEXTS[:-1]:
            c, s = tables[ctx]
            ok[f"{kind}/prefix_{ctx}_of_1M_identical"] = same(c1[:c.shape[0]], c) and same(s1[:s.shape[0]], s)
        # rope_block is freqs_cis's arithmetic (the same call shape gives the same bits)
        ok[f"{kind}/rope_block_is_freqs_cis"] = same(rope_block(RD, 0, 4096, *args),
                                                     freqs_cis.__wrapped__(RD, 4096, *args, device="cpu"))
        # info: is this CPU's table length-dependent at all (2^21 in one call vs the blocks past 2^19)?
        mono = freqs_cis.__wrapped__(RD, 1 << 21, *args, device="cpu")
        n = c1.shape[0]
        out["info"][f"{kind}/one_call_2^21_equals_blocks"] = same(c1, mono.real[:n]) and same(s1, mono.imag[:n])
        out["info"][f"{kind}/one_call_2^21_prefix_equals_2^19"] = same(mono[:ROPE_COMPAT], f)
        del mono
        # the prompt path's complex rows (_rot) are freqs_cis's rows, and rope_ rotates with the same bits
        cos, sin = tables[262144]
        idx = torch.randint(0, 262144, (4096,))
        sl = slice(131000, 131000 + 2048)
        rot_idx = torch.complex(cos[idx.to(cos.device)], sin[idx.to(cos.device)]).cpu()
        rot_sl = torch.complex(cos[sl], sin[sl]).cpu()
        ok[f"{kind}/complex_rows_equal_old"] = same(rot_idx, f[idx]) and same(rot_sl, f[sl])
        for dt in (torch.float32, torch.bfloat16):
            x = (torch.randn(2048, 8, RD) * 3).to(dt)
            for inv in (False, True):
                ra = rope_(x.clone(), f[sl], inverse=inv)
                rb = rope_(x.clone(), rot_sl, inverse=inv)
                ok[f"{kind}/rope_{str(dt).split('.')[1]}_{'inv' if inv else 'fwd'}_same_bits"] = bool(
                    torch.equal(ra.view(torch.int16 if dt == torch.bfloat16 else torch.int32),
                                rb.view(torch.int16 if dt == torch.bfloat16 else torch.int32)))
    # RopeTables: one table a kind, never replaced (graphs hold its address), no complex table kept
    cfg = Cfg(vocab=1, dim=1, inter=1, n_layers=1, n_heads=1, head_dim=512, rope_dim=RD, q_rank=1, o_rank=1, o_groups=1,
              window=128, eps=1e-20, n_routed=1, topk=1, route_scale=1.0, swiglu_limit=10.0, compress_ratios=[0, 2],
              kv_sources=[], index_sources=[], idx_heads=1, idx_dim=128, idx_topk=512, cand_source=-1, cand_blocks=0,
              cand_block=0, hc=4, hc_iters=20, hc_eps=1e-6, rope_theta=10000.0, compress_theta=160000.0,
              rope_factor=16.0, orig_len=65536, beta_fast=32, beta_slow=1, engram_layers=[], engram_rows=[],
              engram_ngram=1, engram_vocab=0, engram_heads=0, engram_dim=0, engram_pad=2, engram_cvocab=0)
    cache_before = freqs_cis.cache_info().currsize
    rt = RopeTables(cfg, device=dev)
    t1 = rt.cs(True, 1048590)
    t2 = rt.cs(True, 262158)                       # a shorter need: the table already made
    t3 = rt.cs(False, 1048590)
    ok["tables/one_per_kind_reused"] = t1 is t2 and t3 is not t1 and len(rt.tables) == 2
    ref = rope_cs(RD, RopeTables.rows(1048590), *KINDS["compressed"], device=dev)
    ok["tables/same_values_as_rope_cs"] = same(t1[0], ref[0]) and same(t1[1], ref[1])
    del ref
    t4 = rt.cs(True, 1048590 + 2 * ROPE_ALIGN)    # a longer need: a new table, the old one kept
    ok["tables/longer_need_adds_keeps_old"] = t4 is not t1 and rt.tables[(True, RopeTables.rows(1048590))] is t1
    ok["tables/no_complex_table_cached"] = freqs_cis.cache_info().currsize == cache_before
    # bytes: the old tables (2^k rows past rope_cap, complex64 + fp32 cos and sin, two kinds) vs these
    mem = {}
    for ctx in CONTEXTS:
        cap = ctx + EXTRA
        p2 = 1 << max(12, (cap - 1).bit_length())
        new = RopeTables.rows(cap)
        mem[ctx] = {"old_rows": p2, "old_MiB": round(2 * p2 * (RD // 2) * (8 + 4 + 4) / 2**20, 1),
                    "new_rows": new, "new_MiB": round(2 * new * (RD // 2) * 8 / 2**20, 1)}
    out["info"]["bytes_two_kinds"] = mem
    rt2 = RopeTables(cfg, device=dev)
    rt2.cs(True, 1048590)
    rt2.cs(False, 1048590)
    ok["tables/nbytes_1M"] = rt2.nbytes() == 2 * RopeTables.rows(1048590) * (RD // 2) * 4 * 2
    out["info"]["torch"] = torch.__version__
    out["info"]["threads"] = torch.get_num_threads()
    out["info"]["block"] = ROPE_BLOCK
    out["passed"] = all(ok.values())
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0 if out["passed"] else 1


def _freq_vector(args) -> torch.Tensor:
    """freqs_cis's fp32 frequencies (its formula, for the fp64 check of the rows past the compat block)."""

    orig_len, base, factor, beta_fast, beta_slow = args
    dim = RD
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if orig_len > 0:
        def corr(rot: float) -> float:
            return dim * math.log(orig_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


if __name__ == "__main__":
    sys.exit(main())
