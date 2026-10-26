"""Ling-3.1-flash memory a rank: what the engine's startup admits at TP2 and TP4 for 128K, 262K and 1M windows,
and what is left free at peak.

The cache, state and buffer bytes are the engine's own estimate (families/bailing_v3/memory.py: what admission prices,
held to the allocations by tests/test_bailing_13_memory.py). Weights a rank are the EXL3 recipes priced from the
shapes (``prior_weights``: EXL3 trellises at bits/8 a weight plus fp16 suh/svh vectors, BF16 where the recipe keeps
it, widths split in whole 128-column blocks as split.unit hands them out), or the engine's own estimate from a
converted checkpoint's headers (--checkpoint). Admission is tensorfold.cuda.capacity's, run as the engine runs it:

    budget  = MemAvailable - reserve       reserve = max(4 GiB, MemTotal / 10), or TENSORFOLD_MEMORY_RESERVE_GIB
    refused   when resident + staging > budget     staging = 3 x the largest layer's (or tensor's) bytes
    window  = the largest w with resident + estimate(w + 8 slots) <= budget, at most the native 262,144

The estimate leaves out the CUDA context, NCCL, the CUDA graphs and the allocator's slack (3.5-4.8 GiB on the
GLM-5.3-Flash TP2 lane, docs/recipes/glm-5.3-flash.md): they come out of the reserve, so a window that fills the
budget leaves reserve - outside free at peak.

1M needs YaRN (config.json has rope_scaling null, bailing_v3/config.py refuses rope_scaling, and admission refuses
an explicit --context past native): those rows say what the memory would hold if it had it.

Usage: python tools/ling_memory_model.py [--mem-available 116.9] [--mem-total 119.7] [--reserve 7]
       [--outside 3.5,4.8] [--checkpoint DIR --world 2 --rank 0]
Torch-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorfold.cuda.capacity import GIB, Weights, config, estimate_weights, make_plan  # noqa: E402
from tensorfold.cuda.geometry import mla_geometry, share  # noqa: E402
from tensorfold.families.bailing_v3 import memory  # noqa: E402
from tensorfold.families.bailing_v3.config import Config  # noqa: E402

CONFIG = REPO / "tests" / "fixtures" / "ling31" / "config.json"     # the bank's config.json (its FP8 quantization)
ROWS = 8                                                             # the engine's MAX_ROWS: slots past the window
WINDOWS = {"128K": 131072, "262K": 262144, "1M": 1048576}
# exl3/recipes/ling-tp{2,4}.json: bits a weight; kv_b, the head gates, KDA's b_proj and the routers stay BF16
RECIPES = {2: dict(kda=4, mla=6, dense=4, shared=5, expert=3, head=6, mtp=4),
           4: dict(kda=5, mla=6, dense=4, shared=5, expert=3, head=6, mtp=4)}


def ling_text() -> dict:
    text = json.loads(CONFIG.read_text())
    text["_quantization"] = {"quant_method": "exl3"}
    return text


def prior_weights(text: dict, world: int, rank: int = 0, *, mtp: bool = True,
                  recipe: dict | None = None) -> tuple[Weights, dict[str, int]]:
    """Resident and staging bytes a rank from the recipe, the way estimate_weights adds a converted checkpoint up:
    every tensor's rank share, staging 3 x the largest group (a layer's tensors, or a top-level tensor)."""

    c = Config.from_dict(text)
    bits = recipe or RECIPES[world]
    D = c.hidden_size

    def cols(n: int) -> int:                 # an output width split in whole 128-column blocks (split.unit)
        return share(n // 128, world, rank) * 128 if n % 128 == 0 and n // 128 >= world else share(n, world, rank)

    def x3(k: int, n: int, b: int, axis: str) -> int:
        """An EXL3 linear on this rank: trellis b/8 bytes a weight, suh (k) and svh (n) fp16; split along the
        output ("col"), the input ("row") or whole ("rep")."""
        if axis == "col":
            n = cols(n)
        elif axis == "row":
            k = cols(k)
        return k * n * b // 8 + (k + n) * 2

    heads = share(c.num_attention_heads, world, rank)
    lin = share(c.lin_heads, world, rank)
    shared = (c.shared_intermediate_size or c.moe_intermediate_size) * (c.n_shared_experts or 1)
    kv_a = -(-(c.kv_lora_rank + c.qk_rope_head_dim) // 128) * 128

    def mla(b: int) -> int:
        qk = c.qk_nope_head_dim + c.qk_rope_head_dim
        return (x3(D, c.q_lora_rank, b, "rep") + x3(c.q_lora_rank, c.num_attention_heads * qk, b, "col")
                + x3(D, kv_a, b, "rep") + x3(c.num_attention_heads * c.v_head_dim, D, b, "row")
                + heads * (c.qk_nope_head_dim + c.v_head_dim) * c.kv_lora_rank * 2      # kv_b, BF16
                + heads * D * 2                                                           # the head gate, BF16
                + (c.q_lora_rank + c.kv_lora_rank) * 2)                                   # q_a and kv_a norms

    def kda(b: int) -> int:
        w = c.lin_heads * c.lin_dim
        return (x3(D, 3 * w, b, "col") + x3(D, w, b, "col") + x3(D, w, b, "col") + x3(w, D, b, "row")
                + lin * D * 2                                       # b_proj, BF16
                + 3 * lin * c.lin_dim * c.conv_kernel * 2           # the q, k, v short convs
                + lin * c.lin_dim * 4 + lin * 4                     # dt_bias, A_log (fp32)
                + c.lin_dim * 2)                                    # o_norm

    def moe(expert_bits: int, shared_bits: int) -> int:
        I = c.moe_intermediate_size
        one = 2 * x3(D, I, expert_bits, "col") + x3(I, D, expert_bits, "row")
        out = c.n_routed_experts * one + c.n_routed_experts * D * 2 + c.n_routed_experts * 4     # router, bias
        return out + 2 * x3(D, shared, shared_bits, "col") + x3(shared, D, shared_bits, "row")

    groups: dict[str, int] = {}
    for i, kind in enumerate(c.layer_kinds):
        att = kda(bits["kda"]) if kind == "kda" else mla(bits["mla"])
        if c.mlp_layer_types[i] == "dense":
            ffn = 2 * x3(D, c.intermediate_size, bits["dense"], "col") + x3(c.intermediate_size, D, bits["dense"], "row")
        else:
            ffn = moe(bits["expert"], bits["shared"])
        groups[f"layers.{i}"] = att + ffn + 2 * D * 2
    if mtp and c.num_nextn_predict_layers:
        m = bits["mtp"]
        groups[f"layers.{c.num_hidden_layers}"] = (mla(m) + moe(m, m) + x3(2 * D, D, m, "rep")    # eh_proj
                                                   + 5 * D * 2)        # enorm, hnorm, final and the block's norms
    vocab = -(-c.vocab_size // 128)
    groups["embed"] = c.vocab_size * D * 2                              # BF16, every rank
    groups["head"] = x3(D, -(-vocab // world) * 128, bits["head"], "rep")
    groups["norm"] = D * 2
    return Weights(sum(groups.values()), 3 * max(groups.values())), groups


def checkpoint_weights(folder: Path, world: int, rank: int, *, mtp: bool) -> Weights:
    """The engine's own estimate (bailing_v3.cuda.engine): split_weights over the checkpoint's headers."""

    from tensorfold.cuda.geometry import split_weights
    from tensorfold.families.glm5_next.cuda.split import rule, unit

    transform = split_weights(rule, world, unit=unit)
    if not mtp:
        prefix = f"model.layers.{Config.from_dict(config(folder)).num_hidden_layers}."
        transform = (lambda t: lambda name, info: (0, 0) if name.startswith(prefix) else t(name, info))(transform)
    return estimate_weights(folder, transform, rank=rank)


def admitted(text: dict, world: int, rank: int, mtp: bool, weights: Weights, budget: int, native: int,
             requested: int | None) -> int:
    """The window the engine allocates (0: refused), as capacity.make_plan decides it."""

    geo = memory.geometry(text, world, rank, mtp=mtp, rows=ROWS)
    plan = make_plan(native, requested, requested is not None, budget, weights, geo)
    return plan.fitting


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mem-available", type=float, default=116.9,
                    help="GiB MemAvailable on an idle node (DSV41's startup lines: budget 101.6-105.3 GiB at the "
                         "default reserve, so 113.6-117.3; 116.9 is the 104.9 typical)")
    ap.add_argument("--mem-total", type=float, default=119.7, help="GiB MemTotal (the default reserve is a tenth)")
    ap.add_argument("--reserve", type=float, action="append",
                    help="GiB TENSORFOLD_MEMORY_RESERVE_GIB to show beside the default (default: 7)")
    ap.add_argument("--outside", default="3.5,4.8", help="GiB outside the estimate: low,high")
    ap.add_argument("--kept", type=float, default=3.0, help="GiB TF_LING_CACHE_GIB (kept prompts, what the window leaves)")
    ap.add_argument("--checkpoint", type=Path, help="a converted checkpoint: weights from its headers (the engine's)")
    ap.add_argument("--world", type=int, help="with --checkpoint: TF_TP_WORLD")
    ap.add_argument("--rank", type=int, default=0, help="with --checkpoint: the rank whose files to read")
    args = ap.parse_args(argv)
    lo, hi = (float(x) for x in args.outside.split(","))
    avail, default_reserve = args.mem_available * GIB, max(4 * GIB, int(args.mem_total * GIB) // 10)
    reserves = [("default", default_reserve)] + [(f"{g:g} GiB", int(g * GIB)) for g in (args.reserve or [7.0])]
    if args.checkpoint:
        text = config(args.checkpoint)
        cases = [(args.world or 2, args.rank, True), (args.world or 2, args.rank, False)]
    else:
        text = ling_text()
        cases = [(2, 0, True), (2, 0, False), (4, 0, True), (4, 2, True)]
    print(f"MemAvailable {args.mem_available:.1f} GiB, MemTotal {args.mem_total:.1f} GiB (default reserve "
          f"{default_reserve / GIB:.1f}); outside the estimate {lo:g}-{hi:g} GiB; kept prompts up to {args.kept:g} GiB")
    for world, rank, mtp in cases:
        if args.checkpoint:
            weights, groups = checkpoint_weights(args.checkpoint, world, rank, mtp=mtp), None
        else:
            weights, groups = prior_weights(text, world, rank, mtp=mtp)
        geo = memory.geometry(text, world, rank, mtp=mtp, rows=ROWS)
        r = memory.Rank.of(text, world, rank, mtp=mtp)
        glm = mla_geometry(text, world, ROWS, minimum_slots=2560, latent=True, mtp=mtp)
        big = max(groups, key=groups.get) if groups else "largest group"
        print(f"\nTP{world} rank {rank}, MTP {'on' if mtp else 'off'}: weights {weights.resident / GIB:.2f} GiB "
              f"resident, {weights.staging / GIB:.2f} GiB staging (3 x {big})")
        for name, w in WINDOWS.items():
            p = memory.parts(r, w + ROWS, rows=ROWS)
            parts = ", ".join(f"{k} {v / GIB:.2f}" for k, v in p.items())
            print(f"  {name:>4} window: estimate {geo.needed(w) / GIB:.2f} GiB ({parts}); GLM-shaped "
                  f"{glm.needed(w) / GIB:.2f}; weights + estimate {(weights.resident + geo.needed(w)) / GIB:.2f}")
        for label, reserve in reserves:
            budget = int(avail) - reserve
            for name, native, requested in (("native", 262144, None), ("1M with YaRN", 1048576, 1048576)):
                window = admitted(text, world, rank, mtp, weights, budget, native, requested)
                head = f"  reserve {label:>8} (budget {budget / GIB:.2f}), {name}:"
                if weights.resident + weights.staging > budget:
                    print(f"{head} refused: weights + staging {(weights.resident + weights.staging) / GIB:.2f} GiB "
                          f"pass the budget")
                    break
                if not window:
                    print(f"{head} refused: no window fits beside the weights")
                    continue
                need = geo.needed(window)
                kept = max(0, min(int(args.kept * GIB), budget - weights.resident - max(weights.staging, need)))
                free = avail - weights.resident - need - kept
                print(f"{head} window {window:,} ({need / GIB:.2f} GiB), kept prompts {kept / GIB:.2f}, free at "
                      f"peak {(free - hi * GIB) / GIB:.1f}-{(free - lo * GIB) / GIB:.1f} GiB")
        staging_room = avail - weights.resident - weights.staging
        full = avail - weights.resident - geo.needed(262144)
        print(f"  admits at all with a reserve up to {staging_room / GIB:.2f} GiB; the full 262,144 window with one up "
              f"to {full / GIB:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
