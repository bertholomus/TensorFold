"""Ling-3.1-flash's CUDA memory on one rank: what bailing_v3.cuda allocates, as a function of the cache slots.

The startup admission (``tensorfold.cuda.capacity.admit``) asks a Geometry for the bytes the caches, recurrent
states and buffers take at a slot capacity, then fits the largest window beside the weights. GLM's ``mla_geometry``
prices another model when handed Ling's config: 18 attention layers from its default interval (Ling has 9 and the
MTP layer), a DSA indexer cache and selection scratch Ling does not have, four residual streams, the latent cache
without its rope key, and GLM's prompt attention scratch. This prices what the engine allocates:

- ``State``: two fp32 KDA recurrent states a layer (cur and 1 - cur), the conv windows, the decode window's
  projections and replay scratch, and every latent layer's cache (kv_lora + qk_rope bf16 a token: 9 layers and the
  MTP layer's);
- three ``Buffers`` (the decode window's, the MTP head's, a prompt chunk's): row arrays and the latent attention's
  chunk partials (their rows capped at TF_LING_LATENT_MB a buffer);
- the EXL3 scratch: the routed experts' (``exl3.experts.Scratch``) and the EXL3 linears' shared one
  (``glm_moe_dsa.cuda.x3.X3Scratch``), the prompt chunk's grown on first use (decoded weights, the unpack cache).

With TF_LING_SHARED_SCRATCH=1 (``SHARED``) the three Buffers hold their call-local scratch once (forward.Shared): one
set of chunk partials for the widest Buffers' rows, and one X3Scratch core (rotated inputs, K-split partials,
staging), not three. The same kernels on the same rows: no output bit changes.

Torch-free, so tools/ling_memory_model.py reads it without torch. tests/test_bailing_13_memory.py holds the
State and Buffers terms to their allocations byte for byte and each EXL3 rule to the code it mirrors. Outside the
estimate (the admission reserve carries them): the CUDA context, NCCL, the graph pool and the allocator's slack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tensorfold.cuda.capacity import Geometry
from tensorfold.cuda.geometry import share

CHUNK = 512                 # keys a latent attention chunk program reads (glm5_next.cuda.latent.CHUNK)
DK = DV = 128               # KDA key and value head widths (the chain kernel's)
X3_ROWS = 128               # most rows one EXL3 linear call takes (glm_moe_dsa.cuda.x3.ROWS)
# the dense latent attention's chunk partials: MiB a Buffers may hold (rows beyond what fits run as row blocks, each
# at its first row's position: the same bits a row). Dense attention at every length makes this grow with capacity:
# 262,144 tokens is 512 chunks, 33.7 MB of partials a row at 32 heads
LATENT_MB = int(os.environ.get("TF_LING_LATENT_MB") or 1024)
# decoded EXL3 weights a prompt chunk's workspace keeps (glm_moe_dsa.cuda.x3.UNPACK_CACHE_MB, the same variable)
UNPACK_CACHE_MB = int(os.environ.get("TF_GLM_UNPACK_CACHE_MB") or 384)
# the decode, MTP and prompt Buffers share their call-local scratch (forward.Shared): one CUDA stream runs their calls
# one after another. Off by default; at 262K with LATENT_MB 270 it saves 0.66 GiB a rank (partials 0.50, EXL3 0.16)
SHARED = os.environ.get("TF_LING_SHARED_SCRATCH", "0") == "1"


def chunks_for(length: int) -> int:
    return -(-int(length) // CHUNK)


def latent_rows(rows: int, heads: int, chunks: int, lw: int) -> int:
    """Rows of chunk partials (po, pm, pl) one attention call takes within LATENT_MB."""

    per_row = chunks * heads * (lw + 2) * 4
    return max(1, min(rows, (LATENT_MB << 20) // max(1, per_row)))


@dataclass(frozen=True)
class Rank:
    """The shapes one rank's buffers are made of (``bailing_v3.cuda.weights`` as loaded)."""

    world: int
    hidden: int
    heads: int                  # this rank's attention heads
    lin_heads: int              # this rank's KDA heads
    q_lora: int
    qk_nope: int
    qk_rope: int
    kv_lora: int
    v_dim: int
    conv: int
    kda_layers: int
    mla_layers: int
    mtp: bool                   # the MTP layer is loaded (its latent cache, its Buffers)
    dense_width: int            # this rank's dense-MLP width, in whole 128-column blocks
    shared_width: int           # this rank's shared-expert width
    expert_width: int           # this rank's routed-expert width
    experts: int
    top_k: int
    vocab_rows: int             # this rank's logits columns (the head's padded slice)
    exl3: bool                  # EXL3 groups (an EXL3 conversion) rather than BF16 matrices

    @property
    def qk_dim(self) -> int:
        return self.qk_nope + self.qk_rope

    @classmethod
    def of(cls, text: dict[str, Any], world: int, rank: int = 0, *, mtp: bool = True) -> "Rank":
        """``text``: the checkpoint's config.json (its text part); ``mtp``: whether the engine loads the MTP layer."""

        from tensorfold.families.bailing_v3.config import Config

        c = Config.from_dict(text)
        quant = text.get("_quantization") or text.get("quantization_config") or text.get("quantization") or {}
        exl3 = quant.get("quant_method") == "exl3"

        def blocks(width: int) -> int:
            """A width as the split hands it out: whole 128-column EXL3 blocks where they divide (split.unit)."""
            return share(width // 128, world, rank) * 128 if exl3 and width % 128 == 0 and width // 128 >= world \
                else share(width, world, rank)

        heads, lin_heads = share(c.num_attention_heads, world, rank), share(c.lin_heads, world, rank)
        shared = (c.shared_intermediate_size or c.moe_intermediate_size) * c.n_shared_experts
        vocab_blocks = -(-c.vocab_size // 128)
        vocab_rows = -(-vocab_blocks // world) * 128 if exl3 else share(c.vocab_size, world, rank)
        return cls(world=world, hidden=c.hidden_size, heads=heads, lin_heads=lin_heads, q_lora=c.q_lora_rank,
                   qk_nope=c.qk_nope_head_dim, qk_rope=c.qk_rope_head_dim, kv_lora=c.kv_lora_rank,
                   v_dim=c.v_head_dim, conv=c.conv_kernel, kda_layers=sum(k == "kda" for k in c.layer_kinds),
                   mla_layers=sum(k != "kda" for k in c.layer_kinds),
                   mtp=bool(mtp and c.num_nextn_predict_layers), dense_width=blocks(c.intermediate_size),
                   shared_width=blocks(shared) if shared else 0, expert_width=blocks(c.moe_intermediate_size),
                   experts=c.n_routed_experts, top_k=c.num_experts_per_tok, vocab_rows=vocab_rows, exl3=exl3)


# -- State and Buffers (forward.State, forward.Buffers) -------------------------------------------------------------

def state_fixed(r: Rank, rows: int) -> int:
    """forward.State without its latent caches: KDA conv windows, the two recurrent states, the window's
    projections and replay scratch, the device positions."""

    n, LL = r.kda_layers, r.lin_heads
    width = 3 * LL * DK + LL
    b = 4 + 4                                                   # pos_dev, mtp_pos_dev
    b += n * (r.conv - 1) * 3 * LL * DK * 2                     # conv
    b += 2 * n * LL * DV * DK * 4                               # rec: cur and 1 - cur
    b += n * rows * width * 2                                   # proj
    if n:
        b += n * rows * LL * (DV * 2 + DK * 4 + DV * 2 + DK * 4 + 4)    # KDAScratchSet: out, k, v, g, b
    return b


def cache_per_slot(r: Rank) -> int:
    """Latent cache bytes a token: kv_lora and the rope key, bf16, every MLA layer and the MTP layer's. Replicated:
    every rank keeps the whole latent (it is shared by all heads), so TP4 holds as much as TP2."""

    return (r.mla_layers + int(r.mtp)) * (r.kv_lora + r.qk_rope) * 2


def partials(r: Rank, rows: int, capacity: int) -> int:
    """LatentScratch's chunk partials (po, pm, pl) of a Buffers made for ``capacity`` slots."""

    chunks = chunks_for(capacity)
    part = latent_rows(rows, r.heads, chunks, r.kv_lora)
    return chunks * part * r.heads * (r.kv_lora + 2) * 4


def _b16_split_k(n: int, k: int) -> int:
    """glm5_next.cuda.qmm.b16_split_k: K slices of a BF16 matmul (B16_BN = B16_BK = 64, SPLIT_TARGET 192)."""

    tiles, steps, sk = -(-n // 64), k // 64, 1
    while sk < 8 and tiles * sk < 192 and steps % (sk * 2) == 0 and steps // (sk * 2) >= 4:
        sk *= 2
    return sk


def buffer_rows(r: Rank, rows: int, prefill: bool) -> int:
    """forward.Buffers without its chunk partials and EXL3 scratch: every row array, the rope table, the latent
    queries and outputs."""

    D, HL, LL = r.hidden, r.heads, r.lin_heads
    half = r.qk_rope // 2
    b = rows * 4 * 2                                            # ids, ids_host (pinned: the same RAM on a GB10)
    b += rows * (4 + 4 + half * 4 * 2 + half * 2 * 2) + half * 4    # rope.Table: step, pos, phase, f32, cos, sin; inv
    b += rows * D * 2 * 2                                       # x, normed
    b += rows * max(LL * DV, 1) * 2                             # stage
    if prefill:
        b += rows * (3 * LL * DK + LL) * 2                      # kproj
        b += rows * LL * (DV * 2 + DK * 4 + DV * 2 + DK * 4 + 4)    # KDAScratch
    b += rows * LL * (DK + DV) * 2                              # ka, kg
    b += rows * r.q_lora * 2                                    # qr
    b += rows * HL * (r.qk_dim + r.qk_nope + r.qk_rope) * 2     # q, qn, qp
    b += rows * (r.kv_lora + r.qk_rope) * 2                     # lat
    b += rows * HL * r.v_dim * 2                                # vn
    b += rows * HL * (2 + 4 + 2)                                # gl, gf, gb
    b += 2 * rows * HL * r.kv_lora * 2 + 4                      # LatentScratch's qa, ol and its dummy
    dl, sl, slots = r.dense_width, r.shared_width, r.top_k + 1
    b += rows * (2 * dl * 2 + dl * 2 + max(dl // 64, 1) * 4)   # gu, act, xs_act
    b += rows * (2 * r.experts * 4 + 2 * slots * 4)             # mlog, msc, pick, wts
    b += rows * (2 * sl * 2 + sl * 2 + max(sl // 64, 1) * 4 + D * 4)   # sgu, sact, sxs, sy
    b += rows * D * 4 + r.world * rows * D * 4                  # part, gath
    # sk: split-K partials of the BF16 matmuls (the KDA b_proj and the MLA head gate stay BF16 in every conversion)
    b += rows * max([_b16_split_k(n, D) * n for n in (LL, HL) if _b16_split_k(n, D) > 1], default=1) * 4
    b += rows * D * 2                                           # fnormed
    b += (1 if prefill else rows) * r.vocab_rows * 2            # logits
    b += rows * D * 2 * 3 + rows * 2 * D * 2                    # hin, me, mx, mcat
    return b


# -- EXL3 scratch ---------------------------------------------------------------------------------------------------

def x3_plan(k: int, n: int, blocks: int = 192, min_tiles: int = 8) -> tuple[int, int]:
    """tensorfold.cuda.exl3.linear.plan: (K splits, warps a program) of a K x N EXL3 linear."""

    kt, nb = k // 16, n // 128
    sk, wk = (1, 8) if nb >= 64 else (1, 4)
    if nb < 64:
        while nb * sk < blocks and sk < 64:
            sk *= 2
    while (wk > 4 or sk > 1) and (kt % (sk * wk) or kt // (sk * wk) < min_tiles):
        if wk > 4:
            wk = 4
        elif sk > 1:
            sk //= 2
        else:
            break
    return sk, wk


def x3_linears(r: Rank) -> list[tuple[int, int]]:
    """(k, n) of the EXL3 linears a rank loads (weights.load's x3 users): KDA qkv, f, g, o; MLA q_a, q_b, kv_a,
    dense; the dense MLPs' and shared experts' gate, up, down; the MTP layer's eh_proj; the head, last."""

    D, HL, LL = r.hidden, r.heads, r.lin_heads
    out = [(D, 3 * LL * DK), (D, LL * DK), (D, LL * DV), (LL * DV, D)] if r.kda_layers else []
    if r.mla_layers or r.mtp:
        # kv_a's stored width: kv_lora + qk_rope in whole 128-column blocks (the loader crops it)
        out += [(D, r.q_lora), (r.q_lora, HL * r.qk_dim), (D, -(-(r.kv_lora + r.qk_rope) // 128) * 128),
                (HL * r.v_dim, D)]
    if r.dense_width:
        out += [(D, r.dense_width), (r.dense_width, D)]
    if r.shared_width:
        out += [(D, r.shared_width), (r.shared_width, D)]
    if r.mtp:
        out.append((2 * D, D))
    out.append((D, r.vocab_rows))
    return out


def x3_scratch(linears: list[tuple[int, int]], *, prefill: bool, rows: int = 0) -> int:
    """X3Scratch for EXL3 linears of (k, n): fp16 rotated inputs, the fp32 K-split partials, fp32 and bf16 staging;
    with ``prefill``, what a prompt chunk of ``rows`` rows grows on first use (exl3.prefill.Workspace: one decoded
    W and one rotated input, the unpack cache's decoded weights, the Hadamard block). The head (last in ``linears``)
    reads a prompt chunk's last row only, through the decode path."""

    xh = max([X3_ROWS * k for k, _ in linears], default=1)
    split = [x3_plan(k, n)[0] * X3_ROWS * n for k, n in linears if x3_plan(k, n)[0] > 1]
    z = max(split, default=1)
    tmp = max([X3_ROWS * n for _, n in linears], default=1)
    b = xh * 2 + z * 4 + tmp * (4 + 2)
    if prefill:
        prompt = linears[:-1] or linears
        b += max(k * n for k, n in prompt) * 2 + rows * max(k for k, _ in prompt) * 2 + 128 * 128 * 2
        b += UNPACK_CACHE_MB << 20
    return b


def expert_tiles(K: int, N: int, gateup: bool) -> tuple[int, int, int, int]:
    """tensorfold.cuda.exl3.experts.default_config: the tile setting of a K -> N expert projection."""

    cands = [(8, 4, 4, 1) if gateup else (8, 4, 1, 1), (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
    for nt, w, sk, pf in cands:
        if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
            return nt, w, sk, pf
    raise ValueError(f"no tile setting divides K={K}, N={N}")


def expert_scratch(D: int, I: int, count: int, rows: int, slots: int, *, prompt: bool) -> int:
    """exl3.experts.Scratch for ``rows`` x ``slots`` pairs of D x I experts (``count`` of them)."""

    P = rows * slots
    gu, dn = expert_tiles(D, I, True)[2], expert_tiles(I, D, False)[2]
    b = P * I * 2 + max(2 * gu * I, dn * D) * P * 4 + 4         # xd, z, no_y
    if not prompt:
        b += 2 * P * D * 2 + P * D * 4                          # xg, xu, y
    maxu = min(P, count)
    b += maxu * 4 + 4 + count * 4 + maxu * (-(-rows // 128) * 128) * 4    # ids, count, counts, members_buf
    return b


# -- the geometry ---------------------------------------------------------------------------------------------------

def parts(r: Rank, capacity: int, *, rows: int = 8, prefill_rows: int = 2048,
          shared: bool | None = None) -> dict[str, int]:
    """Named bytes at ``capacity`` slots: the engine's State, its decode, MTP and prompt Buffers, and their EXL3
    scratch; ``shared`` (None: SHARED): the Buffers' call-local scratch held once (forward.Shared)."""

    shared = SHARED if shared is None else bool(shared)
    decode = 2 if r.mtp else 1                                  # the decode window's Buffers and the MTP head's
    if shared:                                                  # one set, for the widest Buffers' rows
        chunk_partials = partials(r, max(rows, prefill_rows), capacity)
    else:
        chunk_partials = decode * partials(r, rows, capacity) + partials(r, prefill_rows, capacity)
    out = {"latent cache": cache_per_slot(r) * capacity,
           "KDA state": state_fixed(r, rows),
           "chunk partials": chunk_partials,
           "decode buffers": decode * buffer_rows(r, rows, prefill=False),
           "prompt buffers": buffer_rows(r, prefill_rows, prefill=True)}
    if r.exl3:
        lin = x3_linears(r)
        slots = r.top_k + 1
        # the decode and MTP Buffers' X3Scratch: its own core each, or the prompt chunk's (the same sizes: shared)
        core = 0 if shared else decode * x3_scratch(lin, prefill=False)
        out["EXL3 scratch"] = (core + decode * expert_scratch(r.hidden, r.expert_width, r.experts, rows, slots,
                                                              prompt=False)
                               + x3_scratch(lin, prefill=True, rows=prefill_rows)
                               + expert_scratch(r.hidden, r.expert_width, r.experts, prefill_rows, slots, prompt=True))
    return out


def geometry(text: dict[str, Any], world: int, rank: int = 0, *, mtp: bool = True, rows: int = 8,
             prefill_rows: int = 2048, minimum_slots: int = 2560, shared: bool | None = None) -> Geometry:
    """The admission's Geometry for Ling on ``rank`` of ``world``: bytes at a slot capacity; ``rows`` (the widest
    verify window) is also the slots kept past the window."""

    r = Rank.of(text, world, rank, mtp=mtp)
    return Geometry(lambda capacity: sum(parts(r, capacity, rows=rows, prefill_rows=prefill_rows,
                                               shared=shared).values()), rows, minimum_slots)
