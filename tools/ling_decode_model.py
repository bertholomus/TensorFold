"""Ling-3.1-flash decode priors: bytes a rank, round times and tokens a second for MTP chains and block drafters.

A model to judge node results against, not a measurement. Every byte count comes from the checkpoint's shapes and the
EXL3 recipes (ling-tp2 and ling-tp4: experts 3, KDA 4 or 5, MLA 6, dense 4, shared 5, head 6, MTP 4;
kv_b and the head gates BF16, the router BF16) and the split this tree serves (split.unit: experts and shared expert
640/640 at TP2, 384/384/256/256 at TP4, so rank 0 reads 0.3 of them). Times are bytes / bandwidth + glue; the glue
constants are priors. Once an engine has started, its own line

    [tensorfold] verify costs (ms): v1 v2 ... v8; MTP draft m (+s a chained draft, +r a row)

replaces the bandwidth model: pass ``--verify-ms v1,...,v8 --mtp-ms m,s,r`` and only the acceptance stays a prior.
Measured MTP acceptance replaces that too: ``--mtp-accept a1,a2,a3``. verify(8)/verify(1) against the birthday and
correlated priors says how much consecutive tokens share experts.

Acceptance is tokens committed a round (the bonus token included). MTP chains: draft j is accepted with probability
a_j given drafts 1..j-1 were, a_j = a_1 * decay^(j-1) (the head is trained one step ahead; steps 2-3 extend it).
Block drafters (DFlash/DSpark): a constant per-position rate alpha. E(k) = 1 + sum_j prod_{i<=j} a_i.

Depth: a forward p tokens into its cache also reads every latent layer's cache once (1,152 B a token a layer,
the rows and head blocks of a chunk run together and share it through L2; replicated, so TP4 reads as much as TP2) or
does the score and value math, whichever is longer at --tflops; writes and merges the chunk partials (2,056 B a row,
head and chunk, written once and read once); and reads and writes every KDA layer's fp32 state (264 MB at TP2, 132 MB
at TP4), which a round that keeps fewer rows than it verified does once more (commit's replay). The partials walk
every chunk of the cache slots on a CUDA graph replay, as the engine does today (--merge window), or only the chunks
that hold keys, as eager does and the merge fix would (filled). The startup verify costs ran at 64 tokens with the
window's walk: given them, only what depth and the walk add is modelled; given the same costs at a short window too
(--short-ms, an engine started with --context 8192), what a chunk of the walk costs is measured, not priced.

Usage: python tools/ling_decode_model.py [--bw 205] [--world 2 --world 4] [--slots 2:229930,4:262152]
       [--depth 4096,32768,131072,262144] [--tflops 25] [--verify-ms ... [--short-ms ...]] [--mtp-ms ...]
       [--mtp-accept ...]
Stdlib only (no numpy needed).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

GB = 1e9

# -- shapes (config.json of Ling-3.1-flash, HF 7b864daa) ---------------------------------------------------------
D, VOCAB = 4096, 157184
KDA_LAYERS, MLA_LAYERS, DENSE_LAYERS, MOE_LAYERS = 63, 9, 4, 68
EXPERTS, TOP_K, MOE_W, SHARED_W, DENSE_W = 512, 8, 1280, 1280, 10240
KDA_PROJ = 5 * 8192 * D + D * 8192                       # q, k, v, g, f + o, one layer
KDA_BITS_BYTES = 64 * D * 2 + 3 * 8192 * 4 * 2 + 8192 * 4 + 64 * 4 + 128 * 2   # b_proj, convs, dt_bias, A_log, o_norm
MLA_REP = 1536 * D + 576 * D                             # q_a, kv_a: every rank (the shared latent and q-lora)
MLA_SPLIT = 12288 * 1536 + D * 8192                      # q_b, dense: by heads
KV_B = 16384 * 512                                       # BF16, by heads
HEAD_GATE = 64 * D                                       # BF16, by heads
EXPERT = 3 * MOE_W * D                                   # one routed expert (gate, up, down)
SHARED = 3 * SHARED_W * D
DENSE = 3 * DENSE_W * D
ROUTER = EXPERTS * D                                     # BF16, every rank
HEAD = VOCAB * D
EH_PROJ = D * 2 * D

RECIPES = {
    2: dict(kda=4, mla=6, dense=4, shared=5, expert=3, head=6, mtp=4),     # exl3/recipes/ling-tp2.json
    4: dict(kda=5, mla=6, dense=4, shared=5, expert=3, head=6, mtp=4),     # exl3/recipes/ling-tp4.json
}
# rank 0's share of the expert widths (split.unit: whole 128-column blocks; 10 blocks a 1280-wide expert)
EXPERT_SHARE = {1: 1.0, 2: 0.5, 4: 0.3, 6: 0.2}

# -- context: what a forward reads besides the weights (families/bailing_v3/memory.py prices the same buffers) ------
HEADS = LIN_HEADS = 64                                   # MLA query heads and KDA heads, split by rank
CHUNK = 512                                              # keys a latent attention chunk (glm5_next.cuda.latent.CHUNK)
LATENT_ROW = (512 + 64) * 2                              # a token's latent and rope key, bf16
PARTIAL = (512 + 2) * 4                                  # a chunk's partial a row and head: o, m and l, fp32
KDA_STATE = 128 * 128 * 4                                # a KDA head's fp32 state
KDA_ROW = 128 * 2 + 128 * 4 + 128 * 2 + 128 * 4 + 4      # a row's replay scratch a head: out, k, v, g, beta
ATTN_FLOP = 2 * (512 + 64 + 512)                         # a head and key: the scores (latent and rope), the values
NATIVE, MAX_ROWS = 262144, 8                             # the window, and the slots kept past it (engine MAX_ROWS)
CAL_DEPTH = 64                                           # the startup calibration's prompt (engine._calibrate)
SHORT_SLOTS = 8192 + MAX_ROWS                            # an engine started with --context 8192
# effective TFLOPS of the dense latent attention's score and value math (prior: about half of GB10's BF16 tensor
# rate at the 2200 MHz cap for 16-head Triton tiles; the startup verify costs at depth replace it)
TFLOPS = 25.0


def chunks_for(n: float) -> int:
    return -(-math.ceil(n) // CHUNK)


def q(params: float, bits: float) -> float:
    return params * bits / 8


@dataclass
class Bytes:
    """Bytes rank 0 reads (the widest expert share): a forward's flat weights, one row's routed experts, one MTP step
    (the layer-72 block and the head) and one more MTP row's experts."""

    flat: float
    expert_row: float
    mtp_step: float
    mtp_row: float


def rank_bytes(world: int, recipe: dict | None = None) -> Bytes:
    r = recipe or RECIPES[world]
    s = EXPERT_SHARE[world]
    kda = q(KDA_PROJ, r["kda"]) * KDA_LAYERS / world + KDA_BITS_BYTES * KDA_LAYERS / world
    mla = MLA_LAYERS * (q(MLA_REP, r["mla"]) + q(MLA_SPLIT, r["mla"]) / world + (KV_B + HEAD_GATE) * 2 / world)
    dense = q(DENSE, r["dense"]) * DENSE_LAYERS / world
    shared = q(SHARED, r["shared"]) * MOE_LAYERS * s
    head = q(HEAD, r["head"]) / world
    router = ROUTER * 2 * MOE_LAYERS
    flat = kda + mla + dense + shared + head + router
    expert_row = q(EXPERT, r["expert"]) * TOP_K * MOE_LAYERS * s
    m = r["mtp"]
    mtp_row = q(EXPERT, m) * TOP_K * s
    mtp_step = (q(EH_PROJ, m) + q(MLA_REP, m) + q(MLA_SPLIT, m) / world + (KV_B + HEAD_GATE) * 2 / world
                + ROUTER * 2 + q(SHARED, m) * s + mtp_row + head)
    return Bytes(flat, expert_row, mtp_step, mtp_row)


def union_rows(R: int, model: str) -> float:
    """Expert rows' worth a window of R rows reads: independent top-8 picks of 512 ("birthday"), or the overlap DSV41
    measured on its windows (2.3x one row at 4 rows: 0.433 a further row, "correlated")."""

    if model == "birthday":
        return EXPERTS / TOP_K * (1 - (1 - TOP_K / EXPERTS) ** R)
    if model == "correlated":
        return 1 + 0.433 * (R - 1)
    raise ValueError(model)


@dataclass
class Glue:
    """Milliseconds that are not weight bytes (priors): a forward's collectives, launches and host sync; a row's
    sampling and per-row kernels; an MTP step; a block-drafter pass."""

    forward: float
    row: float
    mtp: float
    block: float


GLUE = {2: Glue(6.0, 0.3, 0.8, 1.2), 4: Glue(8.0, 0.3, 1.0, 1.5)}


@dataclass
class Drafter:
    """A DSpark/DFlash-class block drafter: layers of a qwen3 block (GQA) on the target's taps, drafting through the
    target's head; its linears held as 4-bit affine copies (0.5625 B a weight, as dflash2.Drafter does)."""

    layers: int = 5
    hidden: int = D
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    inter: int = 8192
    taps: int = 5
    bytes_per_weight: float = 0.5625
    ring_rows: int = 2176          # its context: a 2,048 sliding window in a ring (as dflash2's)

    def layer_params(self) -> int:
        qo = 2 * self.hidden * self.heads * self.head_dim
        kv = 2 * self.hidden * self.kv_heads * self.head_dim
        return qo + kv + 3 * self.hidden * self.inter

    def params(self) -> int:
        return self.layers * self.layer_params() + self.taps * self.hidden * self.hidden

    def pass_bytes(self, world: int, recipe: dict | None = None) -> float:
        """A block pass (every drafter layer, the head over the block's rows, each layer's attention over its ring)
        and the context update of the kept rows (fc, replicated as dflash2 holds it, and each layer's k/v)."""

        r = recipe or RECIPES[world]
        layers = self.layers * self.layer_params() * self.bytes_per_weight / world
        fc = self.taps * self.hidden * self.hidden * self.bytes_per_weight
        kv = self.layers * 2 * self.hidden * self.kv_heads * self.head_dim * self.bytes_per_weight / world
        ring = self.layers * 2 * self.kv_heads * self.head_dim * 2 * self.ring_rows / world
        return layers + q(HEAD, r["head"]) / world + fc + kv + ring


class Model:
    """Round times on rank 0 at ``depth`` tokens into a cache of ``slots`` (the startup line's cache slots)."""

    def __init__(self, world: int, bw: float, union: str, glue: Glue | None = None, verify_ms: list[float] | None = None,
                 mtp_ms: tuple[float, float, float] | None = None, drafter: Drafter | None = None, *,
                 slots: int = NATIVE + MAX_ROWS, depth: int = CAL_DEPTH, merge: str = "window",
                 tflops: float = TFLOPS, short_ms: list[float] | None = None, short_slots: int = SHORT_SLOTS) -> None:
        if merge not in ("window", "filled"):
            raise ValueError(merge)
        self.world, self.bw, self.union = world, bw * GB / 1e3, union          # bytes a millisecond
        self.b = rank_bytes(world)
        self.g = glue or GLUE[world]
        self.verify_ms, self.mtp_ms = verify_ms, mtp_ms
        self.drafter = drafter or Drafter()
        self.heads, self.lin_heads = HEADS // world, LIN_HEADS // world
        self.slots, self.depth, self.merge, self.tflops = slots, depth, merge, tflops
        self.per_chunk = None
        if verify_ms and short_ms:
            # the A/B: the same engine at two windows differs only in the chunks its partials walk
            walked = chunks_for(slots) - chunks_for(short_slots)
            if walked <= 0:
                raise ValueError("--short-ms must come from a window shorter than --slots")
            self.per_chunk = [(v - s) / walked for v, s in zip(verify_ms, short_ms)]

    # -- context terms ----------------------------------------------------------------------------------------------
    def walk(self, rows: float) -> int:
        """Chunks the partials walk: every chunk of the slots (a graph replay, today) or those that hold keys."""
        return chunks_for(self.slots) if self.merge == "window" else chunks_for(self.depth + rows)

    def partials_ms(self, rows: float, chunks: int, layers: int) -> float:
        if self.per_chunk is not None:                  # measured: interpolated between whole row counts
            lo = min(max(int(rows), 1), MAX_ROWS)
            hi = min(lo + 1, MAX_ROWS)
            per = self.per_chunk[lo - 1] + (self.per_chunk[hi - 1] - self.per_chunk[lo - 1]) * (rows - lo)
            return per * chunks * layers / MLA_LAYERS
        return layers * 2 * rows * self.heads * chunks * PARTIAL / self.bw

    def attention_ms(self, rows: float, depth: int, layers: int, chunks: int) -> float:
        """``layers`` latent layers' dense attention over depth + rows keys: the cache read once or the score and
        value math, whichever is longer, then the partials written and merged."""
        keys = depth + rows
        read = layers * keys * LATENT_ROW / self.bw
        work = layers * rows * self.heads * keys * ATTN_FLOP / (self.tflops * 1e9) if self.tflops else 0.0
        return max(read, work) + self.partials_ms(rows, chunks, layers)

    def context_ms(self, rows: float, layers: int) -> float:
        """What this depth and walk add to a measured cost (the calibration's 64 tokens and full walk taken out), or
        the whole context cost when nothing was measured."""
        ctx = self.attention_ms(rows, self.depth, layers, self.walk(rows))
        measured = self.verify_ms if layers == MLA_LAYERS else self.mtp_ms
        if measured:
            ctx -= self.attention_ms(rows, CAL_DEPTH, layers, chunks_for(self.slots))
        return ctx

    def kda_ms(self, rows: float) -> float:
        """Every KDA layer reads rec[cur], writes rec[1 - cur] and keeps each row's k, v, decay and beta."""
        return KDA_LAYERS * self.lin_heads * (2 * KDA_STATE + rows * KDA_ROW) / self.bw

    def replay_ms(self, keep: float) -> float:
        """commit() keeping fewer rows than it verified: every KDA layer again, from rec[cur] through the kept rows."""
        return self.kda_ms(keep)

    # -- rounds -----------------------------------------------------------------------------------------------------
    def forward_ms(self, R: int) -> float:
        if self.verify_ms:
            return self.verify_ms[R - 1] + self.context_ms(R, MLA_LAYERS)
        weights = (self.b.flat + union_rows(R, self.union) * self.b.expert_row) / self.bw
        return weights + self.g.forward + self.g.row * R + self.kda_ms(R) + self.context_ms(R, MLA_LAYERS)

    def mtp_round_ms(self, k: int, keep: float, reject: float = 0.0) -> float:
        """A round with k MTP drafts: the verify window of k + 1 rows, then the next round's drafts (one call absorbs the
        kept rows and drafts once, k - 1 chained calls draft the rest), and the KDA replay when a draft was rejected
        (``reject``: its probability)."""

        if k == 0:
            return self.forward_ms(1)
        calls = self.context_ms(keep, 1) + (k - 1) * self.context_ms(1, 1)
        if self.mtp_ms:
            m, s, r = self.mtp_ms
            drafts = m + r * max(keep - 1, 0) + s * (k - 1)
        else:
            step = self.b.mtp_step / self.bw + self.g.mtp
            drafts = step + self.b.mtp_row * max(keep - 1, 0) / self.bw + step * (k - 1)
        return self.forward_ms(k + 1) + drafts + calls + reject * self.replay_ms(keep)

    def block_round_ms(self, k: int, keep: float = 1.0, reject: float = 0.0) -> float:
        if k == 0:
            return self.forward_ms(1)
        return (self.forward_ms(k + 1) + self.drafter.pass_bytes(self.world) / self.bw + self.g.block
                + reject * self.replay_ms(keep))


def chain(rates: list[float]) -> float:
    """Tokens a round: 1 (the bonus/serial token) + the expected accepted prefix of the drafts."""

    e, p = 1.0, 1.0
    for a in rates:
        p *= a
        e += p
    return e


def mtp_rates(a1: float, k: int, decay: float) -> list[float]:
    return [min(a1 * decay ** j, 1.0) for j in range(k)]


def solve_a1(e3: float, decay: float) -> float:
    """The first-step rate that gives E(3) = e3 under the decay model (bisection)."""

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if chain(mtp_rates(mid, 3, decay)) < e3:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def solve_alpha(e: float, k: int) -> float:
    lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = (lo + hi) / 2
        if chain([mid] * k) < e:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def best(model: Model, kind: str, rate, kmax: int) -> tuple[int, float, float, float]:
    """The draft count with the most tokens a second: (k, tokens a round, ms a round, tokens a second)."""

    out = []
    for k in range(kmax + 1):
        rates = mtp_rates(rate, k, MTP_DECAY) if kind == "mtp" else [rate] * k
        e = chain(rates)
        reject = 1 - math.prod(rates) if k else 0.0
        ms = model.mtp_round_ms(k, e, reject) if kind == "mtp" else model.block_round_ms(k, e, reject)
        out.append((k, e, ms, 1e3 * e / ms))
    return max(out, key=lambda t: t[3])


MTP_DECAY = 0.9
# Ling-3.1-flash NEXTN acceptance, SGLang cookbook (3 drafts, bonus included): real-text probes, full GSM8K, random tokens
MTP_E3 = {"real text 2.13": 2.13, "GSM8K 2.43": 2.43, "random 2.61": 2.61}
# inclusionAI's Ling-3.0-flash DSpark (8 drafts, verify 9; HF card): Alpaca 3.51, the nine-workload mean 5.29,
# HumanEval 6.57 -> constant per-position rates; "ours" takes 0.04 off each rate (a smaller training set)
BLOCK_E8 = {"chat 3.51": 3.51, "mean 5.29": 5.29, "code 6.57": 6.57}


def measured_mtp(model: Model, accept: list[float]) -> tuple[int, float, float, float]:
    rows = []
    for k in range(1, len(accept) + 1):
        e = chain(accept[:k])
        ms = model.mtp_round_ms(k, e, 1 - math.prod(accept[:k]))
        rows.append((k, e, ms, 1e3 * e / ms))
    return max(rows, key=lambda t: t[3])


def parse_slots(text: str | None, worlds: list[int]) -> dict[int, int]:
    """One slot count for every world, or world:slots pairs (a world left out keeps the native window's)."""

    if not text:
        return {w: NATIVE + MAX_ROWS for w in worlds}
    if ":" not in text:
        return {w: int(text) for w in worlds}
    pairs = dict(item.split(":") for item in text.split(","))
    return {w: int(pairs.get(str(w), NATIVE + MAX_ROWS)) for w in worlds}


def label(n: int) -> str:
    return {4096: "4K", 32768: "32K", 131072: "128K", 262144: "262K", 1048576: "1M"}.get(n, f"{n:,}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", type=int, action="append", choices=(2, 4))
    ap.add_argument("--bw", type=float, default=205.0, help="effective GB/s a rank (PLAN 5.1: 190-210 as built)")
    ap.add_argument("--union", choices=("birthday", "correlated"), default="birthday")
    ap.add_argument("--verify-ms", help="measured verify ms for 1..8 rows (the engine's startup line), comma-separated")
    ap.add_argument("--mtp-ms", help="measured MTP draft, chained-draft and per-row ms, comma-separated")
    ap.add_argument("--kmax-block", type=int, default=7, help="drafts a block round verifies (engine MAX_ROWS - 1)")
    ap.add_argument("--mtp-accept", help="measured MTP acceptance a1,a2,...: draft j accepted given drafts 1..j-1 were")
    ap.add_argument("--slots", help="cache slots (the startup line's): one for every world, or pairs such as "
                                    "2:229930,4:262152 (default: the native window's 262,152)")
    ap.add_argument("--depth", default="4096,32768,131072,262144,1048576",
                    help="tokens already in the cache, comma-separated (depths past the slots are left out)")
    ap.add_argument("--tflops", type=float, default=TFLOPS,
                    help="the latent attention's effective score and value TFLOPS (0: bandwidth only)")
    ap.add_argument("--short-ms", help="with --verify-ms: the same engine's verify ms at a short window (--context 8192)")
    ap.add_argument("--short-slots", type=int, default=SHORT_SLOTS, help="that short engine's cache slots")
    args = ap.parse_args(argv)
    worlds = args.world or [2, 4]
    verify = [float(x) for x in args.verify_ms.split(",")] if args.verify_ms else None
    mtp = tuple(float(x) for x in args.mtp_ms.split(",")) if args.mtp_ms else None
    accept = [float(x) for x in args.mtp_accept.split(",")] if args.mtp_accept else None
    short = [float(x) for x in args.short_ms.split(",")] if args.short_ms else None
    slots = parse_slots(args.slots, worlds)
    depths = [int(x) for x in args.depth.split(",")]
    for world in worlds:
        last = slots[world] - MAX_ROWS                  # the window's last tokens: as deep as this engine goes
        kw = dict(slots=slots[world], tflops=args.tflops, short_ms=short, short_slots=args.short_slots)
        m = Model(world, args.bw, args.union, verify_ms=verify, mtp_ms=mtp, **kw)
        b = m.b
        print(f"TP{world} rank 0 at {args.bw:.0f} GB/s ({args.union} expert overlap), {m.slots:,} cache slots "
              f"({chunks_for(m.slots)} chunks), latent attention at {args.tflops:g} TFLOPS: flat {b.flat / GB:.3f} GB, "
              f"experts {b.expert_row / GB:.3f} GB a row, MTP step {b.mtp_step / GB:.3f} GB, drafter pass "
              f"{m.drafter.pass_bytes(world) / GB:.3f} GB, KDA states {m.kda_ms(0) * m.bw / GB:.3f} GB a forward")
        if m.per_chunk:
            print("  a chunk of the walk, measured (ms by rows): " + " ".join(f"{x * 1e3:.2f}e-3" for x in m.per_chunk))
        print("  forward ms by rows at 64 tokens (as the startup verify costs): "
              + " ".join(f"{m.forward_ms(R):.1f}" for R in range(1, 9)))
        prior = {u: Model(world, args.bw, u, **kw) for u in ("birthday", "correlated")}
        print(f"  verify(8)/verify(1): {m.forward_ms(8) / m.forward_ms(1):.2f} (birthday "
              f"{prior['birthday'].forward_ms(8) / prior['birthday'].forward_ms(1):.2f}, correlated "
              f"{prior['correlated'].forward_ms(8) / prior['correlated'].forward_ms(1):.2f})")
        print(f"  serial: {1e3 / m.forward_ms(1):.1f} tok/s")
        if accept:
            k, e, ms, tps = measured_mtp(m, accept)
            print(f"  MTP measured {','.join(f'{a:.3f}' for a in accept)}: best k={k} E {e:.2f} {ms:.1f} ms -> "
                  f"{tps:.1f} tok/s")
        for name, e3 in MTP_E3.items():
            a1 = solve_a1(e3, MTP_DECAY)
            k, e, ms, tps = best(m, "mtp", a1, 3)
            k3 = chain(mtp_rates(a1, 3, MTP_DECAY))
            tps3 = 1e3 * k3 / m.mtp_round_ms(3, k3, 1 - math.prod(mtp_rates(a1, 3, MTP_DECAY)))
            print(f"  MTP {name:15s} a1 {a1:.3f}: best k={k} E {e:.2f} {ms:.1f} ms -> {tps:.1f} tok/s"
                  f" (k=3: {tps3:.1f})")
        for name, e8 in BLOCK_E8.items():
            alpha = solve_alpha(e8, 8)
            for tag, a in (("3.0-class", alpha), ("ours -0.04", alpha - 0.04)):
                k, e, ms, tps = best(m, "block", a, args.kmax_block)
                print(f"  block {name:10s} {tag:10s} alpha {a:.3f}: best k={k} E {e:.2f} {ms:.1f} ms -> {tps:.1f} tok/s")
        # by depth: the walk as a graph replay does it today | only the chunks that hold keys (eager, the merge fix)
        cols = [("serial", None), ("MTP 2.13", ("mtp", solve_a1(2.13, MTP_DECAY))),
                ("MTP 2.43", ("mtp", solve_a1(2.43, MTP_DECAY))),
                ("block 5.29 ours", ("block", solve_alpha(5.29, 8) - 0.04))]
        if accept:
            cols.insert(1, ("MTP measured", ("measured", accept)))
        print(f"  tok/s by depth, today's walk (every chunk of the slots) | chunks that hold keys; ms a forward of 1 "
              f"and 4 rows today:")
        print(f"    {'depth':>6} " + "".join(f"{n:>21}" for n, _ in cols) + f"{'1 row':>9}{'4 rows':>9}")
        for p in [d for d in depths if d < last] + [last]:
            cells = []
            for _, how in cols:
                pair = []
                for merge in ("window", "filled"):
                    mm = Model(world, args.bw, args.union, verify_ms=verify, mtp_ms=mtp, depth=p, merge=merge, **kw)
                    if how is None:
                        pair.append(f"{1e3 / mm.forward_ms(1):5.1f}   ")
                        continue
                    kind, rate = how
                    k, _, _, tps = (measured_mtp(mm, rate) if kind == "measured"
                                    else best(mm, kind, rate, 3 if kind == "mtp" else args.kmax_block))
                    pair.append(f"{tps:5.1f} k{k}")
                cells.append(f"{pair[0]} | {pair[1]}")
            today = Model(world, args.bw, args.union, verify_ms=verify, mtp_ms=mtp, depth=p, **kw)
            print(f"    {label(p):>6} " + "".join(f"{c:>21}" for c in cells)
                  + f"{today.forward_ms(1):9.1f}{today.forward_ms(4):9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
