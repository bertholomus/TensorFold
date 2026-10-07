"""Decode context parallelism (TF_GLM_DCP=4, a long-context startup mode): the cache's positions interleaved over the
ranks, so each rank holds a quarter of every plane and the window reaches ~4x further.

The design follows ashhart/TensorFold PR #159 (dc43ca6, drowzeys: fused._attention_dcp, _merge_lse, _dcp_combine,
dcp_gather, select), restated for this family's kernels and cache formats:
- Position p lives on rank p % G at local slot p // G: latent, rope key, indexer keys and the MTP head's alike; every
  writer stores only its own rows.
- Selection (full indexer layers, rows past the dense limit): each rank scores its own slots (a token's score is the
  replicated scorer's, bit for bit), keeps its top index_topk (ties to the lower position), the ranks all-gather those
  candidates as (score, position) keys and every rank takes the same global top index_topk: the replicated selection
  exactly (positions are unique, so the merge is tie-free). A rank keeps the selected positions it holds.
- Attention: the ranks all-gather every head's absorbed query; each rank attends all heads over its own keys (rows
  below the dense limit: every visible key it holds; past it: its share of the selection), normalizes each head's
  partial and keeps its log-sum-exp; each head's partials go to the rank that owns the head, merged in rank order.
  Only the slots a rank holds are read: nothing is masked full width (vllm#58980).
Rows never meet, so a verify window's row keeps the serial step's bits (drafted == serial); the bits differ from the
replicated mode's (the merge order differs), so this mode has its own serial reference.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .kvq import rotate as _q_rotate, tile as _q_tile
from .mla_pe import _keys
from .select import BT

DCP = int(os.environ.get("TF_GLM_DCP") or 1)
# TF_GLM_DCP_PIPE (default 1; prompt chunks only): the selection's candidates go to the row owners
# and the queries and partials travel in parts on the comm stream beside the attention, instead of
# one all-gather and one exchange on the main stream. The same kernels over the same rows either
# way: the prompt chunks' bits are the lane's (Ported-from MiaAI-Lab patch 0131).
PIPE = os.environ.get("TF_GLM_DCP_PIPE", "1") != "0"
# TF_GLM_DCP_PROMPT_TILES (default 0): prompt chunks' DCP attention scores mla_pe.PROMPT_KT (16)
# keys a tile instead of the decode windows' 32 (her 0129 put the prompt chunks through her faster
# sparse-attention kernel; ours is the same launch at the prompt kernels' tile). MOVES the prompt
# chunks' bits (the online softmax rescales every 16 keys instead of 32): off until a new serial
# reference is adopted and a KLD run does not worsen.
PROMPT_TILES = os.environ.get("TF_GLM_DCP_PROMPT_TILES", "0") == "1"
PART_ROWS = int(os.environ.get("TF_GLM_DCP_PART_ROWS") or 512)   # a part's rows (the pipe's pieces)


def owner_rows(R: int, G: int) -> int:
    """Rows of a window each rank merges the selection of (the pipe's owner split): rank p takes rows
    p * M .. min(R, (p + 1) * M) - 1, M = ceil(R / G); the last ranks' rows past R are padding."""

    return -(-int(R) // int(G))


def cp_parts(R: int, step: int) -> list[tuple[int, int]]:
    """A window's parts for the pipe: ``step`` rows each, but the first and last half as long when the
    window takes more than one (the first part's query gather and the last part's partial exchange
    are the pipeline's exposed ends). Rows keep their bits in any part (the kernels are per row)."""

    if R <= step:
        return [(0, R)]
    bounds = [0, step // 2]
    while bounds[-1] + step < R:
        bounds.append(bounds[-1] + step)
    bounds.append(R)
    return list(zip(bounds[:-1], bounds[1:]))


def local_slots(capacity: int, world: int) -> int:
    """Cache rows one rank holds for ``capacity`` positions interleaved over ``world`` ranks."""

    return -(-capacity // world) + 1


# -- writers ---------------------------------------------------------------------------------------------------------
@triton.jit
def _write_bf16(X, x_stride, C, POS, W: tl.constexpr, G: tl.constexpr, RANK: tl.constexpr):
    """Program r: row r (position POS + r) into slot (POS + r) // G when this rank holds it."""

    r = tl.program_id(0)
    p = (tl.load(POS) + r).to(tl.int64)
    if p % G == RANK:
        k = tl.arange(0, W)
        tl.store(C + (p // G) * W + k, tl.load(X + r * x_stride + k))


def write_bf16(rows: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor, G: int, rank: int) -> None:
    _write_bf16[(rows.shape[0],)](rows, rows.stride(0), cache, pos, W=cache.shape[1], G=G, RANK=rank, num_warps=4)


# -- selection -------------------------------------------------------------------------------------------------------
@triton.jit
def _scores_dcp(QI, W, w_stride, IK, IS, HQ, OUT, POS, R, NT, scale, wscale, H: tl.constexpr, HP: tl.constexpr,
                D: tl.constexpr, BT: tl.constexpr, RB: tl.constexpr, KV8: tl.constexpr, QB: tl.constexpr,
                G: tl.constexpr, RANK: tl.constexpr):
    """select._scores over this rank's slots t (position t G + RANK): s_t = sum_h w_h relu(scale qi_h . k_t), -inf past
    each row's position. A token's arithmetic is the replicated scorer's (the same dot, head order and scales)."""

    rb = tl.program_id(0)
    tb = tl.program_id(1)
    P = tl.load(POS)
    t = tb * BT + tl.arange(0, BT)
    g = t * G + RANK
    last = P + rb * RB + RB                                   # past every row of the block
    if tb * BT * G + RANK < last:
        d = tl.arange(0, D)
        hh = tl.arange(0, HP)
        hok = hh < H
        if KV8:
            k = tl.load(IK + t[:, None].to(tl.int64) * D + d[None, :], mask=(g < last)[:, None],
                        other=0).to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            ks = tl.load(IS + t.to(tl.int64), mask=g < last, other=0.0)
        elif QB > 0:
            k = _q_tile(IK, IS, t.to(tl.int64), g < last, D, QB, BT)
        else:
            k = tl.load(IK + t[:, None].to(tl.int64) * D + d[None, :], mask=(g < last)[:, None],
                        other=0.0).to(tl.bfloat16)
        for i in tl.static_range(RB):
            r = rb * RB + i
            if r < R:
                bound = P + r + 1
                q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None],
                            other=0.0).to(tl.bfloat16)
                if QB > 0:
                    q = _q_rotate(q.to(tl.float32), HQ, HP, D).to(tl.float16)
                kr = tl.where((g < bound)[:, None], k, 0.0)
                dots = tl.dot(q, tl.trans(kr))
                if KV8:
                    dots = dots * ks[None, :]
                w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
                sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
                sc = tl.where(g < bound, sc, float("-inf"))
                tl.store(OUT + r * NT + t, sc, mask=t < NT)
    else:
        for i in tl.static_range(RB):
            r = rb * RB + i
            if r < R:
                tl.store(OUT + r * NT + t, tl.full((BT,), float("-inf"), tl.float32), mask=t < NT)


@triton.jit
def _candidates(S, s_stride, TOK, t_stride, KEYS, k_stride, K, G: tl.constexpr, RANK: tl.constexpr,
                BLOCK: tl.constexpr):
    """Program r: row r's K local picks (slots, ascending) as int64 keys in select._order_key's order, the score then
    the lower position first: (signed ordered score bits << 32) | (0xFFFFFFFF - position); a pick of an invisible slot
    (-inf) is no candidate (int64 min)."""

    r = tl.program_id(0).to(tl.int64)
    for i0 in range(0, K, BLOCK):
        i = i0 + tl.arange(0, BLOCK)
        ok = i < K
        slot = tl.load(TOK + r * t_stride + i, mask=ok, other=0)
        s = tl.load(S + r * s_stride + slot, mask=ok, other=float("-inf"))
        bits = (s + 0.0).to(tl.int32, bitcast=True)              # select._order_key's: signed, negatives flipped
        order = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
        key = (order.to(tl.int64) << 32) | (0xFFFFFFFF - (slot.to(tl.int64) * G + RANK))
        tl.store(KEYS + r * k_stride + i, tl.where(s == float("-inf"), -9223372036854775807 - 1, key), mask=ok)


# _scores_dcp's launch (keys a program, warps, stages) for a block of n rows over np_max slots. Every launch gives the
# same bits: a token's score is its own dot, and the tile only schedules it (a one-GPU scorer launch bench: 16 of 16
# outputs equal). GB10, FP8 indexer keys, DCP4, ms a call (2026-10-06):
#   8,192 x 2,048: (64, 4, default) 4.63, (128, 2, 2) 6.46, (128, 4, 2) 2.71
#   3,072 x 8,192: 6.81, 10.25, 4.93
#   768 x 32,768: 4.85, 10.94, 5.10
#   192 x 131,072: 5.10, 11.54, 5.23
# So prompt blocks (16 rows or more) take (128, 4, 2) up to 8,192 slots and e3290a0's (64, 4) past them; decode
# windows keep the BT of select.py (d98416c's 128 / 2 / 2; not measured at those shapes, decode equal on TP4).
SCORE_WIDE_SLOTS = 8192


def score_launch(n: int, np_max: int) -> tuple[int, int, int | None]:
    """(BT, num_warps, num_stages or None for Triton's default) of _scores_dcp for n rows over np_max slots."""

    if n < 16:
        return BT, 2, 2
    if np_max <= SCORE_WIDE_SLOTS:
        return 128, 4, 2
    return 64, 4, None


def candidates(qi: torch.Tensor, wts: torch.Tensor, keys, pos_dev: torch.Tensor, n: int, np_max: int, topk: int,
               G: int, rank: int) -> torch.Tensor:
    """Rows pos_dev .. + n: this rank's top-k slots (of np_max scored) as candidate keys [n, topk] int64 (no
    candidate: int64 min)."""

    from . import select as sel
    from .kv8 import Kv8
    from .kvq import KvQ

    H = wts.shape[1]
    D = qi.shape[1] // H
    kv8, qb = isinstance(keys, Kv8), keys.bits if isinstance(keys, KvQ) else 0
    ik, isc = (keys.codes, keys.scales) if kv8 or qb else (keys, keys)
    hq = keys.h if qb else isc
    scores = torch.empty((n, np_max), dtype=torch.float32, device=qi.device)
    rb = 16 if n >= 16 else n
    bt, warps, stages = score_launch(n, np_max)
    _scores_dcp[(triton.cdiv(n, rb), triton.cdiv(np_max, bt))](
        qi, wts, wts.stride(0), ik, isc, hq, scores, pos_dev, n, np_max, D ** -0.5, H ** -0.5, H=H,
        HP=max(16, triton.next_power_of_2(H)), D=D, BT=bt, RB=rb, KV8=kv8, QB=qb, G=G, RANK=rank, num_warps=warps,
        **({} if stages is None else {"num_stages": stages}))
    k = min(topk, np_max)
    local = torch.zeros((n, topk), dtype=torch.int32, device=qi.device)
    if n >= sel.RADIX_ROWS:
        sel._radix_topk[(n,)](scores, scores.stride(0), local, local.stride(0), np_max, K=k, BLOCK=1024, num_warps=4)
    else:
        sel._split_topk(scores, local, k, np_max)
    cand = torch.full((n, topk), -9223372036854775807 - 1, dtype=torch.int64, device=qi.device)
    _candidates[(n,)](scores, scores.stride(0), local, local.stride(0), cand, cand.stride(0), k, G=G, RANK=rank,
                      BLOCK=1024, num_warps=4)
    return cand


# TF_GLM_DCP_KV_GATHER (default 0): prompt chunks gather every rank's latent and rope rows of the
# positions the chunk sees (0141) and each rank attends its own heads over each row's whole top-k in
# one call of the prompt sparse attention, instead of exchanging every row's queries and partials
# (~50 KB a row and layer, moved once a layer a chunk here: the gathered planes' rows). MOVES the
# prompt chunks' bits (one pass over the row's keys instead of the ranks' partials merged by
# log-sum-exp): off until a new serial reference is adopted and a KLD run does not worsen.
# Decode windows keep dcp.attention's path. Supersedes the pipe's exchanges for prompt chunks (the
# owner selection stays).
KVG = os.environ.get("TF_GLM_DCP_KV_GATHER", "0") == "1"


def kvg_row_bytes(width: int, pw: int, kv: str | None) -> int:
    """Bytes a rank gathers per visible token a layer (the gathered planes' row: the latent in the
    lane's format plus the bf16 rope key), for the startup estimate."""

    from . import kv8

    mode = kv8.check_mode(kv or kv8.MODE)
    bits = kv8.latent_bits(mode)
    if kv8.latent_fp8(mode):
        lat = width + 4                                   # an e4m3 code a value plus the fp32 scale
    elif bits:
        lat = width // 32 * (bits * 4 + 2)                # kvq.row_bytes
    else:
        lat = width * 2
    return lat + pw * 2


def _gathered_like(plane, rows: int):
    """An empty plane of ``rows`` rows in ``plane``'s format (the gathered rows of every rank)."""

    dev = plane.device
    if isinstance(plane, KvQ):
        g = KvQ.__new__(KvQ)
        g.bits, g.width, g.h = plane.bits, plane.width, plane.h
        g.codes = torch.empty((rows, plane.codes.shape[1]), dtype=torch.int32, device=dev)
        g.scales = torch.empty((rows, plane.scales.shape[1]), dtype=torch.float16, device=dev)
        return g
    if isinstance(plane, Kv8):
        g = Kv8.__new__(Kv8)
        g.codes = torch.empty((rows, plane.codes.shape[1]), dtype=torch.uint8, device=dev)
        g.scales = torch.empty((rows,), dtype=torch.float32, device=dev)
        return g
    return torch.empty((rows, plane.shape[1]), dtype=plane.dtype, device=dev)


def _gather_views(plane, gathered, S: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The (send, recv) flat pairs whose all-gather fills ``gathered``'s first G * S rows with every
    rank's first S local slots (rank-major: position p at row (p % G) * S + p // G). fp16 scales
    travel viewed as bf16 and uint8 codes as int32 (the collectives' dtypes)."""

    pairs = []
    if isinstance(plane, KvQ):
        pairs.append((plane.codes[:S].reshape(-1), gathered.codes[:G * S].reshape(-1)))
        pairs.append((plane.scales[:S].view(torch.bfloat16).reshape(-1),
                      gathered.scales[:G * S].view(torch.bfloat16).reshape(-1)))
    elif isinstance(plane, Kv8):
        pairs.append((plane.codes[:S].view(torch.int32).reshape(-1),
                      gathered.codes[:G * S].view(torch.int32).reshape(-1)))
        pairs.append((plane.scales[:S], gathered.scales[:G * S]))
    else:
        pairs.append((plane[:S].reshape(-1), gathered[:G * S].reshape(-1)))
    return pairs


def kvg_buffers(b, lc, kr, G: int, S: int):
    """The gathered latent and rope rows of one layer, [G * S, ...] each (every rank's first S local
    slots, rank-major): views of one pair kept on the prompt buffers (``b.cp_kvg``), G x the planes'
    rows, made at the first chunk and reused by every layer and chunk."""

    held = getattr(b, "cp_kvg", None)
    rows = G * lc.shape[0]
    if held is None or held[0].shape[0] < rows or held[0].shape[1:] != lc.shape[1:] \
            or held[1].shape[1:] != kr.shape[1:]:
        held = b.cp_kvg = (_gathered_like(lc, rows), _gathered_like(kr, rows))
    return held[0], held[1]


def kvg_gather(w, lc, kr, b, R: int, host_pos: int):
    """All-gather every rank's first S local slots of this layer's planes (S covers the chunk's
    positions), on the prompt chunks' comm stream after the main stream's cache writes. Returns the
    gathered planes and the event the main stream waits before attending (None in line)."""

    G = w.world
    S = -(-(host_pos + R) // G)
    gl, gr = kvg_buffers(b, lc, kr, G, S)
    main = torch.cuda.current_stream()
    stream = getattr(b, "comm_stream", None)

    def gather() -> None:
        for send, recv in _gather_views(lc, gl, S):
            w.comm.all_gather(send, recv)
        for send, recv in _gather_views(kr, gr, S):
            w.comm.all_gather(send, recv)

    if stream is not None and stream is not main:
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            gather()
            ev = torch.cuda.Event()
            ev.record()
        return gl, gr, ev
    gather()
    return gl, gr, None


def owner_lists(cands: torch.Tensor, thresholds: torch.Tensor, G: int, S: int) -> tuple[torch.Tensor, int]:
    """A row owner's whole lists (0141): cands [G, M, k] (every rank's ``candidates`` of the owner's
    M rows, rank order) and each row's threshold (``_threshold``) -> rows [M, k] int32 and counts
    [M] int32: every rank's real candidates >= the threshold (the keys' own signed order, the same
    set dcp.merge takes) as global positions ascending - the replicated selection's list of the row -
    each position p written as the row (p % G) * S + p // G of the gathered planes, -1 past the count."""

    M, k = thresholds.shape[0], cands.shape[2]
    u = cands.permute(1, 0, 2).reshape(M, G * k)
    keep = (u >= thresholds[:M, None]) & (u != -9223372036854775807 - 1)
    big = torch.iinfo(torch.int64).max
    pos = torch.where(keep, 0xFFFFFFFF - (u & 0xFFFFFFFF), big)
    pos = torch.sort(pos, dim=1).values[:, :k]
    rows = torch.where(pos != big, (pos % G) * S + pos // G, -1).to(torch.int32)
    return rows.contiguous(), keep.sum(1).to(torch.int32)


def kvg_select(w, qi: torch.Tensor, wts: torch.Tensor, keys, host_pos: int, R: int, topk: int,
               pos_dev: torch.Tensor, *, tokens: torch.Tensor, counts: torch.Tensor) -> None:
    """select() for the gathered planes: the same scoring and candidate keys, the candidates to the
    row owners, and each owner keeps every rank's picks of its rows as global positions in
    gathered-plane rows (``owner_lists``); the lists are all-gathered (4 B a token), so every rank
    holds the same list."""

    from . import select as sel

    G, rank = w.world, w.rank
    np_max = sel.sparse_bucket(host_pos, R, G)
    if R >= sel.RADIX_ROWS:                                # prompt chunks: up to the last visible slot
        np_max = min(np_max, -(-(-(-(host_pos + R) // G)) // 64) * 64)
    np_max = min(np_max, keys.shape[0])
    S = -(-(host_pos + R) // G)                            # the gathered planes' local rows a rank
    rows = max(1, min(R, sel.SELECT_BYTES // (4 * np_max)))
    for r0 in range(0, R, rows):
        n = min(rows, R - r0)
        cand = candidates(qi[r0:r0 + n], wts[r0:r0 + n], keys, pos_dev + r0 if r0 else pos_dev, n, np_max, topk, G,
                          rank)
        blk = owner_rows(n, G)
        keys_p = torch.full((G * blk, topk), -9223372036854775807 - 1, dtype=torch.int64, device=cand.device)
        keys_p[:n] = cand
        flip = keys_p ^ (1 << 63)                      # the threshold kernel's uint64 order (sentinel the least)
        every = torch.full((G, blk, topk), -9223372036854775807 - 1, dtype=torch.int64, device=cand.device)
        every[rank] = flip[rank * blk:(rank + 1) * blk]
        peers = [p for p in range(G) if p != rank]
        w.comm.grouped([(flip[p * blk:(p + 1) * blk].reshape(-1), p) for p in peers],
                       [(every[p], p) for p in peers])
        th = torch.empty((blk,), dtype=torch.int64, device=cand.device)
        _threshold[(int(blk),)](every, th, blk, G=G, K=topk, KP=triton.next_power_of_2(topk),
                                num_warps=8 if topk > 1024 else 4)
        own_t, own_c = owner_lists(every ^ (1 << 63), th ^ (1 << 63), G, S)   # back to the keys' own order
        got_l = torch.empty((G * blk, topk), dtype=torch.int32, device=cand.device)
        got_c = torch.empty((G * blk,), dtype=torch.int32, device=cand.device)
        w.comm.all_gather(own_t, got_l)
        w.comm.all_gather(own_c, got_c)
        tokens[r0:r0 + n, :topk] = got_l[:n]
        counts[r0:r0 + n] = got_c[:n]


def kvg_all_visible(R: int, host_pos: int, G: int, S: int, k: int, dev,
                    tokens_full_width: int = 0) -> tuple[torch.Tensor, int]:
    """The gathered-plane lists for a chunk whose rows all see at most k tokens: row r every position
    0 .. host_pos + r ascending as gathered-plane rows ((p % G) * S + p // G), its count; what
    ``owner_lists`` gives a row with fewer than k candidates, so the row has the same list in any
    chunk."""

    p = torch.arange(k, dtype=torch.int64, device=dev)
    rows = ((p % G) * S + p // G).to(torch.int32)
    tokens = rows.expand(R, k).contiguous()
    counts = (torch.arange(R, dtype=torch.int32, device=dev) + int(host_pos) + 1).clamp(max=k)
    if tokens.shape[1] < tokens_full_width:                # merge()'s convention: the tail column -1
        tokens = torch.cat([tokens, tokens.new_full((R, tokens_full_width - tokens.shape[1]), -1)], dim=1)
    return tokens, counts


def kvg_attend(w, qa, qp, gl, gr, ev, b, R: int, out: torch.Tensor, scale: float, topk: int, *,
               selected: bool, within: bool, host_pos: int) -> None:
    """The gathered planes' attention (0141): the rank's own heads of the chunk's rows over each
    row's whole top-k (the gathered-plane lists), one prompt sparse-attention call - no query
    gather, no partials, no merge."""

    from . import mla_pe

    G = w.world
    S = -(-(host_pos + R) // G)
    main = torch.cuda.current_stream()
    if ev is not None:
        main.wait_event(ev)
    if not selected:
        if not within:
            raise ValueError("kvg: rows past the dense limit need the selection (a full indexer layer)")
        b.tokens[:R], b.counts[:R] = kvg_all_visible(R, host_pos, G, S, topk, qa.device,
                                              tokens_full_width=b.tokens.shape[1])
    mla_pe.sparse_attention(qa, qp, gl, gr, b.tokens[:R], b.counts[:R], out, scale)


def merge(every: torch.Tensor, G: int, rank: int, topk: int, tokens: torch.Tensor, counts: torch.Tensor) -> None:
    """Every rank's candidates [G, n, topk] -> the global top-k of each row (the replicated selection: positions are
    unique, so the keys never tie) and, into tokens [n, width] / counts [n], the slots this rank holds of it,
    ascending, -1 past them."""

    n = every.shape[1]
    top = torch.topk(every.permute(1, 0, 2).reshape(n, G * topk), topk, dim=1, sorted=False).values
    p = 0xFFFFFFFF - (top & 0xFFFFFFFF)                       # the global positions
    own = ((p % G) == rank) & (top != -9223372036854775807 - 1)
    slot = torch.where(own, p // G, torch.full_like(p, 1 << 40))
    tokens[:, :topk].copy_(torch.sort(slot, dim=1).values.clamp(max=(1 << 31) - 1).to(torch.int32))
    cnt = own.sum(1).to(torch.int32)
    tokens[:, :topk].masked_fill_(torch.arange(topk, device=every.device)[None, :] >= cnt[:, None], -1)
    if tokens.shape[1] > topk:
        tokens[:, topk:] = -1
    counts.copy_(cnt)


@triton.jit
def _threshold(C, T, M, G: tl.constexpr, K: tl.constexpr, KP: tl.constexpr):
    """Program m: T[m] = the K-th largest of the G candidates C[g, m, :] in uint64 order (0131's
    select_threshold; the keys arrive in the uint64 order, the no-candidate sentinel as 0): the same
    8-pass radix select on the same keys, so the global top-k set of dcp.merge."""

    m = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, KP)
    jok = j < K
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint64)
    fixed = tl.zeros((), dtype=tl.uint64)
    need = K
    for p in tl.static_range(8):
        hist = tl.zeros((256,), dtype=tl.int32)
        for g in tl.static_range(G):
            u = tl.load(C + (g * M + m) * K + j, mask=jok, other=0).to(tl.uint64, bitcast=True)
            match = jok & ((u & fixed) == prefix)
            hist += tl.histogram(((u >> (56 - 8 * p)) & 0xFF).to(tl.int32), 256, mask=match)
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint64) << (56 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint64) << (56 - 8 * p))
    tl.store(T + m, prefix.to(tl.int64, bitcast=True))


@triton.jit
def _keep(CAND, T, SL, CNT, G: tl.constexpr, K: tl.constexpr, KP: tl.constexpr):
    """Program r: this rank's candidates CAND[r] that are real and >= the row's threshold T[r] (the
    uint64 order, 0131's select_keep), in ascending position order, as local slots (position // G)
    -> SL[r] (int32, -1 past them), their number -> CNT[r]: merge()'s own slots and counts."""

    r = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, KP)
    jok = j < K
    prefix = tl.load(T + r).to(tl.uint64, bitcast=True)
    u = tl.load(CAND + r * K + j, mask=jok, other=0).to(tl.uint64, bitcast=True)
    keep = jok & (u >= prefix) & (u != 0)
    kp = keep.to(tl.int32)
    gpos = 4294967295 - u.to(tl.uint32).to(tl.int64)
    at = tl.cumsum(kp, 0) - kp
    n = tl.sum(kp, 0)
    tl.store(SL + r * K + at, (gpos // G).to(tl.int32), mask=keep)
    tl.store(SL + r * K + j, tl.full((KP,), -1, tl.int32), mask=jok & (j >= n))
    tl.store(CNT + r, n)


def owner_select(w, cand: torch.Tensor, n: int, topk: int, tokens: torch.Tensor,
                 counts: torch.Tensor) -> None:
    """The pipe's selection merge: every rank's candidates [n, topk] int64 (this rank's slots of every
    row, _candidates' keys, as dcp.merge's input) -> the same tokens / counts as dcp.merge over the
    all-gathered candidates: the owners take their rows' thresholds (one radix select a row over the
    G candidates) and the thresholds go back to every rank (one int64 a row), whose keep is its own
    slots. The keys are flipped to the uint64 order for the kernels (the top bit: our keys are
    float-ordered as signed int64, the radix and the keep compare uint64)."""

    G, rank = w.world, w.rank
    M = owner_rows(n, G)
    GM = G * M
    # every rank's candidates padded to G * M rows with the sentinel (0 in the uint64 order), so the
    # owners' fixed-size sends and receives always match (the last owner's rows past n are padding)
    keys = torch.zeros((GM, topk), dtype=torch.int64, device=cand.device)
    keys[:n] = cand ^ (1 << 63)
    every = torch.empty((G, M, topk), dtype=torch.int64, device=cand.device)
    every[rank].copy_(keys[rank * M:(rank + 1) * M])
    peers = [p for p in range(G) if p != rank]
    w.comm.grouped([(keys[p * M:(p + 1) * M].reshape(-1), p) for p in peers],
                   [(every[p], p) for p in peers])
    th = torch.empty((M,), dtype=torch.int64, device=cand.device)
    _threshold[(int(M),)](every, th, M, G=G, K=topk, KP=triton.next_power_of_2(topk),
                          num_warps=8 if topk > 1024 else 4)
    th_all = torch.empty((GM,), dtype=torch.int64, device=cand.device)
    w.comm.all_gather(th, th_all)
    slots = torch.empty((GM, topk), dtype=torch.int32, device=cand.device)
    cnt = torch.empty((GM,), dtype=torch.int32, device=cand.device)
    _keep[(int(GM),)](keys, th_all, slots, cnt, G=G, K=topk, KP=triton.next_power_of_2(topk),
                      num_warps=8 if topk > 1024 else 4)
    tokens[:, :topk].copy_(slots[:n])
    if tokens.shape[1] > topk:
        tokens[:, topk:] = -1
    counts.copy_(cnt[:n])


def select(w, qi: torch.Tensor, wts: torch.Tensor, keys, pos: int | None, R: int, topk: int, pos_dev: torch.Tensor,
           *, tokens: torch.Tensor, counts: torch.Tensor, bucket: int | None = None) -> None:
    """select.select_tokens across the ranks' slots: tokens[r, :counts[r]] this rank's slots of row r's global top-k
    (ascending), the rest -1. Rows at or below the dense limit get no use of theirs (attention reads their range).
    ``bucket`` fixes the slots scored (captured graphs, no host position); else they follow the host position."""

    from . import select as sel

    G, rank = w.world, w.rank
    if bucket is not None:
        np_max = bucket
    else:
        start = int(pos) if pos is not None else int(pos_dev.item())
        np_max = sel.sparse_bucket(start, R, G)
        if pos is not None and R >= sel.RADIX_ROWS:            # prompt chunks: up to the last visible slot
            np_max = min(np_max, -(-(-(-(start + R) // G)) // 64) * 64)
    np_max = min(np_max, keys.shape[0])
    rows = max(1, min(R, sel.SELECT_BYTES // (4 * np_max)))
    owners = R > 64 and PIPE and bucket is None
    for r0 in range(0, R, rows):
        n = min(rows, R - r0)
        cand = candidates(qi[r0:r0 + n], wts[r0:r0 + n], keys, pos_dev + r0 if r0 else pos_dev, n, np_max, topk, G,
                          rank)
        if owners:                        # the pipe: candidates to the row owners, one int64 a row back
            owner_select(w, cand, n, topk, tokens[r0:r0 + n], counts[r0:r0 + n])
            continue
        every = torch.empty((G, n, topk), dtype=torch.int64, device=qi.device)
        w.comm.all_gather(cand.view(-1).view(torch.float32), every.view(-1).view(torch.float32))
        merge(every, G, rank, topk, tokens[r0:r0 + n], counts[r0:r0 + n])


# -- attention -------------------------------------------------------------------------------------------------------
@triton.jit
def _attend_dcp(QALL, LC, LS, HQ, PC, TOK, CNT, POS, SEND, R, W: tl.constexpr, H: tl.constexpr, LW: tl.constexpr,
                PW: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, KTT: tl.constexpr, K: tl.constexpr,
                G: tl.constexpr, RANK: tl.constexpr, KV8: tl.constexpr, QB: tl.constexpr):
    """Program (row, head group g): rank g's H heads (QALL [G, R, H, LW + PW]) of row r over this rank's keys of the
    row: its slots of positions 0 .. p while p < K (the dense limit), else its share of the selection (TOK, CNT), in
    chunks of CH merged as _merge merges them. Writes each head's normalized partial (bf16 pairs as fp32 words) and
    log-sum-exp into SEND [G, R, H, LW / 2 + 1] at group g (no key: 0 and -inf)."""

    r = tl.program_id(0)
    grp = tl.program_id(1)
    p = tl.load(POS) + r
    dense = p < K
    if dense:
        n = tl.where(p >= RANK, (p - RANK) // G + 1, 0)
    else:
        n = tl.load(CNT + r)
    hh = tl.arange(0, H)
    k = tl.arange(0, LW)
    kq = tl.arange(0, PW)
    qb = QALL + ((grp * R + r) * H + hh[:, None]) * (LW + PW)
    q = tl.load(qb + k[None, :]).to(tl.bfloat16)
    qp = tl.load(qb + LW + kq[None, :]).to(tl.bfloat16)
    qr = q
    if QB > 0:
        qr = _q_rotate(q.to(tl.float32), HQ, H, LW).to(tl.float16)
    mm = tl.full((H,), float("-inf"), tl.float32)
    ll = tl.zeros((H,), tl.float32)
    oo = tl.zeros((H, LW), tl.float32)
    for c in range(tl.cdiv(n, CH)):
        m = tl.full((H,), float("-inf"), tl.float32)
        l = tl.zeros((H,), tl.float32)
        o = tl.zeros((H, LW), tl.float32)
        for t in range(tl.minimum(CH // KTT, tl.cdiv(n - c * CH, KTT))):
            idx = c * CH + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            if dense:
                rows = idx.to(tl.int64)
            else:
                rows = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
            m, l, o = _keys(q, qr, qp, LC, LS, PC, rows, ok, k, kq, m, l, o, LW, PW, SCALE, KTT, KV8, QB)
        if QB > 0:
            o = _q_rotate(o, HQ, H, LW)
        active = l > 0.0
        next_m = tl.where(active, tl.maximum(mm, m), mm)
        a = tl.where(active, tl.where(mm == float("-inf"), 0.0, tl.exp(mm - next_m)), 1.0)
        b = tl.where(active, tl.exp(m - next_m), 0.0)
        oo = oo * a[:, None] + o * b[:, None]
        ll = ll * a + l * b
        mm = next_m
    has = ll > 0.0
    out = tl.where(has[:, None], oo / tl.where(has, ll, 1.0)[:, None], 0.0).to(tl.bfloat16)
    WORDS: tl.constexpr = LW // 2 + 1
    base = SEND + ((grp * R + r) * H + hh) * WORDS
    pairs = tl.reshape(out, (H, LW // 2, 2))
    lo, hi = tl.split(pairs)
    word = lo.to(tl.uint16, bitcast=True).to(tl.uint32) | (hi.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
    tl.store(base[:, None] + tl.arange(0, LW // 2)[None, :], word.to(tl.float32, bitcast=True))
    tl.store(base + LW // 2, tl.where(has, mm + tl.log(tl.where(has, ll, 1.0)), float("-inf")))


@triton.jit
def _combine(RECV, OUT, R, src_stride, H: tl.constexpr, LW: tl.constexpr, G: tl.constexpr):
    """Program (row, own head): every rank's normalized partial of the head (RECV + src * src_stride, laid out [R, H,
    LW / 2 + 1] words) merged by their log-sum-exps in rank order -> OUT [R, H, LW] bf16."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    WORDS: tl.constexpr = LW // 2 + 1
    j = tl.arange(0, LW // 2)
    mx = float("-inf")
    for s in tl.static_range(G):
        mx = tl.maximum(mx, tl.load(RECV + s * src_stride + (r * H + h) * WORDS + LW // 2))
    acc_lo = tl.zeros((LW // 2,), tl.float32)
    acc_hi = tl.zeros((LW // 2,), tl.float32)
    den = 0.0
    for s in tl.static_range(G):
        base = RECV + s * src_stride + (r * H + h) * WORDS
        ls = tl.load(base + LW // 2)
        wgt = tl.where(ls == float("-inf"), 0.0, tl.exp(ls - mx))
        word = tl.load(base + j).to(tl.uint32, bitcast=True)
        lo = (word & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        hi = (word >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        acc_lo += wgt * lo
        acc_hi += wgt * hi
        den += wgt
    d = tl.where(den > 0.0, den, 1.0)
    out = tl.join(acc_lo / d, acc_hi / d)                     # [LW / 2, 2]: back to the value order
    tl.store(OUT + (r * H + h) * LW + tl.arange(0, LW), tl.reshape(out, (LW,)).to(tl.bfloat16))


class Scratch:
    """A window's DCP exchange buffers (flat, viewed contiguous for each window's rows): every head's absorbed query,
    the partials sent, the partials received."""

    def __init__(self, rows: int, heads: int, lw: int, pw: int, world: int, device) -> None:
        self.rows, self.heads, self.lw, self.pw, self.world = rows, heads, lw, pw, world
        words = lw // 2 + 1
        self.qpack = torch.empty((rows * heads * (lw + pw),), dtype=torch.bfloat16, device=device)
        self.qall = torch.empty((world * rows * heads * (lw + pw),), dtype=torch.bfloat16, device=device)
        self.send = torch.empty((world * rows * heads * words,), dtype=torch.float32, device=device)
        # decode windows gather every rank's whole send (a rank reads its own group); prompt chunks exchange groups
        self.recv = torch.empty(((world if rows <= 64 else 1) * world * rows * heads * words,), dtype=torch.float32,
                                device=device)


def attend(qall: torch.Tensor, cache, pcache: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
           pos: torch.Tensor, send: torch.Tensor, R: int, H: int, G: int, rank: int, scale: float, topk: int) -> None:
    """Every rank's heads (qall [G, R, H, LW + PW]) of the window's rows over the keys this rank holds -> send
    [G, R, H, LW / 2 + 1] (each head's normalized partial and log-sum-exp, laid out by the head's owner)."""

    from .mla_pe import FUSED_LAUNCH, KT, PROMPT_KT as _PROMPT_KT, _planes

    PW = pcache.shape[1]
    LW = qall.shape[-1] - PW
    rows, scales, h32, kv8, qb = _planes(cache)
    ktt = _PROMPT_KT if (PROMPT_TILES and R > 64) else KT
    _attend_dcp[(R, G)](qall, rows, scales, h32, pcache, tokens, counts, pos, send, R, W=tokens.shape[1], H=H, LW=LW,
                        PW=PW, CH=512, SCALE=scale, KTT=ktt, K=topk, G=G, RANK=rank, KV8=kv8, QB=qb,
                        num_warps=FUSED_LAUNCH[0], num_stages=FUSED_LAUNCH[1])


def combine(src: torch.Tensor, stride: int, out: torch.Tensor, G: int) -> None:
    """Every rank's partials of this rank's heads (src, a [R, H, LW / 2 + 1] block every ``stride`` words, in rank
    order) merged by their log-sum-exps -> out [R, H, LW] bf16."""

    R, H, LW = out.shape
    _combine[(R, H)](src, out, R, stride, H=H, LW=LW, G=G, num_warps=4)


def attention(w, qa: torch.Tensor, qp: torch.Tensor, cache, pcache: torch.Tensor, tokens: torch.Tensor,
              counts: torch.Tensor, pos: torch.Tensor, out: torch.Tensor, scale: float, d: Scratch, topk: int,
              stream=None) -> None:
    """Every head's attention over the keys this rank holds, merged across the ranks into this rank's
    heads (out). ``stream``: the prompt chunks' collective stream (the pipe runs its exchanges
    beside the attention there; None or the current stream: everything in line, as before)."""

    R, H, LW = qa.shape
    PW = qp.shape[2]
    G, rank = w.world, w.rank
    if R > 64 and PIPE and stream is not None:
        # the pipe packs and gathers the queries itself, by parts; a whole-chunk gather here would only be
        # overwritten unread (453 MB in a rank and layer at 8,192 rows, on the main stream)
        return pipe_attention(w, qa, qp, cache, pcache, tokens, counts, pos, out, scale, d, topk, stream)
    qpack = d.qpack[:R * H * (LW + PW)].view(R, H, LW + PW)
    qpack[:, :, :LW].copy_(qa)
    qpack[:, :, LW:].copy_(qp)
    qall = d.qall[:G * R * H * (LW + PW)]
    if R <= 64:                                           # bf16 pairs as fp32 words: the RDMA gather takes them
        w.comm.all_gather(qpack.view(-1).view(torch.float32), qall.view(torch.float32))
    else:
        w.comm.all_gather(qpack.view(-1), qall)
    words = LW // 2 + 1
    send = d.send[:G * R * H * words].view(G, R, H, words)
    attend(qall.view(G, R, H, LW + PW), cache, pcache, tokens, counts, pos, send, R, H, G, rank, scale, topk)
    if R <= 64:                                           # decode windows: gather everything, read our group
        recv = d.recv[:G * G * R * H * words].view(G, G, R, H, words)
        w.comm.all_gather(send.view(-1), recv.view(-1))
        combine(recv[:, rank], G * R * H * words, out, G)
    else:                                                 # prompt chunks: each group to its owner
        recv = d.recv[:G * R * H * words].view(G, R, H, words)
        w.comm.grouped([(send[g], g) for g in range(G)], [(recv[g], g) for g in range(G)])
        combine(recv, R * H * words, out, G)


def pipe_attention(w, qa, qp, cache, pcache, tokens, counts, pos, out, scale, d, topk,
                   stream) -> None:
    """attention()'s prompt chunks through the pipe (0131): the queries and partials travel in parts
    of PART_ROWS rows on the comm stream beside the attention on the main stream, a ring of two
    buffers. The same _attend_dcp launch runs over the same rows of each part and the same combine
    merges the same partials: the outputs are attention()'s bit for bit."""

    R, H, LW = qa.shape
    PW = qp.shape[2]
    G, rank = w.world, w.rank
    words = LW // 2 + 1
    main = torch.cuda.current_stream()
    parts = cp_parts(R, min(PART_ROWS, R))
    n_max = max(r1 - r0 for r0, r1 in parts)
    if 2 * n_max > d.rows:                                # the ring does not fit: today's path
        send = d.send[:G * R * H * words].view(G, R, H, words)
        qpack = d.qpack[:R * H * (LW + PW)].view(R, H, LW + PW)
        qpack[:, :, :LW].copy_(qa)
        qpack[:, :, LW:].copy_(qp)
        qall = d.qall[:G * R * H * (LW + PW)]
        w.comm.all_gather(qpack.reshape(-1), qall)
        attend(qall.view(G, R, H, LW + PW), cache, pcache, tokens, counts, pos, send, R, H, G, rank, scale, topk)
        recv = d.recv[:G * R * H * words].view(G, R, H, words)
        w.comm.grouped([(send[g], g) for g in range(G)], [(recv[g], g) for g in range(G)])
        combine(recv, R * H * words, out, G)
        return
    ring = []
    for i in range(2):
        qo = i * G * n_max * H * (LW + PW)
        so = i * G * n_max * H * words
        ring.append(dict(q=d.qall[qo:qo + G * n_max * H * (LW + PW)],
                         send=d.send[so:so + G * n_max * H * words],
                         recv=d.recv[so:so + G * n_max * H * words]))
    qpack = d.qpack[:R * H * (LW + PW)].view(R, H, LW + PW)
    qpack[:, :, :LW].copy_(qa)
    qpack[:, :, LW:].copy_(qp)
    ev_q: list = [None] * len(parts)
    ev_a: list = [None] * len(parts)
    ev_x: list = [None] * len(parts)

    def gather(i: int) -> None:
        r0, r1 = parts[i]
        n = r1 - r0
        t = ring[i % 2]
        if stream is main:
            w.comm.all_gather(qpack[r0:r1].reshape(-1), t["q"][:G * n * H * (LW + PW)])
            return
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            w.comm.all_gather(qpack[r0:r1].reshape(-1), t["q"][:G * n * H * (LW + PW)])
            ev_q[i] = torch.cuda.Event()
            ev_q[i].record()

    def exchange(i: int) -> None:
        r0, r1 = parts[i]
        n = r1 - r0
        t = ring[i % 2]
        sends = [(t["send"][:G * n * H * words].view(G, n, H, words)[g], g) for g in range(G)]
        recvs = [(t["recv"][:G * n * H * words].view(G, n, H, words)[g], g) for g in range(G)]
        if stream is main:
            main.wait_event(ev_a[i])
            w.comm.grouped(sends, recvs)
            return
        stream.wait_event(ev_a[i])
        with torch.cuda.stream(stream):
            w.comm.grouped(sends, recvs)
            ev_x[i] = torch.cuda.Event()
            ev_x[i].record()

    def merge(i: int) -> None:
        r0, r1 = parts[i]
        n = r1 - r0
        t = ring[i % 2]
        if ev_x[i] is not None:
            main.wait_event(ev_x[i])
        combine(t["recv"][:G * n * H * words], n * H * words, out[r0:r1], G)

    gather(0)
    if len(parts) > 1:
        gather(1)
    for i in range(len(parts)):
        r0, r1 = parts[i]
        n = r1 - r0
        t = ring[i % 2]
        if ev_q[i] is not None:
            main.wait_event(ev_q[i])
        attend(t["q"][:G * n * H * (LW + PW)].view(G, n, H, LW + PW), cache, pcache, tokens[r0:r1],
               counts[r0:r1], pos if r0 == 0 else pos + r0, t["send"][:G * n * H * words].view(G, n, H, words),
               n, H, G, rank, scale, topk)
        ev_a[i] = torch.cuda.Event()
        ev_a[i].record()
        if i + 2 < len(parts):
            gather(i + 2)
        exchange(i)
        if i >= 1:
            merge(i - 1)
    merge(len(parts) - 1)
    if stream is not main:
        main.wait_stream(stream)
