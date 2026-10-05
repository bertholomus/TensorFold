"""Routed EXL3 experts of any codebook, a width per expert, one grouped launch a projection; rows never depend on the window."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
ACT_BF16, ACT_F32 = 0, 1          # SwiGLU with the GLM family's bf16 roundings / in fp32
# Half-bits a value: 1..8 bits (2, 4, .. 16) and every half-integer rate 1.5..7.5 (3, 5, .. 15).
K2_SUPPORTED = tuple(range(2, 17))

# windows of this many rows or more (prompt chunks) size their member tiles to the busiest expert (a host read)
EXACT_ROWS = 64
# (n tiles a block, warps, K splits, tiles in flight): GLM's settings, whose arithmetic order this keeps bit for bit
GLM_GATEUP = (8, 4, 4, 1)
GLM_DOWN = (8, 4, 1, 1)
# prompt chunks' grouped launches: (n tiles a block, tiles in flight, member tiles a program) for gate/up and down, the
# weights decoded once for a program's member tiles; warps and K splits stay the window's (they fix each row's bits)
PROMPT_TILES = {"gateup": (8, 1, 2), "down": (8, 1, 2)}
PROMPT = os.environ.get("TF_EXL3_PROMPT_TILES", "1") != "0"
# prompt chunks' kernel: "mma" (64 member rows a program, each weight tile decoded once into shared memory for all of
# them) or "rows" (grouped_rows: member tiles in registers, PROMPT_TILES); both keep every row's bits. The mma kernel
# takes K in steps of MMA_KB k tiles, which must divide the window's chains (K / 16 / (splits x warps)); else rows.
# "mma2": 64 member rows by 256 columns a program, each warp decoding its own weight tiles into mma fragments and
# summing each output in one chain over K (deterministic, independent of the window, but not the decode windows' split
# order: other bits); needs N and K multiples of 256 (GLM-5.3's shapes), else mma.
PROMPT_KERNEL = os.environ.get("TF_EXL3_PROMPT_KERNEL") or "mma"
MMA_KB = 4                        # experts_grouped.cuh's MMA_KB
# "mma3": mma2 with gate/up's input rotation inside (rows made from the layer input in shared memory, no rot_in, no
# rotated copies; bf16 input, up to 4 bits a gate/up weight: the same bits as mma2's setting 0), and with the combine
# down's per-slot epilogue inside it: weighted slot outputs in bf16 (half of z's bytes), added by combine_y.
# grouped_mma3's gate/up columns a program: 4 (512, a TP4 rank's whole width: the rows rotated once a mat; 5.3 against
# 6.1 ms a 2,048-row layer at 2) or 2 (256)
MMA3_NW = int(os.environ.get("TF_EXL3_MMA3_NW") or 4)
# decode windows (fewer than EXACT_ROWS rows), every row's bits the same in all three:
#   "fused": three launches: decode_prep (grouping + gate/up's input rotation), grouped_decode for gate/up with the gate/up
#            epilogue in the program that completes each (expert, 128 columns), grouped_decode for down with each slot's
#            output and the rows' combine in it (GB10, a DeepSeek-V4.1 TP2 rank's layer, 1-16 rows: 7-13% faster than
#            "old", 225-242 GB/s from DRAM; tools/dsv41/expert_decode_bench.py);
#   "cp":    the six launches with grouped_cp_kernel (trellis words copied 16 bytes a lane through shared memory);
#   "old":   the six launches with grouped_kernel (4-byte lane loads).
DECODE = os.environ.get("TF_EXL3_DECODE") or "fused"
DECODE_STAGES = 3                 # each warp's ring of shared-memory stages (grouped_cp_launch's one setting)
# fused decode: gate/up and down launched as programmatic dependents (sm_90+: a grid starts on the SMs its predecessor
# frees; down copies its first trellis steps before it waits)
DECODE_PDL = os.environ.get("TF_EXL3_DECODE_PDL", "1") != "0"
# fused decode with DECODE_PDL: each down program waits only for its own expert's rows (published by the gate/up program
# that finishes the expert's epilogues) instead of the whole gate/up grid, so down overlaps gate/up's last programs
DECODE_READY = os.environ.get("TF_EXL3_DECODE_READY", "1") != "0"
# fused decode: the dead fp32 scratch is dropped from L2 instead of written back to DRAM once read (discard.global.L2):
# gate/up's partials after the epilogue that sums them, the per-slot outputs after the row's combine (with weights).
# The layer's experts streaming through L2 would evict them dirty (~2.5 MB a 6-row layer); no output changes.
DECODE_DISCARD = os.environ.get("TF_EXL3_L2_DISCARD", "1") != "0"


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    srcs = [str(here / f) for f in ("experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu")]
    return load(name="tensorfold_exl3_experts_v14", sources=srcs, extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False)


def codebook_id(name: str) -> int:
    """'3inst' / 'mcg' / 'mul1' (the checkpoint's quantization_config.codebook, or the marker tensor's name)."""

    ids = {"3inst": CB_3INST, "mcg": CB_MCG, "mul1": CB_MUL1}
    if name not in ids:
        raise ValueError(f"unknown EXL3 codebook {name!r}")
    return ids[name]


def k2_of(trellis: torch.Tensor) -> int:
    """Half-bits a value of a trellis int16 [K/16, N/16, 16 * K] (16 * K + 8 for the half-integer rates)."""

    w = trellis.shape[-1]
    if w % 8:
        raise ValueError(f"trellis last dim {w} is not a multiple of 8")
    k2 = w // 8
    if k2 not in K2_SUPPORTED:
        raise ValueError(f"unsupported EXL3 bit width {k2 / 2}")
    return k2


@dataclass
class Exl3RoutedExperts:
    """One layer's routed experts: trellis pointers and widths per projection, stacked suh/svh, the tensors keeping the trellises alive."""

    gate_ptr: torch.Tensor    # int64 [E]
    up_ptr: torch.Tensor
    down_ptr: torch.Tensor
    gate_k2: torch.Tensor     # int32 [E], half-bits a value
    up_k2: torch.Tensor
    down_k2: torch.Tensor
    suh_g: torch.Tensor       # fp16 [E, D]
    suh_u: torch.Tensor
    svh_g: torch.Tensor       # fp16 [E, I]
    svh_u: torch.Tensor
    suh_d: torch.Tensor       # fp16 [E, I]
    svh_d: torch.Tensor       # fp16 [E, D]
    count: int                # E
    dims: int                 # D (model width)
    width: int                # I (expert width on this rank)
    cb: int
    k2_gu: tuple[int, int]    # (min, max) K2 over gate and up
    k2_d: tuple[int, int]
    trellis_bytes: torch.Tensor   # int64 [E], gate + up + down trellis bytes of each expert (for GB/s)
    keep: list = field(default_factory=list, repr=False)
    aligned16: bool = False       # every trellis starts on a 16-byte boundary (the decode kernel's 16-byte copies)

    def nbytes_read(self, ids: Sequence[int]) -> int:
        return int(self.trellis_bytes[list(ids)].sum())


def prepare(gate: Sequence[tuple], up: Sequence[tuple], down: Sequence[tuple], codebook: int | str,
            device="cuda") -> Exl3RoutedExperts:
    """A layer from per-expert (trellis, suh, svh) triples, trellises referenced in place, each at its own width."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    E = len(gate)
    if not (len(up) == len(down) == E) or E == 0:
        raise ValueError("gate, up and down need the same, non-zero number of experts")
    D, I = gate[0][0].shape[0] * 16, gate[0][0].shape[1] * 16
    keep = []

    def table(mats, k, n):
        ptrs, k2s = [], []
        for t, _, _ in mats:
            if t.dtype != torch.int16 or not t.is_contiguous() or t.device.type != "cuda":
                raise ValueError("trellis must be a contiguous CUDA int16 tensor")
            if t.shape[0] * 16 != k or t.shape[1] * 16 != n:
                raise ValueError(f"expert shape {tuple(t.shape)} does not match [{k // 16}, {n // 16}, *]")
            k2s.append(k2_of(t))
            ptrs.append(t.data_ptr())
            keep.append(t)
        return (torch.tensor(ptrs, dtype=torch.int64, device=device),
                torch.tensor(k2s, dtype=torch.int32, device=device), k2s)

    gp, gk, gks = table(gate, D, I)
    upp, uk, uks = table(up, D, I)
    dp, dk, dks = table(down, I, D)

    def stack(mats, j, n):
        out = torch.empty((E, n), dtype=torch.float16, device=device)
        for e, m in enumerate(mats):
            out[e].copy_(m[j].reshape(-1))
        return out

    tb = torch.tensor([(D * I // 256) * (gks[e] + uks[e] + dks[e]) * 16 for e in range(E)], dtype=torch.int64)
    aligned = all(t.data_ptr() % 16 == 0 for t in keep)
    return Exl3RoutedExperts(gp, upp, dp, gk, uk, dk, stack(gate, 1, D), stack(up, 1, D), stack(gate, 2, I),
                             stack(up, 2, I), stack(down, 1, I), stack(down, 2, D), E, D, I, cb,
                             (min(gks + uks), max(gks + uks)), (min(dks), max(dks)), tb, keep, aligned)


def prepare_stacked(gt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor, suh_g, suh_u, svh_g, svh_u, suh_d, svh_d,
                    codebook: int | str) -> Exl3RoutedExperts:
    """A uniform-width layer stacked per projection: trellis [E, K/16, N/16, 16K] (int16, or GLM's int32 words), suh/svh [E, n]."""

    def as16(t):
        return t.view(torch.int16) if t.dtype == torch.int32 else t

    gt, ut, dt = as16(gt), as16(ut), as16(dt)
    E = gt.shape[0]
    return prepare([(gt[e], suh_g[e], svh_g[e]) for e in range(E)], [(ut[e], suh_u[e], svh_u[e]) for e in range(E)],
                   [(dt[e], suh_d[e], svh_d[e]) for e in range(E)], codebook, device=gt.device)


def default_config(K: int, N: int, gateup: bool) -> tuple[int, int, int, int]:
    """The tile setting for a K -> N projection (GLM's where it divides): the shape's alone, so rows stay independent."""

    cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
    for nt, w, sk, pf in cands:
        if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
            return nt, w, sk, pf
    raise ValueError(f"no tile setting divides K={K}, N={N}")


class Scratch:
    """Buffers for up to ``rows`` rows of ``slots`` slots; slots whose pick is not a routed expert are left to the caller.

    ``prompt``: buffers for prompt chunks, whose combine adds nothing for non-routed slots and keeps no per-slot outputs
    (``y`` is allocated on first use by a one-tile call: the A/B against the decode-once path).
    """

    def __init__(self, ex: Exl3RoutedExperts, rows: int, slots: int, cfg_gu=None, cfg_d=None, device="cuda",
                 prompt: bool = False) -> None:
        D, I = ex.dims, ex.width
        self.cfg_gu = cfg_gu or default_config(D, I, True)
        self.cfg_d = cfg_d or default_config(I, D, False)
        P = rows * slots
        # gate / up's rotated rows (rot_in): prompt buffers make them on first use (mma3 rotates inside its programs and
        # never needs them: 1.8 GB at 8,192 rows)
        self.xg = None if prompt else torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xu = None if prompt else torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, I), dtype=torch.float16, device=device)
        # gate and up write 2 * splits * P * I partials, down splits * P * D
        self.z = torch.zeros((max(2 * self.cfg_gu[2] * I, self.cfg_d[2] * D) * P,), dtype=torch.float32, device=device)
        self.y = None if prompt else torch.zeros((P, D), dtype=torch.float32, device=device)
        self.no_y = torch.zeros((1,), dtype=torch.float32, device=device)       # a pointer for calls that skip y
        maxu = min(P, ex.count)
        # the fused decode launches' counters, one per (expert place, 128 columns of I) and per (row, 128 columns of D);
        # zero between launches (each launch's last program of a block resets its counter)
        self.cnt_gu = None if prompt else torch.zeros((maxu * max(1, I // 128),), dtype=torch.int32, device=device)
        self.cnt_d = None if prompt else torch.zeros((rows * max(1, D // 128),), dtype=torch.int32, device=device)
        # ... and the experts' readiness for down (DECODE_READY): a launch number, the number an expert's rows were last
        # published under, the epilogues done so far
        self.epoch = torch.zeros((1,), dtype=torch.int32, device=device)
        self.ready = torch.zeros((max(1, maxu),), dtype=torch.int32, device=device)
        self.ready_cnt = torch.zeros((max(1, maxu),), dtype=torch.int32, device=device)
        self.ids = torch.zeros((maxu,), dtype=torch.int32, device=device)
        self.count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.counts = torch.zeros((ex.count,), dtype=torch.int32, device=device)     # members an expert (prompt chunks)
        # prompt chunks pad an expert's members to whole program groups (up to 8 tiles of 16)
        self.members_buf = torch.full((maxu * -(-rows // 128) * 128,), -1, dtype=torch.int32, device=device)
        self.rows, self.slots, self.count_experts = rows, slots, ex.count

    def window(self, R: int):
        """(ids, members) sized for R rows: the grids only span what R rows can use."""

        maxu = min(R * self.slots, self.count_experts)
        return self.ids[:maxu], self.members_buf[:maxu * R].view(maxu, R)


def routed(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor | None, ex: Exl3RoutedExperts, s: Scratch,
           out: torch.Tensor | None, R: int, limit: float = math.inf, act_mode: int = ACT_F32,
           group: bool = True, prompt: bool | None = None, add: torch.Tensor | None = None,
           kernel: str | None = None, decode: str | None = None,
           before_down: torch.cuda.Event | None = None) -> torch.Tensor:
    """Routed experts of R rows (picks >= E skipped): Y per slot, or ``out`` = the wts-weighted sum when ``wts`` (plus
    ``add`` [R, D] fp32 added last, in the same launch, when given).

    Decode windows make no host sync. Prompt chunks (``group`` and R >= EXACT_ROWS) read the busiest expert's row count
    once; with ``prompt`` (default TF_EXL3_PROMPT_TILES) they group in parallel and decode each weight tile once for
    several member rows (``kernel``, default PROMPT_KERNEL), every row's arithmetic the one-tile launch's (the same
    bits) for "mma" and "rows"; "mma2" / "mma3" sum in their own order. Decode windows take ``decode`` (default DECODE):
    "fused", "cp" or "old", the same bits.

    ``before_down``: an event the launches wait for before the ones that read the per-slot outputs of slots this call
    does not compute (picks >= E, whose y the caller writes into ``s.y``, e.g. from another stream): the fused decode
    waits just before its down launch, every other path before its first launch.
    """

    ext = _ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    if R > s.rows:
        raise ValueError(f"{R} rows but the scratch holds {s.rows}")
    ids, members = s.window(R)
    chunk = group and R >= EXACT_ROWS
    mode = DECODE if decode is None else decode
    if (mode == "fused" and group and not chunk and getattr(ex, "aligned16", False)
            and getattr(s, "cnt_gu", None) is not None and s.y is not None
            and tuple(s.cfg_gu[:2]) == (8, 4) and tuple(s.cfg_d[:3]) == (8, 4, 1) and I % 128 == 0 and D % 128 == 0
            and slots <= 32 and R <= 128 and x.stride(1) == 1):
        return _routed_fused(ext, x, pick, wts, ex, s, out, R, ids, members, limit, act_mode, add, before_down)
    if before_down is not None:
        torch.cuda.current_stream().wait_event(before_down)
    fast = chunk and (PROMPT if prompt is None else prompt) and all(
        n % (16 * t[0]) == 0 for n, t in ((I, PROMPT_TILES["gateup"]), (D, PROMPT_TILES["down"])))
    if fast:
        # a prompt chunk: experts' member counts on the device, the busiest one read back once to size the member
        # groups, then each used expert lists its members (the grouping of any number of rows, in parallel)
        ext.group_count(pick, s.counts, R, slots, E)
        busiest = int(s.counts.max())
        tile = 16 * max(PROMPT_TILES["gateup"][2], PROMPT_TILES["down"][2])
        members = s.members_buf[:ids.shape[0] * max(tile, -(-busiest // tile) * tile)].view(ids.shape[0], -1)
        ext.group_place(pick, s.counts, ids, s.count, members, R, slots, E)
    elif chunk:
        # a prompt chunk: member tiles for the busiest expert's rows only (one host read). Sized for R rows, the grouped
        # grids launched an expert's 16-row tiles for every row (2.6M mostly empty blocks a layer at 2,048 rows).
        busiest = int(torch.bincount(pick[:R].reshape(-1).long(), minlength=E + 1)[:E].max())
        members = s.members_buf[:ids.shape[0] * max(16, -(-busiest // 16) * 16)].view(ids.shape[0], -1)
    if group and not fast:
        ext.group(pick, ids, s.count, members, R, slots, E)
    if s.y is None and not (fast and wts is not None):        # prompt buffers: the one-tile path's per-slot outputs
        s.y = torch.zeros((s.rows * slots, D), dtype=torch.float32, device=x.device)
    kern = PROMPT_KERNEL if kernel is None else kernel
    mma2 = (fast and kern in ("mma2", "mma3") and I % 256 == 0 and D % 256 == 0 and
            max(ex.k2_gu[1], ex.k2_d[1]) <= 10)
    mma3 = (mma2 and kern == "mma3" and x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.stride(0) % 4 == 0
            and ex.k2_gu[1] <= 8)
    if not mma3:                     # mma3 rotates gate/up's rows from x inside its programs
        if s.xg is None:
            s.xg = torch.zeros((s.rows * slots, D), dtype=torch.float16, device=x.device)
            s.xu = torch.zeros((s.rows * slots, D), dtype=torch.float16, device=x.device)
        ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E)
    nt, w, sk, pf = s.cfg_gu
    # mma per projection: a down whose K chains do not split into MMA_KB steps (a TP2 DeepSeek-V4.1 rank: I = 1152)
    # still lets gate/up take the mma kernel
    mma_ok = not mma2 and fast and kern in ("mma", "mma2", "mma3") and I % 64 == 0 and D % 64 == 0
    mma = mma_ok and (D // 16) % (s.cfg_gu[1] * s.cfg_gu[2] * MMA_KB) == 0
    mma_d = mma_ok and (I // 16) % (s.cfg_d[1] * s.cfg_d[2] * MMA_KB) == 0
    if mma3:
        ext.grouped_mma3(x, x.stride(0), ex.suh_g, ex.suh_u, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids,
                         s.count, members, s.z, D, I, P, slots, ex.cb, ex.k2_gu[0], ex.k2_gu[1], MMA3_NW)
    elif mma2:
        # each output one fp32 chain over K: Z [2, P, I] holds whole sums (one split)
        ext.grouped_mma2(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D,
                         I, P, slots, ex.cb, ex.k2_gu[0], ex.k2_gu[1])
    elif mma:
        # every split's chains in one program, their sum from 0 as the epilogue would add them
        ext.grouped_mma(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                        P, sk, slots, ex.cb, w, ex.k2_gu[0], ex.k2_gu[1], 1)
    elif fast:
        # one program runs the K splits in order and writes their sum from 0, as the epilogue would add them
        pnt, ppf, pg = PROMPT_TILES["gateup"]
        ext.grouped_rows(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D,
                         I, P, sk, slots, ex.cb, pnt, w, ppf, pg, ex.k2_gu[0], ex.k2_gu[1], 1)
    else:
        cp = DECODE_STAGES if (mode != "old" and ex.aligned16 and nt == 8 and w == 4) else 0
        ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                    P, sk, slots, ex.cb, nt, w, cp or pf, ex.k2_gu[0], ex.k2_gu[1], int(cp > 0))
    ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, 1 if fast else sk, slots, E,
                        float(limit), act_mode)
    nt, w, sk, pf = s.cfg_d
    dsk = sk                          # the down partials' splits as down_combine reads them
    if mma3 and wts is not None and ex.k2_d[1] <= 8:
        # down with down_combine's per-slot arithmetic inside, times the routing weight, as bf16 Y [P, D] in the tail
        # of the z buffer (gate/up's sums take its first 2 P I floats); the combine adds a row's routed slots
        if out is None:
            out = torch.empty((R, D), dtype=torch.float32, device=x.device)
        yb = s.z.view(torch.bfloat16)[4 * P * I:4 * P * I + P * D]
        ext.grouped_down3(s.xd, ex.down_ptr, ex.down_k2, ids, s.count, members, ex.svh_d, wts, yb, I, D, P, slots,
                          ex.cb, ex.k2_d[0], ex.k2_d[1])
        ext.combine_y(yb, pick, s.no_y if add is None else add, out, R, D, slots, E, int(add is not None))
        return out
    if mma2:
        ext.grouped_mma2(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1,
                         I, D, P, slots, ex.cb, ex.k2_d[0], ex.k2_d[1])
        dsk = 1
    elif mma_d:
        # one split is written as is; several are folded in order from 0 (down_combine's order) into one
        ext.grouped_mma(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1,
                        I, D, P, sk, slots, ex.cb, w, ex.k2_d[0], ex.k2_d[1], int(sk > 1))
        dsk = 1
    elif fast:
        pnt, ppf, pg = PROMPT_TILES["down"]
        ext.grouped_rows(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1,
                         I, D, P, sk, slots, ex.cb, pnt, w, ppf, pg, ex.k2_d[0], ex.k2_d[1], 0)
    else:
        cp = DECODE_STAGES if (mode != "old" and ex.aligned16 and nt == 8 and w == 4) else 0
        ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                    D, P, sk, slots, ex.cb, nt, w, cp or pf, ex.k2_d[0], ex.k2_d[1], int(cp > 0))
    if wts is None:
        ext.down_epilogue(s.z, pick, ex.svh_d, s.y, R, P, D, dsk, slots, E)
        return s.y[:P]
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    # the down epilogue and the combine in one launch (the same arithmetic in the same order as the two); a prompt
    # chunk leaves the per-slot outputs unwritten (453 MB a layer at 2,048 rows that nothing reads)
    ext.down_combine(s.z, pick, ex.svh_d, s.no_y if fast else s.y, wts, s.no_y if add is None else add, out, R, P, D,
                     dsk, slots, E, 0 if fast else 1, int(add is not None))
    return out


def _routed_fused(ext, x, pick, wts, ex, s, out, R, ids, members, limit, act_mode, add, before_down=None):
    """routed()'s decode window in three launches (DECODE "fused"): the six launches' arithmetic, in the same order."""

    D, I, E, slots = ex.dims, ex.width, ex.count, s.slots
    P = R * slots
    has_wts = wts is not None
    has_add = has_wts and add is not None
    if has_wts and out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    o = out if has_wts else s.no_y
    w = wts if has_wts else s.no_y
    a = add if has_add else s.no_y
    ready = int(DECODE_PDL and DECODE_READY)
    ext.decode_prep(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, ids, s.count, members, R, D, slots, E, w,
                    s.y, a, o, int(has_wts), int(has_add), 1, s.epoch, ready)
    ext.grouped_decode(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                       P, s.cfg_gu[2], slots, ex.cb, DECODE_STAGES, ex.k2_gu[0], ex.k2_gu[1], 1, pick, E, ex.svh_g,
                       ex.svh_u, ex.suh_d, s.xd, float(limit), act_mode, s.y, s.no_y, s.no_y, s.no_y, 0, 0, 1,
                       s.cnt_gu, int(DECODE_PDL), s.ready, s.ready_cnt, s.epoch, ready, int(DECODE_DISCARD))
    if before_down is not None:            # the slots computed elsewhere: their y is in s.y before down combines
        torch.cuda.current_stream().wait_event(before_down)
    ext.grouped_decode(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                       D, P, 1, slots, ex.cb, DECODE_STAGES, ex.k2_d[0], ex.k2_d[1], 2, pick, E, ex.svh_d, ex.svh_d,
                       ex.svh_d, s.xd, float(limit), act_mode, s.y, w, a, o, int(has_wts), int(has_add), 1, s.cnt_d,
                       int(DECODE_PDL), s.ready, s.ready_cnt, s.epoch, ready, int(DECODE_DISCARD))
    return out if has_wts else s.y[:P]


def dequant(trellis: torch.Tensor, codebook: int | str) -> torch.Tensor:
    """W_q [K, N] fp16 of one matrix through the kernels' own lane decode (ExLlamaV3's ``reconstruct``); for tests."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    k2 = k2_of(trellis)
    K, N = trellis.shape[0] * 16, trellis.shape[1] * 16
    out = torch.empty((K, N), dtype=torch.float16, device=trellis.device)
    _ext().dequant(trellis.contiguous(), out, k2, cb)
    return out
