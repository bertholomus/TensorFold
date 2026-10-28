"""Beam-1.0 on TF_TP_WORLD tensor-parallel ranks (two: rank 0 and rank 1 on the fabric between two GB10s).

Rank 0 schedules. A request waits for a sequence slot and for the KV pages its prompt and reply need, fills its prompt
chunk by chunk, then decodes one row a step together with every other decoding sequence in one forward; prompt chunks
and decode steps take turns while both have work. Before each step rank 0 shares the step's plan (the slots it
releases, the pages it reserves, the sequences with their first positions and rows, the rows' token ids, the rows it
samples with their sampling) and every rank runs the same forward, so the ranks' page tables stay alike. The head is
split by vocabulary: the ranks' candidates (each one's first maximum and top-k) are gathered and rank 0 draws by the
keyed rule of tensorfold.engine.exact_sampling. Between bursts of work the other ranks wait on the rendezvous store,
not in NCCL.

With a drafter, a decoding sequence's step holds its last token and k drafts: every row is drawn at its own position
by the same keyed rule, and the longest run of drafts that equal the draws is kept, with the draw after it. Decode
rows' bits do not depend on the rows beside them (attention.py), so a reply is the same token for token with drafts
or without. The DFlash2 drafter runs on every rank alike; its context comes from the target's residual stream at five
layers, kept for the rows each step keeps (the next plan names them)."""

from __future__ import annotations

import ctypes
import gc
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows
from tensorfold.families.beam.config import Config

IDLE, STEP = 0, 1                     # plan ops: IDLE sends the other ranks back to the store
PREFILL, DECODE = 0, 1
HEADER = 14                           # op, kind, sequences, rows, releases, reserves, samples, payload length, score,
                                      # context entries, drafted, prompt cache evictions, restores, saves
CTX_WORDS = 5                         # slot, first position, capture row, rows, which
CTX_PREV, CTX_THIS = 0, 1             # a context entry's states: the last step's capture, or this step's
SAMPLE_WORDS = 9                      # row, position, then pack_sampling's 7
PAGE = 256                            # positions a full layer's KV page holds
DEFAULT_CHUNK = 2048                  # prompt rows a forward
DEFAULT_SLOTS = 4                     # sequences decoding together
PROMPT_CACHE = 4                      # prompt cache entries (TF_BEAM_PROMPT_CACHE; 0: none): a prompt's pages up to a
                                      # page boundary before its end and its rings there, for a later prompt that starts
                                      # the same
MIN_PIECE = 129                       # prompt rows come out the same in any piece of this many rows or more (at 128 or
                                      # fewer the attention splits its keys and EXL3 and the router take decode paths)
CACHE_FINE = 768                      # TF_BEAM_CACHE_FINE (0: the chunk grid only): a prompt that came from an entry
                                      # keeps its own at its last page boundary MIN_PIECE rows before its end when that
                                      # is this many rows past its chunk grid point (the piece it adds costs ~700 rows)
DECODE_SHARE = 1.0                    # after a prompt chunk, decoding sequences take steps for this much of its time
DEFAULT_SPEC_K = 3                    # drafts a sequence's step (the vLLM recipe's K)
LOOKUP_SPAN = 8192                    # the prompt-lookup test drafter searches this many latest tokens
SCORE_ROWS = 256                      # a scored prompt's rows a head matmul (its fp32 logits: 78 MB a rank)
FLOOR_GIB = 9.0                       # MemAvailable every node keeps (the recipe's floor)
RESERVE_GIB = 0.6                     # beyond the floor on rank 0: what serving grows after the pool (0.28 GiB
                                      # measured at 2,048-row chunks: the first long prompt's 0.10 GiB of allocator
                                      # blocks, the HTTP app, requests; 0.51 GiB at 4,096-row chunks with the split
                                      # gathers, 0.32 GiB of it allocator blocks)
FOLLOWER_RESERVE_GIB = 0.5            # on the other ranks (0.21 GiB measured at 2,048 rows; 0.32 GiB of allocator
                                      # blocks at 4,096 with the split)
TEXT_BYTES = 680                      # rank 0's host memory a prompt token takes while the server tokenizes a text
                                      # prompt (643 B measured: 272 MiB for 443,552 tokens of code, freed after): rank
                                      # 0 keeps it for a text prompt as long as the window
SIZE_SECONDS = 3.0                    # the pool is sized from MemAvailable's lowest over this long
SETTLE_SECONDS = 240.0                # ... once MemAvailable stops rising (a node gives memory back for a minute or
SETTLE_MIB = 128                      # more after loading): up by less than this over the last 10 s, at most this long


class StepTimes:
    """Debug (TF_BEAM_TIMES=N): each phase's time in a decode step, between synchronized marks (so the phases do not
    overlap as they do unmarked), printed every N decode steps; idle time between bursts is left out.
    TF_BEAM_PROFILE=A,N also prints torch's profile of decode steps A + 1 .. A + N on ranks 1.. (Kineto refuses rank 0's
    scheduler thread; every rank runs the same kernels)."""

    def __init__(self, every: int, rank: int) -> None:
        self.every, self.rank, self.acc, self.rows, self.steps, self.last = every, rank, {}, 0, 0, None
        at = [int(v) for v in os.environ.get("TF_BEAM_PROFILE", "").split(",") if v.strip()]
        self.profile_at, self.profile = (at + [0, 0])[:2] if len(at) != 1 else (at[0], at[0])
        self.decodes, self.prof, self.distinct = 0, None, 0.0

    def mark(self, name: str) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self.last is not None:
            self.acc[name] = self.acc.get(name, 0.0) + now - self.last
        self.last = now

    def experts(self, picks: list, E: int) -> None:
        """The distinct experts a MoE layer of this step read (its rows' picks below E), summed over the layers."""
        if picks:
            n = [int((torch.bincount(p.reshape(-1).long(), minlength=E + 1)[:E] > 0).sum()) for p in picks]
            self.distinct += sum(n) / len(n)

    def begin(self) -> None:
        self.decodes += 1
        if self.profile and self.rank != 0 and self.decodes == self.profile_at + 1:
            self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA])
            self.prof.__enter__()
        self.mark("between steps")

    def end(self, rows: int) -> None:
        self.steps += 1
        self.rows += rows
        if self.prof is not None and self.decodes == self.profile_at + self.profile:
            torch.cuda.synchronize()
            self.prof.__exit__(None, None, None)
            print(f"[beam] rank {self.rank} profile of {self.profile} decode steps:\n"
                  + self.prof.key_averages().table(sort_by="cuda_time_total", row_limit=45), flush=True)
            self.prof = None
        if self.steps == self.every:
            n = self.steps
            parts = ", ".join(f"{k} {v / n * 1e3:.1f}" for k, v in self.acc.items())
            print(f"[beam] rank {self.rank} decode step over {n} steps ({self.rows / n:.1f} rows, "
                  f"{self.distinct / n:.1f} experts a MoE layer): {parts}, total "
                  f"{sum(self.acc.values()) / n * 1e3:.1f} ms", flush=True)
            self.acc, self.rows, self.steps, self.distinct = {}, 0, 0, 0.0


def _f64_bits(x: float) -> int:
    return int(np.array([x], dtype=np.float64).view(np.int64)[0])


def _bits_f64(b: int) -> float:
    return float(np.array([b], dtype=np.int64).view(np.float64)[0])


def pack_sampling(s: Sampling | None) -> list[int]:
    """7 ints: a draw's settings as their exact bits (greedy: all zero)."""

    if s is None or s.temperature <= 0:
        return [0] * 7
    seed = int(s.seed) & 0xFFFFFFFFFFFFFFFF
    return [1, int(s.top_k), seed & 0xFFFFFFFF, seed >> 32, _f64_bits(s.temperature), _f64_bits(s.top_p),
            _f64_bits(s.min_p)]


def unpack_sampling(w) -> Sampling | None:
    if not w[0]:
        return None
    return Sampling((int(w[3]) << 32) | int(w[2]), _bits_f64(w[4]), int(w[1]), _bits_f64(w[5]), _bits_f64(w[6]))


@dataclass
class Plan:
    op: int
    kind: int = DECODE
    releases: list[int] = field(default_factory=list)
    reserves: list[tuple[int, int]] = field(default_factory=list)       # (slot, positions)
    seqs: list[tuple[int, int, int]] = field(default_factory=list)      # (slot, first position, rows)
    ids: list[int] = field(default_factory=list)
    samples: list[tuple[int, int, Sampling | None]] = field(default_factory=list)   # (row, position drawn, sampling)
    score: int = 0                    # a scored prompt chunk: each row's top-``score`` log-probs and the next token's
    next_ids: list[int] = field(default_factory=list)                   # ... the token after each row (-1: none)
    ctx: list[tuple[int, int, int, int, int]] = field(default_factory=list)  # the drafter's context to write (CTX_*)
    drafted: int = 0                  # a sequence's rows after its first are the device drafter's (else from ids)
    evicts: list[int] = field(default_factory=list)                     # prompt cache entries dropped first
    restores: list[tuple[int, int]] = field(default_factory=list)       # (entry, slot): a slot starts from an entry
    saves: list[tuple[int, int, int]] = field(default_factory=list)     # (slot, entry, positions) kept after the step

    def encode(self) -> tuple[list[int], list[int]]:
        payload = list(self.releases)
        for slot, n in self.reserves:
            payload += [slot, n]
        for s in self.seqs:
            payload += list(s)
        payload += list(self.ids)
        for row, pos, sp in self.samples:
            payload += [row, pos, *pack_sampling(sp)]
        if self.score:
            payload += list(self.next_ids)
        for e in self.ctx:
            payload += list(e)
        payload += list(self.evicts)
        for e in self.restores:
            payload += list(e)
        for e in self.saves:
            payload += list(e)
        return [self.op, self.kind, len(self.seqs), len(self.ids), len(self.releases), len(self.reserves),
                len(self.samples), len(payload), self.score, len(self.ctx), self.drafted, len(self.evicts),
                len(self.restores), len(self.saves)], payload

    @classmethod
    def decode(cls, header: list[int], payload: list[int]) -> "Plan":
        op, kind, S, R, nrel, nres, K, _, score, nctx, drafted, nev, nrs, nsv = header
        i = 0
        releases = payload[i:i + nrel]
        i += nrel
        reserves = [(payload[i + 2 * j], payload[i + 2 * j + 1]) for j in range(nres)]
        i += 2 * nres
        seqs = [tuple(payload[i + 3 * j:i + 3 * j + 3]) for j in range(S)]
        i += 3 * S
        ids = payload[i:i + R]
        i += R
        samples = []
        for _ in range(K):
            w = payload[i:i + SAMPLE_WORDS]
            i += SAMPLE_WORDS
            samples.append((w[0], w[1], unpack_sampling(w[2:])))
        next_ids = payload[i:i + R] if score else []
        i += R if score else 0
        ctx = [tuple(payload[i + CTX_WORDS * j:i + CTX_WORDS * (j + 1)]) for j in range(nctx)]
        i += CTX_WORDS * nctx
        evicts = payload[i:i + nev]
        i += nev
        restores = [tuple(payload[i + 2 * j:i + 2 * j + 2]) for j in range(nrs)]
        i += 2 * nrs
        saves = [tuple(payload[i + 3 * j:i + 3 * j + 3]) for j in range(nsv)]
        return cls(op, kind, releases, reserves, seqs, ids, samples, score, next_ids, ctx, drafted, evicts, restores,
                   saves)


@dataclass(eq=False)
class Request:
    prompt: list[int]
    max_tokens: int
    sampling: Sampling | None
    stop_eos: bool
    need: int                          # positions its pages hold: the prompt and the reply
    slot: int = -1
    filled: int = 0                    # prompt positions written
    out: list[int] = field(default_factory=list)
    cancel: bool = False
    done: bool = False
    q: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)
    arrived: float = field(default_factory=time.perf_counter)
    started: float = 0.0
    first_at: float = 0.0
    rounds: int = 0
    stats: dict = field(default_factory=dict)
    score: int = 0                     # prompt log-probs asked (vLLM's prompt_logprobs): top-``score`` a position
    scores: list = field(default_factory=list)
    draft: bool = True                 # the request's steps carry drafts (when the engine has a drafter)
    drafted: int = 0
    accepted: int = 0
    k: int = 0                         # its drafts a step while adaptive (0: not yet set)
    kept: float = 0.0                  # its drafts kept a step, a moving average (adaptive drafts)
    salt: object = None                # its cache_salt: prompt cache entries serve only requests with the same salt
    cached: int = 0                    # prompt positions taken from a prompt cache entry
    save_at: int = 0                   # the prompt position its prefill keeps as an entry (0: none)
    source: int = -1                   # the entry it started from (-1: none)


@dataclass(eq=False)
class Entry:
    """Rank 0's record of a prompt cache entry: the cache_salt and the first ids of the prompt it keeps, and its last
    use."""

    salt: object
    ids: list[int]
    used: int


def prompt_logprobs(prompt: list[int], scores: list, k: int) -> list:
    """vLLM's prompt_logprobs: None, then for each later prompt token its position's top-``k`` ids and the token
    itself, each {"logprob", "rank"} (row i - 1's distribution is token i's)."""

    ids = np.concatenate([s[0] for s in scores])
    lps = np.concatenate([s[1] for s in scores])
    nxt = np.concatenate([s[2] for s in scores])
    out: list = [None]
    for i in range(1, len(prompt)):
        row_ids, row_lp = ids[i - 1].tolist(), lps[i - 1].tolist()
        entry = {str(t): {"logprob": lp, "rank": j + 1} for j, (t, lp) in enumerate(zip(row_ids, row_lp))}
        tok = str(prompt[i])
        if tok not in entry:
            entry[tok] = {"logprob": float(nxt[i - 1]), "rank": k + 1}
        out.append(entry)
    return out


def release_host() -> None:
    """Hand freed host memory back to the system: the pinned staging the loader left in torch's host cache and the
    heap's free pages (on GB10 the GPU's memory is the same memory)."""
    gc.collect()
    torch._C._host_emptyCache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def lookup_drafts(hist: list[int], k: int) -> list[int]:
    """Prompt lookup (a test drafter): up to k tokens that followed the latest earlier occurrence of the history's last
    3, else 2, else 1 tokens, within the latest LOOKUP_SPAN tokens."""

    h = hist[-LOOKUP_SPAN:]
    for m in (3, 2, 1):
        if len(h) <= m:
            continue
        tail = h[-m:]
        for i in range(len(h) - m - 1, -1, -1):
            if h[i:i + m] == tail:
                return h[i + m:i + m + k]
    return []


def mem_available() -> int:
    with open("/proc/meminfo") as f:
        for row in f:
            if row.startswith("MemAvailable:"):
                return int(row.split()[1]) * 1024
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def mem_low(seconds: float) -> int:
    """MemAvailable's lowest over ``seconds`` (the node's other users breathe)."""
    low, end = mem_available(), time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(0.05)
        low = min(low, mem_available())
    return low


def mem_settle(longest: float, rise: int, span: float = 10.0) -> tuple[float, int]:
    """Wait until MemAvailable stops rising: up by less than ``rise`` bytes over the last ``span`` seconds, at most
    ``longest`` seconds. After the load and the warm-up a node gives memory back for a while (one node within a
    minute; another rose 2.2 GiB in the 2 minutes after its pool was sized, so its pool came out 0.15M positions short).
    Returns (seconds waited, bytes MemAvailable rose meanwhile)."""
    t0 = time.monotonic()
    first = mem_available()
    seen = deque([(t0, first)])
    while time.monotonic() - t0 < longest:
        time.sleep(0.5)
        now, avail = time.monotonic(), mem_available()
        seen.append((now, avail))
        while now - seen[0][0] > span:
            seen.popleft()
        if now - t0 >= span and avail - min(v for _, v in seen) < rise:
            break
    return time.monotonic() - t0, mem_available() - first


def boost_room() -> int:
    """Bytes the kernel's watermark boost can still take off MemAvailable. A fragmentation event raises a zone's
    watermarks by up to a pageblock (512 MiB on 64 KiB pages) until kswapd clears it, and MemAvailable takes the low
    watermark off its page cache and its reclaimable kernel memory: what the largest boost would take, less what the
    boost takes now (nothing when the zones are boosted already, or with vm.watermark_boost_factor 0)."""
    page = os.sysconf("SC_PAGE_SIZE")
    try:
        factor = int(open("/proc/sys/vm/watermark_boost_factor").read())
    except (OSError, ValueError):
        factor = 15000
    try:
        block = int(open("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size").read()) // page
    except (OSError, ValueError):
        block = (512 << 20 if page == 65536 else 2 << 20) // page
    if not factor:
        return 0
    zones, low_now, low_most = [], 0, 0
    for row in open("/proc/zoneinfo"):
        f = row.split()
        if f[:1] == ["Node"]:
            zones.append({})
        elif zones and len(f) == 2 and f[0] in ("boost", "low", "high", "managed"):
            zones[-1].setdefault(f[0], int(f[1]))
    total = sum(z.get("managed", 0) for z in zones)
    for z in zones:
        boost, low, high = z.get("boost", 0), z.get("low", 0), z.get("high", 0)
        low_now += low
        if high - boost > 0 and z.get("managed", 0) * 8 >= total:   # the node's large zones (Normal on GB10)
            low_most += low - boost + max(block, (high - boost) * factor // 10000)
        else:
            low_most += low
    kb = {}
    with open("/proc/meminfo") as f:
        for row in f:
            k, _, v = row.partition(":")
            kb[k] = int(v.split()[0])
    cache = (kb.get("Active(file)", 0) + kb.get("Inactive(file)", 0)) * 1024 // page
    slab = kb.get("KReclaimable", kb.get("SReclaimable", 0)) * 1024 // page

    def taken(low: int) -> int:
        return min(cache // 2, low) + min(slab // 2, low)

    return max(0, taken(low_most) - taken(low_now)) * page


def mem_fields() -> dict[str, int]:
    """MiB of the node's meminfo fields and this process's RSS parts (TF_BEAM_MEMDEBUG's deltas)."""
    out: dict[str, int] = {}
    for path, keys in (("/proc/meminfo", ("MemAvailable", "MemFree", "Cached", "AnonPages", "Shmem", "SUnreclaim",
                                          "SReclaimable", "PageTables", "Unevictable", "Mapped")),
                       ("/proc/self/status", ("RssAnon", "RssFile", "RssShmem", "VmPin"))):
        with open(path) as f:
            for row in f:
                k, _, v = row.partition(":")
                if k in keys:
                    out[k] = int(v.split()[0]) // 1024
    return out


def mem_delta(before: dict[str, int], what: str) -> str:
    now = mem_fields()
    return f"[beam] memory {what}: " + ", ".join(f"{k} {now[k] - before[k]:+d}" for k in before if k in now) + " MiB"


class BeamEngine:
    """Beam-1.0 on ``world`` ranks (this one ``rank``): weights, the KV pool, the scheduler (rank 0) or follower."""

    concurrent = True                 # the server calls generate from each request's thread; the scheduler batches
    prompt_cache = False              # no prompt rows are kept across requests

    def __init__(self, model_dir: str | Path, *, rank: int, world: int, master: str, port: int,
                 context: int | None = None, slots: int = DEFAULT_SLOTS, chunk: int = DEFAULT_CHUNK,
                 kv_dtype: str = "fp8", comm=None, drafter: str = "", spec_k: int = DEFAULT_SPEC_K) -> None:
        from tensorfold.cuda.comm import open_comm
        from tensorfold.families.bailing_v3.cuda.engine import memory_line, source_fingerprint
        from tensorfold.families.beam.cuda import forward as F
        from tensorfold.families.beam.cuda import weights as W

        if kv_dtype not in ("fp8", "bf16"):
            raise ValueError(f"Beam's full layers keep fp8 or bf16 keys and values, not {kv_dtype!r}")
        torch.cuda.set_device(0)
        self.device = torch.device("cuda")
        self.rank, self.world, self.tp = rank, world, world
        self.model_dir = Path(model_dir)
        self.memory_line = memory_line
        self.comm = comm if comm is not None else (open_comm(rank, world, master, port) if world > 1 else None)
        if self.comm is not None:
            self.comm.barrier()
        self.lookup = os.environ.get("TF_BEAM_DRAFTER", "") == "lookup"
        self.entries = int(os.environ.get("TF_BEAM_PROMPT_CACHE", PROMPT_CACHE) or 0)
        self.fine = int(os.environ.get("TF_BEAM_CACHE_FINE", CACHE_FINE) or 0)
        self.prompt_cache = self.entries > 0      # the server: prompt rows are kept across requests ...
        self.salt_scoped = True                   # ... and only reused under the same cache_salt
        mine = [source_fingerprint(), slots, chunk, int(kv_dtype == "fp8"), int(context or 0), int(spec_k),
                int(bool(drafter)), int(self.lookup), self.entries]
        seen = self._gather_ints(mine)
        if any(row != seen[0] for row in seen):
            raise RuntimeError("the ranks were started differently (source, slots, chunk, fp8, context, drafts, "
                               f"drafter, lookup, prompt cache): {seen}")
        self.cfg = Config.read(model_dir)
        self.eos = tuple(self.cfg.eos)
        self.slots, self.chunk = int(slots), int(chunk)
        self.full_dtype = torch.float8_e4m3fn if kv_dtype == "fp8" else torch.bfloat16
        self.kv_dtype = kv_dtype
        t0 = time.time()
        vocab_rows = [int(r) for r in os.environ.get("TF_BEAM_VOCAB_ROWS", "").split(",") if r.strip()]
        asked = (vocab_rows + [0] * world)[:world]
        if any(row != asked for row in self._gather_ints(asked)):
            raise RuntimeError("the ranks were started with different TF_BEAM_VOCAB_ROWS")
        self.w = W.load(model_dir, rank=rank, world=world, embed=os.environ.get("TF_BEAM_EMBED", "host"),
                        vocab_rows=vocab_rows or None)
        self.drafter = None
        self.spec_k = int(spec_k) if (drafter or self.lookup) else 0
        if drafter and not self.lookup:
            from tensorfold.families.beam.cuda.drafter import DFlash2

            self.drafter = DFlash2(Path(drafter), self.w, self.comm, slots=slots, k=self.spec_k, device=self.device,
                                   entries=self.entries)
        self._pending: list[tuple[int, int, int, int, int]] = []   # rank 0: context entries for the next plan
        release_host()
        print(memory_line(torch, rank, f"loaded in {time.time() - t0:.0f} s" + (
            ", embedding in the page cache" if self.w.embed_host is not None else "") + (
            f" (drafter {drafter}: {self.drafter.nbytes() / 2 ** 30:.2f} GiB, {self.spec_k} drafts a step)"
            if self.drafter is not None else
            f" (prompt-lookup drafts, {self.spec_k} a step)" if self.lookup else "")), flush=True)
        if self.comm is not None:
            self.comm.ready("loading")
            self.comm.barrier()
        if os.environ.get("TF_BEAM_EXPANDABLE", "1") == "1":     # later segments grow in place: less cached slack
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        self.F = F
        self._bell = 0
        self.request = threading.local()   # the calling request's options the app sets (prompt_logprobs)
        if rank != 0:
            self._boot(context)
            return
        self.cv = threading.Condition()
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.releasing: list[int] = []
        self.ent: list[Entry | None] = [None] * self.entries     # the prompt cache's entries
        self._uses = 0
        self.failed: BaseException | None = None
        self.debt = 0.0                    # seconds of steps the decoding sequences are owed (DECODE_SHARE)
        self.share = float(os.environ.get("TF_BEAM_DECODE_SHARE", DECODE_SHARE))
        self.decode_rows = int(os.environ.get("TF_BEAM_DECODE_ROWS", "0") or 0)   # a decode step's rows at most
        self.k_cap = int(os.environ.get("TF_BEAM_K_CAP", "0") or 0)            # a sequence's drafts at most
        # adaptive drafts (TF_BEAM_K_ADAPT=lo,hi): a sequence's drafts a step follow the drafts it keeps a step (a
        # moving average, weight TF_BEAM_K_EMA on the latest step): lo, plus one for each threshold of TF_BEAM_K_THR it
        # reaches (spec_k at most). Defaults from the seeded rows: prose keeps 1.48 at k=3 and 1.80 at k=5 (best at 3),
        # code 2.33 and 3.26 (best at 5); thresholds 2.0/2.4 against 1.75/2.0: one stream code 46.0 / 45.6 tok/s,
        # prose 32.4 / 32.0 (weight 0.2: code 44.0).
        self.k_adapt = tuple(int(v) for v in os.environ.get("TF_BEAM_K_ADAPT", "").split(",") if v.strip())
        self.k_thr = tuple(float(v) for v in os.environ.get("TF_BEAM_K_THR", "2.0,2.4").split(",") if v.strip())
        self.k_ema = float(os.environ.get("TF_BEAM_K_EMA", "0.3"))
        switch = os.environ.get("TF_BEAM_SWITCH", "")   # the GIL's switch interval in seconds (Python's 0.005):
        if switch:                         # the scheduler waits at most this long for it after a request's thread
            sys.setswitchinterval(float(switch))
        self.knobs = os.environ.get("TF_BEAM_KNOBS", "")   # test: a JSON file of rank 0's knobs ({"decode_rows": n,
        self._knobs_at = None                               # "switch": s, "k": n}), read again whenever it changes
        self.test_pieces = 0                                # test (knob "pieces"): scored prompts in shorter pieces
        booted, failed = threading.Event(), []

        def run() -> None:                 # the warm-up runs on the scheduler's thread: its cuBLAS state is sized in
            try:
                self._boot(context)
            except BaseException as e:     # the constructor raises it
                failed.append(e)
                booted.set()
                return
            booted.set()
            self._loop()

        threading.Thread(target=run, name="beam-scheduler", daemon=True).start()
        booted.wait()
        if failed:
            raise failed[0]

    def _boot(self, context: int | None) -> None:
        """Warm up, then size the KV pool from what MemAvailable leaves and allocate it, on the thread that runs every
        later forward (rank 0's scheduler, the others' main thread)."""

        from tensorfold.families.beam.cuda.cache import Cache

        debug = os.environ.get("TF_BEAM_MEMDEBUG", "") in ("1", "2")
        if os.environ.get("TF_BEAM_MEMDEBUG", "") == "2":
            self._first_uses()
        start = mem_fields() if debug else None
        self.work = self.F.Work(self.w, self.comm, rows=self.chunk, device=self.device)
        self._gather_buf = None
        self._high = None
        every = int(os.environ.get("TF_BEAM_TIMES", "0") or 0)
        self.times = StepTimes(every, self.rank) if every > 0 else None
        # debug: torch's profile of serving's Nth full prompt chunks on ranks 1.. (TF_BEAM_PROFILE_PREFILL=N[,M..]; a
        # pool sized well under the floor: the profiler's host buffers take ~1 GB)
        self._prof_prefill = {int(v) for v in os.environ.get("TF_BEAM_PROFILE_PREFILL", "").split(",") if v.strip()}
        self._prefills = 0
        self._warmup(context)
        release_host()
        print(self.memory_line(torch, self.rank, f"warmed up (peak allocated "
                                                 f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB)"), flush=True)
        if debug:
            print(mem_delta(start, "warm-up"), flush=True)
            start = mem_fields()
        positions = self._pool_positions(context)
        self.cache = Cache(self.cfg, self.world, pages=positions // PAGE, page=PAGE, seqs=self.slots,
                           chunk=self.chunk, full_dtype=self.full_dtype, device=self.device, entries=self.entries)
        self.limit = self.cache.capacity()
        self.context_window = self.limit
        print(self.memory_line(torch, self.rank, f"KV pool {self.limit} positions "
                                                 f"({self.cache.nbytes() / 2 ** 30:.2f} GiB, {self.kv_dtype} full "
                                                 f"layers, {self.slots} slots, {self.entries} prompt cache entries; "
                                                 f"{self.sizing})"), flush=True)
        if debug:
            print(mem_delta(start, f"the pool ({self.cache.nbytes() >> 20} MiB)"), flush=True)
        self._high = torch.cuda.memory_reserved()   # what torch holds after the pool: any growth is logged

    # -- collectives -----------------------------------------------------------------------------------------------

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        if self.comm is None:
            return [list(values)]
        mine = torch.tensor(values, dtype=torch.int64, device=self.device)
        got = torch.empty((self.world * len(values),), dtype=torch.int64, device=self.device)
        self.comm.all_gather(mine, got)
        return got.view(self.world, len(values)).tolist()

    def _share(self, values: list[int] | None, n: int) -> list[int]:
        """Rank 0's ``n`` ints on every rank."""

        if self.comm is None:
            return list(values)
        send = (torch.tensor(values, dtype=torch.int64, device=self.device) if self.rank == 0
                else torch.zeros((n,), dtype=torch.int64, device=self.device))
        got = torch.empty((self.world * n,), dtype=torch.int64, device=self.device)
        self.comm.all_gather(send, got)
        return list(values) if self.rank == 0 else got[:n].tolist()

    def _send(self, plan: Plan) -> None:
        if self.comm is None:
            return
        header, payload = plan.encode()
        self._share(header, HEADER)
        if payload:
            self._share(payload, len(payload))

    def _receive(self) -> Plan:
        header = self._share(None, HEADER)
        payload = self._share(None, header[7]) if header[7] else []
        return Plan.decode(header, payload)

    def _ring(self) -> None:
        store = getattr(self.comm, "store", None)
        if store is not None:
            self._bell += 1
            for r in range(1, self.world):
                store.set(f"tf_beam_step_{self._bell}/{r}", b"1")

    def _await_bell(self) -> None:
        store = getattr(self.comm, "store", None)
        if store is None:
            return
        from datetime import timedelta

        key = f"tf_beam_step_{self._bell + 1}/{self.rank}"
        while True:
            try:
                store.wait([key], timedelta(hours=1))
                break
            except Exception as e:                # an idle hour: wait again (a lost rank 0 raises otherwise)
                if "timeout" not in str(e).lower():
                    raise
        store.delete_key(key)
        self._bell += 1

    # -- one step on every rank ------------------------------------------------------------------------------------

    def _execute(self, plan: Plan, cache=None) -> list[int] | None:
        from tensorfold.families.beam.cuda.cache import batch

        c = self.cache if cache is None else cache
        dr = self.drafter
        for slot in plan.releases:
            c.release(slot)
        for e in plan.evicts:
            c.release(c.seqs + e)
        for e, slot in plan.restores:
            c.restore(e, slot)
            if dr is not None:
                dr.copy_context(c.seqs + e, slot)
        for slot, n in plan.reserves:
            c.reserve(slot, n)
        t = self.times if (self.times is not None and cache is None and plan.op == STEP
                           and plan.kind == DECODE) else None
        if t is not None:
            t.begin()
        prof = None
        if (self._prof_prefill and self.rank != 0 and cache is None and plan.op == STEP and plan.kind == PREFILL
                and len(plan.ids) == self.chunk):     # full chunks only: the Nth of serving's prompts
            self._prefills += 1
            if self._prefills in self._prof_prefill:
                torch.cuda.synchronize()
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA])
                prof.__enter__()
                t_prof = time.perf_counter()
        if dr is not None:
            prev = [e for e in plan.ctx if e[4] == CTX_PREV]
            if prev:
                dr.context(prev, self.work)
        if plan.op != STEP:
            if self.times is not None:
                self.times.last = None             # the wait before the next burst is not a decode step's
            return None
        if t is not None:
            t.mark("drafter context")
        ids = torch.tensor(plan.ids, dtype=torch.int64)    # host: the embedding gathers its rows there
        if plan.drafted:                          # the drafter writes each sequence's rows after its first
            sampling = {row: sp for row, _, sp in plan.samples}
            starts, at = [], 0
            for slot, pos, rows in plan.seqs:
                if rows > 1:
                    starts.append((slot, pos, at, rows - 1, sampling.get(at)))
                at += rows
            dr.draft(ids, starts, self.work)
        if t is not None:
            t.mark("draft")
        b = batch(c, list(plan.seqs), self.device, chunk=plan.kind == PREFILL)
        if t is not None:
            t.mark("batch")
            self.work.picks = []
        taps = dr is not None and (plan.kind == DECODE or any(e[4] == CTX_THIS for e in plan.ctx))
        if taps:
            dr.start(len(plan.ids))
        y = self.F.forward(self.w, ids, b, c, self.work, capture=dr.layers if taps else (),
                           tap=dr.tap if taps else None)
        if t is not None:
            t.mark("forward")
            t.experts(self.work.picks, self.cfg.experts)
            self.work.picks = None
        if dr is not None:
            this = [e for e in plan.ctx if e[4] == CTX_THIS]
            if this:
                dr.context(this, self.work)
        for slot, e, m in plan.saves:             # after the step's rings and context: the entry's copies end at m
            c.release(c.seqs + e)                 # (what it held goes now, after this step's restores)
            c.save(slot, e, m)
            if dr is not None:
                dr.copy_context(slot, c.seqs + e)
        scores = self._score(y, plan.next_ids, plan.score) if plan.score else None
        tokens = None
        if plan.samples:
            rows = torch.tensor([row for row, _, _ in plan.samples], dtype=torch.int64, device=self.device)
            tokens = self._sample(self.F.logits(self.w, y.index_select(0, rows)), plan.samples)
        drafts = ids.tolist() if plan.drafted and self.rank == 0 else None
        if t is not None:
            t.mark("head and sampling")
            t.end(len(plan.ids))
        elif self.times is not None and cache is None:
            self.times.last = None                         # a prefill step's time is not a decode step's
        if prof is not None:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            print(f"[beam] rank {self.rank} profile of prompt chunk {self._prefills} ({len(plan.ids)} rows at "
                  f"{plan.seqs[0][1]}, {time.perf_counter() - t_prof:.3f} s):\n"
                  + prof.key_averages().table(sort_by="cuda_time_total", row_limit=40), flush=True)
        if self._high is not None and torch.cuda.memory_reserved() > self._high:
            self._high = torch.cuda.memory_reserved()
            print(self.memory_line(torch, self.rank, f"grew after a {'prefill' if plan.kind == PREFILL else 'decode'}"
                                                     f" step of {len(plan.ids)} rows at {plan.seqs[:4]}"), flush=True)
        return tokens, scores, drafts

    def _score(self, y: torch.Tensor, next_ids: list[int], k: int):
        """Every row's top-``k`` log-probs over the whole vocabulary and the next token's (the ranks' shards merged:
        their logsumexps added, their top-k candidates ranked together); rank 0 returns (ids, log-probs, next
        token's log-prob) as host arrays [R, k], [R, k], [R], the others None."""

        off, width = self.w.head_lo, self.w.head.shape[0]
        nid_all = torch.tensor(next_ids, dtype=torch.int64, device=self.device) - off
        out_ids, out_lp, out_next = [], [], []
        for r0 in range(0, y.shape[0], SCORE_ROWS):
            r1 = min(y.shape[0], r0 + SCORE_ROWS)
            lg = self.F.logits(self.w, y[r0:r1])
            lse = torch.logsumexp(lg, dim=-1)
            tv, ti = torch.topk(lg, k, dim=-1)
            nid = nid_all[r0:r1]
            inside = (nid >= 0) & (nid < width)
            nxt = torch.where(inside, lg.gather(1, nid.clamp(0, width - 1)[:, None])[:, 0],
                              torch.full_like(lse, float("-inf")))
            packed = torch.cat([lse[:, None].view(torch.int32), nxt[:, None].view(torch.int32),
                                tv.contiguous().view(torch.int32), (ti + off).to(torch.int32)], dim=1).contiguous()
            if self.comm is not None:
                got = torch.empty((self.world * packed.numel(),), dtype=torch.int32, device=self.device)
                self.comm.all_gather(packed, got)
                got = got.view(self.world, r1 - r0, -1)
            else:
                got = packed[None]
            if self.rank != 0:
                continue
            vals = got.view(torch.float32)
            total = torch.logsumexp(vals[:, :, 0].double(), dim=0)                     # [b]
            cand_v = torch.cat([vals[r, :, 2:2 + k] for r in range(self.world)], dim=1)
            cand_i = torch.cat([got[r, :, 2 + k:2 + 2 * k] for r in range(self.world)], dim=1)
            best_v, at = torch.topk(cand_v, k, dim=-1)
            out_ids.append(cand_i.gather(1, at).cpu().numpy())
            out_lp.append((best_v.double() - total[:, None]).float().cpu().numpy())
            out_next.append((vals[:, :, 1].double().max(dim=0).values - total).float().cpu().numpy())
        if self.rank != 0:
            return None
        return np.concatenate(out_ids), np.concatenate(out_lp), np.concatenate(out_next)

    def _sample(self, lg: torch.Tensor, samples: list[tuple[int, int, Sampling | None]]) -> list[int] | None:
        """Every rank's candidates for the sampled rows; rank 0 returns the drawn tokens, the others None."""

        from tensorfold.cuda.sampling import comm_gather, nucleus_rows

        K, width = lg.shape
        off = self.w.head_lo
        gv, gi = lg.max(dim=-1)                       # greedy: this shard's first maximum
        parts = [gv[:, None].contiguous().view(torch.int32), (gi + off).to(torch.int32)[:, None]]
        C = min(width, max((sp.top_k + MARGIN for _, _, sp in samples if sp is not None and sp.top_k), default=0))
        if C:
            tv, ti = torch.topk(lg, C, dim=-1, sorted=False)
            parts += [tv.contiguous().view(torch.int32), (ti + off).to(torch.int32)]
        packed = torch.cat(parts, dim=1).contiguous()  # [K, 2 + 2C]: fp32 bits, ids
        if self.comm is not None:
            got = torch.empty((self.world * packed.numel(),), dtype=torch.int32, device=self.device)
            self.comm.all_gather(packed, got)
            got = got.view(self.world, K, -1)
        else:
            got = packed[None]
        drawn: dict[int, int] = {}
        for j, (_, pos, sp) in enumerate(samples):     # top_k off: the shared nucleus rule, a collective a row
            if sp is not None and not sp.top_k:
                gather = comm_gather(self.comm) if self.comm is not None else None
                drawn[j] = nucleus_rows(lg[j:j + 1], [pos], sp, offset=off,
                                        **({"gather": gather} if gather is not None else {}))[0]
        if self.rank != 0:
            return None
        g = got.cpu().numpy()
        vals = g.view(np.float32)
        out = []
        for j, (_, pos, sp) in enumerate(samples):
            if j in drawn:
                out.append(int(drawn[j]))
            elif sp is None:                           # the whole vocabulary's first maximum: rank 0 holds lower ids
                best = 0
                for r in range(1, self.world):
                    if vals[r, j, 0] > vals[best, j, 0]:
                        best = r
                out.append(int(g[best, j, 1]))
            else:
                v = np.concatenate([vals[r, j, 2:2 + C] for r in range(self.world)])
                i = np.concatenate([g[r, j, 2 + C:2 + 2 * C] for r in range(self.world)]).astype(np.int64)
                out.append(choose_rows(v[None], i[None], [pos], sp)[0])
        return out

    # -- memory --------------------------------------------------------------------------------------------------

    def _per_position(self) -> int:
        hkv = self.cfg.kv_heads // self.world
        return 2 * len(self.cfg.full) * hkv * self.cfg.head_dim * (1 if self.full_dtype == torch.float8_e4m3fn else 2)

    def _first_uses(self) -> None:
        """TF_BEAM_MEMDEBUG=2: what each library's first call costs the host (cuBLAS, NCCL, a Triton kernel)."""
        start, dev = mem_fields(), self.device

        def note(what: str) -> None:
            torch.cuda.synchronize()
            print(mem_delta(start, f"first use: {what}"), flush=True)

        a = torch.randn(16, 5120, device=dev).bfloat16()
        b = torch.randn(4096, 5120, device=dev).bfloat16()
        torch.mm(a, b.t(), out_dtype=torch.float32)
        note("cuBLAS bf16 matmul, fp32 out, 16 rows")
        torch.mm(torch.randn(2048, 5120, device=dev).bfloat16(), b.t())
        note("cuBLAS bf16 matmul, 2048 rows")
        if self.comm is not None:
            for n in (64, 2048 * 5120):
                x, g = torch.zeros(n, device=dev), torch.empty(n * self.world, device=dev)
                self.comm.all_gather(x, g)
                note(f"NCCL all-gather of {n * 4} bytes")
            del x, g
        self.F.ops.rms_norm(a, torch.ones(5120, device=dev, dtype=torch.bfloat16), 1e-6)
        note("a Triton kernel")
        del a, b

    def _pool_positions(self, context: int | None) -> int:
        """Positions of the full layers' pool: what MemAvailable leaves above the floor and the reserve (on rank 0 also
        a window-long text prompt's tokenizing), at most the asked context (else the checkpoint's), the same on every
        rank."""

        cfg = self.cfg
        hkv = cfg.kv_heads // self.world
        win = cfg.layers - len(cfg.full)
        rings = 2 * win * (self.slots + self.entries) * 528 * hkv * cfg.head_dim * 2
        buf = 2 * (cfg.window - 1 + self.chunk) * hkv * cfg.head_dim * 2
        floor = float(os.environ.get("TF_BEAM_FLOOR_GIB", FLOOR_GIB)) * 2 ** 30
        reserve = float(os.environ.get("TF_BEAM_RESERVE_GIB", RESERVE_GIB) if self.rank == 0 else
                        os.environ.get("TF_BEAM_FOLLOWER_RESERVE_GIB", FOLLOWER_RESERVE_GIB)) * 2 ** 30
        text = int(os.environ.get("TF_BEAM_TEXT_BYTES", TEXT_BYTES)) if self.rank == 0 else 0
        waited, rose = mem_settle(float(os.environ.get("TF_BEAM_SETTLE_SECONDS", SETTLE_SECONDS)),
                                  int(os.environ.get("TF_BEAM_SETTLE_MIB", SETTLE_MIB)) << 20)
        low, boost = mem_low(float(os.environ.get("TF_BEAM_SIZE_SECONDS", SIZE_SECONDS))), boost_room()
        room = low - boost - floor - reserve - rings - buf
        fits = max(0, int(room // (self._per_position() + text))) // PAGE * PAGE
        self.sizing = (f"settled after {waited:.0f} s ({rose / 2 ** 30:+.2f} GiB), "
                       f"room {room / 2 ** 30:.2f} GiB: MemAvailable low {low / 2 ** 30:.2f}, watermark boost "
                       f"allowance {boost / 2 ** 30:.2f}, reserve {reserve / 2 ** 30:.2f}"
                       + (f", text prompt allowance {fits * text / 2 ** 30:.2f}" if text else ""))
        want = int(context) if context else cfg.max_positions
        want = -(-want // PAGE) * PAGE
        if self.comm is not None:
            self.comm.ready("sizing the KV pool")       # each rank's settle wait is its own: meet in the store first
        both = min(row[0] for row in self._gather_ints([min(fits, want)]))
        if both < PAGE:
            raise MemoryError(f"no room for a KV pool: MemAvailable leaves {room / 2 ** 30:.1f} GiB above the "
                              f"{floor / 2 ** 30:.0f} GiB floor and the {reserve / 2 ** 30:.2f} GiB reserve")
        if both < want and self.rank == 0:
            print(f"[tensorfold] the KV pool holds {both} positions, not the {want} asked: the floor and reserve "
                  "leave no more", flush=True)
        return both

    def _warmup(self, context: int | None) -> None:
        """A three-chunk prompt (random ids; then one id, every row to the same experts: their widest per-expert rows;
        then a short last chunk: the later chunks read the window layers' rings and give the drafter its context the
        way a long prompt's do), one decode row, one drafted step, a scored prompt, a short one and a prompt chunk again
        through a throwaway pool: the kernels compile and the allocator keeps the widest step's activations (and the
        drafter's), with the decode attention partials for the whole context, before the pool is sized (nothing is
        handed back after; the throwaway pool's rings and window buffer are the real pool's shapes, so the real pool
        takes their blocks)."""

        from tensorfold.families.beam.cuda.cache import Cache

        rng = np.random.default_rng(0)
        n, k, dr = self.chunk, self.spec_k, self.drafter
        m = 2 * n + n // 4                    # the prompt's length
        start = mem_fields() if os.environ.get("TF_BEAM_MEMDEBUG", "") == "1" else None

        def note(what: str) -> None:
            if start is not None:
                torch.cuda.synchronize()
                print(mem_delta(start, f"{what} (torch reserved {torch.cuda.memory_reserved() / 2 ** 30:.2f} GiB)"),
                      flush=True)
        c = Cache(self.cfg, self.world, pages=-(-(m + 2 + k) // PAGE), page=PAGE, seqs=self.slots, chunk=n,
                  full_dtype=self.full_dtype, device=self.device, entries=self.entries)
        ids = [int(t) for t in rng.integers(0, 151643, m)]
        ids[n:2 * n] = [ids[0]] * n
        lo = max(0, n - dr.window) if dr is not None else 0
        self._execute(Plan(STEP, PREFILL, reserves=[(0, m + 2 + k)], seqs=[(0, 0, n)], ids=ids[:n],
                           ctx=[(0, lo, lo, n - lo, CTX_THIS)] if dr is not None else []), cache=c)
        note("warm-up prompt chunk")
        for p0 in (n, 2 * n):
            rows = min(n, m - p0)
            lo = max(p0, m - dr.window) if dr is not None else m
            self._execute(Plan(STEP, PREFILL, seqs=[(0, p0, rows)], ids=ids[p0:p0 + rows],
                               samples=[(rows - 1, m, None)] if p0 + rows == m else [],
                               ctx=[(0, lo, lo - p0, p0 + rows - lo, CTX_THIS)] if lo < p0 + rows else []), cache=c)
            note(f"warm-up prompt chunk at {p0}" + (" (one id: every row to the same experts)" if p0 == n else ""))
        self._execute(Plan(STEP, DECODE, seqs=[(0, m, 1)], ids=ids[:1], samples=[(0, m + 1, None)]), cache=c)
        note("warm-up decode row")
        if k:
            self._execute(Plan(STEP, DECODE, seqs=[(0, m + 1, 1 + k)], ids=ids[:1 + k],
                               samples=[(i, m + 2 + i, None) for i in range(1 + k)],
                               ctx=[(0, m, 0, 1, CTX_PREV)] if dr is not None else [], drafted=int(dr is not None)),
                          cache=c)
        note("warm-up drafted step")
        self._execute(Plan(STEP, PREFILL, seqs=[(0, 0, SCORE_ROWS)], ids=ids[:SCORE_ROWS], score=8,
                           next_ids=ids[1:SCORE_ROWS + 1]), cache=c)
        note("warm-up scored prompt")
        short = (self.F.SMALL_EXPERT_ROWS + self.work.small_rows) // 2      # the experts' one-tile prompt path
        self._execute(Plan(STEP, PREFILL, seqs=[(0, 0, short)], ids=ids[:short], samples=[(short - 1, short, None)]),
                      cache=c)
        note("warm-up short prompt")
        # a prompt chunk once more, after the short steps (serving's first long prompt still grows the allocator once,
        # by 0.10 GiB a rank at 2,048-row chunks and 0.32 GiB at 4,096 with the split gathers, inside the reserve)
        self._execute(Plan(STEP, PREFILL, seqs=[(0, 0, n)], ids=ids[:n]), cache=c)
        note("warm-up prompt chunk after the short steps")
        cfg = self.cfg
        rows = max(16, self.slots * (1 + k))
        span = int(context) if context else cfg.max_positions
        parts = -(-span // self.F.A.PART)
        self.work.att.get(parts * (cfg.kv_heads // self.world) * rows * (cfg.heads // cfg.kv_heads), self.device)
        torch.cuda.synchronize()
        del c

    # -- rank 0: requests and the scheduler ----------------------------------------------------------------------

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 stop_eos: bool = True, *, constraint=None, background: bool = False, vision=None,
                 cache_salt=None) -> dict:
        """One request (any server thread): queued, scheduled with the others, its tokens handed to ``on_tokens``
        as they are drawn (True stops it)."""

        if constraint is not None or vision is not None:
            raise ValueError("Beam's CUDA engine serves text replies without a grammar")
        prompt = [int(t) for t in prompt]
        if not prompt:
            raise ValueError("empty prompt")
        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        req = Request(prompt, max_tokens, sampling if sampling is not None and sampling.temperature > 0 else None,
                      bool(stop_eos), need=len(prompt) + max_tokens,
                      score=int(getattr(self.request, "prompt_logprobs", 0) or 0), draft=bool(draft), salt=cache_salt)
        with self.cv:
            if self.failed is not None:
                raise RuntimeError(f"the engine stopped: {self.failed!r}")
            self.waiting.append(req)
            self.cv.notify()
        stopped = False
        while True:
            item = req.q.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            if not stopped and on_tokens(item):
                stopped = True
                with self.cv:
                    req.cancel = True
                    self.cv.notify()
        if req.score and req.scores:
            req.stats["prompt_logprobs"] = prompt_logprobs(prompt, req.scores, req.score)
        return req.stats

    def _finish(self, r: Request) -> None:
        now = time.perf_counter()
        r.done = True
        self.running.remove(r)
        self.releasing.append(r.slot)
        r.stats = {"prefill_s": (r.first_at or now) - r.started, "queued_s": r.started - r.arrived, "cached": r.cached,
                   "rounds": r.rounds, "decode_s": now - (r.first_at or now), "drafted": r.drafted,
                   "accepted": r.accepted}
        r.q.put(None)

    def _plan(self) -> Plan | None:
        """The next step (under ``cv``): finish what was cancelled, admit what fits (from the prompt cache entry that
        holds the longest start of its prompt, if any), then a prompt chunk or a decode row for every decoding
        sequence. None: nothing to run."""

        for r in [r for r in self.running if r.cancel]:
            self._finish(r)
        live = {r.slot for r in self.running}
        ctx, self._pending = [e for e in self._pending if e[0] in live], []
        releases, self.releasing = self.releasing, []
        c = self.cache
        evicts, restores, saves = [], [], []
        taken = 0                                 # pages the requests admitted here take
        free_pages = len(c.free) + c.freed(releases)
        busy = {r.slot for r in self.running}
        free_slots = [s for s in range(self.slots) if s not in busy]
        reserves = []
        while self.waiting and free_slots:
            r = self.waiting[0]
            if r.cancel:
                self.waiting.popleft()
                r.stats = {"cached": 0}
                r.q.put(None)
                continue
            hit = self._lookup(r)
            pages = -(-r.need // PAGE) - (len(self.ent[hit].ids) // PAGE if hit >= 0 else 0)
            while pages > free_pages:             # entries give their pages back, least recently used first
                e = self._victim({hit} | {x for x, _ in restores}, releases, evicts)
                if e < 0:
                    break
                evicts.append(e)
                self.ent[e] = None
                free_pages = len(c.free) + c.freed(releases + [self.slots + x for x in evicts]) - taken
            if pages > free_pages:
                break
            self.waiting.popleft()
            r.slot = free_slots.pop(0)
            r.started = time.perf_counter()
            free_pages -= pages
            taken += pages
            if hit >= 0:
                restores.append((hit, r.slot))
                r.filled = r.cached = len(self.ent[hit].ids)
                r.source = hit
                self.ent[hit].used = self._use()
            if self.entries and not r.score:
                r.save_at = self._save_point(r)
            reserves.append((r.slot, r.need))
            self.running.append(r)
        filling = [r for r in self.running if r.filled < len(r.prompt)]
        decoding = [r for r in self.running if r.filled == len(r.prompt)]
        if not filling and not decoding:
            return (Plan(IDLE, releases=releases, reserves=reserves, evicts=evicts, restores=restores)
                    if releases or reserves or evicts else None)
        if filling and (not decoding or self.debt <= 0):
            # the prompt with the fewest rows left goes first, so a short chat does not wait behind a long prefill (a
            # prompt's chunks stay where they were: the bits do not change)
            r = min(filling, key=lambda q: len(q.prompt) - q.filled)
            p0 = r.filled
            end = min(p0 // self.chunk * self.chunk + self.chunk, len(r.prompt))   # pieces end on the chunk grid,
            if r.save_at > p0:                                                       # at the prompt's end and at
                end = min(end, r.save_at)                                            # the entry it keeps
            if r.score and self.test_pieces:      # a row-invariance check (knob "pieces"): a scored prompt's pieces
                cut = (p0 // self.test_pieces + 1) * self.test_pieces   # also end on this step's multiples, every
                if p0 + MIN_PIECE <= cut <= end - MIN_PIECE:             # piece MIN_PIECE rows or more
                    end = cut
            n = end - p0
            last = p0 + n == len(r.prompt)
            nxt = (r.prompt[p0 + 1:p0 + n + 1] + [-1])[:n] if r.score else []
            if self.drafter is not None:          # the prompt rows inside the drafter's window give its context (the
                lo = max(p0, (r.save_at or len(r.prompt)) - self.drafter.window)   # window before an entry's end too)
                if lo < p0 + n:
                    ctx = ctx + [(r.slot, lo, lo - p0, p0 + n - lo, CTX_THIS)]
            if r.save_at and p0 + n == r.save_at:  # an entry keeps the prompt to here, unless one already does
                e = self._entry_for(r)
                if e >= 0:
                    saves.append((r.slot, e, r.save_at))
            return Plan(STEP, PREFILL, releases, reserves, [(r.slot, p0, n)], r.prompt[p0:p0 + n],
                        [(n - 1, len(r.prompt), r.sampling)] if last else [], r.score, nxt, ctx,
                        evicts=evicts, restores=restores, saves=saves)
        seqs, ids, samples = [], [], []
        row, drafted = 0, 0
        per = max(1, self.decode_rows // len(decoding)) if self.decode_rows else 0   # rows a sequence may take
        for r in decoding:
            pos = len(r.prompt) + len(r.out) - 1
            k = max(0, min(self.spec_k, r.max_tokens - len(r.out) - 1)) if r.draft else 0
            if per:
                k = min(k, per - 1)
            if self.k_cap:
                k = min(k, self.k_cap)
            if self.k_adapt and k:
                if not r.k:
                    r.k, r.kept = self.k_adapt[0], 1.5
                k = min(k, r.k)
            d: list[int] = []
            if k and self.lookup:
                d = lookup_drafts(r.prompt + r.out, k)
            elif k:
                d = [0] * k                       # the device drafter fills these rows
                drafted = 1
            seqs.append((r.slot, pos, 1 + len(d)))
            ids += [r.out[-1]] + d
            for i in range(1 + len(d)):
                samples.append((row + i, pos + 1 + i, r.sampling))
            row += 1 + len(d)
        return Plan(STEP, DECODE, releases, reserves, seqs, ids, samples, ctx=ctx, drafted=drafted, evicts=evicts,
                    restores=restores)

    # -- the prompt cache (rank 0's records; every rank's pages and rings follow the plans) --------------------------

    def _use(self) -> int:
        self._uses += 1
        return self._uses

    def _lookup(self, r: Request) -> int:
        """The entry that holds the longest start of r's prompt (short of its last id, whose logits it needs) under
        its cache_salt, or -1. An entry past the prompt's last chunk grid point needs MIN_PIECE rows after it: a
        fresh prefill fills those rows in one piece of at least that many. A request for prompt log-probs fills every
        row."""

        if not self.entries or r.score:
            return -1
        n = len(r.prompt)
        grid = (n - 1) // self.chunk * self.chunk
        best, at = -1, 0
        for i, x in enumerate(self.ent):
            m = len(x.ids) if x is not None else 0
            if (at < m < n and (m <= grid or n - m >= MIN_PIECE) and x.salt == r.salt
                    and r.prompt[:m] == x.ids):
                best, at = i, m
        return best

    def _save_point(self, r: Request) -> int:
        """Where r's prefill keeps an entry, or 0. A fresh prompt: its last chunk grid point before its last id (no
        piece added). A prompt that came from an entry (an agent's next turn, a re-sent prompt) is likely to come
        again: its last page boundary MIN_PIECE rows before its end, when that is ``fine`` rows or more past both
        its grid point and where it resumed. Every piece is then a page-aligned part of one grid chunk with MIN_PIECE
        rows or more, or a fresh prefill's own last chunk: the rows a fresh prefill computes."""

        n = len(r.prompt)
        grid = (n - 1) // self.chunk * self.chunk
        if r.cached and self.fine:
            at = (n - MIN_PIECE) // PAGE * PAGE
            if at - max(grid, r.filled) >= self.fine:
                return at
        return grid if grid > r.filled else 0

    def _victim(self, keep: set, releases: list[int], evicts: list[int]) -> int:
        """The least recently used entry (not in ``keep``) whose eviction gives pages back, or -1."""

        c = self.cache
        dropped = releases + [self.slots + x for x in evicts]
        base = c.freed(dropped)
        for i in sorted((i for i, x in enumerate(self.ent) if x is not None and i not in keep),
                        key=lambda i: self.ent[i].used):
            if c.freed(dropped + [self.slots + i]) > base:
                return i
        return -1

    def _entry_for(self, r: Request) -> int:
        """The entry r's prompt to ``save_at`` goes to: the one r started from while it still holds a start of r's
        prompt (r's entry takes its place, so streams that resume together keep each other's entries), else an empty
        one, else the least recently used. Its old pages and rings go when the save runs (after the step's restores,
        so an entry a sequence starts from in the same step can take it); -1 when an entry already holds it."""

        m = r.save_at
        if any(x is not None and len(x.ids) == m and x.salt == r.salt and x.ids == r.prompt[:m] for x in self.ent):
            return -1
        free = [i for i, x in enumerate(self.ent) if x is None]
        src = self.ent[r.source] if r.source >= 0 else None
        if src is not None and src.salt == r.salt and src.ids == r.prompt[:len(src.ids)]:
            e = r.source
        else:
            e = free[0] if free else min(range(self.entries), key=lambda i: self.ent[i].used)
        self.ent[e] = Entry(r.salt, r.prompt[:m], self._use())
        return e

    def _deliver(self, plan: Plan, result) -> None:
        now = time.perf_counter()
        tokens, scores, drafted_ids = result
        with self.cv:
            if plan.kind == PREFILL:
                slot, p0, n = plan.seqs[0]
                r = next(r for r in self.running if r.slot == slot)
                r.filled = p0 + n
                if scores is not None:
                    r.scores.append(scores)
                got = [(r, tokens[0])] if tokens else []
                if got:
                    r.first_at = now
                for r, t in got:
                    r.out.append(int(t))
                    if not r.cancel:
                        r.q.put([int(t)])
                    if (r.stop_eos and t in self.eos) or len(r.out) >= r.max_tokens:
                        self._finish(r)
                return
            ids = drafted_ids if drafted_ids is not None else plan.ids
            by_slot = {r.slot: r for r in self.running}
            j = 0
            for slot, pos, rows in plan.seqs:
                r = by_slot[slot]
                drawn, drafts = [int(t) for t in tokens[j:j + rows]], ids[j + 1:j + rows]
                n = 0                             # drafts equal to the draws before them
                while n < rows - 1 and drafts[n] == drawn[n]:
                    n += 1
                keep: list[int] = []
                done = False
                for t in drawn[:n + 1]:
                    keep.append(t)
                    if (r.stop_eos and t in self.eos) or len(r.out) + len(keep) >= r.max_tokens:
                        done = True
                        break
                r.out += keep
                r.rounds += 1
                r.drafted += rows - 1
                r.accepted += len(keep) - 1
                if self.k_adapt and r.k and rows > 1:
                    r.kept += self.k_ema * (n - r.kept)
                    r.k = min(self.k_adapt[0] + sum(r.kept >= t for t in self.k_thr), self.k_adapt[-1], self.spec_k)
                if not r.cancel:
                    r.q.put(keep)
                if done:
                    self._finish(r)
                elif self.drafter is not None:    # rows 0..n hold the kept positions' states
                    self._pending.append((slot, pos, j, n + 1, CTX_PREV))
                j += rows

    def _read_knobs(self) -> None:
        try:
            at = os.stat(self.knobs).st_mtime_ns
        except OSError:
            return
        if at == self._knobs_at:
            return
        self._knobs_at = at
        try:
            d = json.loads(Path(self.knobs).read_text())
        except ValueError:
            return
        if "decode_rows" in d:
            self.decode_rows = int(d["decode_rows"])
        if "switch" in d:
            sys.setswitchinterval(float(d["switch"]))
        if "k" in d:
            self.k_cap = int(d["k"])
        if "adapt" in d:
            self.k_adapt = tuple(int(v) for v in d["adapt"])
        if "thr" in d:
            self.k_thr = tuple(float(v) for v in d["thr"])
        if "ema" in d:
            self.k_ema = float(d["ema"])
        if "pieces" in d:
            self.test_pieces = int(d["pieces"])
        print(f"[beam] knobs {d} (switch interval {sys.getswitchinterval()} s)", flush=True)

    def _loop(self) -> None:
        awake = False
        try:
            while True:
                if self.knobs:
                    self._read_knobs()
                with self.cv:
                    while not awake and not self.waiting:
                        self.cv.wait()
                    plan = self._plan()
                if plan is None:
                    if awake:
                        self._send(Plan(IDLE))
                        awake = False
                    if self.times is not None:
                        self.times.last = None
                    continue
                if not awake:
                    self._ring()
                    awake = True
                t0 = time.perf_counter()
                self._send(plan)
                result = self._execute(plan)
                if plan.op == STEP:
                    if plan.kind == PREFILL and not plan.samples:
                        torch.cuda.synchronize()       # the chunk's own time, for the decoding sequences' share
                    self._deliver(plan, result)
                    dt = time.perf_counter() - t0
                    with self.cv:                      # a chunk owes the decoding sequences its time; their steps pay it
                        self.debt = dt * self.share if plan.kind == PREFILL else self.debt - dt
                else:                                  # an IDLE plan sends the others back to the store
                    awake = False
        except BaseException as exc:                   # noqa: BLE001  every waiting request hears it
            import traceback

            traceback.print_exc()
            with self.cv:
                self.failed = exc
                for r in list(self.running) + list(self.waiting):
                    r.q.put(RuntimeError(f"the engine stopped: {exc!r}"))
                self.running.clear()
                self.waiting.clear()

    # -- ranks 1.. --------------------------------------------------------------------------------------------------

    def follow(self) -> None:
        """Ranks 1..: run every step rank 0 plans, forever."""

        while True:
            self._await_bell()
            while True:
                plan = self._receive()
                self._execute(plan)
                if plan.op == IDLE:
                    break
