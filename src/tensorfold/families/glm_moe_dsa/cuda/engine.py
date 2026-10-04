"""GLM-5.3 on TF_TP_WORLD NCCL ranks; both sample by one keyed rule from the same gathered candidates."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

MAX_ROWS = 8                          # the widest verify window (a pending token and up to 7 drafts)
# a prompt chunk's rows (TF_GLM_PREFILL_ROWS): 8,192 runs each micro-batch's experts on 4,096 rows, twice the rows a
# weight read of 2,048-row chunks (32.5k-token prefill on TP4: 32.0 s against 35.9 at 4,096 and ~40 at 2,048)
PROMPT_ROWS = int(os.environ.get("TF_GLM_PREFILL_ROWS") or 8192)
# the prompt buffers' bytes a row past geometry's 2,048 (measured on TP4: torch allocated 82.6 GiB at 8,192 rows
# against 77.7 at 2,048, 64k context, before the rotated expert rows went lazy), counted by the startup estimate
PROMPT_ROW_BYTES = 860_000
GRAPH_ROWS = (1, 2, 3, 4, 5, 6)       # verify windows captured as CUDA graphs
DENSE_CAPACITY = 2560                 # cache slots while DSA attention stays dense (contexts up to 2,048 tokens)
DEFAULT_POLICY = "auto"


class GlmEngine:
    """GLM-5.3 on ``TF_TP_WORLD`` ranks (this one ``rank``): EXL3 routed experts, BF16 attention, MTP drafting."""

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, policy: str = DEFAULT_POLICY,
                 drafter: Path | None = None, context: int | None = None, context_explicit: bool | None = None,
                 serial_only: bool = False, comm=None, prefill_rows: int | None = None, parallel: int = 1) -> None:
        """``comm``: a communicator with ``all_gather`` and ``barrier`` instead of NCCL between machines (tests).
        ``parallel`` > 1: up to that many requests decoded together (multi.MultiDecoder), each with the window
        ``context`` asks for, every stream's cache a slot of one pool."""

        import torch

        from tensorfold.cuda.capacity import admit
        from tensorfold.cuda.comm import NCCL
        from tensorfold.cuda.geometry import PREFILL_ROWS, mla_geometry, split_weights
        from tensorfold.families.glm5_next.cuda import latent
        from tensorfold.families.glm5_next.cuda.split import rule

        from . import dcp as dcp_mod, forward as fwd, kv8
        from .decode import Engine as Decoder
        from .weights import load

        torch.cuda.set_device(0)
        self.torch = torch
        self.rank = rank
        self.policy = "0" if serial_only else policy
        self.serial_only = serial_only
        world = int(os.environ.get("TF_TP_WORLD", "2"))
        self.world = world
        if comm is None:
            # every replay of a captured forward is synchronized before the next uncaptured collective (the decode
            # loops sample after torch.cuda.synchronize()), so NCCL can skip its graph-mixing support; with grouped
            # send/recv an in-graph all-gather of a [1, 6144] partial takes 35 us instead of 51 (tools/nccl_bench.py)
            os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")
            # one channel: a prompt chunk's row-share reductions run beside the other micro-batch's compute, and with
            # fewer NCCL blocks (and less staging traffic at once) more of them hide (26k-token prefill on TP4 with two
            # micro-batches: 41.0 s at 1 channel, 41.4 at 2, 41.8 at 4); decode partials go over RDMA (rdma.py)
            os.environ.setdefault("NCCL_MAX_NCHANNELS", "1")
            comm = NCCL(rank, world, master, port, gather=os.environ.get("TF_NCCL_GATHER") or "p2p")
            from tensorfold.families.glm_moe_dsa.cuda.weights import Config

            cfg0 = Config.read(model_dir)
            most = MAX_ROWS * max(1, int(parallel)) * cfg0.hidden * 4      # a concurrent round's rows: every stream's
            if dcp_mod.DCP > 1:                       # dcp: a decode window's partials for every rank's heads
                most = max(most, world * MAX_ROWS * (cfg0.heads // world) * (cfg0.kv_lora // 2 + 1) * 4)
            comm = _rdma(comm, rank, world, most)
        self.comm = comm
        self.comm.barrier()
        from tensorfold.families.glm_moe_dsa.cuda.weights import Config, draft_vocab, embed_split, embed_transform

        cfg = Config.read(model_dir)
        G = dcp_mod.DCP
        if G > 1 and G != world:
            raise ValueError(f"TF_GLM_DCP={G}: decode context parallelism interleaves the cache over every rank, "
                             f"so it must equal the {world} ranks")
        # GLM-5.3 has no k-pool: the dense limit is index_topk visible tokens
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        # each rank holds its vocabulary span of the embedding (weights.embed_span), not the whole table
        rows = PROMPT_ROWS if prefill_rows is None else int(prefill_rows)
        streams = max(1, int(parallel))
        if streams > 1 and G > 1:
            raise ValueError("--parallel and TF_GLM_DCP (decode context parallelism) do not run together yet")
        self.capacity_plan = admit(model_dir, context if explicit else cfg.dense_limit, explicit, torch,
                                   lambda text: self._geometry(text, world, dcp=G, rows=rows, streams=streams),
                                   embed_transform(split_weights(rule, world), cfg.vocab, world, rank),
                                   rank=rank, world=world, gather=self._gather_ints)
        self.limit = self.capacity_plan["context_window"]
        capacity = self.capacity_plan["cache_slots"]
        long_context = self.limit > cfg.dense_limit
        prefill_rows = PROMPT_ROWS if prefill_rows is None else int(prefill_rows)
        from .decode import DEPTH_COST
        from .multi import CUT_STREAMS, DRAFT_CUT, EXTENTS, FILL_ROWS, KEEP, QUICK_ROWS

        mine = [int(drafter is not None), capacity, int(long_context), int(serial_only), int(latent.ENABLED),
                prefill_rows, int(embed_split()), kv8.mode_code(kv8.MODE), draft_vocab(), G, streams,
                FILL_ROWS if streams > 1 else 0, QUICK_ROWS if streams > 1 else 0, int(DEPTH_COST * 1e6),
                int(KEEP), int(EXTENTS), int(DRAFT_CUT * 1e6), CUT_STREAMS]
        both = self._gather_ints(mine)
        if any(row != both[0] for row in both):
            raise RuntimeError("the ranks were started with different settings (draft model, context, TF_GLM_LATENT,"
                               " TF_GLM_EMBED_SPLIT, TF_GLM_KV, TF_GLM_DRAFT_VOCAB, TF_GLM_DCP, --parallel, "
                               "TF_GLM_FILL_ROWS, TF_GLM_QUICK_ROWS, TF_GLM_DEPTH_COST, TF_GLM_KEEP_SLOTS, "
                               "TF_GLM_EXTENTS, TF_GLM_DRAFT_CUT, TF_GLM_DRAFT_CUT_STREAMS):"
                               f" rank 0 {both[0]} vs {both[1:]}; give every rank the same flags")
        if rank == 0 and kv8.parse(kv8.MODE) != ("bf16", "bf16"):
            lat, idx = kv8.parse(kv8.MODE)
            name = {"bf16": "bf16", "fp8": "FP8 (e4m3, a power-of-two scale a token)"}
            print(f"[tensorfold] cache TF_GLM_KV={kv8.MODE}: the latent "
                  f"{name.get(lat) or f'in exllamav3 {lat[1:]}-bit groups (H32-rotated)'}, the indexer keys "
                  f"{name.get(idx) or f'in exllamav3 {idx[1:]}-bit groups (H32-rotated)'}, the rope key bf16", flush=True)
        if rank == 0 and G > 1:
            print(f"[tensorfold] decode context parallelism over {G} ranks: positions p % {G}, a quarter of the cache "
                  "a rank", flush=True)
        w = load(model_dir, rank=rank)
        w.comm = self.comm
        w.meta["long_context"] = long_context
        w.meta["dcp"] = G
        self.comm.barrier()
        if w.mtp is None and not serial_only:
            print("[tensorfold] this checkpoint has no MTP layer: every round decodes one token", flush=True)
            self.policy = "0"
        self.w = w
        self.master = master
        self.concurrent = streams > 1
        self.multi = self.scheduler = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import Link, MultiDecoder

            depth = 0 if self.policy == "0" else min(int(self.policy) if str(self.policy).isdigit() else 3, MAX_ROWS - 1)
            self.e = None
            self.multi = MultiDecoder(w, slots=streams, slot_cap=capacity, depth=depth, prefill_rows=prefill_rows)
            self.multi.warm()
            if world > 1:
                from .multi import Watchdog

                self.multi.watch = Watchdog(self.comm, _store(self.comm), rank=rank, world=world, host=master)
            if rank == 0:
                if world > 1:
                    self.multi.link = Link(_store(self.comm), rank=0, world=world, host=master)
                self.scheduler = Scheduler(self.multi, max_streams=streams)
                from .multi import EXTENTS

                pool = (f"extents of one {capacity}-row pool" if EXTENTS else f"a {capacity}-row cache slot each")
                print(f"[tensorfold] up to {streams} requests decode together, each with a {self.limit}-token window "
                      f"({pool}); {depth} MTP drafts a round", flush=True)
        else:
            self.e = Decoder(w, capacity=capacity, max_rows=MAX_ROWS, prefill_rows=prefill_rows, graphs=True,
                             graph_rows=GRAPH_ROWS, long_context=long_context)
        self.eos = tuple(w.cfg.eos)
        self.model_dir = Path(model_dir)
        self.request = threading.local()    # the calling request's policy and stop-at-EOS (``app.GlmApp``)
        self.cache = []                     # kept conversations (decode.Snapshot), least recently used first
        self.live: list[int] = []

    @staticmethod
    def _geometry(text: dict, world: int, kv: str | None = None, dcp: int = 1, rows: int | None = None,
                  streams: int = 1):
        """The Flash MLA geometry minus the KDA and hyper-connection terms, with the MTP head counted, and this family's
        own cache a slot: Flash counts a 512-wide latent and three indexer planes a layer, this family keeps the latent
        plus a 64-wide rope key a layer (and the MTP head's) and one indexer key plane a full-indexer layer (96,640 B a
        slot at TP4 against Flash's 126,912; TF_GLM_KV idx8 93,912, fp8 53,780, q8 55,992, q6 45,880, q5 40,824, q4
        35,768), and its token selection holds at most select.SELECT_BYTES at once. With ``dcp`` ranks interleaving the
        positions a rank holds 1/dcp of every slot, plus the prompt chunks' exchange buffers (dcp.Scratch). A quantized
        latent without dcp adds the prompt chunks' latent scratch (forward.qscratch_slots, 1 KiB a slot), and prompt
        chunks wider than geometry's 2,048 rows PROMPT_ROW_BYTES a row. With ``streams`` > 1 (--parallel) every slot
        of the window is held once a stream, plus the rounds' wider buffers."""

        from tensorfold.cuda.geometry import PREFILL_ROWS, Geometry, mla_geometry

        from . import kv8
        from .forward import qscratch_slots
        from .select import SELECT_BYTES

        prompt_rows = PROMPT_ROWS if rows is None else int(rows)
        t = dict(text)
        t.pop("linear_attn_config", None)              # no KDA layers: drop its state terms
        t["layer_types"] = ["deepseek_sparse_attention"] * int(t["num_hidden_layers"])
        t["hc_mult"] = 1                               # plain residuals: one stream
        t["swiglu_limit"] = 0.0
        t.setdefault("index_kpool", 1)                 # raw-token selection: one token per pool in the estimate
        flash = mla_geometry(t, world, MAX_ROWS, minimum_slots=DENSE_CAPACITY, latent=True)
        mtp = int(text.get("num_nextn_predict_layers") or 0) > 0
        layers = int(text["num_hidden_layers"]) + int(mtp)
        full = sum(kind == "full" for kind in text.get("indexer_types") or []) + int(mtp)
        own = kv8.slot_bytes(layers, int(text["kv_lora_rank"]), int(text["qk_rope_head_dim"]), full,
                             int(text.get("index_head_dim", 128)), kv or kv8.MODE)
        lo, hi = 1 << 20, 1 << 21                      # Flash's bytes a slot past its dense window
        flash_slot = (flash.bytes_at(hi) - flash.bytes_at(lo)) / (hi - lo)
        extra = 0
        if dcp > 1:
            own = own / dcp
            hl = int(text["num_attention_heads"]) // world
            lw, pw = int(text["kv_lora_rank"]), int(text["qk_rope_head_dim"])
            r = prompt_rows                            # the prompt buffers' qpack, qall, send and recv; ~20 MiB decode
            extra = r * hl * ((lw + pw) * 2 * (1 + dcp) + (lw // 2 + 1) * 4 * 2 * dcp) + (24 << 20)
            extra += 2 * r * dcp * 2048 * 8            # a prompt block's gathered selection candidates
        extra += max(0, prompt_rows - PREFILL_ROWS) * PROMPT_ROW_BYTES      # chunks wider than geometry's 2,048 rows
        qs_row = 2 * int(text["kv_lora_rank"])
        if streams > 1:
            from .multi import EXTENTS, FILL_ROW_BYTES, FILL_ROWS

            own = own if EXTENTS else own * streams    # extents: one window's pool for every stream
            extra += 1 << 30                           # the rounds' buffers (every stream's rows), graphs' pool
            extra += FILL_ROWS * FILL_ROW_BYTES        # short prompts' batched fills

        def bytes_at(slots: int) -> int:
            return (int(flash.bytes_at(slots) + (own - flash_slot) * slots) + 3 * SELECT_BYTES + extra
                    + qscratch_slots(slots, dcp, kv) * qs_row)

        return Geometry(bytes_at, flash.reserve, flash.minimum_slots)

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int32, device="cuda")
        got = torch.empty((self.world * len(values),), dtype=torch.int32, device="cuda")
        self.comm.all_gather(mine, got)
        return [got[i * len(values):(i + 1) * len(values)].tolist() for i in range(self.world)]

    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank (a length, then the values, through the all-gather)."""

        torch = self.torch
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32, device="cuda")
        got = torch.empty((self.world,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(n, got)
        count = int(got[0].item())
        buf = (torch.tensor(values, dtype=torch.int32, device="cuda") if self.rank == 0
               else torch.zeros((count,), dtype=torch.int32, device="cuda"))
        allv = torch.empty((self.world * count,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]

    # -- the server-facing surface --------------------------------------------------------------------------------
    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 constraint=None, **_: Any) -> dict[str, Any]:
        """One request end to end; identical keyed sampling on every rank, verified drafts propose only."""

        from tensorfold.engine.grammar import pack

        if self.scheduler is not None:
            if constraint is not None:
                raise ValueError("structured output is not served with --parallel on GLM-5.3 yet: send text without "
                                 "response_format, or start without --parallel")
            stop_eos = bool(getattr(self.request, "stop_eos", True))
            return self.scheduler.submit(list(prompt), max_tokens, sampling, draft and self.policy != "0", on_tokens,
                                         stop_eos=stop_eos)
        spec = getattr(self.request, "policy", None) or self.policy
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        code = [1 if spec not in ("", "0") else 0, min(int(spec) if str(spec).isdigit() else 3, MAX_ROWS - 1), 0, 0]
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        header = [max_tokens, int(stop_eos), int(draft), 0, seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF,
                  seed >> 62, *_f64_ints(sampling.temperature if sampling else 0.0),
                  int(sampling.top_k) if sampling else 0, *_f64_ints(sampling.top_p if sampling else 1.0),
                  *_f64_ints(sampling.min_p if sampling else 0.0), int(constraint is not None)] + code
        self._share(header)
        self._share(list(prompt))
        if constraint is not None:                     # the request's grammar: every rank compiles the same
            self._share(pack(constraint))
        return self._run(list(prompt), max_tokens, sampling, stop_eos, on_tokens, code, draft)

    def follow(self) -> None:
        """Follower rank: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling

        if self.multi is not None:
            from .multi import Link

            self.multi.follow(Link(_store(self.comm), rank=self.rank, world=self.world, host=self.master))
            return
        while True:
            (max_tokens, stop_eos, draft, _cached, s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi,
             shaped, kind, most, _a, _b) = self._share(None)
            prompt = self._share(None)
            constraint = None
            if shaped:
                packed = self._share(None)
                if packed:
                    from tensorfold.engine import grammar

                    constraint = grammar.compiler(self, self.model_dir, self.eos).follow(packed)
            temperature = _ints_f64(t_lo, t_hi)
            seed = (s_top << 62) | (s_hi << 31) | s_lo
            sampling = (Sampling(seed, temperature, top_k, _ints_f64(p_lo, p_hi), _ints_f64(m_lo, m_hi))
                        if temperature > 0 else None)
            self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None,
                      [kind, most, 0, 0], bool(draft), constraint)

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool,
             on_tokens: Callable[[list[int]], Any], code: list[int], draft: bool, constraint=None) -> dict[str, Any]:
        self.e.constraint, self.e.window = constraint, None       # every rank walks and masks the same rows
        try:
            return self._run_once(prompt, max_tokens, sampling, stop_eos, on_tokens, code, draft)
        finally:
            self.e.constraint = self.e.window = None

    def _run_once(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool,
                  on_tokens: Callable[[list[int]], Any], code: list[int], draft: bool) -> dict[str, Any]:
        from .decode import DEPTH_COST, AcceptPolicy, DepthPolicy, mtp_decode, prefill, serial_decode

        t0 = time.perf_counter()
        first = prefill(self.e, prompt, sampling)
        stats: dict[str, Any] = {"prefill_s": time.perf_counter() - t0, "cached": getattr(self.e, "cached", 0)}
        on_tokens([first])
        if max_tokens <= 1 or (stop_eos and first in self.eos):
            return stats
        policy = None
        if code[0]:
            policy = AcceptPolicy(code[1]) if DEPTH_COST > 0 else DepthPolicy(code[1], fixed=True)
        res = (serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens)
               if policy is None or self.w.mtp is None or not draft
               else mtp_decode(self.e, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                               on_tokens=on_tokens))
        stats.update(decode_s=res.seconds, rounds=res.rounds, tokens_per_second=round(res.tokens_per_second, 3),
                     sha256=__import__("hashlib").sha256(json.dumps(res.tokens).encode()).hexdigest()[:16])
        if self.w.comm is None or self.w.rank == 0:
            print(f"[tensorfold] decode {len(res.tokens)} tok {res.tokens_per_second:.2f} tok/s rounds {res.rounds}"
                  f" drafted {res.drafted} accepted {res.accepted}", flush=True)
        return stats


def _store(comm):
    """The rendezvous store under the communicator (the RDMA hybrid wraps NCCL's)."""

    for c in (comm, getattr(comm, "comm", None), getattr(comm, "nccl", None)):
        store = getattr(c, "store", None)
        if store is not None:
            return store
    raise RuntimeError("--parallel on several ranks needs the communicator's rendezvous store")


def _rdma(comm, rank: int, world: int, max_bytes: int):
    """NCCL plus the RDMA gather (tensorfold.cuda.rdma) for fp32 all-gathers up to ``max_bytes`` (decode windows'
    partials, sampling) when every rank opens it, else NCCL alone; TF_GLM_RDMA=0 keeps NCCL. Same bytes either way: a
    decode window's 156 all-gathers take 23 instead of 35 us each at one row, 36 instead of 56 at four (rdma_bench.py)."""

    if world < 2 or os.environ.get("TF_GLM_RDMA", "1") == "0":
        return comm
    from tensorfold.cuda.rdma import Hybrid, RdmaGather

    try:
        rdma = RdmaGather(comm.store, rank, world, max_bytes=max_bytes)
    except Exception as exc:                  # noqa: BLE001  (every rank sees the same refusal)
        print(f"[tensorfold] decode all-gathers stay on NCCL: {exc}", flush=True)
        return comm
    if rank == 0:
        print(f"[tensorfold] decode all-gathers: RDMA writes over {', '.join(rdma.devices)}", flush=True)
    return Hybrid(comm, rdma)


def _f64_ints(x: float) -> list[int]:
    import struct

    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _ints_f64(lo: int, hi: int) -> float:
    import struct

    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]
