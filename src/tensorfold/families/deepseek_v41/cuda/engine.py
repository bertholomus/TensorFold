"""DeepSeek-V4.1 on TF_TP_WORLD ranks behind TensorFold's CUDA server: rank 0 serves, the others mirror each request.

One request at a time. Sampling is keyed by (seed, position, token) on the gathered logits, which every rank holds
bit for bit, so the ranks agree without a broadcast. DSpark drafts are verified against the target's own keyed
samples: drafted output equals serial output, at temperature 0 and above.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch

PREFILL_CHUNK = int(os.environ.get("TF_DS_PREFILL_CHUNK") or 512)
GRAPHS = os.environ.get("TF_DS_GRAPHS", "1") != "0"
# decoder SWA bounded replay (CED's prefill, DeepSeek's deployment mode): the decoder runs only over the last 128
# prompt tokens; off by default (exact prefill) until its agreement and needle recall are measured
REPLAY = os.environ.get("TF_DS_REPLAY", "0") == "1"
CHUNK_LOG = os.environ.get("TF_DS_CHUNK_LOG", "0") == "1"
# drafts verified a round chosen from the confidence head's prefix survival and measured window costs (0: fixed k)
ADAPTIVE = os.environ.get("TF_DS_ADAPTIVE", "0") == "1"
# before serving: caches at full size, the Triton kernels prompt chunks use, every decode graph (each window size at each
# context bucket) and the drafter's graph, so no request pays a kernel build or a graph capture
WARM = os.environ.get("TF_DS_WARM", "1") != "0"
# the tower's warm-up image: "small" (default) or "largest" (logs the largest image's time and memory peak)
VISION_WARM = os.environ.get("TF_DS_VISION_WARM") or "small"
# synthetic prompt lengths for the warm-up: row and key counts of 1, multiples of 16 and others (Triton specializes
# integer arguments on those), short prompts on the window ring, every pick-block bucket of the prompt attention, prompts
# past the decoder replay window at even and odd offsets (pointer alignment), a second chunk
WARM_LENGTHS = tuple(int(v) for v in (os.environ.get("TF_DS_WARM_LENGTHS") or
                                      "1,2,3,16,17,32,33,34,48,65,66,96,130,131,160,256,258,259,512,514,1024,2113").split(","))


class DsEngine:
    def __init__(self, model_dir: Path, *, rank: int, world: int, master: str, port: int, drafts: int = 3,
                 context: int | None = None, engram_dir: str | None = None, vision: bool = False,
                 vision_urls: bool = False, parallel: int = 1) -> None:
        from tensorfold.cuda.comm import NCCL

        from ..ops import compressed_token_map
        from .dspark import Drafter
        from .model import Comm, Engram, Model
        from .weights import load

        torch.cuda.set_device(0)
        self.rank, self.world, self._master = rank, world, master
        # NCCL only moves prompt chunks' partials ([2048, 5120] fp32): the Simple protocol on 4 channels took 3.8 ms a
        # gather on the CX7 link against 7.1 with NCCL's choice (decode windows go over the RDMA gather)
        os.environ.setdefault("NCCL_PROTO", "Simple")
        os.environ.setdefault("NCCL_MIN_NCHANNELS", "4")
        nccl = NCCL(rank, world, master, port) if world > 1 else None
        self.nccl = nccl
        self.w = load(model_dir, rank, world, dspark=drafts > 0)
        cfg = self.w.cfg
        eng = None
        engram_dir = engram_dir or _default_engram(model_dir)
        if engram_dir:
            cache = Path(os.environ.get("TF_DS_TOKEN_MAP") or (Path.home() / ".cache" / "dsv41_token_map.json"))
            if cache.exists():
                tm = json.loads(cache.read_text())
            else:
                tm, n = compressed_token_map(Path(model_dir) / "tokenizer.json")
                assert n == cfg.engram_cvocab, (n, cfg.engram_cvocab)
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(tm))
                except OSError:
                    pass
            eng = Engram(engram_dir, cfg, tm, rank, world)
        elif rank == 0:
            print("[tensorfold] WARNING: no Engram tables (TF_DS_ENGRAM): output will be degraded", flush=True)
        self.model = Model(self.w, Comm(nccl, world, rdma_bytes=8 << 20), eng)
        # images (--vision): the tower runs on rank 0, which shares each image span's rows; every rank routes the span
        # with the gates' VL bias and keeps Engram out of it
        self.vision = None
        self.tower = None
        self.vcfg = None
        if vision:
            from .vision import DsVision, Tower, VisionConfig
            from .weights import attach_vl_bias

            self.vcfg = VisionConfig.read(json.loads((Path(model_dir) / "config.json").read_text()))
            if self.vcfg is None:
                raise ValueError("--vision: this checkpoint has no vision tower (vision_config)")
            found = attach_vl_bias(self.w, [os.environ.get("TF_DS_VISION_EXTRA"), str(model_dir), engram_dir,
                                            str(Path(model_dir).resolve().parent / "DeepSeek-V4.1-Flash-extra")])
            if rank == 0:
                if not found:
                    print("[tensorfold] WARNING: no ffn.gate.bias_vl found (TF_DS_VISION_EXTRA): image tokens route "
                          "with the text bias", flush=True)
                before = torch.cuda.memory_allocated()
                self.tower = Tower(self.vcfg, model_dir)
                self.vision = DsVision(self.vcfg, allow_urls=bool(vision_urls))
                print(f"[tensorfold] vision tower on rank 0: {(torch.cuda.memory_allocated() - before) / 2**30:.2f} "
                      f"GiB; VL bias for {found} gates", flush=True)
        self.drafter = Drafter(self.model) if drafts > 0 and self.w.dspark is not None else None
        from .graph import GraphRunner

        self.runner = GraphRunner(self.model, drafts + 1) if GRAPHS else None
        self.drafts = drafts
        self.limit = int(context or 65536)
        self.max_rows = drafts + 1
        self.model.rope_cap = self.limit + self.max_rows + 8      # every cache reads the same RoPE table
        self.eos = (int(json.loads((Path(model_dir) / "config.json").read_text()).get("eos_token_id", 1)),)
        self.request = threading.local()
        self.quiet = False
        # --parallel N: up to N requests decoded together (multi.MultiDecoder), each in a slot of one cache pool
        self.concurrent = int(parallel) > 1
        self.multi = self.scheduler = None
        if self.concurrent:
            import sys

            from .multi import MultiDecoder

            # the scheduler thread shares the GIL with every reply's HTTP thread: at Python's default 5 ms switch
            # interval it waited ~10 ms a round for the GIL between rounds (4 streams); 0.5 ms hands it back sooner
            sys.setswitchinterval(float(os.environ.get("TF_DS_SWITCH_INTERVAL") or 0.0005))

            self.multi = MultiDecoder(self, slots=int(parallel), cap=self.limit + self.max_rows + 8)
            mu = self.multi                              # the warm-up's prompts run on slot 0 over the whole window
            self.sc = self.model.pool_view(mu.pool, 0, 0, mu.extents.total)
            self.dc = mu.slots[0].dc
        if nccl is not None:
            nccl.barrier()
            mine = torch.tensor([int(self.vcfg is not None), int(parallel)], dtype=torch.int64, device="cuda")
            every = torch.empty((world * 2,), dtype=torch.int64, device="cuda")
            nccl.all_gather(mine, every)
            flags = every.view(world, 2).tolist()
            if any(f != flags[0] for f in flags):
                raise ValueError("--vision and --parallel must be the same on every rank (rank 0 and the workers run "
                                 f"the same steps): {flags}")
        if WARM:
            self.warm()
        if self.concurrent:
            if WARM:
                self.multi.warm()
            if world > 1:
                from .multi import Link, Watchdog

                self.multi.watch = Watchdog(self.model.comm.nccl, nccl.store, rank=rank, world=world, host=master)
                if rank == 0:
                    self.multi.link = Link(nccl.store, rank=0, world=world, host=master)
            if rank == 0:
                from tensorfold.cuda.scheduler import Scheduler

                self.scheduler = Scheduler(self.multi, max_streams=int(parallel))
                print(f"[tensorfold] --parallel {int(parallel)}: {int(parallel)} streams share one window of "
                      f"{self.multi.extents.total} tokens (an extent each)", flush=True)
        self._memory_ceiling()
        if rank == 0:
            print(f"[tensorfold] DeepSeek-V4.1 engine ready: {world} rank(s), context {self.limit}, "
                  f"{'DSpark ' + str(drafts) + ' drafts' if self.drafter else 'serial decode'}", flush=True)

    def _memory_ceiling(self) -> None:
        """After the warm-up, torch may grow only while the host keeps TF_DS_MEM_FLOOR_GIB (default 2; 0: no ceiling)
        available: a step that would need more fails with CUDA's out-of-memory error (that request fails, the lane's
        watchdog restarts it if a rank broke) instead of the kernel's OOM killer, which on a unified-memory GB10 took
        the user's systemd manager, and its restart started every enabled user unit."""

        floor = float(os.environ.get("TF_DS_MEM_FLOOR_GIB") or 2.0)
        if floor <= 0:
            return
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            available = int(info["MemAvailable"].split()[0]) * 1024
        except (OSError, KeyError, ValueError):
            return
        torch.cuda.synchronize()
        total = torch.cuda.mem_get_info()[1]
        reserved = torch.cuda.memory_reserved()
        ceiling = reserved + max(0, available - int(floor * 2**30))
        torch.cuda.set_per_process_memory_fraction(min(1.0, ceiling / total))
        print(f"[tensorfold] rank {self.rank}: allocator ceiling {ceiling / 2**30:.1f} GiB (holds "
              f"{reserved / 2**30:.1f}, host available {available / 2**30:.1f}, floor {floor:.1f})", flush=True)

    def warm(self) -> None:
        """Same steps on every rank, in the same order (prefills and graph captures issue real collectives)."""

        t0 = time.perf_counter()
        m = self.model
        if not self.concurrent:
            self.sc = m.new_cache(self.limit + self.max_rows + 8)
            if self.drafter is not None:
                self.dc = self.drafter.new_cache()
        ids = [1000 + (i * 7919) % 60000 for i in range(max(WARM_LENGTHS))]
        self.quiet = True
        try:
            for n in WARM_LENGTHS:
                if n + 8 <= self.limit:
                    if self.concurrent:     # --parallel decodes through round graphs (multi.warm): prompt kernels only
                        self._sample(self.prefill(self.sc, self.dc, ids[:n]), [n], None)
                    else:
                        self._run(ids[:n], 6, None, False, lambda new: None, self.drafter is not None)
            if not self.concurrent:
                self._run(ids[:17], 3, None, False, lambda new: None, False)
            if self.vcfg is not None:
                self.vision_warm(ids)
        finally:
            self.quiet = False
        t1 = time.perf_counter()
        info = {}
        if self.runner is not None and not self.concurrent:
            info = self.runner.warm(self.sc, self.limit)
            if self.drafter is not None:
                self._draft_graph(self.sc, self.dc, 0, 0)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if self.rank == 0:
            print(f"[tensorfold] warm-up: {len(WARM_LENGTHS)} prompt lengths {t1 - t0:.1f}s, decode graphs {info}, "
                  f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)

    def vision_warm(self, ids: list[int]) -> None:
        """The image paths' kernels on every rank (synthetic span rows, nothing shared), and the tower once at its
        largest grid on rank 0 (its time and memory peak logged)."""

        k = 70
        prompt = list(ids[:k + 20])
        prompt[5:5 + k] = [self.vcfg.image_token_id] * k
        rows = torch.zeros((k, self.vcfg.model_dim), dtype=torch.bfloat16, device="cuda")
        if self.concurrent:
            self.prefill(self.sc, self.dc, prompt, image=(list(range(5, 5 + k)), rows))
        else:
            self._run(prompt, 4, None, False, lambda new: None, self.drafter is not None,
                      image=(list(range(5, 5 + k)), rows))
        if self.tower is not None:
            from .vision import Picture, plan_grid

            side = 4096 if VISION_WARM == "largest" else 448        # the tower has no shape-tuned kernels to build
            n_h, n_w, bh, bw = plan_grid(side, side, self.vcfg)
            p = self.vcfg.patch
            pic = Picture(torch.zeros((bh // p * (bw // p), 3, p, p), dtype=torch.bfloat16), bh // p, bw // p, n_h, n_w)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            self.tower.span_rows(pic)
            torch.cuda.synchronize()
            print(f"[tensorfold] vision warm-up: {VISION_WARM} image ({pic.tokens} tokens, {bh}x{bw} px) "
                  f"{1000 * (time.perf_counter() - t0):.0f} ms, peak +{(torch.cuda.max_memory_allocated() - base) / 2**30:.2f} "
                  f"GiB", flush=True)

    def _share_rows(self, rows: torch.Tensor | None, n: int) -> torch.Tensor:
        """Rank 0's image-span rows [n, dim] bf16 on every rank."""

        dim = self.w.cfg.dim
        if self.world == 1:
            return rows
        send = rows.contiguous() if self.rank == 0 else torch.zeros((n, dim), dtype=torch.bfloat16, device="cuda")
        recv = torch.empty((self.world * n * dim,), dtype=torch.bfloat16, device="cuda")
        self.nccl.all_gather(send.view(-1), recv)
        return recv[:n * dim].view(n, dim)

    def _make_draft_graph(self, sc, dc, tok: int, pos: int):
        """A drafter graph bound to one cache pair (a concurrent slot's), captured in the shared graph pool."""

        from .dspark import DraftGraph

        dg = DraftGraph(self.drafter, sc, dc)
        dg.token.fill_(tok)
        dg.q0.fill_(pos)
        if self.runner is None:
            return dg
        if self.runner.pool is None:
            self.runner.pool = torch.cuda.graph_pool_handle()
        dg.capture(self.runner.pool)
        return dg

    def _draft_graph(self, sc, dc, tok: int, pos: int):
        dg = getattr(self, "_dg", None)
        if dg is None or dg.sc is not sc or dg.dc is not dc:
            from .dspark import DraftGraph

            if getattr(self.runner, "warmed", False):
                print("[tensorfold] WARNING: drafter graph captured while serving", flush=True)
            dg = self._dg = DraftGraph(self.drafter, sc, dc)
            dg.token.fill_(tok)
            dg.q0.fill_(pos)
            if self.runner.pool is None:
                self.runner.pool = torch.cuda.graph_pool_handle()
            dg.capture(self.runner.pool)
        return dg

    # -- request mirroring -----------------------------------------------------------------------------------------
    def _share(self, values: list[int] | None) -> list[int]:
        if self.world == 1:
            return list(values or [])
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int64, device="cuda")
        got = torch.empty((self.world,), dtype=torch.int64, device="cuda")
        self.nccl.all_gather(n, got)
        count = int(got[0])
        buf = (torch.tensor(values, dtype=torch.int64, device="cuda") if self.rank == 0
               else torch.zeros((count,), dtype=torch.int64, device="cuda"))
        allv = torch.empty((self.world * count,), dtype=torch.int64, device="cuda")
        if count:
            self.nccl.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 constraint=None, vision=None, **_: Any) -> dict[str, Any]:
        if constraint is not None:
            raise ValueError("structured output is not served by the DeepSeek-V4.1 engine yet")
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        if self.scheduler is not None:
            if vision is not None and getattr(vision, "spans", None) and self.tower is None:
                raise ValueError("image input needs the server started with --vision")
            return self.scheduler.submit(list(prompt), max_tokens, sampling, bool(draft), on_tokens,
                                         stop_eos=stop_eos, vision=vision)
        positions, rows = [], None
        if vision is not None and getattr(vision, "spans", None):
            if self.tower is None:
                raise ValueError("image input needs the server started with --vision")
            positions = vision.positions()
            if positions and positions[-1] >= len(prompt):
                raise ValueError("an image span lies past the prompt")
            rows = torch.cat([self.tower.span_rows(pic) for _, pic in vision.spans])
        seed = (sampling.seed if sampling else 0) & ((1 << 63) - 1)
        header = [max_tokens, int(stop_eos), int(draft), seed, *_f64(sampling.temperature if sampling else 0.0),
                  int(sampling.top_k) if sampling else 0, *_f64(sampling.top_p if sampling else 1.0),
                  *_f64(sampling.min_p if sampling else 0.0), len(positions)]
        self._share(header)
        self._share(list(prompt))
        image = None
        if positions:
            self._share(positions)
            image = (positions, self._share_rows(rows, len(positions)))
        return self._run(list(prompt), max_tokens, sampling, stop_eos, on_tokens, draft, image=image)

    def follow(self) -> None:
        from tensorfold.engine.exact_sampling import Sampling

        if self.multi is not None:
            from .multi import Link

            self.multi.follow(Link(self.nccl.store, rank=self.rank, world=self.world, host=self._master))
            return
        while True:
            (max_tokens, stop_eos, draft, seed, t0, t1, top_k, p0, p1, m0, m1, n_img) = self._share(None)
            prompt = self._share(None)
            image = None
            if n_img:
                positions = self._share(None)
                image = (positions, self._share_rows(None, n_img))
            temperature = _f64_back(t0, t1)
            sampling = (Sampling(seed, temperature, top_k, _f64_back(p0, p1), _f64_back(m0, m1))
                        if temperature > 0 else None)
            self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None, bool(draft), image=image)

    # -- one request -----------------------------------------------------------------------------------------------
    def _sample(self, logits: torch.Tensor, positions: list[int], sampling) -> list[int]:
        from tensorfold.cuda.sampling import sample_rows

        return sample_rows(logits, positions, sampling)

    def prefill(self, sc, dc, prompt: list[int], image: tuple | None = None) -> torch.Tensor:
        """The prompt into ``sc`` (and the drafter's cache ``dc``, when given) in PREFILL_CHUNK chunks: the last row's
        logits. ``image``: (ascending positions of the prompt's image-span tokens, their rows [k, dim] bf16)."""

        steps = self.prefill_steps(sc, dc, prompt, image)
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value

    def prefill_steps(self, sc, dc, prompt: list[int], image: tuple | None = None, start: int = 0, snap=None):
        """``prefill`` one chunk a step (a generator: each ``next`` runs one chunk; its return value is the last
        row's logits), so a concurrent lane can decode between a long prompt's chunks. ``start`` (a chunk boundary):
        the caches already hold the prompt's rows before it, a kept prompt's (``multi``), and its chunks are a fresh
        prefill's own from there; ``snap(end)`` runs after each chunk that ends on a chunk boundary."""

        m = self.model
        assert start % PREFILL_CHUNK == 0 and start < len(prompt), (start, len(prompt))
        sc.length = start
        del sc.host[start:]
        use_drafts = dc is not None
        if use_drafts:
            dc.absorbed = 0
        last = None
        replay = max(0, len(prompt) - self.w.cfg.window) if getattr(self, "replay_mode", REPLAY) else None
        host = prompt
        if image is not None:
            import bisect

            positions, rows = image
            pos_dev = torch.tensor(positions, dtype=torch.long, device="cuda")
            host = list(prompt)
            for p in positions:
                host[p] = -1                  # Engram's hashing: no n-gram reaches into an image span
        def ahead(a: int) -> None:                     # the chunk at ``a``: its Engram rows start reading now
            if m.engram is None or m.engram.bg is None or a >= len(prompt):
                return
            hs = m.engram.hashes(host, a, min(len(prompt), a + PREFILL_CHUNK) - a)
            lo, hi = m.engram.cols
            for i in self.w.cfg.engram_layers:
                if i < len(self.w.layers):
                    m.engram.prefetch(i, hs[:, self.w.cfg.engram_layers.index(i), lo:hi], lane=1)

        for s in range(start, len(prompt), PREFILL_CHUNK):
            e = min(len(prompt), s + PREFILL_CHUNK)
            if s == start:
                ahead(s)
            ahead(e)                                   # the next chunk's rows read while this one runs
            ids = torch.tensor(prompt[s:e], dtype=torch.long, device="cuda")
            taps: list | None = [] if use_drafts else None
            block = None
            if image is not None:
                i0, i1 = bisect.bisect_left(positions, s), bisect.bisect_left(positions, e)
                if i1 > i0:
                    block = (pos_dev[i0:i1] - s, rows[i0:i1])
            tc = time.perf_counter()
            out = m.forward(sc, ids, s, taps=taps, host_ids=host[s:e], replay=replay, image=block)
            if CHUNK_LOG:
                torch.cuda.synchronize()
                print(f"[tensorfold] rank {self.rank} chunk at {s}: {time.perf_counter() - tc:.2f}s, allocated "
                      f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB, peak {torch.cuda.max_memory_allocated() / 2**30:.2f}, "
                      f"reserved {torch.cuda.memory_reserved() / 2**30:.2f}", flush=True)
                from .model import TIMES
                if TIMES and self.rank == 0:            # TF_DS_TIMING=1: the chunk's sections (ms), then reset
                    print("[tensorfold] chunk sections ms " + " ".join(f"{k} {1000 * v:.1f}" for k, v in
                                                                      sorted(TIMES.items(), key=lambda kv: -kv[1])),
                          flush=True)
                    TIMES.clear()
            if out is not None:
                last = out
            if use_drafts and taps:
                self.drafter.absorb(dc, sc, torch.cat(taps, -1), m.taps_start)
            if snap is not None and e % PREFILL_CHUNK == 0:
                snap(e)
            if e < len(prompt):
                yield e
        return last

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool, on_tokens: Callable,
             draft: bool, image: tuple | None = None) -> dict[str, Any]:
        """``image``: (ascending positions of the prompt's image-span tokens, their rows [k, dim] bf16)."""
        m = self.model
        eos = self.eos if stop_eos else ()
        if len(prompt) + max_tokens > self.limit:
            max_tokens = max(1, self.limit - len(prompt))
        need = len(prompt) + max_tokens + self.max_rows + 8
        if getattr(self, "sc", None) is None or self.sc.cap < need:
            self.sc = m.new_cache(max(need, self.limit + self.max_rows + 8))
        sc = self.sc
        use_drafts = bool(draft) and self.drafter is not None
        dc = None
        if use_drafts:
            if getattr(self, "dc", None) is None:
                self.dc = self.drafter.new_cache()
            dc = self.dc
        t0 = time.perf_counter()
        last = self.prefill(sc, dc, prompt, image)
        self.last_prefill_logits = last
        torch.cuda.empty_cache()          # a long prompt's chunk buffers back to the node (unified memory)
        first = self._sample(last, [len(prompt)], sampling)[0]
        stats: dict[str, Any] = {"prefill_s": time.perf_counter() - t0}
        t1 = time.perf_counter()
        if use_drafts:
            out, st = self._spec(sc, dc, first, max_tokens, sampling, eos, on_tokens)
            stats.update(rounds=st[0], drafted=st[1], accepted=st[2], **self.last_spec_times)
        else:
            out = self._serial(sc, first, max_tokens, sampling, eos, on_tokens)
        dt = time.perf_counter() - t1
        stats.update(decode_s=dt, tokens=len(out), tokens_per_second=(len(out) - 1) / dt if dt > 0 else 0.0,
                     sha256=hashlib.sha256(json.dumps(out).encode()).hexdigest()[:16])
        if self.rank == 0 and not self.quiet:
            print(f"[tensorfold] prompt {len(prompt)} prefill {stats['prefill_s']:.2f}s decode {len(out)} tok "
                  f"{stats['tokens_per_second']:.2f} tok/s"
                  + (f" rounds {stats['rounds']} drafted {stats['drafted']} accepted {stats['accepted']}"
                     if use_drafts else ""), flush=True)
        return stats

    def _serial(self, sc, first, max_tokens, sampling, eos, on_tokens) -> list[int]:
        out = [first]
        on_tokens([first])
        tok = first
        while len(out) < max_tokens and tok not in eos:
            p = sc.length
            if self.runner is not None:
                lg, _ = self.runner.forward(sc, [tok], p, False)
            else:
                lg = self.model.forward(sc, torch.tensor([tok], dtype=torch.long, device="cuda"), p, host_ids=[tok])
            tok = self._sample(lg, [p + 1], sampling)[0]
            out.append(tok)
            on_tokens([tok])
        return out

    def _choose_k(self, conf, kmax: int, draft_ms: float) -> int:
        """The k in 1..kmax with the most expected tokens a millisecond: (1 + sum of prefix survivals) over the draft
        plus the measured cost of a k-draft window (unmeasured sizes: the nearest measured one, + 7 ms a row)."""

        surv = _survival(conf[:kmax])
        cost = getattr(self, "vcost", {})
        best, best_v = kmax, -1.0
        for k in range(1, kmax + 1):
            if k in cost:
                c = cost[k]
            elif cost:
                near = min(cost, key=lambda q: abs(q - k))
                c = cost[near] + 7.0 * (k - near)
            else:
                c = 33.0 + 7.0 * k
            v = (1.0 + sum(surv[:k])) / (draft_ms + c)
            if v > best_v:
                best, best_v = k, v
        return best

    def _spec(self, sc, dc, first, max_tokens, sampling, eos, on_tokens):
        """Drafts verified k at a time against the target's keyed samples (the serial rule)."""

        m, d = self.model, self.drafter
        out = [first]
        on_tokens([first])
        tok = first
        rounds = drafted = accepted = 0
        t_draft = t_verify = t_absorb = 0.0
        while len(out) < max_tokens and tok not in eos:
            P = sc.length
            ta = time.perf_counter()
            if self.runner is not None:
                drafts, _conf = self._draft_graph(sc, dc, tok, P).run(tok, P)
            else:
                drafts, _conf = d.draft(dc, sc, tok, P)
            tb = time.perf_counter()
            t_draft += tb - ta
            kk = min(self.drafts, max_tokens - len(out), len(drafts))
            if ADAPTIVE and kk > 1:
                kk = self._choose_k(_conf, kk, (tb - ta) * 1000)
            window = [tok] + drafts[:kk]
            if self.runner is not None:
                lg, tapt = self.runner.forward(sc, window, P, True)
                taps = [tapt]
            else:
                taps = []
                lg = m.forward(sc, torch.tensor(window, dtype=torch.long, device="cuda"), P, all_logits=True,
                               taps=taps, host_ids=window)
            target = self._sample(lg, [P + 1 + i for i in range(kk + 1)], sampling)
            tc = time.perf_counter()
            t_verify += tc - tb
            if ADAPTIVE:
                cost = self.__dict__.setdefault("vcost", {})
                cost[kk] = (tc - tb) * 1000 if kk not in cost else 0.8 * cost[kk] + 0.2 * (tc - tb) * 1000
            a = 0
            while a < kk and drafts[a] == target[a]:
                a += 1
            new = drafts[:a] + [target[a]]
            rounds += 1
            drafted += kk
            accepted += a
            sc.length = P + a + 1
            d.absorb(dc, sc, torch.cat(taps, -1)[:a + 1], P)
            t_absorb += time.perf_counter() - tc
            for i, t in enumerate(new):
                if t in eos:
                    new = new[:i + 1]
                    break
            new = new[:max_tokens - len(out)]
            out += new
            on_tokens(new)
            tok = out[-1]
        self.last_spec_times = {"draft_ms": 1000 * t_draft / max(rounds, 1), "verify_ms": 1000 * t_verify / max(rounds, 1),
                                "absorb_ms": 1000 * t_absorb / max(rounds, 1)}
        return out, (rounds, drafted, accepted)


def _survival(conf) -> list[float]:
    import math

    out, s = [], 1.0
    for c in conf:
        s *= 1.0 / (1.0 + math.exp(-float(c)))
        out.append(s)
    return out


def _default_engram(model_dir: Path) -> str | None:
    """A sibling folder holding the original Engram shards (``*Engram*``), else None."""

    parent = Path(model_dir).resolve().parent
    for cand in sorted(parent.glob("*Engram*")):
        if any(cand.glob("*.safetensors")):
            return str(cand)
    return None


def _f64(x: float) -> list[int]:
    lo, hi = struct.unpack("<2i", struct.pack("<d", float(x)))
    return [lo, hi]


def _f64_back(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<2i", int(lo), int(hi)))[0]
