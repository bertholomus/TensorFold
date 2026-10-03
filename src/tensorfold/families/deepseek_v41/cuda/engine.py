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


class DsEngine:
    def __init__(self, model_dir: Path, *, rank: int, world: int, master: str, port: int, drafts: int = 3,
                 context: int | None = None, engram_dir: str | None = None) -> None:
        from tensorfold.cuda.comm import NCCL

        from ..ops import compressed_token_map
        from .dspark import Drafter
        from .model import Comm, Engram, Model
        from .weights import load

        torch.cuda.set_device(0)
        self.rank, self.world = rank, world
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
        self.drafter = Drafter(self.model) if drafts > 0 and self.w.dspark is not None else None
        self.drafts = drafts
        self.limit = int(context or 65536)
        self.max_rows = drafts + 1
        self.eos = (int(json.loads((Path(model_dir) / "config.json").read_text()).get("eos_token_id", 1)),)
        self.request = threading.local()
        if nccl is not None:
            nccl.barrier()
        if rank == 0:
            print(f"[tensorfold] DeepSeek-V4.1 engine ready: {world} rank(s), context {self.limit}, "
                  f"{'DSpark ' + str(drafts) + ' drafts' if self.drafter else 'serial decode'}", flush=True)

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
                 constraint=None, **_: Any) -> dict[str, Any]:
        if constraint is not None:
            raise ValueError("structured output is not served by the DeepSeek-V4.1 engine yet")
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        seed = (sampling.seed if sampling else 0) & ((1 << 63) - 1)
        header = [max_tokens, int(stop_eos), int(draft), seed, *_f64(sampling.temperature if sampling else 0.0),
                  int(sampling.top_k) if sampling else 0, *_f64(sampling.top_p if sampling else 1.0),
                  *_f64(sampling.min_p if sampling else 0.0)]
        self._share(header)
        self._share(list(prompt))
        return self._run(list(prompt), max_tokens, sampling, stop_eos, on_tokens, draft)

    def follow(self) -> None:
        from tensorfold.engine.exact_sampling import Sampling

        while True:
            (max_tokens, stop_eos, draft, seed, t0, t1, top_k, p0, p1, m0, m1) = self._share(None)
            prompt = self._share(None)
            temperature = _f64_back(t0, t1)
            sampling = (Sampling(seed, temperature, top_k, _f64_back(p0, p1), _f64_back(m0, m1))
                        if temperature > 0 else None)
            self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None, bool(draft))

    # -- one request -----------------------------------------------------------------------------------------------
    def _sample(self, logits: torch.Tensor, positions: list[int], sampling) -> list[int]:
        from tensorfold.cuda.sampling import sample_rows

        return sample_rows(logits, positions, sampling)

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool, on_tokens: Callable,
             draft: bool) -> dict[str, Any]:
        m = self.model
        eos = self.eos if stop_eos else ()
        if len(prompt) + max_tokens > self.limit:
            max_tokens = max(1, self.limit - len(prompt))
        sc = m.new_cache(len(prompt) + max_tokens + self.max_rows + 8)
        use_drafts = bool(draft) and self.drafter is not None
        dc = self.drafter.new_cache() if use_drafts else None
        t0 = time.perf_counter()
        last = None
        for s in range(0, len(prompt), PREFILL_CHUNK):
            ids = torch.tensor(prompt[s:s + PREFILL_CHUNK], dtype=torch.long, device="cuda")
            taps: list | None = [] if use_drafts else None
            last = m.forward(sc, ids, s, taps=taps)
            if use_drafts:
                self.drafter.absorb(dc, sc, torch.cat(taps, -1), s)
        first = self._sample(last, [len(prompt)], sampling)[0]
        stats: dict[str, Any] = {"prefill_s": time.perf_counter() - t0}
        t1 = time.perf_counter()
        if use_drafts:
            out, st = self._spec(sc, dc, first, max_tokens, sampling, eos, on_tokens)
            stats.update(rounds=st[0], drafted=st[1], accepted=st[2])
        else:
            out = self._serial(sc, first, max_tokens, sampling, eos, on_tokens)
        dt = time.perf_counter() - t1
        stats.update(decode_s=dt, tokens=len(out), tokens_per_second=(len(out) - 1) / dt if dt > 0 else 0.0,
                     sha256=hashlib.sha256(json.dumps(out).encode()).hexdigest()[:16])
        if self.rank == 0:
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
            lg = self.model.forward(sc, torch.tensor([tok], dtype=torch.long, device="cuda"), p)
            tok = self._sample(lg, [p + 1], sampling)[0]
            out.append(tok)
            on_tokens([tok])
        return out

    def _spec(self, sc, dc, first, max_tokens, sampling, eos, on_tokens):
        """Drafts verified k at a time against the target's keyed samples (the serial rule)."""

        m, d = self.model, self.drafter
        out = [first]
        on_tokens([first])
        tok = first
        rounds = drafted = accepted = 0
        while len(out) < max_tokens and tok not in eos:
            P = sc.length
            drafts, _conf = d.draft(dc, sc, tok, P)
            kk = min(self.drafts, max_tokens - len(out), len(drafts))
            window = [tok] + drafts[:kk]
            taps: list = []
            lg = m.forward(sc, torch.tensor(window, dtype=torch.long, device="cuda"), P, all_logits=True, taps=taps)
            target = self._sample(lg, [P + 1 + i for i in range(kk + 1)], sampling)
            a = 0
            while a < kk and drafts[a] == target[a]:
                a += 1
            new = drafts[:a] + [target[a]]
            rounds += 1
            drafted += kk
            accepted += a
            sc.length = P + a + 1
            d.absorb(dc, sc, torch.cat(taps, -1)[:a + 1], P)
            for i, t in enumerate(new):
                if t in eos:
                    new = new[:i + 1]
                    break
            new = new[:max_tokens - len(out)]
            out += new
            on_tokens(new)
            tok = out[-1]
        return out, (rounds, drafted, accepted)


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
