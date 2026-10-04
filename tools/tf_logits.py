"""The lane's logits at every position of the KL reference panel's rows, for kld_ref_score.py (the scoring host's
pipeline/kld-ref): each row's input_ids as a fresh prompt through this engine's prompt path (the lane's settings: cache
format, prompt kernels), the final norm on every row, the head as decode steps run it (the row-invariant EXL3 linear),
the ranks' vocabulary slices gathered on rank 0. Rank 0 either scores each row against the reference in-process
(kld_ref_score's measures) or writes A_DIR/rowNNN.safetensors ("logits" bf16 [L, vocab], "input_ids") for the scorer.

usage (TP4, lane stopped; tp4_run.sh):
  python3 tools/tf_logits.py MODEL RANK MASTER PORT CONTEXT REF_DIR [A_DIR]
  REF_DIR: the reference (manifest.json, rowNNN.safetensors with input_ids and fp32 logits); its kld_ref_score.py is
  imported from TF_KL_SCORER (default REF_DIR/../kld_ref_score.py, else REF_DIR/kld_ref_score.py). With A_DIR the
  rows are written instead of scored. TF_KL_ROWS=N scores the first N rows only. TF_KL_AB=V1,V2,...: score the panel
  once per prompt-path variant in one load (each "+"-joined from deq0 / deqbf16 / deqfp16 / deqauto: TF_GLM_PROMPT_DEQ,
  and inv0 / inv1: TF_GLM_PROMPT_INVARIANT), each with its own summary.
  Rank 0 scores each head call's 128 positions on its GPU as they come: against the reference's same rows, read
  straight from the file into a fixed buffer and dropped from the page cache after (GB10's CUDA free memory does not
  count cached pages, and a long panel's rows are 10 GB each), so nothing but the per-position results stays. A long
  row reports its progress (forward, reference reads, scoring) and the dense positions (0-2,047) and the ones where
  DSA selects (2,048+) apart.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


def scorer(ref: Path):
    for path in (os.environ.get("TF_KL_SCORER"), ref.parent / "kld_ref_score.py", ref / "kld_ref_score.py"):
        if path and Path(path).is_file():
            spec = importlib.util.spec_from_file_location("kld_ref_score", str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("kld_ref_score.py not found: set TF_KL_SCORER")


class RefRows:
    """One reference row's fp32 logits, a few positions at a time: pread from the safetensors file into a pinned buffer
    (no mmap of the 10 GB row), the pages dropped from the page cache after, copied to ``device``."""

    def __init__(self, path: Path, rows_max: int, device) -> None:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            info = json.loads(f.read(n))["logits"]
        if info["dtype"] != "F32":
            raise SystemExit(f"{path}: logits are {info['dtype']}, not F32")
        self.L, self.width = info["shape"]
        self.row_bytes = 4 * self.width
        self.off = 8 + n + info["data_offsets"][0]
        self.fd = os.open(path, os.O_RDONLY)
        pin = torch.device(device).type == "cuda"
        self.host = torch.empty((rows_max, self.width), dtype=torch.float32, pin_memory=pin)
        self.view = memoryview(self.host.numpy()).cast("B")
        self.dev = self.host if not pin else torch.empty((rows_max, self.width), dtype=torch.float32, device=device)

    def fetch(self, r0: int, r1: int) -> int:
        """Positions r0 .. r1-1 into the host buffer (the CPU only: it overlaps the GPU's queued work)."""

        n = r1 - r0
        want, pos, got = n * self.row_bytes, self.off + r0 * self.row_bytes, 0
        while got < want:
            k = os.preadv(self.fd, [self.view[got:want]], pos + got)
            if k <= 0:
                raise OSError(f"short read of the reference at position {r0 + got // self.row_bytes}")
            got += k
        os.posix_fadvise(self.fd, pos, want, os.POSIX_FADV_DONTNEED)
        return n

    def upload(self, n: int) -> torch.Tensor:
        if self.dev is not self.host:
            self.dev[:n].copy_(self.host[:n])
        return self.dev[:n]

    def read(self, r0: int, r1: int) -> torch.Tensor:
        return self.upload(self.fetch(r0, r1))

    def close(self) -> None:
        os.close(self.fd)


def meminfo() -> str:
    """CUDA's free memory beside the kernel's view (GB10: cached file pages count as used for CUDA)."""

    kb = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            kb[k] = int(v.split()[0])
    free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else float("nan")
    return (f"CUDA free {free:.1f} GiB; MemFree {kb['MemFree'] / 2**20:.1f}, MemAvailable {kb['MemAvailable'] / 2**20:.1f}, "
            f"Cached {kb['Cached'] / 2**20:.1f} GiB")


class RowScore:
    """One row scored block by block with kld_ref_score's definitions: KL(A, B) = KL(P_B || P_A) and KL(B, A) over every
    position, log-probs and top-1 agreement over positions 0 .. L-2 (each against the next token)."""

    def __init__(self, mod, length: int, vocab: int, ids: torch.Tensor) -> None:
        self.mod, self.L, self.vocab, self.ids = mod, length, vocab, ids
        self.kl_ab: list[torch.Tensor] = []
        self.kl_ba: list[torch.Tensor] = []
        self.lp = [0.0, 0.0]
        self.same = self.after_same = self.after_n = 0

    def add(self, a: torch.Tensor, b: torch.Tensor, a0: int) -> None:
        """Positions a0 .. a0 + len(a) - 1: A's logits ``a`` and the reference's ``b`` ([n, >= vocab] each)."""

        import torch.nn.functional as F

        mod = self.mod
        kv = min(self.vocab, a.shape[-1], b.shape[-1])
        self.kl_ab.append(mod.kl_rows(a, b, kv).cpu())
        self.kl_ba.append(mod.kl_rows(b, a, kv).cpu())
        m = min(a.shape[0], self.L - 1 - a0)                # positions with a next token
        if m <= 0:
            return
        tgt = self.ids[a0 + 1:a0 + 1 + m].to(a.device).long().view(-1, 1)
        for j, x in enumerate((a, b)):
            for c0 in range(0, m, mod.CHUNK):
                c1 = min(m, c0 + mod.CHUNK)
                lsm = F.log_softmax(x[c0:c1, :kv].float(), dim=-1)
                self.lp[j] += lsm.gather(-1, tgt[c0:c1]).sum().item()
        same = a[:m, :kv].argmax(dim=-1) == b[:m, :kv].argmax(dim=-1)
        self.same += int(same.sum())
        dense = max(0, min(m, mod.DENSE - a0))
        self.after_same += int(same[dense:].sum())
        self.after_n += m - dense

    def result(self) -> dict:
        kl_ab, kl_ba = torch.cat(self.kl_ab), torch.cat(self.kl_ba)
        n = self.L - 1
        out = {"kl_ab": kl_ab.mean().item(), "kl_ba": kl_ba.mean().item(), "kl_ab_tokens": kl_ab,
               "logprob_a": self.lp[0], "logprob_b": self.lp[1], "count": n, "top1_agree": self.same / n}
        if kl_ab.numel() > self.mod.DENSE:
            out["kl_ab_first_2048"] = kl_ab[:self.mod.DENSE].mean().item()
            out["kl_ab_after_2048"] = kl_ab[self.mod.DENSE:].mean().item()
            out["top1_after_2048"] = self.after_same / max(1, self.after_n)
        return out


@torch.no_grad()
def main() -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd, glue
    from tensorfold.families.glm_moe_dsa.cuda import x3 as x3mod
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    model, rank, master, port, context = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), \
        int(sys.argv[5])
    ref = Path(sys.argv[6])
    a_dir = Path(sys.argv[7]) if len(sys.argv) > 7 else None

    def say(*a):
        if rank == 0:
            print(*a, flush=True)

    eng = GlmEngine(model, rank=rank, master=master, port=port, policy="3", context=context, context_explicit=True)
    e, w = eng.e, eng.w
    c = w.cfg
    man = json.loads((ref / "manifest.json").read_text()) if rank == 0 else None
    rows = eng._share([man["rows"], man["length"], man["vocab_size"]] if rank == 0 else None)
    n_rows, length, vocab = rows
    n_rows = min(n_rows, int(os.environ.get("TF_KL_ROWS") or n_rows))
    blocks = -(-c.vocab // 128)
    spans = [x3mod.vocab_slice(blocks, w.world, r) for r in range(w.world)]
    width = w.head.n                                    # every rank's padded slice width
    kept = [max(0, min(mine * 128, c.vocab - lo * 128)) for lo, mine, _ in spans]
    mod = scorer(ref) if rank == 0 and a_dir is None else None
    piece = 128                                         # positions a head call (and a scoring step)
    if a_dir is not None and rank == 0:
        a_dir.mkdir(parents=True, exist_ok=True)
    what = f"writing {a_dir}" if a_dir else "scoring"
    say(f"== {n_rows} rows x {length} tokens, vocab {vocab}; settings TF_GLM_KV={os.environ.get('TF_GLM_KV', '')} "
        f"TF_GLM_PROMPT_DEQ={x3mod.PROMPT_DEQ}; {what}")
    say(f"== memory after the load: {meminfo()}")

    def score(label: str) -> None:
        """The panel's rows through the prompt path as it is set now, scored (or written) row by row."""

        per_row, toks, lp, count = [], [], [0.0, 0.0], 0
        for i in range(n_rows):
            t0 = time.perf_counter()
            path = ref / f"row{i:03d}.safetensors"
            rf = safe_open(str(path), framework="pt", device="cpu") if rank == 0 else None
            ids_t = rf.get_tensor("input_ids") if rank == 0 else None
            ids = eng._share(ids_t.tolist() if rank == 0 else None)
            e.reset()
            e.kept = None
            logits: list = []                               # rank 0, writing: the whole row
            rs = RowScore(mod, len(ids), vocab, ids_t) if rank == 0 and a_dir is None else None
            refs = RefRows(path, piece, w.device) if rs is not None else None
            if refs is not None and refs.L < len(ids):
                raise SystemExit(f"{path}: {refs.L} reference positions for {len(ids)} tokens")
            t_read = t_gpu = t_score = 0.0
            for start in range(0, len(ids), e.prefill_rows):
                chunk = ids[start:start + e.prefill_rows]
                R = fwd.stage(w, st, b, chunk)
                fwd.compute(w, st, b, R, logits=False, nch=fwd.chunks_for(st, R), host_pos=st.pos)
                glue.rmsnorm(b.x[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
                for r0 in range(0, R, piece):           # the head as decode steps run it, 128 rows a call
                    r1 = min(R, r0 + piece)
                    n = r1 - r0
                    part = torch.empty((n, width), dtype=torch.bfloat16, device=w.device)
                    w.head(b.fnormed[r0:r1], part, b.x3, prefill=False)
                    got = torch.empty((w.world * n * width // 2,), dtype=torch.float32, device=w.device)
                    w.comm.all_gather(part.view(torch.float32).view(-1), got)
                    if rank != 0:
                        continue
                    g = got.view(torch.bfloat16).view(w.world, n, width)
                    a = torch.cat([g[r, :, :kept[r]] for r in range(w.world)], dim=1)
                    if rs is None:
                        logits.append(a.cpu())
                        continue
                    t1 = time.perf_counter()
                    refs.fetch(start + r0, start + r1)
                    t2 = time.perf_counter()
                    torch.cuda.synchronize()                # this chunk's forward and this piece's head
                    t3 = time.perf_counter()
                    rs.add(a, refs.upload(n), start + r0)   # syncs: the per-position results come back to the host
                    t_score += time.perf_counter() - t3
                    t_gpu += t3 - t2
                    t_read += t2 - t1
                fwd.commit(w, st, b, R, R)
                if rs is not None and len(ids) > 4 * mod.DENSE:
                    say(f"   row {i}: {start + R} / {len(ids)} positions, {time.perf_counter() - t0:.1f} s (reference "
                        f"reads {t_read:.1f} s, waiting on the forward {t_gpu:.1f} s, scoring {t_score:.1f} s)")
            torch.cuda.synchronize()
            if rank != 0:
                continue
            if a_dir is not None:
                save_file({"logits": torch.cat(logits), "input_ids": torch.tensor(ids, dtype=torch.int64)},
                          str(a_dir / f"row{i:03d}.safetensors"))
                say(f"row {i:3d}: written ({time.perf_counter() - t0:.1f} s)")
                continue
            refs.close()
            s = rs.result()
            toks.append(s.pop("kl_ab_tokens"))
            lp[0] += s["logprob_a"]
            lp[1] += s["logprob_b"]
            count += s["count"]
            per_row.append(s)
            split = (f"  [first {mod.DENSE}: {s['kl_ab_first_2048']:.5f}, after: {s['kl_ab_after_2048']:.5f}]"
                     if "kl_ab_after_2048" in s else "")
            say(f"row {i:3d}: KL(A, B) {s['kl_ab']:.5f}  KL(B, A) {s['kl_ba']:.5f}  top-1 {s['top1_agree']:.4f}{split} "
                f"({time.perf_counter() - t0:.1f} s)")
            if len(ids) > 4 * mod.DENSE:
                say(f"   memory: {meminfo()}")
        if rank == 0 and per_row:
            allt = torch.cat(toks)
            n = len(per_row)
            say(f" -- KL divergence (A, B): {sum(r['kl_ab'] for r in per_row) / n:.8f}")
            say(f" -- KL divergence (B, A): {sum(r['kl_ba'] for r in per_row) / n:.8f}")
            say(f" -- KL (A, B) per token: median {allt.median().item():.8f}   p90 {allt.quantile(0.9).item():.8f}")
            say(f" -- top-1 agreement: {sum(r['top1_agree'] for r in per_row) / n:.4f}")
            if all("kl_ab_after_2048" in r for r in per_row):
                say(f" -- KL (A, B), positions 0-{mod.DENSE - 1} (dense): "
                    f"{sum(r['kl_ab_first_2048'] for r in per_row) / n:.8f}   positions {mod.DENSE}+ (DSA selects): "
                    f"{sum(r['kl_ab_after_2048'] for r in per_row) / n:.8f}   top-1 there: "
                    f"{sum(r['top1_after_2048'] for r in per_row) / n:.4f}")
            say(f" -- perplexity A {math.exp(-lp[0] / count):.6f}   B {math.exp(-lp[1] / count):.6f}")

    st, b = e.st, e.pbuf
    variants = [v for v in (os.environ.get("TF_KL_AB") or "").split(",") if v] or [""]
    for label in variants:
        for setting in [x for x in label.split("+") if x]:
            if setting.startswith("deq"):
                x3mod.PROMPT_DEQ = setting[3:]
            elif setting in ("inv0", "inv1"):
                fwd.invariant.INVARIANT = setting == "inv1"
            else:
                raise SystemExit(f"TF_KL_AB: unknown setting {setting}")
        if label:
            say(f"== variant {label}: TF_GLM_PROMPT_DEQ={x3mod.PROMPT_DEQ} "
                f"TF_GLM_PROMPT_INVARIANT={int(fwd.invariant.INVARIANT)}")
        score(label)
    eng.comm.barrier()


if __name__ == "__main__":
    main()
