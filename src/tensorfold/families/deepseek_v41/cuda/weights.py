"""Per-rank weights of DeepSeek-V4.1 on TP ranks (our split, DESIGN.md section 2).

Routed and shared experts: intermediate 2304 split in halves (gate/up by output column, down by input row).
Attention: heads split (wq_b columns, wo_a slices 0-3 / 4-7, wo_b rows, sinks); wq_a, wkv, compressor, indexer
replicated. Engram wkv: input rows split by hash column (12 columns a rank). Head: vocabulary halves. Embedding and
everything per-token small: replicated. EXL3 splits stay on 128-wide Hadamard blocks, so each part is exact.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda.exl3 import experts as exl3_experts
from tensorfold.cuda.exl3.linear import Exl3Linear

from ..config import Cfg


class Shards:
    """Header offsets of every tensor in a folder; reads a whole tensor or a dim-0 slice straight from the file."""

    DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I16": torch.int16,
          "I32": torch.int32, "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "U8": torch.uint8}

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.entries: dict[str, tuple[Path, int, dict]] = {}
        for path in sorted(self.root.glob("*.safetensors")):
            with open(path, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
            header.pop("__metadata__", None)
            for name, e in header.items():
                self.entries[name] = (path, 8 + size, e)
        self._files: dict[Path, object] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def _fh(self, path: Path):
        f = self._files.get(path)
        if f is None:
            f = open(path, "rb", buffering=0)
            self._files[path] = f
        return f

    def get(self, name: str, rows: tuple[int, int] | None = None) -> torch.Tensor:
        """The tensor (CPU), or rows [lo, hi) of its first dimension, read without the rest."""

        path, base, e = self.entries[name]
        shape = list(e["shape"])
        dt = self.DT[e["dtype"]]
        lo, hi = e["data_offsets"]
        if rows is not None and shape:
            per = (hi - lo) // max(shape[0], 1)
            lo, hi = lo + rows[0] * per, lo + rows[1] * per
            shape[0] = rows[1] - rows[0]
        f = self._fh(path)
        f.seek(base + lo)
        buf = bytearray(hi - lo)
        view = memoryview(buf)
        got = 0
        while got < len(buf):
            n = f.readinto(view[got:])
            if not n:
                raise IOError(f"short read of {name}")
            got += n
        t = torch.frombuffer(buf, dtype=dt) if buf else torch.empty(0, dtype=dt)
        return t.reshape(shape)

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()


class RankCache:
    """This rank's own bytes of the checkpoint, in the order the loader takes them, in one file (TF_DS_RANK_CACHE, a
    directory): the column and row slices the TP split keeps, the dtypes the loader converts to. The first start writes
    it beside its normal load; later starts read it ahead of use with parallel large reads instead of a seek and a
    read per tensor of every shard (each rank used to read whole trellises to keep half their columns). A key of the
    checkpoint (shard names, sizes, times), the parsed config, the rank split and this file's own source picks the
    file; anything else rebuilds it. The cache is optional: a write error (disk full) turns it off for that start, and a
    file that reads short or out of order is deleted so the next start writes it again."""

    MAGIC = b"TFDSRK01"
    AHEAD = 512 << 20                                    # bytes read ahead of the loader

    def __init__(self, path: Path, index: list | None) -> None:
        self.path, self.index, self.i = path, index, 0
        self.reading = index is not None
        if self.reading:
            from concurrent.futures import ThreadPoolExecutor

            self.fd = os.open(path, os.O_RDONLY)
            self.pool = ThreadPoolExecutor(int(os.environ.get("TF_DS_RANK_CACHE_READERS") or 8))
            self.ahead: deque = deque()
            self.next, self.queued = 0, 0
        else:
            self.tmp = path.with_suffix(".partial")
            self.f = open(self.tmp, "wb")
            self.index, self.at = [], 0

    @classmethod
    def open(cls, root: str | None, model_dir: Path, rank: int, world: int, extra: str) -> "RankCache | None":
        if not root:
            return None
        d = Path(root)
        key = hashlib.sha256()
        key.update(Path(__file__).read_bytes())
        key.update(f"{rank}/{world}/{extra}".encode())
        for f in sorted(Path(model_dir).glob("*.safetensors")) + [Path(model_dir) / "config.json"]:
            st = f.stat()
            key.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
        path = d / f"rank{rank}of{world}-{key.hexdigest()[:20]}.bin"
        try:
            with open(path, "rb") as f:
                f.seek(-16, os.SEEK_END)
                tail = f.read(16)
                if tail[8:] != cls.MAGIC:
                    raise ValueError("no footer")
                at = struct.unpack("<Q", tail[:8])[0]
                f.seek(at)
                index = json.loads(f.read()[:-16])
            return cls(path, index)
        except (OSError, ValueError):
            pass
        try:
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob(f"rank{rank}of{world}-*"):     # one file a rank split: older keys go
                old.unlink()
            # a rank's file holds its share of the checkpoint plus the replicated tensors: no write without room
            # for the whole checkpoint over world, a tenth more, and 2 GiB
            need = sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors")) * 11 // (10 * world)
            if shutil.disk_usage(d).free < need + (2 << 30):
                print(f"[tensorfold] rank {rank}: rank cache off (needs ~{need / 2**30:.0f} GiB free in {d})",
                      flush=True)
                return None
            return cls(path, None)
        except OSError:
            return None

    def _fill(self) -> None:
        while self.next < len(self.index) and self.queued < self.AHEAD:
            _, _, _, off, n = self.index[self.next]
            self.ahead.append(self.pool.submit(self._read, off, n))
            self.queued += n
            self.next += 1

    def _read(self, off: int, n: int):
        buf = np.empty(n, dtype=np.uint8)                 # no zero fill: pages are first touched inside preadv
        view, got = memoryview(buf), 0
        while got < n:
            k = os.preadv(self.fd, [view[got:]], off + got)
            if k <= 0:
                raise IOError(f"short read of {self.path}")
            got += k
        return buf

    def take(self, key: str) -> torch.Tensor:
        if self.i >= len(self.index) or self.index[self.i][0] != key:
            self.path.unlink(missing_ok=True)             # (the next start writes it again)
            raise KeyError(f"rank cache out of order at {self.i}: {key}")
        name, dtype, shape, _, n = self.index[self.i]
        self._fill()
        try:
            buf = self.ahead.popleft().result()
        except OSError:
            self.path.unlink(missing_ok=True)
            raise
        self.queued -= n
        self.i += 1
        dt = getattr(torch, dtype)
        t = torch.frombuffer(buf, dtype=dt) if n else torch.empty(0, dtype=dt)
        return t.reshape(shape)

    def put(self, key: str, t: torch.Tensor) -> None:
        t = t.contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy() if t.numel() else b""
        self.f.write(memoryview(raw))
        self.index.append((key, str(t.dtype).split(".")[1], list(t.shape), self.at, int(t.numel() * t.element_size())))
        self.at += int(t.numel() * t.element_size())

    def close(self, ok: bool = True) -> bool:
        """True when a written file was published (or every kept tensor was read back)."""

        if self.reading:
            for f in self.ahead:
                f.cancel()
            self.pool.shutdown(wait=True)
            os.close(self.fd)
            if ok and self.i != len(self.index):
                self.path.unlink(missing_ok=True)
                return False
            return ok
        try:
            if ok:
                body = json.dumps(self.index).encode()
                self.f.write(body + struct.pack("<Q", self.at) + self.MAGIC)
                self.f.flush()
                os.fsync(self.f.fileno())
            self.f.close()
            if ok:
                os.replace(self.tmp, self.path)
                return True
            self.tmp.unlink(missing_ok=True)
        except OSError:
            self.tmp.unlink(missing_ok=True)
        return False


def _trim() -> None:
    """Hand the loader's freed host buffers back to the OS: glibc keeps them in its arenas otherwise (7 GiB of a
    rank's memory after a load from the rank cache, whose eight readers each fill an arena), memory the page cache and
    the allocator ceiling then lack."""

    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _kept(sh: Shards, key: str, make) -> torch.Tensor:
    """A CPU tensor the loader keeps: from the rank cache when it is reading, else made (and written to it)."""

    rc = getattr(sh, "cache", None)
    if rc is not None and rc.reading:
        return rc.take(key)
    t = make()
    if rc is not None:
        try:
            rc.put(key, t)
        except OSError as exc:                            # a full or failing cache disk: carry on from the shards
            print(f"[tensorfold] rank cache off ({exc}); loading from the shards", flush=True)
            rc.close(ok=False)
            sh.cache = None
    return t


def _codebook(sh: Shards, prefix: str) -> str:
    return "mul1" if f"{prefix}.mul1" in sh else "mcg" if f"{prefix}.mcg" in sh else "3inst"


def exl3_parts(sh: Shards, prefix: str, cols: tuple[int, int] | None = None, rows: tuple[int, int] | None = None,
               device: str | None = "cuda"):
    """(trellis, suh, svh, codebook) of a group, optionally the output columns or input rows [lo, hi) only (multiples
    of 128); on the GPU unless ``device`` is None."""

    key = f"{prefix}|{cols}|{rows}"
    if rows is not None:
        tr = _kept(sh, key + "|tr", lambda: sh.get(f"{prefix}.trellis", rows=(rows[0] // 16, rows[1] // 16)))
        suh = _kept(sh, key + "|suh", lambda: sh.get(f"{prefix}.suh")[rows[0]:rows[1]].contiguous())
    else:
        tr = _kept(sh, key + "|tr", lambda: sh.get(f"{prefix}.trellis")[:, cols[0] // 16:cols[1] // 16].contiguous()
                   if cols is not None else sh.get(f"{prefix}.trellis"))
        suh = _kept(sh, key + "|suh", lambda: sh.get(f"{prefix}.suh"))
    svh = _kept(sh, key + "|svh", lambda: sh.get(f"{prefix}.svh")[cols[0]:cols[1]].contiguous() if cols is not None
                else sh.get(f"{prefix}.svh"))
    tr, suh, svh = tr.contiguous(), suh.contiguous(), svh.contiguous()
    if device is not None:
        tr, suh, svh = tr.to(device), suh.to(device), svh.to(device)
    return (tr, suh, svh, _codebook(sh, prefix))


def pack_trellises(mats: list) -> list:
    """Copy CPU trellises into one GPU buffer (one allocation a layer, no per-tensor rounding); views in order."""

    total = sum(t.numel() for t in mats)
    buf = torch.empty((total,), dtype=torch.int16, device="cuda")
    out, o = [], 0
    for t in mats:
        n = t.numel()
        v = buf[o:o + n].view(t.shape)
        v.copy_(t)
        out.append(v)
        o += n
    return out


def linear(sh: Shards, prefix: str, cols=None, rows=None) -> Exl3Linear:
    tr, suh, svh, cb = exl3_parts(sh, prefix, cols, rows)
    return Exl3Linear.from_tensors(tr, suh, svh, cb)


@dataclass
class Layer:
    idx: int
    ratio: int
    hc_attn: tuple                 # (fn [24, hc*d] f32, scale [3], base [24])
    hc_ffn: tuple
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    wq_a: Exl3Linear
    q_norm: torch.Tensor
    wq_b: Exl3Linear               # local heads' columns
    wkv: Exl3Linear
    kv_norm: torch.Tensor
    sink: torch.Tensor             # local heads [Hl] f32
    wo_a: list                     # local groups' slices
    wo_b: Exl3Linear               # local groups' input rows
    comp_wkv: Exl3Linear | None = None
    comp_wgate: Exl3Linear | None = None
    comp_norm: torch.Tensor | None = None
    idx_wq_b: Exl3Linear | None = None
    idx_proj: torch.Tensor | None = None     # [idx_heads, d] f32
    idx_wk: Exl3Linear | None = None
    idx_k_norm: torch.Tensor | None = None
    gate_w: torch.Tensor | None = None       # [E, d] f32
    gate_b: torch.Tensor | None = None       # [E] f32
    gate_b_vl: torch.Tensor | None = None    # [E] f32: the bias image-span tokens pick experts with (attach_vl_bias)
    experts: object = None                   # Exl3RoutedExperts: routed + the shared expert as the last entry
    engram_wkv: Exl3Linear | None = None     # local hash columns' input rows
    engram_qk: torch.Tensor | None = None    # q_weight * k_weight [hc, d] f32


@dataclass
class Weights:
    cfg: Cfg
    rank: int
    world: int
    embed: torch.Tensor                      # [V, d] bf16 (replicated)
    norm: torch.Tensor
    head: Exl3Linear                         # this rank's vocabulary columns
    vocab_lo: int
    vocab_hi: int
    layers: list = field(default_factory=list)
    model_dir: Path | None = None
    dspark: "DSparkWeights | None" = None


@dataclass
class DSparkWeights:
    blocks: list                  # Layer per stage (window-only attention, dspark_routed experts)
    main_proj: Exl3Linear         # [len(taps) * d -> d]
    main_norm: torch.Tensor
    norm: torch.Tensor
    markov_embed: torch.Tensor    # [V, rank] bf16
    markov_head: torch.Tensor     # [V, rank] fp16
    conf: torch.Tensor            # [1, d + rank] fp16


def load_block(sh: Shards, cfg: Cfg, p: str, i: int, rank: int, world: int, n_experts: int) -> Layer:
    """One block (a backbone layer ``layers.i`` or a DSpark stage ``mtp.j``, numbered i = n_layers + j)."""

    d, H, hd = cfg.dim, cfg.n_heads, cfg.head_dim
    Hl = H // world
    gl = cfg.o_groups // world
    inter_l = cfg.inter // world
    ratio = cfg.compress_ratios[i]

    def plain(name: str, dtype=None) -> torch.Tensor:
        return _kept(sh, f"{name}|{dtype}", lambda: (sh.get(name).to(dtype) if dtype is not None
                                                      else sh.get(name)).contiguous()).cuda()

    lay = Layer(
        idx=i, ratio=ratio,
        hc_attn=(plain(f"{p}.hc_attn_fn", torch.float32), plain(f"{p}.hc_attn_scale", torch.float32),
                 plain(f"{p}.hc_attn_base", torch.float32)),
        hc_ffn=(plain(f"{p}.hc_ffn_fn", torch.float32), plain(f"{p}.hc_ffn_scale", torch.float32),
                plain(f"{p}.hc_ffn_base", torch.float32)),
        attn_norm=plain(f"{p}.attn_norm.weight"), ffn_norm=plain(f"{p}.ffn_norm.weight"),
        wq_a=linear(sh, f"{p}.attn.wq_a"), q_norm=plain(f"{p}.attn.q_norm.weight"),
        wq_b=linear(sh, f"{p}.attn.wq_b", cols=(rank * Hl * hd, (rank + 1) * Hl * hd)),
        wkv=linear(sh, f"{p}.attn.wkv"), kv_norm=plain(f"{p}.attn.kv_norm.weight"),
        sink=plain(f"{p}.attn.attn_sink", torch.float32)[rank * Hl:(rank + 1) * Hl].contiguous(),
        wo_a=[linear(sh, f"{p}.attn.wo_a.slice.{g}") for g in range(rank * gl, (rank + 1) * gl)],
        wo_b=linear(sh, f"{p}.attn.wo_b", rows=(rank * gl * cfg.o_rank, (rank + 1) * gl * cfg.o_rank)),
    )
    if i in cfg.kv_sources:
        lay.comp_wkv = linear(sh, f"{p}.attn.compressor.wkv")
        lay.comp_norm = plain(f"{p}.attn.compressor.norm.weight")
        if ratio > 1:
            lay.comp_wgate = linear(sh, f"{p}.attn.compressor.wgate")
    if i in cfg.index_sources:
        lay.idx_wq_b = linear(sh, f"{p}.attn.indexer.wq_b")
        lay.idx_proj = plain(f"{p}.attn.indexer.weights_proj.weight", torch.float32)
        if i in cfg.kv_sources:
            lay.idx_wk = linear(sh, f"{p}.attn.indexer.wk")
            lay.idx_k_norm = plain(f"{p}.attn.indexer.k_norm.weight")
    lay.gate_w = plain(f"{p}.ffn.gate.weight", torch.float16)
    lay.gate_b = plain(f"{p}.ffn.gate.bias", torch.float32)
    cols = (rank * inter_l, (rank + 1) * inter_l)
    gate, up, down = [], [], []
    cb = None
    for e in list(range(n_experts)) + ["shared"]:
        ep = f"{p}.ffn.experts.{e}" if e != "shared" else f"{p}.ffn.shared_experts"
        g = exl3_parts(sh, f"{ep}.w1", cols=cols, device=None)
        u = exl3_parts(sh, f"{ep}.w3", cols=cols, device=None)
        dn = exl3_parts(sh, f"{ep}.w2", rows=cols, device=None)
        cb = cb or g[3]
        assert g[3] == u[3] == dn[3] == cb
        gate.append(g[:3])
        up.append(u[:3])
        down.append(dn[:3])
    trs = pack_trellises([m[0] for m in gate + up + down])
    E1 = len(gate)
    gate = [(trs[k], m[1].cuda(), m[2].cuda()) for k, m in enumerate(gate)]
    up = [(trs[E1 + k], m[1].cuda(), m[2].cuda()) for k, m in enumerate(up)]
    down = [(trs[2 * E1 + k], m[1].cuda(), m[2].cuda()) for k, m in enumerate(down)]
    lay.experts = exl3_experts.prepare(gate, up, down, cb)
    if i in cfg.engram_layers:
        n_cols = (cfg.engram_ngram - 1) * cfg.engram_heads
        per = n_cols // world * cfg.engram_dim
        lay.engram_wkv = linear(sh, f"{p}.engram.wkv", rows=(rank * per, (rank + 1) * per))
        lay.engram_qk = (plain(f"{p}.engram.q_weight", torch.float32)
                         * plain(f"{p}.engram.k_weight", torch.float32)).contiguous()
    return lay


def load(model_dir: str | Path, rank: int, world: int, n_layers: int | None = None, log=print,
         dspark: bool = False) -> Weights:
    import time

    t0 = time.time()
    cfg = Cfg.read(model_dir)
    sh = Shards(model_dir)
    # TF_DS_RANK_CACHE=<dir>: this rank's bytes in one file (RankCache), written on the first start
    sh.cache = RankCache.open(os.environ.get("TF_DS_RANK_CACHE"), Path(model_dir), rank, world,
                              f"{n_layers}/{bool(dspark)}/{cfg!r}")
    if sh.cache is not None:
        log(f"[tensorfold] rank {rank}: rank cache {'reading' if sh.cache.reading else 'writing'} {sh.cache.path}")
    try:
        return _load(sh, cfg, model_dir, rank, world, n_layers, log, dspark, t0)
    except BaseException:
        if sh.cache is not None:
            sh.cache.close(ok=False)
        raise


def _load(sh, cfg, model_dir, rank, world, n_layers, log, dspark, t0) -> Weights:
    import time

    d, H, hd = cfg.dim, cfg.n_heads, cfg.head_dim
    Hl = H // world
    gl = cfg.o_groups // world
    inter_l = cfg.inter // world
    assert inter_l % 128 == 0 and (Hl * hd) % 128 == 0

    def plain(name: str, dtype=None) -> torch.Tensor:
        return _kept(sh, f"{name}|{dtype}", lambda: (sh.get(name).to(dtype) if dtype is not None
                                                      else sh.get(name)).contiguous()).cuda()

    vocab_l = cfg.vocab // world
    w = Weights(cfg, rank, world, embed=plain("embed.weight"), norm=plain("norm.weight"),
                head=linear(sh, "head", cols=(rank * vocab_l, (rank + 1) * vocab_l)),
                vocab_lo=rank * vocab_l, vocab_hi=(rank + 1) * vocab_l, model_dir=Path(model_dir))
    L = cfg.n_layers if n_layers is None else int(n_layers)
    for i in range(L):
        lay = load_block(sh, cfg, f"layers.{i}", i, rank, world, cfg.n_routed)
        w.layers.append(lay)
        if i % 5 == 4 or i == L - 1:
            log(f"[tensorfold] rank {rank}: layers 0..{i} loaded ({time.time() - t0:.0f} s, "
                f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
    if dspark and cfg.dspark_block:
        n = len(cfg.compress_ratios) - cfg.n_layers
        last = f"mtp.{n - 1}"

        def plain(name: str, dtype=None) -> torch.Tensor:
            return _kept(sh, f"{name}|{dtype}", lambda: (sh.get(name).to(dtype) if dtype is not None
                                                          else sh.get(name)).contiguous()).cuda()

        blocks = [load_block(sh, cfg, f"mtp.{j}", cfg.n_layers + j, rank, world, cfg.dspark_routed) for j in range(n)]
        w.dspark = DSparkWeights(blocks, linear(sh, "mtp.0.main_proj"), plain("mtp.0.main_norm.weight"),
                                 plain(f"{last}.norm.weight"), plain(f"{last}.markov_head.embed.weight"),
                                 plain(f"{last}.markov_head.head.weight"), plain(f"{last}.confidence_head.proj.weight"))
        log(f"[tensorfold] rank {rank}: DSpark ({n} stages) loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    if sh.cache is not None:
        reading = sh.cache.reading
        kept = sh.cache.close(ok=True)
        log(f"[tensorfold] rank {rank}: weights in {time.time() - t0:.0f} s ("
            + ("from the rank cache" if reading else "written to the rank cache" if kept
               else "the rank cache write failed and was dropped") + ")")
    sh.close()
    _trim()
    return w


def attach_vl_bias(w: Weights, folders: list) -> int:
    """The MoE gates' VL bias (``ffn.gate.bias_vl``: inside an image span the gate picks experts with it) from the
    first folder that holds it; an EXL3 pack may leave it out (DeepSeek's original checkpoint has it). Returns the
    number of gates given one."""

    for folder in folders:
        if not folder or not Path(folder).is_dir():
            continue
        sh = Shards(folder)
        found = 0
        blocks = list(w.layers) + (list(w.dspark.blocks) if w.dspark is not None else [])
        for lay in blocks:
            name = (f"layers.{lay.idx}" if lay.idx < w.cfg.n_layers else f"mtp.{lay.idx - w.cfg.n_layers}") + \
                ".ffn.gate.bias_vl"
            if name in sh:
                lay.gate_b_vl = sh.get(name).to(torch.float32).contiguous().cuda()
                found += 1
        sh.close()
        if found:
            return found
    return 0

