"""Per-rank weights of DeepSeek-V4.1 on TP ranks (our split, DESIGN.md section 2).

Routed and shared experts: intermediate 2304 split in halves (gate/up by output column, down by input row).
Attention: heads split (wq_b columns, wo_a slices 0-3 / 4-7, wo_b rows, sinks); wq_a, wkv, compressor, indexer
replicated. Engram wkv: input rows split by hash column (12 columns a rank). Head: vocabulary halves. Embedding and
everything per-token small: replicated. EXL3 splits stay on 128-wide Hadamard blocks, so each part is exact.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

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


def _codebook(sh: Shards, prefix: str) -> str:
    return "mul1" if f"{prefix}.mul1" in sh else "mcg" if f"{prefix}.mcg" in sh else "3inst"


def exl3_parts(sh: Shards, prefix: str, cols: tuple[int, int] | None = None, rows: tuple[int, int] | None = None,
               device: str | None = "cuda"):
    """(trellis, suh, svh, codebook) of a group, optionally the output columns or input rows [lo, hi) only (multiples
    of 128); on the GPU unless ``device`` is None."""

    if rows is not None:
        tr = sh.get(f"{prefix}.trellis", rows=(rows[0] // 16, rows[1] // 16))
        suh = sh.get(f"{prefix}.suh")[rows[0]:rows[1]]
    else:
        tr = sh.get(f"{prefix}.trellis")
        suh = sh.get(f"{prefix}.suh")
    svh = sh.get(f"{prefix}.svh")
    if cols is not None:
        tr = tr[:, cols[0] // 16:cols[1] // 16]
        svh = svh[cols[0]:cols[1]]
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
        t = sh.get(name)
        return (t.to(dtype) if dtype is not None else t).contiguous().cuda()

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
    d, H, hd = cfg.dim, cfg.n_heads, cfg.head_dim
    Hl = H // world
    gl = cfg.o_groups // world
    inter_l = cfg.inter // world
    assert inter_l % 128 == 0 and (Hl * hd) % 128 == 0

    def plain(name: str, dtype=None) -> torch.Tensor:
        t = sh.get(name)
        return (t.to(dtype) if dtype is not None else t).contiguous().cuda()

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
            t = sh.get(name)
            return (t.to(dtype) if dtype is not None else t).contiguous().cuda()

        blocks = [load_block(sh, cfg, f"mtp.{j}", cfg.n_layers + j, rank, world, cfg.dspark_routed) for j in range(n)]
        w.dspark = DSparkWeights(blocks, linear(sh, "mtp.0.main_proj"), plain("mtp.0.main_norm.weight"),
                                 plain(f"{last}.norm.weight"), plain(f"{last}.markov_head.embed.weight"),
                                 plain(f"{last}.markov_head.head.weight"), plain(f"{last}.confidence_head.proj.weight"))
        log(f"[tensorfold] rank {rank}: DSpark ({n} stages) loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    sh.close()
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

