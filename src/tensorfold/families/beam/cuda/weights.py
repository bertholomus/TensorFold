"""Load one tensor-parallel rank of Beam-1.0's EXL3 pack.

Attention splits by heads (q, gate and the gate bias by query head, k and v by key head, o_proj by its input rows),
the dense SwiGLU layers and the routed experts by width (gate/up columns, down rows), the head by vocabulary rows.
Every split falls on 128-column blocks, so each rank's EXL3 groups keep whole Hadamard blocks and a rank's partial
sums add up to the whole layer. The head's vocabulary rows may split unevenly (``vocab_rows``: a node with less room
keeps fewer; each logit is its own dot product, so the logits keep their bits, tools/beam_head_split_check.py).
The embedding splits like the head, by vocabulary rows, or stays whole in the page
cache (``embed="host"``: each rank copies its rows from the file's pages, which MemAvailable counts as available);
the norms and the router stay whole on every rank. A layer's routed experts sit in one allocation a projection (a slab), each expert's trellis
a contiguous slot in it."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.cuda.exl3 import experts as exl3_experts
from tensorfold.cuda.exl3.linear import Exl3Linear
from tensorfold.families.beam.config import Config

PREFIX = "model.layers."


@dataclass
class AttnW:
    q: Exl3Linear                 # [hidden -> rank's query heads * 128]
    k: Exl3Linear                 # [hidden -> rank's key heads * 128]
    v: Exl3Linear
    g: Exl3Linear                 # the output gate's projection, the rank's query heads
    o: Exl3Linear                 # [rank's query heads * 128 -> hidden], a partial sum
    q_norm: torch.Tensor          # bf16 [128]
    k_norm: torch.Tensor
    gate_bias: torch.Tensor       # bf16 [rank's query heads]


@dataclass
class DenseW:
    gate: Exl3Linear              # [hidden -> rank's width]
    up: Exl3Linear
    down: Exl3Linear              # [rank's width -> hidden], a partial sum


@dataclass
class MoEW:
    router: torch.Tensor          # fp32 [experts, hidden], as the pack stores it
    bias: torch.Tensor            # fp32 [experts], the selection bias
    ex: exl3_experts.Exl3RoutedExperts
    slabs: list = field(default_factory=list, repr=False)


@dataclass
class LayerW:
    index: int
    sliding: bool
    attn: AttnW
    mlp: DenseW | MoEW
    ln_in: torch.Tensor           # bf16 [hidden]: input, post-attention, pre-feedforward, post-feedforward norms
    ln_post_attn: torch.Tensor
    ln_pre_ff: torch.Tensor
    ln_post_ff: torch.Tensor


@dataclass
class Weights:
    cfg: Config
    rank: int
    world: int
    layers: dict[int, LayerW]
    embed: torch.Tensor | None    # bf16 [rank's vocab rows, hidden], the head's rows
    head: torch.Tensor | None     # bf16 [rank's vocab rows, hidden]
    head_lo: int                  # the rank's first vocab row (the head's and the embedding's)
    norm: torch.Tensor            # bf16 [hidden]
    embed_host: torch.Tensor | None = None   # bf16 [vocab, hidden] on the host, read in place from the pack's pages
    vocab_starts: tuple[int, ...] = ()       # every rank's first vocab row

    def nbytes(self) -> int:
        seen, total = set(), 0

        def add(t):
            nonlocal total
            if isinstance(t, torch.Tensor) and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                total += t.numel() * t.element_size()

        for t in (self.embed, self.head, self.norm):
            add(t)
        for lw in self.layers.values():
            for lin in (lw.attn.q, lw.attn.k, lw.attn.v, lw.attn.g, lw.attn.o):
                add(lin.words), add(lin.suh), add(lin.svh)
            if isinstance(lw.mlp, DenseW):
                for lin in (lw.mlp.gate, lw.mlp.up, lw.mlp.down):
                    add(lin.words), add(lin.suh), add(lin.svh)
            else:
                for t in (lw.mlp.router, lw.mlp.bias, *lw.mlp.slabs):
                    add(t)
        return total


def span(n: int, world: int, rank: int, unit: int = 128) -> tuple[int, int]:
    """The rank's [lo, hi) of n items in whole units (an even split: Beam's shapes divide at 1, 2 and 4 ranks)."""
    if n % (unit * world):
        raise ValueError(f"{n} does not split into {world} ranks of whole {unit}-blocks")
    per = n // world
    return rank * per, (rank + 1) * per


def vocab_spans(n: int, world: int, rows=None, unit: int = 16) -> list[tuple[int, int]]:
    """Every rank's [lo, hi) of the vocabulary: even, or ``rows`` (each rank's count, whole units, summing to n)."""
    if not rows:
        return [span(n, world, r, unit=unit) for r in range(world)]
    rows = [int(r) for r in rows]
    if len(rows) != world or sum(rows) != n or any(r <= 0 or r % unit for r in rows):
        raise ValueError(f"vocab rows {rows}: {world} positive counts of whole {unit}-row units summing to {n}")
    lo = [sum(rows[:r]) for r in range(world)]
    return [(lo[r], lo[r] + rows[r]) for r in range(world)]


def mapped(path: Path, begin: int, n: int) -> np.ndarray:
    """n uint16 words of the file from byte ``begin``, read in place (read-only) after one pass that brings the pages
    into the page cache."""
    with open(path, "rb", buffering=0) as f:
        f.seek(begin)
        left = 2 * n
        while left > 0:
            got = len(f.read(min(left, 64 << 20)))
            if not got:
                break
            left -= got
    return np.memmap(path, dtype=np.uint16, mode="r", offset=begin, shape=(n,))


class Pack:
    """The pack's tensors by name, read with O_DIRECT straight to the device (``rows``: a slice of the first axis)."""

    def __init__(self, model_dir: Path) -> None:
        files = sorted(p for p in model_dir.glob("*.safetensors"))
        self.st = SafeTensors(files)

    def __contains__(self, name: str) -> bool:
        return name in self.st

    def shape(self, name: str) -> list[int]:
        return list(self.st.where[name][4])

    def get(self, name: str, device, rows: tuple[int, int] | None = None) -> torch.Tensor:
        if rows is None:
            return self.st.get(name, device)
        from tensorfold.cuda.direct_read import DTYPES

        path, begin, n, dtype, shape = self.st.where[name]
        per = n // shape[0]
        lo, hi = rows
        raw = self.st.reader.read(path, begin + lo * per, (hi - lo) * per, device)
        return raw.view(DTYPES[dtype]).reshape([hi - lo] + shape[1:])

    def map(self, name: str) -> np.ndarray:
        """A bf16 tensor's words in place in the page cache: uint16 [shape]."""
        path, begin, n, dtype, shape = self.st.where[name]
        assert dtype == "BF16", (name, dtype)
        return mapped(path, begin, n // 2).reshape(shape)

    def close(self) -> None:
        self.st.close()


def _codebook(pack: Pack, p: str) -> str:
    return "mul1" if f"{p}.mul1" in pack else "mcg" if f"{p}.mcg" in pack else "3inst"


def linear(pack: Pack, p: str, device, *, cols: tuple[int, int] | None = None,
           rows: tuple[int, int] | None = None) -> Exl3Linear:
    """EXL3 group ``p`` on this rank: ``cols`` keeps output columns [lo, hi), ``rows`` input rows [lo, hi)."""
    cb = _codebook(pack, p)
    if rows is not None:
        trellis = pack.get(f"{p}.trellis", device, rows=(rows[0] // 16, rows[1] // 16))
        suh = pack.get(f"{p}.suh", device)[rows[0]:rows[1]]
        svh = pack.get(f"{p}.svh", device)
    else:
        trellis = pack.get(f"{p}.trellis", device)
        suh = pack.get(f"{p}.suh", device)
        svh = pack.get(f"{p}.svh", device)
        if cols is not None:
            trellis = trellis[:, cols[0] // 16:cols[1] // 16]
            svh = svh[cols[0]:cols[1]]
    return Exl3Linear.from_tensors(trellis.contiguous(), suh.contiguous(), svh.contiguous(), cb, device=device)


def experts(pack: Pack, L: int, cfg: Config, rank: int, world: int, device) -> tuple:
    """Layer L's routed experts on this rank: gate/up columns and down rows [lo, hi) of the expert width, each
    projection's trellises in one slab."""
    lo, hi = span(cfg.moe_width, world, rank)
    E, D = cfg.experts, cfg.hidden
    base = f"{PREFIX}{L}.mlp.experts."
    cb = _codebook(pack, f"{base}0.gate_proj")
    t0 = pack.st.where[f"{base}0.gate_proj.trellis"][4]           # [D/16, I/16, 16 * bits]
    w = t0[2]
    slab_g = torch.empty((E, D // 16, (hi - lo) // 16, w), dtype=torch.int16, device=device)
    slab_u = torch.empty_like(slab_g)
    td = pack.st.where[f"{base}0.down_proj.trellis"][4]           # [I/16, D/16, 16 * bits]
    slab_d = torch.empty((E, (hi - lo) // 16, D // 16, td[2]), dtype=torch.int16, device=device)
    gate, up, down = [], [], []
    for e in range(E):
        p = f"{base}{e}."
        for proj, slab, out in (("gate_proj", slab_g, gate), ("up_proj", slab_u, up)):
            t = pack.get(f"{p}{proj}.trellis", device)
            if tuple(t.shape) != (D // 16, cfg.moe_width // 16, w):
                raise ValueError(f"{p}{proj}: trellis {tuple(t.shape)} is not [{D // 16}, {cfg.moe_width // 16}, {w}]")
            slab[e].copy_(t[:, lo // 16:hi // 16])
            out.append((slab[e], pack.get(f"{p}{proj}.suh", device).contiguous(),
                        pack.get(f"{p}{proj}.svh", device)[lo:hi].contiguous()))
        slab_d[e].copy_(pack.get(f"{p}down_proj.trellis", device, rows=(lo // 16, hi // 16)))
        down.append((slab_d[e], pack.get(f"{p}down_proj.suh", device)[lo:hi].contiguous(),
                     pack.get(f"{p}down_proj.svh", device).contiguous()))
    ex = exl3_experts.prepare(gate, up, down, cb, device=device)
    return ex, [slab_g, slab_u, slab_d]


def layer(pack: Pack, cfg: Config, L: int, rank: int, world: int, device) -> LayerW:
    p = f"{PREFIX}{L}."
    hd = cfg.head_dim
    qlo, qhi = span(cfg.heads * hd, world, rank, unit=hd)
    klo, khi = span(cfg.kv_heads * hd, world, rank, unit=hd)

    def plain(name: str) -> torch.Tensor:
        return pack.get(name, device).contiguous()

    attn = AttnW(q=linear(pack, f"{p}self_attn.q_proj", device, cols=(qlo, qhi)),
                 k=linear(pack, f"{p}self_attn.k_proj", device, cols=(klo, khi)),
                 v=linear(pack, f"{p}self_attn.v_proj", device, cols=(klo, khi)),
                 g=linear(pack, f"{p}self_attn.gate_proj", device, cols=(qlo, qhi)),
                 o=linear(pack, f"{p}self_attn.o_proj", device, rows=(qlo, qhi)),
                 q_norm=plain(f"{p}self_attn.q_norm.weight"), k_norm=plain(f"{p}self_attn.k_norm.weight"),
                 gate_bias=pack.get(f"{p}self_attn.gate_bias", device)[qlo // hd:qhi // hd].contiguous())
    if cfg.routed(L):
        ex, slabs = experts(pack, L, cfg, rank, world, device)
        mlp = MoEW(router=plain(f"{p}mlp.gate.weight"),
                   bias=plain(f"{p}mlp.gate.e_score_correction_bias").float(), ex=ex, slabs=slabs)
    else:
        lo, hi = span(cfg.intermediate, world, rank)
        mlp = DenseW(gate=linear(pack, f"{p}mlp.gate_proj", device, cols=(lo, hi)),
                     up=linear(pack, f"{p}mlp.up_proj", device, cols=(lo, hi)),
                     down=linear(pack, f"{p}mlp.down_proj", device, rows=(lo, hi)))
    return LayerW(index=L, sliding=cfg.sliding(L), attn=attn, mlp=mlp, ln_in=plain(f"{p}input_layernorm.weight"),
                  ln_post_attn=plain(f"{p}post_attention_layernorm.weight"),
                  ln_pre_ff=plain(f"{p}pre_feedforward_layernorm.weight"),
                  ln_post_ff=plain(f"{p}post_feedforward_layernorm.weight"))


def host_table(words: np.ndarray) -> torch.Tensor:
    """bf16 words in place as a read-only CPU tensor (nothing writes them)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.from_numpy(words).view(torch.bfloat16)


def load(model_dir: str | Path, *, rank: int, world: int, device="cuda", layers=None, embed: bool | str = True,
         head: bool = True, vocab_rows=None, log=print) -> Weights:
    """One rank's weights; ``layers``: the subset to load (all by default). embed: True or "device" (the rank's
    vocabulary rows on the device), "host" (the whole table in the page cache) or False. vocab_rows: each rank's
    head rows (an even split by default)."""
    import time

    root = Path(model_dir)
    cfg = Config.read(root)
    device = torch.device(device)
    pack = Pack(root)
    t0 = time.time()
    try:
        out: dict[int, LayerW] = {}
        for L in (range(cfg.layers) if layers is None else layers):
            out[L] = layer(pack, cfg, L, rank, world, device)
            if log and (L % 8 == 7 or L == cfg.layers - 1):
                log(f"[beam] rank {rank}/{world}: layer {L} loaded, {time.time() - t0:.0f} s, "
                    f"{torch.cuda.memory_allocated(device) / 2**30:.2f} GiB")
        spans = vocab_spans(cfg.vocab, world, vocab_rows)
        vlo, vhi = spans[rank]
        place = embed if isinstance(embed, str) else ("device" if embed else "")
        w = Weights(cfg=cfg, rank=rank, world=world, layers=out,
                    embed=pack.get("model.embed_tokens.weight", device, rows=(vlo, vhi)) if place == "device" else None,
                    head=pack.get("lm_head.weight", device, rows=(vlo, vhi)) if head else None, head_lo=vlo,
                    norm=pack.get("model.norm.weight", device).contiguous(),
                    embed_host=host_table(pack.map("model.embed_tokens.weight")) if place == "host" else None,
                    vocab_starts=tuple(lo for lo, _ in spans))
    finally:
        pack.close()
    return w
