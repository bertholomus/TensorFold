"""Load one rank of GLM-5.3's EXL3 checkpoint: trellis groups wherever the conversion made them (routed experts at any
width and codebook, and ExLlamaV3's attention, MLP, MTP and head groups), stored weights elsewhere, world-parameterized."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.geometry import share, share_lo

from tensorfold.families.glm5_next.cuda import latent
from tensorfold.families.glm5_next.cuda.qmm import B16, make_b16, stack_b16

from .x3 import X3, X3Pair

PREFIX = "model.language_model."      # also bare "model." (the reference layout): both tried below


@dataclass(frozen=True)
class Config:
    """The architecture numbers the loader, buffers and kernels need (see config.Config for the full set)."""

    hidden: int
    layers: int
    vocab: int
    eps: float
    heads: int
    q_lora: int
    kv_lora: int
    qk_nope: int
    qk_rope: int
    v_dim: int
    index_heads: int
    index_dim: int
    index_topk: int
    indexer_types: list[str]
    experts: int
    top_k: int
    moe_width: int
    shared_width: int
    dense_width: int
    routed_scale: float
    norm_topk: bool
    shared: int
    streams: int                 # always 1: GLM-5.3 has no hyper-connections
    lin_heads: int               # always 0: no KDA layers (kept so Flash's Buffers constructor runs)
    quant: str                   # always "exl3": this family's only read path
    rope_theta: float
    mlp_kinds: list[str]       # per layer: "dense" or "moe"
    eos: tuple[int, ...]
    mtp_layers: int

    @property
    def qk_dim(self) -> int:
        return self.qk_nope + self.qk_rope

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        from tensorfold.families.glm_moe_dsa.config import Config as Full

        raw = json.loads((Path(model_dir) / "config.json").read_text())
        full = Full.from_dict(raw)
        dense = int(raw.get("first_k_dense_replace") or sum(k == "dense" for k in full.mlp_layer_types))
        return cls(hidden=full.hidden_size, layers=full.num_hidden_layers, vocab=full.vocab_size,
                   eps=full.rms_norm_eps, heads=full.num_attention_heads, q_lora=full.q_lora_rank,
                   kv_lora=full.kv_lora_rank, qk_nope=full.qk_nope_head_dim, qk_rope=full.qk_rope_head_dim,
                   v_dim=full.v_head_dim, index_heads=full.index_n_heads, index_dim=full.index_head_dim,
                   index_topk=full.index_topk, indexer_types=list(full.indexer_types),
                   experts=full.n_routed_experts, top_k=full.num_experts_per_tok,
                   moe_width=full.moe_intermediate_size, shared_width=full.moe_intermediate_size * full.n_shared_experts,
                   dense_width=full.intermediate_size, routed_scale=full.routed_scaling_factor,
                   norm_topk=full.norm_topk_prob, shared=full.n_shared_experts, streams=1, lin_heads=0,
                   quant="exl3", rope_theta=full.rope_theta,
                   mlp_kinds=["moe" if k == "sparse" else "dense" for k in full.mlp_layer_types],
                   eos=full.eos_token_id, mtp_layers=full.num_nextn_predict_layers)

    @property
    def dense_limit(self) -> int:
        """Largest context (tokens) where DSA's top-k keeps every visible key: past it rows attend sparsely."""

        return self.index_topk


@dataclass
class IndexW:
    """One full-indexer layer's weights: key projection, LayerNorm, per-head scoring weights, query projection."""

    kw: B16                   # [index_dim, hidden] bf16
    qb: B16 | X3              # [heads * index_dim, q_lora]
    ln_w: torch.Tensor        # [index_dim] bf16
    ln_b: torch.Tensor        # [index_dim] bf16
    weights: torch.Tensor     # [heads, hidden] bf16 (raw per-head weights; the scorer folds the scales)


@dataclass
class DSAW:
    """One MLA layer's weights, with kv_b's per-head key/value blocks kept whole (BF16, never quantized)."""

    q_a: B16 | X3             # [q_lora, hidden]
    q_b: B16 | X3             # [heads * qk_dim, q_lora]
    kv_a: B16 | X3            # [kv_lora + qk_rope, hidden] (an EXL3 group pads it to 640; X3 crops)
    kv_k: B16                 # [heads * qk_dim, kv_lora] key rows of kv_b
    kv_v: B16                 # [heads * v_dim, kv_lora] value rows
    o: B16 | X3               # [hidden, heads * v_dim]
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    heads: int
    index: IndexW | None = None     # only a "full" layer holds one; "shared" layers reuse the group's selection
    absorb: Any = None              # latent.AbsorbW: kv_b split per head, for attention on the latent cache


@dataclass
class MLPW:
    gu: B16 | X3Pair          # this rank's [gate | up] rows
    down: B16 | X3            # [hidden, this rank's width]
    width: int


@dataclass
class RoutedW:
    """One layer's routed experts on this rank for ``tensorfold.cuda.exl3.experts`` (any codebook, a width per matrix).

    ``ex`` is the prepared layer (trellis pointers, widths, stacked scales); on a CPU load (the tests) it stays None
    and ``parts`` keeps the rank's (trellis, suh, svh) triples so shapes can be checked without a GPU.
    """

    count: int                # E
    dims: int                 # D (model width)
    width: int                # I on this rank
    codebook: str
    ex: Any = None            # tensorfold.cuda.exl3.experts.Exl3RoutedExperts
    parts: dict | None = None


@dataclass
class MoEW:
    router: torch.Tensor      # [E, hidden] bf16
    bias: torch.Tensor        # [E] fp32
    experts: RoutedW          # the E routed experts (EXL3, any width)
    shared: MLPW | None       # the shared expert (EXL3 group or BF16)


@dataclass
class LayerW:
    index: int
    kind: str                 # always "dsa"
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    attn_hc: None = None
    ffn_hc: None = None
    kda: None = None
    dsa: DSAW | None = None
    mlp: MLPW | None = None
    moe: MoEW | None = None


@dataclass
class MTPW:
    enorm: torch.Tensor
    hnorm: torch.Tensor
    eh: B16                   # [hidden, 2 * hidden]
    norm: torch.Tensor        # shared_head.norm
    layer: LayerW             # DSA (own full indexer) + the full MoE, plain residual


@dataclass
class Weights:
    cfg: Config
    embed: torch.Tensor       # [vocab, hidden] bf16 (replicated)
    layers: list[LayerW]
    norm: torch.Tensor
    head: B16 | X3            # this rank's vocabulary slice (an EXL3 head pads every rank to one width)
    mtp: MTPW | None
    rank: int
    world: int
    device: torch.device
    comm: Any = None
    meta: dict = field(default_factory=dict)
    draft_head: None = None   # BF16 heads keep the verification head; no quantized copy is needed

    @property
    def vocab_offset(self) -> int:
        per = self.meta.get("vocab_per_rank")         # an EXL3 head: whole 128-column blocks a rank
        if per is not None:
            return min(self.rank * int(per), self.cfg.vocab)   # padded tails of last ranks hold no rows
        return share_lo(self.cfg.vocab, self.world, self.rank)

    def nbytes(self) -> int:
        total = 0
        seen: set[int] = set()

        def add(t: Any) -> None:
            nonlocal total
            if isinstance(t, torch.Tensor) and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                total += t.numel() * t.element_size()
            elif isinstance(t, (X3, X3Pair)):
                total += t.nbytes()
            elif isinstance(t, (B16, RoutedW, DSAW, MLPW, MoEW, LayerW, MTPW, IndexW)):
                for v in vars(t).values():
                    add(v)
            elif isinstance(t, (list, tuple)):
                for v in t:
                    add(v)
            elif hasattr(t, "gate_ptr"):                  # a prepared Exl3RoutedExperts
                add([v for v in vars(t).values() if isinstance(v, (torch.Tensor, list))])
            elif isinstance(t, dict):
                for v in t.values():
                    add(v)

        add(self.embed)
        add(self.layers)
        add(self.norm)
        add(self.head)
        add(self.mtp)
        return total


def load(model_dir: str | Path, *, rank: int, device: str = "cuda") -> Weights:
    """Read one of ``TF_TP_WORLD`` ranks from a full checkpoint or rank folder, MTP layer included.

    Each projection is read the way the checkpoint stores it: an EXL3 trellis group (``X.trellis`` with its scales and
    codebook marker) becomes an ``X3`` on the row-invariant EXL3 linear, a stored ``X.weight`` a BF16 ``B16``. The
    split rules (``split.rule``) hand each rank its share of either form.
    """

    from tensorfold.families.glm5_next.cuda.split import RankReader

    from . import x3 as x3mod

    world = int(os.environ.get("TF_TP_WORLD", "2"))
    cfg = Config.read(model_dir)
    dev = torch.device(device)
    rd = RankReader(model_dir, rank, cfg_hint={"qk_dim": cfg.qk_dim, "qk_nope": cfg.qk_nope, "v_dim": cfg.v_dim})
    HL = share(cfg.heads, world, rank)                # 64 heads: 16 a rank at TP4, 11/.../10 at TP6
    x3_users: list = []

    def name(raw: str) -> str:
        """The checkpoint's name for a model tensor: bare (the reference layout) or prefixed (a merged conversion)."""

        if "model." + raw in rd.index:
            return "model." + raw
        if PREFIX + raw in rd.index:
            return PREFIX + raw
        if raw in rd.index:
            return raw
        raise KeyError(f"{raw}: the checkpoint holds no such tensor")

    def has(raw: str) -> bool:
        try:
            name(raw)
            return True
        except KeyError:
            return False

    def t(raw: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        x = rd.get(name(raw))
        if dtype is not None:
            x = x.to(dtype)
        return x.to(dev)

    def group(p: str) -> dict:
        """An EXL3 group's parts on this rank (trellis, in/out scales, codebook marker, bias)."""

        got = {"trellis": rd.get(name(p + "trellis"))}
        for part in ("suh", "su", "svh", "sv", "bias", "mul1", "mcg", "3inst"):
            if has(p + part):
                got[part] = rd.get(name(p + part))
        return got

    def x3(p: str, crop: int = 0, n_pad: int = 0) -> X3:
        g = group(p)
        suh = g.get("suh", g.get("su"))
        svh = g.get("svh", g.get("sv"))
        lin = x3mod.make(g["trellis"], suh, svh, x3mod.codebook(g), dev, bias=g.get("bias")).lin
        out = X3(lin, n_pad=n_pad, crop=crop)
        x3_users.append(out)
        return out

    def proj(p: str, crop: int = 0) -> B16 | X3:
        """``p`` (``...q_a_proj.``): its EXL3 group when the checkpoint made one, else its stored weight."""

        if has(p + "trellis"):
            return x3(p, crop=crop)
        return make_b16(t(p + "weight").contiguous())

    def b16(raw: str) -> B16:
        return make_b16(t(raw).contiguous())

    def stack(names: list[str]) -> B16:
        return stack_b16([t(n) for n in names])

    def indexer(i: int) -> IndexW:
        p = f"layers.{i}.self_attn.indexer."
        return IndexW(b16(p + "wk.weight"), proj(p + "wq_b."), t(p + "k_norm.weight", torch.bfloat16),
                      t(p + "k_norm.bias", torch.bfloat16),
                      t(p + "weights_proj.weight", torch.bfloat16).view(cfg.index_heads, cfg.hidden))

    def dsa(i: int, full: bool = True) -> DSAW:
        p = f"layers.{i}.self_attn."
        # kv_b_proj is stored head-major [H * (qk_nope + v_dim), kv_lora]: each head's qk_nope key rows, then its
        # v_dim value rows (ExLlamaV3 views it [H, nope + v, kv_lora]). The absorb kernel dots a head's whole
        # qk_dim query against wk [H, qk_dim, kv_lora]: the rope rows are zero, so it computes q_nope . W_UK and the
        # rope term comes from the separate q_pe . k_pe score (mla_pe).
        w = t(p + "kv_b_proj.weight", torch.bfloat16)
        per = w.view(HL, cfg.qk_nope + cfg.v_dim, cfg.kv_lora)
        k_nope = per[:, :cfg.qk_nope]                                     # [HL, qk_nope, kv_lora]
        v_rows = per[:, cfg.qk_nope:]                                     # [HL, v_dim, kv_lora]
        wk = torch.zeros((HL, cfg.qk_dim, cfg.kv_lora), dtype=torch.float32, device=w.device)
        wk[:, :cfg.qk_nope] = k_nope.float()
        kv_k = make_b16(k_nope.reshape(HL * cfg.qk_nope, cfg.kv_lora).contiguous())
        kv_v = make_b16(v_rows.reshape(HL * cfg.v_dim, cfg.kv_lora).contiguous())
        absorb = latent.AbsorbW(wk, v_rows.float())
        return DSAW(proj(p + "q_a_proj."), proj(p + "q_b_proj."),
                    proj(p + "kv_a_proj_with_mqa.", crop=cfg.kv_lora + cfg.qk_rope),
                    kv_k, kv_v, proj(p + "o_proj."), t(p + "q_a_layernorm.weight", torch.bfloat16),
                    t(p + "kv_a_layernorm.weight", torch.bfloat16), HL, indexer(i) if full else None, absorb)

    def mlp(p: str) -> MLPW:
        if has(p + "gate_proj.trellis"):
            gu = X3Pair(x3(p + "gate_proj."), x3(p + "up_proj."))
            return MLPW(gu, x3(p + "down_proj."), gu.gate.n)
        gu = stack([p + "gate_proj.weight", p + "up_proj.weight"])
        return MLPW(gu, make_b16(t(p + "down_proj.weight").contiguous()), share(gu.n, world))

    def moe(p: str) -> MoEW:
        from tensorfold.cuda.exl3 import experts as x3experts

        router = t(p + "gate.weight", torch.bfloat16).contiguous()
        bias = t(p + "gate.e_score_correction_bias", torch.float32).contiguous()
        # every rank holds ALL routed experts, each at its share of the width (the split rules cut gate/up
        # trellis columns and down rows: EXL3_RULES). The router picks global expert ids, so the expert set must
        # stay whole on every rank; the rank's fp32 partial is summed over ranks by the all-gather.
        # (An expert-subset split would need a per-rank id remap AND unsplit expert reads; it is not wired.)
        mine = cfg.experts
        experts_of = range(cfg.experts)
        parts: dict[str, list] = {}
        cb = None
        D = I = 0
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            rows = []
            for e in experts_of:
                g = group(f"{p}experts.{e}.{proj_name}.")
                cb = cb or x3mod.codebook(g)
                trellis = g["trellis"].to(dev).contiguous()
                if proj_name == "gate_proj":
                    D, I = trellis.shape[0] * 16, trellis.shape[1] * 16
                rows.append((trellis, g.get("suh", g.get("su")).to(dev, torch.float16),
                             g.get("svh", g.get("sv")).to(dev, torch.float16)))
            parts[proj_name] = rows
        if not mine:
            # a rank with no routed experts (only possible below the real scale): a zero-width layer
            return MoEW(router, bias, RoutedW(0, D or cfg.hidden, I or cfg.moe_width, cb or "mul1", None, parts),
                        None if not cfg.shared else mlp(p + "shared_experts."))
        ex = None
        if dev.type == "cuda":
            ex = x3experts.prepare(parts["gate_proj"], parts["up_proj"], parts["down_proj"], cb, device=dev)
        routed = RoutedW(mine, D, I, cb, ex, None if ex is not None else parts)
        return MoEW(router, bias, routed, None if not cfg.shared else mlp(p + "shared_experts."))

    def layer(i: int) -> LayerW:
        lw = LayerW(i, "dsa", t(f"layers.{i}.input_layernorm.weight", torch.bfloat16),
                    t(f"layers.{i}.post_attention_layernorm.weight", torch.bfloat16))
        lw.dsa = dsa(i, full=cfg.indexer_types[i] == "full")
        if cfg.mlp_kinds[i] == "dense":
            lw.mlp = mlp(f"layers.{i}.mlp.")
        else:
            lw.moe = moe(f"layers.{i}.mlp.")
        return lw

    def lm_head(meta: dict) -> B16 | X3:
        """This rank's vocabulary slice: BF16 rows vocab/world, or an EXL3 head in whole 128-column blocks."""

        if "lm_head.trellis" in rd.index:
            g = {"trellis": rd.get("lm_head.trellis")}
            for part in ("suh", "su", "svh", "sv", "mul1", "mcg", "3inst"):
                if "lm_head." + part in rd.index:
                    g[part] = rd.get("lm_head." + part)
            blocks = g["trellis"].shape[1] // 8                     # the trellis's 16-column tiles, 8 a block
            lo, mine, per = x3mod.vocab_slice(blocks, world, rank)
            svh = g.get("svh", g.get("sv"))
            tr = g["trellis"][:, lo * 8:(lo + mine) * 8].contiguous()
            head = X3(x3mod.make(tr, g.get("suh", g.get("su")), svh[lo * 128:(lo + mine) * 128].contiguous(),
                                 x3mod.codebook(g), dev).lin, n_pad=per * 128,
                      crop=max(0, min(mine * 128, cfg.vocab - lo * 128)))
            x3_users.append(head)
            meta["vocab_per_rank"] = per * 128
            return head
        vl = share(cfg.vocab, world, rank)
        lo = share_lo(cfg.vocab, world, rank)
        return make_b16(t("lm_head.weight")[lo:lo + vl].to(dev).contiguous())

    embed = t("embed_tokens.weight", torch.bfloat16).contiguous().to(dev)
    try:
        built = [layer(i) for i in range(cfg.layers)]
        meta: dict = {}
        head = lm_head(meta)
        mtpw = None
        if cfg.mtp_layers and has(f"layers.{cfg.layers}.enorm.weight"):
            i = cfg.layers
            mtpw = MTPW(t(f"layers.{i}.enorm.weight", torch.bfloat16), t(f"layers.{i}.hnorm.weight", torch.bfloat16),
                        proj(f"layers.{i}.eh_proj."), t(f"layers.{i}.shared_head.norm.weight", torch.bfloat16),
                        LayerW(i, "dsa", t(f"layers.{i}.input_layernorm.weight", torch.bfloat16),
                               t(f"layers.{i}.post_attention_layernorm.weight", torch.bfloat16)))
            mtpw.layer.dsa = dsa(i, full=True)
            mtpw.layer.moe = moe(f"layers.{i}.mlp.")
        w = Weights(cfg, embed, built, t("norm.weight", torch.bfloat16), head, mtpw, rank, world, dev)
        w.meta.update(layers=list(range(cfg.layers)), x3=x3_users, **meta)
    finally:
        rd.close()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return w
