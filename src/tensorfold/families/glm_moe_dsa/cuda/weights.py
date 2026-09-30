"""Load one rank of GLM-5.3's EXL3 checkpoint: 4-bit trellis routed experts, BF16 everywhere else, world-parameterized."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tensorfold.cuda import experts as grouped

from tensorfold.families.glm5_next.cuda.exl3_mm import Exl3Experts, words as exl3_words
from tensorfold.families.glm5_next.cuda import latent
from tensorfold.families.glm5_next.cuda.qmm import B16, make_b16, stack_b16

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
    qb: B16                   # [heads * index_dim, q_lora] bf16
    ln_w: torch.Tensor        # [index_dim] bf16
    ln_b: torch.Tensor        # [index_dim] bf16
    weights: torch.Tensor     # [heads, hidden] bf16 (raw per-head weights; the scorer folds the scales)


@dataclass
class DSAW:
    """One MLA layer's weights, with kv_b's per-head key/value blocks kept whole (BF16, never quantized)."""

    q_a: B16                  # [q_lora, hidden]
    q_b: B16                  # [heads * qk_dim, q_lora]
    kv_a: B16                 # [kv_lora + qk_rope, hidden]
    kv_k: B16                 # [heads * qk_dim, kv_lora] key rows of kv_b
    kv_v: B16                 # [heads * v_dim, kv_lora] value rows
    o: B16                    # [hidden, heads * v_dim]
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    heads: int
    index: IndexW | None = None     # only a "full" layer holds one; "shared" layers reuse the group's selection
    absorb: Any = None              # latent.AbsorbW: kv_b split per head, for attention on the latent cache


@dataclass
class MLPW:
    gu: B16                   # this rank's [gate | up] rows
    down: B16                 # [hidden, this rank's width]
    width: int


@dataclass
class MoEW:
    router: torch.Tensor      # [E, hidden] bf16
    bias: torch.Tensor        # [E] fp32
    experts: Exl3Experts      # the E routed experts (4-bit trellis)
    shared: MLPW | None       # the shared expert (BF16)


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
    head: B16                 # this rank's [vocab/world, hidden] slice
    mtp: MTPW | None
    rank: int
    world: int
    device: torch.device
    comm: Any = None
    meta: dict = field(default_factory=dict)
    draft_head: None = None   # BF16 heads keep the verification head; no quantized copy is needed

    @property
    def vocab_offset(self) -> int:
        return self.rank * (self.cfg.vocab // self.world)

    def nbytes(self) -> int:
        total = 0
        seen: set[int] = set()

        def add(t: Any) -> None:
            nonlocal total
            if isinstance(t, torch.Tensor) and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                total += t.numel() * t.element_size()
            elif isinstance(t, (B16, Exl3Experts, DSAW, MLPW, MoEW, LayerW, MTPW, IndexW)):
                for v in vars(t).values():
                    add(v)
            elif isinstance(t, (list, tuple)):
                for v in t:
                    add(v)
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
    """Read one of ``TF_TP_WORLD`` ranks from a full checkpoint or rank folder, MTP layer included."""

    from tensorfold.families.glm5_next.cuda.split import RankReader

    world = int(os.environ.get("TF_TP_WORLD", "2"))
    cfg = Config.read(model_dir)
    dev = torch.device(device)
    rd = RankReader(model_dir, rank)
    HL = cfg.heads // world

    def name(raw: str) -> str:
        """The checkpoint's name for a model tensor: bare (the reference layout) or prefixed (a merged conversion)."""

        if "model." + raw in rd.index:
            return "model." + raw
        if PREFIX + raw in rd.index:
            return PREFIX + raw
        if raw in rd.index:
            return raw
        raise KeyError(f"{raw}: the checkpoint holds no such tensor")

    def t(raw: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        x = rd.get(name(raw))
        if dtype is not None:
            x = x.to(dtype)
        return x.to(dev)

    def b16(raw: str) -> B16:
        return make_b16(t(raw).contiguous())

    def stack(names: list[str]) -> B16:
        return stack_b16([t(n) for n in names])

    def indexer(i: int) -> IndexW:
        p = f"layers.{i}.self_attn.indexer."
        return IndexW(b16(p + "wk.weight"), b16(p + "wq_b.weight"), t(p + "k_norm.weight"),
                      t(p + "k_norm.bias"), t(p + "weights_proj.weight").view(cfg.index_heads, cfg.hidden))

    def dsa(i: int, full: bool = True) -> DSAW:
        p = f"layers.{i}.self_attn."
        # kv_b_proj is stored [H * (qk_nope + v_dim), kv_lora]: per-head blocks of key rows then value rows
        w = t(p + "kv_b_proj.weight")
        kv_k = make_b16(w[:HL * cfg.qk_nope].contiguous())
        kv_v = make_b16(w[HL * cfg.qk_nope:].contiguous())
        absorb = latent.AbsorbW.from_rows(w[:HL * cfg.qk_nope].float(), w[HL * cfg.qk_nope:].float(), HL)
        return DSAW(b16(p + "q_a_proj.weight"), b16(p + "q_b_proj.weight"), b16(p + "kv_a_proj_with_mqa.weight"),
                    kv_k, kv_v, b16(p + "o_proj.weight"), t(p + "q_a_layernorm.weight"),
                    t(p + "kv_a_layernorm.weight"), HL, indexer(i) if full else None, absorb)

    def mlp(p: str) -> MLPW:
        gu = stack([p + "gate_proj.weight", p + "up_proj.weight"])
        return MLPW(gu, make_b16(t(p + "down_proj.weight").contiguous()), gu.n // world)

    def moe(p: str) -> MoEW:
        router = t(p + "gate.weight", torch.bfloat16).contiguous()
        bias = t(p + "gate.e_score_correction_bias", torch.float32).contiguous()
        parts: dict[str, tuple] = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            ts, us, vs = [], [], []
            for e in range(cfg.experts):
                raw = f"{p}experts.{e}.{proj}."
                ts.append(exl3_words(rd.get(name(raw + "trellis"))))
                us.append(rd.get(name(raw + "suh")))
                vs.append(rd.get(name(raw + "svh")))
            parts[proj] = (torch.stack(ts).to(dev), torch.stack(us).to(dev), torch.stack(vs).to(dev))
            del ts, us, vs
        (gt, sg, vg), (ut, su, vu), (dt, sd, vd) = parts["gate_proj"], parts["up_proj"], parts["down_proj"]
        return MoEW(router, bias,
                    Exl3Experts(gt, ut, dt, sg, su, vg, vu, sd, vd, cfg.experts, int(vg.shape[1]), int(vd.shape[1])),
                    None if not cfg.shared else mlp(p + "shared_experts."))

    def layer(i: int) -> LayerW:
        lw = LayerW(i, "dsa", t(f"layers.{i}.input_layernorm.weight"), t(f"layers.{i}.post_attention_layernorm.weight"))
        lw.dsa = dsa(i, full=cfg.indexer_types[i] == "full")
        if cfg.mlp_kinds[i] == "dense":
            lw.mlp = mlp(f"layers.{i}.mlp.")
        else:
            lw.moe = moe(f"layers.{i}.mlp.")
        return lw

    embed = t("embed_tokens.weight", torch.bfloat16).contiguous().to(dev)
    try:
        built = [layer(i) for i in range(cfg.layers)]
        vl = cfg.vocab // world
        head = make_b16(t("lm_head.weight")[rank * vl:(rank + 1) * vl].to(dev).contiguous())
        mtpw = None
        if cfg.mtp_layers:
            i = cfg.layers
            mtpw = MTPW(t(f"layers.{i}.enorm.weight"), t(f"layers.{i}.hnorm.weight"),
                        b16(f"layers.{i}.eh_proj.weight"), t(f"layers.{i}.shared_head.norm.weight"),
                        LayerW(i, "dsa", t(f"layers.{i}.input_layernorm.weight"),
                               t(f"layers.{i}.post_attention_layernorm.weight")))
            mtpw.layer.dsa = dsa(i, full=True)
            mtpw.layer.moe = moe(f"layers.{i}.mlp.")
        w = Weights(cfg, embed, built, t("norm.weight"), head, mtpw, rank, world, dev)
        w.meta.update(layers=list(range(cfg.layers)))
    finally:
        rd.close()
    torch.cuda.empty_cache()
    return w
