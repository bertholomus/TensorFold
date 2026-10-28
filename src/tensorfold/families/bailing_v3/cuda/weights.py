"""Load one rank of Ling-3.1-flash's checkpoint: KDA and MLA layers in their storage forms (EXL3
groups as ``X3``, stored matrices as ``B16``), the router and KDA state pieces plain, world-parameterized."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tensorfold.cuda.geometry import share, share_lo

from tensorfold.families.glm5_next.cuda import latent
from tensorfold.families.glm5_next.cuda.qmm import B16, make_b16, stack_b16

from tensorfold.families.glm_moe_dsa.cuda.x3 import X3, X3Pair, vocab_slice

PREFIX = "model."                     # BailingMoeV3's layout: model.layers.N..., the head at the top level
# the config's expert SwiGLU clamp (the serving stacks apply it, the reference file does not);
# "0" serves without it
CLAMP = os.environ.get("TF_LING_SWIGLU_CLAMP", "1") != "0"
# the routed experts' trellises, one allocation a projection a layer (``Slab``; TF_LING_PACK_EXPERTS=0: one an expert,
# as before). An expert's trellis at TP2 (3 bits: 983,040 bytes) is a block of torch's small pool, two to a 2 MiB
# segment whose last 128 KiB nothing that stays can use: 6.7% of the experts' bytes, 6.4 GiB a rank (an OOM at TP2).
# The kernels read each expert by its own pointer (exl3.experts.prepare), so where its bytes sit changes no output bit.
PACK = os.environ.get("TF_LING_PACK_EXPERTS", "1") != "0"
SLOT_ALIGN = 512             # torch's caching allocator starts every block on 512 bytes: each slot starts on it too
# TF_LING_EMBED_SPLIT=1: each rank holds only its vocabulary span of the embedding (``embed_span``), and forward.embed
# (glm_moe_dsa's) gathers every rank's rows of the tokens: copies, the whole table's bits. Half the 1.2 GiB table a rank
# at TP2, for one in-graph all-gather a forward and an MTP step, so off by default. Every rank must agree (the engine
# checks it at startup); a tool that loads one rank without its peers keeps it off.
EMBED_SPLIT = os.environ.get("TF_LING_EMBED_SPLIT", "0") == "1"


def embed_span(vocab: int, world: int, rank: int, split: bool | None = None) -> tuple[int, int]:
    """(first token, rows) of the embedding one rank holds: the whole table, or with the split (None: EMBED_SPLIT)
    its span of whole 128-row blocks, the EXL3 head's (vocab_slice), which glm_moe_dsa's embed_pick reads back. Ling's
    157,184 tokens at TP2: 78,592 rows a rank."""

    if not (EMBED_SPLIT if split is None else split) or world == 1:
        return 0, vocab
    lo, mine, _ = vocab_slice(-(-vocab // 128), world, rank)
    first = min(lo * 128, vocab)
    return first, min((lo + mine) * 128, vocab) - first


def embed_priced(transform, vocab: int, world: int, rank: int):
    """A startup weight transform that prices this rank's embed_span rows of the embedding, not the whole table."""

    rows = embed_span(vocab, world, rank)[1]

    def priced(name: str, info: dict):
        shape = info.get("shape") or []
        if name.endswith("word_embeddings.weight") and len(shape) == 2:
            return transform(name, {**info, "shape": [rows, shape[1]]})
        return transform(name, info)

    return priced


class Slab:
    """``count`` pieces of one size in one allocation on ``device`` (a projection's expert trellises): slot e holds the
    e-th piece put, each slot the first piece's bytes rounded up to SLOT_ALIGN. A larger piece, or one past ``count``,
    gets an allocation of its own (the per-expert way), so mixed widths stay correct, only less packed."""

    def __init__(self, count: int, device: torch.device) -> None:
        self.count, self.device = count, device
        self.buf: torch.Tensor | None = None
        self.slot = self.used = 0

    def put(self, src: torch.Tensor) -> torch.Tensor:
        """``src``'s bytes in the next slot: a contiguous tensor of its dtype and shape on the slab's device."""

        n = src.numel() * src.element_size()
        if self.buf is None:
            self.slot = -(-n // SLOT_ALIGN) * SLOT_ALIGN
            self.buf = torch.empty((self.count * self.slot,), dtype=torch.uint8, device=self.device)
        if n > self.slot or self.used == self.count:
            return src.to(self.device).contiguous()
        at = self.used * self.slot
        self.used += 1
        out = self.buf[at:at + n].view(src.dtype).view(src.shape)
        out.copy_(src)
        return out


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
    lin_heads: int               # the KDA heads (all of them: num_kv_heads_for_linear_attn is refused otherwise)
    lin_dim: int
    conv: int
    lower: float
    experts: int
    top_k: int
    n_group: int
    topk_group: int
    moe_width: int
    shared_width: int
    dense_width: int
    routed_scale: float
    norm_topk: bool
    rope_theta: float
    layer_kinds: list[str]       # per layer: "kda" or "mla"
    mlp_kinds: list[str]         # per layer: "dense" or "moe"
    eos: tuple[int, ...]
    mtp_layers: int
    quant: str = "exl3"          # always "exl3": this family's only read path
    # the SwiGLU clamp per decoder layer (config.Config.swiglu_limit; inf: none): routed experts, shared expert
    expert_limits: tuple[float, ...] = ()
    shared_limits: tuple[float, ...] = ()

    def swiglu_limit(self, layer: int, shared: bool = False) -> float:
        """A layer's expert SwiGLU clamp L, inf past the lists (the MTP layer, index ``layers``)."""

        limits = self.shared_limits if shared else self.expert_limits
        return limits[layer] if 0 <= layer < len(limits) else math.inf

    @property
    def qk_dim(self) -> int:
        return self.qk_nope + self.qk_rope

    @property
    def kv_b_per_head(self) -> int:
        return self.qk_nope + self.v_dim

    @property
    def dense_limit(self) -> int:
        """Largest context (tokens) the MLA layers attend densely: every visible key, at any length."""

        return 1 << 62

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        from tensorfold.families.bailing_v3.config import Config as Full

        raw = json.loads((Path(model_dir) / "config.json").read_text())
        full = Full.from_dict(raw)
        return cls(hidden=full.hidden_size, layers=full.num_hidden_layers, vocab=full.vocab_size,
                   eps=full.rms_norm_eps, heads=full.num_attention_heads, q_lora=full.q_lora_rank,
                   kv_lora=full.kv_lora_rank, qk_nope=full.qk_nope_head_dim, qk_rope=full.qk_rope_head_dim,
                   v_dim=full.v_head_dim, lin_heads=full.lin_heads, lin_dim=full.lin_dim,
                   conv=full.conv_kernel, lower=full.kda_lower_bound, experts=full.n_routed_experts,
                   top_k=full.num_experts_per_tok, n_group=full.n_group, topk_group=full.topk_group,
                   moe_width=full.moe_intermediate_size,
                   # the shared expert's width: moe_shared_expert_intermediate_size x num_shared_experts
                   # (modeling's BailingMoeV3SparseMoeBlock; Ling-3.1-flash: 1280, the routed width too)
                   shared_width=(full.shared_intermediate_size or full.moe_intermediate_size)
                   * full.n_shared_experts,
                   dense_width=full.intermediate_size, routed_scale=full.routed_scaling_factor,
                   norm_topk=full.norm_topk_prob, rope_theta=full.rope_theta,
                   layer_kinds=["kda" if k == "kda" else "mla" for k in full.layer_kinds],
                   mlp_kinds=["moe" if k == "sparse" else "dense" for k in full.mlp_layer_types],
                   eos=full.eos_token_id, mtp_layers=full.num_nextn_predict_layers,
                   # TF_LING_SWIGLU_CLAMP=0: no clamp anywhere, the reference file's forward, for A/B runs
                   expert_limits=full.expert_swiglu_limits if CLAMP else (),
                   shared_limits=full.shared_swiglu_limits if CLAMP else ())


@dataclass
class KDAW:
    """One KDA layer: full-rank q, k, v, decay f and gate g projections, one beta logit a head, the short conv.

    The projections are separate linears (each EXL3 group has its own Hadamard sign vectors, so they cannot share
    one call), except q | k | v when the conversion quantized them as one group (exllamav3's fused qkv_proj: one
    call writes the rank's [q | k | v] rows). Forward writes q | k | v into P's first three columns, the beta logits
    at ``b_off``, f into A and g into G. No low-rank stage: the reference's no_kda_lora makes f and g full rank.
    """

    q: B16 | X3 | None       # [LL * lin_dim, hidden] (None: ``qkv`` holds q, k and v)
    k: B16 | X3 | None
    v: B16 | X3 | None
    f: B16 | X3              # the decay input ("a" in the chain signature)
    g: B16 | X3              # the output gate
    b: B16                   # [LL, hidden] the beta logits (P's trailing columns at b_off)
    conv: torch.Tensor       # [3 * LL * lin_dim, conv] bf16: the q | k | v channels concatenated
    a_log: torch.Tensor      # [LL] fp32
    dt_bias: torch.Tensor    # [LL * lin_dim] fp32
    norm: torch.Tensor       # [lin_dim] bf16 (the gated RMSNorm weight)
    o: B16 | X3              # [hidden, LL * lin_dim]
    heads: int
    qkv: X3 | None = None    # [3 * LL * lin_dim, hidden]: the fused group, the rank's q | k | v rows in that order

    @property
    def b_off(self) -> int:
        return 3 * self.heads * 128


@dataclass
class MLAW:
    """One RoPE MLA layer: low-rank q, the latent kv_a, per-head kv_b kept whole (BF16), a sigmoid head gate."""

    q_a: B16 | X3            # [q_lora, hidden] replicated
    q_norm: torch.Tensor     # [q_lora] bf16
    q_b: B16 | X3            # [HL * qk_dim, q_lora] rows by head: qk_nope key columns then the rope slice
    kv_a: B16 | X3           # [kv_lora + qk_rope, hidden] replicated (an EXL3 group pads it to 640; X3 crops)
    kv_norm: torch.Tensor    # [kv_lora] bf16 (the latent norm, not the rope key)
    kv_k: B16                # [HL * qk_nope, kv_lora] key rows of kv_b
    kv_v: B16                # [HL * v_dim, kv_lora] value rows
    gate: B16 | X3           # [HL, hidden] the head gate (sigmoid(fp32(.)), applied after W_UV)
    dense: B16 | X3          # [hidden, HL * v_dim]
    heads: int
    absorb: Any = None       # latent.AbsorbW: kv_b split per head, for attention on the latent cache


@dataclass
class MLPW:
    gu: B16 | X3Pair         # this rank's [gate | up] rows
    down: B16 | X3           # [hidden, this rank's width]
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
    router: torch.Tensor      # [E, hidden] bf16 (fp32 math: the logits upcast at the router)
    bias: torch.Tensor        # [E] fp32
    experts: RoutedW          # the E routed experts (EXL3, any width)
    shared: MLPW | None       # the shared expert (EXL3 5)


@dataclass
class LayerW:
    index: int
    kind: str                 # "kda" or "mla"
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    kda: KDAW | None = None
    mla: MLAW | None = None
    mlp: MLPW | None = None
    moe: MoEW | None = None


@dataclass
class MTPW:
    enorm: torch.Tensor
    hnorm: torch.Tensor
    eh: B16 | X3              # [hidden, 2 * hidden]: input [enorm(embedding) | hnorm(hidden)]
    norm: torch.Tensor        # final_layernorm: its output chains into the next draft row and the head
    layer: LayerW             # the MLA block and the full MoE, plain residual


@dataclass
class Weights:
    cfg: Config
    embed: torch.Tensor       # [rows, hidden] bf16: the whole table, or this rank's embed_span (TF_LING_EMBED_SPLIT)
    layers: list[LayerW]
    norm: torch.Tensor
    head: B16 | X3            # this rank's vocabulary slice
    mtp: MTPW | None
    rank: int
    world: int
    device: torch.device
    comm: Any = None
    meta: dict = field(default_factory=dict)
    embed_lo: int = 0         # the first token of this rank's embedding rows

    @property
    def vocab_offset(self) -> int:
        lo = self.meta.get("vocab_lo")                # an EXL3 head: the rank's whole-128-column-block span start
        if lo is not None:
            return int(lo)                            # balanced uneven spans: NOT rank * per
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
            elif isinstance(t, (B16, RoutedW, KDAW, MLAW, MLPW, MoEW, LayerW, MTPW)):
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


def load(model_dir: str | Path, *, rank: int, device: str = "cuda", mtp: bool = True,
         pack: bool | None = None) -> Weights:
    """Read one of ``TF_TP_WORLD`` ranks from a full checkpoint or rank folder, the MTP layer included unless
    ``mtp`` is False (TF_LING_MTP=0: its tensors are neither read nor held). ``pack`` (None: PACK): each layer's
    routed-expert trellises in one ``Slab`` a projection rather than one allocation an expert.

    Each projection is read the way the checkpoint stores it: an EXL3 trellis group (``X.trellis`` with its
    scales and codebook marker) becomes an ``X3`` on the row-invariant EXL3 linear, a stored ``X.weight`` a
    BF16 ``B16``. The split rules (``split.rule``) hand each rank its share of either form; kv_b's rows stay
    whole (BF16) and split per head by the config's qk_nope + v_dim.
    """

    from tensorfold.families.glm5_next.cuda.split import RankReader

    from tensorfold.families.glm_moe_dsa.cuda import x3 as x3mod

    world = int(os.environ.get("TF_TP_WORLD", "2"))
    cfg = Config.read(model_dir)
    dev = torch.device(device)
    pack = PACK if pack is None else bool(pack)
    rd = RankReader(model_dir, rank, cfg_hint={"qk_dim": cfg.qk_dim, "qk_nope": cfg.qk_nope, "v_dim": cfg.v_dim})
    HL = share(cfg.heads, world, rank)                # 64 heads: 32 a rank at TP2
    LL = share(cfg.lin_heads, world, rank)            # the KDA heads (the same 64)
    x3_users: list = []

    def name(raw: str) -> str:
        if "model." + raw in rd.index:
            return "model." + raw
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

    def kda(i: int) -> KDAW:
        p = f"layers.{i}.attention."
        # the checkpoint's three short convs, or the conversion's one conv1d (exllamav3 writes q | k | v fused);
        # the split hands either form over as this rank's q | k | v channels (split.SEGMENTS)
        if has(p + "conv1d.weight"):
            conv = t(p + "conv1d.weight")
        else:
            conv = torch.cat([t(p + f"{x}_conv1d.weight") for x in "qkv"])
        conv = conv.reshape(3 * LL * cfg.lin_dim, cfg.conv).contiguous()
        if has(p + "qkv_proj.trellis"):       # one EXL3 group: the rank's [q_r | k_r | v_r] rows (split.SEGMENTS)
            qkv = x3(p + "qkv_proj.")
            if qkv.n != 3 * LL * cfg.lin_dim:
                raise ValueError(f"layer {i}: qkv_proj holds {qkv.n} rows on rank {rank}, not 3 x {LL} heads")
            q = k = v = None
        else:
            qkv, q, k, v = None, proj(p + "q_proj."), proj(p + "k_proj."), proj(p + "v_proj.")
        return KDAW(q, k, v, proj(p + "f_proj."),
                    proj(p + "g_proj."), make_b16(t(p + "b_proj.weight").reshape(LL, cfg.hidden).contiguous()),
                    conv, t(p + "A_log", torch.float32).contiguous(), t(p + "dt_bias", torch.float32).contiguous(),
                    t(p + "o_norm.weight", torch.bfloat16), proj(p + "o_proj."), LL, qkv)

    def mla(i: int) -> MLAW:
        p = f"layers.{i}.attention."
        # kv_b_proj is stored head-major [H * (qk_nope + v_dim), kv_lora]: each head's qk_nope key rows, then
        # its v_dim value rows (the reference views it [H, nope + v, kv_lora] and splits the two). The forward
        # absorbs each head's q_nope (gathered contiguous) against wk [H, qk_nope, kv_lora]: the absorb kernels
        # take a power-of-two head width (128; Ling's qk_dim 192 is not), and the rope term comes from the
        # separate q_pe . k_pe score (mla_pe).
        if has(p + "kv_b_proj.trellis"):
            raise ValueError("kv_b_proj must stay BF16: the absorb kernels read its rows per head")
        w = t(p + "kv_b_proj.weight", torch.bfloat16)
        per = w.view(HL, cfg.kv_b_per_head, cfg.kv_lora)
        k_nope = per[:, :cfg.qk_nope]                                     # [HL, qk_nope, kv_lora]
        v_rows = per[:, cfg.qk_nope:]                                     # [HL, v_dim, kv_lora]
        kv_k = make_b16(k_nope.reshape(HL * cfg.qk_nope, cfg.kv_lora).contiguous())
        kv_v = make_b16(v_rows.reshape(HL * cfg.v_dim, cfg.kv_lora).contiguous())
        return MLAW(proj(p + "q_a_proj."), t(p + "q_a_layernorm.weight", torch.bfloat16),
                    proj(p + "q_b_proj."),
                    proj(p + "kv_a_proj_with_mqa.", crop=cfg.kv_lora + cfg.qk_rope),
                    t(p + "kv_a_layernorm.weight", torch.bfloat16), kv_k, kv_v,
                    proj(p + "g_proj.", crop=HL), proj(p + "dense."), HL,
                    latent.AbsorbW(k_nope.float(), v_rows.float()))

    def mlp(p: str) -> MLPW:
        if has(p + "gate_proj.trellis"):
            gu = X3Pair(x3(p + "gate_proj."), x3(p + "up_proj."))
            return MLPW(gu, x3(p + "down_proj."), gu.gate.n)
        gu = stack([p + "gate_proj.weight", p + "up_proj.weight"])
        return MLPW(gu, make_b16(t(p + "down_proj.weight").contiguous()), gu.n // 2)     # the rank's rows, as loaded

    def moe(p: str) -> MoEW:
        from tensorfold.cuda.exl3 import experts as x3experts

        router = t(p + "gate.weight", torch.bfloat16).contiguous()
        bias = t(p + "gate.expert_bias", torch.float32).contiguous()
        # every rank holds ALL routed experts, each at its share of the width (the split rules cut gate/up
        # trellis columns and down rows: EXL3_RULES). The router picks global expert ids, so the expert set
        # must stay whole on every rank; the rank's fp32 partial is summed over ranks by the all-gather.
        mine = cfg.experts
        experts_of = range(cfg.experts)
        parts: dict[str, list] = {}
        cb = None
        D = I = 0
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            rows = []
            slab = Slab(len(experts_of), dev) if pack else None
            for e in experts_of:
                g = group(f"{p}experts.{e}.{proj_name}.")
                cb = cb or x3mod.codebook(g)
                trellis = slab.put(g["trellis"]) if slab is not None else g["trellis"].to(dev).contiguous()
                if proj_name == "gate_proj":
                    D, I = trellis.shape[0] * 16, trellis.shape[1] * 16
                rows.append((trellis, g.get("suh", g.get("su")).to(dev, torch.float16),
                             g.get("svh", g.get("sv")).to(dev, torch.float16)))
            parts[proj_name] = rows
        if not mine:
            # a rank with no routed experts (only possible below the real scale): a zero-width layer
            return MoEW(router, bias, RoutedW(0, D or cfg.hidden, I or cfg.moe_width, cb or "mul1", None, parts),
                        None if not cfg.shared_width else mlp(p + "shared_experts."))
        ex = None
        if dev.type == "cuda":
            ex = x3experts.prepare(parts["gate_proj"], parts["up_proj"], parts["down_proj"], cb, device=dev)
        routed = RoutedW(mine, D, I, cb, ex, None if ex is not None else parts)
        return MoEW(router, bias, routed, None if not cfg.shared_width else mlp(p + "shared_experts."))

    def layer(i: int) -> LayerW:
        lw = LayerW(i, cfg.layer_kinds[i], t(f"layers.{i}.input_layernorm.weight", torch.bfloat16),
                    t(f"layers.{i}.post_attention_layernorm.weight", torch.bfloat16))
        if cfg.layer_kinds[i] == "kda":
            lw.kda = kda(i)
        else:
            lw.mla = mla(i)
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
            meta["vocab_lo"] = lo * 128                             # this rank's first token id (uneven spans)
            return head
        vl = share(cfg.vocab, world, rank)
        lo = share_lo(cfg.vocab, world, rank)
        return make_b16(t("lm_head.weight")[lo:lo + vl].to(dev).contiguous())

    # the whole table on every rank, or with TF_LING_EMBED_SPLIT this rank's vocabulary span only (forward.embed
    # gathers the other ranks' rows)
    e0, en = embed_span(cfg.vocab, world, rank)
    embed = rd.get(name("word_embeddings.weight"))[e0:e0 + en].to(device=dev, dtype=torch.bfloat16, copy=True)
    try:
        built = [layer(i) for i in range(cfg.layers)]
        meta: dict = {}
        head = lm_head(meta)
        mtpw = None
        if mtp and cfg.mtp_layers and has(f"layers.{cfg.layers}.enorm.weight"):
            i = cfg.layers
            mtpw = MTPW(t(f"layers.{i}.enorm.weight", torch.bfloat16), t(f"layers.{i}.hnorm.weight", torch.bfloat16),
                        proj(f"layers.{i}.eh_proj."), t(f"layers.{i}.final_layernorm.weight", torch.bfloat16),
                        LayerW(i, "mla", t(f"layers.{i}.input_layernorm.weight", torch.bfloat16),
                               t(f"layers.{i}.post_attention_layernorm.weight", torch.bfloat16)))
            mtpw.layer.mla = mla(i)
            mtpw.layer.moe = moe(f"layers.{i}.mlp.")
        w = Weights(cfg, embed, built, t("norm.weight", torch.bfloat16), head, mtpw, rank, world, dev, embed_lo=e0)
        w.meta.update(layers=list(range(cfg.layers)), x3=x3_users, **meta)
    finally:
        rd.close()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return w
