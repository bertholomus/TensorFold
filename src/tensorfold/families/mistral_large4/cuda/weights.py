"""Per-rank weights of Mistral-Large-4 read straight from the EXL3 pack (note D3 §1-§4; layout D2 §1-§2).

The pack (``quant/pack_ml4.py``, PACK-LAYOUT.md v1) carries HF tensor names; this loader reads them verbatim — no
on-disk rename, no index.json lookup (headers are authority, like ``Shards``). One 4-rank lane: routed experts are
**EP4** (whole experts, rank r owns ``local_ids(r)`` = its e//32 block, D2 §2) and everything else — attention
(q/o, kv_a+norm, kv_b, ssmax), indexer, shared expert, dense L0, router, norms, embedding, LeRoPE alpha, head — is
**replicated** on every rank. No requantization and no column split: every ``-nq`` tensor keeps its source
precision, and the only casts are widenings (D3 §2): norms / router / ssmax / ``idx.proj`` → F32, LeRoPE alpha →
F32, kv_a / kv_b / wk / embed keep BF16, and the indexer's fp8 ``wq_b`` dequantizes to fp16 with its 128×128 fp32
scale grid (``dequant_fp8_block``) — source bytes untouched.

Integrity (D3 §4) is loud and early: ``verify_manifest`` asserts the manifest fields and the partial-pack head
combination (§6), the per-layer header preflight checks every expected tensor's presence + dtype + shape and every
forbidden one's absence (readers carry no ``kv_a*``, shared-indexer layers no ``indexer.*``), the EXL3 group
headers must parse with the mul1 marker at ``0x83DCD12D``, and ``load`` gates on the per-rank expert-id set, on
bytes read == plan, and on ``torch.cuda.memory_allocated()`` vs the per-rank plan.

Phase 1 scope: these tensors arrive on the GPU at the right precision; the engine that consumes them is phase 2
(families/mistral_large4/__init__.py refuses ``cuda_engine`` until then). TODO(phase-2, D1 §Extension points):
attention/kernels read these slots; TODO(D2 §1): cli/engine 4-rank launch — ``cli.py`` and ``cli_args.py`` still
carry 2-rank-only semantics and deepseek_v41's ``cuda_engine`` still refuses ``tp not in (1, 2)``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import Cfg

# -- pack constants (mirror quant/pack_ml4.py and cuda/exl3/format.py; kept local so CPU tests import without
#    numpy — a test pins these against the upstream values) --------------------------------------------------

PACK_VERSION = 1
LM = "model.language_model"
LP = LM + ".layers.{L}."                                  # layer-name prefix
MARKER_VALUE = {"mul1": 0x83DCD12D, "mcg": 0xCBAC1FED}    # == cuda/exl3/format.py MARKERS (pinned by a test)
#: the pack v1 recipe (D3 §4.1): one recipe ships, asserted not branched on
EXPECTED_BITS = {"experts": 2.5, "shared": 5.0, "attn": 5.0, "dense": 5.0}
EXPECTED_CALIBRATION = {"rows": 128, "cols": 2048}
#: safetensors header dtype strings -> item bytes (same names as Shards.DT)
DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "U8": 1}


# -- name shim (D3 §1): every pack tensor name template -> how the loader keeps it -----------------------------
#
# "how": "exl3" = the group goes to Exl3Linear / exl3_experts as stored; the others name the widening cast (or
# kept dtype) for a -nq tensor. "fp8deq" / "fp8scale" are the indexer wq_b pair, dequantized together.

LAYER_SHIM: dict[str, tuple[str, str]] = {
    "self_attn.q_proj": ("exl3", "attn.q"),
    "self_attn.o_proj": ("exl3", "attn.o"),
    "mlp.gate_proj": ("exl3", "dense.gate"),
    "mlp.up_proj": ("exl3", "dense.up"),
    "mlp.down_proj": ("exl3", "dense.down"),
    "mlp.shared_experts.gate_proj": ("exl3", "experts.shared.gate"),
    "mlp.shared_experts.up_proj": ("exl3", "experts.shared.up"),
    "mlp.shared_experts.down_proj": ("exl3", "experts.shared.down"),
    "mlp.experts.{e}.gate_proj": ("exl3", "experts.{e}.gate"),
    "mlp.experts.{e}.up_proj": ("exl3", "experts.{e}.up"),
    "mlp.experts.{e}.down_proj": ("exl3", "experts.{e}.down"),
    "input_layernorm.weight": ("f32", "attn_norm"),
    "post_attention_layernorm.weight": ("f32", "ffn_norm"),
    "self_attn.kv_a_proj_with_mqa.weight": ("bf16", "kv_a"),
    "self_attn.kv_a_layernorm.weight": ("f32", "kv_a_norm"),
    "self_attn.kv_b_proj.weight": ("bf16", "kv_b"),
    "self_attn.softmax_scaling.scale": ("f32", "ssmax_s"),
    "self_attn.softmax_scaling.bias": ("f32", "ssmax_b"),
    "self_attn.indexer.wq_b.weight": ("fp8deq", "idx.wq_b"),
    "self_attn.indexer.wq_b.weight_scale_inv": ("fp8scale", "idx.wq_b_scale"),
    "self_attn.indexer.wk.weight": ("bf16", "idx.wk"),
    "self_attn.indexer.weights_proj.weight": ("f32", "idx.proj"),
    "self_attn.indexer.k_norm.weight": ("f32", "idx.k_norm"),
    "self_attn.indexer.k_norm.bias": ("f32", "idx.k_norm_b"),
    "self_attn.indexer.q_input_norm.weight": ("f32", "idx.q_input_norm"),
    "self_attn.indexer.q_input_norm.bias": ("f32", "idx.q_input_norm_b"),
    "self_attn.indexer.softmax_scaling.scale": ("f32", "idx.ssmax_s"),
    "self_attn.indexer.softmax_scaling.bias": ("f32", "idx.ssmax_b"),
    "mlp.gate.weight": ("f32", "gate_w"),
}

GLOBALS_SHIM: dict[str, str] = {                             # pack name -> slot
    f"{LM}.embed_tokens.weight": "embed",                    # BF16 kept (replicated)
    f"{LM}.rotary_emb.alpha": "lerope_alpha",                # -> F32
    f"{LM}.norm.weight": "norm",                             # -> F32
    "lm_head.weight": "head_bf16",                           # only when the pack head is the source head
}

#: cast + kept-bytes for each -nq "how": (target torch dtype or None for source, cache bytes per source byte)
_NQ_CAST = {"f32": torch.float32, "bf16": None, "fp8deq": torch.float16, "fp8scale": None}


# -- EP4 expert split (D2 §2; the default is the whole-numbering rule e//32) ------------------------------------

def expert_rank(e: int, world: int = 4, n_experts: int = 128, rule: str = "block") -> int:
    """The rank that holds expert e. "block" (default): e // (E // world) — D2 §2's e//32; "interleave": e % world
    (kept as an interface only — Phase B may A/B it, D2 §9; Phase 1 implements no routing either way)."""

    if n_experts % world:
        raise ValueError(f"EP4 needs whole experts a rank: {n_experts} experts do not divide into {world} ranks")
    if rule == "block":
        return e // (n_experts // world)
    if rule == "interleave":
        return e % world
    raise ValueError(f"unknown expert split rule {rule!r} (block, interleave)")


def local_ids(rank: int, world: int = 4, n_experts: int = 128, rule: str = "block") -> list[int]:
    """The global expert ids rank r keeps, in pack order (D3 §2: exactly E/world of them, shared not among them)."""

    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} outside world {world}")
    return [e for e in range(n_experts) if expert_rank(e, world, n_experts, rule) == rank]


# -- fp8 block dequant (D3 §5.3) --------------------------------------------------------------------------------

def dequant_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor, block: int = 128) -> torch.Tensor:
    """FP8 e4m3 ``weight`` [N, K] x fp32 scale grid ``scale_inv`` [ceil(N/block), ceil(K/block)] -> fp16 [N, K].

    The same arithmetic as ``cuda/nvfp4/format.py`` ``dequant("fp8block")`` (e4m3 → f32, multiply by the scale of
    its block), rounded to fp16 here (D3 §2 keeps the dequantized wq_b as fp16). Source bytes untouched; CPU or
    GPU, ~15 lines, only at load."""

    if weight.ndim != 2:
        raise ValueError(f"wq_b must be [N, K], got shape {tuple(weight.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"wq_b must be F8_E4M3, got {weight.dtype}")
    if scale_inv.dtype != torch.float32:
        raise ValueError(f"wq_b scale must be F32, got {scale_inv.dtype}")
    n, k = weight.shape
    want = (math.ceil(n / block), math.ceil(k / block))
    if tuple(scale_inv.shape) != want:
        raise ValueError(f"wq_b scale grid must be exactly {block}x{block} blocks: expected {want}, got "
                         f"{tuple(scale_inv.shape)} (D3 §4)")
    scales = scale_inv.repeat_interleave(block, 0).repeat_interleave(block, 1)[:n, :k]
    return (weight.to(torch.float32) * scales).to(torch.float16)


# -- the expected tensor sets (D3 §2: what a layer holds is decided only by the config lists; §4: shape/dtype) ----

def _group(name: str, k: int, n: int, bits: float, codebook: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """One EXL3 group's expected header entries at width ``bits``: trellis I16 [K/16, N/16, 16*bits], scales
    F16 [K] / [N], and the codebook marker when it has one (half widths always; pack_ml4 writes the mul1 marker
    for every mul1 group)."""

    w = int(16 * bits)
    out = {f"{name}.trellis": ("I16", (k // 16, n // 16, w)), f"{name}.suh": ("F16", (k,)),
           f"{name}.svh": ("F16", (n,))}
    if codebook in MARKER_VALUE:
        out[f"{name}.{codebook}"] = ("I32", ())
    return out


def expected_layer_quant(cfg: Cfg, bits: dict[str, float], codebook: str, i: int) -> dict[str, tuple[str, tuple]]:
    """Every tensor ``L<i>.safetensors`` must hold — and the set it must not exceed (assert absence by set
    equality). Trellis shapes are derived from the config dims + the manifest bits, which reproduces D3 §4's
    exact numbers on the real dims ([384, 512, 80] q_proj, [512, 384, 80] o_proj, [384, 384, 80] shared,
    [384, 384, 40] routed, [384, 1536, 80] / [1536, 384, 80] dense L0)."""

    d, out = cfg.dim, {}
    out.update(_group(f"{LP.format(L=i)}self_attn.q_proj", d, cfg.q_out, bits["attn"], codebook))
    out.update(_group(f"{LP.format(L=i)}self_attn.o_proj", cfg.o_in, d, bits["attn"], codebook))
    if cfg.is_dense(i):
        for t, (k, n) in (("gate", (d, cfg.dense_inter)), ("up", (d, cfg.dense_inter)), ("down", (cfg.dense_inter, d))):
            out.update(_group(f"{LP.format(L=i)}mlp.{t}_proj", k, n, bits["dense"], codebook))
        return out
    for t, (k, n) in (("gate", (d, cfg.inter)), ("up", (d, cfg.inter)), ("down", (cfg.inter, d))):
        out.update(_group(f"{LP.format(L=i)}mlp.shared_experts.{t}_proj", k, n, bits["shared"], codebook))
    for e in range(cfg.n_routed):
        for t, (k, n) in (("gate", (d, cfg.inter)), ("up", (d, cfg.inter)), ("down", (cfg.inter, d))):
            out.update(_group(f"{LP.format(L=i)}mlp.experts.{e}.{t}_proj", k, n, bits["experts"], codebook))
    return out


def expected_layer_nq(cfg: Cfg, i: int) -> dict[str, tuple[str, tuple]]:
    """Every tensor ``L<i>-nq.safetensors`` must hold, at source precision (BF16 everywhere but the indexer's
    fp8 wq_b + its fp32 scale grid). Readers carry no kv_a*; shared-indexer layers carry no ``indexer.*``; the
    dense layer carries no router (D3 §2)."""

    d = cfg.dim
    out = {
        f"{LP.format(L=i)}input_layernorm.weight": ("BF16", (d,)),
        f"{LP.format(L=i)}post_attention_layernorm.weight": ("BF16", (d,)),
        f"{LP.format(L=i)}self_attn.kv_b_proj.weight": ("BF16", (cfg.kv_b_rows(i), cfg.kv_lora)),
        f"{LP.format(L=i)}self_attn.softmax_scaling.scale": ("BF16", (cfg.n_heads,)),
        f"{LP.format(L=i)}self_attn.softmax_scaling.bias": ("BF16", (cfg.n_heads,)),
    }
    if not cfg.is_reader(i):
        out[f"{LP.format(L=i)}self_attn.kv_a_proj_with_mqa.weight"] = ("BF16", (cfg.kv_a_rows(i), d))
        out[f"{LP.format(L=i)}self_attn.kv_a_layernorm.weight"] = ("BF16", (cfg.kv_lora,))
    if cfg.is_full_indexer(i):
        p = f"{LP.format(L=i)}self_attn.indexer."
        out[p + "wq_b.weight"] = ("F8_E4M3", (cfg.idx_heads * cfg.idx_dim, d))
        out[p + "wq_b.weight_scale_inv"] = ("F32", (math.ceil(cfg.idx_heads * cfg.idx_dim / 128), math.ceil(d / 128)))
        out[p + "wk.weight"] = ("BF16", (cfg.idx_dim, d))
        out[p + "weights_proj.weight"] = ("BF16", (cfg.idx_heads, d))
        out[p + "k_norm.weight"] = ("BF16", (cfg.idx_dim,))
        out[p + "k_norm.bias"] = ("BF16", (cfg.idx_dim,))
        out[p + "q_input_norm.weight"] = ("BF16", (d,))
        out[p + "q_input_norm.bias"] = ("BF16", (d,))
        out[p + "softmax_scaling.scale"] = ("BF16", (cfg.idx_heads,))
        out[p + "softmax_scaling.bias"] = ("BF16", (cfg.idx_heads,))
    if not cfg.is_dense(i):
        out[f"{LP.format(L=i)}mlp.gate.weight"] = ("BF16", (cfg.n_routed, d))
    return out


def expected_globals(cfg: Cfg, head_mode: str) -> dict[str, tuple[str, tuple]]:
    """``globals.safetensors``: embedding, the one LeRoPE alpha table, the final norm (+ the bf16 head exactly
    when the pack head mode is "source")."""

    out = {f"{LM}.embed_tokens.weight": ("BF16", (cfg.vocab, cfg.dim)),
           f"{LM}.rotary_emb.alpha": ("BF16", (cfg.rd // 2,)),
           f"{LM}.norm.weight": ("BF16", (cfg.dim,))}
    if head_mode == "source":
        out["lm_head.weight"] = ("BF16", (cfg.vocab, cfg.dim))
    return out


def expected_head(cfg: Cfg, head_bits: float, codebook: str) -> dict[str, tuple[str, tuple]]:
    """``head.safetensors``: the EXL3 lm_head group, full vocabulary on every rank (replicated, D3 §2)."""

    return _group("lm_head", cfg.dim, cfg.vocab, head_bits, codebook)


# -- header index (headers are authority; stdlib only, so CPU tests read real files without numpy) ---------------

class PackIndex:
    """Every pack tensor's header entry by name: (file, dtype, shape, data bytes, offset). Built from
    ``*.safetensors`` headers (or fabricated in tests); no tensor payloads are touched."""

    def __init__(self, model_dir: str | Path | None = None):
        self.root = Path(model_dir) if model_dir is not None else None
        self.entries: dict[str, tuple[str, str, tuple, int, int]] = {}
        self.file_at: dict[str, int] = {}                    # file -> header length + 8 (where its data starts)
        self.file_sizes: dict[str, int] = {}

    def load(self) -> "PackIndex":
        assert self.root is not None
        for path in sorted(self.root.glob("*.safetensors")):
            header, base = read_header(path)
            self.file_at[path.name], self.file_sizes[path.name] = base, path.stat().st_size
            for name, e in header.items():
                if name in self.entries:
                    raise ValueError(f"{name}: defined twice ({self.entries[name][0]}, {path.name}) — a pack "
                                     f"defines every tensor once (PACK-LAYOUT §Invariants)")
                lo, hi = e["data_offsets"]
                self.entries[name] = (path.name, e["dtype"], tuple(e["shape"]), hi - lo, lo)
        return self

    def add(self, name: str, file: str, dtype: str, shape: tuple, nbytes: int = 0, off: int = 0) -> None:
        self.entries[name] = (file, dtype, tuple(shape), nbytes, off)

    def names_in(self, file: str) -> dict[str, tuple[str, str, tuple, int, int]]:
        return {n: e for n, e in self.entries.items() if e[0] == file}

    def marker_value(self, name: str) -> int:
        """The int32 marker scalar stored under ``name`` (reads 4 bytes of the pack, read-only)."""

        file, dtype, shape, nbytes, off = self.entries[name]
        if dtype != "I32" or shape or nbytes != 4:
            raise ValueError(f"{name}: not an I32 scalar marker ({dtype}, {shape}, {nbytes} bytes)")
        with open(self.root / file, "rb") as f:
            f.seek(self.file_at[file] + off)
            raw = f.read(4)
        if len(raw) != 4:
            raise IOError(f"{name}: short read of the marker")
        return struct.unpack("<i", raw)[0]


def read_header(path: str | Path) -> tuple[dict, int]:
    """A safetensors file's header (without ``__metadata__``) and where its data starts (8 + header length)."""

    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


# -- integrity gates (D3 §4) ------------------------------------------------------------------------------------

def verify_manifest(manifest: dict, config: dict, index: PackIndex | None = None) -> None:
    """Gate 1 (D3 §4.1 + §6): the manifest fields against the config, and the file set when an index is given.
    The source paths are NOT checked (they name the machine that built the pack)."""

    v = manifest.get("pack_version")
    assert v == PACK_VERSION, f"ml4_pack.json: pack_version {v!r}, this loader reads {PACK_VERSION}"
    assert manifest.get("codebook") in ("mul1", "mcg", "3inst"), f"ml4_pack.json: codebook {manifest.get('codebook')!r}"
    assert manifest.get("bits") == EXPECTED_BITS, \
        f"ml4_pack.json: bits {manifest.get('bits')!r}, the pack v1 recipe is {EXPECTED_BITS}"
    assert manifest.get("calibration") == EXPECTED_CALIBRATION, \
        f"ml4_pack.json: calibration {manifest.get('calibration')!r}, expected {EXPECTED_CALIBRATION}"
    assert manifest.get("tensors") == sum(f.get("tensors", 0) for f in manifest["files"].values()), \
        "ml4_pack.json: tensors != the summed per-file tensor counts"

    t = config.get("text_config", config)
    assert t.get("vocab_size") == 204800, f"config: vocab {t.get('vocab_size')}, the ML4 pack vocab is 204800"
    lo, hi = (manifest.get("layers") or [None, None])[:2]
    n = t.get("num_hidden_layers")
    assert lo == 0 and n == hi - lo + 1, \
        f"config num_hidden_layers {n} != manifest layers {manifest.get('layers')} (a pack covers 0..k-1)"
    assert n <= manifest.get("num_hidden_layers_source", 0), \
        "config num_hidden_layers above the source model's count"
    qcfg = config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config") or {}
    assert qcfg.get("ml4_bits") == manifest.get("bits"), \
        f"config quantization_config.ml4_bits {qcfg.get('ml4_bits')!r} != manifest bits"

    partial, head = manifest.get("partial"), manifest.get("head")
    full_pack = partial is False and n == manifest.get("num_hidden_layers_source")
    assert isinstance(partial, bool), f"ml4_pack.json: partial {partial!r} is not a bool"
    if partial:
        # A truncated pack (pack_ml4.py --layers 0-K) covers [0, K] with K == the truncated config's
        # num_hidden_layers - 1 and K >= 0; the lo == 0 / n == hi + 1 assert above forces the rest.
        # The head is ORTHOGONAL to the extent: a partial pack of the real quant carries head "quant".
        assert hi >= 0, f"partial pack: layers {manifest.get('layers')}, must cover [0, K] with K >= 0"
    # the §6 head pairing, asserted BOTH ways, for full and partial packs alike (the recipe constants
    # above describe the quant, the head mode describes the head — node smoke 2026-11-06):
    if head == "quant":
        assert qcfg.get("head_bits") == 6, f"head 'quant': head_bits must be 6, got {qcfg.get('head_bits')!r}"
    elif head == "source":
        assert qcfg.get("head_bits") == 16, \
            f"head 'source': the bf16 source head needs head_bits 16, got {qcfg.get('head_bits')!r}"
    elif head == "none":
        assert t.get("tie_word_embeddings") is True, "--head none needs tie_word_embeddings (loader ties)"
    else:
        raise AssertionError(f"pack head mode must be 'quant', 'source' or 'none', got {head!r}")
    if full_pack and manifest.get("num_hidden_layers_source") == 72:
        assert [lo, hi] == [0, 71], f"full pack of the 72-layer ML4 must cover layers [0, 71], got " \
                                    f"{manifest.get('layers')}"

    if index is None:
        return
    files = manifest["files"]
    for fn, entry in files.items():
        assert isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64, f"{fn}: no sha256 field"
    have = set(index.file_sizes)
    want = {fn for fn, e in files.items() if fn.endswith(".safetensors")}
    for fn in sorted(want - have):
        raise AssertionError(f"ml4_pack.json names {fn}, the pack folder does not hold it")
    for fn in sorted(have - want):
        raise AssertionError(f"{fn} is in the pack folder but not in ml4_pack.json")
    for fn in sorted(want):
        assert files[fn]["bytes"] == index.file_sizes[fn], \
            f"{fn}: manifest bytes {files[fn]['bytes']} != on-disk size {index.file_sizes[fn]}"
    kinds = {fn: files[fn]["kind"] for fn in want}
    for i in range(lo, hi + 1):
        assert kinds.get(f"L{i:02d}.safetensors") == "exl3-layer" and kinds.get(f"L{i:02d}-nq.safetensors") \
            == "nq-layer", f"layer {i}: the pack must carry L{i:02d}.safetensors + L{i:02d}-nq.safetensors"
    assert kinds.get("globals.safetensors") == "globals", "the pack must carry globals.safetensors"
    assert (("head.safetensors" in kinds) == (head == "quant")), \
        f"head mode {head!r}: head.safetensors present == {head == 'quant'} required (D3 §6, both ways)"
    tensor_bytes = sum(e[3] for e in index.entries.values())
    assert manifest.get("tensor_bytes") == tensor_bytes, \
        f"manifest tensor_bytes {manifest.get('tensor_bytes')} != summed tensor bytes {tensor_bytes}"
    assert manifest.get("tensors") == len(index.entries), \
        f"manifest tensors {manifest.get('tensors')} != tensors in the pack files {len(index.entries)}"


def check_layer_headers(index: PackIndex, cfg: Cfg, bits: dict[str, float], codebook: str, i: int) -> None:
    """Gate 3 (D3 §4.3), header-level, before any allocation: the layer's two files hold exactly the expected
    sets — missing, unexpected, wrong dtype or shape all abort, naming the tensor and the file (readers carry no
    kv_a*, shared-indexer layers no indexer.*: absence is asserted by the set equality too)."""

    for file, expected in ((f"L{i:02d}.safetensors", expected_layer_quant(cfg, bits, codebook, i)),
                           (f"L{i:02d}-nq.safetensors", expected_layer_nq(cfg, i))):
        have = index.names_in(file)
        missing = sorted(set(expected) - set(have))
        extra = sorted(set(have) - set(expected))
        if missing:
            raise ValueError(f"{file}: missing tensors {missing[:6]}{'...' if len(missing) > 6 else ''} "
                             f"(expected from the config lists, D3 §2)")
        if extra:
            raise ValueError(f"{file}: unexpected tensors {extra[:6]}{'...' if len(extra) > 6 else ''} (a reader "
                             f"carries no kv_a*, a shared-indexer layer no indexer.*, a dense layer no shared "
                             f"experts or router)")
        for name, (dt, shape) in expected.items():
            got_dt, got_shape = have[name][1], have[name][2]
            if got_dt != dt or tuple(got_shape) != tuple(shape):
                raise ValueError(f"{name} in {file}: header says {got_dt} {tuple(got_shape)}, expected {dt} "
                                 f"{tuple(shape)} (D3 §4)")


def check_exl3_markers(index: PackIndex, codebook: str) -> None:
    """Gate 3's second half: every group's codebook marker carries the format's value — the mul1 marker is
    0x83DCD12D (D3 §4; the int32 bit pattern cuda/exl3/format.py MARKERS pins)."""

    if codebook not in MARKER_VALUE:
        return                                            # 3inst groups carry no marker tensor
    suffix = "." + codebook
    for name, e in sorted(index.entries.items()):
        if not name.endswith(suffix):
            continue
        got = index.marker_value(name)
        want = struct.unpack("<i", struct.pack("<I", MARKER_VALUE[codebook]))[0]
        if got != want:
            raise ValueError(f"{name}: marker value {got:#010x}, expected {MARKER_VALUE[codebook]:#010x}")


def check_globals_head(index: PackIndex, cfg: Cfg, head_mode: str, head_bits: float, codebook: str) -> None:
    """Gate 3 for the two model-wide files: ``globals.safetensors`` and ``head.safetensors`` hold exactly the
    expected sets (the bf16 head tensor exactly when the head mode is "source", the group exactly when "quant")."""

    for file, expected in ((f"globals.safetensors", expected_globals(cfg, head_mode)),
                           (f"head.safetensors", expected_head(cfg, head_bits, codebook) if head_mode == "quant"
                            else {})):
        have = index.names_in(file)
        missing, extra = sorted(set(expected) - set(have)), sorted(set(have) - set(expected))
        if missing or extra:
            raise ValueError(f"{file}: missing {missing[:6]}, unexpected {extra[:6]} (head mode {head_mode!r}: "
                             f"D3 §6 ties the bf16 head tensor and head.safetensors to the mode, both ways)")
        for name, (dt, shape) in expected.items():
            if have[name][1] != dt or tuple(have[name][2]) != tuple(shape):
                raise ValueError(f"{name} in {file}: header says {have[name][1]} {tuple(have[name][2])}, "
                                 f"expected {dt} {tuple(shape)} (D3 §4)")


def verify_pack(model_dir: str | Path, log=print) -> dict:
    """Gate 2 (D3 §4.2): one streaming pass rehashing every pack file against ``sha256.txt``, and the file set
    against ``ml4_pack.json``. Once per node before the first serve; it also proves the layer files were not
    truncated in transit. Returns {files, bytes, seconds}."""

    import time

    t0 = time.time()
    root = Path(model_dir)
    manifest = json.loads((root / "ml4_pack.json").read_text())
    verify_manifest(manifest, json.loads((root / "config.json").read_text()))
    lines = (root / "sha256.txt").read_text().splitlines()
    listed = {}
    for line in lines:
        digest, _, name = line.partition("  ")
        assert digest and name, f"sha256.txt: unreadable line {line!r}"
        listed[name] = digest
    named = {fn for fn, e in manifest["files"].items()}
    # pack_ml4.py hashes EVERY file of the pack dir into sha256.txt (its docstring: config.json and
    # model.safetensors.index.json included); the manifest's files[] names only the pack payloads
    # (safetensors + aux). So: everything the manifest names must be hashed, only those two static
    # loader inputs may be hashed extra, and no .safetensors may sit in the folder unnamed (real-pack
    # finding, node smoke 2026-11-06; the old set(sha256.txt) == files[] equality refused every real pack).
    extra = set(listed) - named
    assert extra <= {"config.json", "model.safetensors.index.json"}, \
        f"sha256.txt lists files ml4_pack.json does not name: {sorted(extra - {'config.json', 'model.safetensors.index.json'})[:5]}"
    missing = sorted(named - set(listed))
    assert not missing, f"ml4_pack.json names {missing[:5]}, sha256.txt does not hash them"
    stray = {p.name for p in root.iterdir() if p.is_file() and p.suffix == ".safetensors"} - named
    assert not stray, f"{sorted(stray)[:5]}: .safetensors in the pack folder not named in ml4_pack.json"
    total = 0
    for name in sorted(listed):
        p = root / name
        assert p.is_file(), f"{name}: named in sha256.txt, missing from the pack"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                h.update(b)
        if h.hexdigest() != listed[name]:
            raise ValueError(f"{name}: sha256 mismatch (refusing to load a pack that changed)")
        total += p.stat().st_size
        log(f"[ml4-verify] {name}: ok")
    report = {"files": len(listed), "bytes": total, "seconds": time.time() - t0}
    log(f"[ml4-verify] {report['files']} files, {total / 2**30:.2f} GiB, {report['seconds']:.0f} s: ok")
    return report


# -- the per-rank plan (D3 §2/§3): what one rank reads and keeps, exactly ----------------------------------------

def split_bytes(suffix: str, nbytes: int, world: int) -> tuple[int, int]:
    """(read, kept) bytes of one expert tensor when a rank keeps a quarter of every expert: gate and up their columns (the
    whole trellis is read for that), down its rows, the sign vectors whole on disk, cut on the side that is split."""

    proj, part = suffix.split(".")[-2:]
    down = proj == "down_proj"
    if part == "trellis":
        return (nbytes // world if down else nbytes), nbytes // world
    if part == "suh":
        return nbytes, nbytes // world if down else nbytes
    if part == "svh":
        return nbytes, nbytes if down else nbytes // world
    return nbytes, nbytes


def rank_plan(cfg: Cfg, codebook: str, index: PackIndex, rank: int, world: int, head_mode: str,
              expert_rule: str = "block", split: bool = False) -> dict:
    """The rank's read bytes and rank-cache (kept-bytes) plan: routed/world + every replicated byte — D3 §5.4's
    sizing input, computed from the pack headers, the config lists and the EP4 ``local_ids`` split. One pass
    classifies every name the index holds (a second completeness gate: a tensor no rule classifies aborts);
    another rank's experts are simply not this rank's bytes. ``split``: a quarter of every expert instead (``split_bytes``)."""

    ids = set(range(cfg.n_routed)) if split else set(local_ids(rank, world, cfg.n_routed, expert_rule))
    lp = f"{LM}.layers."
    marker = f".{codebook}" if codebook in MARKER_VALUE else None
    read = cache = routed_read = repl_read = 0
    for name, e in sorted(index.entries.items()):
        nbytes, rbytes, cbytes = e[3], e[3], e[3]
        if marker is not None and name.endswith(marker):
            rbytes = cbytes = 0                        # the codebook marker is membership-checked, never read
        if name.startswith(lp):
            i_str, _, suffix = name[len(lp):].partition(".")
            int(i_str)                                 # a layer tensor: the index must be a layer number
            if suffix.startswith("mlp.experts."):
                if int(suffix[len("mlp.experts."):].split(".", 1)[0]) not in ids:
                    continue                           # an expert this rank does not hold (EP4)
                if split and rbytes:
                    rbytes, cbytes = split_bytes(suffix, nbytes, world)
                routed_read += rbytes
            elif split and suffix.startswith("mlp.shared_experts.") and rbytes:
                rbytes, cbytes = split_bytes(suffix, nbytes, world)
                repl_read += rbytes
            elif suffix in LAYER_SHIM:
                cbytes = _nq_cache_bytes(suffix, e)
                repl_read += rbytes
            elif suffix.rsplit(".", 1)[0] in LAYER_SHIM and LAYER_SHIM[suffix.rsplit(".", 1)[0]][0] == "exl3":
                repl_read += rbytes                    # a replicated group's part (q/o, dense, shared expert)
            else:
                raise ValueError(f"layer tensor outside the name shim: {name} (D3 §1)")
        elif name in GLOBALS_SHIM:
            if GLOBALS_SHIM[name] in ("lerope_alpha", "norm"):
                cbytes = nbytes * (4 // DT_BYTES[e[1]])    # widened to F32 at keep
            repl_read += rbytes
        elif name == "lm_head.weight" or name.startswith("lm_head."):
            assert name == "lm_head.weight" or head_mode == "quant", \
                f"{name}: the head is mode {head_mode!r}, the pack must not carry a quant head"
            assert name != "lm_head.weight" or head_mode != "quant", \
                "a quant-head pack carries the bf16 head nowhere (it lives in head.safetensors only)"
            repl_read += rbytes
        else:
            raise ValueError(f"pack tensor outside the rank plan: {name}")
        read += rbytes
        cache += cbytes
    return {"rank": rank, "world": world, "expert_ids": sorted(ids), "routed_bytes": routed_read,
            "replicated_bytes": repl_read, "read_bytes": read, "cache_bytes": cache}


def _nq_cache_bytes(suffix: str, e: tuple) -> int:
    """Kept (rank-cache) bytes of one -nq tensor: widened to F32 doubles BF16, the fp8 wq_b doubles to fp16,
    its scale grid is consumed by the dequant and never kept."""

    how = LAYER_SHIM[suffix][0]
    if how == "fp8scale":
        return 0
    if how == "f32":
        return e[3] * (4 // DT_BYTES[e[1]])
    if how == "fp8deq":
        return e[3] * 2
    return e[3]


# -- the weight holder (phase 2's engine reads these slots) ------------------------------------------------------

@dataclass
class Ml4Layer:
    """One layer's kept tensors on this rank. None marks the slot its config list forbids (D3 §2)."""

    idx: int
    rope: bool
    kv_source: int                  # own layer id when the layer owns its cache, else its source's id
    indexer: str                    # "full" | "shared"
    dense: bool
    attn_norm: torch.Tensor         # f32 [d]
    ffn_norm: torch.Tensor          # f32 [d]
    q: object                       # Exl3Linear [d -> H*qk_head] (replicated whole)
    o: object                       # Exl3Linear [H*v_head -> d]
    kv_a: torch.Tensor | None       # bf16 [576|512, d] — None on readers
    kv_a_norm: torch.Tensor | None  # f32 [kv_lora] — None on readers
    kv_b: torch.Tensor              # bf16 [rows(i), kv_lora]
    ssmax_s: torch.Tensor           # f32 [H]
    ssmax_b: torch.Tensor           # f32 [H]
    idx_wq_b: torch.Tensor | None = None        # fp16 [ih*id, d] (dequantized) — full indexer layers only
    idx_wk: torch.Tensor | None = None          # bf16 [id, d]
    idx_proj: torch.Tensor | None = None        # f32 [ih, d]
    idx_k_norm: torch.Tensor | None = None      # f32 [id]
    idx_k_norm_b: torch.Tensor | None = None    # f32 [id]
    idx_q_norm: torch.Tensor | None = None      # f32 [d] (biased LayerNorm on the block input)
    idx_q_norm_b: torch.Tensor | None = None    # f32 [d]
    idx_ssmax_s: torch.Tensor | None = None     # f32 [ih]
    idx_ssmax_b: torch.Tensor | None = None     # f32 [ih]
    gate_w: torch.Tensor | None = None          # f32 [E, d] — None on the dense layer
    dense_gate: object = None                   # Exl3Linear [d -> dense_inter] — dense layers only
    dense_up: object = None
    dense_down: object = None                   # [dense_inter -> d]
    experts: object = None                      # Exl3RoutedExperts: local_ids' 32 routed + the shared as last
    expert_ids: tuple = ()                      # global ids of ``experts`` entries 0..E-1, shared last separate


@dataclass
class Ml4Weights:
    cfg: Cfg
    rank: int
    world: int
    embed: torch.Tensor             # bf16 [V, d] replicated
    norm: torch.Tensor              # f32 [d]
    alpha: torch.Tensor             # f32 [rd/2] LeRoPE, one table for the model (D1 §7)
    head: object                    # Exl3Linear (head_bits) full vocab, or None when the head is bf16/tied
    head_bf16: torch.Tensor | None  # bf16 [V, d] when the pack head mode is "source"
    vocab_lo: int                   # always 0 / vocab: replicated (D3 §2), kept for the engine's interface
    vocab_hi: int
    layers: list = field(default_factory=list)
    expert_ids: tuple = ()          # local_ids(rank) as loaded
    plan: dict | None = None
    model_dir: Path | None = None
    split: bool = False             # a layer's ``experts`` hold a quarter of every expert (and of the shared one), not whole ones


# -- the load path (CUDA; import inside: numpy enters with the shared Shards/RankCache) ---------------------------

def load(model_dir: str | Path, rank: int = 0, world: int = 4, expert_rule: str = "block", log=print) -> Ml4Weights:
    """Read one rank's weights from the pack (D3 §2's per-rank load plan). Any missing tensor, unexpected tensor,
    wrong dtype, short read or rank-cache order break aborts naming the tensor and the file; the pack files are
    read-only and the rank cache self-deletes on corruption."""

    import time

    t0 = time.time()
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    cfg = Cfg.from_dict(config)
    manifest = json.loads((model_dir / "ml4_pack.json").read_text())
    index = PackIndex(model_dir).load()
    verify_manifest(manifest, config, index)
    bits, codebook = manifest["bits"], manifest["codebook"]
    lo, hi = manifest["layers"][:2]
    assert hi - lo + 1 == cfg.n_layers, f"pack covers layers {lo}..{hi}, config says {cfg.n_layers}"

    for i in range(lo, hi + 1):                       # gate 3: header preflight before any allocation
        check_layer_headers(index, cfg, bits, codebook, i)
    head_mode = manifest["head"]
    head_bits = {"quant": 6.0, "source": 16.0}.get(head_mode, 0.0)
    check_globals_head(index, cfg, head_mode, head_bits, codebook)
    check_exl3_markers(index, codebook)

    split = world > 1 and os.environ.get("TF_ML4_EXPERTS", "split") == "split"
    ids = list(range(cfg.n_routed)) if split else local_ids(rank, world, cfg.n_routed, expert_rule)
    plan = rank_plan(cfg, codebook, index, rank, world, head_mode, expert_rule, split)

    from tensorfold.cuda.exl3 import load as shared    # numpy lands here (a GPU host has it; D3 §1's lift)

    class CountingShards(shared.Shards):
        """Shards with a byte meter on cold reads: D3 §3's 'reading != plan is a failure'."""

        def __init__(self, root):
            super().__init__(root)
            self.bytes_read = 0

        def get(self, name, rows=None):
            t = super().get(name, rows)
            self.bytes_read += t.numel() * t.element_size()
            return t

    extra = (f"ml4|{'split' if split else expert_rule}|{lo}-{hi}|{head_mode}|{cfg.n_layers}layers|"
             f"{hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]}")
    rc = shared.open_rank_cache(os.environ.get("TF_ML4_RANK_CACHE"), model_dir, rank, world, extra,
                                loader_file=Path(__file__), need_bytes=plan["cache_bytes"], prefix="ml4rank")
    sh = CountingShards(model_dir)
    sh.cache = rc
    if rc is not None:
        log(f"[ml4-load] rank {rank}: rank cache {'reading' if rc.reading else 'writing'} {rc.path}")
    cold = rc is None or not rc.reading
    try:
        w = _load(sh, index, cfg, codebook, model_dir, rank, world, ids, head_mode, plan, log, t0, split)
    except BaseException:
        if rc is not None:
            rc.close(ok=False)
        raise
    if rc is not None:
        rc.close(ok=True)
        log(f"[ml4-load] rank {rank}: rank cache {'read' if not cold else 'written'} "
            f"({plan['cache_bytes'] / 2**30:.1f} GiB plan)")
    sh.close()

    # D3 §4's load-time gates: the GPU holds what the plan said (kept dtype), and a cold start read exactly the
    # planned bytes from the pack (D3 §3: reading != plan is a failure, not a curiosity).
    torch.cuda.synchronize()
    got = torch.cuda.memory_allocated()
    tol = plan["cache_bytes"] // 50 + (1 << 20)        # 2 % + 1 GiB: allocator rounding of the packed buffers
    assert abs(got - plan["cache_bytes"]) <= tol, \
        f"GPU bytes {got / 2**30:.2f} GiB off kept-plan {plan['cache_bytes'] / 2**30:.2f} GiB (D3 §4)"
    if cold:
        assert abs(sh.bytes_read - plan["read_bytes"]) <= plan["read_bytes"] // 50 + (1 << 20), \
            f"bytes read {sh.bytes_read / 2**30:.2f} GiB != plan {plan['read_bytes'] / 2**30:.2f} GiB (D3 §3)"
    log(f"[ml4-load] rank {rank}: bytes={sh.bytes_read} plan={plan['read_bytes']} "
        f"(routed/world {plan['routed_bytes']} + replicated {plan['replicated_bytes']}), "
        f"tensors={len(index.entries)} in pack, {len(w.layers)} layers, {time.time() - t0:.0f} s, "
        f"cache={'hit' if not cold else ('miss' if rc is not None else 'off')}")
    return w


def _load(sh, index: PackIndex, cfg: Cfg, codebook: str, model_dir: Path, rank: int, world: int, ids: list[int],
          head_mode: str, plan: dict, log, t0, split: bool = False) -> Ml4Weights:
    import time

    from tensorfold.cuda.exl3 import experts as exl3_experts
    from tensorfold.cuda.exl3.load import _kept, exl3_parts, linear, pack_trellises

    def plain(name: str, how: str) -> torch.Tensor:
        """One -nq tensor at its kept precision (D3 §2: the only casts widen; source dtype was asserted at
        preflight from the same header this read uses)."""

        target = _NQ_CAST[how]

        def make():
            t = sh.get(name)
            if how == "fp8deq":                          # the wq_b pair: dequantized together, bytes untouched
                scale = sh.get(name.replace(".weight", ".weight_scale_inv"))
                return dequant_fp8_block(t, scale).contiguous()
            return (t.to(target) if target is not None else t).contiguous()

        return _kept(sh, f"{name}|{how}", make).cuda()

    def nq(i: int, suffix: str) -> torch.Tensor:
        return plain(LP.format(L=i) + suffix, LAYER_SHIM[suffix][0])

    w = Ml4Weights(cfg=cfg, rank=rank, world=world,
                   embed=plain(f"{LM}.embed_tokens.weight", "bf16"),
                   norm=plain(f"{LM}.norm.weight", "f32"),
                   alpha=plain(f"{LM}.rotary_emb.alpha", "f32"),
                   head=None, head_bf16=None, vocab_lo=0, vocab_hi=cfg.vocab,
                   expert_ids=tuple(ids), plan=plan, model_dir=model_dir, split=split)
    if head_mode == "quant":
        w.head = linear(sh, "lm_head")                       # replicated, full vocabulary
    elif head_mode == "source":
        w.head_bf16 = plain("lm_head.weight", "bf16")        # partial packs: the bf16 source head
    elif head_mode != "none":
        raise ValueError(f"unknown head mode {head_mode!r}")

    for i in range(cfg.n_layers):
        p = LP.format(L=i)
        lay = Ml4Layer(
            idx=i, rope=cfg.is_rope(i), kv_source=cfg.kv_source_layers[i], indexer=cfg.indexer_types[i],
            dense=cfg.is_dense(i), attn_norm=nq(i, "input_layernorm.weight"),
            ffn_norm=nq(i, "post_attention_layernorm.weight"),
            q=linear(sh, p + "self_attn.q_proj"), o=linear(sh, p + "self_attn.o_proj"),
            kv_a=None if cfg.is_reader(i) else nq(i, "self_attn.kv_a_proj_with_mqa.weight"),
            kv_a_norm=None if cfg.is_reader(i) else nq(i, "self_attn.kv_a_layernorm.weight"),
            kv_b=nq(i, "self_attn.kv_b_proj.weight"),
            ssmax_s=nq(i, "self_attn.softmax_scaling.scale"), ssmax_b=nq(i, "self_attn.softmax_scaling.bias"))
        if cfg.is_full_indexer(i):
            lay.idx_wk = nq(i, "self_attn.indexer.wk.weight")
            lay.idx_proj = nq(i, "self_attn.indexer.weights_proj.weight")
            lay.idx_k_norm = nq(i, "self_attn.indexer.k_norm.weight")
            lay.idx_k_norm_b = nq(i, "self_attn.indexer.k_norm.bias")
            lay.idx_q_norm = nq(i, "self_attn.indexer.q_input_norm.weight")
            lay.idx_q_norm_b = nq(i, "self_attn.indexer.q_input_norm.bias")
            lay.idx_ssmax_s = nq(i, "self_attn.indexer.softmax_scaling.scale")
            lay.idx_ssmax_b = nq(i, "self_attn.indexer.softmax_scaling.bias")
        if cfg.is_dense(i):
            lay.dense_gate = linear(sh, p + "mlp.gate_proj")
            lay.dense_up = linear(sh, p + "mlp.up_proj")
            lay.dense_down = linear(sh, p + "mlp.down_proj")
        else:
            lay.gate_w = nq(i, "mlp.gate.weight")
            gate, up, down = [], [], []
            cut = None
            if split:
                q = cfg.inter // world
                cut = (rank * q, (rank + 1) * q)
            for ep in [f"{p}mlp.experts.{e}" for e in ids] + [f"{p}mlp.shared_experts"]:
                g = exl3_parts(sh, f"{ep}.gate_proj", cols=cut, device=None)    # 32 routed (local_ids) + shared appended
                u = exl3_parts(sh, f"{ep}.up_proj", cols=cut, device=None)
                dn = exl3_parts(sh, f"{ep}.down_proj", rows=cut, device=None)
                assert g[3] == u[3] == dn[3] == codebook, f"{ep}: codebook disagrees with the manifest"
                gate.append(g[:3])
                up.append(u[:3])
                down.append(dn[:3])
            trs = pack_trellises([m[0] for m in gate + up + down])
            n = len(gate)
            gate = [(trs[k], m[1].cuda(), m[2].cuda()) for k, m in enumerate(gate)]
            up = [(trs[n + k], m[1].cuda(), m[2].cuda()) for k, m in enumerate(up)]
            down = [(trs[2 * n + k], m[1].cuda(), m[2].cuda()) for k, m in enumerate(down)]
            lay.experts = exl3_experts.prepare(gate, up, down, codebook)
            lay.expert_ids = tuple(ids)                  # the shared expert sits after these (D1 §Extension pt. 4)
            assert lay.experts.count == len(ids) + 1, \
                f"layer {i}: {lay.experts.count} expert slots, want {len(ids) + 1} (D2 §2)"
        w.layers.append(lay)
        if i % 6 == 5 or i == cfg.n_layers - 1:
            log(f"[ml4-load] rank {rank}: layers 0..{i} loaded ({time.time() - t0:.0f} s, "
                f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB)")

    for i, lay in enumerate(w.layers):                       # together, after the rest: cp4_cuda.wq_b_from_pack frees half of what they hold
        if cfg.is_full_indexer(i):
            lay.idx_wq_b = nq(i, "self_attn.indexer.wq_b.weight")     # fp8 -> fp16 (dequant_fp8_block)

    moe = [lay for lay in w.layers if lay.expert_ids]
    assert all(lay.expert_ids == tuple(ids) for lay in moe), \
        f"rank {rank}: a layer loaded expert ids != local_ids {tuple(ids)} (D3 §4)"
    if cfg.n_routed and moe:
        assert len(moe) == sum(1 for i in range(cfg.n_layers) if not cfg.is_dense(i)), \
            "every sparse layer must carry the rank's expert set"
    return w
