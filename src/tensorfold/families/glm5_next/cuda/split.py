"""Read each rank directly from checkpoint bytes or a saved rank folder, preserving packed groups at every split boundary."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

from tensorfold.cuda.geometry import share, share_lo

ROW = (
    r"\.mlp\.experts\.\d+\.(gate|up)_proj\.",
    r"\.mlp\.shared_experts\.(gate|up)_proj\.",
    r"\.mlp\.(gate|up)_proj\.",
    r"\.self_attn\.(q|k|v)_proj\.",
    r"\.self_attn\.(q|k|v)_conv1d\.",
    r"\.self_attn\.(f_b|g_b|b)_proj\.",
    r"\.self_attn\.(A_log|dt_bias)$",
    r"\.self_attn\.(q_b|kv_b)_proj\.",
    # BailingMoeV3 ("attention.", no self_attn): full-rank q/k/v, the decay input f, the output gate g and the
    # beta logits b (b_proj's 64 rows are one a head, so an even world splits it like any other row tensor).
    r"\.attention\.(q|k|v|f|g|b)_proj\.",
    r"\.attention\.(q|k|v)_conv1d\.",
    r"\.attention\.(A_log|dt_bias)$",
    r"\.attention\.(q_b|kv_b)_proj\.",
    # the EXL3 conversion's fused KDA tensors (exllamav3's GatedDeltaNet keeps q | k | v as one qkv_proj group and
    # writes the three short convs as one conv1d): each third splits by head (SEGMENTS)
    r"\.attention\.qkv_proj\.",
    r"\.attention\.conv1d\.weight$",
)
COL = (
    r"\.mlp\.experts\.\d+\.down_proj\.",
    r"\.mlp\.shared_experts\.down_proj\.",
    r"\.mlp\.down_proj\.",
    r"\.self_attn\.o_proj\.",
    r"\.attention\.(o_proj|dense)\.",      # the KDA out projection and the MLA one (named dense)
)
REP = (
    r"^lm_head\.", r"embed_tokens\.", r"^model(\.language_model)?\.norm\.weight$",   # bare or merged final norm
    r"_layernorm\.weight$", r"\.hc_(attn|ffn)_(fn|base|scale)$", r"\.mlp\.gate\.(weight|e_score_correction_bias)$",
    r"\.self_attn\.indexer\.", r"\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.", r"\.self_attn\.(f_a|g_a)_proj\.",
    r"\.self_attn\.o_norm\.weight$", r"\.(eh_proj)\.", r"\.(enorm|hnorm)\.weight$", r"\.shared_head\.norm\.weight$",
    # BailingMoeV3: its embedding table is word_embeddings, its chain norm final_layernorm, its router bias
    # expert_bias; q_a / kv_a and the KDA gated-RMSNorm weight are replicated like their GLM counterparts.
    r"word_embeddings\.", r"\.attention\.(q_a_proj|kv_a_proj_with_mqa)\.", r"\.attention\.o_norm\.weight$",
    r"\.mlp\.gate\.(weight|expert_bias)$", r"\.final_layernorm\.weight$",
)
RUN = 128 << 20          # most bytes one read of neighbouring tensors takes (``RankReader.prefetch``)
GAP = 1 << 20            # most unused bytes such a read spans between two tensors (more reads other layers twice)
READERS = 8              # reads in flight: past this the layers are built slower than they are read
DTYPE_BYTES = {"U32": 4, "I32": 4, "F32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "U8": 1, "I8": 1, "I64": 8,
               "F64": 8}
# the files a rank folder needs besides its weights (the tokenizer, chat template and configs)
SMALL = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "processor_config.json", "model.safetensors.index.json")


# EXL3 gate/up split tile columns and svh; down splits tile rows and suh; each replicates the remaining tensors.
EXL3_EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(trellis|suh|svh|mcg|mul1|3inst)$")
EXL3_RULES = {("gate", "trellis"): "dim1", ("gate", "suh"): "rep", ("gate", "svh"): "row",
              ("down", "trellis"): "row", ("down", "suh"): "row", ("down", "svh"): "rep"}
# Any other EXL3 group (attention, dense and shared MLPs, head, MTP projections): its plain weight's rule decides.
# A trellis is [K/16, N/16, words] (input tiles first), the transpose of a plain [N, K] weight: an output split
# ("row") cuts the trellis's second axis and svh, an input split ("col") its first axis and suh; the codebook
# marker and the other scale vector are replicated. Rank parts stay whole 128-column Hadamard blocks (``unit``).
EXL3_PART = re.compile(r"\.(trellis|suh|svh|su|sv|mcg|mul1|3inst|bias)$")
EXL3_BY_KIND = {"row": {"trellis": "dim1", "svh": "row", "sv": "row", "bias": "row"},
                "col": {"trellis": "row", "suh": "row", "su": "row"}}
BLOCK = 128              # EXL3's Hadamard block: 8 of the trellis's 16-column tiles, 128 suh / svh entries
# tensors whose split axis is equal segments that each split on their own: the conversion's fused q | k | v (a
# rank keeps [q_r | k_r | v_r], the layout the KDA chain reads) and the three short convs written as one
SEGMENTS = ((re.compile(r"\.attention\.qkv_proj\."), 3), (re.compile(r"\.attention\.conv1d\.weight$"), 3))


def rule(name: str) -> str:
    if name.startswith("model.visual."):
        return "drop"
    m = EXL3_EXPERT.search(name)
    if m:
        proj, part = m.groups()
        return EXL3_RULES.get(("gate" if proj == "up" else proj, part), "rep")
    base = name
    p = EXL3_PART.search(name)
    if p:
        base = name[:p.start()] + ".weight"
    hits = [kind for kind, pats in (("row", ROW), ("col", COL), ("rep", REP)) if any(re.search(r, base) for r in pats)]
    if len(hits) != 1:
        raise ValueError(f"{name}: split rule is ambiguous or missing ({hits})")
    if p:
        return EXL3_BY_KIND.get(hits[0], {}).get(p.group(1), "rep")
    return hits[0]


HEAD_GROUP = (
    (re.compile(r"\.self_attn\.q_b_proj\."), lambda c: c["qk_dim"]),
    (re.compile(r"\.self_attn\.kv_b_proj\."), lambda c: c["qk_nope"] + c["v_dim"]),
    (re.compile(r"\.attention\.q_b_proj\."), lambda c: c["qk_dim"]),            # BailingMoeV3's names
    (re.compile(r"\.attention\.kv_b_proj\."), lambda c: c["qk_nope"] + c["v_dim"]),
)


def head_group(name: str, cfg: dict) -> int:
    """The rows one head contributes to ``name`` (0: not head-major), so uneven worlds split by whole heads.

    ``cfg``: the model config's qk_dim (nope + rope), qk_nope and v_dim. q_b's rows are head-major
    qk_dim a head; kv_b's are qk_nope + v_dim a head (key rows then value rows, per head) — GLM-5.3-Flash's
    qk_dim 256 / kv_b 512 and BailingMoeV3's 192 / 256 both come from the config, never hard-coded.
    """

    for pat, rows_of in HEAD_GROUP:
        if pat.search(name):
            return int(rows_of(cfg))
    return 0


def segments(name: str) -> int:
    """Equal segments ``name``'s split axis concatenates, each split on its own (1: one plain axis)."""

    for pat, n in SEGMENTS:
        if pat.search(name):
            return n
    return 1


def unit(name: str, shape: list[int], kind: str, world: int | None = None, cfg: dict | None = None) -> int:
    """Items of ``name``'s split axis a rank's span moves in, per segment (``split_bytes``' ``group``); 1: any.

    An EXL3 group's parts move in whole 128-column Hadamard blocks: the trellis's split axis in 8 16-column tiles,
    suh / svh in 128 entries. A span that cut a block would leave a rank half a rotation it cannot undo (the expert
    kernels refuse such a width: experts.cpp's N % 128 check). Where the world divides an axis's blocks this is the
    plain split byte for byte (every GLM-5.3 axis at TP2 and TP4, Ling's attention and dense axes); Ling's 1280-wide
    routed and shared experts at TP4 become 384/384/256/256 (10 blocks, the first 10 % 4 ranks one extra), GLM-5.3's
    2048 at W=6 384/384/384/384/256/256. An axis with fewer whole blocks than ranks (test-scale shapes) keeps the
    16-column tile split, its scale vector following the trellis's tiles. A bias (no EXL3 group in these families
    has one) splits plainly. Head-major rows (q_b, kv_b, with ``cfg``'s qk_dim / qk_nope / v_dim) move in whole
    heads, an EXL3 group's in whole heads and whole blocks at once (Ling's 192-row q_b heads: 384 rows, 2 heads in
    3 blocks); an even world never changes either.
    """

    W = _world() if world is None else world
    if kind not in ("row", "col", "dim1") or len(shape) < (1 if kind == "row" else 2):
        return 1
    n, seg = (shape[0] if kind == "row" else shape[1]), segments(name)
    if n % seg:
        return 1
    n //= seg
    if EXL3_EXPERT.search(name) or EXL3_PART.search(name):
        part = name.rsplit(".", 1)[-1]
        if part not in ("trellis", "suh", "svh", "su", "sv"):
            return 1
        tile = 1 if part == "trellis" else 16           # the axis's items one 16-column tile holds
        block = BLOCK // 16 * tile
        g = head_group(name, cfg) if cfg else 0         # a head-major output axis (q_b): whole heads as well
        if g and g % 16 == 0:
            block = math.lcm(block, g // 16 * tile)
        if n % block == 0 and n // block >= W:
            return block
        return tile if n % tile == 0 else 1
    g = head_group(name, cfg) if cfg else 0
    return g if g and n % g == 0 else 1


def spans(n: int, world: int, rank: int, group: int = 1, parts: int = 1) -> list[tuple[int, int]]:
    """The rank's runs (first item, items) of an ``n``-item split axis: ``parts`` equal segments, each a balanced
    uneven share of whole ``group``s (the first ranks one extra group)."""

    if parts < 1 or n % parts:
        raise ValueError(f"{n} items do not make {parts} equal segments")
    seg = n // parts
    if group < 1 or seg % group:
        raise ValueError(f"a segment of {seg} items does not split in groups of {group}")
    lo, take = share_lo(seg // group, world, rank) * group, share(seg // group, world, rank) * group
    return [(s * seg + lo, take) for s in range(parts)]


def read_header(path: str | Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


WORLD = int(os.environ.get("TF_TP_WORLD", "2"))    # ranks the checkpoint is split across


def _world() -> int:
    """Ranks the checkpoint is split across: per-call env wins over the import-time default.

    Import-time WORLD is baked before TF_TP_WORLD may exist (e.g. engine sets it at runtime), so
    helpers that receive world=None must not trust the stale global. Callers that pass world
    explicitly are unaffected.
    """

    try:
        return int(os.environ.get("TF_TP_WORLD", WORLD))
    except ValueError:
        return WORLD


def split_bytes(raw: np.ndarray, shape: list[int], itemsize: int, kind: str, rank: int,
                world: int | None = None, group: int = 1, parts: int = 1) -> tuple[np.ndarray, list[int]]:
    """A tensor's bytes -> rank's part of them and its shape (a balanced uneven share of the split axis).

    ``group``: the split axis's items travel in whole groups of this many rows/columns (a head-major
    tensor's rows: one head's 512 kv_b rows; an EXL3 group's 128-column Hadamard blocks). The balanced
    partition then runs on the group count, so a rank's span never lands inside a group. ``parts``: the
    axis is that many equal segments (a fused q | k | v), each split this way; the rank's runs concatenate.
    """

    W = _world() if world is None else world
    if kind == "col" and len(shape) != 2:
        raise ValueError(f"col split needs a 2-D tensor, got {shape}")
    n = shape[0] if kind == "row" else (shape[1] if kind in ("col", "dim1") else 0)
    if kind != "rep" and (not 0 < n or rank >= W or rank < 0):
        raise ValueError(f"{kind} split of {shape} into {W} ranks")
    if kind == "rep":
        return raw, list(shape)
    if kind == "row":
        rows = shape[0]
        if rows % group:
            raise ValueError(f"row split of {shape} in groups of {group}")
        per = raw.size // rows                # one row's bytes
        runs = spans(rows, W, rank, group, parts)
        take = sum(t for _, t in runs)
        if len(runs) == 1:
            return raw[runs[0][0] * per:(runs[0][0] + take) * per], [take] + list(shape[1:])
        return np.concatenate([raw[a * per:(a + t) * per] for a, t in runs]), [take] + list(shape[1:])
    if kind in ("col", "dim1"):                          # the second axis (col: of a 2-D tensor)
        inner = int(np.prod(shape[2:])) * itemsize
        view = raw.reshape(shape[0], shape[1] * inner)
        runs = spans(shape[1], W, rank, group, parts)
        pieces = [view[:, a * inner:(a + t) * inner] for a, t in runs]
        part = np.ascontiguousarray(pieces[0]) if len(pieces) == 1 else np.concatenate(pieces, axis=1)
        return part.reshape(-1), [shape[0], sum(t for _, t in runs)] + list(shape[2:])
    raise ValueError(kind)


def split_device(raw, shape: list[int], itemsize: int, kind: str, rank: int, world: int | None = None,
                 group: int = 1, parts: int = 1):
    """``split_bytes`` for a uint8 tensor on the GPU: the rank's part (a new contiguous tensor) and its shape.

    ``group`` / ``parts``: whole-group spans in equal segments, as in ``split_bytes``.
    """

    import torch

    W = _world() if world is None else world
    if kind == "rep":
        return raw.clone(), list(shape)
    if kind == "row":
        if shape[0] <= 0 or shape[0] % group:
            raise ValueError(f"row split of leading dim {shape} into {W} in groups of {group}")
        per = raw.numel() // shape[0]
        runs = spans(shape[0], W, rank, group, parts)
        pieces = [raw[a * per:(a + t) * per] for a, t in runs]
        data = pieces[0].clone() if len(pieces) == 1 else torch.cat(pieces)
        return data, [sum(t for _, t in runs)] + list(shape[1:])
    if kind in ("col", "dim1"):
        if len(shape) < 2 or (kind == "col" and len(shape) != 2) or shape[1] <= 0:
            raise ValueError(f"{kind} split needs a non-empty second dim, got {shape}")
        inner = int(np.prod(shape[2:])) * itemsize
        view = raw.view(shape[0], shape[1] * inner)
        runs = spans(shape[1], W, rank, group, parts)
        pieces = [view[:, a * inner:(a + t) * inner] for a, t in runs]
        part = pieces[0].contiguous() if len(pieces) == 1 else torch.cat(pieces, dim=1)
        return part.reshape(-1), [shape[0], sum(t for _, t in runs)] + list(shape[2:])
    raise ValueError(kind)


def torch_dtype(dtype: str):
    import torch

    return {"U32": torch.uint32, "I32": torch.int32, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16,
            "I16": torch.int16, "U16": torch.uint16, "U8": torch.uint8, "I8": torch.int8, "I64": torch.int64,
            "F64": torch.float64}[dtype]


def rank_files(model_dir: str | Path, rank: int) -> list[Path]:
    return sorted(Path(model_dir).glob(f"*.rank{rank}.safetensors"))


class RankReader:
    """Read stored-dtype CPU tensors for one rank from the full checkpoint or its pre-split folder."""

    def __init__(self, model_dir: str | Path, rank: int, cfg_hint: dict | None = None) -> None:
        """``cfg_hint``: the model config's ``qk_dim`` / ``v_head_dim`` (head-major group sizes at uneven worlds)."""

        from tensorfold.cuda.direct_read import ReadAhead, Reader, SafeTensors

        self.dir, self.rank = Path(model_dir), rank
        self.cfg_hint = cfg_hint
        self.io = Reader()                                # O_DIRECT reads where the file system allows them
        self.reads = ReadAhead(self.io, READERS, RUN, GAP)
        mine = rank_files(self.dir, rank)
        other = sorted(set(self.dir.glob("*.rank*.safetensors")) - set(mine))
        if other and not mine:
            raise ValueError(f"{self.dir} holds another rank's share ({other[0].name}): give rank {rank} its own "
                             "folder or the full checkpoint")
        self.split = bool(mine)
        if self.split:
            self.folder = SafeTensors(mine, self.io)
            self.index = dict.fromkeys(self.folder.keys())
            return
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            names = json.loads(index.read_text())["weight_map"]
        else:
            names = {k: p.name for p in sorted(self.dir.glob("*.safetensors")) for k in read_header(p)[0]
                     if k != "__metadata__"}
        self.files: dict[str, tuple[dict, int]] = {}
        self.index = dict(names)

    def prefetch(self, names, device=None) -> None:
        """Start reading ``names`` (a layer's hundreds of small expert tensors) in shared reads; with a CUDA ``device``, ``get`` returns them uploaded."""

        items = []
        for name in dict.fromkeys(names):
            span = self._span(name)
            items.append((name, span[0], span[1], span[2], span))
        self.reads.queue(items, device, self._cut)

    @property
    def ahead(self) -> dict:
        return self.reads.ahead

    def get(self, name: str):
        out = self.reads.take(name)
        return out if out is not None else self._read(name)

    def close(self) -> None:
        self.reads.close()

    def _span(self, name: str) -> tuple[str, int, int, str, list[int], str, int, int]:
        """(file, first byte, end byte, split kind, shape, dtype, group, segments) of the bytes this rank reads."""

        if self.split:                                    # a rank folder holds the rank's tensors as they are
            path, begin, n, dtype, shape = self.folder.where[name]
            return str(path), begin, begin + n, "rep", list(shape), dtype, 0, 1
        file = str(self.dir / self.index[name])
        if file not in self.files:
            self.files[file] = read_header(file)
        header, base = self.files[file]
        info = header[name]
        kind = rule(name)
        if kind == "drop":
            raise KeyError(f"{name} is not used by the engine")
        a, b = info["data_offsets"]
        shape = list(info["shape"])
        W = _world()
        group, parts = self._group(name, shape, kind), segments(name) if kind != "rep" else 1
        if kind == "row" and shape and shape[0] > 0 and parts == 1 and shape[0] % (group or 1) == 0:
            # the rank's rows are one run (whole groups, the balanced share): read only those
            (lo, take), = spans(shape[0], W, self.rank, group or 1)
            per = (b - a) // shape[0]
            a, b, kind, shape, group = a + lo * per, a + (lo + take) * per, "rep", [take] + shape[1:], 0
        return file, base + a, base + b, kind, shape, info["dtype"], group, parts

    def _group(self, name: str, shape: list[int], kind: str = "row") -> int:
        """``unit`` for ``name`` (0: none): whole heads for head-major rows, whole Hadamard blocks for EXL3 parts.

        Head-major tensors (q_b, kv_b) at a world that does not divide their head count need whole-head spans:
        the raw row share of an uneven split lands inside a head and the loader's per-head key/value slices would
        mix heads. An EXL3 group's split axis moves in 128-column blocks (``unit``). Where the world divides the
        groups the plain split gives the same bytes.
        """

        if self.split or not shape:
            return 0
        u = unit(name, shape, kind, _world(), self.cfg_hint)
        return 0 if u == 1 else u

    def _tensor(self, raw: np.ndarray, span: tuple, own: bool):
        """The rank's tensor from the span's bytes; ``own``: never a view of ``raw`` (a shared read's buffer)."""

        import torch

        _, _, _, kind, shape, dtype, group, parts = span
        data, shape = split_bytes(raw, shape, DTYPE_BYTES[dtype], kind, self.rank, group=group or 1, parts=parts)
        if not data.size:                    # none of the axis on this rank (widths under a tile a rank: test scale)
            return torch.empty(shape, dtype=torch_dtype(dtype))
        if own and np.may_share_memory(data, raw):
            data = data.copy()
        return torch.from_numpy(data).view(torch_dtype(dtype)).reshape(shape)

    def _read(self, name: str):
        if self.split:
            return self.folder.get(name)
        span = self._span(name)
        return self._tensor(self.io.read(span[0], span[1], span[2] - span[1]).numpy(), span, own=False)

    def _cut(self, raw, span: tuple):
        """The rank's tensor from its span's bytes, a view of a shared read (so copied), on the host or the device."""

        if not raw.is_cuda:
            return self._tensor(raw.numpy(), span, own=True)
        _, _, _, kind, shape, dtype, group, parts = span
        data, shape = split_device(raw, shape, DTYPE_BYTES[dtype], kind, self.rank, group=group or 1, parts=parts)
        if not data.numel():
            import torch

            return torch.empty(shape, dtype=torch_dtype(dtype), device=raw.device)
        return data.view(torch_dtype(dtype)).reshape(shape)

def write(path: str, tensors: list[tuple[str, str, list[int], np.ndarray]], metadata: dict | None) -> None:
    header: dict = {}
    offset = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.size]}
        offset += data.size
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, data in tensors:
            f.write(memoryview(data))
    os.replace(tmp, path)


def cfg_hint(model_dir: str | Path) -> dict | None:
    """The head-major group sizes ``unit`` reads (qk_dim, qk_nope, v_dim) from a checkpoint's config.json, or None."""

    path = Path(model_dir) / "config.json"
    if not path.exists():
        return None
    t = json.loads(path.read_text())
    t = t.get("text_config", t)
    nope, rope, v = t.get("qk_nope_head_dim"), t.get("qk_rope_head_dim"), t.get("v_head_dim")
    if None in (nope, rope, v):
        return None
    return {"qk_dim": int(nope) + int(rope), "qk_nope": int(nope), "v_dim": int(v)}


def _rank_written(out: str | Path, rank: int) -> dict[str, tuple[Path, int, int]]:
    """Names the rank folder already holds: name -> (rank file, first byte, end byte) of the written copy.

    A replicated name can arrive in more than one shard (the conversion writes the MTP layer's own pieces
    into two); whatever an earlier rank file holds must not be written again.
    """

    seen: dict[str, tuple[Path, int, int]] = {}
    for path in rank_files(out, rank):
        header, base = read_header(path)
        for name, info in header.items():
            if name != "__metadata__":
                a, b = info["data_offsets"]
                seen[name] = (path, base + a, base + b)
    return seen


def _same_written(path: Path, a: int, b: int, data: np.ndarray) -> bool:
    """``data`` equals the bytes at ``path[a:b]``, compared a chunk at a time (neither side materialized whole)."""

    if b - a != data.size:
        return False
    with open(path, "rb") as f:
        f.seek(a)
        for i in range(0, data.size, RUN):
            if f.read(min(RUN, data.size - i)) != data[i:i + RUN].tobytes():
                return False
    return True


def split_file(src: str | Path, out: str | Path, rank: int, cfg: dict | None = None) -> dict:
    """One checkpoint file -> OUT/<stem>.rank<R>.safetensors with the rank's part of every tensor it keeps.

    ``cfg``: ``cfg_hint``'s head sizes, so head-major rows split in whole heads (the loader's own rule). A
    name the rank folder already holds is written once: its copies are compared byte for byte and an
    unequal pair raises (the split never silently keeps the first of two differing copies).
    """

    header, base = read_header(src)
    metadata = header.pop("__metadata__", None)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    stem = os.path.basename(str(src)).replace(".safetensors", "")
    summary: dict[str, int] = {"rep": 0, "row": 0, "col": 0, "dim1": 0, "drop": 0}
    part = []
    W = _world()
    written = _rank_written(out, rank) if Path(out).is_dir() else {}
    for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
        info = header[name]
        kind = rule(name)
        if kind == "drop":
            summary[kind] += 1
            continue
        a, b = info["data_offsets"]
        itemsize = DTYPE_BYTES[info["dtype"]]
        data, shape = split_bytes(mm[base + a:base + b], info["shape"], itemsize, kind, rank, world=W,
                                  group=unit(name, info["shape"], kind, W, cfg),
                                  parts=segments(name) if kind != "rep" else 1)
        if int(np.prod(shape)) * itemsize != data.size:
            raise ValueError(f"{name}: {shape} does not match {data.size} bytes")
        if name in written:                          # a replicated name: one copy a rank, and they must agree
            path, wa, wb = written[name]
            if not _same_written(path, wa, wb, data):
                held = path.name.replace(f".rank{rank}.safetensors", ".safetensors")
                raise ValueError(f"{name}: rank {rank}'s copies differ between {held} and {src.name}")
            summary["dup"] = summary.get("dup", 0) + 1
            continue
        summary[kind] += 1
        part.append((name, info["dtype"], shape, data))
    os.makedirs(out, exist_ok=True)
    if part:
        write(os.path.join(str(out), f"{stem}.rank{rank}.safetensors"), part, metadata)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, help="the checkpoint (e.g. the snapshot `tensorfold pull` downloaded)")
    p.add_argument("--rank", type=int, required=True,
                   help=f"the rank this machine serves (0..{_world() - 1}; TF_TP_WORLD sets the world)")
    p.add_argument("out", type=Path, help="the folder to write (then: tensorfold serve OUT --tp W --rank R ...)")
    args = p.parse_args(argv)
    files = sorted(args.model_dir.glob("model-*.safetensors"))
    if not files:
        raise SystemExit(f"{args.model_dir}: no model-*.safetensors files")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SMALL:
        if (args.model_dir / name).exists():
            shutil.copyfile(args.model_dir / name, args.out / name)
    cfg = cfg_hint(args.model_dir)
    for src in files:
        stem = src.name.replace(".safetensors", "")
        if (args.out / f"{stem}.rank{args.rank}.safetensors").exists():
            continue
        print(src.name, split_file(src, args.out, args.rank, cfg), flush=True)
    print(f"rank {args.rank}'s share of {len(files)} files in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
