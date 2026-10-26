"""Check a Ling-3.1-flash checkpoint's tensor-parallel split from its headers alone: every rank's share at TP W.

    python tools/ling_split_check.py MODEL_DIR [--world 2 --world 4] [--table]

MODEL_DIR is the EXL3 conversion (the served form) or the banked FP8 source; only config.json and the safetensors
headers are read, and a name two or more shards hold has its copies' bytes read too (to tell identical copies, a
note, from differing ones, a problem that fails the run). The arithmetic is split.py's own (rule, unit, segments,
spans: what RankReader and split_file cut with), so a pass here is a pass of the bytes each rank will read.

For each world, tensor and rank:
  1. rule: every name rules exactly once (row / col / dim1 / rep) and nothing is dropped;
  2. partition: the ranks' runs cover the split axis once, in order, segment by segment;
  3. Hadamard blocks: an EXL3 trellis / suh / svh span never cuts a 128-column block, and the parts of one group
     agree on every rank's columns (the trellis's tiles and the scale vector on the same side);
  4. heads: per layer and rank, every KDA tensor (q | k | v fused or not, f, g, b, the convs, A_log, dt_bias,
     o_proj) and every MLA tensor (q_b, kv_b, the head gate, dense) holds the same whole heads, summing to the
     config's heads over the ranks;
  5. widths: per layer and rank, gate / up's output width equals down's input width for the dense MLP, the shared
     expert and the routed experts (one width for all routed experts of a layer), summing to the config's.
Then each rank's weight bytes (the head at its vocabulary slice, as the loader keeps it). An FP8 block scale whose
grid does not follow its weight's split is a note, not a problem: the FP8 bank is the conversion's source, never
split for serving. Exit status 1 on any problem; ``--table`` prints every role's rank shapes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfold.cuda.geometry import share, share_lo  # noqa: E402
from tensorfold.families.bailing_v3.config import Config  # noqa: E402
from tensorfold.families.glm5_next.cuda import split  # noqa: E402

ITEMSIZE = {**split.DTYPE_BYTES, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1}
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
SCALE_SIDE = {"suh": "in", "su": "in", "svh": "out", "sv": "out"}
FP8_BLOCK = 128
# KDA head dims (lin_dim) one head holds, per tensor, on the split axis (o_proj: its input columns)
KDA_PER_HEAD = {"qkv_proj": 3, "conv1d": 3, "q_proj": 1, "k_proj": 1, "v_proj": 1, "f_proj": 1, "g_proj": 1,
                "q_conv1d": 1, "k_conv1d": 1, "v_conv1d": 1, "dt_bias": 1, "o_proj": 1}
KDA_ONE = ("b_proj", "A_log")                     # one row (entry) a head


def _same_bytes(path_a: Path, a: int, b: int, path_b: Path, c: int, d: int) -> bool:
    """The bytes at path_a[a:b] equal those at path_b[c:d], read a chunk at a time."""

    if b - a != d - c:
        return False
    with open(path_a, "rb") as x, open(path_b, "rb") as y:
        x.seek(a)
        y.seek(c)
        for i in range(0, b - a, 1 << 24):
            n = min(1 << 24, b - a - i)
            if x.read(n) != y.read(n):
                return False
    return True


def headers(model_dir: Path, dups: list[tuple[str, str]] | None = None) -> dict[str, tuple[str, list[int]]]:
    """name -> (dtype, shape) over every safetensors file of ``model_dir`` (headers only).

    A name more than one file holds is reported through ``dups`` as ("note", line) when its copies are
    byte-identical (the split keeps one) or ("problem", line) when they differ; those copies are the only
    weights this tool ever reads.
    """

    out: dict[str, tuple[str, list[int]]] = {}
    first: dict[str, tuple[Path, int, int]] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        header, base = split.read_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in out:
                if dups is not None:
                    pa, fa, fb = first[name]
                    what = f"{name} is in {pa.name} and {path.name}"
                    a, b = info["data_offsets"]
                    if (info["dtype"], list(info["shape"])) == out[name] and \
                            _same_bytes(path, base + a, base + b, pa, fa, fb):
                        dups.append(("note", f"{what}: byte-identical copies (the split keeps one)"))
                    else:
                        dups.append(("problem", f"{what}: the copies differ"))
                continue
            out[name] = (info["dtype"], list(info["shape"]))
            a, b = info["data_offsets"]
            first[name] = (path, base + a, base + b)
    if not out:
        raise SystemExit(f"{model_dir}: no safetensors headers")
    return out


def head_slice(blocks: int, world: int, rank: int) -> tuple[int, int]:
    """(first block, blocks) of the EXL3 head a rank keeps: x3.vocab_slice's balanced whole-128-column span."""

    return share_lo(blocks, world, rank), share(blocks, world, rank)


def plan(name: str, shape: list[int], world: int, hint: dict) -> tuple[str, int, int, list[list[tuple[int, int]]]]:
    """(kind, unit, segments, each rank's runs on the split axis) for ``name``: split.py's arithmetic only."""

    kind = split.rule(name)
    if kind in ("rep", "drop") or not shape:
        return kind, 1, 1, []
    u = split.unit(name, shape, kind, world, hint)
    parts = split.segments(name)
    n = shape[0] if kind == "row" else shape[1]
    return kind, u, parts, [split.spans(n, world, r, u, parts) for r in range(world)]


def role(cfg: Config, name: str) -> str:
    """``name`` with its layer and expert numbers blanked, prefixed by the layer's attention kind."""

    m = LAYER.search(name)
    abstract = re.sub(r"\.\d+\.", ".#.", name)
    if not m:
        return abstract
    li = int(m.group(1))
    kind = "MTP" if li >= cfg.num_hidden_layers else cfg.layer_kinds[li].upper()
    return f"{kind} {abstract}"


def _measure(cfg: Config, name: str, part: str | None, extent: int) -> tuple[str, float] | None:
    """(what a rank's share of ``name`` measures, how much): a layer's KDA / MLA heads or an MLP's width."""

    m = LAYER.search(name)
    if not m or part in ("mcg", "mul1", "3inst", "bias") or name.endswith("weight_scale_inv"):
        return None
    li, rest = int(m.group(1)), m.group(2)
    attn = cfg.layer_kinds[li] if li < cfg.num_hidden_layers else "mla"     # the MTP layer attends with MLA
    if rest.startswith("attention."):
        what = rest[len("attention."):].split(".")[0]
        if attn == "kda":
            per = KDA_PER_HEAD.get(what, 0) * cfg.lin_dim or (1 if what in KDA_ONE else 0)
            return (f"L{li} KDA heads", extent / per) if per else None
        per = {"q_b_proj": cfg.qk_head_dim, "kv_b_proj": cfg.kv_b_per_head, "g_proj": 1,
               "dense": cfg.v_head_dim}.get(what, 0)
        return (f"L{li} MLA heads", extent / per) if per else None
    mm = re.match(r"mlp\.(experts\.\d+\.|shared_experts\.|)(gate|up|down)_proj\.", rest)
    if mm:
        who = {"": "dense MLP", "shared_experts.": "shared expert"}.get(mm.group(1), "routed experts")
        return f"L{li} {who} width ({'in' if mm.group(2) == 'down' else 'out'})", extent
    return None


def check(names: dict[str, tuple[str, list[int]]], world: int,
          config: dict) -> tuple[list[str], list[str], dict, list[int]]:
    """(problems, notes, role table, weight bytes per rank) of the checkpoint ``names`` split over ``world`` ranks.

    The role table maps each role (``role``) to (kind, unit, segments, every rank's shape) and each measure
    ("L4 routed experts width (out)", "L7 MLA heads") to ("measure", 1, 1, every rank's amount).
    """

    cfg = Config.from_dict(config)
    hint = {"qk_dim": cfg.qk_head_dim, "qk_nope": cfg.qk_nope_head_dim, "v_dim": cfg.v_head_dim}
    problems: list[str] = []
    notes: set[str] = set()
    table: dict[str, tuple] = {}
    total = [0] * world
    measures: dict[tuple[str, int], dict[float, str]] = defaultdict(dict)    # (what, rank) -> {amount: a name}
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(dict)       # (EXL3 prefix, side) -> part -> cols
    weight_runs: dict[str, list] = {}
    for name, (dtype, shape) in sorted(names.items()):
        try:
            kind, u, parts, runs = plan(name, shape, world, hint)
        except ValueError as e:
            problems.append(f"{name}: {e}")
            continue
        if kind == "drop":
            problems.append(f"{name}: dropped by the split rules (Ling serves every tensor it ships)")
            continue
        size = ITEMSIZE.get(dtype)
        if size is None:
            problems.append(f"{name}: dtype {dtype} is not one the reader knows")
            continue
        p = split.EXL3_PART.search(name)
        part = p.group(1) if p else None
        axis = 0 if kind == "row" else 1
        shapes = []
        for r in range(world):
            s = list(shape)
            if runs:
                s[axis] = sum(t for _, t in runs[r])
            if name.startswith("lm_head.") and part in ("trellis", "svh", "sv"):   # the loader's vocabulary slice
                ax, tile = (1, 8) if part == "trellis" else (0, 128)
                s[ax] = head_slice(shape[ax] // tile, world, r)[1] * tile
            elif name == "lm_head.weight":
                s[0] = share(shape[0], world, r)
            shapes.append(tuple(s))
            total[r] += math.prod(s) * size
        table.setdefault(role(cfg, name), (kind, u, parts, shapes))
        if not runs:
            continue
        if name.endswith(".weight") and kind in ("row", "col"):
            weight_runs[name] = runs
        n = shape[axis]
        flat = sorted(run for r in range(world) for run in runs[r])
        at = 0
        for a, t in flat:                                                  # 2. one partition of the axis
            if a != at:
                problems.append(f"{name}: the ranks' runs leave a gap or overlap at {at}")
                break
            at = a + t
        else:
            if at != n:
                problems.append(f"{name}: the ranks' runs cover {at} of {n}")
        if part == "trellis" or part in SCALE_SIDE:                       # 3. whole Hadamard blocks
            per_tile = 1 if part == "trellis" else 16                       # axis items a 16-column tile holds
            cut = next(((a, t) for a, t in flat if a % (8 * per_tile) or t % (8 * per_tile)), None)
            if cut:
                problems.append(f"{name}: span {cut} cuts a 128-column Hadamard block ({world} ranks)")
            side = ("out" if kind == "dim1" else "in") if part == "trellis" else SCALE_SIDE[part]
            groups[(name[:p.start()], side)][part] = [[(a * 16 // per_tile, t * 16 // per_tile) for a, t in rr]
                                                      for rr in runs]
        for r in range(world):                                             # 4. and 5.: what the rank holds
            got = _measure(cfg, name, part, sum(t for _, t in runs[r]) * (16 if part == "trellis" else 1))
            if got:
                measures[(got[0], r)].setdefault(got[1], name)
    for (prefix, side), parts in sorted(groups.items()):                   # 3. a group's parts agree
        if len({json.dumps(c) for c in parts.values()}) > 1:
            problems.append(f"{prefix}: its {side}put-side parts {sorted(parts)} split different columns")
    for name, runs in weight_runs.items():                                 # FP8 scale grids against the weight
        scale = name[:-len("weight")] + "weight_scale_inv"
        if scale not in names:
            continue
        axis = 0 if split.rule(name) == "row" else 1
        n = names[name][1][axis]
        try:
            sruns = plan(scale, names[scale][1], world, hint)[3]
        except ValueError as e:
            problems.append(f"{scale}: {e}")
            continue
        want = [[(a * FP8_BLOCK, min((a + t) * FP8_BLOCK, n) - a * FP8_BLOCK) for a, t in rr if t] for rr in sruns]
        if want != [[run for run in rr if run[1]] for rr in runs]:
            notes.add(role(cfg, scale))
    whats = sorted({w for w, _ in measures})
    for what in whats:
        per_rank: list[float | None] = []
        for r in range(world):
            vals = measures[(what, r)]
            if len(vals) != 1:
                problems.append(f"{what}, rank {r}: its tensors disagree ({dict(list(vals.items())[:3])})")
                per_rank.append(None)
                continue
            (v, first), = vals.items()
            if v != int(v):
                problems.append(f"{what}, rank {r}: {first} holds {v:g} heads, not a whole number")
            per_rank.append(v)
        if None in per_rank:
            continue
        if "KDA" in what:
            want = cfg.lin_heads
        elif "MLA" in what:
            want = cfg.num_attention_heads
        elif "routed" in what:
            want = cfg.moe_intermediate_size
        elif "shared" in what:
            want = (cfg.shared_intermediate_size or cfg.moe_intermediate_size) * cfg.n_shared_experts
        else:
            want = cfg.intermediate_size
        if sum(per_rank) != want:
            problems.append(f"{what}: the ranks hold {per_rank}, {sum(per_rank):g} in all, not {want}")
        table[what] = ("measure", 1, 1, [int(v) for v in per_rank])
    for what in whats:                                                     # 5. gate / up out == down in
        if what.endswith("(out)"):
            other = what[:-len("(out)")] + "(in)"
            if what in table and other in table and table[what][3] != table[other][3]:
                problems.append(f"{what[:-6]}: gate / up out {table[what][3]} != down in {table[other][3]}")
    return problems, sorted(notes), table, total


def summary(table: dict) -> dict[str, dict[tuple, int]]:
    """Each measure without its layer number -> {every rank's amount: layers}: one line a kind of measure."""

    out: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
    for what, row in table.items():
        if row[0] == "measure":
            out[what.split(" ", 1)[1]][tuple(row[3])] += 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--world", type=int, action="append", help="ranks (repeat for several; default 2 and 4)")
    ap.add_argument("--table", action="store_true", help="print every role's rank shapes")
    args = ap.parse_args(argv)
    config = json.loads((args.model_dir / "config.json").read_text())
    dups: list[tuple[str, str]] = []
    names = headers(args.model_dir, dups)
    bad = sum(1 for kind, _ in dups if kind == "problem")
    for kind, line in dups:
        print(f"  {'NOTE' if kind == 'note' else 'PROBLEM'} {line}")
    for world in args.world or [2, 4]:
        problems, notes, table, total = check(names, world, config)
        print(f"TP{world}: {len(names)} tensors, {len(problems)} problem(s); weight GiB a rank: "
              + " / ".join(f"{b / 2 ** 30:.3f}" for b in total))
        for what, seen in sorted(summary(table).items()):
            for amounts, layers in sorted(seen.items()):
                print(f"  {what}: {list(amounts)} a rank ({layers} layer{'s' if layers > 1 else ''})")
        if args.table:
            for name, (kind, u, parts, shapes) in sorted(table.items()):
                if kind != "measure":
                    print(f"    {name}: {kind}, unit {u}, {parts} segment(s) -> "
                          f"{[list(s) for s in sorted(set(shapes), key=shapes.index)]}")
        for line in notes:
            print(f"  NOTE {line}: the FP8 block scales do not follow the weight's split at TP{world} (an FP8 "
                  "rank split is not a serving path: convert first)")
        for line in problems[:50]:
            print("  PROBLEM", line)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        bad += len(problems)
    print("PASS" if not bad else f"FAIL ({bad} problems)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
