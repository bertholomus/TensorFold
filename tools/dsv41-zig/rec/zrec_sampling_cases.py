"""Goldens for the served sampler's wide top_k and top_k-off rows (zig/src/families/dsv41/sampling_cases.zig), from the
Python line's own rules on the CPU: exact_sampling.choose_rows over sample_rows' candidates (the top_k + MARGIN largest,
of equal values the lower ids, as sampling.zig takes them), and cuda.sampling.nucleus_rows on one rank for top_k 0.

Each row is synthetic and exact in float32 (token i: the top `bits` bits of splitmix64(gen + i * golden) times
2^-shift, less offset, plus bump at the peaks), so the Zig test rebuilds it bit for bit. numpy's float64 exp and log
must be libm's here, as on the served image (np_libm.py); the script checks that first. A top_k-off case is kept only
where nucleus_rows (torch's exp for the shares) and the same rule with libm's exp (the Zig's) draw the same token.

  docker run --rm --network none --user "$(id -u):$(id -g)" -v <python line>/src:/src:ro -v <worktree>:/wt -w /wt \
    -e PYTHONPATH=/src nvcr.io/nvidia/pytorch:26.07-py3 \
    python3 tools/dsv41-zig/rec/zrec_sampling_cases.py zig/src/families/dsv41/sampling_cases.zig
"""

import ctypes
import ctypes.util
from fractions import Fraction
import math
import sys

import numpy as np
import torch

from tensorfold.cuda.sampling import MASS, nucleus_rows
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows, uniform

GOLDEN = 0x9E3779B97F4A7C15
U = np.uint64

# name, gen, vocab, bits, shift, offset, bump, peaks
SHAPES = [
    ("peaked", 11, 129280, 20, 16, 8.0, 12.0, (7, 999, 1234, 4321, 77777)),
    ("flat", 12, 129280, 20, 20, 0.5, 0.0, ()),
    ("ties", 13, 129280, 4, 1, 4.0, 0.0, ()),
    ("steep", 14, 129280, 20, 12, 128.0, 0.0, ()),
    ("mid", 15, 129280, 16, 10, 32.0, 0.0, ()),
    ("small", 16, 1000, 20, 14, 32.0, 0.0, ()),
]
# (temperature, top_p, min_p)
TOPK0 = [(1.0, 0.95, 0.0), (0.6, 0.95, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.05), (1.0, 0.95, 0.05),
         (0.8, 0.9, 0.3), (1.0, 0.0, 0.0), (1.3, 0.999, 0.0)]
WIDE = [57, 100, 1000, 200000]
WIDE_P = [(1.0, 0.95, 0.0), (0.7, 0.9, 0.05), (1.0, 1.0, 0.0)]
NARROW = [1, 20, 56]
NARROW_P = [(1.0, 0.95, 0.0), (0.6, 0.8, 0.1)]
DRAWS = 3
SUMS = [129, 136, 200, 1000, 1008, 4099, 8192, 8193, 16385, 20000, 129280]
CEILS = [(0.95, 100), (0.1, 10), (0.5, 3), (0.5, 4), (0.9, 7), (0.999, 1 << 40), (0.95, 129280 << 40),
         (0.7, (1 << 57) - 1), (1e-300, 12345), (5e-324, 1), (0.3, 0), (0.9999999999999999, (1 << 40) + 1)]


def mix(x):
    with np.errstate(over="ignore"):
        x = x ^ (x >> U(30))
        x = x * U(0xBF58476D1CE4E5B9)
        x = x ^ (x >> U(27))
        x = x * U(0x94D049BB133111EB)
        return x ^ (x >> U(31))


def keys(gen, n):
    with np.errstate(over="ignore"):
        return mix(U(gen) + np.arange(n, dtype=np.uint64) * U(GOLDEN))


def row(gen, vocab, bits, shift, offset, bump, peaks):
    v = (keys(gen, vocab) >> U(64 - bits)).astype(np.float64) * 2.0 ** -shift - offset
    for p in peaks:
        v[p] += bump
    out = v.astype(np.float32)
    assert (out.astype(np.float64) == v).all(), "a row value is not exact in float32"
    return out


def libm_check():
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.exp.restype = libm.log.restype = ctypes.c_double
    libm.exp.argtypes = libm.log.argtypes = [ctypes.c_double]
    rng = np.random.default_rng(7)
    xs = np.concatenate([-rng.random(100000) * 60.0, np.array([0.0, -27.7, -28.0, -745.0])])
    us = rng.integers(1, 2 ** 53, 100000).astype(np.float64) * 2.0 ** -53
    bad = int((np.exp(xs).view(np.uint64) != np.array([libm.exp(float(x)) for x in xs]).view(np.uint64)).sum())
    bad += int((np.log(us).view(np.uint64) != np.array([libm.log(float(u)) for u in us]).view(np.uint64)).sum())
    if bad:
        sys.exit(f"numpy's float64 exp/log differ from libm's on {bad} values here: run on a host without AVX-512F")


def expect_topk(v, position, s):
    width = len(v)
    count = min(width, s.top_k + MARGIN)
    if count < width:
        pick = np.lexsort((np.arange(width), -v))[:count]
        return choose_rows(v[pick][None], pick.astype(np.int64)[None], [position], s)[0]
    return choose_rows(v[None], np.arange(width, dtype=np.int64)[None], [position], s)[0]


def nucleus_libm(v, position, s):
    """nucleus_rows' rule on one whole row with numpy's (libm's) exp for the shares."""

    scaled = v.astype(np.float64) / max(float(s.temperature), 1e-6)
    top = scaled.max()
    mass = np.floor(np.exp(scaled - top) * MASS).astype(np.int64)
    order = np.lexsort((np.arange(len(v)), -scaled))
    sv, sm = scaled[order], mass[order]
    keep = len(sv)
    if 0.0 < s.top_p < 1.0:
        need = math.ceil(Fraction(s.top_p) * int(mass.sum()))
        keep = int((np.cumsum(sm) < need).sum()) + 1
    if s.min_p > 0.0:
        keep = min(keep, int((sv >= top + s.min_log).sum()))
    score = sv[:keep] - np.log(-np.log(uniform(s.seed, position, order[:keep])))
    return int(order[int(np.argmax(score))]), int((mass != torch.floor(torch.exp(torch.from_numpy(scaled - top)) * MASS)
                                                   .to(torch.int64).numpy()).sum())


def zf(x):
    return repr(float(x))


def main(out_path):
    libm_check()
    rng = np.random.default_rng(1020)
    cases, dropped, share_diffs = [], 0, 0
    for at, (name, gen, vocab, bits, shift, offset, bump, peaks) in enumerate(SHAPES):
        v = row(gen, vocab, bits, shift, offset, bump, peaks)
        sets = [(0, p) for p in TOPK0] + [(k, p) for k in WIDE for p in WIDE_P] + [(k, p) for k in NARROW for p in NARROW_P]
        for top_k, (t, top_p, min_p) in sets:
            for _ in range(DRAWS):
                seed = int(rng.integers(0, 2 ** 63))
                position = int(rng.integers(0, 2 ** 20))
                s = Sampling(seed=seed, temperature=t, top_k=top_k, top_p=top_p, min_p=min_p)
                if top_k == 0:
                    want = nucleus_rows(torch.from_numpy(v)[None], [position], s)[0]
                    other, diffs = nucleus_libm(v, position, s)
                    share_diffs += diffs
                    if other != want:
                        dropped += 1
                        continue
                else:
                    want = expect_topk(v, position, s)
                cases.append((at, seed, position, t, top_k, top_p, min_p, int(want)))
        print(f"{name}: {sum(1 for c in cases if c[0] == at)} cases", file=sys.stderr)
    sums = []
    for n in SUMS:
        a = (keys(99, n) >> U(11)).astype(np.float64) * 2.0 ** -53
        one, rows = np.add.reduce(a), a[None].sum(axis=-1, keepdims=True)[0, 0]
        assert one.view(np.uint64) == rows.view(np.uint64), f"np.add.reduce and sum(axis=-1) differ at n {n}"
        sums.append((n, int(np.float64(one).view(np.uint64))))
    ceils = [(p, n, math.ceil(Fraction(p) * n)) for p, n in CEILS]
    lines = [
        "//! Goldens for sampling.zig, written by tools/dsv41-zig/rec/zrec_sampling_cases.py (do not edit): each case's row",
        "//! rebuilt from `shapes` and the token the Python line draws (exact_sampling.choose_rows on sample_rows' candidates,",
        "//! cuda.sampling.nucleus_rows for top_k 0); numpy's sums past its pairwise block and buffer; ceil(Fraction(p) * n).",
        f"//! numpy {np.__version__}, torch {torch.__version__.split('+')[0]}: {len(cases)} cases, {dropped} top_k-off cases dropped where"
        f" torch's and libm's",
        f"//! exp draw apart ({share_diffs} shares one unit apart over all top_k-off cases).",
        "",
        "/// Token i of a row: the top `bits` bits of splitmix64(gen + i * 0x9E3779B97F4A7C15) times 2^-shift, less offset,",
        "/// plus bump at each peak (exact in f32).",
        "pub const Shape = struct { gen: u64, vocab: usize, bits: u6, shift: u11, offset: f64, bump: f64, peaks: []const usize };",
        "pub const Case = struct { shape: usize, seed: u64, position: u64, temperature: f64, top_k: usize, top_p: f64, min_p: f64, want: i64 };",
        "pub const Sum = struct { n: usize, bits: u64 };",
        "pub const Ceil = struct { p: f64, n: u64, want: u64 };",
        "",
        "pub const shapes = [_]Shape{",
    ]
    for name, gen, vocab, bits, shift, offset, bump, peaks in SHAPES:
        pk = ", ".join(str(p) for p in peaks)
        lines.append(f"    .{{ .gen = {gen}, .vocab = {vocab}, .bits = {bits}, .shift = {shift}, .offset = {zf(offset)}, "
                     f".bump = {zf(bump)}, .peaks = &.{{{' ' + pk + ' ' if pk else ''}}} }}, // {name}")
    lines += ["};", "", "pub const cases = [_]Case{"]
    for at, seed, position, t, top_k, top_p, min_p, want in cases:
        lines.append(f"    .{{ .shape = {at}, .seed = 0x{seed:016x}, .position = {position}, .temperature = {zf(t)}, .top_k = {top_k}, "
                     f".top_p = {zf(top_p)}, .min_p = {zf(min_p)}, .want = {want} }},")
    lines += ["};", "", "/// np.add.reduce of n doubles: splitmix64(99 + i * golden)'s top 53 bits times 2^-53.",
              "pub const sums = [_]Sum{"]
    lines += [f"    .{{ .n = {n}, .bits = 0x{b:016x} }}," for n, b in sums]
    lines += ["};", "", "pub const ceils = [_]Ceil{"]
    lines += [f"    .{{ .p = {zf(p)}, .n = {n}, .want = {w} }}," for p, n, w in ceils]
    lines += ["};", ""]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}: {len(cases)} cases, {dropped} dropped, {share_diffs} share differences", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
