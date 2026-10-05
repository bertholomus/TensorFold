"""EXL3 linear layers on CUDA (``linear.cu``), any codebook and width: y = x @ W + bias for 1 to 128 rows, row-invariant, no cuBLAS."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from . import format as fmt

CODEBOOK_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_exl3_linear_v6",
                sources=[str(here / "linear.cpp"), str(here / "linear.cu"), str(here / "linear_grouped.cu")],
                extra_include_paths=[str(here)], extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                verbose=False)


def k2_of(bits: float) -> int:
    return int(2 * fmt.check_bits(bits))


def plan(k: int, n: int, blocks: int = 192, min_tiles: int = 8) -> tuple[int, int]:
    """(K splits, warps a program) for a K x N layer: the shape's alone, so a layer keeps one reduction for every row count."""

    kt, nb = k // 16, n // 128
    sk, wk = (1, 8) if nb >= 64 else (1, 4)
    if nb < 64:                                   # a narrow layer needs K splits to fill the SMs at all
        while nb * sk < blocks and sk < 64:
            sk *= 2
    while (wk > 4 or sk > 1) and (kt % (sk * wk) or kt // (sk * wk) < min_tiles):
        if wk > 4:
            wk = 4
        elif sk > 1:
            sk //= 2
        else:
            break
    return sk, wk


def strips(words: torch.Tensor) -> torch.Tensor:
    """Trellis words [K/16, N/16, W] -> [N/128, K/16, 8, W]: each 128-column block's tiles in k order (a copy)."""

    kt, nt, w = words.shape
    return words.view(kt, nt // 8, 8, w).permute(1, 0, 2, 3).contiguous()


@dataclass
class Exl3Linear:
    """One EXL3 layer on the GPU: trellis words (``layout``), fp16 scales, optional fp16 bias."""

    words: torch.Tensor           # int32, [N/128, K/16, 8, 8 * bits] ("strips") or [K/16, N/16, 8 * bits] ("stored")
    suh: torch.Tensor             # fp16 [K]
    svh: torch.Tensor             # fp16 [N]
    bias: torch.Tensor | None     # fp16 [N]
    bits: float
    codebook: str
    k: int
    n: int
    layout: str = "strips"
    split: tuple[int, int] | None = None     # (K splits, warps a program); plan(k, n) when None, fixed thereafter

    @classmethod
    def from_tensors(cls, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: str,
                     bias: torch.Tensor | None = None, device: str | torch.device = "cuda",
                     layout: str = "strips") -> "Exl3Linear":
        """From a group's tensors: trellis int16 [K/16, N/16, 16 * bits], suh/svh fp16 (or packed su/sv sign words), codebook name."""

        bits = fmt.bits_of(trellis.shape)
        if codebook not in CODEBOOK_IDS:
            raise ValueError(f"unknown EXL3 codebook {codebook!r}")
        if not float(bits).is_integer() and codebook != "mul1":
            raise ValueError(f"{bits}-bit EXL3 tiles need the mul1 codebook")
        k, n = 16 * trellis.shape[0], 16 * trellis.shape[1]
        if k % 128 or n % 128:
            raise ValueError(f"K={k} and N={n} must be multiples of 128")

        def scales(t: torch.Tensor, size: int) -> torch.Tensor:
            if t.dtype == torch.int16 and t.numel() * 16 == size:
                t = torch.from_numpy(fmt.unpack_signs(t))
            if t.dtype != torch.float16 or t.numel() != size:
                raise ValueError(f"scales must be fp16 [{size}] or int16 sign words [{size // 16}]")
            return t.reshape(size).to(device).contiguous()

        words = trellis.to(device).contiguous().view(torch.int32)
        if layout == "strips":
            words = strips(words)
        elif layout != "stored":
            raise ValueError(f"layout must be 'strips' or 'stored', got {layout!r}")
        b = None if bias is None else bias.to(device=device, dtype=torch.float16).contiguous()
        return cls(words, scales(suh, k), scales(svh, n), b, bits, codebook, k, n, layout)

    @classmethod
    def load(cls, model_dir: str | Path, prefix: str, device: str | torch.device = "cuda",
             layout: str = "strips") -> "Exl3Linear":
        """Layer ``prefix`` (e.g. "model.layers.0.self_attn.q_proj") of an EXL3 checkpoint folder."""

        from safetensors import safe_open

        root = Path(model_dir)
        index = root / "model.safetensors.index.json"
        if index.exists():
            import json

            weight_map = json.loads(index.read_text())["weight_map"]
            rel = weight_map.get(prefix + ".trellis")
            if rel is None:  # a group outside the index's coverage: find the shard that holds it
                for part in sorted(root.glob("*.safetensors")):
                    with safe_open(str(part), framework="pt") as g:
                        if f"{prefix}.trellis" in set(g.keys()):
                            rel = part.name
                            break
            if rel is None:
                raise ValueError(f"{prefix}.trellis is in no file of {root}")
            file = root / rel
        else:
            file = root / "model.safetensors"
        with safe_open(str(file), framework="pt") as f:
            names = set(f.keys())

            def get(part: str) -> torch.Tensor | None:
                return f.get_tensor(f"{prefix}.{part}") if f"{prefix}.{part}" in names else None

            parts = {p: get(p) for p in fmt.PARTS}
        codebook = "mul1" if parts["mul1"] is not None else "mcg" if parts["mcg"] is not None else "3inst"
        suh = parts["suh"] if parts["suh"] is not None else parts["su"]
        svh = parts["svh"] if parts["svh"] is not None else parts["sv"]
        return cls.from_tensors(parts["trellis"], suh, svh, codebook, parts["bias"], device, layout)

    @property
    def k2(self) -> int:
        return k2_of(self.bits)

    @property
    def strides(self) -> tuple[int, int]:
        """(words between k tiles, words between 128-column blocks) in ``words``."""

        tw = 4 * self.k2
        if self.layout == "strips":
            return 8 * tw, (self.k // 16) * 8 * tw
        return (self.n // 16) * tw, 8 * tw

    def nbytes(self) -> int:
        return self.words.numel() * 4

    def __post_init__(self) -> None:
        if self.split is None:
            self.split = plan(self.k, self.n)

    @property
    def counters(self) -> torch.Tensor:
        c = getattr(self, "_counters", None)
        if c is None:
            c = torch.zeros((8 * (self.n // 128),), dtype=torch.int32, device=self.words.device)
            self._counters = c
        return c

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
                 xh: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        """y [M, N] = x [M, K] @ W + bias for M = 1..128; scratch ``xh`` fp16 [M, K] and ``z`` fp32 (SK > 1) allocated when not given."""

        if x.dim() != 2 or x.shape[1] != self.k or not 1 <= x.shape[0] <= 128:
            raise ValueError(f"x must be [1..128, {self.k}], got {tuple(x.shape)}")
        x = x.contiguous()
        m = x.shape[0]
        if out is None:
            out = torch.empty((m, self.n), dtype=out_dtype or x.dtype, device=x.device)
        sk, wk = self.split
        if xh is None:
            xh = torch.empty((m, self.k), dtype=torch.float16, device=x.device)
        if sk > 1 and z is None:
            z = torch.empty((sk * m * self.n,), dtype=torch.float32, device=x.device)
        sk_stride, nb_stride = self.strides
        ext = _ext()
        ext.rot_in(x, self.suh, xh)
        ext.linear(xh, self.words, sk_stride, nb_stride, self.svh, self.bias, out, z if sk > 1 else None,
                   self.counters, self.k2, CODEBOOK_IDS[self.codebook], sk, wk)
        return out

    def grouped(self, x: torch.Tensor, out: torch.Tensor | None = None,
                out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """The same y as ``__call__`` (bit for bit) through the grouped kernels (``Exl3Group`` of this layer alone):
        x and ``out`` may be row-strided views, and both launches are programmatic dependent launches."""

        if self.split[1] not in (4, 8):                     # a plan the grouped kernels do not take
            return self(x.contiguous(), out=out, out_dtype=out_dtype)
        g = getattr(self, "_group1", None)
        if g is None:
            g = self._group1 = Exl3Group([self])
        return g([x], None if out is None else [out], [out_dtype or x.dtype])[0]

    def unpack(self, out: torch.Tensor | None = None) -> torch.Tensor:
        """W_q [K, N] fp16 decoded on the GPU from either layout, into ``out`` when given."""

        w = out if out is not None else torch.empty((self.k, self.n), dtype=torch.float16, device=self.words.device)
        _ext().unpack(self.words, w, *self.strides, self.k2, CODEBOOK_IDS[self.codebook])
        return w


def _rows(x: torch.Tensor) -> torch.Tensor:
    """x as the grouped kernel reads rows: unit column stride, 16-byte aligned rows a multiple of 4 elements apart."""

    if x.dim() == 2 and x.stride(1) == 1 and x.stride(0) % 4 == 0 and x.data_ptr() % 16 == 0:
        return x
    return x.contiguous()


# grouped launches as programmatic dependent launches (sm_90+): each may start while the kernel before it finishes,
# reading only weights until that kernel is done (TF_EXL3_PDL=0: plain launches); the bits are the same either way
PDL = os.environ.get("TF_EXL3_PDL", "1") != "0"
# TF_EXL3_L2_DISCARD=1 (default): a grouped launch's split-K partials (fp32 Z) are dropped from L2 once the block that
# sums them has read them (discard.global.L2), so they are never written back to DRAM: the partials are dead then, and
# on a decode step the layer's weights streaming through L2 would evict them dirty (~2 MB a 6-row layer). No output
# changes (only DRAM write traffic).
DISCARD = os.environ.get("TF_EXL3_L2_DISCARD", "1") != "0"


class Exl3Group:
    """Up to ``glinear_max()`` EXL3 layers of the same rows in few launches (``linear_grouped.cu``): one ``rot_many``
    launch rotating every layer's input (each its own suh), then one ``glinear`` launch for each (bits, codebook, warps
    a program) set. A layer keeps its own plan, K ranges and reduction order, so every output has the bits of the
    layer's own ``__call__``, whatever shares the launch. Inputs and outputs may be row-strided views (unit column
    stride, 16-byte aligned rows): the slices of one tensor in, column blocks of one tensor out, no copies."""

    def __init__(self, layers: list):
        self.layers = list(layers)
        if len({id(layer) for layer in self.layers}) != len(self.layers):
            raise ValueError("a layer may appear once in a group (its split-K counters are its own)")
        for layer in self.layers:
            if layer.split[1] not in (4, 8):
                raise ValueError(f"grouped launches take 4 or 8 warps a program, not {layer.split[1]}")
        self.cap = int(_ext().glinear_max())
        if len(self.layers) > self.cap:
            raise ValueError(f"at most {self.cap} layers a group")
        self.suh = [layer.suh for layer in self.layers]
        sets: dict = {}
        for i, layer in enumerate(self.layers):
            sets.setdefault((layer.k2, CODEBOOK_IDS[layer.codebook], layer.split[1]), []).append(i)
        self.launches = []
        for (k2, cb, wk), part in sets.items():
            ls = [self.layers[i] for i in part]
            self.launches.append((part, k2, cb, wk, [la.words for la in ls], [la.strides[0] for la in ls],
                                  [la.strides[1] for la in ls], [la.svh for la in ls], [la.bias for la in ls],
                                  [la.counters for la in ls], [la.split[0] for la in ls], [la.n for la in ls]))

    def __call__(self, xs: list, outs: list | None = None, out_dtypes: list | None = None) -> list:
        """ys[i] [M, N_i] = xs[i] [M, K_i] @ W_i + bias_i for M = 1..128 (the same tensor may be every layer's input);
        ``outs`` are written in place when given, else made with ``out_dtypes`` (default: the input's dtype)."""

        if len(xs) != len(self.layers):
            raise ValueError(f"{len(self.layers)} inputs expected, got {len(xs)}")
        m = xs[0].shape[0]
        for x, layer in zip(xs, self.layers):
            if x.dim() != 2 or x.shape[1] != layer.k or x.shape[0] != m or not 1 <= m <= 128:
                raise ValueError(f"x must be [{m} (1..128), {layer.k}], got {tuple(x.shape)}")
        if outs is None:
            outs = [torch.empty((m, layer.n), dtype=(out_dtypes[i] if out_dtypes else None) or xs[i].dtype,
                                device=xs[0].device) for i, layer in enumerate(self.layers)]
        return self.rotated(self.rotate(xs), outs)

    def rotate(self, xs: list) -> list:
        """Every layer's rotated input rows (``buffers``) from xs, one rot_many launch."""

        xh = self.buffers(xs[0].shape[0], xs[0].device)
        _ext().rot_many([_rows(x) for x in xs], self.suh, xh, PDL)
        return xh

    def buffers(self, m: int, dev) -> list:
        """Rotated-input rows for m rows: [m, K_i] fp16 a layer, one buffer, each layer's rows 16-byte aligned."""

        ks = [layer.k for layer in self.layers]
        buf = torch.empty((m * sum(ks),), dtype=torch.float16, device=dev)
        xh, o = [], 0
        for k in ks:                                          # (K % 128)
            xh.append(buf[o:o + m * k].view(m, k))
            o += m * k
        return xh

    def rotated(self, xh: list, outs: list | None = None, out_dtypes: list | None = None, rot: list | None = None,
                rope: tuple | None = None) -> list:
        """The glinear launches on rotated rows ``xh`` (``buffers``; written by rot_many or by a kernel that folds the
        rotation in, the same bits): ys[i] = xh[i] @ W_i + bias_i. ``rot``: a layer's outputs also rotated into the
        next layer's input rows, (suh, rows [M, *] fp16, column offset) or None a layer (the bits rot_many makes of
        y). ``rope``: (cos, sin fp32 [*, rd / 2], positions int64 [M], head dim, rope dim, layer flags): the flagged
        layers' bf16 outputs leave with rope_heads applied (its bits)."""

        m = xh[0].shape[0]
        dev = xh[0].device
        if outs is None:
            outs = [torch.empty((m, layer.n), dtype=(out_dtypes[i] if out_dtypes else None) or torch.float16,
                                device=dev) for i, layer in enumerate(self.layers)]
        ext = _ext()
        for part, k2, cb, wk, words, s_k, s_nb, svh, bias, counters, sks, ns in self.launches:
            zn = sum(sk * m * n for sk, n in zip(sks, ns) if sk > 1)
            z = torch.empty((zn,), dtype=torch.float32, device=dev) if zn else None
            dc = int(DISCARD and z is not None)
            if rope is not None and any(rope[5][i] for i in part):
                rs = [rot[i] for i in part] if rot is not None else [None] * len(part)
                ext.glinear([xh[i] for i in part], words, s_k, s_nb, svh, bias, [outs[i] for i in part], z, counters,
                            sks, k2, cb, wk, PDL, [r[0] if r else None for r in rs], [r[1] if r else None for r in rs],
                            [r[2] if r else 0 for r in rs], [int(bool(rope[5][i])) for i in part], rope[0], rope[1],
                            rope[2], rope[3], rope[4], discard=dc)
            elif rot is not None and any(rot[i] is not None for i in part):
                rs = [rot[i] for i in part]
                ext.glinear([xh[i] for i in part], words, s_k, s_nb, svh, bias, [outs[i] for i in part], z, counters,
                            sks, k2, cb, wk, PDL, [r[0] if r else None for r in rs], [r[1] if r else None for r in rs],
                            [r[2] if r else 0 for r in rs], discard=dc)
            else:
                ext.glinear([xh[i] for i in part], words, s_k, s_nb, svh, bias, [outs[i] for i in part], z, counters,
                            sks, k2, cb, wk, PDL, discard=dc)
        return outs


def unpack_cuda(trellis: torch.Tensor, codebook: str) -> torch.Tensor:
    """W_q [K, N] fp16 from trellis int16 [K/16, N/16, 16 * bits] on the GPU (``decode.cuh``)."""

    bits = fmt.bits_of(trellis.shape)
    words = trellis.cuda().contiguous().view(torch.int32)
    k, n = 16 * trellis.shape[0], 16 * trellis.shape[1]
    w = torch.empty((k, n), dtype=torch.float16, device=words.device)
    tw = 4 * k2_of(bits)
    _ext().unpack(words, w, (n // 16) * tw, 8 * tw, k2_of(bits), CODEBOOK_IDS[codebook])
    return w
