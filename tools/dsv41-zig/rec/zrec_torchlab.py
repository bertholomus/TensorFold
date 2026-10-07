"""The Zig port's torch-op kernels (zig/kernels/cuda/dsv41_torch.cu) against the served build's own torch code on this
GPU, byte for byte: ops.rms_norm (128 and 512 wide, 1 to 2,048 rows), _compress's ratio-2 softmax with the weighted
sum, and ops.rope_ (forward and inverse); with OPS_FATBIN (dsv41_ops.cu), tf_ds_topk_i64 against torch.topk's values
(as a set: _topk_finish sorts them). One JSON line a test and a summary.

  python zrec_torchlab.py FATBIN MODEL_DIR [OPS_FATBIN]
"""

import ctypes
import json
import sys

import numpy as np
import torch

from tensorfold.families.deepseek_v41.config import Cfg
from tensorfold.families.deepseek_v41.ops import rms_norm, rope_

cu = ctypes.CDLL("libcuda.so.1")


def ok(r: int, what: str) -> None:
    if r != 0:
        raise RuntimeError(f"{what}: CUresult {r}")


def load(path: str, names=("tf_ds_rms_norm_kernel", "tf_ds_compress2_kernel", "tf_ds_rope_kernel")):
    torch.zeros(1, device="cuda")            # torch's primary context, current on this thread
    mod = ctypes.c_void_p()
    img = open(path, "rb").read()
    ok(cu.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(img)), "cuModuleLoadData")
    out = {}
    for name in names:
        f = ctypes.c_void_p()
        ok(cu.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), name)
        out[name] = f
    return out


def launch(f, grid, block, smem, args) -> None:
    holders = list(args)
    arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in holders])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    ok(cu.cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem, stream, arr, None),
       "cuLaunchKernel")


def ptr(t: torch.Tensor) -> ctypes.c_uint64:
    return ctypes.c_uint64(t.data_ptr())


def last_pow2(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def rms_lanes(d: int, rows: int) -> int:
    """Reduce.cuh's block width for a contiguous last-dim float mean (vectorized: d / 4), 512 threads at most."""

    dim0, dim1 = d // 4, rows
    d0 = last_pow2(dim0) if dim0 < 512 else 512
    d1 = last_pow2(dim1) if dim1 < 512 else 512
    bw = min(d0, 32)
    bh = min(d1, 512 // bw)
    return min(d0, 512 // bh)


def differ(a: torch.Tensor, b: torch.Tensor) -> int:
    ai = a.contiguous().view(torch.int16 if a.element_size() == 2 else torch.int32)
    bi = b.contiguous().view(torch.int16 if b.element_size() == 2 else torch.int32)
    return int((ai != bi).sum())


def main() -> None:
    k = load(sys.argv[1])
    eps = float(Cfg.read(sys.argv[2]).eps)
    g = torch.Generator(device="cuda").manual_seed(20261007)
    lines, bad = [], 0

    def report(**kw):
        nonlocal bad
        bad += int(kw.get("differ", 0) > 0)
        lines.append(kw)
        print(json.dumps(kw), flush=True)

    # ops.rms_norm
    for d in (128, 512):
        for rows in (1, 2, 3, 4, 5, 8, 15, 16, 17, 31, 64, 127, 512, 1024, 1025, 2048):
            for scale in (1.0, 1e-3, 40.0):
                x = (torch.randn(rows, d, device="cuda", generator=g) * scale).to(torch.bfloat16)
                if scale == 40.0:
                    x[:, ::7] = (x[:, ::7].float() * 1e-4).to(torch.bfloat16)
                w = (torch.randn(d, device="cuda", generator=g) * 0.5 + 1.0).to(torch.bfloat16)
                ref = rms_norm(x, w, eps)
                out = torch.empty_like(x)
                lanes = rms_lanes(d, rows)
                h = max(1, min(512 // lanes, 64))
                factor = float(np.float32(rows) / np.float32(rows * d))
                launch(k["tf_ds_rms_norm_kernel"], ((rows + h - 1) // h, 1, 1), (lanes, h, 1),
                       lanes * h * 4 if lanes > 32 else 0,
                       [ptr(x), ctypes.c_longlong(d), ptr(w), ptr(out), ctypes.c_longlong(d), ctypes.c_int(rows),
                        ctypes.c_int(d), ctypes.c_float(factor), ctypes.c_float(eps)])
                torch.cuda.synchronize()
                report(op="rms_norm", d=d, rows=rows, scale=scale, lanes=lanes, differ=differ(out, ref))

    # _compress at ratio 2: (kv * score.softmax(dim=1)).sum(1).to(bf16)
    c = 512
    for groups in (1, 2, 7, 64, 1024):
        for case in ("plain", "wide", "ties", "zeros"):
            kv = torch.randn(groups, 2, c, device="cuda", generator=g)
            sc = torch.randn(groups, 2, c, device="cuda", generator=g)
            if case == "wide":
                sc = sc * 40.0
                kv = kv * 1e3
            elif case == "ties":
                sc[:, 1] = sc[:, 0]
            elif case == "zeros":
                kv[:, 0, ::3] = 0.0
                kv[:, 1, ::3] = -0.0
                kv[:, 0, 1::5] = -0.0
            ref = (kv * sc.softmax(dim=1)).sum(1).to(torch.bfloat16)
            out = torch.empty((groups, c), dtype=torch.bfloat16, device="cuda")
            n = groups * c
            launch(k["tf_ds_compress2_kernel"], (min(4096, (n + 255) // 256), 1, 1), (256, 1, 1), 0,
                   [ptr(kv), ptr(sc), ptr(out), ctypes.c_int(groups), ctypes.c_int(c)])
            torch.cuda.synchronize()
            report(op="compress2", groups=groups, case=case, differ=differ(out, ref))

    # ops.rope_ on the last 64 dims, rows' rotations from fp32 cos / sin tables at row positions
    half = 32
    table = 4096
    ang = torch.rand(table, half, device="cuda", generator=g) * 6.2831853
    cos, sin = torch.cos(ang).contiguous(), torch.sin(ang).contiguous()
    rope_bad = 0
    for width in (128, 512):
        for rows in (1, 3, 1024):
            for inverse in (False, True):
                x = torch.randn(rows, width, device="cuda", generator=g).to(torch.bfloat16)
                pos = torch.randint(0, table, (rows,), device="cuda", generator=g)
                f = torch.complex(cos[pos], sin[pos])
                ref = x.clone()
                rope_(ref[..., -2 * half:], f, inverse=inverse)
                out = x.clone()
                n = rows * half
                launch(k["tf_ds_rope_kernel"], (min(4096, (n + 255) // 256), 1, 1), (256, 1, 1), 0,
                       [ptr(out), ctypes.c_longlong(width), ctypes.c_int(width - 2 * half), ptr(cos), ptr(sin),
                        ptr(pos), ctypes.c_int(rows), ctypes.c_int(half), ctypes.c_int(int(inverse))])
                torch.cuda.synchronize()
                report(op="rope", width=width, rows=rows, inverse=inverse, differ=differ(out, ref))
    # torch's keys.topk(k).values of unique int64 keys (the indexer's: score bits above, an index below), as a set
    if len(sys.argv) > 3:
        t = load(sys.argv[3], ("tf_ds_topk_i64_kernel",))["tf_ds_topk_i64_kernel"]
        for rows, n, k, ks, case in ((1, 1114, 512, 1114, "unique"), (181, 1114, 512, 1114, "unique"),
                                     (181, 1025, 512, 1025, "unique"), (64, 4097, 512, 4097, "unique"),
                                     (3, 600, 7, 640, "unique"), (2, 513, 512, 513, "unique"), (5, 2000, 1, 2000, "unique"),
                                     (181, 1114, 512, 1114, "ties"), (181, 1114, 512, 1114, "dups"),
                                     (16, 1114, 512, 1114, "masked")):
            hi = torch.randint(-2**31, 2**31 - 1, (rows, ks), device="cuda", generator=g, dtype=torch.int64)
            if case in ("ties", "dups"):
                hi = hi % 5 - 2                                # a few high words: long runs of equal leading digits
            lo = torch.randperm(2**20, device="cuda", generator=g)[:ks].to(torch.int64).expand(rows, ks)
            keys = (hi << 32) | (2**31 - 1 - lo)
            if case == "dups":
                keys[:, 1::3] = keys[:, 0:ks - 1:3][:, :keys[:, 1::3].shape[1]]   # repeated values (a multiset)
            if case == "masked":
                keys[:, n // 3:] = torch.iinfo(torch.int64).min + torch.arange(ks - n // 3, device="cuda")
            view = keys[:, :n]
            ref = view.topk(k, dim=1, sorted=False).values.sort(dim=1).values
            out = torch.full((rows, k), 7, dtype=torch.int64, device="cuda")
            launch(t, (rows, 1, 1), (1024, 1, 1), 0,
                   [ptr(keys), ctypes.c_longlong(ks), ctypes.c_int(n), ctypes.c_int(k), ptr(out)])
            torch.cuda.synchronize()
            got = out.sort(dim=1).values
            report(op="topk_i64", rows=rows, n=n, k=k, ks=ks, case=case, differ=int((got != ref).sum()))
    print(json.dumps({"summary": {"tests": len(lines), "failed": bad, "eps": eps}}), flush=True)


if __name__ == "__main__":
    main()
