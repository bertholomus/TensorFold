"""Where a Beam rank's memory goes outside torch's allocator: MemAvailable, torch's reserved bytes and CUDA's per-thread
stack limit after each start-up stage, on one GPU with one routed layer (world 1). A stage's hidden cost is its
MemAvailable drop less its growth in torch's reserved bytes.

usage: python -m tools.beam_memprobe --pack DIR [--layer 5] [--rows 2048] [--shrink]"""

from __future__ import annotations

import argparse
import ctypes
import glob

import torch


def avail() -> int:
    with open("/proc/meminfo") as f:
        for row in f:
            if row.startswith("MemAvailable:"):
                return int(row.split()[1]) * 1024
    return 0


def cudart():
    for p in glob.glob("/usr/local/cuda/lib64/libcudart.so*") + ["libcudart.so"]:
        try:
            return ctypes.CDLL(p)
        except OSError:
            continue
    raise OSError("no libcudart")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--shrink", action="store_true", help="set the stack limit back to 1 KiB at the end")
    a = ap.parse_args()
    rt = cudart()
    last = {"a": avail(), "r": 0}

    def stage(name: str) -> None:
        torch.cuda.synchronize() if torch.cuda.is_initialized() else None
        v = ctypes.c_size_t(0)
        if torch.cuda.is_initialized():
            rt.cudaDeviceGetLimit(ctypes.byref(v), 0)
        now, res = avail(), (torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0)
        drop, grow = (last["a"] - now) / 2**20, (res - last["r"]) / 2**20
        print(f"{name:28s} MemAvailable drop {drop:9.1f} MiB, torch reserved +{grow:9.1f} MiB, hidden {drop - grow:8.1f} "
              f"MiB, stack {v.value} B/thread", flush=True)
        last["a"], last["r"] = now, res

    stage("start")
    torch.zeros(1, device="cuda")
    stage("cuda init")
    torch.mm(torch.ones(64, 64, device="cuda", dtype=torch.bfloat16), torch.ones(64, 64, device="cuda",
                                                                                 dtype=torch.bfloat16))
    stage("cublas")
    from tensorfold.families.beam.config import Config
    from tensorfold.families.beam.cuda import cache as C
    from tensorfold.families.beam.cuda import forward as F
    from tensorfold.families.beam.cuda import weights as W
    from pathlib import Path

    dev = torch.device("cuda")
    cfg = Config.read(a.pack)
    pack = W.Pack(Path(a.pack))
    lw = W.layer(pack, cfg, a.layer, 0, 1, dev)
    pack.close()
    stage(f"load layer {a.layer}")
    fake = W.Weights(cfg=cfg, rank=0, world=1, layers={a.layer: lw}, embed=None, head=None, head_lo=0, norm=lw.ln_in)
    work = F.Work(fake, None, rows=a.rows, device=dev)
    T = a.rows
    cache = C.Cache(cfg, 1, pages=-(-(T + 1) // 256), page=256, seqs=1, chunk=T, full_dtype=torch.float8_e4m3fn,
                    device=dev)
    cache.reserve(0, T + 1)
    stage("work + cache")
    x = torch.randn(T, cfg.hidden, device=dev, dtype=torch.float32)
    y = torch.randn(T, cfg.hidden, device=dev, dtype=torch.bfloat16)
    b = C.batch(cache, [(0, 0, T)], dev, chunk=True)
    att = F.attention(lw, y, b, cache, work)
    stage("prompt attention")
    mo = F.mlp(lw, y, work, x)
    stage("prompt routed experts")
    b1 = C.batch(cache, [(0, T, 1)], dev, chunk=False)
    F.attention(lw, y[:1], b1, cache, work)
    stage("decode attention")
    F.mlp(lw, y[:1], work, x[:1])
    stage("decode routed experts")
    if a.shrink:
        rt.cudaDeviceSetLimit(0, ctypes.c_size_t(1024))
        stage("stack limit -> 1 KiB")
        F.mlp(lw, y[:1], work, x[:1])
        F.attention(lw, y[:1], b1, cache, work)
        stage("decode again")
        F.mlp(lw, y, work, x)
        stage("prompt experts again")
    del att, mo


if __name__ == "__main__":
    main()
