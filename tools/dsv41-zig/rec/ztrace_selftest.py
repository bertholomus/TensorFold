"""ztrace self-test, run with CUDA_INJECTION64_PATH=.../ztrace.so and ZTRACE_FILE set: a torch elementwise op, a
Triton kernel and a cuBLAS GEMM, each under its own phase. Prints one JSON line: for each phase the launches the trace
holds (name, how they were caught, parameters), and whether the device addresses came out as <ptr>."""

import ctypes
import json
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _add1(X, Y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(Y + i, tl.load(X + i, mask=m) + 1, mask=m)


zt = ctypes.CDLL(os.environ["CUDA_INJECTION64_PATH"])
zt.ztrace_phase.argtypes = [ctypes.c_char_p]
x = torch.arange(4096, device="cuda", dtype=torch.float32)
zt.ztrace_phase(b"selftest.torch")
y = x * 2
zt.ztrace_phase(b"selftest.triton")
z = torch.empty_like(x)
_add1[(4,)](x, z, 4096, BLOCK=1024)
zt.ztrace_phase(b"selftest.blas")
a = torch.randn(64, 64, device="cuda")
b = a @ a
torch.cuda.synchronize()
zt.ztrace_phase(b"other")
zt.ztrace_flush()
rows = [json.loads(line) for line in open(os.environ["ZTRACE_FILE"])]
out = {"summary": [r["summary"] for r in rows if "summary" in r][-1:], "phases": {}}
for r in rows:
    if r.get("phase", "").startswith("selftest."):
        out["phases"].setdefault(r["phase"], []).append(
            {"name": r["name"][:80], "via": r["via"], "grid": r["grid"], "block": r["block"], "pdl": r["pdl"],
             "params": r["params"][:160]})
ok = all(any("<ptr>" in e["params"] for e in out["phases"].get(p, [])) for p in ("selftest.torch", "selftest.triton"))
tri = [e for e in out["phases"].get("selftest.triton", []) if e["name"].startswith("_add1")]
ok = ok and bool(tri) and "00100000" in tri[0]["params"]     # n = 4096, little-endian
out["ok"] = ok
print(json.dumps(out))
