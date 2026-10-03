"""A prompt chunk's sparse attention on one GB10, GLM-5.3 TP4 rank shapes (16 heads, 512-wide latent, 64-wide rope
key, 2,048 selected tokens a row): ms a call over a bf16 latent, over a quantized one dequantized in registers, and
over the quantized one through the fp16 scratch (mla_pe.unpack_q, timed with it), with the compiled kernels'
registers, spills and shared memory.

usage (one GPU, tf container): python3 tools/bench_prompt_sparse.py [ROWS=2048] [CONTEXT=38000] [BITS=5]
"""

from __future__ import annotations

import sys
import time

import torch

H, LW, PW, TOPK = 16, 512, 64, 2048


def timed(fn, reps: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def kernels(fn) -> str:
    """Compiled variants of fn (a triton.jit function): registers, spills, shared bytes."""

    out = []
    for dev_cache in getattr(fn, "device_caches", {}).values():
        cache = dev_cache[0] if isinstance(dev_cache, tuple) else dev_cache
        for k in cache.values():
            out.append(f"regs {k.n_regs} spills {k.n_spills} shared {k.metadata.shared}")
    return "; ".join(out)


def main() -> None:
    from tensorfold.families.glm_moe_dsa.cuda import kvq, mla_pe

    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 38000
    bits = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    g = torch.Generator().manual_seed(0)
    lat = (torch.randn((n, LW), generator=g) * 2).to(torch.bfloat16).cuda()
    pc = (torch.randn((n, PW), generator=g)).to(torch.bfloat16).cuda()
    plane = kvq.KvQ(n, LW, bits, "cuda")
    kvq.write(lat, plane, torch.zeros((1,), dtype=torch.int32, device="cuda"))
    pos = n - R
    tokens = torch.stack([torch.randperm(pos + r, generator=g)[:TOPK].sort().values for r in range(R)])
    tokens = torch.cat([tokens, torch.full((R, 1), -1)], 1).to(torch.int32).cuda()
    counts = torch.full((R,), TOPK, dtype=torch.int32, device="cuda")
    qa = (torch.randn((R, H, LW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    qp = (torch.randn((R, H, PW), generator=g) * 0.05).to(torch.bfloat16).cuda()
    out = torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda")
    scr = torch.empty((n, LW), dtype=torch.float16, device="cuda")
    s = 256 ** -0.5
    print(f"{R} rows, {TOPK} tokens a row, context {n}, latent {bits}-bit groups", flush=True)
    print(f"bf16 latent: {timed(lambda: mla_pe.sparse_attention(qa, qp, lat, pc, tokens, counts, out, s)):.2f} ms",
          flush=True)
    print(f"q{bits} in registers: {timed(lambda: mla_pe.sparse_attention(qa, qp, plane, pc, tokens, counts, out, s)):.2f}"
          " ms", flush=True)

    def scratch():
        mla_pe.unpack_q(plane, n, scr)
        mla_pe.sparse_attention(qa, qp, plane, pc, tokens, counts, out, s, qscratch=scr)

    print(f"q{bits} via the fp16 scratch (unpack included): {timed(scratch):.2f} ms; unpack alone "
          f"{timed(lambda: mla_pe.unpack_q(plane, n, scr)):.2f} ms", flush=True)
    sbf = torch.empty((n, LW), dtype=torch.bfloat16, device="cuda")

    def scratch_bf16():
        mla_pe.unpack_q(plane, n, sbf)
        mla_pe.sparse_attention(qa, qp, plane, pc, tokens, counts, out, s, qscratch=sbf)

    print(f"q{bits} via the bf16 latent-domain scratch (unpack included): {timed(scratch_bf16):.2f} ms; unpack alone "
          f"{timed(lambda: mla_pe.unpack_q(plane, n, sbf)):.2f} ms", flush=True)
    print(f"_sparse_rows_pe variants: {kernels(mla_pe._sparse_rows_pe)}", flush=True)


if __name__ == "__main__":
    main()
