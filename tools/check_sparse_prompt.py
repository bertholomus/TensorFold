"""glm_moe_dsa's sparse MLA for prompt chunks: _sparse_rows_pe (a program a row, chunk partials merged in registers)
against the chunk programs + _merge it replaces, bit for bit, and ms a 2,048-row chunk.

Selections are each row's top-2,048 of random scores over its visible tokens (ascending, as select.py writes them):
"independent" draws every row's scores afresh (the least key reuse between rows), "correlated" adds small per-row
noise to one score vector (neighbouring rows pick mostly the same tokens, as a real prompt's rows do). Rows below the
dense limit have count 0 and must stay untouched.

usage (one GPU, in the tf container): python3 tools/check_sparse_prompt.py [ROWS=2048] [POSITIONS=1900,6000,13000,24000]
  [LAUNCHES=4:3,..]   (the fused kernel's warps:stages to try)
"""

from __future__ import annotations

import sys
import time

import torch

from tensorfold.families.glm_moe_dsa.cuda import mla_pe

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
POSITIONS = [int(p) for p in (sys.argv[2] if len(sys.argv) > 2 else "1900,6000,13000,24000").split(",")]


def selection(pos: int, R: int, topk: int, correlated: bool, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    W = topk + 1
    tokens = torch.full((R, W), -1, dtype=torch.int32, device="cuda")
    counts = torch.zeros((R,), dtype=torch.int32, device="cuda")
    n = pos + R
    base = torch.rand((n,), generator=g, device="cuda")
    for r0 in range(0, R, 256):
        r1 = min(R, r0 + 256)
        noise = torch.rand((r1 - r0, n), generator=g, device="cuda")
        sc = base[None, :] + 0.05 * noise if correlated else noise
        vis = torch.arange(n, device="cuda")[None, :] <= (pos + torch.arange(r0, r1, device="cuda"))[:, None]
        sc = torch.where(vis, sc, -1.0)
        k = min(topk, n)
        pick = torch.topk(sc, k, dim=1).indices.sort(dim=1).values.to(torch.int32)
        visible = pos + torch.arange(r0, r1, device="cuda") + 1
        tokens[r0:r1, :k] = pick
        counts[r0:r1] = torch.where(visible > topk, torch.full_like(visible, k), torch.zeros_like(visible)).to(torch.int32)
    return tokens, counts


def main() -> None:
    g = torch.Generator(device="cuda").manual_seed(7)
    H, LW, PW, topk = 16, 512, 64, 2048
    cap = max(POSITIONS) + ROWS + 64
    cache = (torch.randn((cap, LW), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    pcache = (torch.randn((cap, PW), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    scale = 256 ** -0.5
    fused = mla_pe.FUSED_ROWS
    bad = 0
    for correlated in (False, True):
        for pos in POSITIONS:
            tokens, counts = selection(pos, ROWS, topk, correlated, g)
            qa = (torch.randn((ROWS, H, LW), device="cuda", generator=g) * 2).to(torch.bfloat16)
            qp = (torch.randn((ROWS, H, PW), device="cuda", generator=g) * 2).to(torch.bfloat16)
            junk = torch.randn((ROWS, H, LW), device="cuda", generator=g).to(torch.bfloat16)
            outs, ms = {}, {}
            launches = [tuple(int(v) for v in c.split(":")) for c in (sys.argv[3] if len(sys.argv) > 3 else "4:3").split(",")]
            for label, rows, launch in [("chunks+merge", 1 << 30, launches[0])] + [
                    (f"fused {c[0]}w{c[1]}s", fused, c) for c in launches]:
                mla_pe.FUSED_ROWS = rows
                mla_pe.FUSED_LAUNCH = launch
                out = junk.clone()
                mla_pe.sparse_attention(qa, qp, cache, pcache, tokens, counts, out, scale)
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(5):
                    mla_pe.sparse_attention(qa, qp, cache, pcache, tokens, counts, out, scale)
                torch.cuda.synchronize()
                ms[label] = (time.perf_counter() - t) / 5 * 1e3
                outs[label] = out
            mla_pe.FUSED_ROWS = fused
            line = f"{'correlated' if correlated else 'independent'} pos {pos}: {int((counts > 0).sum())} sparse rows; "
            line += f"chunks+merge {ms['chunks+merge']:.2f} ms"
            for k in outs:
                if k == "chunks+merge":
                    continue
                same = torch.equal(outs[k].view(torch.int16), outs["chunks+merge"].view(torch.int16))
                bad += not same
                line += f", {k} {ms[k]:.2f} ms{'' if same else ' DIFF'}"
            print(line, flush=True)
    print(f"{bad} cases differ", flush=True)


if __name__ == "__main__":
    main()
