"""One GLM-5.3 TP4 MoE layer's routed experts on a prompt chunk, one GB10: ms a call and the kernels inside, for the
shipped prompt path and the decode-once one, with their outputs bit-compared.

Random 3-bit mul1 trellis words at rank 0's shapes (256 experts, 6144 -> 512 gate/up, 512 -> 6144 down; the decode is
data-independent). Each row picks 8 distinct experts by a Zipf popularity (exponent SKEW: 0 is uniform) plus the shared
slot (id E), as glue.select lays them out.

usage (one GPU, tf container): python3 tools/bench_prompt_experts.py [ROWS=2048,4096] [SKEW=0,0.6,1.2] [TILES=4:2:4,..]
(TILES: decode-once settings to try: grouped_rows' n tiles:tiles in flight:member tiles a program, for gate/up and
down alike or "gate/up+down", mma for grouped_mma, mma2 for grouped_mma2: other bits, so its relative error is shown)
"""

from __future__ import annotations

import math
import sys
import time

import torch

from tensorfold.cuda.exl3 import experts as x3experts

E, D, I, BITS, TOPK = 256, 6144, 512, 3, 8


def layer() -> x3experts.Exl3RoutedExperts:
    g = torch.Generator(device="cuda").manual_seed(0)

    def mat(k: int, n: int):
        t = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * BITS), dtype=torch.int16, device="cuda", generator=g)
        suh = (torch.randn(k, device="cuda", generator=g) * 0.1 + 1).half()
        svh = (torch.randn(n, device="cuda", generator=g) * 0.01).half()
        return t, suh, svh

    gate, up, down = zip(*[(mat(D, I), mat(D, I), mat(I, D)) for _ in range(E)])
    return x3experts.prepare(list(gate), list(up), list(down), "mul1")


def routing(R: int, skew: float, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    pop = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** skew
    pop = pop[torch.randperm(E, generator=g)]
    pick = torch.multinomial(pop.expand(R, E), TOPK, replacement=False, generator=g).int()
    pick = torch.cat([pick, torch.full((R, 1), E, dtype=torch.int32)], 1).cuda()
    w = torch.rand((R, TOPK), generator=g, dtype=torch.float32) + 0.1
    wts = torch.cat([w / w.sum(1, keepdim=True), torch.ones((R, 1))], 1).cuda()
    return pick, wts


def timed(fn, reps: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def kernels(fn) -> str:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    times: dict[str, float] = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            name = ev.name.replace("(anonymous namespace)::", "").replace("void ", "").split("<")[0].split("(")[0][:40]
            times[name] = times.get(name, 0.0) + ev.time_range.elapsed_us() / 1e3
    return ", ".join(f"{k} {v:.2f}" for k, v in sorted(times.items(), key=lambda kv: -kv[1])[:6])


def main() -> None:
    rows = [int(r) for r in (sys.argv[1] if len(sys.argv) > 1 else "2048,4096").split(",")]
    skews = [float(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,0.6,1.2").split(",")]
    ex = layer()
    x = (torch.randn((max(rows), D), device="cuda") * 0.5).to(torch.bfloat16)
    tb = int(ex.trellis_bytes.sum())
    print(f"layer: {E} experts, {tb / 2**20:.0f} MiB of trellis on this rank", flush=True)
    # a setting is "nt:pf:g" for both projections or "gate/up+down" (grouped_rows), or "mma" (grouped_mma)
    tiles = [(t,) if t in ("mma", "mma2") else
             [tuple(int(v) for v in p.split(":")) for p in (t.split("+") * 2)[:2]]
             for t in (sys.argv[3] if len(sys.argv) > 3 else "8:1:2,mma").split(",")]
    for R in rows:
        s = x3experts.Scratch(ex, R, TOPK + 1)
        for skew in skews:
            pick, wts = routing(R, skew)
            cnt = torch.bincount(pick[:, :TOPK].reshape(-1).long(), minlength=E)
            outs = {}
            line = f"R={R} skew {skew}: busiest expert {int(cnt.max())} rows (mean {R * TOPK / E:.0f})"
            for name, fast, tile in [("per-tile", False, None)] + [(f"decode-once {t}", True, t) for t in tiles]:
                out = torch.empty((R, D), dtype=torch.float32, device="cuda")
                if tile is not None and tile[0] in ("mma", "mma2"):
                    x3experts.PROMPT_KERNEL = tile[0]
                elif tile is not None:
                    x3experts.PROMPT_KERNEL = "rows"
                    x3experts.PROMPT_TILES = {"gateup": tile[0], "down": tile[1]}

                def call():
                    if fast or R <= 2048:
                        x3experts.routed(x[:R], pick, wts, ex, s, out, R, limit=math.inf,
                                         act_mode=x3experts.ACT_BF16, prompt=fast)
                        return
                    for r0 in range(0, R, 2048):        # the one-tile grouping holds 2,048 rows: rows are independent
                        n = min(2048, R - r0)
                        x3experts.routed(x[r0:r0 + n], pick[r0:r0 + n], wts[r0:r0 + n], ex, s, out[r0:r0 + n], n,
                                         limit=math.inf, act_mode=x3experts.ACT_BF16, prompt=False)

                try:
                    ms = timed(call)
                except Exception as err:                      # noqa: BLE001  (a setting the launcher refuses)
                    line += f"\n   {name}: {err}"
                    continue
                ref = next(iter(outs.values()), None)
                same = None if ref is None else torch.equal(out.view(torch.int32), ref.view(torch.int32))
                err = None if ref is None else float((out.double() - ref.double()).norm() / ref.double().norm())
                outs[name] = out.clone()
                tf = 2 * R * TOPK * 3 * D * I / ms / 1e9
                line += (f"\n   {name}: {ms:.2f} ms ({tb / ms / 1e6:.0f} GB/s of trellis, {tf:.1f} TFLOPS), bit-equal "
                         f"{same}{'' if err is None else f' (rel. error {err:.2e})'}; kernels ms: {kernels(call)}")
            print(line, flush=True)
        del s


if __name__ == "__main__":
    main()
