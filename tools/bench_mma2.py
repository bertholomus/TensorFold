"""The prompt chunks' grouped expert matmuls alone on one GB10 (GLM-5.3 TP4 rank 0's shapes, random 3-bit mul1
trellis): ms and TFLOPS of each launch (gate/up, down) for grouped_mma, grouped_mma2 and grouped_mma3 (and its down /
combine), on the grouping and rotated rows routed() leaves for a chunk of ROWS rows.

usage (one GPU, tf container): python3 tools/bench_mma2.py [ROWS=2048] [SKEW=0]
"""

from __future__ import annotations

import math
import sys
import time

import torch

from tensorfold.cuda.exl3 import experts as x3experts

E, D, I, BITS, TOPK = 256, 6144, 512, 3, 8


def layer():
    g = torch.Generator(device="cuda").manual_seed(0)

    def mat(k, n):
        t = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * BITS), dtype=torch.int16, device="cuda", generator=g)
        return t, (torch.randn(k, device="cuda", generator=g) * 0.1 + 1).half(), \
            (torch.randn(n, device="cuda", generator=g) * 0.01).half()

    gate, up, down = zip(*[(mat(D, I), mat(D, I), mat(I, D)) for _ in range(E)])
    return x3experts.prepare(list(gate), list(up), list(down), "mul1")


def timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    skew = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    ex = layer()
    g = torch.Generator().manual_seed(1)
    pop = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** skew
    pick = torch.multinomial(pop[torch.randperm(E, generator=g)].expand(R, E), TOPK, replacement=False,
                             generator=g).int()
    pick = torch.cat([pick, torch.full((R, 1), E, dtype=torch.int32)], 1).cuda()
    w = torch.rand((R, TOPK), generator=g) + 0.1
    wts = torch.cat([w / w.sum(1, keepdim=True), torch.ones((R, 1))], 1).cuda()
    x = (torch.randn((R, D), device="cuda") * 0.5).to(torch.bfloat16)
    s = x3experts.Scratch(ex, R, TOPK + 1, prompt=True)
    out = torch.empty((R, D), dtype=torch.float32, device="cuda")
    x3experts.PROMPT_KERNEL = "mma2"
    x3experts.routed(x, pick, wts, ex, s, out, R, limit=math.inf, act_mode=x3experts.ACT_BF16)     # fills the buffers
    ext = x3experts._ext()
    ids = s.ids[:min(R * (TOPK + 1), E)]
    busiest = int(s.counts.max())
    members = s.members_buf[:ids.shape[0] * max(32, -(-busiest // 32) * 32)].view(ids.shape[0], -1)
    P = R * (TOPK + 1)
    live = int(s.counts.sum())
    fl_gu, fl_d = 2 * live * D * I * 2, 2 * live * I * D
    print(f"R={R} skew {skew}: {live} member rows, busiest expert {busiest}", flush=True)
    nt, wgu, skgu, _ = s.cfg_gu
    _, wd, skd, _ = s.cfg_d
    t_gu = timed(lambda: ext.grouped_mma(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count,
                                         members, s.z, 2, D, I, P, skgu, TOPK + 1, ex.cb, wgu, ex.k2_gu[0],
                                         ex.k2_gu[1], 1))
    t_d = timed(lambda: ext.grouped_mma(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count,
                                        members, s.z, 1, I, D, P, skd, TOPK + 1, ex.cb, wd, ex.k2_d[0], ex.k2_d[1],
                                        int(skd > 1)))
    print(f"  grouped_mma: gate/up {t_gu:.2f} ms ({fl_gu / t_gu / 1e9:.1f} TFLOPS), down {t_d:.2f} ms "
          f"({fl_d / t_d / 1e9:.1f} TFLOPS)", flush=True)
    t_gu = timed(lambda: ext.grouped_mma2(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count,
                                          members, s.z, 2, D, I, P, TOPK + 1, ex.cb, ex.k2_gu[0], ex.k2_gu[1]))
    t_d = timed(lambda: ext.grouped_mma2(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count,
                                         members, s.z, 1, I, D, P, TOPK + 1, ex.cb, ex.k2_d[0], ex.k2_d[1]))
    print(f"  grouped_mma2: gate/up {t_gu:.2f} ms ({fl_gu / t_gu / 1e9:.1f} TFLOPS), down {t_d:.2f} ms "
          f"({fl_d / t_d / 1e9:.1f} TFLOPS)", flush=True)
    for nw in (2, 4, 1):
        t_gu = timed(lambda: ext.grouped_mma3(x, x.stride(0), ex.suh_g, ex.suh_u, ex.gate_ptr, ex.up_ptr, ex.gate_k2,
                                              ex.up_k2, ids, s.count, members, s.z, D, I, P, TOPK + 1, ex.cb,
                                              ex.k2_gu[0], ex.k2_gu[1], nw))
        print(f"  grouped_mma3 gate/up (rotation inside), setting {nw}: {t_gu:.2f} ms "
              f"({fl_gu / t_gu / 1e9:.1f} TFLOPS)", flush=True)
    t_rot = timed(lambda: ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, TOPK + 1, E))
    print(f"  (the rot_in mma3 replaces: {t_rot:.2f} ms)", flush=True)
    yb = s.z.view(torch.bfloat16)[4 * P * I:4 * P * I + P * D]
    t_d3 = timed(lambda: ext.grouped_down3(s.xd, ex.down_ptr, ex.down_k2, ids, s.count, members, ex.svh_d, wts, yb, I,
                                           D, P, TOPK + 1, ex.cb, ex.k2_d[0], ex.k2_d[1]))
    t_cy = timed(lambda: ext.combine_y(yb, pick, s.no_y, out, R, D, TOPK + 1, E, 0))
    t_dc = timed(lambda: ext.down_combine(s.z, pick, ex.svh_d, s.no_y, wts, s.no_y, out, R, P, D, 1, TOPK + 1, E, 0,
                                          0))
    print(f"  grouped_down3 (epilogue inside, bf16 slots): {t_d3:.2f} ms ({fl_d / t_d3 / 1e9:.1f} TFLOPS) + combine_y "
          f"{t_cy:.2f} ms; the down_combine it replaces {t_dc:.2f} ms", flush=True)
    # grouped_mma2 with every member row folded onto 32 token rows: the rows' A reads (and Z writes) hit L2
    folded = torch.where(members >= 0, ((members >> 5) % 32) * 32 + (members & 31), members).contiguous()
    t_gu = timed(lambda: ext.grouped_mma2(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count,
                                          folded, s.z, 2, D, I, P, TOPK + 1, ex.cb, ex.k2_gu[0], ex.k2_gu[1]))
    t_d = timed(lambda: ext.grouped_mma2(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count,
                                         folded, s.z, 1, I, D, P, TOPK + 1, ex.cb, ex.k2_d[0], ex.k2_d[1]))
    print(f"  grouped_mma2, rows folded onto 32 (A from L2): gate/up {t_gu:.2f} ms "
          f"({fl_gu / t_gu / 1e9:.1f} TFLOPS), down {t_d:.2f} ms ({fl_d / t_d / 1e9:.1f} TFLOPS)", flush=True)

if __name__ == "__main__":
    main()
