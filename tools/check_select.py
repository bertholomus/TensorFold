"""select.select_tokens after row blocking and 16-row scoring programs vs the one-shot RB=1 selection it replaced: tokens and
counts equal for decode windows and prompt chunks past the dense limit. usage (one GPU): python3 tools/check_select.py"""

import time

import torch
import triton

from tensorfold.families.glm_moe_dsa.cuda import select


def old_select(qi, wts, keys, pos, R, topk, pos_dev, tokens, counts, bucket=None):
    H = wts.shape[1]
    D = qi.shape[1] // H
    np_max = bucket if bucket is not None else select.sparse_bucket(int(pos), R)
    np_max = min(np_max, keys.shape[0])
    scores = torch.empty((R, np_max), dtype=torch.float32, device=qi.device)
    select._scores[(R, triton.cdiv(np_max, 64))](qi, wts, wts.stride(0), keys, scores, pos_dev, R, np_max, D ** -0.5,
                                                 H ** -0.5, H=H, HP=max(16, triton.next_power_of_2(H)), D=D, BT=64,
                                                 RB=1, num_warps=4)
    width = tokens.shape[1]
    k = min(topk, np_max)
    picked = torch.topk(select._order_key(scores), k, dim=1, sorted=False).indices
    picked = torch.sort(picked, dim=1).values
    tokens.zero_()
    tokens[:, :k] = picked.to(torch.int32)
    if width > k:
        tokens[:, k:] = -1
    q = pos_dev.to(torch.int64) + torch.arange(R, device=qi.device)
    counts.copy_(torch.where(q + 1 > topk, tokens.ne(-1).sum(1).clamp(max=width), torch.zeros_like(counts)))


def main():
    torch.manual_seed(0)
    dev = "cuda"
    H, D, topk = 32, 128, 2048
    cap = 300000
    keys = torch.randn((cap, D), device=dev).to(torch.bfloat16)
    bad = n = 0
    for pos, R in ((2048, 1), (2050, 4), (6677, 6), (30000, 3), (2048, 2048), (14000, 2048), (60000, 2048),
                   (130000, 2048), (250000, 2048), (4096, 300), (6677, 1), (30000, 1), (130000, 1), (130000, 4),
                   (250000, 1), (250000, 4), (250000, 6), (40000, 15)):
        qi = torch.randn((R, H * D), device=dev).to(torch.bfloat16)
        wts = torch.randn((R, H), device=dev).to(torch.bfloat16)
        pos_dev = torch.tensor([pos], dtype=torch.int32, device=dev)
        ta = torch.empty((R, topk + 1), dtype=torch.int32, device=dev)
        tb = torch.empty_like(ta)
        ca = torch.empty((R,), dtype=torch.int32, device=dev)
        cb = torch.empty_like(ca)
        for _ in range(2):                              # warm (compiles), then time the second call
            torch.cuda.synchronize()
            t = time.perf_counter()
            old_select(qi, wts, keys, pos, R, topk, pos_dev, ta, ca)
            torch.cuda.synchronize()
            t_old = time.perf_counter() - t
            select.select_tokens(qi, wts, keys, pos, R, topk, pos_dev, tokens=tb, counts=cb)
            torch.cuda.synchronize()
        before = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        select.select_tokens(qi, wts, keys, pos, R, topk, pos_dev, tokens=tb, counts=cb)
        torch.cuda.synchronize()
        t_new = time.perf_counter() - t
        peak = torch.cuda.max_memory_allocated() / 2**20
        same = torch.equal(ta, tb) and torch.equal(ca, cb)
        n += 1
        bad += not same
        print(f"pos {pos:6d} R {R:4d}: same {same}; old {t_old * 1e3:8.1f} ms, new {t_new * 1e3:8.1f} ms "
              f"(peak {peak:.0f} MiB)", flush=True)
        del qi, wts, ta, tb
        torch.cuda.empty_cache()
    print(f"select: {n} cases, {bad} differ", flush=True)


if __name__ == "__main__":
    main()
