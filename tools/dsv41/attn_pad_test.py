"""Prompt-chunk sparse attention: pick blocks rounded up to a power of two give the same bits as the exact block
count (masked blocks leave the online softmax unchanged). One GPU.

  python3 attn_pad_test.py
"""

import json

import torch
import triton

from tensorfold.families.deepseek_v41.cuda import kernels as K


def run(q, wsrc, wlo, codes, scales, idx, pos, sink, n_idx, nblk, window=128, bn=32, hb=16):
    rows, h, hd = q.shape
    out = torch.empty_like(q)
    K._sparse_attn_part[(rows, h // hb, 1)](q, wsrc, wlo, codes, scales, idx, pos, out, out, out, sink, out,
                                            hd ** -0.5, wsrc.shape[0], n_idx, H=h, HD=hd, HB=hb, WIN=window, BN=bn,
                                            RING=False, HAS_COMP=True, PACKED=True, SPLITS=1, NBLK=nblk, FINAL=True,
                                            num_warps=4, num_stages=1)
    return out


def bits(x, y):
    return bool(torch.equal(x.view(torch.int16), y.view(torch.int16)))


def main():
    torch.manual_seed(0)
    dev = "cuda"
    h, hd, window, bn = 32, 512, 128, 32
    res = []
    for rows, n_idx in [(40, 17), (64, 33), (100, 65), (130, 129), (257, 257), (300, 300), (77, 500)]:
        q = torch.randn((rows, h, hd), device=dev).to(torch.bfloat16)
        wsrc = torch.randn((rows, hd), device=dev).to(torch.bfloat16)
        wlo = torch.zeros((1,), dtype=torch.int64, device=dev)
        pos = torch.arange(rows, dtype=torch.int64, device=dev)
        n_comp = 2 * n_idx + 8
        codes = torch.randint(0, 256, (n_comp, hd // 2), dtype=torch.uint8, device=dev)
        scales = torch.randint(100, 126, (n_comp, hd // 16), dtype=torch.uint8, device=dev)
        idx = torch.stack([torch.randperm(n_comp, device=dev)[:n_idx].sort().values for _ in range(rows)])
        idx[: rows // 3, n_idx // 2:] = -1
        sink = torch.randn((h,), device=dev)
        picks = triton.cdiv(n_idx, bn)
        a = run(q, wsrc, wlo, codes, scales, idx.contiguous(), pos, sink, n_idx, window // bn + picks)
        b = run(q, wsrc, wlo, codes, scales, idx.contiguous(), pos, sink, n_idx,
                window // bn + triton.next_power_of_2(picks))
        w = K.sparse_attn(q, sink, wsrc, wlo, False, (codes, scales), idx.contiguous(), pos, hd ** -0.5, window)
        res.append({"rows": rows, "n_idx": n_idx, "picks": picks, "padded": triton.next_power_of_2(picks),
                    "equal": bits(a, b), "wrapper_equal": bits(a, w), "nan": bool(a.isnan().any())})
        print(json.dumps(res[-1]), flush=True)
    print(json.dumps({"all_equal": all(r["equal"] and r["wrapper_equal"] for r in res)}))


if __name__ == "__main__":
    main()
