"""Do the head's logits keep their bits when the vocabulary splits unevenly over the ranks? Each split's fp32 logits
(torch.mm of bf16 rows and that rank's head rows, out fp32, the engine's call) are compared bit for bit with the even
split's, for decode and prompt row counts. Runs on one GPU with a random head of Beam's shape.
usage: python tools/beam_head_split_check.py [vocab rows of rank 0 ...]"""

import sys

import torch

V, D = 151936, 5120


def logits(y, head, cuts):
    return torch.cat([torch.mm(y, head[lo:hi].t(), out_dtype=torch.float32) for lo, hi in zip(cuts, cuts[1:])], 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    head = (torch.randn(V, D, device="cuda") * 0.02).bfloat16()
    even = [0, V // 2, V]
    firsts = [int(a) for a in sys.argv[1:]] or [44512, 52992, 60000, 64000]
    bad = 0
    for rows in (1, 2, 4, 8, 12, 16, 32, 64, 256, 2048):
        y = torch.randn(rows, D, device="cuda").bfloat16()
        want = logits(y, head, even)
        for f in firsts:
            got = logits(y, head, [0, f, V])
            same = torch.equal(got.view(torch.int32), want.view(torch.int32))
            bad += not same
            if not same:
                print(f"rows {rows} rank-0 rows {f}: {(got != want).sum().item()} logits differ")
    print("HEAD_SPLIT_SAME" if not bad else f"HEAD_SPLIT_DIFFERS in {bad} cases")
