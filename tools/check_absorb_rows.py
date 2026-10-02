"""glm_moe_dsa's prompt-chunk absorb (q . W_UK) and expand (o_lat . W_UV), every way mla_pe can run them, against latent's
kernels: elements whose bits differ (must be 0) and ms a call at a 2,048-row chunk (TP4 shapes: 16 heads, 256 -> 512 ->
256, bf16 weights as latent.AbsorbW holds them). usage (one GPU, tf container): python3 tools/check_absorb_rows.py [R=2048]
"""

from __future__ import annotations

import sys
import time

import torch

from tensorfold.families.glm5_next.cuda import latent
from tensorfold.families.glm_moe_dsa.cuda import mla_pe

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2048


def timed(fn, reps: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main() -> None:
    g = torch.Generator().manual_seed(5)
    H, D, LW, DV = 16, 256, 512, 256
    a = latent.AbsorbW((torch.randn((H, D, LW), generator=g) * 0.05).cuda(),
                       (torch.randn((H, DV, LW), generator=g) * 0.05).cuda())
    q = (torch.randn((R, H, D), generator=g) * 2).to(torch.bfloat16).cuda()
    ol = (torch.randn((R, H, LW), generator=g) * 2).to(torch.bfloat16).cuda()
    qa = torch.empty((R, H, LW), dtype=torch.bfloat16, device="cuda")
    vn = torch.empty((R, H, DV), dtype=torch.bfloat16, device="cuda")
    want_qa = latent.absorb_q(q, a, qa).clone()
    want_v = latent.expand_v(ol, a, vn).clone()
    t_qa = timed(lambda: latent.absorb_q(q, a, qa))
    t_v = timed(lambda: latent.expand_v(ol, a, vn))
    print(f"R={R}: latent absorb {t_qa:.2f} ms, expand {t_v:.2f} ms", flush=True)
    saved = mla_pe.ABSORB
    try:
        for kind in ("triton", "cuda"):
            mla_pe.ABSORB = kind
            got_qa = mla_pe.absorb_q(q, a, qa).clone()
            got_v = mla_pe.expand_v(ol, a, vn).clone()
            bad_qa = int((got_qa.view(torch.int16) != want_qa.view(torch.int16)).sum())
            bad_v = int((got_v.view(torch.int16) != want_v.view(torch.int16)).sum())
            t_qa = timed(lambda: mla_pe.absorb_q(q, a, qa))
            t_v = timed(lambda: mla_pe.expand_v(ol, a, vn))
            print(f"   {kind}: absorb {t_qa:.2f} ms ({bad_qa} of {got_qa.numel()} differ), expand {t_v:.2f} ms "
                  f"({bad_v} of {got_v.numel()} differ)", flush=True)
    finally:
        mla_pe.ABSORB = saved


if __name__ == "__main__":
    main()
