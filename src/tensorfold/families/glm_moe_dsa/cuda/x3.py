"""EXL3 projections for GLM-5.3's non-expert matrices: the row-invariant EXL3 linear in 128-row calls, the prompt GEMM.

ExLlamaV3's GLM-5.3 conversion quantizes q_a, q_b, kv_a, o_proj, the indexer's wq_b, the dense and shared-expert
MLPs, the MTP's eh_proj and the head as EXL3 trellis groups; only kv_b, the indexer's wk / weights_proj / k_norm,
the router and the norms stay plain (fp16 / bf16 / fp32). ``X3`` wraps one group so the family's ``mm`` can
dispatch on it next to the Flash path's ``B16``.

Two shapes differ from the plain weight: an EXL3 group's output width is padded to whole 128-column Hadamard
blocks (kv_a's 576 = 512 latent + 64 rope is stored as 640), so ``crop`` keeps the model's columns; and a head
split over ranks in whole blocks gives ranks different widths, so ``n_pad`` pads every rank to one width with -inf.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

ROWS = 128               # most rows one call of the EXL3 linear takes


class X3Scratch:
    """Shared fp16 input rotations, fp32 K-split partials and staging blocks for every X3 of a model."""

    def __init__(self, users: list, device, *, prefill: bool) -> None:
        from tensorfold.cuda.exl3.prefill import Workspace

        xh = z = tmp = 1
        for u in users:
            lin = u.lin
            xh = max(xh, ROWS * lin.k)
            if lin.split[0] > 1:
                z = max(z, lin.split[0] * ROWS * lin.n)
            tmp = max(tmp, ROWS * lin.n)
        self.xh = torch.empty((xh,), dtype=torch.float16, device=device)
        self.z = torch.empty((z,), dtype=torch.float32, device=device)
        # staging for outputs the linear cannot write in place (cropped widths, non-contiguous slices)
        self.tmp = {torch.float32: torch.empty((tmp,), dtype=torch.float32, device=device),
                    torch.bfloat16: torch.empty((tmp,), dtype=torch.bfloat16, device=device)}
        self.big: dict = {}                  # prompt-chunk staging (cropped groups only), grown on first use
        self.prefill = Workspace() if prefill else None

    def staging(self, dtype: torch.dtype, rows: int, n: int) -> torch.Tensor:
        if rows <= ROWS:
            return self.tmp[dtype][:rows * n].view(rows, n)
        buf = self.big.get(dtype)
        if buf is None or buf.numel() < rows * n:
            buf = self.big[dtype] = torch.empty((rows * n,), dtype=dtype, device=self.xh.device)
        return buf[:rows * n].view(rows, n)

    def nbytes(self) -> int:
        own = [self.xh, self.z, *self.tmp.values(), *self.big.values()]
        return sum(t.numel() * t.element_size() for t in own) + (self.prefill.nbytes() if self.prefill else 0)


@dataclass
class X3:
    """One EXL3 group on this rank: y [R, n] = (x [R, k] @ W)[:, :crop], -inf past it up to ``n_pad``."""

    lin: object              # tensorfold.cuda.exl3.linear.Exl3Linear
    n_pad: int = 0           # the width callers see; 0 = the kept width
    crop: int = 0            # the model's columns of the group's output (<= lin.n); 0 = all of them

    def __post_init__(self) -> None:
        stored = int(self.lin.n)
        self.crop = int(self.crop) or stored
        if not 0 < self.crop <= stored:
            raise ValueError(f"X3: crop {self.crop} outside the group's {stored} columns")
        self.n_pad = int(self.n_pad) or self.crop
        if self.n_pad < self.crop:
            raise ValueError(f"X3: padded width {self.n_pad} below the kept {self.crop}")
        _ = self.lin.counters            # allocate now, never inside a CUDA graph capture

    @property
    def n(self) -> int:
        return self.n_pad

    @property
    def k(self) -> int:
        return int(self.lin.k)

    def nbytes(self) -> int:
        lin = self.lin
        parts = [lin.words, lin.suh, lin.svh, lin.counters] + ([lin.bias] if lin.bias is not None else [])
        return sum(t.numel() * t.element_size() for t in parts)

    def __call__(self, x: torch.Tensor, out: torch.Tensor, sc: X3Scratch, *, prefill: bool = False) -> torch.Tensor:
        """out [R, n] (bf16 or fp32, unit column stride) for x [R, k] bf16; each row's bits never depend on R."""

        lin, R = self.lin, x.shape[0]
        stored, keep = int(lin.n), self.crop
        if x.shape[1] != lin.k or out.shape[0] != R or out.shape[1] != self.n_pad or out.stride(1) != 1:
            raise ValueError(f"X3: x {tuple(x.shape)} / out {tuple(out.shape)} do not match K={lin.k}, N={self.n_pad}")
        if self.n_pad > keep:
            out[:, keep:].fill_(float("-inf"))
        if prefill and sc.prefill is not None and R > ROWS:
            from tensorfold.cuda.exl3.prefill import matmul

            if stored == keep:
                matmul(lin, x, out[:, :keep], sc.prefill)       # the prompt GEMM takes any row stride
            else:
                y = sc.staging(out.dtype, R, stored)
                matmul(lin, x, y, sc.prefill)
                out[:, :keep].copy_(y[:, :keep])
            return out
        sk = lin.split[0]
        for r0 in range(0, R, ROWS):
            r1 = min(R, r0 + ROWS)
            rows = r1 - r0
            dst = out[r0:r1, :keep]
            direct = stored == keep and dst.is_contiguous()
            y = dst if direct else sc.staging(out.dtype, rows, stored)
            lin(x[r0:r1].contiguous(), out=y, xh=sc.xh[:rows * lin.k].view(rows, lin.k),
                z=sc.z[:sk * rows * stored] if sk > 1 else None)
            if not direct:
                dst.copy_(y[:, :keep])
        return out


@dataclass
class X3Pair:
    """Two groups reading the same input written side by side: the dense / shared MLP's [gate | up]."""

    gate: X3
    up: X3

    @property
    def lin(self):                       # scratch sizing reads the wider of the two (they match)
        return self.gate.lin

    @property
    def n(self) -> int:
        return self.gate.n + self.up.n

    @property
    def k(self) -> int:
        return self.gate.k

    def nbytes(self) -> int:
        return self.gate.nbytes() + self.up.nbytes()

    def __call__(self, x: torch.Tensor, out: torch.Tensor, sc: X3Scratch, *, prefill: bool = False) -> torch.Tensor:
        g = self.gate.n
        self.gate(x, out[:, :g], sc, prefill=prefill)
        self.up(x, out[:, g:], sc, prefill=prefill)
        return out


def codebook(parts: dict) -> str:
    """A group's codebook from its marker tensor (ExLlamaV3 stores an empty ``mul1`` / ``mcg`` scalar)."""

    for cb in ("mul1", "mcg"):
        if parts.get(cb) is not None:
            return cb
    return "3inst"


def make(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, cb: str, device, *,
         bias: torch.Tensor | None = None) -> X3:
    from tensorfold.cuda.exl3.linear import Exl3Linear

    lin = Exl3Linear.from_tensors(trellis.contiguous(), suh.to(torch.float16), svh.to(torch.float16), cb,
                                  None if bias is None else bias.to(torch.float16), device=device)
    return X3(lin)


def vocab_slice(blocks: int, world: int, rank: int) -> tuple[int, int, int]:
    """(first 128-column block, blocks on this rank, blocks every rank pads to) of a head split over ``world`` ranks.

    The EXL3 linear works in whole 128-column Hadamard blocks, and GLM-5.3's 154,880-token vocabulary is 1,210
    blocks: 302.5 a rank at TP4. Ranks take ceil(blocks / world) blocks each (the last takes the rest) and pad
    their logits to the same width with -inf, so every rank's slice starts at ``rank * per * 128``.
    """

    per = -(-blocks // world)
    lo = min(rank * per, blocks)
    return lo, min(per, blocks - lo), per
