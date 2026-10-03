"""exl3.prefill's dequantized mode (matmul's ``deq``): the weight made in the model's basis once a call (_dequant, in
place over the decoded W_q) against format.dequantize's float64 weight, and the plain GEMM on it against the rotated
path (rot_in, the fixed-tile GEMM and its H128 epilogue) within the weight's bf16 / fp16 rounding; the Triton GEMM's
rows never depend on the call's row count."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import linear, prefill

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def _layer(bits: float, kt: int, nt: int, seed: int = 0, bias: bool = False):
    rng = np.random.default_rng(seed)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(kt, nt, fmt.tile_words(bits))).astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(16 * kt) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(16 * nt) * 0.05).astype(np.float16))
    b = torch.from_numpy((rng.standard_normal(16 * nt) * 0.05).astype(np.float16)) if bias else None
    return trellis, suh, svh, linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1", bias=b)


@pytest.mark.parametrize("mode,tol", [("bf16", 4e-3), ("fp16", 6e-4)])
@pytest.mark.parametrize("bits", (2, 3, 4))
def test_the_dequantized_weight_is_formats(mode, tol, bits):
    trellis, suh, svh, layer = _layer(bits, 32, 24, seed=bits)
    ws = prefill.Workspace()
    k, n = layer.k, layer.n
    wq = ws._grow("w", k * n, "cuda")[:k * n].view(k, n)
    prefill._ext().unpack(layer.words, wq, *layer.strides, layer.k2, prefill.CODEBOOK_IDS["mul1"])
    wd = wq.view(torch.bfloat16) if mode == "bf16" else wq
    prefill._dequant[(k // 128, n // 128)](wq, wd, ws.hadamard("cuda"), layer.suh, layer.svh, N=n,
                                           SCALE=prefill.DEQ_SCALE, num_warps=8)
    want = torch.from_numpy(fmt.dequantize(trellis, suh, svh, bits, "mul1"))
    err = ((wd.double().cpu() - want).norm() / want.norm()).item()
    assert err < tol, err


@pytest.mark.parametrize("mode,tol", [("bf16", 5e-3), ("fp16", 1.5e-3)])
@pytest.mark.parametrize("rows", (129, 1000, 4096))
@pytest.mark.parametrize("nt", (32, 128))                      # N = 512 (128-column programs), 2,048 (256)
@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float32))
def test_the_plain_gemm_matches_the_rotated_path(mode, tol, rows, nt, out_dtype):
    _, _, _, layer = _layer(3, 64, nt, seed=nt + rows)
    ws = prefill.Workspace()
    x = (torch.randn((rows, layer.k), device="cuda") * 2).to(torch.bfloat16)
    want = torch.empty((rows, layer.n), dtype=out_dtype, device="cuda")
    got = torch.empty_like(want)
    prefill.matmul(layer, x, want, ws, deq="0")
    prefill.matmul(layer, x, got, ws, deq=mode)
    err = ((got.double() - want.double()).norm() / want.double().norm()).item()
    assert err < tol, err


def test_the_triton_gemm_keeps_each_rows_bits_and_adds_the_bias():
    _, _, _, layer = _layer(4, 64, 128, seed=7, bias=True)
    ws = prefill.Workspace()
    x = (torch.randn((1000, layer.k), device="cuda") * 2).to(torch.bfloat16)
    whole = torch.empty((1000, layer.n), dtype=torch.float32, device="cuda")
    prefill.matmul(layer, x, whole, ws, deq="fp16")
    for lo, hi in ((0, 1), (3, 300), (517, 1000)):
        part = torch.empty((hi - lo, layer.n), dtype=torch.float32, device="cuda")
        prefill.matmul(layer, x[lo:hi], part, ws, deq="fp16")
        assert torch.equal(part.view(torch.int32), whole[lo:hi].view(torch.int32)), (lo, hi)
    rot = torch.empty_like(whole)
    prefill.matmul(layer, x, rot, ws, deq="0")                  # the rotated path adds the bias after its epilogue
    assert ((whole - rot).norm() / rot.norm()).item() < 1.5e-3
