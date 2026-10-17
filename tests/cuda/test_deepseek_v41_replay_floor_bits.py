"""TF_DS_REPLAY_FLOOR: a chunk the floor sends to the gather path runs its attention as one split, so its rows get the
bits the same rows get inside a longer prompt chunk (one split, the prompt path), not the split decode variant's."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from tensorfold.families.deepseek_v41.cuda import kernels as K

H, HD, W = 32, 512, 128                         # one rank's heads (of 64, two ranks), the head dim, the window


@pytest.mark.parametrize("packed", [False, True])
def test_a_floor_chunk_gets_a_longer_chunks_bits(packed):
    g = torch.Generator(device="cuda").manual_seed(0)
    floor, n_long, n_short, picks, n_comp = 1000, 48, 8, 40, 300
    start = floor + 60                          # the longer chunk's first position; every window reaches the floor
    pos = torch.arange(start, start + n_long, device="cuda")
    keys = torch.randn(start + n_long - floor, HD, generator=g, device="cuda").to(torch.bfloat16)  # from the floor
    wlo = torch.tensor([floor], dtype=torch.int64, device="cuda")
    q = torch.randn(n_long, H, HD, generator=g, device="cuda").to(torch.bfloat16)
    sink = torch.randn(H, generator=g, device="cuda")
    if packed:                                  # FP4 codes with E4M3 scales of 1.0
        comp = (torch.randint(0, 256, (n_comp, HD // 2), generator=g, device="cuda").to(torch.uint8),
                torch.full((n_comp, HD // 16), 0x38, dtype=torch.uint8, device="cuda"))
    else:
        comp = torch.randn(n_comp, HD, generator=g, device="cuda").to(torch.bfloat16)
    idx = torch.randint(0, n_comp, (n_long, picks), generator=g, device="cuda")
    idx[:, picks - 5:] = -1                     # a few empty picks (40 is not a multiple of 16: the padding too)
    scale = HD ** -0.5
    longer = K.sparse_attn(q, sink, keys, wlo, False, comp, idx, pos, scale, W)
    s = slice(n_long - n_short, n_long)         # its last rows as a chunk of their own (DECODE_ROWS rows or fewer)
    alone = K.sparse_attn(q[s].contiguous(), sink, keys, wlo, False, comp, idx[s].contiguous(), pos[s].contiguous(),
                          scale, W, one_split=True)
    assert torch.equal(alone.view(torch.int16), longer[s].view(torch.int16))
