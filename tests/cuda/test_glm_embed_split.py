"""GLM-5.3's embedding split by vocabulary (weights.embed_span, forward.embed): every rank's glue.embed_span rows,
gathered in rank order and picked by glue.embed_pick, are the whole table's rows bit for bit (every bf16 pattern:
signed zeros, NaN payloads, infinities), at TP2 to TP8 and on vocabularies with a partial last block or fewer blocks
than ranks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@pytest.fixture(autouse=True)
def _split(monkeypatch):
    monkeypatch.delenv("TF_GLM_EMBED_SPLIT", raising=False)


def _table(vocab: int, dims: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(-32768, 32768, (vocab, dims), generator=g, dtype=torch.int32).to(torch.int16)
    return bits.view(torch.bfloat16).cuda()


def _rows(R: int, dims: int) -> torch.Tensor:
    return torch.empty((R, dims), dtype=torch.bfloat16, device="cuda")


def _sends(ids: torch.Tensor, table: torch.Tensor, spans) -> list[torch.Tensor]:
    """What each rank sends: the rows of the tokens it holds, zeros for the others (forward.embed's first step)."""

    from tensorfold.families.glm_moe_dsa.cuda import glue

    out = []
    for first, n in spans:
        rows = _rows(ids.shape[0], table.shape[1])
        out.append(glue.embed_span(ids, table[first:first + n].contiguous(), first, rows) if n else rows.zero_())
    return out


@pytest.mark.parametrize("vocab,world", [(154880, 4), (154880, 2), (154880, 6), (5000, 8), (300, 4), (129, 2)])
def test_split_rows_equal_the_whole_tables(vocab, world):
    from tensorfold.families.glm5_next.cuda import glue as fglue
    from tensorfold.families.glm_moe_dsa.cuda import glue
    from tensorfold.families.glm_moe_dsa.cuda.weights import embed_span

    dims = 6144 if vocab == 154880 else 256
    table = _table(vocab, dims, vocab + world)
    spans = [embed_span(vocab, world, r) for r in range(world)]
    edges = sorted({t for first, n in spans for t in (first - 1, first, first + n - 1, first + n) if 0 <= t < vocab})
    g = torch.Generator().manual_seed(world)
    for R in (1, 4, 8, 300):
        picks = (edges + torch.randint(0, vocab, (R,), generator=g).tolist())[:R]
        ids = torch.tensor(picks, dtype=torch.int32, device="cuda")
        want = fglue.embed(ids, table, dims, 1, _rows(R, dims))
        got = glue.embed_pick(ids, torch.stack(_sends(ids, table, spans)), _rows(R, dims), vocab)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (vocab, world, R)


class _Gather:
    """A one-process stand-in for the ranks' all_gather: every rank's send, computed up front, in rank order."""

    def __init__(self, sends: list[torch.Tensor], rank: int) -> None:
        self.sends, self.rank = sends, rank

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        assert send.dtype == torch.float32                           # bf16 pairs as words: the RDMA path's dtype
        assert torch.equal(send.view(torch.int32), self.sends[self.rank].view(torch.int32).view(-1))
        recv.copy_(torch.stack(self.sends).view(torch.float32).view(-1))


@pytest.mark.parametrize("R", (1, 4, 8))
def test_forward_embed_takes_each_row_from_its_holder(R):
    from tensorfold.families.glm5_next.cuda import glue as fglue
    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd
    from tensorfold.families.glm_moe_dsa.cuda.weights import embed_span

    vocab, dims, world = 154880, 6144, 4
    table = _table(vocab, dims, R)
    ids = torch.tensor([154879, 0, 38784, 116223, 77567, 77568, 38783, 116224][:R], dtype=torch.int32, device="cuda")
    want = fglue.embed(ids, table, dims, 1, _rows(R, dims))
    spans = [embed_span(vocab, world, r) for r in range(world)]
    sends = _sends(ids, table, spans)
    for rank, (first, n) in enumerate(spans):
        w = SimpleNamespace(embed=table[first:first + n].contiguous(), embed_lo=first, world=world,
                            cfg=SimpleNamespace(vocab=vocab), comm=_Gather(sends, rank))
        b = SimpleNamespace(gath=torch.full((world * 8 * dims,), float("nan"), device="cuda"))
        out = _rows(8, dims)[:R]
        fwd.embed(w, b, ids, out)
        assert torch.equal(out.view(torch.int16), want.view(torch.int16)), (R, rank)
