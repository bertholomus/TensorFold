"""The prefill-regression fix commits (fix plan F4-F6), host side on CPU."""

from __future__ import annotations

import os
import subprocess
import sys


def _unpack_mb(env: dict) -> int:
    """x3.UNPACK_CACHE_MB as a fresh interpreter reads it under env (the setting is read once at import)."""

    code = "from tensorfold.families.glm_moe_dsa.cuda import x3; print(x3.UNPACK_CACHE_MB)"
    base = {k: v for k, v in os.environ.items() if not k.startswith("TF_GLM_")}
    out = subprocess.run([sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[-1])


def test_unpack_cache_default_follows_dcp() -> None:
    assert _unpack_mb({}) == 384                                    # replicated: the micro-batches reuse entries
    assert _unpack_mb({"TF_GLM_DCP": "1"}) == 384
    assert _unpack_mb({"TF_GLM_DCP": "4"}) == 0                     # DCP4: one batch a chunk, nothing to reuse
    assert _unpack_mb({"TF_GLM_DCP": "4", "TF_GLM_UNPACK_CACHE_MB": "384"}) == 384      # an explicit value wins
    assert _unpack_mb({"TF_GLM_UNPACK_CACHE_MB": "0"}) == 0


def test_score_launch_by_shape() -> None:
    from tensorfold.families.glm_moe_dsa.cuda import dcp
    from tensorfold.families.glm_moe_dsa.cuda.select import BT

    assert dcp.score_launch(1, 512) == (BT, 2, 2)                    # decode windows keep select.BT's launch
    assert dcp.score_launch(15, 1 << 20) == (BT, 2, 2)
    assert dcp.score_launch(8192, 2048) == (128, 4, 2)               # prompt blocks up to 8,192 slots
    assert dcp.score_launch(3072, 8192) == (128, 4, 2)
    assert dcp.score_launch(2048, 8256) == (64, 4, None)             # past them: e3290a0's launch
    assert dcp.score_launch(192, 131072) == (64, 4, None)


class _Comm:
    """Records the collectives a call makes (sizes only); moves nothing."""

    def __init__(self) -> None:
        self.gathers: list[int] = []
        self.groups = 0

    def all_gather(self, send, recv) -> None:
        self.gathers.append(send.numel())

    def grouped(self, sends, recvs) -> None:
        self.groups += 1


def _attention_call(monkeypatch, stream):
    """dcp.attention for a 128-row prompt chunk on CPU, with the kernels and the pipe stubbed out."""

    import torch

    from tensorfold.families.glm_moe_dsa.cuda import dcp

    R, H, LW, PW, G = 128, 2, 8, 4, 4
    piped: list[int] = []
    monkeypatch.setattr(dcp, "PIPE", True)
    monkeypatch.setattr(dcp, "pipe_attention", lambda *a, **k: piped.append(a[1].shape[0]))
    monkeypatch.setattr(dcp, "attend", lambda *a, **k: None)
    monkeypatch.setattr(dcp, "combine", lambda *a, **k: None)
    comm = _Comm()
    w = type("W", (), {"world": G, "rank": 1, "comm": comm})()
    d = dcp.Scratch(R, H, LW, PW, G, torch.device("cpu"))
    qa = torch.zeros((R, H, LW), dtype=torch.bfloat16)
    qp = torch.zeros((R, H, PW), dtype=torch.bfloat16)
    out = torch.zeros((R, H, LW), dtype=torch.bfloat16)
    tokens = torch.zeros((R, 8), dtype=torch.int32)
    counts = torch.zeros((R,), dtype=torch.int32)
    pos = torch.zeros((1,), dtype=torch.int32)
    dcp.attention(w, qa, qp, None, None, tokens, counts, pos, out, 1.0, d, 2048, stream=stream)
    return comm, piped, R * H * (LW + PW)


def test_pipe_does_not_gather_the_whole_chunk_first(monkeypatch) -> None:
    comm, piped, _ = _attention_call(monkeypatch, stream=object())
    assert piped == [128]                       # the chunk went to the pipe
    assert comm.gathers == [] and comm.groups == 0   # and nothing was gathered before it (the pipe gathers by parts)


def test_inline_path_still_gathers_once(monkeypatch) -> None:
    comm, piped, per_rank = _attention_call(monkeypatch, stream=None)
    assert piped == []
    assert comm.gathers == [per_rank] and comm.groups == 1      # one query gather, one partial exchange, as before
