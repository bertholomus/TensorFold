"""The config's expert SwiGLU clamp: min(silu(g), L) * clip(u, -L, L), L = 10 on the routed experts
of layers 65-71 and the shared expert of 64-71, never the MTP layer, as the serving stacks apply it (the reference
file ignores the lists; TF_LING_SWIGLU_CLAMP=0 serves without them).

- config: the bank's lists give exactly those layers (inf elsewhere and past the lists); malformed lists are refused
  (a clamp on a dense layer, an entry for the MTP layer, a limit that is not a positive bf16 value, not a list);
- the engine config carries the limits and the switch drops them; moe_block hands each layer's limits to the routed
  experts (ACT_BF16_POST) and to the shared expert (act.silu_mul), with every kernel recorded instead of run;
- the shared expert's kernel (bailing_v3/cuda/act.py): tests/bailing_kernel_checks.py "swiglu" in a child process
  (natively on CUDA, else Triton's interpreter).

The routed experts' copy of the rule (tensorfold.cuda.exl3.experts ACT_BF16_POST) is CUDA only:
tests/cuda/test_exl3_experts.py::test_post_silu_clamp_mode.

Run: python -m pytest tests/test_bailing_9_swiglu.py -q
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "tests"))

BANK = Path(os.environ.get("LING31_FP8_DIR", "/models/Ling-3.1-flash"))
INF = math.inf
LISTS = ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list")


def from_dict(raw: dict):
    from tensorfold.families.bailing_v3.config import Config

    return Config.from_dict(raw)


def _fake(**over) -> dict:
    """bailing_fakes' tiny config (3 layers, layer 0 dense) with keys replaced."""

    from bailing_fakes import CFG

    raw = copy.deepcopy(CFG)
    raw.update(over)
    return raw


def _ling_like() -> dict:
    """The fake config at Ling-3.1-flash's depth: 72 layers, 4 dense, the bank's two lists."""

    raw = _fake(num_hidden_layers=72, layer_group_size=8, first_k_dense_replace=4)
    raw[LISTS[0]] = [0] * 65 + [10] * 7
    raw[LISTS[1]] = [0] * 64 + [10] * 8
    return raw


@pytest.mark.skipif(not BANK.is_dir(), reason="the banked Ling-3.1-flash checkpoint is not on this host")
def test_the_bank_clamps_routed_65_to_71_and_shared_64_to_71():
    cfg = from_dict(json.loads((BANK / "config.json").read_text()))
    assert [cfg.swiglu_limit(i) for i in range(73)] == [INF] * 65 + [10.0] * 7 + [INF]
    assert [cfg.swiglu_limit(i, shared=True) for i in range(73)] == [INF] * 64 + [10.0] * 8 + [INF]
    assert len(cfg.expert_swiglu_limits) == len(cfg.shared_swiglu_limits) == 72    # the MTP layer (72) is past both


def test_lists_parse_per_layer():
    cfg = from_dict(_fake(expert_swiglu_limit_list=[0, None, 4.0], share_expert_swiglu_limit_list=[0, 0.75]))
    assert [cfg.swiglu_limit(i) for i in range(4)] == [INF, INF, 4.0, INF]
    assert [cfg.swiglu_limit(i, shared=True) for i in range(4)] == [INF, 0.75, INF, INF]   # a short list ends early
    raw = _fake()
    for key in LISTS:
        del raw[key]
    cfg = from_dict(raw)
    assert cfg.expert_swiglu_limits == cfg.shared_swiglu_limits == ()
    assert all(cfg.swiglu_limit(i, s) == INF for i in range(4) for s in (False, True))


@pytest.mark.parametrize("key", LISTS)
@pytest.mark.parametrize("value,why", [
    ([4.0, 0, 0], "dense MLP layer 0"),
    ([0, 0, 0, 4.0], "4 entries for 3 layers"),
    ([0, -1.0, 0], "not a positive bf16 value"),
    ([0, 0.1, 0], "not a positive bf16 value"),            # 0.1 is not a bf16 number: where it rounds is unpinned
    ([0, float("nan"), 0], "not a positive bf16 value"),
    (10.0, "a list of per-layer limits"),
])
def test_refusals(key, value, why):
    with pytest.raises(ValueError) as e:
        from_dict(_fake(**{key: value}))
    assert key in str(e.value) and why in str(e.value), str(e.value)


def test_engine_config_carries_the_limits_and_the_switch_drops_them(tmp_path, monkeypatch):
    import subprocess

    from tensorfold.families.bailing_v3.cuda import weights as W

    (tmp_path / "config.json").write_text(json.dumps(_fake()))
    c = W.Config.read(tmp_path)
    assert [c.swiglu_limit(i) for i in range(4)] == [INF, INF, 4.0, INF]
    assert [c.swiglu_limit(i, shared=True) for i in range(4)] == [INF, 4.0, 4.0, INF]
    monkeypatch.setattr(W, "CLAMP", False)
    c = W.Config.read(tmp_path)
    assert all(c.swiglu_limit(i, s) == INF for i in range(4) for s in (False, True))
    # the switch is read from the environment when the module loads: "0" turns it off, anything else leaves it on
    code = f"import sys; sys.path.insert(0, {str(SRC)!r}); from tensorfold.families.bailing_v3.cuda import weights; " \
           "print(weights.CLAMP)"
    for env, want in (("0", "False"), ("1", "True"), (None, "True")):
        import os

        e = {k: v for k, v in os.environ.items() if k != "TF_LING_SWIGLU_CLAMP"}
        if env is not None:
            e["TF_LING_SWIGLU_CLAMP"] = env
        r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and r.stdout.strip() == want, (env, r.stdout, r.stderr[-2000:])


def test_moe_block_hands_each_layer_its_limits(tmp_path, monkeypatch):
    """Ling-3.1-flash's layout: the routed experts get L at 65-71 with ACT_BF16_POST everywhere (inf: ACT_BF16's
    bits), the shared expert L at 64-71, the MTP layer (72) neither; in prompt chunks and decode windows alike."""

    torch = pytest.importorskip("torch")
    from tensorfold.cuda.exl3 import experts as x3experts
    from tensorfold.families.bailing_v3.cuda import act, forward as F
    from tensorfold.families.bailing_v3.cuda import weights as W

    calls: dict = {}
    monkeypatch.setattr(F, "mm", lambda *a, **k: None)
    monkeypatch.setattr(F, "gather", lambda w, b, R: None)
    monkeypatch.setattr(F.glue, "router", lambda *a, **k: None)
    monkeypatch.setattr(F.route_mod, "route", lambda *a, **k: None)
    monkeypatch.setattr(x3experts, "routed",
                        lambda *a, limit, act_mode, **k: calls.__setitem__("routed", (limit, act_mode)))
    monkeypatch.setattr(act, "silu_mul", lambda gu, out, xs, limit: calls.__setitem__("shared", limit))

    def buffers(prefill: bool):
        z = torch.zeros((2, 8))
        return SimpleNamespace(prefill=prefill, normed=z, sgu=z, sact=z, sxs=z, sy=z.clone(), mlog=z, msc=z, pick=z,
                               wts=z, moe=None, part=z.clone())

    def layer(i: int):
        shared = SimpleNamespace(gu=None, down=None)
        return SimpleNamespace(index=i, moe=SimpleNamespace(shared=shared, router=None, bias=None,
                                                            experts=SimpleNamespace(ex=None)))

    for clamp in (True, False):
        (tmp_path / "config.json").write_text(json.dumps(_ling_like()))
        monkeypatch.setattr(W, "CLAMP", clamp)
        w = SimpleNamespace(cfg=W.Config.read(tmp_path))
        L = 10.0 if clamp else INF
        for i, want in ((4, (INF, INF)), (63, (INF, INF)), (64, (INF, L)), (65, (L, L)), (71, (L, L)),
                        (72, (INF, INF))):
            for prefill in (True, False):
                calls.clear()
                F.moe_block(layer(i), w, buffers(prefill), 2)
                assert calls == {"routed": (want[0], x3experts.ACT_BF16_POST), "shared": want[1]}, (clamp, i, prefill)


def test_shared_expert_kernel_against_the_clamp_rule():
    """bailing_v3/cuda/act.py: L = inf gives glue.silu_mul's bits; finite L the clamp after silu on the kernel's own
    bf16 silu, one rounding of the product; NaN kept; rows window-independent (tests/bailing_kernel_checks.py)."""

    import os
    import subprocess

    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = dict(os.environ)
    if dev == "cpu":
        env["TRITON_INTERPRET"] = "1"
    r = subprocess.run([sys.executable, str(REPO / "tests" / "bailing_kernel_checks.py"), "swiglu", dev], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-4000:] + r.stderr[-4000:]
