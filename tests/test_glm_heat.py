"""Heat pacing (heat.py, TF_GLM_HEAT_PACE): off unless set; at a chunk boundary every rank waits while the hottest GPU is
at HI or its last chunk's rise would reach TOP, until the hottest is at LO; a chunk run from a pause sets how far under
TOP the next pause must cool. Fake GPUs here: each second of pause cools every GPU by ``cool`` C."""

from __future__ import annotations

import importlib
import time
import types

import pytest


def load(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TF_GLM_HEAT_PACE", raising=False)
    else:
        monkeypatch.setenv("TF_GLM_HEAT_PACE", value)
    from tensorfold.families.glm_moe_dsa.cuda import heat

    heat = importlib.reload(heat)
    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    return heat


class Gpus:
    def __init__(self, heat, temps, cool=2):
        self.t, self.cool, self.clock, self.reads = list(temps), cool, 0.0, 0
        heat._temps = self.read
        heat.time = types.SimpleNamespace(time=lambda: self.clock, monotonic=lambda: self.clock, sleep=self.sleep,
                                          strftime=time.strftime, gmtime=time.gmtime)

    def read(self, comm, world):
        self.reads += 1
        return list(self.t)

    def sleep(self, s):
        self.clock += s
        self.t = [x - self.cool * s for x in self.t]


def test_off_unless_set(monkeypatch):
    heat = load(monkeypatch, None)
    g = Gpus(heat, [95, 95])
    heat.pace(None, 2, 0, first=True)
    assert heat.PACE is None and g.reads == 0 and heat.paused == 0


@pytest.mark.parametrize("value, want", [("1", (78, 66, 82)), ("80:70", (80, 70, 84)), ("80:70:83", (80, 70, 83)),
                                         ("0", None)])
def test_parse(monkeypatch, value, want):
    assert load(monkeypatch, value).PACE == want


@pytest.mark.parametrize("value", ["70:80", "80", "80:70:79", "80:0"])
def test_parse_refuses(monkeypatch, value):
    with pytest.raises(ValueError):
        load(monkeypatch, value)


def test_under_hi_runs_on(monkeypatch):
    heat = load(monkeypatch, "78:66:82")
    Gpus(heat, [70, 77])
    heat.pace(None, 2, 0, first=True)
    assert heat.paused == 0 and heat.pauses == []


def test_at_hi_every_rank_waits_until_lo(monkeypatch):
    heat = load(monkeypatch, "78:66:82")
    g = Gpus(heat, [79, 74])
    heat.pace(None, 2, 0, first=True)
    assert max(g.t) <= 66 and heat.paused == 7 and len(heat.pauses) == 1     # 79 -> 65 at 2 C a second


def test_a_rise_that_would_reach_top_pauses_early(monkeypatch):
    heat = load(monkeypatch, "80:66:82")
    g = Gpus(heat, [60, 60])
    heat.pace(None, 2, 0, first=True)                 # an idle reading: no rise from it
    g.t = [74, 75]
    heat.pace(None, 2, 0)                             # the first loaded reading
    assert heat.paused == 0
    g.t = [77, 79]                                    # under HI, but 79 + 4 would reach TOP
    heat.pace(None, 2, 0)
    assert heat._rise == [3, 4] and heat.paused > 0 and max(g.t) <= 66


def test_a_chunk_from_a_pause_sets_the_next_pause(monkeypatch):
    heat = load(monkeypatch, "78:66:82")
    g = Gpus(heat, [80, 70])
    heat.pace(None, 2, 0, first=True)                 # waits 80 -> 66
    assert g.t == [66, 56]
    g.t = [86, 75]                                    # that chunk rose 20 / 19 C from idle (the jump and its heat)
    heat.pace(None, 2, 0)
    assert heat._cold == [20, 19]
    assert max(t + c for t, c in zip(g.t, heat._cold)) < 82                  # cooled until the next chunk fits,
    assert g.t == [60, 49]                                                   # past LO: 60 + 20 < 82


def test_a_new_prompt_forgets_the_last_rises(monkeypatch):
    heat = load(monkeypatch, "78:66:82")
    g = Gpus(heat, [70, 70])
    heat.pace(None, 2, 0, first=True)
    g.t = [75, 76]
    heat.pace(None, 2, 0)
    g.t = [77, 77]
    heat.pace(None, 2, 0)
    assert heat._rise is not None
    g.t = [60, 60]
    heat.pace(None, 2, 0, first=True)
    assert heat._rise is None and heat._last is None
