"""TF_DS_REPLAY_FLOOR: a prompt chunk of RING_EXTRA rows or fewer whose window reaches below the bounded replay's cut
reads its window by the gather path (from the cut), not from the ring, whose slots below the cut hold the previous
request's keys. Off, and wherever the window stays at or above the cut, the choice is today's."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.cuda import model

W = 128                                         # DeepSeek-V4.1's attention window


def test_off_is_todays_choice(monkeypatch):
    monkeypatch.setattr(model, "REPLAY_FLOOR", False)
    assert model.ring_mode_for(8, 8184, 8184, W)           # below the cut, yet ring mode: the stale-key read
    assert model.ring_mode_for(16, 8184, 8184, W)
    assert not model.ring_mode_for(17, 8184, 8184, W)      # more than RING_EXTRA rows: the gather path either way


def test_on_a_small_chunk_at_the_cut_gathers(monkeypatch):
    monkeypatch.setattr(model, "REPLAY_FLOOR", True)
    # a prompt of 8,312 tokens (2048 x 4 + 120): the replay cut at 8,184 leaves an 8-row last chunk whose window
    # (8,057 .. 8,183) lies below the cut
    assert not model.ring_mode_for(8, 8184, 8184, W)
    assert not model.ring_mode_for(16, 8184 + 100, 8184, W)  # its window still reaches 27 positions below the cut


def test_on_unchanged_where_the_window_stays_above_the_cut(monkeypatch):
    monkeypatch.setattr(model, "REPLAY_FLOOR", True)
    assert model.ring_mode_for(8, 8184 + W - 1, 8184, W)     # the window starts exactly at the cut
    assert model.ring_mode_for(8, 8184 + 5 * W, 8184, W)
    assert model.ring_mode_for(1, 0, 0, W)                   # no replay (floor 0): decode-sized chunks as before
    assert model.ring_mode_for(16, 50_000, 0, W)
    assert not model.ring_mode_for(17, 50_000, 0, W)
