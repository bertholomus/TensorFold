"""tensorfold.cuda.rdma's wiring on CPU (no RDMA device, no GPU; the extension stood in): a gather hands its proxy the
idle back-off at create (TF_RDMA_IDLE_S as microseconds, TF_RDMA_NAP_US) and passes ``idle_stats`` through; the env
parses as documented (0: never naps; unset: 2 s, 100 us)."""

import importlib
import threading

import pytest

pytest.importorskip("torch")

from tensorfold.cuda import rdma  # noqa: E402


class Store:
    """The ranks' rendezvous store: ``set``, and a ``get`` that waits for its key."""

    def __init__(self) -> None:
        self.d, self.cv = {}, threading.Condition()

    def set(self, key, value) -> None:
        with self.cv:
            self.d[key] = value
            self.cv.notify_all()

    def get(self, key):
        with self.cv:
            assert self.cv.wait_for(lambda: key in self.d, timeout=10), f"no {key}"
            return self.d[key]


class Ext:
    """rdma_gather's bindings, recorded: a handle a create; each rank's queue pairs as local_info gives them."""

    def __init__(self) -> None:
        self.created, self.started, self.lock = [], [], threading.Lock()

    def create(self, *args):
        with self.lock:
            self.created.append(args)
            return len(self.created) - 1

    def local_info(self, h):
        return [([10 + h], [20 + h], [bytes(16)], 4096, [30 + h]) for _ in range(2)]

    def connect(self, h, infos):
        assert len(infos) == 2

    def start(self, h):
        self.started.append(h)

    def stop(self, h):
        pass

    def idle_stats(self, h):
        return [3 + h, 1]


def test_a_gather_hands_its_proxy_the_idle_back_off(monkeypatch):
    ext = Ext()
    monkeypatch.setattr(rdma, "_ext", lambda: ext)
    monkeypatch.setattr(rdma, "IDLE_S", 0.25)
    monkeypatch.setattr(rdma, "NAP_US", 80)
    store, out = Store(), {}

    def make(r: int) -> None:
        out[r] = rdma.RdmaGather(store, r, 2, max_bytes=4096, slots=4, devices=["mlx5_0"], gid_index=3)

    ts = [threading.Thread(target=make, args=(r,)) for r in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    assert sorted(out) == [0, 1] and sorted(ext.started) == [0, 1]
    assert [a[-2:] for a in ext.created] == [(250000, 80), (250000, 80)]          # (idle_us, nap_us) at create
    assert out[0].idle_stats() == (3 + out[0].h, 1) and isinstance(out[1].idle_stats(), tuple)


def test_the_idle_back_off_s_env(monkeypatch):
    try:
        monkeypatch.setenv("TF_RDMA_IDLE_S", "0")
        monkeypatch.setenv("TF_RDMA_NAP_US", "50")
        m = importlib.reload(rdma)
        assert m.IDLE_S == 0.0 and m.NAP_US == 50                                  # (0: the proxy never naps)
        monkeypatch.setenv("TF_RDMA_IDLE_S", "0.5")
        assert importlib.reload(rdma).IDLE_S == 0.5
        monkeypatch.delenv("TF_RDMA_IDLE_S")
        monkeypatch.delenv("TF_RDMA_NAP_US")
        m = importlib.reload(rdma)
        assert m.IDLE_S == 2.0 and m.NAP_US == 100
    finally:
        importlib.reload(rdma)
