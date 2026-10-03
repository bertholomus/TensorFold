"""GLM-5.3's concurrent decoding, the host side (CPU): rank 0's steps reach every follower in order over the TCP
links, and a round's tables give each stream's rows its positions, slot base and segment, with the round's passes
and bucket."""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest


def test_link_sends_every_step_to_every_follower_in_order():
    from torch.distributed import TCPStore

    from tensorfold.families.glm_moe_dsa.cuda.multi import Link

    world = 4
    store = TCPStore("127.0.0.1", 0, None, True, timeout=timedelta(seconds=30), wait_for_workers=False)
    port = store.port
    leader = Link(store, rank=0, world=world, host="127.0.0.1")
    got = {r: [] for r in range(1, world)}

    def follow(rank: int) -> None:
        mine = TCPStore("127.0.0.1", port, None, False, timeout=timedelta(seconds=30))
        link = Link(mine, rank=rank, world=world, host="127.0.0.1")
        while True:
            op = link.receive()
            if op is None or op[0] == "stop":
                return
            got[rank].append(op)

    threads = [threading.Thread(target=follow, args=(r,)) for r in range(1, world)]
    for t in threads:
        t.start()
    ops = [["admit", [1, 2, 3], 8, None, True, True], ["round", []], ["round", [0]], ["finish", [0]], ["drop"]]
    for op in ops:
        leader.send(op)
    leader.send(["stop"])
    for t in threads:
        t.join(timeout=30)
    assert all(got[r] == ops for r in got), got


def test_tables_give_each_stream_its_rows():
    from tensorfold.families.glm_moe_dsa.cuda.rows import Tables

    t = Tables(16, 4, None)
    n = t.fill([(0, 5, 4), (8192, 2047, 2), (16384, 6000, 4)], 2048)
    assert n == 10 and t.R == 10
    assert t.pos_host[:n].tolist() == [5, 6, 7, 8, 2047, 2048, 6000, 6001, 6002, 6003]
    assert t.base_host[:n].tolist() == [0] * 4 + [8192] * 2 + [16384] * 4
    assert t.seg_host.tolist() == [[0, 4], [4, 2], [6, 4], [0, 0]]
    assert t.key() == (10, True, True, 8192)             # dense rows, sparse rows, the deepest row's bucket
    assert t.fill([(0, 100, 1)], 2048) == 1 and t.key() == (1, True, False, 0)
    assert t.fill([(0, 3000, 4), (8192, 2048, 1)], 2048) == 5 and t.key() == (5, False, True, 4096)
    with pytest.raises(ValueError):
        t.fill([(0, 0, 8), (0, 0, 8), (0, 0, 8)], 2048)      # more rows than the tables hold


def test_a_lost_rank_breaks_the_lane_on_every_rank():
    """A follower that goes away (its process killed: its sockets close) breaks the lane on rank 0 at once, and rank 0
    tells the other followers: every rank aborts its RDMA waits and NCCL, nobody keeps waiting."""

    import socket
    import time
    from types import SimpleNamespace

    from torch.distributed import TCPStore

    from tensorfold.families.glm_moe_dsa.cuda.multi import Watchdog

    world = 4
    store = TCPStore("127.0.0.1", 0, None, True, timeout=timedelta(seconds=30), wait_for_workers=False)
    calls = {r: [] for r in range(world)}

    def comm(rank: int):
        return SimpleNamespace(rdma=SimpleNamespace(abort=lambda why, r=rank: calls[r].append(("rdma", why))),
                               nccl=SimpleNamespace(abort=lambda r=rank: calls[r].append(("nccl",)),
                                                    async_error=lambda: 0))

    dogs = {0: Watchdog(comm(0), store, rank=0, world=world, host="127.0.0.1", timeout=30, every=0.2)}
    for r in range(1, world):
        client = TCPStore("127.0.0.1", store.port, None, False, timeout=timedelta(seconds=30))
        dogs[r] = Watchdog(comm(r), client, rank=r, world=world, host="127.0.0.1", timeout=30, every=0.2)
    time.sleep(1.0)
    assert all(d.broken is None for d in dogs.values())         # heartbeats keep an idle lane alive
    dead = dogs[2]
    dead.broken = "killed"                                      # its threads stop; its process "dies":
    for sock in dead.socks:
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
    deadline = time.time() + 5
    while time.time() < deadline and any(dogs[r].broken is None for r in (0, 1, 3)):
        time.sleep(0.05)
    for r in (0, 1, 3):
        assert dogs[r].broken is not None, r
        assert ("nccl",) in calls[r] and any(c[0] == "rdma" for c in calls[r]), calls[r]
    assert "rank 2" in dogs[0].broken and dogs[1].broken == "rank 0 lost a rank"
