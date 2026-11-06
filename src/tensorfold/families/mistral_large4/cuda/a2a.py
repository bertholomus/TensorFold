"""The EP4 MoE all-to-all (note D2 §3): two phases over the ``Comm`` protocol, counts then data.

Per MoE layer, every rank dispatches its rows' routed picks to the ranks that own the experts
(``expert_rank`` = ``e // (E // world)``, D2 §2's e//32 — the phase-1 loader's ``local_ids`` split)
and gets the expert work back:

* **counts**: each rank all-gathers a row of ``world`` int32 counts, so every rank holds the full
  ``world x world`` pair matrix; the reply needs no second count exchange (D2 §3: "the reply
  mirrors the dispatch slots" — each side predicts the other's reply count from its own metadata);
* **data**: for every active peer pair in a fixed global order, one ``comm.exchange`` group ships
  ``[expert ids i32 | weights f32 | hidden rows]`` to the peer and receives the peer's, and a
  matching reply group carries the expert work back on the mirrored slots.

That is a layer's only cross-rank traffic (D2 §0.4: under DP4 attention is rank-local; a rank with
no rows of its own this round still runs the layer so the dispatches landing on its experts are
served — see ``Ml4Model._layer``'s zero-row path). The code touches no NCCL symbol itself: it
speaks the ``tensorfold.cuda.comm.Comm`` protocol (``all_gather`` + ``exchange``), so the RoCE
custom all-to-all (D2 O2) drops in behind the same protocol without touching this file.

Two combine regimes over the same transport:

* ``run_local`` (the CPU/eager regime, and what ``tests/test_ml4_engine.py`` gates): the receiver
  runs each routed pick separately and returns one vector per pick; the owner scatters them to
  ``[row, slot]`` and sums slots ``0..topk-1`` in order — exactly the world=1 eager loop's add
  order, so with fp32 payloads the EP4 output is **bit-exact** to all-experts-local. The engine's
  accuracy argument is then "the same numbers, moved", not "close enough".
* ``run_batch`` (the CUDA regime): the receiver sums its own share per row — one fp16 vector per
  received row, D2 §3's byte-lean shape — through ``exl3_experts.routed``; the owner adds the
  partials in destination-rank order. fp16 experts are not bit-reproducible across layouts
  anyway, so this buys the smaller wire where the exactness test cannot apply.

Pair building is eager Python (rows x ranks) — the reference semantics the phase-B CUDA-graph
decode body ports to device ints; at a 2048-row chunk the loop costs ~1 ms a layer, noise beside
the GEMMs it feeds.
"""
from __future__ import annotations

import itertools

import torch

from .dp4 import trace_tag

F32 = torch.float32


def route_dest(idx: torch.Tensor, world: int, n_routed: int) -> torch.Tensor:
    """Destination rank per routed pick, D2 §2's block rule (``e // (E // world)``)."""

    if n_routed % world:
        raise ValueError(f"EP4 needs whole experts a rank: {n_routed} experts do not divide into {world} ranks")
    return idx // (n_routed // world)


def row_pairs(dest_row, world: int) -> list[tuple[int, list[int]]]:
    """One row's picks grouped by destination rank: ``[(dst, [slots ascending])]``, dst ascending."""

    return [(d, [s for s, e in enumerate(dest_row) if e == d]) for d in range(world)
            if any(e == d for e in dest_row)]


def _pair_buffers(slots, idx_row: torch.Tensor, w_row: torch.Tensor, topk: int, dev) -> tuple:
    """One pair's (eids i32[topk], wts f32[topk]) built on ``dev`` — the ROUTING tensors' device.

    m2b8 root cause (load4 2026-11-06, m2b6 build): these two buffers were allocated without a
    device, so they landed on the CPU while ``idx``/``w`` were cuda; loading cuda values into a
    cpu buffer (``eids[slots] = <cuda>``) raised "Expected all tensors to be on the same device,
    but found at least two devices, cuda:0 and cpu!" in the FIRST CUDA world>1 forward ever run
    — the m2b6 activation probe. Followers died there and rank 0 blocked in the layer's counts
    all-gather waiting for ranks that had already left the lane (the probe timeout, not this
    helper, is that guard). Tests pin the device with the meta device: tests/test_ml4_engine.py."""

    eids = torch.full((topk,), -1, dtype=torch.int32, device=dev)
    eids[slots] = idx_row[slots].to(torch.int32)
    wt = torch.zeros(topk, dtype=F32, device=dev)
    wt[slots] = w_row[slots].float()
    return eids, wt


def moe_all_to_all(comm, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, *, n_routed: int,
                   run_local=None, run_batch=None) -> torch.Tensor:
    """One MoE layer's routed part for this rank's ``x`` rows: ``out [R, dim]`` = the weighted
    routed contributions summed (the shared expert and residual stay with the caller).

    ``run_local(e_global, rows_1x_dim) -> contribution 1xDIM`` runs one routed pick unweighted
    (CPU regime; ``moe_all_to_all`` applies the router weight). ``run_batch(recv_x, eids, wts) ->
    (vecs n_rows x DIM, n_valid_slots)`` runs a received batch's own-share partials (CUDA regime);
    its vectors land one per received row and the reply mirrors those rows exactly. Exactly one of
    the two is passed."""

    if (run_local is None) == (run_batch is None):
        raise ValueError("moe_all_to_all: pass exactly one of run_local / run_batch")
    rank, world, dev = comm.rank, comm.world, x.device
    R, topk, dim = x.shape[0], idx.shape[1], x.shape[-1]
    dest = route_dest(idx.cpu(), world, n_routed).tolist() if R else []

    # pairs this rank sends, per destination rank: (row, eids (topk) i32, wts (topk) f32), row order
    to_send: list[list] = [[] for _ in range(world)]
    own = []                                                  # (row, slots) picked on this rank
    for r in range(R):
        for dst, slots in row_pairs(dest[r], world):
            if dst == rank:
                own.append((r, slots))
                continue
            eids, wt = _pair_buffers(slots, idx[r], w[r], topk, dev)   # on x's device (m2b8)
            to_send[dst].append((r, eids, wt))

    counts = torch.empty(world * world, dtype=torch.int32, device=dev)
    with trace_tag(comm, "a2a counts"):
        comm.all_gather(torch.tensor([len(t) for t in to_send], dtype=torch.int32, device=dev), counts)
    done = getattr(comm, "gather_done", None)                 # m2b11: C drives WHICH pairs exchange and at
    if done is not None:                                      # what size, so it must have LANDED before
        done()                                                # any pairing decision (m2b8's agree() lesson;
    C = counts.view(world, world)                             # no-op on NCCL — int() reads sync anyway)

    contrib = torch.zeros((R, topk, dim), dtype=F32, device=dev) if run_local is not None else None
    partials: list[list] = [] if run_local is None else None
    if run_local is not None:                                 # own picks: no wire, straight to the adder
        for r, slots in own:
            for s in slots:
                contrib[r, s] = (run_local(int(idx[r, s]), x[r:r + 1]) * w[r, s]).float()
    else:
        for r, slots in own:                                  # one own-share partial per row
            eids = torch.full((1, topk), -1, dtype=torch.int32, device=dev)
            eids[0, slots] = idx[r, slots].to(torch.int32)
            wt = torch.zeros((1, topk), dtype=F32, device=dev)
            wt[0, slots] = w[r, slots].float()
            partials.append((r, rank, run_batch(x[r:r + 1], eids, wt)[0][0]))

    for i, j in itertools.combinations(range(world), 2):      # fixed global order: every rank agrees
        if int(C[i, j]) + int(C[j, i]) == 0:
            continue
        if rank not in (i, j):                                # a pair's exchanges involve only its two members
            continue
        peer = j if rank == i else i
        out_list, n_in = to_send[peer], int(C[peer, rank])

        def _ship(items, n):                                  # [eids | wts | rows] for one direction
            if not items:
                return (torch.zeros((0, topk), dtype=torch.int32, device=dev),
                        torch.zeros((0, topk), dtype=F32, device=dev),
                        torch.zeros((0, dim), dtype=x.dtype, device=dev))
            return (torch.stack([it[1] for it in items]).to(dev), torch.stack([it[2] for it in items]).to(dev),
                    torch.stack([x[it[0]] for it in items]).to(dev))

        m_out, w_out, x_out = _ship(out_list, len(out_list))
        m_in = torch.zeros((n_in, topk), dtype=torch.int32, device=dev)
        w_in = torch.zeros((n_in, topk), dtype=F32, device=dev)
        x_in = torch.zeros((n_in, dim), dtype=x.dtype, device=dev)
        with trace_tag(comm, f"a2a({i},{j})dispatch"):
            comm.exchange([m_out, w_out, x_out], [m_in, w_in, x_in], peer)

        if run_local is not None:                             # receiver: one vector per routed pick
            vecs, rep = [], []
            for p in range(n_in):
                for s in range(topk):
                    e = int(m_in[p, s])
                    if e >= 0:
                        vecs.append(run_local(e, x_in[p:p + 1]) * w_in[p, s])
                        rep.append((p, s))
            vec_t = torch.cat(vecs, 0) if vecs else torch.zeros((0, dim), dtype=x.dtype, device=dev)
            rep_t = (torch.tensor(rep, dtype=torch.int32) if rep else torch.zeros((0, 2), dtype=torch.int32)).to(dev)
            n_reply = int((m_out >= 0).sum())                 # the peer predicts the same from its meta
        else:                                                 # receiver: one own-share partial per row
            vec_t, n_valid = run_batch(x_in, m_in, w_in)
            rep_t = torch.stack([torch.arange(n_in, dtype=torch.int32),
                                 torch.full((n_in,), -1, dtype=torch.int32)], 1).to(dev)
            # m2b11 ROOT CAUSE (load5 2026-11-06, m2b10 build — both load4 probe timeouts were this too):
            # the reply RECV is what pairs against the PEER's reply SEND, and the peer sends one partial
            # per row IT dispatched to me = len(out_list) HERE. Sizing it by n_in (rows I received) only
            # pairs when C[i,j] == C[j,i] — false for any real routing. Mode B: rank 0 owned every row,
            # so C[0,j] > 0 and C[j,0] = 0; rank 0's reply group was empty (NCCL skips zero-size legs)
            # while the follower posted send/recv into a peer that had already moved to the next layer's
            # counts all_gather — a silent P2P deadlock (rank 0 GPU 96%, rank 1 CPU 97%, rounds 0, no
            # error). run_local sizes its reply RECV from its own dispatch (m_out below) and is correct;
            # the CUDA regime simply never shared that rule until now.
            n_reply = len(out_list)                           # one reply per row dispatched OUT

        rep_back = torch.zeros((n_reply, 2), dtype=torch.int32, device=dev)
        vec_back = torch.zeros((n_reply, dim), dtype=x.dtype, device=dev)
        with trace_tag(comm, f"a2a({i},{j})reply"):            # the pair's out=[..] vs in=[..] lines MUST
            comm.exchange([rep_t, vec_t.to(x.dtype)], [rep_back, vec_back], peer)   # mirror across ranks
        for k in range(n_reply):                              # owner scatters by its own send order
            p, s = int(rep_back[k, 0]), int(rep_back[k, 1])
            row = out_list[p][0]
            if contrib is not None:
                contrib[row, s] = vec_back[k].float()
            else:
                partials.append((row, peer, vec_back[k]))

    mark = getattr(comm, "trace", None)                        # m2b11: combine is local (no wire), but the
    if mark is not None:                                       # trace names it — a rank parked BEFORE the
        mark(f"a2a combine rows={R} got={len(partials) if partials else 0}")   # combine is missing replies

    if contrib is not None:
        acc = torch.zeros((R, dim), dtype=F32, device=dev)
        for s in range(topk):                                 # the world=1 eager loop's exact add order
            acc += contrib[:, s]
        return acc
    acc = torch.zeros((R, dim), dtype=F32, device=dev)
    for r, _dst, v in sorted(partials, key=lambda t: (t[0], t[1])):   # destination-rank order per row
        acc[r] += v.float()
    return acc
