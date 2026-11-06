"""Where a step's time goes on one rank: CUDA events around every leaf call, scopes for the call sites, a kineto pass.

A leaf's GPU time runs from its first event to its last, so a host-bound step shows launch gaps as op time; the kineto
table has the kernels alone, and its sum against the wall says how much of the step the GPU was busy. A collective's
time includes its wait for the slowest rank.
"""
from __future__ import annotations

import json
import time

import torch

from tensorfold.families.mistral_large4.cuda import cp4

LEAVES = ("embed", "lin", "mat", "router", "local_topk", "attend", "absorb_q", "expand_o", "experts", "logits")
HELPERS = ("rms_norm", "layer_norm_bias", "ssmax_apply", "rotate_leading", "gptj_rotate", "route_softmax", "kth_key",
           "filter_selection", "merge_partials")
SCOPES = (("_select", "select"), ("_attention", "attn"), ("_moe", "moe"))


class Prof:
    def __init__(self) -> None:
        self.on, self.rows, self.scope, self.inside = False, [], [], 0

    def leaf(self, owner, name: str, label: str | None = None) -> None:
        fn, label = getattr(owner, name), label or name

        def call(*args, **kwargs):
            if not self.on or self.inside:
                return fn(*args, **kwargs)
            self.inside += 1
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            h0 = time.perf_counter()
            e0.record()
            try:
                out = fn(*args, **kwargs)
            finally:
                self.inside -= 1
            e1.record()
            where = "/".join(self.scope)
            self.rows.append((f"{where}/{label}" if where else label, e0, e1, time.perf_counter() - h0))
            return out

        setattr(owner, name, call)

    def within(self, owner, name: str, label: str) -> None:
        fn = getattr(owner, name)

        def call(*args, **kwargs):
            if not self.on:
                return fn(*args, **kwargs)
            self.scope.append(label)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            h0 = time.perf_counter()
            e0.record()
            try:
                out = fn(*args, **kwargs)
            finally:
                self.scope.pop()
            e1.record()
            self.rows.append((f"[{label}]", e0, e1, time.perf_counter() - h0))
            return out

        setattr(owner, name, call)

    def attach(self, ops, fwd, comm) -> None:
        for name in LEAVES:
            self.leaf(ops, name)
        for name in HELPERS:
            self.leaf(cp4, name)
        self.leaf(fwd.pool, "store")
        for name, label in SCOPES:
            self.within(fwd, name, label)
        self.within(cp4, "ep_dispatch", "ep")
        if comm is not None:
            self.leaf(comm, "all_gather")
            self.leaf(comm, "alltoall")

    def table(self, what: str, wall: float, steps: int, rank: int, out: str = "") -> None:
        torch.cuda.synchronize()
        agg: dict = {}
        for label, e0, e1, host in self.rows:
            t = agg.setdefault(label, [0, 0.0, 0.0])
            t[0] += 1
            t[1] += e0.elapsed_time(e1)
            t[2] += host * 1e3
        self.rows = []
        leaves = sum(t[1] for k, t in agg.items() if not k.startswith("["))
        print(f"[prof] rank {rank} {what}: {wall * 1e3 / steps:.1f} ms a step, leaves {leaves / steps:.1f} ms", flush=True)
        for k, t in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            print(f"[prof] rank {rank} {what} {k:34s} calls {t[0] / steps:8.1f} gpu {t[1] / steps:10.3f} ms "
                  f"host {t[2] / steps:10.3f} ms", flush=True)
        if out:
            json.dump({"what": what, "wall_ms": wall * 1e3 / steps,
                       "rows": {k: [t[0] / steps, t[1] / steps, t[2] / steps] for k, t in agg.items()}},
                      open(f"{out}-{what}-rank{rank}.json", "w"), indent=0)


def kernels(step, what: str, rank: int, top: int = 28) -> None:
    """One ``step`` under kineto: the kernels alone, and how busy the GPU was."""

    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        step()
        torch.cuda.synchronize()
    wall = time.time() - t0
    ev = [e for e in p.key_averages() if e.self_device_time_total > 0]
    busy = sum(e.self_device_time_total for e in ev) / 1e3
    print(f"[prof] rank {rank} {what} kernels: {len(ev)} kinds, {sum(e.count for e in ev)} launches, busy {busy:.1f} ms of "
          f"wall {wall * 1e3:.1f} ms (the wall carries the profiler's own cost)", flush=True)
    for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:top]:
        print(f"[prof] rank {rank} {what} {e.self_device_time_total / 1e3:10.3f} ms x{e.count:<6d} {e.key[:90]}", flush=True)
