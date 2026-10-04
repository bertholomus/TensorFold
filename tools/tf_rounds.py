"""A concurrent round's compute on one rank alone: TF_TP_WORLD=4 (rank 0's shard), the collectives replaced by copies as
tf_profile's TF_PROFILE_LOCAL does (fp32 partials gather as this rank's plus zeros, ints as copies). Chat streams
decode together through the --parallel scheduler (MTP-3, greedy, end tokens ignored); for each group size (1, 2, 4
streams at once) it prints the wall time a round without and with the profiler, the host's stages a round, kernel
time a round by name, and at the end each kernel's cost a row: its slope from the smallest group to the largest.
The replies are not the lane's (one rank's partials), but every round verifies each stream's pending token and its
drafts (4 rows a stream at MTP-3), the same shapes and kernels as the lane's rounds; the fabric is not in it.

usage (one GPU, tf container, models mounted): python3 tools/tf_rounds.py MODEL [CONTEXT=8192] [SECONDS=6]
  [GROUPS=1,2,4]   (TF_ROUNDS_EAGER=1: eager rounds, no graph replays; TF_ROUNDS_TOP: kernels listed, default 24;
  TF_ROUNDS_SWITCH: sys.setswitchinterval seconds, for the client threads' GIL handoffs; TF_ROUNDS_TEXT=FILE with
  TF_ROUNDS_DEPTH=N: each stream's prompt is its own N-token slice of FILE and a request to continue it, so the
  rounds read caches N tokens deep: sparse attention and the indexer at depth; CONTEXT must hold N plus the reply)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch


HIDDEN = 6144


class Alone:
    """This rank on its own: fp32 partials gather as its own plus zeros, anything else as copies (int settings: every
    rank agrees; sampling's (value, token) pairs: every rank's best is this rank's, so greedy picks this rank's argmax
    instead of a zero rank's token 0 and the streams' tokens vary as a reply's do)."""

    def __init__(self, rank: int, world: int) -> None:
        self.rank, self.world = rank, world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if send.dtype == torch.float32 and send.numel() % HIDDEN == 0:      # a partial (rows x hidden)
            recv.zero_()
            recv.view(self.world, -1)[self.rank].copy_(send.view(-1))
        else:
            recv.view(self.world, -1).copy_(send.view(1, -1).expand(self.world, -1))

    def barrier(self) -> None:
        torch.cuda.synchronize()

    def grouped(self, sends, recvs) -> None:
        """Decode context parallelism's prompt-chunk exchange (each group to the rank that owns it): alone, every group
        comes back as this rank sent it (the replies are not the lane's; solo and concurrent runs see the same)."""

        for (s, _), (r, _) in zip(sends, recvs):
            r.copy_(s)


class Quiet:
    """No followers to tell and no peers to watch (multi.Link / multi.Watchdog stand-in)."""

    broken = None
    busy = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    def send(self, op) -> None:
        pass


class Calls:
    """Host seconds and calls of the round's parts that wait on the GPU or the gathers (TF_ROUNDS_TRACE=1): each
    wrapped call is timed from its entry (after a device sync, so earlier work is not charged to it) to its return."""

    def __init__(self) -> None:
        self.s: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)

    def wrap(self, owner, name: str, label: str) -> None:
        fn = getattr(owner, name)

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.s[label] += time.perf_counter() - t
                self.n[label] += 1

        setattr(owner, name, timed)

    def report(self, since: tuple, rounds: int, until: tuple) -> str:
        (s0, n0), (s1, n1) = since, until
        return ", ".join(f"{k} {1e3 * (v - s0.get(k, 0.0)) / rounds:.2f} ms x{(n1[k] - n0.get(k, 0)) / rounds:.1f}"
                         for k, v in sorted(s1.items()))


def kernels(prof) -> dict[str, float]:
    times: dict[str, float] = defaultdict(float)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            name = ev.name if len(ev.name) < 90 else ev.name[:87] + "..."
            times[name] += ev.time_range.elapsed_us() / 1e3
    return times


@torch.no_grad()
def main() -> None:
    model = Path(sys.argv[1])
    context = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    groups = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,2,4").split(",")]
    top = int(os.environ.get("TF_ROUNDS_TOP") or 24)
    if os.environ.get("TF_ROUNDS_SWITCH"):                # the interpreter's thread switch interval (s): GIL handoffs
        sys.setswitchinterval(float(os.environ["TF_ROUNDS_SWITCH"]))
    os.environ["TF_GLM_EMBED_SPLIT"] = "0"          # alone, no rank holds the others' embedding rows: the whole table

    from torch.profiler import ProfilerActivity, profile

    from tensorfold.families.glm_moe_dsa.cuda import engine as eng_mod, forward as fwd, multi as multi_mod
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp

    fwd.PROMPT_REDUCE = "gather"
    calls = Calls() if os.environ.get("TF_ROUNDS_TRACE") == "1" else None
    if calls is not None:
        from tensorfold.families.glm5_next.cuda import decode as g5dec
        from tensorfold.families.glm_moe_dsa.cuda import mtp as mtp_mod

        calls.wrap(g5dec, "sample_rows", "sample_rows")
        calls.wrap(mtp_mod, "mtp_compute_rows", "mtp_rows")
        calls.wrap(multi_mod.MultiDecoder, "_sample_spans", "sample_spans")
        calls.wrap(multi_mod.MultiDecoder, "_draft_all", "draft_all")
        calls.wrap(multi_mod.MultiDecoder, "_forward", "forward")
    multi_mod.Watchdog = multi_mod.Link = Quiet
    eng_mod._store = lambda comm: None
    t0 = time.time()
    eng = eng_mod.GlmEngine(model, rank=0, master="127.0.0.1", port=29690, policy="3", context=context,
                            context_explicit=True, comm=Alone(0, int(os.environ.get("TF_TP_WORLD", "4"))),
                            parallel=max(groups))
    multi, sched = eng.multi, eng.scheduler
    print(f"-- loaded in {time.time() - t0:.0f} s: {len(multi.slots)} slots of {multi.slot_cap} rows, MTP depth "
          f"{multi.depth}, {'eager' if os.environ.get('TF_ROUNDS_EAGER') == '1' else 'graph'} rounds", flush=True)
    app = GlmApp(eng, model, "GLM")
    topics = ["how a hash table works", "the history of the printing press", "how vaccines train the immune system",
              "the rules of chess for a beginner", "how a CPU executes an instruction", "the water cycle",
              "how compilers optimize loops", "the causes of the French Revolution"]
    text, depth = os.environ.get("TF_ROUNDS_TEXT"), int(os.environ.get("TF_ROUNDS_DEPTH") or 0)
    if text and depth:
        raw = Path(text).read_text(errors="ignore")
        bodies = []
        for i in range(len(topics)):
            start = (i * 7919 * 64) % max(1, len(raw) - depth * 8)
            bodies.append(app.tok.decode(app.tok.encode(raw[start:start + depth * 8],
                                                        add_special_tokens=False).ids[:depth]))
        contents = [b + "\n\nContinue the text above." for b in bodies]
    else:
        contents = [f"Write a detailed explanation of {t}." for t in topics]
    chat = [app._prepare({"messages": [{"role": "user", "content": c}], "max_tokens": 4096, "temperature": 0,
                          "chat_template_kwargs": {"enable_thinking": False}}, True).prompt for c in contents]
    print(f"-- prompts of {min(len(c) for c in chat)}-{max(len(c) for c in chat)} tokens", flush=True)
    seen = {}
    for n in groups:
        stop = [False]
        threads = [threading.Thread(target=lambda q=q: sched.submit(list(q), 4096, None, True, lambda new: stop[0],
                                                                   stop_eos=False)) for q in chat[:n]]
        for th in threads:
            th.start()
        while len([s for s in list(multi.streams.values()) if not s.done]) < n or multi.filling:
            time.sleep(0.05)
        seen0 = len(multi.round_log)                     # three whole rounds of every stream (graphs captured)
        while sum(1 for x in multi.round_log[seen0:] if x[0] == n) < 3:
            time.sleep(0.05)
        time.sleep(1.0)
        a = len(multi.round_log)
        routed = []                                      # distinct routed experts of a round's last MoE layer
        spans0 = multi._sample_spans

        def spy(logits, spans, *, draft=False, **kw):
            if not draft:
                R = sum(sp[1] for sp in spans)
                p = multi.buf.pick[:R]
                routed.append(int(torch.unique(p[p < eng.w.cfg.experts]).numel()))
            return spans0(logits, spans, draft=draft, **kw)

        multi._sample_spans = spy
        quiet0 = dict(multi.stage_s)
        c0 = (dict(calls.s), dict(calls.n)) if calls is not None else None
        time.sleep(3.0)
        plain = multi.round_log[a:]
        multi._sample_spans = spans0
        quiet = {k: 1e3 * (multi.stage_s[k] - quiet0[k]) / max(1, len(plain)) for k in multi.stage_s}
        c1 = (dict(calls.s), dict(calls.n)) if calls is not None else None
        graphs = multi.graphs
        if os.environ.get("TF_ROUNDS_EAGER") == "1":
            multi.graphs = None
        logged = len(multi.round_log)
        stages0 = dict(multi.stage_s)
        with profile(activities=[ProfilerActivity.CUDA]) as kp:
            time.sleep(secs)
            torch.cuda.synchronize()
        log = multi.round_log[logged:]
        multi.graphs = graphs
        stop[0] = True
        for th in threads:
            th.join()
        rounds = max(1, len(log))
        times = {k: v / rounds for k, v in kernels(kp).items()}
        total = sum(times.values())
        rows = sum(x[1] for x in log) / rounds
        wall = sum(x[2] for x in plain) / max(1, len(plain)) * 1e3
        seen[n] = (rows, times)
        print(f"-- {n} stream{'s' if n > 1 else ''} at once: {rows:.1f} rows a round; wall {wall:.1f} ms a round "
              f"({len(plain)} rounds unprofiled; distinct routed experts in the last MoE layer "
              f"{sum(routed) / max(1, len(routed)):.1f}), {sum(x[2] for x in log) / rounds * 1e3:.1f} ms profiled; kernel time "
              f"{total:.2f} ms a round ({len(log)} rounds)", flush=True)
        print("   host stages a round, unprofiled (ms): " + ", ".join(f"{k} {v:.2f}" for k, v in quiet.items()), flush=True)
        if calls is not None:
            print("   calls a round, unprofiled: " + calls.report(c0, max(1, len(plain)), c1), flush=True)
        print("   host stages a round, profiled (ms): " + ", ".join(
            f"{k} {1e3 * (multi.stage_s[k] - stages0[k]) / rounds:.2f}" for k in multi.stage_s), flush=True)
        for name, t in sorted(times.items(), key=lambda kv: -kv[1])[:top]:
            print(f"    {t:7.3f} ms a round {100 * t / max(total, 1e-9):5.1f}%  {name}", flush=True)
    lo, hi = min(seen), max(seen)
    if hi > lo and seen[hi][0] > seen[lo][0]:
        (r0, t0s), (r1, t1s) = seen[lo], seen[hi]
        dr = r1 - r0
        slope = {k: (t1s.get(k, 0.0) - t0s.get(k, 0.0)) / dr for k in set(t0s) | set(t1s)}
        base = sum(t0s.values()) - r0 * sum(slope.values())
        print(f"-- a row's kernel time ({lo} -> {hi} streams, {r0:.1f} -> {r1:.1f} rows): "
              f"{sum(slope.values()):.3f} ms a row on a {base:.2f} ms base", flush=True)
        for name, s in sorted(slope.items(), key=lambda kv: -kv[1])[:top]:
            print(f"    {s:7.3f} ms a row  ({t0s.get(name, 0.0):6.3f} -> {t1s.get(name, 0.0):6.3f} ms)  {name}",
                  flush=True)


if __name__ == "__main__":
    main()
