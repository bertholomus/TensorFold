"""One CP4 stream on the real pack, no server: load, pool, rounds, greedy decode, memory and time on every step.

    python3 tools/ml4/cp4_run.py PACK --rank R --world 4 --master IP --port P --rows 262144 --tokens ids.json --decode 8

``--tokens`` is a JSON list of ids; ``--len N`` draws N ids instead (a speed and memory run: its text means nothing).
``--deep POS`` copies the rows the prompt wrote across the whole pool and runs one more round and the decode at position
POS: the time and the memory of a full-depth round without the hours of prefill in front of it. ``CP4_TRACE=1`` prints
every call of the first round that adds 16 MiB or more of host memory. ``--prof N`` ends the run with N decode steps and
a round under cp4_prof's events, then one of each under kineto. ``--tp N`` runs N decode tokens through tp.py's step from
the end of the prompt, eager then in a CUDA graph, and compares their ids with the rounds'. ``--streams 2,4,8`` then times
the step with that many rows: random tokens over the prompt's rows, a stand-in for that many streams at this depth.
``--nll N`` teacher-forces ``--tokens``: every row of the rounds, then its last N tokens through the decode step, each with
its loss, top id and the gap to the second id, into OUT.nll.json (tools/ml4/cp4_nll_diff.py compares two of them).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from cp4_prof import Prof, kernels
from tensorfold.cuda.comm import open_comm
from tensorfold.families.mistral_large4.cuda import cp4, tp
from tensorfold.families.mistral_large4.cuda.cp4_cuda import CudaOps, trim_host
from tensorfold.families.mistral_large4.cuda.cp4_fused import forward
from tensorfold.families.mistral_large4.cuda.engine import read_mem_avail
from tensorfold.families.mistral_large4.cuda.model import rms_norm, softcap
from tensorfold.families.mistral_large4.cuda.tp_cuda import FusedTp, fast_comm
from tensorfold.families.mistral_large4.cuda.weights import load


def row_check(a, lane, seq, st, w, total, first) -> None:
    """A verify step's rows against the same tokens one step each: the residual stream at every layer and the logits, bit for bit."""

    n, L = a.rowcheck, w.cfg.n_layers
    lane.taps = torch.zeros((L + 1, n, w.cfg.dim), dtype=torch.float32, device="cuda")
    lane.tap_at = {i: i for i in range(L + 1)}
    graphs = tp.Graphs(lane)
    toks = [first] + torch.randint(1000, 100000, (n - 1,), generator=torch.Generator().manual_seed(1)).tolist()
    ps = [total + a.tp + i for i in range(n)]

    def rows(idx):
        return tp.make_rows([seq] * len(idx), [ps[i] for i in idx], a.rank, "cuda", n=tp.width(st, ps[-1] + 1, a.world, w.cfg.idx_topk),
                            k=w.cfg.idx_topk, scratch=a.rows)

    one = []
    for i in range(n):
        _, logits = graphs.run(torch.tensor([toks[i]], dtype=torch.long, device="cuda"), rows([i]))
        one.append((logits[0].clone(), lane.taps[:, 0].clone()))
    _, logits = graphs.run(torch.tensor(toks, dtype=torch.long, device="cuda"), rows(list(range(n))))
    for i in range(n):
        x = lane.taps[:, i]
        bad = (x != one[i][1]).flatten(1).any(1).nonzero().flatten().tolist()
        diff = float((logits[i] - one[i][0]).abs().max())
        if a.rank == 0:
            at = bad[0] if bad else -1
            gap = float((x[at] - one[i][1][at]).abs().max()) if bad else 0.0
            print(f"[cp4] rowcheck row {i}: {len(bad)} of {L + 1} layers differ, first {at} by {gap:.3g}; logits differ by {diff:.3g}, "
                  f"argmax {int(logits[i].argmax())} {int(one[i][0].argmax())}", flush=True)
    lane.taps, lane.tap_at = None, {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--master", default="")
    ap.add_argument("--port", type=int, default=29561)
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--tokens", default="")
    ap.add_argument("--len", type=int, default=0)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--block", type=int, default=cp4.BLOCK)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--deep", type=int, default=0)
    ap.add_argument("--copy", type=int, default=0, help="rows of the prompt's copy on every rank: its rounds run over it")
    ap.add_argument("--slots", action="store_true", help="a prompt's rounds run over its rows gathered a layer ahead")
    ap.add_argument("--prof", type=int, default=0)
    ap.add_argument("--tp", type=int, default=0)
    ap.add_argument("--streams", default="")
    ap.add_argument("--rowcheck", type=int, default=0, help="rows of one step against that many steps of one row: where they part")
    ap.add_argument("--nll", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    t0 = time.time()

    def mem(what: str) -> None:
        st = dict(line.split(":", 1) for line in open("/proc/self/status") if line.startswith("Rss"))
        host = ", ".join(f"{k[3:].lower()} {int(v.split()[0]) / 2 ** 20:.2f}" for k, v in st.items() if k != "RssFile")
        print(f"[cp4] rank {a.rank} {time.time() - t0:7.1f} s {what}: MemAvailable {read_mem_avail() / 2 ** 30:.2f} GiB, "
              f"torch alloc {torch.cuda.memory_allocated() / 2 ** 30:.2f}, reserved "
              f"{torch.cuda.memory_reserved() / 2 ** 30:.2f}, peak {torch.cuda.max_memory_reserved() / 2 ** 30:.2f}, "
              f"host {host}", flush=True)

    def host_objects() -> None:
        """What the Python heap still holds after the load: the largest host arrays."""

        import gc

        import numpy as np

        sizes: dict = {}
        for o in gc.get_objects():
            try:
                if isinstance(o, np.ndarray) and o.base is None:
                    sizes[f"ndarray {o.dtype} {o.shape}"] = sizes.get(f"ndarray {o.dtype} {o.shape}", 0) + o.nbytes
                elif isinstance(o, torch.Tensor) and o.device.type == "cpu":
                    key = f"tensor {o.dtype} {tuple(o.shape)}"
                    sizes[key] = sizes.get(key, 0) + o.numel() * o.element_size()
            except Exception:                                    # noqa: BLE001  (a half-built object in the walk)
                continue
        top = sorted(sizes.items(), key=lambda kv: -kv[1])[:8]
        print(f"[cp4] rank {a.rank} host arrays total {sum(sizes.values()) / 2 ** 30:.2f} GiB, top {top}", flush=True)

    page = os.sysconf("SC_PAGE_SIZE")

    def anon() -> int:
        fields = open("/proc/self/statm").read().split()
        return (int(fields[1]) - int(fields[2])) * page

    def trace(owner, names, who: str) -> list:
        """Wrap ``names`` on ``owner``; the returned list ends the trace when cleared of its first item."""

        live = [True]
        top = [torch.cuda.max_memory_allocated()]

        def wrap(name, fn):
            def call(*args, **kwargs):
                if not live:
                    return fn(*args, **kwargs)
                before = anon()
                out = fn(*args, **kwargs)
                grown = anon() - before
                if grown >= 16 << 20:
                    print(f"[cp4] rank {a.rank} trace {who}.{name} +{grown / 2 ** 20:.0f} MiB (anon {anon() / 2 ** 30:.2f})",
                          flush=True)
                peak = torch.cuda.max_memory_allocated()
                if peak >= top[0] + (8 << 20):
                    top[0] = peak
                    print(f"[cp4] rank {a.rank} trace {who}.{name} raised the peak to {peak / 2 ** 30:.3f} GiB, live "
                          f"{torch.cuda.memory_allocated() / 2 ** 30:.3f}", flush=True)
                return out
            return call

        for name in names:
            setattr(owner, name, wrap(name, getattr(owner, name)))
        return live

    torch.cuda.set_device(0)
    mem("start")
    comm = open_comm(a.rank, a.world, a.master, a.port) if a.world > 1 else None
    if comm is not None:
        comm.barrier()
        comm = fast_comm(comm)
    mem("comm up")
    w = load(a.pack, rank=a.rank, world=a.world)
    mem("loaded")
    ops = CudaOps()
    ops.prepare(w)
    torch.cuda.empty_cache()
    mem("kv_b fp16, embedding mapped")
    if a.rank == 0 and os.environ.get("CP4_HOST_OBJECTS"):
        host_objects()
    pool = cp4.KvPool(w.cfg, a.rows + 1 + a.copy, device="cuda", kind=os.environ.get("TF_ML4_KV_KIND") or "fp8")
    mem(f"pool {a.rows} rows {pool.nbytes() / 2 ** 30:.2f} GiB")
    fwd = forward(w, comm, pool, ops)
    if a.slots:
        fwd.open_slots(a.rows * a.world)
    st = cp4.Stripe(a.world, a.block)
    seq = cp4.Seq(st, 0, a.rows, *((a.rows + 1, a.copy) if a.copy else ()))
    wb = a.round or a.world * a.block
    traces = []
    if os.environ.get("CP4_TRACE"):
        traces.append(trace(ops, ("embed", "lin", "mat", "router", "local_topk", "attend", "absorb_q", "expand_o", "experts",
                                  "logits"), "ops"))
        traces.append(trace(fwd, ("_layer", "_select", "_attention", "_moe"), "fwd"))
        if comm is not None:
            traces.append(trace(comm, ("all_gather", "alltoall"), "comm"))

    if a.tokens:
        ids = json.load(open(a.tokens))
    else:
        ids = torch.randint(1000, w.cfg.vocab - 1000, (a.len,), generator=torch.Generator().manual_seed(11)).tolist()
    every_id = torch.tensor(ids, dtype=torch.long, device="cuda")
    if a.nll:
        ids = ids[:-a.nll]
    total = len(ids)
    end = (a.deep + wb if a.deep else total) + a.decode + a.nll
    assert st.rows_for(end, 0) <= a.rows, f"{end} tokens do not fit {a.rows} rows a rank"
    ids_t = torch.tensor(ids, dtype=torch.long)
    logits = None
    head = w.head if w.head is not None else w.head_bf16
    forced: dict = {"rounds": [], "steps": []}

    def graded(lg: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Rows of (position, loss of the next id, top id, gap to the second) for logits [R, V] at ``pos``."""

        lg = lg.float()
        nxt = every_id[(pos + 1).clamp(max=every_id.numel() - 1)]
        v, i = lg.topk(2, dim=1)
        return torch.stack([pos.float(), lg.logsumexp(1) - lg.gather(1, nxt[:, None])[:, 0], i[:, 0].float(), v[:, 0] - v[:, 1]], 1)

    def grade_round(f: cp4.Flight) -> None:
        got = torch.zeros((f.rnd.own.numel(), 4), device="cuda")
        for c in range(0, got.shape[0], 128):                    # the head at a decode's width: 105 MB of logits
            h = rms_norm(f.x[c:c + 128], w.norm, w.cfg.eps)
            got[c:c + 128] = graded(softcap(ops.logits(head, h), w.cfg.softcap), f.rnd.own[c:c + 128])
        forced["rounds"] += cp4.gather_rows(comm, got, f.rnd.counts).tolist()

    with torch.inference_mode():
        for start in range(0, total, wb):
            n = min(wb, total - start)
            p = torch.arange(start, start + n)
            mine = ids_t[start:start + n][st.rank(p) == a.rank].cuda()
            t1 = time.time()
            if a.nll:
                f = fwd.begin(seq, start, n, mine, total - 1 if start + n == total else None)
                fwd.advance(f, w.cfg.n_layers)
                logits = fwd.finish(f)
                for g in (f.a, f.b) if hasattr(f, "a") else (f,):
                    grade_round(g)
            else:
                logits = fwd.round(seq, start, n, mine, last=total - 1 if start + n == total else None)
            torch.cuda.synchronize()
            dt = time.time() - t1
            if a.rank == 0:
                mem(f"round {start // wb + 1}/{-(-total // wb)} pos {start + n} {dt:.2f} s, {n / dt:.0f} tok/s")
            if start == 0:
                trim_host()
                for live in traces:
                    live.clear()
        mem("prefilled")
        if a.nll:
            stepped = os.environ.get("TF_ML4_TP", "1") != "0"
            if stepped:
                lane = FusedTp(fwd)
                lane.use(st)
                graphs = tp.Graphs(lane)
            t1 = time.time()
            for i in range(a.nll):
                p = total + i
                tok_i = every_id[p:p + 1]
                at = torch.tensor([p], device="cuda")
                if stepped:
                    rows = tp.make_rows([seq], [p], a.rank, "cuda", n=tp.width(st, p + 1, a.world, w.cfg.idx_topk),
                                        k=w.cfg.idx_topk, scratch=a.rows)
                    part = graphs.run(tok_i, rows)[1]
                    every = torch.empty((a.world, *part.shape), dtype=part.dtype, device="cuda")
                    comm.all_gather(part.reshape(-1), every.view(-1))
                    row = graded(every.permute(1, 0, 2).reshape(1, -1), at)
                else:
                    owner = int(st.rank(torch.tensor(p)))
                    lg = fwd.round(seq, p, 1, tok_i if owner == a.rank else tok_i[:0], last=p)
                    row = graded(lg[None], at) if lg is not None else torch.zeros((1, 4), device="cuda")
                    every = torch.empty((a.world, 4), device="cuda")
                    comm.all_gather(row.view(-1), every.view(-1))
                    row = every[owner:owner + 1]
                forced["steps"] += row.tolist()
            torch.cuda.synchronize()
            if a.rank == 0:
                for part, rows_of in forced.items():
                    print(f"[cp4] forced {part}: {len(rows_of)} rows, mean loss "
                          f"{sum(r[1] for r in rows_of[:-1]) / max(len(rows_of) - 1, 1):.5f}", flush=True)
                print(f"[cp4] forced steps {a.nll} in {time.time() - t1:.1f} s", flush=True)
                if a.out:
                    json.dump(forced, open(f"{a.out}.nll.json", "w"))
            mem("done")
            return
        if a.deep:
            assert total % wb == 0 and a.deep % wb == 0 and not a.tokens, "--deep takes whole rounds of drawn ids"
            done = st.rows_for(total, 0)
            for plane in (*pool.lat.values(), *pool.kpe.values(), *pool.keys.values()):
                for c in range(done, a.rows, done):
                    m = min(done, a.rows - c)
                    plane[c:c + m].copy_(plane[:m])
                for c in range(total, a.copy, total):
                    m = min(total, a.copy - c)
                    plane[seq.rep + c:seq.rep + c + m].copy_(plane[seq.rep:seq.rep + m])
            torch.cuda.synchronize()
            mem(f"pool filled from {done} rows")
            p = torch.arange(a.deep, a.deep + wb)
            mine = ids_t[:wb][st.rank(p) == a.rank].cuda()
            t1 = time.time()
            logits = fwd.round(seq, a.deep, wb, mine, last=a.deep + wb - 1)
            torch.cuda.synchronize()
            dt = time.time() - t1
            if a.rank == 0:
                mem(f"deep round at {a.deep}: {dt:.2f} s, {wb / dt:.0f} tok/s")
            total = a.deep + wb

        def pick(lg) -> int:
            mine = torch.tensor([-1 if lg is None else int(lg.argmax())], dtype=torch.int64, device="cuda")
            if comm is None:
                return int(mine)
            every = torch.empty((a.world,), dtype=torch.int64, device="cuda")
            comm.all_gather(mine, every)
            return int(every.max())

        out, top = [], None
        if logits is not None:
            v, i = logits.float().topk(5)
            top = list(zip(i.tolist(), [round(x, 3) for x in v.tolist()]))
            print(f"[cp4] rank {a.rank} top-5 after the prompt: {top}", flush=True)
        tok = pick(logits)
        t1 = time.time()
        for s in range(a.decode):
            out.append(tok)
            pos = total + s
            mine = torch.tensor([tok], dtype=torch.long, device="cuda")
            mine = mine if int(st.rank(torch.tensor(pos))) == a.rank else mine[:0]
            tok = pick(fwd.round(seq, pos, 1, mine, last=pos))
        out.append(tok)
        torch.cuda.synchronize()
        dt = time.time() - t1
        if a.rank == 0:
            print(f"[cp4] greedy ids {out}", flush=True)
            print(f"[cp4] decode {a.decode} tokens {dt:.2f} s, {a.decode / max(dt, 1e-9):.2f} tok/s", flush=True)
            if a.out:
                json.dump({"ids": out, "top": top}, open(a.out, "w"))
        if a.tp:
            lane = tp.Ml4Tp(fwd)
            lane.use(st)
            fused = FusedTp(fwd)
            fused.use(st)
            first, ref = out[0], out + [tok]

            def run(steps, graphs=None) -> list:
                got, t = [], first
                t1 = time.time()
                for i in range(steps):
                    got.append(t)
                    p = total + i
                    tt = torch.tensor([t], dtype=torch.long, device="cuda")
                    rows = tp.make_rows([seq], [p], a.rank, "cuda", n=tp.width(st, p + 1, a.world, w.cfg.idx_topk),
                                        k=w.cfg.idx_topk, scratch=a.rows)
                    if graphs is None:
                        t = int(lane.best(lane.step(tt, rows))[0])
                    else:
                        t = int(graphs.run(tt, rows)[0][0])
                torch.cuda.synchronize()
                got.append(t)
                return got, time.time() - t1

            got, dt = run(a.tp)
            same = sum(1 for x, y in zip(got, ref) if x == y)
            if a.rank == 0:
                print(f"[cp4] tp eager {a.tp} tokens {dt:.2f} s, {a.tp / dt:.2f} tok/s; ids {got}; {same}/{min(len(got), len(ref))} "
                      f"as the rounds", flush=True)
            for name, one in (("graph", lane), ("fused graph", fused)):
                graphs = tp.Graphs(one)
                got, dt = run(a.tp, graphs)
                mem(f"{name} built")
                got, dt = run(a.tp, graphs)
                same = sum(1 for x, y in zip(got, ref) if x == y)
                if a.rank == 0:
                    print(f"[cp4] tp {name} {a.tp} tokens {dt:.2f} s, {a.tp / dt:.2f} tok/s; ids {got}; "
                          f"{same}/{min(len(got), len(ref))} as the rounds", flush=True)
            if a.prof:
                rows = tp.make_rows([seq], [total], a.rank, "cuda", n=tp.width(st, total + 1, a.world, w.cfg.idx_topk),
                                    k=w.cfg.idx_topk, scratch=a.rows)
                tt = torch.tensor([first], dtype=torch.long, device="cuda")
                kernels(lambda: graphs.run(tt, rows), "fused step", a.rank, top=44)
            if a.rowcheck:
                row_check(a, fused, seq, st, w, total, first)
            for n in (int(x) for x in a.streams.split(",") if x):
                tokens = torch.randint(1000, 100000, (n,), generator=torch.Generator().manual_seed(n)).cuda()

                def rows_at(i: int):
                    ps = [total + i - 64 * j for j in range(n)]
                    return tp.make_rows([seq] * n, ps, a.rank, "cuda", n=tp.width(st, max(ps) + 1, a.world, w.cfg.idx_topk),
                                        k=w.cfg.idx_topk, scratch=a.rows)

                for i in range(4):
                    graphs.run(tokens, rows_at(i))
                torch.cuda.synchronize()
                t1 = time.time()
                for i in range(4, 4 + a.tp):
                    tokens = graphs.run(tokens, rows_at(i))[0][:n].clone()
                torch.cuda.synchronize()
                dt = (time.time() - t1) / a.tp
                if a.rank == 0:
                    print(f"[cp4] tp {n} streams: {dt * 1e3:.1f} ms a step, {n / dt:.1f} tok/s in all, {1 / dt:.1f} a stream", flush=True)
                if os.environ.get("STEP_TRACE") == str(n):
                    from torch.profiler import ProfilerActivity, profile

                    rows = rows_at(4 + a.tp)
                    with profile(activities=[ProfilerActivity.CUDA]) as p:
                        graphs.run(tokens, rows)
                        torch.cuda.synchronize()
                    p.export_chrome_trace(f"{a.out}.r{a.rank}.json")
                if a.prof and n == max(int(x) for x in a.streams.split(",")):
                    rows = rows_at(4 + a.tp)
                    kernels(lambda: graphs.run(tokens, rows), f"fused step {n}", a.rank, top=30)
            mem("tp done")
        if a.prof:
            pos = total + a.decode
            start = -(-(pos + 2 * a.prof + 2) // wb) * wb

            def step() -> None:
                nonlocal pos, tok
                mine = torch.tensor([tok], dtype=torch.long, device="cuda")
                mine = mine if int(st.rank(torch.tensor(pos))) == a.rank else mine[:0]
                tok = pick(fwd.round(seq, pos, 1, mine, last=pos))
                pos += 1

            def one_round() -> None:
                nonlocal start
                p = torch.arange(start, start + wb)
                fwd.round(seq, start, wb, ids_t[:wb].repeat(-(-wb // len(ids)))[:wb][st.rank(p) == a.rank].cuda())
                start += wb

            assert st.rows_for(start + 2 * wb, 0) <= a.rows, f"--prof needs {st.rows_for(start + 2 * wb, 0)} rows"
            prof = Prof()
            prof.attach(ops, fwd, comm)
            prof.on = True
            t1 = time.time()
            for _ in range(a.prof):
                step()
            torch.cuda.synchronize()
            prof.on = False
            prof.table("decode", time.time() - t1, a.prof, a.rank, a.out)
            mem("decode profiled")
            kernels(step, "decode", a.rank)
            prof.on = True
            t1 = time.time()
            one_round()
            torch.cuda.synchronize()
            prof.on = False
            prof.table("round", time.time() - t1, 1, a.rank, a.out)
            mem("round profiled")
            kernels(one_round, "round", a.rank)
    mem("done")


if __name__ == "__main__":
    main()
