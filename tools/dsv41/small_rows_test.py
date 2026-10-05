"""Bits and speed of the verify window's small kernels (kernels.SMALL_SWITCHES "hc_split", "hc_rot", "hc_rb",
"rowmm_parts", "qkv_split", "attn_pf") on synthetic inputs of the real shapes (no weights loaded: a few MB of GPU
memory): every output of the new path against the old one (torch.equal on the bits) at 1..16 rows and several input
scales, each row alone against the same row in the window, then CUDA-graph timing of the MoE mHC step (gather
stand-in -> mixes + finish -> router matmul -> route) old against new, interleaved (--hog MiB: with the side stream's
paced L2 prefetch beside it).

  python3 small_rows_test.py [--rows 1,2,3,4,5,6,8,16] [--trials 40] [--iters 50] [--out F]
"""

import argparse
import json
import statistics

import torch

BF16, F32 = torch.bfloat16, torch.float32


def beq(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        iv = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))
    return torch.equal(a, b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="1,2,3,4,5,6,8,16")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--dim", type=int, default=5120)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--profile", action="store_true", help="per-kernel exclusive and total time of each variant's step")
    p.add_argument("--hog", type=int, default=0, help="MiB of L2 prefetch (exl3/prefetch.py, paced) forked beside each "
                   "step from a 512 MiB buffer: the decode graphs' side-stream DRAM traffic")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    D, E = a.dim, a.experts
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(5)
    sw0 = dict(K.SMALL_SWITCHES)
    NEW = ("hc_split", "hc_rot", "hc_rb", "hc_xpf", "rowmm_parts", "qkv_split", "attn_pf", "hc_defer", "topk_fused",
           "merge_reg", "hc_dots")
    sink = torch.cuda.Stream()                  # (switch "hc_defer") the Sinkhorn half's side stream

    def sink_on():
        return sink if K.hc_defer_on(D) else None

    def join():
        torch.cuda.current_stream().wait_stream(sink)

    def mode(on):
        for k in NEW:
            K.set_switch(k, on if isinstance(on, bool) else (k in on))

    def rnd(*shape, scale=1.0, dtype=BF16):
        return (torch.randn(shape, generator=g, device=dev) * scale).to(dtype)

    from tensorfold.cuda.exl3.linear import _ext
    ext = _ext()
    suhs = [(torch.randn(D, generator=g, device=dev) * 0.5).to(torch.float16) for _ in range(4)]
    fn = rnd(24, 4 * D, scale=0.02, dtype=F32)
    scale = torch.rand(3, generator=g, device=dev) * 2
    base = rnd(24, scale=0.5, dtype=F32)
    norm_w = (1 + rnd(D, scale=0.1, dtype=F32)).to(BF16)
    gate_w = rnd(E, D, scale=0.03, dtype=torch.float16)
    gate_b = rnd(E, scale=0.1, dtype=F32)
    res = {"bits": {}, "time": {}}

    def record(name, ok):
        r = res["bits"].setdefault(name, [0, 0])
        r[0] += 1
        r[1] += int(bool(ok))

    def mhc_inputs(R, sc):
        h = rnd(R, 4, D, scale=sc)
        gathered = rnd(2, R, D, scale=sc, dtype=F32)
        pre_in = torch.rand(R, 4, generator=g, device=dev) + 0.1
        x0 = torch.empty(R, D, dtype=BF16, device=dev)
        part = torch.empty(R * K.HC_BLOCKS * 32, dtype=F32, device=dev)
        po, post, comb = (torch.empty(R, 4, device=dev), torch.empty(R, 4, device=dev),
                          torch.empty(R, 4, 4, device=dev))
        mode(False)
        K.hc_pre(h, fn, scale, base, pre_in, norm_w, 1e-6, 1e-6, 20, x0, po, post, comb, part)
        return h, gathered, pre_in, post, comb

    def mhc_run(h, gathered, pre_in, post, comb, posted=True):
        R = h.shape[0]
        x = torch.empty(R, D, dtype=BF16, device=dev)
        pre_out = torch.empty(R, 4, device=dev)
        p2, c2 = post.clone(), comb.clone()
        part = torch.zeros(R * K.HC_BLOCKS * 32, dtype=F32, device=dev)     # (slots 25..31 of a block: unwritten)
        if posted:
            hn = torch.empty_like(h)
            K.hc_pre2(h, fn, scale, base, pre_in, norm_w, 1e-6, 1e-6, 20, x, pre_out, p2, c2, part,
                      gathered=gathered, h_out=hn, sink=sink_on())
        else:
            hn = h
            K.hc_pre2(h, fn, scale, base, pre_in, norm_w, 1e-6, 1e-6, 20, x, pre_out, p2, c2, part, sink=sink_on())
        join()
        return hn, x, pre_out, p2, c2, part

    rows = [int(r) for r in a.rows.split(",")]
    scales = [1e-3, 0.05, 1.0, 30.0, 1.0, 5.0, 300.0, 0.3]
    for trial in range(a.trials):
        sc = scales[trial % len(scales)]
        for R in rows:
            h, gathered, pre_in, post, comb = mhc_inputs(R, sc)
            for posted in (True, False):
                mode(False)
                old = mhc_run(h, gathered, pre_in, post, comb, posted)
                for name, on in (("new", True), ("hc_split", ("hc_split",)), ("hc_rb", ("hc_rb",)),
                                 ("hc_defer", ("hc_split", "hc_defer")),
                                 ("hc_dots", ("hc_split", "hc_defer", "hc_dots", "hc_rb"))):
                    mode(on)
                    new = mhc_run(h, gathered, pre_in, post, comb, posted)
                    # (part: the mixes' partials, every one compared; h_out only when posted)
                    record(f"mhc {name} {'posted' if posted else 'plain'} == old",
                           all(beq(u, v) for u, v in zip(old, new)))
                # the finish's rotated rows (switch "hc_rot") against rot_many of its output rows
                mode(True)
                x3 = torch.empty(R, D, dtype=BF16, device=dev)
                rot = [(su, torch.empty(R, D, dtype=torch.float16, device=dev)) for su in suhs]
                hn3 = torch.empty_like(h)
                K.hc_pre2(h, fn, scale, base, pre_in, norm_w, 1e-6, 1e-6, 20, x3, torch.empty(R, 4, device=dev),
                          post.clone(), comb.clone(), torch.zeros(R * K.HC_BLOCKS * 32, dtype=F32, device=dev),
                          gathered=gathered if posted else None, h_out=hn3 if posted else None, rot=rot)
                refs = [torch.empty(R, D, dtype=torch.float16, device=dev) for _ in suhs]
                ext.rot_many([x3] * len(suhs), suhs, refs, False)
                record(f"mhc rot {'posted' if posted else 'plain'} == rot_many",
                       beq(x3, old[1]) and all(beq(t[1], u) for t, u in zip(rot, refs)))
                # row invariance of the new path: row i alone
                mode(True)
                new = mhc_run(h, gathered, pre_in, post, comb, posted)
                i = (trial * 7 + R) % R
                one = mhc_run(h[i:i + 1].contiguous(), gathered[:, i:i + 1].contiguous(), pre_in[i:i + 1].contiguous(),
                              post[i:i + 1].contiguous(), comb[i:i + 1].contiguous(), posted)
                record(f"mhc new {'posted' if posted else 'plain'} rows", all(beq(u[i:i + 1], v) for u, v in
                                                                              zip(new[:5], one[:5])))
            # router matmul + route
            x = rnd(R, D, scale=[0.05, 1.0, 7.0, 0.3][trial % 4])
            mode(False)
            ref = K.rowmm(x, gate_w)
            record("rowmm2 (old) == rowmm", beq(K.rowmm2(x, gate_w), ref))
            mode(True)
            i = (trial * 5 + R) % R
            perm = torch.randperm(R, generator=torch.Generator().manual_seed(trial * 100 + R)).to(dev)
            # the gate as chunk sums: summed in K order (torch's fp32 adds) == rowmm; the route's picks and weights
            parts = K.rowmm_gate(x, gate_w)
            if parts.dim() == 3:
                lg = torch.zeros(R, E, dtype=F32, device=dev)
                for t in range(parts.shape[1]):
                    lg = lg + parts[:, t]
                record("gate parts summed == rowmm", beq(lg, ref))
                one = K.rowmm_gate(x[i:i + 1].contiguous(), gate_w)
                record("gate parts rows", beq(one[0], parts[i]))
                record("gate parts perm", beq(K.rowmm_gate(x[perm].contiguous(), gate_w), parts[perm]))
            outs = []
            for lgx in (ref, parts):
                pk = torch.empty(R, 7, dtype=torch.int32, device=dev)
                wt = torch.empty(R, 7, dtype=F32, device=dev)
                K.route(lgx, gate_b, 6, 2.5, E, pk, wt)
                outs.append((pk, wt))
            record("route(parts) == route(rowmm)", beq(outs[0][0], outs[1][0]) and beq(outs[0][1], outs[1][1]))
    # q_kv_norm: the rotations split into programs of their own (switch "qkv_split") against one program a row
    from tensorfold.families.deepseek_v41.ops import freqs_cis
    f = freqs_cis(64, 4096, 65536, 160000.0, 16.0, 32, 1)
    cos, sin = f.real.contiguous().float().to(dev), f.imag.contiguous().float().to(dev)
    qn = (1 + rnd(1280, scale=0.1, dtype=F32)).to(BF16)
    kn = (1 + rnd(512, scale=0.1, dtype=F32)).to(BF16)
    su = [(torch.randn(1280, generator=g, device=dev) * 0.5).to(torch.float16) for _ in range(2)]
    for trial in range(a.trials):
        for R in rows:
            qa = rnd(R, 1280, scale=[0.3, 1.0, 3.0][trial % 3])
            ykv = rnd(R, 512, scale=[0.3, 1.0, 30.0][trial % 3])
            pos = torch.randint(0, 4096, (R,), generator=g, device=dev, dtype=torch.int64)
            slots = torch.randperm(4 * 144, generator=torch.Generator().manual_seed(trial + R))[:R].to(dev)
            for nrot in (1, 2):
                outs = []
                for on in (False, True):
                    K.set_switch("qkv_split", on)
                    ring = torch.zeros(4 * 144, 512, dtype=BF16, device=dev)
                    rot = [(su[i], torch.empty(R, 1280, dtype=torch.float16, device=dev)) for i in range(nrot)]
                    o = K.q_kv_norm(qa, qn, 1e-6, rot, ykv, kn, cos, sin, pos, ring, slots, True, 64)
                    outs.append([o, ring] + [t[1] for t in rot])
                record("q_kv_norm split == one program", all(beq(u, v) for u, v in zip(outs[0], outs[1])))
    # the decode attention (split keys, merge + inverse RoPE + wo_a's rotation) with and without "attn_pf"
    from tensorfold.families.deepseek_v41.ops import fp4_pack
    hd, rd, Hl, gh, RS = 512, 64, 32, 8, 144
    sink_a = rnd(Hl, scale=1.0, dtype=F32)
    suh_o = (torch.randn(4 * gh * hd, generator=g, device=dev) * 0.5).to(torch.float16)
    for trial in range(a.trials):
        for R in rows:
            ring = rnd(4 * RS, hd)
            cp = fp4_pack(rnd(1024, hd), 16, True)
            sk = rnd(R, Hl, hd, scale=[0.05, 0.3, 1.0, 3.0][trial % 4])
            pos = torch.randint(0, 1000, (R,), generator=g, device=dev, dtype=torch.int64)
            slot = torch.randint(0, 4, (R,), generator=g, device=dev, dtype=torch.int64)
            idx = torch.sort(torch.randint(0, 512, (R, 512), generator=g, device=dev), dim=-1).values
            idx = torch.where(idx < (pos[:, None] + 1) // 2, idx, -1)
            outs = []
            for on, mreg in ((False, False), (True, False), (True, True)):
                K.set_switch("attn_pf", on)
                K.set_switch("merge_reg", mreg)                # (the last variant: the round-2 merge)
                xh = torch.empty(4 * R * gh * hd, dtype=torch.float16, device=dev)
                o = K.sparse_attn(sk, sink_a, ring, torch.zeros(1, dtype=torch.int64, device=dev), True, cp, idx, pos,
                                  hd ** -0.5, 128, wbase=slot * RS, cbase=torch.zeros(R, dtype=torch.int64, device=dev),
                                  ring_rows=RS, rot=(cos, sin, rd, suh_o, xh, gh))
                outs.append((o.clone(), xh.clone()))
            record("attn (attn_pf) == old", beq(outs[0][0], outs[1][0]) and beq(outs[0][1], outs[1][1]))
            record("attn (merge_reg) == old", beq(outs[0][0], outs[2][0]) and beq(outs[0][1], outs[2][1]))
            # row invariance of the new attention: row i alone (its own ring slot base, picks and position)
            i = (trial * 3 + R) % R
            xh1 = torch.empty(4 * 1 * gh * hd, dtype=torch.float16, device=dev)
            o1 = K.sparse_attn(sk[i:i + 1].contiguous(), sink_a, ring, torch.zeros(1, dtype=torch.int64, device=dev),
                               True, cp, idx[i:i + 1].contiguous(), pos[i:i + 1].contiguous(), hd ** -0.5, 128,
                               wbase=(slot * RS)[i:i + 1].contiguous(),
                               cbase=torch.zeros(1, dtype=torch.int64, device=dev), ring_rows=RS,
                               rot=(cos, sin, rd, suh_o, xh1, gh))
            xr = outs[2][1].view(4, R, gh * hd)[:, i]
            record("attn (merge_reg) rows", beq(o1[0], outs[2][0][i]) and beq(xh1.view(4, gh * hd), xr))
    # the indexer's selection (switch "topk_fused") against torch's top-k + _topk_finish: unique keys of scores with
    # ties, -inf entries and both signs, at several widths
    for trial in range(a.trials):
        for R in rows:
            for n in (512, 1024, 2048, 4096):
                sc = torch.randn(R, n, generator=g, device=dev) * [0.5, 3.0, 100.0][trial % 3]
                sc = torch.where(torch.rand(R, n, generator=g, device=dev) < 0.2, sc.round(), sc)        # ties
                sc = torch.where(torch.rand(R, n, generator=g, device=dev) < 0.1, float("-inf"), sc)
                bits = (sc + 0.0).view(torch.int32)
                ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
                keys = (ordered << 32) | (0xFFFFFFFF - torch.arange(n, device=dev, dtype=torch.int64))
                vis = torch.randint(1, n + 1, (R,), generator=g, device=dev, dtype=torch.int64)
                outs = []
                for on in (False, True):
                    K.set_switch("topk_fused", on)
                    outs.append(K.topk_select(keys, 512, vis))
                i = (trial + R) % R
                K.set_switch("topk_fused", True)
                one = K.topk_select(keys[i:i + 1].contiguous(), 512, vis[i:i + 1].contiguous())
                record("topk_select fused == torch", beq(outs[0], outs[1]))
                record("topk_select fused rows", beq(one[0], outs[1][i]))
    ok_all = all(v[0] == v[1] for v in res["bits"].values())
    print(json.dumps({"bits": {k: f"{v[1]}/{v[0]}" for k, v in res["bits"].items()}, "all_equal": ok_all}, indent=1),
          flush=True)
    res["all_equal"] = ok_all

    if not a.no_time:
        # the MoE mHC step as the decode graph runs it: gather stand-in (a copy, no PDL) -> mixes + finish (PDL)
        # -> router matmul -> route; a graph of ``iters`` steps a variant, replays interleaved
        variants = {"old": False, "new": True, "hc_split": ("hc_split",), "hc_rb": ("hc_rb",),
                    "parts": ("rowmm_parts",), "new_nodefer": tuple(k for k in NEW if k != "hc_defer")}
        hog, side = None, None
        if a.hog:
            from tensorfold.cuda.exl3 import prefetch as PF
            big = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
            n = (512 << 20) // (a.hog << 20)
            hog = [PF.ranges([big[i * (a.hog << 20):(i + 1) * (a.hog << 20)]]) for i in range(n)]
            side = PF.SideStream()
        for R in rows:
            h, gathered, pre_in, post, comb = mhc_inputs(R, 1.0)
            src = gathered.clone()
            graphs = {}
            alive = []                      # each graph's buffers, alive as long as the graphs (they hold raw pointers)
            for name, on in variants.items():
                mode(on)
                hs = [h.clone(), torch.empty_like(h)]
                x = torch.empty(R, D, dtype=BF16, device=dev)
                pre = [pre_in.clone(), torch.empty_like(pre_in)]
                pc = [post.clone(), comb.clone()]
                part = torch.empty(R * K.HC_BLOCKS * 32, dtype=F32, device=dev)
                logits = torch.empty(R, E, dtype=F32, device=dev)
                pick = torch.empty(R, 7, dtype=torch.int32, device=dev)
                wts = torch.empty(R, 7, dtype=F32, device=dev)

                def step(j):
                    if hog is not None:
                        side.fork(hog[j % len(hog)])
                    gathered.copy_(src)
                    K.hc_pre2(hs[j % 2], fn, scale, base, pre[j % 2], norm_w, 1e-6, 1e-6, 20, x, pre[1 - j % 2],
                              pc[0], pc[1], part, gathered=gathered, h_out=hs[1 - j % 2], sink=sink_on())
                    K.route(K.rowmm_gate(x, gate_w), gate_b, 6, 2.5, E, pick, wts)
                    if sink_on() is not None:
                        join()                       # (rounds.py: before the sublayer's gather)

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for j in range(2):
                        step(j)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    for j in range(a.iters):
                        step(j)
                    if hog is not None:
                        side.join()
                graphs[name] = gr
                alive.append((hs, x, pre, pc, part, logits, pick, wts))
            if a.profile:
                from torch.profiler import ProfilerActivity, profile
                for n, gr in graphs.items():
                    gr.replay()
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        gr.replay()
                        torch.cuda.synchronize()
                    ev = sorted(((e.name, e.time_range.start, e.time_range.end) for e in prof.events()
                                 if e.device_type == torch.autograd.DeviceType.CUDA), key=lambda t: t[1])
                    tot, end = {}, None
                    for nm, t0, t1 in ev:
                        ex = t1 - (t0 if end is None else max(end, t0))
                        end = t1 if end is None else max(end, t1)
                        x = tot.setdefault(nm[:40], [0.0, 0.0, 0])
                        x[0] += max(0.0, ex)
                        x[1] += t1 - t0
                        x[2] += 1
                    print(json.dumps({"R": R, "variant": n, "excl_us": {k: round(v[0] / v[2], 2) for k, v in tot.items()},
                                      "dur_us": {k: round(v[1] / v[2], 2) for k, v in tot.items()}}), flush=True)
            ts = {n: [] for n in graphs}
            for rd in range(a.rounds):
                order = list(graphs) if rd % 2 == 0 else list(graphs)[::-1]
                for n in order:
                    graphs[n].replay()
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[n].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    ts[n].append(e0.elapsed_time(e1) * 1000 / a.iters)
            r = {n: round(statistics.median(v), 2) for n, v in ts.items()}
            res["time"][R] = r
            print(json.dumps({"R": R, "us_a_step": r}), flush=True)
            del graphs, alive
    for k, v in sw0.items():
        K.set_switch(k, v)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
