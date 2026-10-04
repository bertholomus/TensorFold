"""Bit checks of the decode round's new small kernels (kernels.SMALL_SWITCHES) against the kernels they replace, on
real weights of a few layers and random activations at several scales: every output torch.equal to the old path's, at
1 .. 16 rows, and row invariance (each row alone against inside the call, rows in random order).

  python3 small_kernels_test.py --model M [--layers 0,2,20,24] [--trials 6] [--out F]
"""

import argparse
import json
import random

import torch

BF16, F32 = torch.bfloat16, torch.float32


def beq(a, b):
    """Bitwise equality (torch.equal, but NaN payloads compare by their bits too)."""

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        iv = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))
    return torch.equal(a, b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="0,2,20,24")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.cuda.exl3.linear import Exl3Group, _ext
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda import weights as W

    cfg = Cfg.read(a.model)
    sh = W.Shards(a.model)
    lays = [W.load_block(sh, cfg, f"layers.{i}", i, 0, 2, 0) for i in [int(x) for x in a.layers.split(",")]]
    for L in lays:
        if L.idx_proj is not None:
            L.idx_proj_h = L.idx_proj.to(torch.float16).contiguous()
    ext = _ext()
    g = torch.Generator(device="cuda").manual_seed(5)
    rng = random.Random(3)
    res = {}

    def rnd(*shape, scale=1.0, dtype=BF16):
        return (torch.randn(shape, generator=g, device="cuda") * scale).to(dtype)

    def record(name, ok):
        r = res.setdefault(name, [0, 0])
        r[0] += 1
        r[1] += int(bool(ok))

    D = cfg.dim
    rows_list = list(range(1, 17))
    for trial in range(a.trials):
        scale = [1e-3, 0.05, 1.0, 30.0, 1.0, 5.0][trial % 6]
        for L in lays:
            # -- rowmm: the MoE gate (fp16 weights) and the indexer's weights_proj (fp16) ------------------------
            for name, w in (("rowmm gate", L.gate_w), ("rowmm idx_proj", getattr(L, "idx_proj_h", None))):
                if w is None:
                    continue
                for R in rows_list:
                    x = rnd(R, D, scale=scale)
                    old = K.rowmm(x, w)
                    new = K.rowmm2(x, w)
                    record(f"{name} == old", beq(old, new))
                    # row invariance: each row alone, and the rows permuted
                    perm = torch.randperm(R, generator=torch.Generator().manual_seed(trial * 100 + R))
                    newp = K.rowmm2(x[perm].contiguous(), w)
                    record(f"{name} rows", beq(newp, new[perm]) and
                           all(beq(K.rowmm2(x[i:i + 1], w), new[i:i + 1]) for i in range(0, R, max(1, R // 3))))
            # -- mHC: hc_post + hc_pre (old) against hc_pre2 (fused) ---------------------------------------------
            for which in ("hc_attn", "hc_ffn"):
                fn, sc, bs = getattr(L, which)
                for R in (1, 2, 3, 6, 11, 16):
                    h = rnd(R, 4, D, scale=scale)
                    pre_in = torch.rand(R, 4, generator=g, device="cuda") + 0.1
                    gathered = (torch.randn(2, R, D, generator=g, device="cuda") * scale)
                    # post / comb of a real finish (realistic mixes)
                    x0 = torch.empty(R, D, dtype=BF16, device="cuda")
                    part = torch.empty(R * K.HC_BLOCKS * 32, dtype=F32, device="cuda")
                    po, post, comb = (torch.empty(R, 4, device="cuda"), torch.empty(R, 4, device="cuda"),
                                      torch.empty(R, 4, 4, device="cuda"))
                    K.hc_pre(h, fn, sc, bs, pre_in, L.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, x0, po, post,
                             comb, part)
                    # old: hc_post in place, then hc_pre
                    ho = h.clone()
                    p1, c1 = post.clone(), comb.clone()
                    K.hc_post(gathered, ho, p1, c1, ho)
                    xo = torch.empty(R, D, dtype=BF16, device="cuda")
                    preo = torch.empty(R, 4, device="cuda")
                    K.hc_pre(ho, fn, sc, bs, pre_in, L.ffn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, xo, preo, p1, c1,
                             part)
                    # new, fused, without and with the rotation outputs
                    for use_rot in (False,):
                        hn = torch.empty_like(h)
                        p2, c2 = post.clone(), comb.clone()
                        xn = torch.empty(R, D, dtype=BF16, device="cuda")
                        pren = torch.empty(R, 4, device="cuda")
                        part2 = torch.empty_like(part)
                        rot, ref = None, None
                        if use_rot:
                            suhs = [L.wq_a.suh, L.wkv.suh] + ([L.comp_wkv.suh] if L.comp_wkv is not None else [])
                            rot = [(s_, torch.empty(R, D, dtype=torch.float16, device="cuda")) for s_ in suhs]
                            ref = [torch.empty(R, D, dtype=torch.float16, device="cuda") for _ in suhs]
                            ext.rot_many([xo] * len(suhs), suhs, ref, False)
                        K.hc_pre2(h, fn, sc, bs, pre_in, L.ffn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, xn, pren, p2,
                                  c2, part2, gathered=gathered, h_out=hn)
                        ok = (beq(hn, ho) and beq(xn, xo) and beq(pren, preo) and
                              beq(p2, p1) and beq(c2, c1))
                        if use_rot:
                            ok = ok and all(beq(t[1], r_) for t, r_ in zip(rot, ref))
                        record(f"hc fused{'+rot' if use_rot else ''} == old", ok)
                        # row invariance: row i alone
                        i = rng.randrange(R)
                        hn1 = torch.empty_like(h[i:i + 1])
                        xn1 = torch.empty(1, D, dtype=BF16, device="cuda")
                        pre1 = torch.empty(1, 4, device="cuda")
                        p3, c3 = post[i:i + 1].clone(), comb[i:i + 1].clone()
                        part3 = torch.empty(K.HC_BLOCKS * 32, dtype=F32, device="cuda")
                        K.hc_pre2(h[i:i + 1].contiguous(), fn, sc, bs, pre_in[i:i + 1].contiguous(), L.ffn_norm,
                                  cfg.eps, cfg.hc_eps, cfg.hc_iters, xn1, pre1, p3, c3, part3,
                                  gathered=gathered[:, i:i + 1].contiguous(), h_out=hn1)
                        record("hc fused rows", beq(hn1, hn[i:i + 1]) and beq(xn1, xn[i:i + 1]) and
                               beq(pre1, pren[i:i + 1]) and beq(c3, c2[i:i + 1]))
                    # not posted (the first sublayer, after Engram)
                    xa = torch.empty(R, D, dtype=BF16, device="cuda")
                    xb = torch.empty(R, D, dtype=BF16, device="cuda")
                    pa, pb = torch.empty(R, 4, device="cuda"), torch.empty(R, 4, device="cuda")
                    qa_, qb_ = torch.empty(R, 4, device="cuda"), torch.empty(R, 4, device="cuda")
                    ca, cb_ = torch.empty(R, 4, 4, device="cuda"), torch.empty(R, 4, 4, device="cuda")
                    K.hc_pre(h, fn, sc, bs, pre_in, L.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, xa, pa, qa_, ca,
                             part)
                    K.hc_pre2(h, fn, sc, bs, pre_in, L.attn_norm, cfg.eps, cfg.hc_eps, cfg.hc_iters, xb, pb, qb_, cb_,
                              torch.empty_like(part))
                    record("hc plain == old", beq(xa, xb) and beq(pa, pb) and beq(qa_, qb_)
                           and beq(ca, cb_))
            # -- q RMSNorm + the rotations of wq_b (and the indexer's wq_b) ---------------------------------------
            for R in rows_list[:: 3] + [16]:
                qa = rnd(R, cfg.q_rank, scale=scale)
                old = K.rmsnorm(qa, L.q_norm, cfg.eps)
                tg = [L.wq_b] + ([L.idx_wq_b] if L.idx_wq_b is not None else [])
                ref = [torch.empty(R, cfg.q_rank, dtype=torch.float16, device="cuda") for _ in tg]
                ext.rot_many([old] * len(tg), [t.suh for t in tg], ref, False)
                rot = [(t.suh, torch.empty(R, cfg.q_rank, dtype=torch.float16, device="cuda")) for t in tg]
                new = K.rmsnorm_rot(qa, L.q_norm, cfg.eps, rot)
                record("rmsnorm+rot == old", beq(old, new) and all(beq(r_[1], f) for r_, f in
                                                                           zip(rot, ref)))
            # -- wo_a's group with wo_b's rotation folded into its epilogue ---------------------------------------
            ga = Exl3Group(L.wo_a)
            gb = Exl3Group([L.wo_b])
            for R in (1, 2, 5, 6, 16):
                o = rnd(R, len(L.wo_a) * L.wo_a[0].k, scale=scale)
                og = o.view(R, len(L.wo_a), -1)
                u = torch.empty(R, sum(w.n for w in L.wo_a), dtype=BF16, device="cuda")
                outs, c = [], 0
                for wo in L.wo_a:
                    outs.append(u[:, c:c + wo.n])
                    c += wo.n
                ga([og[:, j] for j in range(len(L.wo_a))], outs)
                ref = torch.empty(R, L.wo_b.k, dtype=torch.float16, device="cuda")
                ext.rot_many([u], [L.wo_b.suh], [ref], False)
                yref = gb([u], out_dtypes=[F32])[0]
                u2 = torch.empty_like(u)
                outs2, c = [], 0
                for wo in L.wo_a:
                    outs2.append(u2[:, c:c + wo.n])
                    c += wo.n
                xh = ga.buffers(R, "cuda")
                ext.rot_many([og[:, j] for j in range(len(L.wo_a))], ga.suh, xh, True)
                xb2 = gb.buffers(R, "cuda")
                offs = []
                c = 0
                for wo in L.wo_a:
                    offs.append((L.wo_b.suh, xb2[0], c))
                    c += wo.n
                ga.rotated(xh, outs2, rot=offs)
                y2 = gb.rotated(xb2, out_dtypes=[F32])[0]
                record("wo_a->wo_b fold == old", beq(u2, u) and beq(xb2[0], ref) and
                       beq(y2, yref))
    # -- the indexer's glue: fp4_qd of q, the weights, scores -> candidates -> keys -> top-k -> sorted, masked -----
    from tensorfold.families.deepseek_v41.cuda.model import apply_candidates
    from tensorfold.families.deepseek_v41.ops import fp4_qd

    for trial in range(a.trials * 4):
        scale = [1e-3, 0.05, 1.0, 30.0, 1e-30, 4.0][trial % 6]
        for R in (1, 2, 3, 6, 16):
            iq = rnd(R, cfg.idx_heads, cfg.idx_dim, scale=scale)
            if trial % 5 == 4:          # exact ties and zeros: multiples of a block's scale / 4
                iq = (torch.randint(-24, 25, iq.shape, generator=g, device="cuda").float() * 0.125).to(BF16)
            record("fp4_qd_p2 == fp4_qd", beq(K.fp4_qd_p2(iq), fp4_qd(iq, 32, e4m3_scale=False)))
    Lx = next(L for L in lays if getattr(L, "idx_proj_h", None) is not None)
    sc_w = cfg.idx_dim ** -0.5 * cfg.idx_heads ** -0.5
    for trial in range(a.trials):
        for R in range(1, 17):
            x = rnd(R, D, scale=[0.05, 1.0, 7.0][trial % 3])
            record("rowmm_wts == old", beq(K.rowmm_wts(x, Lx.idx_proj_h, sc_w),
                                           K.rowmm(x, Lx.idx_proj_h).to(BF16) * sc_w))
    for trial in range(a.trials * 2):
        for R in (1, 2, 5, 6, 16):
            for nb, cap in ((512, 2048), (1024, 4096), (4096, 8192)):
                codes = torch.randint(0, 256, (cap, cfg.idx_dim // 2), generator=g, device="cuda", dtype=torch.int32
                                      ).to(torch.uint8)
                scl = torch.randint(118, 134, (cap, cfg.idx_dim // 32), generator=g, device="cuda", dtype=torch.int32
                                    ).to(torch.uint8)
                iq = fp4_qd(rnd(R, cfg.idx_heads, cfg.idx_dim), 32, e4m3_scale=False)
                wts = K.rowmm(rnd(R, D), Lx.idx_proj_h).to(BF16) * sc_w
                vis = torch.randint(1, nb + 1, (R,), generator=g, device="cuda", dtype=torch.int64)
                base = torch.randint(0, cap - nb, (R,), generator=g, device="cuda", dtype=torch.int64)
                nblk = -(-nb // cfg.cand_block)
                cand = torch.rand(R, nblk, generator=g, device="cuda") < 0.6
                for use_cand in (False, True):
                    for use_base in (False, True):
                        bs = base if use_base else None
                        score = K.index_score(iq, (codes, scl), wts, vis, nb, base=bs)
                        if use_cand:
                            apply_candidates(score, cand, cfg.cand_block)
                        kk = min(cfg.idx_topk, nb)
                        top = K.topk_indices(score, kk)
                        ref = torch.where(top < vis[:, None], top, -1).contiguous()
                        keys = K.index_keys(iq, (codes, scl), wts, vis, nb, base=bs, cand=cand if use_cand else None,
                                            cand_block=cfg.cand_block)
                        new = K.topk_select(keys, kk, vis)
                        record("index keys+topk == old", beq(new, ref))
                        # row invariance: each row alone
                        i = rng.randrange(R)
                        k1 = K.index_keys(iq[i:i + 1].contiguous(), (codes, scl), wts[i:i + 1].contiguous(),
                                          vis[i:i + 1].contiguous(), nb, base=None if bs is None else bs[i:i + 1],
                                          cand=cand[i:i + 1].contiguous() if use_cand else None,
                                          cand_block=cfg.cand_block)
                        record("index keys rows", beq(K.topk_select(k1, kk, vis[i:i + 1].contiguous()), new[i:i + 1]))
    # -- the compressor's FP4 cache writes ------------------------------------------------------------------------
    from tensorfold.families.deepseek_v41.cuda.model import store_rows

    for trial in range(a.trials * 4):
        scale = [1e-3, 0.05, 1.0, 30.0, 1e-30, 300.0, 2000.0, 1e-6][trial % 8]
        for R in (1, 2, 5, 16):
            for d, block, e4 in ((cfg.head_dim, 16, True), (cfg.idx_dim, 32, False)):
                cap = 64
                x = rnd(R, d, scale=scale)
                if trial % 5 == 3:
                    x = (torch.randint(-48, 49, x.shape, generator=g, device="cuda").float() * 0.0625).to(BF16)
                rows = torch.randperm(cap, generator=torch.Generator().manual_seed(trial * 31 + R))[:R].cuda()
                ca = (torch.zeros(cap, d // 2, dtype=torch.uint8, device="cuda"),
                      torch.zeros(cap, d // block, dtype=torch.uint8, device="cuda"))
                cb = (ca[0].clone(), ca[1].clone())
                store_rows(ca, rows, x, block, e4)
                K.fp4_store(x, cb, rows, block, e4)
                record(f"fp4_store({'e4m3' if e4 else 'pow2'}) == store_rows", beq(ca[0], cb[0]) and beq(ca[1], cb[1]))
    # -- route (1 warp a row) and the cand-source layer's fast path ----------------------------------------------
    from tensorfold.families.deepseek_v41.cuda import rounds as RD
    from tensorfold.families.deepseek_v41.cuda.model import _candidates

    for trial in range(a.trials * 2):
        for L in lays:
            for R in (1, 2, 6, 16):
                lg = K.rowmm(rnd(R, D, scale=[0.3, 1.0, 3.0][trial % 3]), L.gate_w)
                outs = []
                for sw in (False, True):
                    K.set_switch("rowmm", sw)
                    pk = torch.empty(R, cfg.topk + 1, dtype=torch.int32, device="cuda")
                    wt = torch.empty(R, cfg.topk + 1, device="cuda")
                    K.route(lg, L.gate_b, cfg.topk, cfg.route_scale, L.experts.count - 1, pk, wt)
                    outs.append((pk, wt))
                K.set_switch("rowmm", True)
                record("route 1-warp == old", beq(outs[0][0], outs[1][0]) and beq(outs[0][1], outs[1][1]))
    for trial in range(a.trials * 2):
        for R in (1, 3, 16):
            for nb in (1024, 4096, 32768):
                sc_ = torch.randn(R, nb, generator=g, device="cuda")
                vis = torch.randint(1, nb + 1, (R,), generator=g, device="cuda", dtype=torch.int64)
                sc_ = torch.where(torch.arange(nb, device="cuda")[None] < vis[:, None], sc_, float("-inf"))
                if trial % 2:
                    sc_[:, ::7] = float("-inf")
                    sc_[:, 3::11] = 0.0
                    sc_[:, 5::13] = -0.0
                record("cand fast == _candidates", beq(RD._candidates_fast(sc_, vis, cfg.cand_blocks, cfg.cand_block),
                                                       _candidates(sc_, vis[:, None], cfg.cand_blocks, cfg.cand_block)))
                kk = min(cfg.idx_topk, nb)
                top = K.topk_indices(sc_, kk)
                ref = torch.where(top < vis[:, None], top, -1)
                record("score_keys+topk == old", beq(K.topk_select(K.score_keys(sc_), kk, vis), ref))
    # -- attention: merge + inverse RoPE + wo_a's rotation in one launch ------------------------------------------
    from tensorfold.families.deepseek_v41.cuda.model import wo_a_rot
    from tensorfold.families.deepseek_v41.ops import fp4_pack, freqs_cis

    hd, rd, Hl = cfg.head_dim, cfg.rope_dim, cfg.n_heads // 2
    f = freqs_cis(rd, 4096, cfg.orig_len, cfg.compress_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    cos, sin = f.real.contiguous().float().cuda(), f.imag.contiguous().float().cuda()
    La = lays[0]
    for impat in (0,):
        ok = True
        for trial in range(a.trials * 4):
            for R in (1, 2, 6, 16):
                RS = 144
                ring = rnd(4 * RS, hd)
                ncomp = 1024
                cp = fp4_pack(rnd(ncomp, hd), 16, True)
                q = rnd(R, Hl, hd, scale=[0.05, 0.3, 1.0][trial % 3])
                pos = torch.randint(200, 1000, (R,), generator=g, device="cuda", dtype=torch.int64)
                slot = torch.randint(0, 4, (R,), generator=g, device="cuda", dtype=torch.int64)
                wbase = slot * RS
                idx = torch.sort(torch.randint(0, 512, (R, 512), generator=g, device="cuda"), dim=-1).values
                idx = torch.where(idx < (pos[:, None] + 1) // 2, idx, -1)
                cbase = torch.zeros(R, dtype=torch.int64, device="cuda")
                o1 = K.sparse_attn(q, La.sink, ring, torch.zeros(1, dtype=torch.int64, device="cuda"), True, cp, idx,
                                   pos, hd ** -0.5, cfg.window, wbase=wbase, cbase=cbase, ring_rows=RS)
                K.rope_heads(o1, cos, sin, pos, rd, inverse=True)
                xref = Exl3Group(La.wo_a).rotate([o1.view(R, len(La.wo_a), -1)[:, j] for j in range(len(La.wo_a))])
                suh, xh, gh = wo_a_rot(La, R, "cuda", hd)
                o2 = K.sparse_attn(q, La.sink, ring, torch.zeros(1, dtype=torch.int64, device="cuda"), True, cp, idx,
                                   pos, hd ** -0.5, cfg.window, wbase=wbase, cbase=cbase, ring_rows=RS,
                                   rot=(cos, sin, rd, suh, xh[0], gh))
                good = beq(o1, o2) and all(beq(u, v) for u, v in zip(xref, xh))
                ok &= good
                record("attn merge+rope+rot == old", good)
    # -- q: RMSNorm + rotation, wq_b with q's RoPE in its epilogue -------------------------------------------------
    from tensorfold.families.deepseek_v41.cuda.model import q_proj

    for trial in range(a.trials * 2):
        for L in lays:
            for R in (1, 2, 5, 16):
                qa = rnd(R, cfg.q_rank, scale=[0.3, 1.0, 3.0][trial % 3])
                pos = torch.randint(0, 4096, (R,), generator=g, device="cuda", dtype=torch.int64)
                qr0 = K.rmsnorm(qa, L.q_norm, cfg.eps)
                q0 = L.wq_b.grouped(qr0, out_dtype=BF16).view(R, -1, hd)
                K.rope_heads(q0, cos, sin, pos, rd)
                idx = L.idx_wq_b is not None
                iq0 = L.idx_wq_b.grouped(qr0, out_dtype=BF16) if idx else None
                qr1, q1, iq1 = q_proj(L, qa, cfg.eps, idx=idx, rope=(cos, sin, pos, hd, rd))
                record("q_proj(+rope) == rmsnorm + mm + rope", beq(qr0, qr1) and beq(q0.view(R, -1), q1) and
                       (not idx or beq(iq0, iq1)))
    # -- q RMSNorm (+ rotations) and the window KV's norm + RoPE + ring write in one launch ------------------------
    for trial in range(a.trials * 3):
        for L in lays:
            for R in (1, 2, 5, 16):
                qa = rnd(R, cfg.q_rank, scale=[0.3, 1.0, 3.0][trial % 3])
                ykv = rnd(R, hd, scale=[0.3, 1.0, 30.0][trial % 3])
                pos = torch.randint(0, 4096, (R,), generator=g, device="cuda", dtype=torch.int64)
                ring0 = torch.zeros(4 * 144, hd, dtype=BF16, device="cuda")
                ring1 = ring0.clone()
                slots = torch.randperm(4 * 144, generator=torch.Generator().manual_seed(trial + R))[:R].cuda()
                tg = [L.wq_b] + ([L.idx_wq_b] if L.idx_wq_b is not None else [])
                rot0 = [(t.suh, torch.empty(R, cfg.q_rank, dtype=torch.float16, device="cuda")) for t in tg]
                rot1 = [(t.suh, torch.empty(R, cfg.q_rank, dtype=torch.float16, device="cuda")) for t in tg]
                for quant in (True, False):
                    o0 = K.rmsnorm_rot(qa, L.q_norm, cfg.eps, rot0)
                    k0 = K.kv_norm_rope(ykv, L.kv_norm, cos, sin, pos, ring0, slots, cfg.eps, quant, rd)
                    o1 = K.q_kv_norm(qa, L.q_norm, cfg.eps, rot1, ykv, L.kv_norm, cos, sin, pos, ring1, slots, quant,
                                     rd)
                    record("q_kv_norm == rmsnorm_rot + kv_norm_rope", beq(o0, o1) and beq(ring0, ring1) and
                           all(beq(u[1], v[1]) for u, v in zip(rot0, rot1)))
    out = {k: {"checks": v[0], "equal": v[1], "all": v[0] == v[1]} for k, v in res.items()}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
