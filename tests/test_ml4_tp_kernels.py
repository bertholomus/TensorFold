"""tp_kernels.py against the torch forms in tp.py and model.py. On a node at real dims; on a CPU host with
TRITON_INTERPRET=1 (the same code path).

    python tests/test_ml4_tp_kernels.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from tensorfold.families.mistral_large4.cuda import cp4, fp8kv   # noqa: E402
from tensorfold.families.mistral_large4.cuda import model as M   # noqa: E402
from tensorfold.families.mistral_large4.cuda import cp4_kernels as CK  # noqa: E402
from tensorfold.families.mistral_large4.cuda import tp_kernels as K    # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1"
DEV = "cpu" if INTERP else "cuda"
F16, F32 = torch.float16, torch.float32
D = 6144


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.pow(2).mean().sqrt().clamp_min(1e-30))


def rnd(*shape, scale=1.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(DEV)


def pos_of(R):
    return (torch.arange(R) * 7919 + 13).to(DEV)


def inv_freq():
    return (1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))).to(DEV)


def test_add_norm(R=3, W=4):
    every, x, w = rnd(W, R, D, seed=1), rnd(R, D, seed=2), rnd(D, seed=3)
    ref = x.clone()
    s = every[0]
    for k in range(1, W):
        s = s + every[k]
    ref = ref + s
    n = M.rms_norm(ref, w, 1e-6)
    got, n32, n16 = K.add_norm(every, x.clone(), w, 1e-6)
    assert torch.equal(got, ref), "the residual"
    assert rel(n32, n) < 1e-5 and rel(n16.float(), n) < 1e-2
    _, n32, _ = K.add_norm(None, ref, w, 1e-6)
    assert rel(n32, n) < 1e-5


def test_layer_norm(R=3):
    x, w, b = rnd(R, D, seed=4), rnd(D, seed=5), rnd(D, seed=6)
    assert rel(K.layer_norm16(x, w, b, 1e-6).float(), M.layer_norm_bias(x, w, b, 1e-6)) < 2e-3


def test_q(R=2, H=4, rd=64):
    for rope in (rd, 0):
        nope = 128 - rope
        q = rnd(R, H * 128, seed=7).to(F16)
        s, b = rnd(H, scale=0.1, seed=8), rnd(H, scale=0.1, seed=9) + 1
        wuk = rnd(H, nope, 512, scale=0.05, seed=10).to(F16)
        pos = pos_of(R)
        ref = M.ssmax_apply(q.float().view(R, H, 128), s, b, pos)
        lat = torch.einsum("qhn,hnc->qhc", ref[..., :nope].to(F16).float(), wuk.float()).to(F16)
        out = torch.empty((R, H * 512 + H * rope), dtype=F16, device=DEV)
        K.q_absorb(q, pos, s, b, wuk, 128, out)
        assert rel(out[:, :H * 512].float().view(R, H, 512), lat.float()) < 5e-3
        if rope:
            pe = M.rotate_leading(ref[..., nope:], inv_freq(), pos, rope).to(F16)
            K.q_rope(q, pos, s, b, inv_freq(), H, 128, rope, out[:, H * 512:])
            assert rel(out[:, H * 512:].float().view(R, H, rope), pe.float()) < 5e-3


def test_qi(R=2, H=4):
    for rope in (64, 0):
        q = rnd(R, H * 128, seed=11).to(F16)
        s, b = rnd(H, scale=0.1, seed=12), rnd(H, scale=0.1, seed=13) + 1
        pos = pos_of(R)
        ref = M.rotate_leading(M.ssmax_apply(q.float().view(R, H, 128), s, b, pos), inv_freq(), pos, rope).to(F16)
        out = torch.empty((R, H * 128 + 8), dtype=F16, device=DEV)
        K.qi_prep(q, pos, s, b, inv_freq(), H, 128, rope, out)
        assert rel(out[:, :H * 128].float().view(R, H, 128), ref.float()) < 5e-3


def test_kv_row(R=3):
    for rope in (64, 0):
        fmt = fp8kv.row_format("f32x128", 512, rope)
        ckv = rnd(R, 512 + rope, seed=14)
        w = rnd(512, seed=15) + 1
        pos = pos_of(R)
        lat = M.rms_norm(ckv[:, :512], w, 1e-6)
        kpe = M.gptj_rotate(ckv[:, 512:].float(), inv_freq(), pos) if rope else None
        ref = fp8kv.encode(lat, kpe, fmt)
        plane = torch.zeros((R + 2, fmt.row_bytes), dtype=torch.uint8, device=DEV)
        wr = torch.arange(R, device=DEV) + 1
        K.kv_row(ckv, w, pos, wr, inv_freq(), plane, 1e-6, 512, rope)
        a, b = fp8kv.decode(plane[1:R + 1], fmt)
        ra, rb = fp8kv.decode(ref, fmt)
        same = float((plane[1:R + 1, :512] == ref[:, :512]).float().mean())
        # the interpreter's e4m3 cast drops the carry into the exponent: only a GPU is held to the codes
        assert (same > 0.95) if INTERP else (same > 0.998 and rel(a, ra) < 0.3), f"latent {same}"
        assert torch.equal(plane[1:R + 1, 512:528], ref[:, 512:528]) or not INTERP, "scales"
        if rope:
            assert rel(b, rb) < 5e-3


def test_key_row(R=3):
    for rope in (64, 0):
        k = rnd(R, 128, seed=16)
        w, b = rnd(128, seed=17) + 1, rnd(128, seed=18)
        pos = pos_of(R)
        ref = M.layer_norm_bias(k, w, b, 1e-6)
        ref = M.rotate_leading(ref.view(R, 1, 1, -1), inv_freq(), pos, rope).view(R, -1).to(F16)
        plane = torch.zeros((R + 2, 128), dtype=F16, device=DEV)
        K.key_row(k, w, b, pos, torch.arange(R, device=DEV) + 1, inv_freq(), plane, 1e-6, 128, rope)
        assert rel(plane[1:R + 1].float(), ref.float()) < 5e-3


def test_merge_expand(R=2, W=4, H=4):
    o = rnd(W, R, H, 512, seed=19).to(F16)
    lse = rnd(W, R, H, seed=20)
    lse[1, 0, 1] = float("-inf")
    wuv = rnd(H, 128, 512, scale=0.05, seed=21).to(F16)
    ref = cp4.merge_partials(o, lse)
    ref = torch.einsum("qhc,hvc->qhv", ref.to(F16).float(), wuv.float())
    stride = R * H * 512 + 2 * R * H
    buf = torch.zeros((W, stride), dtype=F16, device=DEV)
    for k in range(W):
        buf[k, :R * H * 512] = o[k].reshape(-1)
        buf[k, R * H * 512:].view(F32)[:] = lse[k].reshape(-1)
    got = K.merge_expand(buf, wuv, R, torch.empty((R, H * 128), dtype=F16, device=DEV))
    assert rel(got.float().view(R, H, 128), ref) < 5e-3


def test_route(R=3, E=128):
    lg = rnd(R, E, seed=22)
    lut = torch.full((E,), 33, dtype=torch.int32, device=DEV)
    lut[32:64] = torch.arange(32, dtype=torch.int32, device=DEV)
    w, idx = M.route_softmax(lg, 4)
    pick, wts = K.route(lg, lut, 4, 33)
    assert torch.equal(pick, lut[idx])
    assert rel(wts, torch.where(pick < 33, w, torch.zeros_like(w))) < 1e-5


def test_act(n=5000):
    g, u = rnd(2, n, seed=23), rnd(2, n, seed=24)
    assert rel(K.silu_mul(g, u).float(), F.silu(g) * u) < 1e-2


def test_score(R=2, W=4, IL=8, n=1024):
    q = rnd(R, W * IL, 128, seed=25).to(F16)
    keys = rnd(n + 200, 128, seed=26).to(F16)
    wts = rnd(R, W * IL, scale=0.2, seed=27)
    base = torch.tensor([0, 100], device=DEV)[:R]
    vis = torch.tensor([n, n // 2 + 3], dtype=torch.int32, device=DEV)[:R]
    T = 40 + IL * 128
    every = torch.zeros((W, R, T), dtype=F16, device=DEV)
    for k in range(W):
        every[k, :, 40:] = q[:, k * IL:(k + 1) * IL].reshape(R, -1)
    out = torch.empty((R, n), dtype=F32, device=DEV)
    K.score_step(every, 40, wts, 0.5, IL, 128, keys, base, vis, n, out)
    for r in range(R):
        ref = torch.empty((1, n), dtype=F32, device=DEV)
        CK.index_scores(q[r:r + 1], wts[r:r + 1] * 0.5, keys[int(base[r]):], vis[r:r + 1], n, ref)
        ok = out[r] > float("-inf")
        assert int(ok.sum()) == int(vis[r]) and rel(out[r][ok], ref[0][ok]) < 1e-3


def test_attend(R=2, W=4, H=16, cap=300, Kn=64):
    for rd in (64, 0):
        fmt = fp8kv.row_format("f32x128", 512, rd)
        plane = fp8kv.encode(rnd(cap, 512, seed=28), rnd(cap, rd, seed=29) if rd else None, fmt)
        q = rnd(R, W * H, 512, scale=0.05, seed=30).to(F16)
        pe = rnd(R, W * H, rd, scale=0.05, seed=31).to(F16) if rd else None
        rows = torch.sort(torch.randint(0, cap - 10, (R, Kn), generator=torch.Generator().manual_seed(32))).values
        rows = rows.to(torch.int32).to(DEV)
        cnt = torch.tensor([Kn, 17], dtype=torch.int32, device=DEV)[:R]
        base = torch.tensor([0, 7], device=DEV)[:R]
        o = torch.empty((R, W * H, 512), dtype=F16, device=DEV)
        lse = torch.empty((R, W * H), dtype=F32, device=DEV)
        CK.sparse_mla(q, pe, rows + base[:, None].to(torch.int32), cnt, plane, 0.1, o, lse)
        T = H * (512 + rd)
        every = torch.zeros((W, R, T), dtype=F16, device=DEV)
        for k in range(W):
            every[k, :, :H * 512] = q[:, k * H:(k + 1) * H].reshape(R, -1)
            if rd:
                every[k, :, H * 512:] = pe[:, k * H:(k + 1) * H].reshape(R, -1)
        stride = R * H * 512 + 2 * R * H
        blocks = torch.zeros((W, stride), dtype=F16, device=DEV)
        K.attend_step(every, H, rd, rows, cnt, base, plane, 0.1, blocks)
        for k in range(W):
            got = blocks[k, :R * H * 512].view(R, H, 512)
            err = rel(got.float(), o[:, k * H:(k + 1) * H].float())         # p rounds to fp16 against another running max
            assert err < 3e-3, f"o off by {err}"
            assert rel(blocks[k, R * H * 512:].view(F32).view(R, H), lse[:, k * H:(k + 1) * H]) < 1e-3


def test_select():
    if INTERP:
        return
    W, k = 4, 2048
    st = cp4.Stripe(W, 64)
    for R, n, vis, step in ((1, 2048, [2048], 0.0), (2, 2048, [1500, 700], 0.0), (1, 8192, [8192], 0.25),
                            (8, 4096, [4096, 3000, 2049, 2048, 2047, 1, 0, 4000], 0.0),
                            (4, 65536, [65536, 40000, 2500, 30000], 1.0), (1, 262144, [262144], 0.0)):
        score = rnd(W, R, n, seed=R + n)
        if step:
            score = (score / step).round() * step                # many equal scores
        v = torch.tensor(vis, dtype=torch.int32, device=DEV)
        t = torch.arange(n, device=DEV)
        score = torch.where(t[None, None] < v[None, :, None], score, torch.full_like(score, float("-inf")))
        keys = []
        for r in range(W):
            key = cp4.score_keys(score[r], st.pos(r, t)[None].expand(R, n))
            keys.append(key.masked_fill(score[r] == float("-inf"), cp4.PAD_KEY))
        top = [torch.topk(key, k, dim=1).values for key in keys]
        thresh = cp4.kth_key(torch.cat(top, dim=1), k)
        got_keys, got_rows = zip(*(K.local_top(score[r].contiguous(), v, k, st, r) for r in range(W)))
        every = torch.stack(got_keys)
        for r in range(W):
            assert torch.equal(got_keys[r].sort(dim=1, descending=True).values, top[r]), f"local keys {R} {n} rank {r}"
            assert (got_rows[r][:, 1:] > got_rows[r][:, :-1]).logical_or(got_rows[r][:, 1:] == cp4.NO_ROW).all(), "rows out of order"
            rows, cnt = K.pick_rows(every, got_rows[r], r)
            ref_rows, ref_cnt = cp4.filter_selection(got_keys[r], got_rows[r], thresh)
            assert torch.equal(cnt, ref_cnt), f"count {R} {n} rank {r}: {cnt} {ref_cnt}"
            assert torch.equal(rows, ref_rows), f"rows {R} {n} rank {r}"


def test_select_round():
    if INTERP:
        return
    W, k = 4, 2048
    g = torch.Generator().manual_seed(9)
    for R in (1, 5, 300):
        key = (torch.randint(-2 ** 40, 2 ** 40, (W, R, k), generator=g) << 20).to(DEV)
        key[:, R // 2, 1000:] = cp4.PAD_KEY                     # a row with few real keys
        key[0, 0, :5] = key[1, 0, :5]                           # equal keys can not happen, but the digits must still end
        rows = torch.sort(torch.randint(0, 2 ** 20, (R, k), generator=g, dtype=torch.int32), dim=1).values.to(DEV)
        cand = key.permute(1, 0, 2).reshape(R, -1)
        thresh = K.kth_rows(key)
        real = cand != cp4.PAD_KEY
        assert torch.equal((cand >= thresh[:, None]) & real, (cand >= cp4.kth_key(cand, k)[:, None]) & real), f"kth {R}"
        got, cnt = K.keep_rows(key[1].contiguous(), rows, thresh, 7)
        ref, ref_cnt = cp4.filter_selection(key[1], rows, thresh)
        assert torch.equal(cnt, ref_cnt), f"count {R}"
        assert torch.equal(got, torch.where(ref == cp4.NO_ROW, ref, ref + 7)), f"rows {R}"


def test_score8(R=70, n=33000, k=2048):
    """score8's sums against e4m3 values summed in fp32; top8 against local_top, fast, through the list's end and on short rows."""

    if INTERP:
        return
    from types import SimpleNamespace

    st = SimpleNamespace(world=4, block=64)
    q, keys = rnd(R, 32, 128, seed=1).to(F16), rnd(n, 128, seed=2).to(F16)
    keys[n // 2:] = keys[n // 2:n // 2 + 1] * rnd(n - n // 2, 1, seed=3).to(F16).abs()          # rows alike: long lists
    w = rnd(R, 32, seed=4) * 0.05
    vis = torch.randint(n // 2, n + 1, (R,), device=DEV, dtype=torch.int32)
    vis[:3] = torch.tensor([17, k, 3000], device=DEV, dtype=torch.int32)
    vis[3:R // 2] = n                                                                            # long enough to bound
    wide = (n + 7) // 8 * 8
    out = torch.empty((R, wide), device=DEV)
    top = torch.empty((R, wide // 8), device=DEV)
    q8, qw = CK.score8_query(q, w)
    CK.score8(q8, qw, 0, keys, vis, n, out, top)

    def e4m3(t):
        amax = t.float().abs().amax(-1, keepdim=True)
        return (t.float() * (torch.full_like(amax, 448.0) / amax)).to(torch.float8_e4m3fn).float(), amax / 448.0   # 448.0 / amax rounds twice

    qe, qs = e4m3(q)
    ke, ks = e4m3(keys)
    ref = torch.relu(torch.einsum("rhd,td->rht", qe[:8], ke)) * (w[:8] * qs[:8, :, 0])[:, :, None]
    ref = ref.sum(1) * ks[:, 0][None, :]
    t = torch.arange(n, device=DEV)[None, :]
    live = t < vis[:8, None]
    r = rel(out[:8, :n][live], ref[live])
    assert r < 1e-5, f"score8 {r}"
    assert torch.equal(out[:8, :n][~live], torch.full_like(out[:8, :n][~live], float("-inf"))), "score8 past vis"
    groups = out.view(R, wide // 8, 8).amax(2)
    keep = torch.arange(wide // 8, device=DEV)[None, :] < ((vis.long() + 7) // 8)[:, None]
    assert torch.equal(groups[keep], top[keep]), "score8 largest"
    want_k, want_r = K.local_top(out[:, :n].contiguous(), vis, k, st, 1)
    got_k = torch.empty_like(want_k)
    got_r = torch.empty_like(want_r)
    slow = torch.zeros(1, dtype=torch.int32, device=DEV)
    K.top8(out, top, vis, k, st, 1, got_k, got_r, slow)
    assert torch.equal(got_k, want_k) and torch.equal(got_r, want_r), "top8"
    assert 0 < slow.item() < R, f"top8 went through the rows {slow.item()} times of {R}"


def test_score4(R=70, n=33000):
    """score4's sums against the values its own bytes hold summed in fp32; what a row keeps; the largest of every eight."""

    if INTERP:
        return
    q, keys = rnd(R, 32, 128, seed=1).to(F16), rnd(n, 128, seed=2).to(F16)
    w = rnd(R, 32, seed=4) * 0.05
    vis = torch.randint(n // 2, n + 1, (R,), device=DEV, dtype=torch.int32)
    vis[:3] = torch.tensor([17, 2048, 3000], device=DEV, dtype=torch.int32)
    vis[3:R // 2] = n
    wide = (n + 7) // 8 * 8
    out = torch.empty((R, wide), device=DEV)
    top = torch.empty((R, wide // 8), device=DEV)
    q4, k4 = CK.score4_rows(q, w), CK.score4_rows(keys)
    CK.score4(q4, 0, k4, vis, n, out, top)

    def values(rows):
        grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=DEV)
        grid = torch.cat([grid, -grid])
        codes = rows[..., :64]
        v = torch.stack([grid[(codes & 15).long()], grid[(codes >> 4).long()]], -1).flatten(-2)
        scale = rows[..., 64:72].contiguous().view(torch.float8_e4m3fn).float()
        return (v.view(*v.shape[:-1], 8, 16) * scale[..., None]).flatten(-2), rows[..., 72:76].contiguous().view(F32)[..., 0]

    qv, qf = values(q4)
    kv, kf = values(k4)
    for got, want in ((kv * kf[:, None], keys.float()), (qv * (qf / w)[..., None], q.float())):
        assert (got - want).norm() < 0.12 * want.norm(), "score4 rows"
    ref = (torch.relu(torch.einsum("rhd,td->rht", qv[:8], kv)) * qf[:8, :, None]).sum(1) * kf[None, :]
    t = torch.arange(n, device=DEV)[None, :]
    live = t < vis[:8, None]
    r = rel(out[:8, :n][live], ref[live])
    assert r < 1e-5, f"score4 {r}"
    assert torch.equal(out[:8, :n][~live], torch.full_like(out[:8, :n][~live], float("-inf"))), "score4 past vis"
    groups = out.view(R, wide // 8, 8).amax(2)
    keep = torch.arange(wide // 8, device=DEV)[None, :] < ((vis.long() + 7) // 8)[:, None]
    assert torch.equal(groups[keep], top[keep]), "score4 largest"


def test_q_scale():
    H, m, qk = 64, 40, 128
    pos = torch.arange(1000, 1000 + m, device=DEV)
    s, b = rnd(H, scale=0.1), rnd(H, seed=2)
    q = rnd(m, H * qk, seed=3).to(F16)
    for nope in (64, 128):
        out = torch.empty((H, m, nope), dtype=F16, device=DEV)
        K.q_scale(q, pos, s, b, qk, out)
        ref = M.ssmax_apply(q.float().view(m, H, qk), s, b, pos)[..., :nope].to(F16).transpose(0, 1)
        assert rel(out, ref) < 2e-3, f"q_scale {nope}"


def test_merge_rows():
    W, n, H, V = 4, 12, 64, 128
    o = rnd(W, n, H, V, seed=4).to(F16)
    lse = rnd(W, n, H, seed=5)
    lse[2, 3] = float("-inf")
    got = K.merge_rows(o, lse)
    ref = cp4.merge_partials(o.float(), lse).reshape(n, H * V)
    assert rel(got, ref) < 2e-3, "merge_rows"


def main() -> int:
    bad = 0
    for name, t in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                t()
                print(f"ok   {name}", flush=True)
            except BaseException as exc:                         # noqa: BLE001
                import traceback

                bad += 1
                print(f"FAIL {name}: {exc}", flush=True)
                traceback.print_exc()
    return bad


if __name__ == "__main__":
    sys.exit(main())
