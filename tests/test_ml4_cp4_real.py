"""CP4 at real dims on a CPU host (DESIGN-CP4 §4 M5, §11 gate 2). Too heavy for the quick suite: a large CPU host, ~10 minutes.

    python tests/test_ml4_cp4_real.py m5            # real-size layers, seeded weights: the merge again, at real shapes
    python tests/test_ml4_cp4_real.py fp8 <dir>     # fp8 rows against fp16 rows on the pack's own kv_a / kv_b tensors

<dir> holds L02/L03/L35/L39/L71-nq.safetensors copied from the pack (rope writer, NoPE writer, the readers' source,
two of its readers). q_proj is EXL3 in the pack, so the fp8 run draws queries at chosen sharpness; the node run repeats
it on captured activations.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
HOST_TESTS = Path(os.environ.get("ML4_HOST_TESTS") or sys.exit("set ML4_HOST_TESTS to the reference suite"))
if str(HOST_TESTS) not in sys.path:
    sys.path.insert(0, str(HOST_TESTS))

import torch                                                     # noqa: E402

from test_ml4_cp4 import cos, rms, run_ranks                     # noqa: E402
from tensorfold.families.mistral_large4.config import Cfg        # noqa: E402
from tensorfold.families.mistral_large4.cuda import cp4, fp8kv   # noqa: E402
from tensorfold.families.mistral_large4.cuda import model as M   # noqa: E402
from tensorfold.families.mistral_large4.cuda.weights import Ml4Layer, Ml4Weights  # noqa: E402

W = 4


def real_cfg(n_layers, rope_flags, kv_src, idx_types, topk=2048):
    return Cfg(vocab=512, dim=6144, n_layers=n_layers, n_heads=64, kv_lora=512, nope=64, rd=64, v_head=128, qk_head=128,
               inter=64, dense_inter=64, n_routed=8, topk=2, n_shared=1, route_scale=1.0, eps=1e-6, index_norm_eps=1e-6,
               idx_heads=32, idx_dim=128, idx_topk=topk, softcap=30.0, sm_scale=128 ** -0.5, rope_flags=rope_flags,
               kv_source_layers=kv_src, kv_sources={s for i, s in enumerate(kv_src) if s != i},
               indexer_types=idx_types, mlp_layer_types=["dense"] * n_layers, rope_theta=1e4, rope_type="yarn",
               rope_factor=2048.0, orig_len=512, attention_factor=1.0, tie_word_embeddings=True, hidden_act="silu")


def seeded_weights(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    d = cfg.dim

    def r(*shape, s=1.0):
        return torch.randn(*shape, generator=g) * s

    layers = []
    for L in range(cfg.n_layers):
        rope = cfg.is_rope(L)
        nope_w = cfg.qk_head - (cfg.rd if rope else 0)
        writer = cfg.kv_source_layers[L] == L
        full = cfg.indexer_types[L] == "full"
        lay = Ml4Layer(idx=L, rope=rope, kv_source=cfg.kv_source_layers[L], indexer=cfg.indexer_types[L], dense=True,
                       attn_norm=1 + r(d, s=0.1), ffn_norm=1 + r(d, s=0.1), q=r(cfg.n_heads * cfg.qk_head, d, s=d ** -0.5),
                       o=r(d, cfg.n_heads * cfg.v_head, s=(cfg.n_heads * cfg.v_head) ** -0.5),
                       kv_a=r(cfg.kv_lora + (cfg.rd if rope else 0), d, s=d ** -0.5) if writer else None,
                       kv_a_norm=(1 + r(cfg.kv_lora, s=0.1)) if writer else None,
                       kv_b=r(cfg.n_heads * (nope_w + cfg.v_head), cfg.kv_lora, s=cfg.kv_lora ** -0.5),
                       ssmax_s=r(cfg.n_heads, s=0.05).abs(), ssmax_b=1 + r(cfg.n_heads, s=0.05))
        if full:
            lay.idx_wq_b = r(cfg.idx_heads * cfg.idx_dim, d, s=d ** -0.5)
            lay.idx_wk = r(cfg.idx_dim, d, s=d ** -0.5)
            lay.idx_proj = r(cfg.idx_heads, d, s=d ** -0.5)
            lay.idx_k_norm, lay.idx_k_norm_b = 1 + r(cfg.idx_dim, s=0.1), r(cfg.idx_dim, s=0.05)
            lay.idx_q_norm, lay.idx_q_norm_b = 1 + r(d, s=0.1), r(d, s=0.05)
            lay.idx_ssmax_s, lay.idx_ssmax_b = r(cfg.idx_heads, s=0.05).abs(), 1 + r(cfg.idx_heads, s=0.05)
        lay.dense_gate = r(cfg.dense_inter, d, s=d ** -0.5)
        lay.dense_up = r(cfg.dense_inter, d, s=d ** -0.5)
        lay.dense_down = r(d, cfg.dense_inter, s=cfg.dense_inter ** -0.5)
        layers.append(lay)
    return Ml4Weights(cfg=cfg, rank=0, world=1, embed=r(cfg.vocab, d), norm=1 + r(d, s=0.1), alpha=r(cfg.rd // 2, s=0.01),
                      head=None, head_bf16=r(cfg.vocab, d, s=d ** -0.5), vocab_lo=0, vocab_hi=cfg.vocab, layers=layers)


def reference_stack(w, tokens, chunk=128):
    """Single-rank layers from model.py's own functions, the queries in chunks so attend()'s [S, K, T] mask fits; the
    keys and rows are complete before any query runs, which is what one-shot eager does. Returns streams and selections."""

    cfg, N = w.cfg, tokens.numel()
    model = M.Ml4Model(w)
    x = w.embed[tokens].float()[None]
    pos = torch.arange(N)
    streams, sels, prev = [], {}, None
    planes = {}
    for L, lay in enumerate(w.layers):
        rope_w = cfg.rd if lay.rope else 0
        nope_w = cfg.qk_head - rope_w
        xn = M.rms_norm(x, lay.attn_norm, cfg.eps)
        if lay.kv_source == L:
            ckv = xn @ lay.kv_a.float().t()
            lat = M.rms_norm(ckv[..., :cfg.kv_lora], lay.kv_a_norm, cfg.eps)
            kpe = M.gptj_rotate(ckv[..., cfg.kv_lora:].reshape(-1, rope_w), model.inv_freq, pos).view(1, N, rope_w) \
                if rope_w else None
            planes[L] = (lat, kpe)
        rows_lat, rows_kpe = planes[lay.kv_source]
        q = (xn @ lay.q.float().t()).view(1, N, cfg.n_heads, cfg.qk_head)
        q = M.ssmax_apply(q, lay.ssmax_s, lay.ssmax_b, pos)
        q_nope = q[..., :nope_w]
        q_pe = M.rotate_leading(q[..., nope_w:], model.inv_freq, pos, rope_w) if rope_w else q[..., nope_w:]
        if lay.indexer == "full":
            keys = M.indexer_keys(xn, lay.idx_wk, lay.idx_k_norm, lay.idx_k_norm_b, cfg.index_norm_eps, model.inv_freq,
                                  rope_w, pos)
            prev = []
            for c0 in range(0, N, chunk):
                c1 = min(N, c0 + chunk)
                sc = M.indexer_scores(xn[:, c0:c1], pos[c0:c1], qn_w=lay.idx_q_norm, qn_b=lay.idx_q_norm_b,
                                      wq_b_w=lay.idx_wq_b, wp_w=lay.idx_proj, ss_s=lay.idx_ssmax_s, ss_b=lay.idx_ssmax_b,
                                      eps=cfg.index_norm_eps, n_heads=cfg.idx_heads, head_dim=cfg.idx_dim,
                                      inv_freq=model.inv_freq, rope_w=rope_w, keys=keys[:, :c1])
                sel = M.select_topk(sc + M.causal_mask(sc, offset=c0), cfg.idx_topk)
                prev.append(torch.nn.functional.pad(sel, (0, cfg.idx_topk - sel.shape[-1]), value=-1))
            prev = torch.cat(prev, 1)
            sels[L] = prev
        w_uk, w_uv = model.folds[L]
        out = []
        for c0 in range(0, N, chunk):
            c1 = min(N, c0 + chunk)
            out.append(M.attend(q_nope[:, c0:c1], q_pe[:, c0:c1], rows_lat[:, :c1],
                                None if rows_kpe is None else rows_kpe[:, :c1], w_uk, w_uv, cfg.sm_scale,
                                sel=prev[:, c0:c1], past_len=c0))
        x = x + torch.cat(out, 1).reshape(1, N, -1) @ lay.o.float().t()
        xn2 = M.rms_norm(x, lay.ffn_norm, cfg.eps)
        x = x + (torch.nn.functional.silu(xn2 @ lay.dense_gate.float().t()) * (xn2 @ lay.dense_up.float().t())) \
            @ lay.dense_down.float().t()
        streams.append(x[0].clone())
    return streams, sels


class SpyOps(cp4.Ops):
    """Keeps a round's local scores: the same numbers the top-K ran on."""

    def local_topk(self, qi, wts, keys, vis, n, k, pos_of):
        per_head = torch.relu(torch.einsum("qhd,td->qht", qi.float(), keys[:n].float()))
        self.scores = torch.einsum("qh,qht->qt", wts.float(), per_head)
        return super().local_topk(qi, wts, keys, vis, n, k, pos_of)


class Forced(cp4.Ml4Cp):
    """The real selection runs and is recorded, then the reference's selection drives the attention: one flip of a key
    at the K-th place changes a row by ~3 % of its attention output, and fp32 sums in a different order do flip keys
    whose scores agree to the last bit or two. So the merge is held to exactness on its own inputs (gate a), the flips
    to the rounding of the scores (gate b), and the arithmetic after selection to the §4 numbers (gate c)."""

    ref: dict
    seen: list

    def _select(self, L, lay, qi_raw, wts, rnd):
        real_rows, real_cnt = super()._select(L, lay, qi_raw, wts, rnd)
        st = rnd.seq.stripe
        want = self.ref[L][0][rnd.qpos]
        safe = want.clamp(min=0)
        mine = (want >= 0) & (st.rank(safe) == self.rank)
        rows = torch.where(mine, st.row(safe).to(torch.int32), torch.full_like(safe, cp4.NO_ROW, dtype=torch.int32))
        rows = rows.sort(dim=1).values
        self.seen.append((L, rnd.qpos.clone(), real_rows.clone(), real_cnt.clone(), rows.clone(),
                          mine.sum(1).to(torch.int32), self.ops.scores.clone()))
        return rows, mine.sum(1).to(torch.int32)


def check_selection(st, seen, K):
    """seen[k] = rank k's records. (merged sets vs top-K of the same scores: differences, flips vs the reference, the
    worst flip's score gap relative to the K-th score)."""

    checked = merged_differ = flips = 0
    worst_gap = 0.0
    for i in range(len(seen[0])):
        L, qpos = seen[0][i][0], seen[0][i][1]
        Q = qpos.numel()
        top = int(qpos.max()) + 1
        full = torch.full((Q, top), float("-inf"))
        got = torch.zeros((Q, top), dtype=torch.bool)
        want = torch.zeros((Q, top), dtype=torch.bool)
        for k in range(len(seen)):
            _, _, real_rows, real_cnt, ref_rows, ref_cnt, score = seen[k][i]
            n = score.shape[1]
            p = st.pos(k, torch.arange(n))
            vis = st.vis(k, qpos)
            keep = p[None, :] <= qpos[:, None]
            assert bool((keep == (torch.arange(n)[None, :] < vis[:, None])).all()), "vis disagrees with causality"
            full[:, p] = torch.where(keep, score, torch.full_like(score, float("-inf")))
            q_i, c_i = (torch.arange(K)[None, :] < real_cnt[:, None].long()).nonzero(as_tuple=True)
            got[q_i, st.pos(k, real_rows[q_i, c_i].long())] = True
            q_i, c_i = (torch.arange(K)[None, :] < ref_cnt[:, None].long()).nonzero(as_tuple=True)
            want[q_i, st.pos(k, ref_rows[q_i, c_i].long())] = True
        sel = M.select_topk(full, min(K, top))
        same = torch.zeros((Q, top), dtype=torch.bool)
        ok = (sel >= 0) & (torch.gather(full, 1, sel.clamp(min=0)) > float("-inf"))
        q_i, c_i = ok.nonzero(as_tuple=True)
        same[q_i, sel[q_i, c_i]] = True
        checked += Q
        merged_differ += int((same != got).any(1).sum())
        bad = (want != got).any(1).nonzero().squeeze(1)
        flips += bad.numel()
        for q in bad.tolist():
            a, b = full[q][want[q] & ~got[q]], full[q][got[q] & ~want[q]]
            kth = float(full[q][got[q]].min()) if got[q].any() else 1.0
            gap = float((a.max() - b.min()).abs()) if a.numel() and b.numel() else float("inf")
            worst_gap = max(worst_gap, gap / max(abs(kth), 1e-30))
            print(f"    flip L{L} p{int(qpos[q])}: {a.numel()} key(s), gap/kth {gap / max(abs(kth), 1e-30):.2e}", flush=True)
    return checked, merged_differ, flips, worst_gap


def m5(N=8192, block=512):
    t0 = time.time()
    # rope writer full, NoPE writer full, rope shared, reader of layer 1 (NoPE, shared)
    cfg = real_cfg(4, [1, 0, 1, 0], [0, 1, 2, 1], ["full", "full", "shared", "shared"])
    w = seeded_weights(cfg, 7)
    tokens = torch.randint(0, cfg.vocab, (N,), generator=torch.Generator().manual_seed(8))
    cache = os.environ.get("CP4_M5_REF")
    if cache and os.path.exists(cache):
        ref, ref_sel = torch.load(cache)
    else:
        ref, ref_sel = reference_stack(w, tokens)
        if cache:
            torch.save((ref, ref_sel), cache)
    print(f"reference done {time.time() - t0:.0f} s", flush=True)
    st = cp4.Stripe(W, block)

    def work(rank, comm):
        rows = st.rows_for(N, 0) + block
        fwd = Forced(w, comm, cp4.KvPool(cfg, rows, kind="raw"), SpyOps())
        fwd.ref, fwd.seen = ref_sel, []
        seq = cp4.Seq(st, 0, rows)
        col = []
        for start in range(0, N, W * block):
            m = min(W * block, N - start)
            p = torch.arange(start, start + m)
            fwd.round(seq, start, m, tokens[p[st.rank(p) == rank]], collect=col)
        return col, fwd.seen

    out = run_ranks(work)
    print(f"striped done {time.time() - t0:.0f} s", flush=True)
    worst_c, worst_m = 1.0, 0.0
    for L in range(cfg.n_layers):
        got = torch.zeros_like(ref[L])
        for o in out:
            for (l, pos, x) in o[0]:
                if l == L:
                    got[pos] = x
        c, m = cos(ref[L], got), float((ref[L] - got).abs().max()) / rms(ref[L])
        worst_c, worst_m = min(worst_c, c), max(worst_m, m)
        print(f"  layer {L}: cosine {c:.9f}, max-abs/rms {m:.2e}", flush=True)
    checked, merged, flips, gap = check_selection(st, [o[1] for o in out], cfg.idx_topk)
    print(f"M5: selection {checked} checked: merged != top-K of its own scores {merged}; flips against the reference "
          f"{flips}, worst gap/kth {gap:.2e}; forced selection: worst cosine {worst_c:.9f}, max-abs/rms {worst_m:.2e}; "
          f"{time.time() - t0:.0f} s", flush=True)
    assert merged == 0 and gap <= 1e-5 and worst_c >= 0.99999 and worst_m <= 1e-3


# -- gate 2 on the pack's own tensors --------------------------------------------------------------------------------------

def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        data = f.read()
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "F8_E4M3": torch.float8_e4m3fn}
    out = {}
    for k, e in head.items():
        if k == "__metadata__":
            continue
        lo, hi = e["data_offsets"]
        out[k] = torch.frombuffer(bytearray(data[lo:hi]), dtype=dt[e["dtype"]]).view(e["shape"])
    return out


def pick(t, name):
    (k,) = [k for k in t if k.endswith(name)]
    return t[k]


def attention(q_lat, q_pe, lat, kpe, sel, wuv, sm):
    """Batched sparse MLA: q_lat [n, H, C], sel [n, K] -> o [n, H, v] fp32."""

    rows = lat[sel]                                               # [n, K, C]
    s = torch.einsum("nhc,nkc->nhk", q_lat, rows)
    if kpe is not None:
        s = s + torch.einsum("nhr,nkr->nhk", q_pe, kpe[sel])
    p = torch.softmax(s * sm, -1)
    o = torch.einsum("nhk,nkc->nhc", p, rows)
    return torch.einsum("nhc,hvc->nhv", o, wuv), p


def fp8_real(root, N=8192, Qn=256, K=2048):
    root = Path(root)
    g = torch.Generator().manual_seed(21)
    d, H, C, sm = 6144, 64, 512, 128 ** -0.5
    fmts = list(fp8kv.SCALES)
    report = []
    t35 = read_safetensors(root / "L35-nq.safetensors")
    plan = [("L02 rope writer", "L02", True, None), ("L03 NoPE writer", "L03", False, None),
            ("L35 NoPE source", "L35", False, None), ("L39 reader of L35", "L39", False, t35),
            ("L71 reader of L35", "L71", False, t35)]
    inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))
    # hidden rows: a dictionary of "topics" plus noise, so queries and keys correlate and softmax has something to pick
    topics = torch.randn(96, d, generator=g)
    which = torch.randint(0, 96, (N,), generator=g)
    x = topics[which] + 0.5 * torch.randn(N, d, generator=g)
    pos = torch.arange(N)
    for name, tag, rope, source in plan:
        t = read_safetensors(root / f"{tag}-nq.safetensors")
        own = source if source is not None else t
        xn = M.rms_norm(x, pick(t, "input_layernorm.weight"), 1e-6)
        ckv = xn @ pick(own, "kv_a_proj_with_mqa.weight").float().t()
        lat = M.rms_norm(ckv[:, :C], pick(own, "kv_a_layernorm.weight"), 1e-6)
        kpe = M.gptj_rotate(ckv[:, C:], inv, pos) if rope else None
        nope_w = 64 if rope else 128
        wb = pick(t, "kv_b_proj.weight").float().view(H, nope_w + 128, C)
        wuk, wuv = wb[:, :nope_w], wb[:, nope_w:]
        lat16 = lat.half().float()
        kpe16 = None if kpe is None else kpe.half().float()
        qp = torch.arange(N - Qn, N)                               # the deep queries see the whole plane
        sel = torch.stack([torch.randperm(int(p), generator=g)[:K] for p in qp])
        # a query aimed at a few rows of its own selection: sharp attention is where the rows' error shows
        for sharp in (0.5, 2.0, 8.0):
            tgt = lat16[sel[:, :8]].mean(1)                        # [Qn, C]
            q_lat = tgt[:, None, :] * torch.randn(Qn, H, 1, generator=g).abs() + 0.3 * torch.randn(Qn, H, C, generator=g)
            q_lat = q_lat * (sharp / (rms(tgt) ** 2 * C * sm))
            q_pe = torch.randn(Qn, H, 64, generator=g) if rope else None
            base, pb = attention(q_lat, q_pe, lat16, kpe16, sel, wuv, sm)
            eff = float((1.0 / pb.pow(2).sum(-1)).mean())
            for f in fmts:
                fmt = fp8kv.row_format(f, C, 64 if rope else 0)
                l8, k8 = fp8kv.decode(fp8kv.encode(lat, kpe, fmt), fmt)
                got, _ = attention(q_lat, q_pe, l8, k8, sel, wuv, sm)
                b, o = base.reshape(Qn * H, -1).double(), got.reshape(Qn * H, -1).double()
                row_cos = (b * o).sum(1) / (b.norm(dim=1) * o.norm(dim=1) + 1e-30)
                report.append((name, sharp, f, eff, float((l8 - lat16).norm() / lat16.norm()), cos(base, got),
                               float(row_cos.min()), float((b - o).norm() / b.norm())))
                print(f"{name:20s} sharp {sharp:3.1f} eff rows {eff:8.1f} {f:8s} row err {report[-1][4]:.4f} cos "
                      f"{report[-1][5]:.6f} worst head cos {report[-1][6]:.6f} rel {report[-1][7]:.5f}", flush=True)
    return report


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "m5"
    torch.set_num_threads(int(os.environ.get("CP4_THREADS") or 8))
    if what == "m5":
        m5()
    else:
        fp8_real(sys.argv[2])
