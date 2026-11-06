"""CPU tests for the ML4 eager model forward (D1 items 1-10) against the reference suite.

Run directly, no pytest, no numpy (the host-test venv is Python 3.9 + torch CPU only):

    $ML4_HOST_TESTS/.venv/bin/python tests/test_ml4_model.py

The refs (``$ML4_HOST_TESTS/refs``, not part of this repository) are imported READ-ONLY via sys.path, never edited,
never copied. Gates: dense-regime agreement at the
published 0.999999-class cosine, greedy-token agreement cached vs one-shot, and per-mechanism
invariants (reader on the source latent, frozen shared selection, NoPE non-rotation, ssmax
identity, softcap bound, router renorm).

Known ref divergence: ``refs/tiny_model.py`` rotates the indexer on NoPE
layers (it passes the rope table at every layer), while the engine does NOT (mistral_large4.py:210
sets qk_rope_head_dim=0 on NoPE layers, mla_attn._indexer_rope_ early-returns on 0 width,
PACK-LAYOUT: "neither attention nor indexer rotates", loader-test/tiny_diag.py::own_scores uses
ix.use_rope). This model implements the ENGINE semantics. Cross-implementation gates therefore run
in two regimes that provably do not touch the divergence: (A) the default tiny config with the
visible context inside the index budget (no selection computed anywhere), and (B) an all-rotating
config variant that keeps the full sparse machinery live. The default config past the budget is
still run end to end: the layers before the divergence are compared to the ref, the divergence is
pinpointed exactly to layer 3's indexer rotation choice, and the cross cosines are reported.
"""
from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
HOST_TESTS = Path(os.environ.get("ML4_HOST_TESTS") or sys.exit("set ML4_HOST_TESTS to the reference suite"))
if not HOST_TESTS.is_dir():
    raise SystemExit(f"reference suite not found at {HOST_TESTS} (set ML4_HOST_TESTS)")
if str(HOST_TESTS) not in sys.path:
    sys.path.insert(0, str(HOST_TESTS))

import torch                                                     # noqa: E402
from refs import indexer as IX                                   # noqa: E402
from refs import mla as MLA                                      # noqa: E402
from refs import moe as MOE                                      # noqa: E402
from refs import rope_lerope as RL                               # noqa: E402
from refs import softcap as SC                                   # noqa: E402
from refs import ssmax as SS                                     # noqa: E402
from refs import tiny_model as TM                                # noqa: E402

from tensorfold.families.mistral_large4.config import Cfg, kv_sharing_schedule   # noqa: E402
from tensorfold.families.mistral_large4.cuda import model as M                   # noqa: E402
from tensorfold.families.mistral_large4.cuda.weights import Ml4Layer, Ml4Weights # noqa: E402


def cos(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float((a @ b / (a.norm() * b.norm() + 1e-30)).item())


# -- fixtures: the reference TinyConfig W dict -> our Cfg / Ml4Weights (same seed, same tensors) ---------------------------

def tiny_cfg(tc: TM.TinyConfig) -> Cfg:
    rope_flags = [0 if nr else 1 for nr in tc.no_rope]
    kv = list(tc.kv_source)
    return Cfg(vocab=tc.vocab, dim=tc.hidden, n_layers=tc.n_layers, n_heads=tc.heads, kv_lora=tc.kv_lora,
               nope=tc.nope, rd=tc.rope, v_head=tc.v, qk_head=tc.nope + tc.rope, inter=tc.moe_inter,
               dense_inter=tc.dense_inter, n_routed=tc.n_experts, topk=tc.top_k_experts, n_shared=tc.shared_experts,
               route_scale=1.0, eps=tc.rms_eps, index_norm_eps=tc.idx_eps, idx_heads=tc.index_heads,
               idx_dim=tc.index_head_dim, idx_topk=tc.index_topk, softcap=tc.softcap,
               sm_scale=(tc.nope + tc.rope) ** -0.5, rope_flags=rope_flags, kv_source_layers=kv,
               kv_sources=kv_sharing_schedule(kv, rope_flags, tc.n_layers), indexer_types=list(tc.indexer_types),
               mlp_layer_types=list(tc.mlp_types), rope_theta=tc.rope_theta, rope_type="yarn",
               rope_factor=tc.yarn_factor, orig_len=tc.yarn_original_max, attention_factor=1.0,
               tie_word_embeddings=True, hidden_act="silu", q_rank=None)


def tiny_weights(tc: TM.TinyConfig, W: dict) -> Ml4Weights:
    layers = []
    for L in range(tc.n_layers):
        p = f"L{L}."
        dense = tc.mlp_types[L] == "dense"
        experts = None
        if not dense:
            experts = [tuple(W[p + f"mlp.e{e}.{t}"] for t in ("gate", "up", "down")) for e in range(tc.n_experts)]
            experts.append((W[p + "mlp.shared.gate"], W[p + "mlp.shared.up"], W[p + "mlp.shared.down"]))
        layers.append(Ml4Layer(
            idx=L, rope=not tc.no_rope[L], kv_source=tc.kv_source[L], indexer=tc.indexer_types[L], dense=dense,
            attn_norm=W[p + "in_ln"], ffn_norm=W[p + "post_ln"], q=W[p + "attn.q_proj"], o=W[p + "attn.o_proj"],
            kv_a=W.get(p + "attn.kv_a"), kv_a_norm=W.get(p + "attn.kv_a_norm"), kv_b=W[p + "attn.kv_b"],
            ssmax_s=W[p + "attn.ss_s"], ssmax_b=W[p + "attn.ss_b"],
            idx_wq_b=W.get(p + "ix.wq_b"), idx_wk=W.get(p + "ix.wk"), idx_proj=W.get(p + "ix.wp"),
            idx_k_norm=W.get(p + "ix.kn_w"), idx_k_norm_b=W.get(p + "ix.kn_b"),
            idx_q_norm=W.get(p + "ix.qn_w"), idx_q_norm_b=W.get(p + "ix.qn_b"),
            idx_ssmax_s=W.get(p + "ix.ss_s"), idx_ssmax_b=W.get(p + "ix.ss_b"),
            gate_w=W.get(p + "mlp.gate_router"),
            dense_gate=W.get(p + "mlp.gate"), dense_up=W.get(p + "mlp.up"), dense_down=W.get(p + "mlp.down"),
            experts=experts, expert_ids=tuple(range(tc.n_experts))))
    # refs tie the head to the embedding; the real pack is untied — the head slot is separate either way
    return Ml4Weights(cfg=tiny_cfg(tc), rank=0, world=1, embed=W["emb"], norm=W["norm"], alpha=W["alpha"], head=None,
                      head_bf16=W["emb"], vocab_lo=0, vocab_hi=tc.vocab, layers=layers)


@dataclass
class AllRope(TM.TinyConfig):
    """Schedule-variant fixture: every layer rotates. The full sparse machinery stays live (selection,
    shared reuse, reader) but the ref's indexer-rotation divergence (NoPE layers) cannot trigger, so
    this is the cross-implementation gate in the SPARSE regime. A fixture in our test, not a change
    to the refs."""

    @property
    def no_rope(self):
        return [False] * self.n_layers


def gate_pair(tc, seed=1105):
    """(ref model, our model) over the SAME tiny weights."""
    W = TM.random_weights(tc, seed)
    return TM.TinyML4(tc, W=W), M.Ml4Model(tiny_weights(tc, W)), W


def greedy_pair(ref, mine, prompt, n_new):
    """Both models greedy-decode n_new tokens through their own caches following the REF's token
    stream; the ref also re-prefills the whole sequence each step (cache-less one-shot). Returns
    per-step logits cosines (ours vs ref-cached, ours vs ref-one-shot) and greedy agreement."""
    rc, mc = ref.reset_cache(), M.new_cache(mine.cfg)
    rlg, mlg = ref.forward(prompt, rc), mine.forward(prompt, mc)
    seq = prompt.clone()
    onelog = ref.forward(seq, ref.reset_cache())
    cos_ref, cos_os, agree = [], [], True
    agree = agree and bool(rlg[-1].argmax().item() == mlg[-1].argmax().item())
    for _ in range(n_new):
        nxt = rlg[-1].argmax().view(1, 1).clone()
        seq = torch.cat([seq, nxt], 1)
        rlg = ref.forward(nxt, rc)
        mlg = mine.forward(nxt, mc)
        onelog = ref.forward(seq, ref.reset_cache())
        cos_ref.append(cos(mlg[-1], rlg[-1]))
        cos_os.append(cos(mlg[-1], onelog[-1]))
        agree = agree and bool(mlg[-1].argmax().item() == rlg[-1].argmax().item()
                               == onelog[-1].argmax().item())
    return cos_ref, cos_os, agree


# -- gates: our model vs the reference ---------------------------------------------------------------------------

def test_e2e_dense_regime_default_config():
    """Gate A: the default tiny config (8 layers, real ratios, NoPE 3/7, reader 7<-3, shared
    indexers, dense L0) with the visible context inside the index budget (past+S <= 8) — neither
    implementation computes a selection, so this is the pure dense e2e. Prefill 5 + 3 decodes:
    per-step logits cos >= 0.999999 vs the ref AND vs the ref's one-shot re-prefill, greedy tokens
    agree, per-layer residuals (collect) agree at the same class, logits bounded by softcap."""
    tc = TM.TinyConfig()
    ref, mine, _ = gate_pair(tc)
    torch.manual_seed(1105)
    prompt = torch.randint(0, tc.vocab, (1, 5))
    cos_ref, cos_os, agree = greedy_pair(ref, mine, prompt, 3)
    assert min(cos_ref) >= 0.999999 and min(cos_os) >= 0.999999, f"cos {cos_ref} {cos_os}"
    assert agree, "greedy tokens diverged"
    col_r, col_m = [], []
    ref.forward(prompt, ref.reset_cache(), collect=col_r)
    mine.forward(prompt, M.new_cache(mine.cfg), collect=col_m)
    cl = [cos(a, b) for a, b in zip(col_r, col_m)]
    assert min(cl) >= 0.999999, f"per-layer residual cos {cl}"
    lg_m = mine.forward(prompt, M.new_cache(mine.cfg))
    assert lg_m.shape == (5, tc.vocab) and float(lg_m.abs().max()) <= 30.0 + 1e-4
    return {"steps": len(cos_ref), "logits_cos_min": min(cos_ref), "oneshot_cos_min": min(cos_os),
            "layer_residual_cos_min": min(cl), "greedy_agree": 1}


def test_e2e_sparse_regime_all_rope():
    """Gate B: sparse regime (prefill 12 > index_topk 8, then 4 decodes) with every schedule intact
    (reader, shared reuse, dense L0) but all layers rotating — the divergence-free sparse gate.
    Per-step logits cos >= 0.999999 vs the ref and vs one-shot; cached greedy == one-shot greedy."""
    tc = AllRope()
    ref, mine, _ = gate_pair(tc)
    torch.manual_seed(31)
    prompt = torch.randint(0, tc.vocab, (1, 12))
    cos_ref, cos_os, agree = greedy_pair(ref, mine, prompt, 4)
    assert min(cos_ref) >= 0.999999 and min(cos_os) >= 0.999999, f"cos {cos_ref} {cos_os}"
    assert agree, "greedy tokens diverged in the sparse regime"
    return {"steps": len(cos_ref), "logits_cos_min": min(cos_ref), "oneshot_cos_min": min(cos_os),
            "greedy_agree": 1}


def test_internal_e2e_default_config_sparse():
    """Gate C: the default config past the budget (selection live; the layer-3 NoPE-indexer
    divergence is active downstream). Our OWN cached decode must match our own one-shot re-prefill
    (the ref suite's published internal gate: same greedy tokens, cos >= 0.999 per step), and the
    cross-implementation agreement must hold for every layer BEFORE the divergence point (the
    layers whose selection the two implementations compute identically): layers 0-2."""
    tc = TM.TinyConfig()
    ref, mine, W = gate_pair(tc)
    mine2 = M.Ml4Model(tiny_weights(tc, W))
    torch.manual_seed(7)
    prompt = torch.randint(0, tc.vocab, (1, 12))
    mc = M.new_cache(mine.cfg)
    lg = mine.forward(prompt, mc)
    seq, cos_i, agree = prompt.clone(), [], True
    for _ in range(2):
        nxt = lg[-1].argmax().view(1, 1)
        seq = torch.cat([seq, nxt], 1)
        lg = mine.forward(nxt, mc)
        lg2 = mine2.forward(seq, M.new_cache(mine2.cfg))
        cos_i.append(cos(lg[-1], lg2[-1]))
        agree = agree and bool(lg[-1].argmax().item() == lg2[-1].argmax().item())
    assert min(cos_i) >= 0.999 and agree, f"cached vs one-shot diverged: {cos_i} {agree}"
    col_r, col_m = [], []
    ref.forward(prompt, ref.reset_cache(), collect=col_r)
    mine.forward(prompt, M.new_cache(mine.cfg), collect=col_m)
    cl = [cos(a, b) for a, b in zip(col_r, col_m)]
    assert min(cl[:3]) >= 0.999999, f"layers before the divergence differ: {cl[:3]}"
    return {"internal_cos_min": min(cos_i), "greedy_agree": 1,
            "cross_layer_cos": [round(c, 7) for c in cl]}


def test_nope_indexer_does_not_rotate_engine_semantics():
    """The divergence proof. Recompute the layer-3 (NoPE, full indexer) indexer keys, scores and
    selection from the REF PRIMITIVES at rope_dim=0 (engine semantics) and rope_dim=8 (the tiny
    ref's hardcoded choice): our plane and selection equal the rope_dim=0 computation and differ
    from the rope_dim=8 one — pinning the entire ref gap to that one choice, layer 3."""
    tc = TM.TinyConfig()
    ref, mine, W = gate_pair(tc)
    torch.manual_seed(7)
    prompt = torch.randint(0, tc.vocab, (1, 12))
    col, sel_log = [], []
    cache = M.new_cache(mine.cfg)
    mine.forward(prompt, cache, collect=col, sel_out=sel_log)
    sels = dict(sel_log)
    xn3 = MLA.rms_norm(col[2], W["L3.in_ln"], tc.rms_eps)
    pos = torch.arange(12)
    kw = dict(kn_w=W["L3.ix.kn_w"], kn_b=W["L3.ix.kn_b"], eps=tc.idx_eps, inv_freq=ref.inv_freq)
    k0 = IX.indexer_keys(xn3, W["L3.ix.wk"], positions=pos, rope_dim=0, **kw)
    k8 = IX.indexer_keys(xn3, W["L3.ix.wk"], positions=pos, rope_dim=tc.rope, **kw)
    assert torch.allclose(cache["ixk"][3], k0, atol=1e-6), "NoPE indexer plane is not the unrotated ref keys"
    assert not torch.allclose(cache["ixk"][3], k8, atol=1e-4), "NoPE plane unexpectedly equals the rotated keys"
    sc0 = IX.indexer_scores(xn3, pos, qn_w=W["L3.ix.qn_w"], qn_b=W["L3.ix.qn_b"], wq_b_w=W["L3.ix.wq_b"],
                            wk_w=W["L3.ix.wk"], kn_w=W["L3.ix.kn_w"], kn_b=W["L3.ix.kn_b"], wp_w=W["L3.ix.wp"],
                            ss_s=W["L3.ix.ss_s"], ss_b=W["L3.ix.ss_b"], eps=tc.idx_eps, n_heads=tc.index_heads,
                            head_dim=tc.index_head_dim, inv_freq=ref.inv_freq, rope_dim=0, keys=k0)
    sel0 = IX.select_topk(sc0 + IX.causal_mask(sc0, offset=0), tc.index_topk)
    assert torch.equal(sels[3], sel0), "layer-3 selection differs from the engine-semantics ref computation"
    assert torch.allclose(ref.inv_freq, mine.inv_freq, atol=1e-7), "rope tables differ"
    return {"nope_plane_equals_rope0": 1, "nope_plane_differs_from_rope8": 1, "selection_matches_engine": 1}


def test_shared_indexer_selection_frozen():
    """Shared layers reuse the nearest preceding full layer's selection UNCHANGED (int32-identical),
    per forward, in the sparse regime; under the budget nothing selects."""
    tc = TM.TinyConfig()
    _, mine, _ = gate_pair(tc)
    torch.manual_seed(7)
    prompt = torch.randint(0, tc.vocab, (1, 12))
    sel_log = []
    mine.forward(prompt, M.new_cache(mine.cfg), sel_out=sel_log)
    sels = dict(sel_log)
    assert sels[1] is not None and sels[4] is not None and sels[6] is not None
    assert torch.equal(sels[2], sels[1]) and torch.equal(sels[5], sels[4]) and torch.equal(sels[7], sels[6])
    sel_log2 = []
    mine.forward(torch.tensor([[1]]), M.new_cache(mine.cfg), sel_out=sel_log2)   # fresh cache: past+S=1 <= 8
    assert all(s is None for _, s in sel_log2), "budget not binding must select nothing"
    return {"shared_pairs": "2<-1,5<-4,7<-6", "frozen_bit_equal": 1, "in_budget_dense": 1}


def test_reader_appends_nothing_and_uses_source_latent():
    """KV reader (D1 §3): layer 7 reads layer 3's latent rows and publishes NOTHING (no lat/kpe/ixk
    plane of its own); with its MoE zeroed (refs' own ``override`` hook) its residual delta equals
    the ref reader's attention delta at the 0.999999 class; and its OWN kv_b is live (changing it
    changes the output). Run on AllRope so every upstream layer matches the ref exactly."""
    tc = AllRope()
    ref, mine, W = gate_pair(tc)
    torch.manual_seed(31)
    prompt = torch.randint(0, tc.vocab, (1, 12))
    cache = M.new_cache(mine.cfg)
    mine.forward(prompt, cache)
    assert cache["lat"][7] is None and cache["kpe"][7] is None and cache["ixk"][7] is None, \
        "reader/shared layers must publish no planes"
    assert cache["lat"][3].shape[1] == 12 and cache["ixk"][3].shape[1] == 12
    zero = {f"L7.mlp.e{e}.down": torch.zeros_like(W[f"L7.mlp.e{e}.down"]) for e in range(tc.n_experts)}
    zero["L7.mlp.shared.down"] = torch.zeros_like(W["L7.mlp.shared.down"])
    ref0 = TM.TinyML4(tc, W=W, override=zero)
    rcol = []
    ref0.forward(prompt, ref0.reset_cache(), collect=rcol)
    mine0 = M.Ml4Model(tiny_weights(tc, {**W, **zero}))
    mcol = []
    mine0.forward(prompt, M.new_cache(mine0.cfg), collect=mcol)
    dcos = cos(mcol[7] - mcol[6], rcol[7] - rcol[6])
    assert dcos >= 0.999999, f"reader attention delta differs from the ref: {dcos}"
    mine1 = M.Ml4Model(tiny_weights(tc, {**W, "L7.attn.kv_b": W["L7.attn.kv_b"] * 2.0}))
    mcol1 = []
    mine1.forward(prompt, M.new_cache(mine1.cfg), collect=mcol1)
    assert cos(mcol1[7] - mcol1[6], mcol[7] - mcol[6]) < 0.999999, "reader ignored its own kv_b"
    return {"reader_publishes_nothing": 1, "reader_delta_cos_vs_ref": round(dcos, 8), "own_kv_b_live": 1}


# -- invariants: our primitives against the refs (read-only calls) ---------------------------------------------------

def test_primitives_match_refs():
    """rms_norm, biased LayerNorm, ssmax (incl. identity == plain softmax), softcap (bound + cap 0),
    the YaRN+LeRoPE table, partial rotate (r=0 identity, zero table identity), the softmax router
    (== route_fp32, renorm to 1) and stable select_topk (ties to lower index)."""
    torch.manual_seed(1105)
    x = torch.randn(2, 7, 128)
    pos = torch.arange(7)
    w = 1.0 + 0.02 * torch.randn(128)
    assert torch.allclose(M.rms_norm(x, w, 1e-6), MLA.rms_norm(x, w, 1e-6), atol=1e-7, rtol=0)
    b = 0.02 * torch.randn(128)
    assert torch.equal(M.layer_norm_bias(x, w, b, 1e-6), IX.layer_norm_bias(x, w, b, 1e-6))
    s, sb = 0.1 * torch.randn(4), 1.0 + 0.1 * torch.randn(4)
    q4 = torch.randn(1, 7, 4, 24)
    assert torch.equal(M.ssmax_apply(q4, s, sb, pos), SS.ssmax_apply(q4, s, sb, pos))
    zs, zo = SS.identity_params(4)
    assert torch.equal(M.ssmax_apply(q4, zs, zo, pos), q4), "ssmax identity must be plain softmax"
    big = torch.randn(1000) * 500.0
    assert torch.equal(M.softcap(big, 30.0), SC.softcap(big, 30.0))
    assert float(M.softcap(big, 30.0).abs().max()) <= 30.0 and torch.equal(M.softcap(big, 0.0), big)
    alpha = 0.15 * torch.randn(4)
    inv = M.lerope_inv_freq(tiny_cfg(TM.TinyConfig()), alpha)
    assert torch.allclose(inv, RL.lerope_inv_freq(8, 10000.0, alpha, factor=2048.0, original_max=512), atol=0, rtol=0)
    x4 = torch.randn(1, 7, 3, 24)
    assert torch.equal(M.rotate_leading(x4, inv, pos, 8), RL.partial_rotate(x4, inv, pos, 8))
    assert torch.equal(M.rotate_leading(x4, inv, pos, 0), x4), "r=0 rotation must be identity"
    zinv = torch.zeros(4)
    tok = torch.randn(7, 8)
    assert torch.equal(M.gptj_rotate(tok, zinv, pos), tok), "zero table == identity rotation"
    gate = 0.1 * torch.randn(8, 128)
    xr = x.reshape(-1, 128)
    w1, i1 = M.route_softmax(xr @ gate.t(), 2)
    w2, i2 = MOE.route_fp32(xr, gate, 2)
    assert torch.equal(w1, w2) and torch.equal(i1, i2), "router differs from route_fp32"
    assert torch.allclose(w1.sum(-1), torch.ones(w1.shape[0]), atol=1e-6), "router weights must renorm to 1"
    sc = torch.randn(1, 5, 11)
    sc[:, 1, 3] = sc[:, 1, 4] = 10.0                                  # exact tie at the top: lower index must win
    top = M.select_topk(sc, 4)
    assert torch.equal(top, IX.select_topk(sc, 4)) and top[0, 1].tolist()[:2] == [3, 4]
    return {"norms": 1, "ssmax_identity_is_softmax": 1, "softcap_bound": 1, "rope_table_and_rotate": 1,
            "router_equals_route_fp32": 1, "topk_ties": 1}


def test_indexer_matches_ref_primitive():
    """indexer_keys / indexer_scores / select against the ref primitives on identical weights."""
    tc = TM.TinyConfig()
    torch.manual_seed(1105)
    W = TM.random_weights(tc, 1105)
    x = torch.randn(1, 9, 128) * 0.3
    pos = torch.arange(9)
    p = "L0.ix."
    inv = RL.lerope_inv_freq(tc.rope, tc.rope_theta, W["alpha"], factor=tc.yarn_factor,
                             original_max=tc.yarn_original_max)
    k_mine = M.indexer_keys(x, W[p + "wk"], W[p + "kn_w"], W[p + "kn_b"], tc.idx_eps, inv, 8, pos)
    keys = IX.indexer_keys(x, W[p + "wk"], W[p + "kn_w"], W[p + "kn_b"], tc.idx_eps, inv, 8, pos)
    assert torch.equal(k_mine, keys), "indexer keys differ bit-wise from the ref"
    sc_mine = M.indexer_scores(x, pos, qn_w=W[p + "qn_w"], qn_b=W[p + "qn_b"], wq_b_w=W[p + "wq_b"],
                               wp_w=W[p + "wp"], ss_s=W[p + "ss_s"], ss_b=W[p + "ss_b"], eps=tc.idx_eps,
                               n_heads=tc.index_heads, head_dim=tc.index_head_dim, inv_freq=inv, rope_w=8, keys=keys)
    sc_ref = IX.indexer_scores(x, pos, qn_w=W[p + "qn_w"], qn_b=W[p + "qn_b"], wq_b_w=W[p + "wq_b"],
                               wk_w=W[p + "wk"], kn_w=W[p + "kn_w"], kn_b=W[p + "kn_b"], wp_w=W[p + "wp"],
                               ss_s=W[p + "ss_s"], ss_b=W[p + "ss_b"], eps=tc.idx_eps, n_heads=tc.index_heads,
                               head_dim=tc.index_head_dim, inv_freq=inv, rope_dim=8, keys=keys)
    assert torch.allclose(sc_mine, sc_ref, atol=1e-6), f"score cos {cos(sc_mine, sc_ref)}"
    assert torch.equal(M.select_topk(sc_mine + M.causal_mask(sc_mine, 0), tc.index_topk),
                       IX.select_topk(sc_ref + IX.causal_mask(sc_ref, 0), tc.index_topk))
    return {"keys_bit_equal": 1, "scores_cos": round(cos(sc_mine, sc_ref), 10), "selection_equal": 1}


def test_fold_kv_b_shapes_and_absorb():
    """W_UK/W_UV fold (engine mla_attn.py:452-460): shapes on rope and NoPE layers, and the folded
    k/v equal the ref's per-head einsum (refs/mla.py::_kv_per_head) — absorbed == unabsorbed."""
    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    for L in (0, 3):                                    # L0 rope, L3 NoPE
        nope_w = tc.nope if L == 0 else tc.nope + tc.rope
        uk, uv = M.fold_kv_b(W[f"L{L}.attn.kv_b"], tc.heads, nope_w, tc.v)
        assert uk.shape == (tc.kv_lora, tc.heads * nope_w) and uv.shape == (tc.kv_lora, tc.heads * tc.v)
        lat = torch.randn(1, 6, tc.kv_lora)
        ref = MLA.MLARef(hidden=tc.hidden, heads=tc.heads, kv_lora=tc.kv_lora, nope=nope_w,
                         rope=0 if L else tc.rope, v=tc.v, sm_scale=(tc.nope + tc.rope) ** -0.5,
                         eps=tc.rms_eps, q_proj=W[f"L{L}.attn.q_proj"], kv_b=W[f"L{L}.attn.kv_b"],
                         o_proj=W[f"L{L}.attn.o_proj"], ss_s=W[f"L{L}.attn.ss_s"], ss_b=W[f"L{L}.attn.ss_b"],
                         kv_a=None, kv_a_norm=None)
        k_ref, v_ref = ref._kv_per_head(lat)
        assert torch.allclose((lat @ uk).view(1, 6, tc.heads, nope_w), k_ref, atol=1e-6)
        assert torch.allclose((lat @ uv).view(1, 6, tc.heads, tc.v), v_ref, atol=1e-6)
    return {"rope_and_nope_folds": 1, "fold_equals_ref_einsum": 1}


def test_untied_head_slot():
    """The head is its own slot (the real pack: untied EXL3/bf16 head). head_bf16=emb reproduces the
    tied ref at gate class; swapping the head changes the logits while the embedding stays; the
    softcap bounds the untied head too."""
    tc = TM.TinyConfig()
    ref, mine, W = gate_pair(tc)
    torch.manual_seed(1105)
    prompt = torch.randint(0, tc.vocab, (1, 5))
    lw2 = tiny_weights(tc, W)
    lw2.head_bf16 = 5.0 * torch.randn_like(W["emb"])
    m2 = M.Ml4Model(lw2)
    l1 = mine.forward(prompt, M.new_cache(mine.cfg))
    l2 = m2.forward(prompt, M.new_cache(m2.cfg))
    ref_lg = ref.forward(prompt, ref.reset_cache())
    assert cos(l1, ref_lg) >= 0.999999 and cos(l1, l2) < 0.99
    assert float(l2.abs().max()) <= 30.0 + 1e-4
    return {"tied_head_matches_ref": 1, "head_slot_live": 1}


def test_module_is_pure_torch():
    """The CPU acceptance path: importing model.py pulled in neither triton nor numpy."""
    leaked = sorted(m for m in sys.modules if m.split(".")[0] in ("triton", "numpy"))
    assert not leaked, f"import set: {leaked}"
    return {"no_triton_numpy": 1}


def test_slot_contract_dense_head_exl3_on_cpu():
    """Slot contract (node smoke step 1b, 2026-11-06): a quantized pack hands the dense MLP and the
    head as Exl3Linear — a module with .k/.n/__call__(fp16 rows), NO .float()/.t()/indexing. The
    dense branch once called silu_expert's plain-tensor math on those slots and 500'd the first
    token on the node. This exercises the Exl3Linear branch of _lin — dense layer 0 AND the head —
    on CPU with duck-typed stubs, so any future raw-tensor op on a quant slot fails here, not there."""

    class StubExl3:
        """Duck-types cuda.exl3.linear.Exl3Linear exactly as _lin uses it: .k, .n, __call__(fp16
        rows) -> float()-able rows. Every other attribute raises, like the real class."""

        def __init__(self, w):
            self._wt = w.float().t().contiguous()            # [k, n]: y = x @ Wt
            self.k, self.n, self.calls = w.shape[1], w.shape[0], 0

        def __call__(self, x, out=None, out_dtype=None):
            assert x.dtype is torch.float16 and x.ndim == 2 and x.shape[1] == self.k, \
                f"stub saw {x.dtype} {tuple(x.shape)}, Exl3Linear contract is (rows, k) fp16"
            self.calls += 1
            return (x.float() @ self._wt).half()             # kernel-shaped: fp16 out

        def __getattr__(self, name):
            raise AttributeError(f"'StubExl3' object has no attribute '{name}' (Exl3Linear surface)")

    tc = TM.TinyConfig()
    ref, mine, _ = gate_pair(tc)
    mine2 = M.Ml4Model(tiny_weights(tc, _))
    lay0 = mine2.w.layers[0]
    assert lay0.dense, "fixture: tiny layer 0 is the dense layer"
    dense = [StubExl3(t) for t in (lay0.dense_gate, lay0.dense_up, lay0.dense_down)]
    lay0.dense_gate, lay0.dense_up, lay0.dense_down = dense
    head = StubExl3(mine2.w.head_bf16)
    mine2.w.head, mine2.w.head_bf16 = head, None             # force the head through the module branch
    torch.manual_seed(1105)
    prompt = torch.randint(0, tc.vocab, (1, 5))
    lg = mine.forward(prompt, M.new_cache(mine.cfg))         # plain-tensor regime
    lg2 = mine2.forward(prompt, M.new_cache(mine2.cfg))      # Exl3Linear slots (fp16 in/out)
    assert all(d.calls > 0 for d in dense) and head.calls > 0, \
        f"slots never called as modules: dense={[d.calls for d in dense]} head={head.calls}"
    assert lg2.shape == lg.shape and torch.isfinite(lg2).all()
    rel = float((lg2 - lg).abs().max() / lg.abs().max().clamp_min(1e-9))
    assert rel < 0.05, f"logits drifted {rel:.4f} in the Exl3Linear slot regime"
    try:
        lay0.dense_gate.float
        raise AssertionError("stub exposed .float(), Exl3Linear must not")
    except AttributeError:
        pass
    return {"dense_slot_calls": [d.calls for d in dense], "head_slot_calls": head.calls,
            "logits_rel_drift": round(rel, 5)}


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        try:
            r = t()
            print(f"PASS {t.__name__}: {r}")
        except Exception:
            fails += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed")
    sys.exit(1 if fails else 0)
