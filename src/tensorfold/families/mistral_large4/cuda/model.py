"""Mistral-Large-4 text model forward (D1 items 1-10) — eager torch, CPU-fp32 and world=1 CUDA.

This is the phase-2a model: the full prefill/decode forward (classical absorbed MLA, K != V,
per-layer NoPE schedule, ssmax on main + indexer queries, KV-sharing readers, per-full-layer DSA
indexer with shared-indexer reuse, LeRoPE/YaRN table, fp32 softmax router top-4 + shared, dense
layer 0, logit softcap 30, untied EXL3 head), consuming the phase-1 ``Ml4Weights`` slot structure
(``cuda/weights.py``). Two hosts share one forward core:

* **CPU fp32** — the reference acceptance path: every tensor is a plain torch tensor and every function
  here is pure eager torch, so ``tests/test_ml4_model.py`` runs it against the reference suite
  (``$ML4_HOST_TESTS/refs``, not part of this repository) without CUDA, Triton or numpy;
* **world=1 CUDA** — the debug/smoke structure (all 128 experts + shared local): linear slots are
  ``Exl3Linear`` (chunked to its 128-row decode kernel; prefill grouping is phase 2b engine work)
  and ``experts`` is ``exl3_experts.Exl3RoutedExperts`` driven exactly like dsv41's ``Model.moe``
  (shared expert as the last pick slot, weight 1.0).

Kernel note (D1): the Triton adaptation points (sparse_attn V-span HD−RD, rope leading-mode,
biased LayerNorm, softmax route) are phase-2b work — they land in this family's ``kernels.py``
next to the CUDA-graph decode body (D1 §Extension points 2). The eager functions below are their
reference semantics; dsv41's own kernel files stay untouched either way.

Reference semantics proven by the reference suite (refs/mla.py, refs/indexer.py, refs/rope_lerope.py, refs/moe.py,
refs/ssmax.py, refs/softcap.py; engine cites D1 §1-§10). One deliberate difference to
``refs/tiny_model.py``: the indexer on a NoPE layer does NOT rotate — the engine sets
qk_rope_head_dim=0 there (mistral_large4.py:210, mla_attn.py:565-572 "neither attention nor
indexer rotates", PACK-LAYOUT.md), while the tiny ref passes the rope table at every layer.

Scheduler/engine integration (MultiDecoder, EP4 all-to-all, DP4, CUDA graphs) is phase 2b; B == 1.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from ..config import Cfg
from .weights import Ml4Layer, Ml4Weights

F32 = torch.float32


# -- rope: YaRN table with LeRoPE alpha folded (D1 §7; refs/rope_lerope.py) ---------------------------------------

def yarn_inv_freq(dim: int, base: float, device=None, *, factor: float, original_max: int,
                  beta_fast: float = 32.0, beta_slow: float = 1.0, truncate: bool = True) -> torch.Tensor:
    """YaRN inverse frequencies (exllamav3 util/rope.py::yarn_inv_freq, HF ramp). dim rotated dims
    -> dim/2 freqs. attention_factor 1.0 leaves cos/sin unscaled, so it never appears here."""

    pos_freqs = base ** (torch.arange(0, dim, 2, device=device).float() / dim)
    inv_ex = 1.0 / pos_freqs

    def find_correction_dim(num_rotations):
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = find_correction_dim(beta_fast), find_correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim // 2, dtype=F32, device=device) - low) / (high - low)
    efac = 1 - torch.clamp(linear_func, 0, 1).float()
    return (1.0 / (factor * pos_freqs)) * (1 - efac) + inv_ex * efac


def lerope_inv_freq(cfg: Cfg, alpha: torch.Tensor) -> torch.Tensor:
    """The one table every rotating layer and every full indexer shares: YaRN ramp × exp(alpha)
    (mla_attn.py:423-427 fold; alpha is global in globals.safetensors)."""

    inv = yarn_inv_freq(cfg.rd, cfg.rope_theta, alpha.device, factor=cfg.rope_factor,
                        original_max=cfg.orig_len)
    assert alpha.shape == inv.shape, f"rotary_emb.alpha {tuple(alpha.shape)} != rope table {tuple(inv.shape)}"
    return inv.float() * alpha.float().exp()


def gptj_rotate(x: torch.Tensor, inv_freq: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Interleaved (GPT-J) rotation of the last dim of x by angle pos * inv_freq[i] on pair i.
    Bit-identical no-op when inv_freq is all zero (cos=1, sin=0)."""

    r = x.shape[-1]
    nf = inv_freq[: r // 2]
    pos = positions.to(F32).reshape(-1).to(x.device)
    ang = pos[:, None] * nf[None, :]
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


def rotate_leading(x4: torch.Tensor, inv_freq: torch.Tensor, positions: torch.Tensor, r: int) -> torch.Tensor:
    """Rotate only the LEADING r dims of the last axis ((B, S, H, D) rows; the indexer's
    _indexer_rope_ form, mla_attn.py:565-589). r == 0 (every NoPE layer, D1 §2) is a no-op."""

    if r == 0:
        return x4
    rows = x4[..., :r].reshape(-1, r)
    pos = positions.reshape(-1)
    reps = rows.shape[0] // pos.numel()
    head = gptj_rotate(rows, inv_freq, pos.repeat_interleave(reps))
    out = x4.clone()
    out[..., :r] = head.view(x4.shape[:-1] + (r,))
    return out


# -- norms, ssmax, softcap (D1 §4/§8; refs) ----------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(F32)
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def layer_norm_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """Biased LayerNorm (torch.nn.LayerNorm semantics, fp32) — the indexer's q_input_norm and
    k_norm with index_norm_eps 1e-6 (D1 §5; dsv41 has weight-only RMS, so this is new math)."""

    return F.layer_norm(x.to(F32), (x.shape[-1],), weight.float(), bias.float(), eps)


def ssmax_scale(s: torch.Tensor, b: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Per-head scale s_h*ln(pos+1)+b_h (pos 0 -> b_h); positions (S,) or (B, S)."""

    return torch.log1p(positions.to(F32)).unsqueeze(-1) * s.float() + b.float()


def ssmax_apply(q: torch.Tensor, s: torch.Tensor, b: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Scale every (token, head) row of q (..., H, D) fp32. Applied to the WHOLE query before the
    rope split; a per-head scalar commutes with rotation, so the order never matters (refs/ssmax)."""

    sc = ssmax_scale(s, b, positions)
    if q.dim() == 4 and sc.dim() == 2 and q.shape[0] != sc.shape[0]:
        sc = sc.unsqueeze(0)
    return q.to(F32) * sc.reshape(*sc.shape[:-1], sc.shape[-1], 1)


def softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """cap*tanh(l/cap), cap 0 == off (mistral_large4 final_logit_softcapping=30, D1 §8)."""

    if not cap:
        return logits
    return torch.tanh(logits / cap) * cap


# -- DSA indexer (D1 §5/§6; refs/indexer.py) ----------------------------------------------------------------------

def indexer_keys(x: torch.Tensor, wk_w: torch.Tensor, kn_w: torch.Tensor, kn_b: torch.Tensor,
                 eps: float, inv_freq: torch.Tensor, rope_w: int, positions: torch.Tensor) -> torch.Tensor:
    """(B, S, D_i) keys: biased LayerNorm(k) then rotate the leading rope_w dims (0 on NoPE)."""

    k = layer_norm_bias(x @ wk_w.float().t(), kn_w, kn_b, eps)
    B, S, D = k.shape
    return rotate_leading(k.view(B, S, 1, D), inv_freq, positions, rope_w).view(B, S, D)


def indexer_scores(x: torch.Tensor, positions: torch.Tensor, *, qn_w: torch.Tensor, qn_b: torch.Tensor,
                   wq_b_w: torch.Tensor, wp_w: torch.Tensor, ss_s: torch.Tensor, ss_b: torch.Tensor,
                   eps: float, n_heads: int, head_dim: int, inv_freq: torch.Tensor, rope_w: int,
                   keys: torch.Tensor) -> torch.Tensor:
    """fp32 (B, S, T): relu(q·k·D^-0.5) head-weighted by weights_proj(x)·H^-0.5, summed over the
    index heads. Queries come from the biased LayerNorm of the BLOCK INPUT (no q-LoRA latent)."""

    q = layer_norm_bias(x, qn_w, qn_b, eps)
    q = (q @ wq_b_w.float().t()).view(x.shape[0], x.shape[1], n_heads, head_dim)
    q = ssmax_apply(q, ss_s, ss_b, positions)
    q = rotate_leading(q, inv_freq, positions, rope_w)
    per_head = F.relu(torch.einsum("bshd,btd->bsht", q.float(), keys.float()) * (head_dim ** -0.5))
    w = (x @ wp_w.float().t()).float() * (n_heads ** -0.5)
    return torch.einsum("bsh,bsht->bst", w, per_head)


def causal_mask(scores: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Additive causal mask over (B, S, T): row i sees keys 0..offset+i."""

    S, T = scores.shape[-2:]
    allowed = torch.arange(T, device=scores.device)[None, :] <= \
        (torch.arange(S, device=scores.device) + offset)[:, None]
    return torch.where(allowed, 0.0, float("-inf")).expand_as(scores)


def select_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Per-row top-K: score descending, ties to the lower index (stable sort, refs select), int32,
    -1 where fewer than K keys are visible — the padded tail; the rows whose whole visible set
    fits select exactly it (dense-equivalent, the reference's sparse-in-budget property)."""

    sc = scores.masked_fill(~torch.isfinite(scores), float("-inf"))
    order = torch.sort(sc, dim=-1, descending=True, stable=True).indices[..., :k]
    valid = torch.isfinite(scores).gather(-1, order.clamp_min(0).long())
    return torch.where(valid, order, order.new_full((), -1)).to(torch.int32)


# -- attention: classical absorbed MLA (D1 §1; refs/mla.py) -------------------------------------------------------

def fold_kv_b(kv_b: torch.Tensor, heads: int, nope_w: int, v_head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """kv_b [H*(nope_w+v), kv_lora] -> W_UK (kv_lora, H*nope_w), W_UV (kv_lora, H*v).

    The engine folds kv_b once at load into w_uk/w_uv (mla_attn.py:452-460) and scores
    q_nope·(latent@W_UK) + q_pe·k_pe, output o_lat@W_UV → o_proj — K and V are different
    projections of one 512-wide latent row. NoPE layers pass nope_w = qk_head (the whole key
    head is nope, their kv_b carries the extra rows)."""

    kv_lora = kv_b.shape[1]
    assert kv_b.shape[0] == heads * (nope_w + v_head), f"kv_b rows {kv_b.shape[0]} != {heads}({nope_w}+{v_head})"
    w = kv_b.float().view(heads, nope_w + v_head, kv_lora)
    return (w[:, :nope_w].permute(2, 0, 1).reshape(kv_lora, heads * nope_w).contiguous(),
            w[:, nope_w:].permute(2, 0, 1).reshape(kv_lora, heads * v_head).contiguous())


def attend(q_nope: torch.Tensor, q_pe: torch.Tensor, latent: torch.Tensor, k_pe_roped: Optional[torch.Tensor],
           w_uk: torch.Tensor, w_uv: torch.Tensor, sm_scale: float, sel: Optional[torch.Tensor] = None,
           past_len: int = 0) -> torch.Tensor:
    """Attention over latent rows (this layer's or its KV source's cache): unabsorbed scores
    identical to the engine's absorbed form (refs/mla). sm_scale = qk_head^-0.5, no YaRN mscale.
    sel: (B,S,k) int32 selection (-1 padded) or None for causal dense."""

    B, S, H, _ = q_nope.shape
    T = latent.shape[1]
    nope_w, v_dim = q_nope.shape[-1], w_uv.shape[1] // H
    k = (latent @ w_uk).view(B, T, H, nope_w)
    v = (latent @ w_uv).view(B, T, H, v_dim)
    scores = torch.einsum("bshd,bthd->bhst", q_nope, k)
    if k_pe_roped is not None:
        scores = scores + torch.einsum("bshr,btr->bhst", q_pe, k_pe_roped)
    scores = scores * sm_scale
    if sel is None:
        allow = (torch.arange(T, device=scores.device)[None, None, None, :] <=
                 (torch.arange(S, device=scores.device) + past_len)[None, None, :, None]).expand(B, H, S, T)
    else:
        sel = sel.view(B, S, -1).long()
        chosen = (sel[..., None] == torch.arange(T, device=scores.device)).any(-2)
        causal = (torch.arange(T, device=scores.device)[None, None, None, :] <=
                  (torch.arange(S, device=scores.device) + past_len)[None, None, :, None])
        allow = chosen.unsqueeze(1) & causal
    p = torch.softmax(scores.masked_fill(~allow, float("-inf")), dim=-1)
    return torch.einsum("bhst,bthd->bshd", p, v)


# -- experts (D1 §9/§10; refs/moe.py) -----------------------------------------------------------------------------

def route_softmax(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 softmax over all routed logits -> top-k -> renormalize over the k (bias 0,
    routed_scaling_factor 1.0, D1 §9). Weights == softmax restricted to the selected logits."""

    w_full = torch.softmax(logits.to(F32), dim=-1)
    top_w, top_i = w_full.topk(top_k, dim=-1)
    return top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-30), top_i


def silu_expert(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    """GatedMLP: down(silu(gate(x)) * up(x)) — plain SiLU, no swiglu clamp (D1 §9/§10)."""

    return F.silu(x @ gate_w.float().t()) * (x @ up_w.float().t()) @ down_w.float().t()


# -- the cache (refs/tiny_model naive planes; the engine swaps this for paged planes in 2b) ------------------------

def new_cache(cfg: Cfg) -> dict:
    """Per-layer planes: (1,T,kv_lora) normalized latent rows, (1,T,rd) roped k_pe rows, and the
    full-indexer layers' own (1,T,D_i) key plane. Readers and shared layers keep zero-width
    planes: they publish nothing (D1 §3; refs semantics — the plane starts when the budget binds)."""

    n = cfg.n_layers
    return {"lat": [None] * n, "kpe": [None] * n, "ixk": [None] * n, "past": 0}


# -- the model ----------------------------------------------------------------------------------------------------

class Ml4Model:
    """One rank's ML4 forward over phase-1 ``Ml4Weights``. CPU: plain tensors (fp32). CUDA world=1:
    Exl3Linear/Exl3RoutedExperts slots from ``weights.load(model_dir, rank=0, world=1)``."""

    def __init__(self, w: Ml4Weights):
        self.w = w
        self.cfg = w.cfg
        # layer 0 is dense (experts None on every build): detect on the first SPARSE layer's slot type
        # layer 0 is dense (experts None on every build), so detect on ANY sparse layer: the CPU regime
        # holds expert triples as tuples, the CUDA regime an Exl3RoutedExperts slot set (has .count as int)
        self.cuda_experts = any(lay.experts is not None and not isinstance(lay.experts, (tuple, list))
                                for lay in w.layers)
        self.comm = None                                        # set by the engine when world > 1 (cuda/a2a.py)
        self.inv_freq = lerope_inv_freq(self.cfg, w.alpha.float()) if w.cfg.rd else None
        self.folds = [fold_kv_b(lay.kv_b, self.cfg.n_heads, self.cfg.qk_head - (self.cfg.rd if lay.rope else 0),
                                self.cfg.v_head) for lay in w.layers]
        self.prev_sel: Optional[torch.Tensor] = None       # nearest preceding full layer, per forward
        self._scratch: dict = {}
        self._slot_lut: dict = {}
        self._expert_pos: dict = {}

    # -- linear adapters: torch tensors on CPU, Exl3Linear on CUDA ------------------------------------------------

    def _lin(self, mod, x: torch.Tensor) -> torch.Tensor:
        """x (..., k) @ W (n, k).T with widening fp32 math. Exl3Linear is its own decode kernel
        (rows <= 128 per call; chunked here — grouped prefill is phase 2b engine plumbing)."""

        if isinstance(mod, torch.Tensor):
            return x.float() @ mod.float().t()
        k = mod.k
        rows = x.reshape(-1, k).contiguous()
        if rows.shape[0] == 0:                               # DP4 idle rank: no rows to matmul
            return torch.zeros((*x.shape[:-1], mod.n), dtype=F32, device=x.device)
        outs = [mod(rows[i:i + 128].half()).float() for i in range(0, rows.shape[0], 128)]
        out = outs[0] if len(outs) == 1 else torch.cat(outs, 0)
        return out.view(*x.shape[:-1], mod.n)

    @staticmethod
    def _mat(mat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ mat.float().t()

    # -- experts ---------------------------------------------------------------------------------------------------

    def _moe(self, lay: Ml4Layer, xr: torch.Tensor) -> torch.Tensor:
        """fp32 partial [R, d]: sum_e w_e·expert_e(x) + shared(x). CPU: eager loop over the weight
        triples (shared last). CUDA: exl3_experts.routed with the shared as the last pick slot,
        weight 1.0 — the dsv41 packing (D1 §9), picks mapped global id -> local slot."""

        cfg = self.cfg
        if self.cuda_experts or xr.shape[0] == 0:
            logits = self._mat(lay.gate_w, xr)
        else:                                                # CPU regime batch-1 invariant (see _shared_cpu)
            logits = torch.cat([self._mat(lay.gate_w, xr[r:r + 1]) for r in range(xr.shape[0])], 0)
        w, idx = route_softmax(logits, cfg.topk)
        if self.comm is not None:                            # EP4: two-phase all-to-all (cuda/a2a.py, D2 §3)
            from .a2a import moe_all_to_all

            if not self.cuda_experts:
                trip = lay.experts
                pos = self._expert_pos.get(lay.idx)
                if pos is None:
                    pos = {int(e): k for k, e in enumerate(lay.expert_ids)}
                    self._expert_pos[lay.idx] = pos
                routed = moe_all_to_all(self.comm, xr, idx, w, n_routed=cfg.n_routed,
                                        run_local=lambda e, rows: silu_expert(rows, *trip[pos[e]]))
                return routed + self._shared_cpu(trip[-1], xr)   # shared expert: rank-local (D2 §2 replicated)
            routed = moe_all_to_all(self.comm, xr, idx, w, n_routed=cfg.n_routed, run_batch=self._ep4_batch(lay))
            return routed + self._shared_cuda(lay, xr)
        if not self.cuda_experts:
            triples: Sequence = lay.experts
            out = torch.zeros(xr.shape[0], cfg.dim, dtype=F32, device=xr.device)
            for r in range(xr.shape[0]):
                row = xr[r:r + 1]
                for slot in range(cfg.topk):                    # slot-ascending add order, batch-1 (see _shared_cpu)
                    g, u, d = triples[int(idx[r, slot])]
                    out[r] = out[r] + w[r, slot] * silu_expert(row, g, u, d)[0]
                out[r] = out[r] + self._shared_cpu(triples[-1], row)[0]
            return out
        from tensorfold.cuda.exl3 import experts as exl3_experts

        ex = lay.experts
        dev = lay.gate_w.device
        lut = self._slot_lut.get(lay.idx)
        if lut is None:                                      # global expert id -> local slot (shared last)
            lut = torch.full((cfg.n_routed,), ex.count, dtype=torch.int32, device=dev)
            for slot, gid in enumerate(lay.expert_ids):
                lut[gid] = slot
            self._slot_lut[lay.idx] = lut
        R, slots = xr.shape[0], cfg.topk + 1
        pick = torch.cat([lut[idx.reshape(-1)].view(R, cfg.topk),
                          torch.full((R, 1), ex.count - 1, dtype=torch.int32, device=dev)], 1).contiguous()
        wts = torch.cat([w, torch.ones((R, 1), dtype=F32, device=dev)], 1).contiguous()
        key = (lay.idx, R <= 64)
        s = self._scratch.get(key)
        if s is None or s.rows < R:
            self._scratch[key] = s = exl3_experts.Scratch(ex, max(R, 128), slots, device=dev)
        return exl3_experts.routed(xr.half().contiguous(), pick, wts, ex, s, None, R, act_mode=exl3_experts.ACT_F32)

    def _shared_cpu(self, shared_triple: Sequence, xr: torch.Tensor) -> torch.Tensor:
        """Shared expert in the CPU regime at BATCH 1 per row: torch CPU GEMM rounds differently at
        different M (gemv vs gemm blocking), so every expert application in this regime — routed
        picks in the a2a included — evaluates one row at a time. Then splitting the batch across
        ranks (DP4) cannot perturb a single bit; the CUDA regime is the exl3 batch regime instead
        (fp16 partials, D2's documented rounding)."""

        out = torch.zeros(xr.shape[0], self.cfg.dim, dtype=F32, device=xr.device)
        for r in range(xr.shape[0]):
            out[r] = silu_expert(xr[r:r + 1], *shared_triple)[0]
        return out

    def _ep4_batch(self, lay: Ml4Layer):
        """Receiver-side CUDA own-share partial (D2 §3): ``routed`` over a received batch (its expert
        ids are this rank's by the block rule), the shared expert excluded — the owner adds its own
        share locally, and partials sum in destination-rank order."""

        from tensorfold.cuda.exl3 import experts as exl3_experts

        ex = lay.experts

        def run(recv_x: torch.Tensor, eids: torch.Tensor, wts: torch.Tensor):
            dev = recv_x.device
            lut = self._slot_lut.get(lay.idx)
            if lut is None:
                lut = torch.full((self.cfg.n_routed,), ex.count, dtype=torch.int32, device=dev)
                for slot, gid in enumerate(lay.expert_ids):
                    lut[gid] = slot
                self._slot_lut[lay.idx] = lut
            n = eids.shape[0]
            if n == 0:
                return torch.zeros((0, self.cfg.dim), dtype=F32, device=dev), 0
            pick = torch.where(eids >= 0, lut[eids.clamp(min=0)], ex.count).contiguous()
            key = ("ep4", lay.idx, n <= 64)
            s = self._scratch.get(key)
            if s is None or s.rows < n:
                self._scratch[key] = s = exl3_experts.Scratch(ex, max(n, 128), self.cfg.topk, device=dev)
            vec = exl3_experts.routed(recv_x.half().contiguous(), pick, wts.contiguous(), ex, s, None, n,
                                      act_mode=exl3_experts.ACT_F32)
            return vec.float(), int((eids >= 0).sum())

        return run

    def _shared_cuda(self, lay: Ml4Layer, xr: torch.Tensor) -> torch.Tensor:
        """The shared expert on CUDA slots: ``routed`` with every row picking the last slot, weight 1.0."""

        from tensorfold.cuda.exl3 import experts as exl3_experts

        ex, R = lay.experts, xr.shape[0]
        if R == 0:
            return torch.zeros((0, self.cfg.dim), dtype=F32, device=xr.device)
        pick = torch.full((R, 1), ex.count - 1, dtype=torch.int32, device=xr.device)
        wts = torch.ones((R, 1), dtype=F32, device=xr.device)
        key = ("shared", lay.idx, R <= 64)
        s = self._scratch.get(key)
        if s is None or s.rows < R:
            self._scratch[key] = s = exl3_experts.Scratch(ex, max(R, 128), 1, device=xr.device)
        return exl3_experts.routed(xr.half().contiguous(), pick, wts, ex, s, None, R,
                                   act_mode=exl3_experts.ACT_F32).float()

    # -- one layer (D1 §1-§6, §9, §10 in forward order) --------------------------------------------------------------

    def _layer(self, L: int, x: torch.Tensor, positions: torch.Tensor, cache: dict) -> torch.Tensor:
        cfg, lay = self.cfg, self.w.layers[L]
        B, S, _ = x.shape
        if S == 0:                       # DP4 idle rank (D2 §1): no rows this round, but the layer's
            if not lay.dense:            # MoE a2a is lane-wide — serve dispatches landing on our experts
                self._moe(lay, x.reshape(0, cfg.dim))
            return x, None               # attention/indexer/caches: nothing to compute without rows
        past = cache["past"]
        rope_w = cfg.rd if lay.rope else 0                   # D1 §2: NoPE rotates nothing
        nope_w = cfg.qk_head - rope_w
        xn = rms_norm(x, lay.attn_norm, cfg.eps)

        # KV rows: writers own kv_a (+kv_a_layernorm on the latent, BEFORE kv_b); readers read the
        # source's rows with their OWN kv_b fold and append nothing (D1 §3).
        src = lay.kv_source
        if src == lay.idx:
            ckv = self._mat(lay.kv_a, xn)
            lat = rms_norm(ckv[..., :cfg.kv_lora], lay.kv_a_norm, cfg.eps)
            if rope_w:
                kpe = gptj_rotate(ckv[..., cfg.kv_lora:].reshape(-1, rope_w), self.inv_freq,
                                  positions).view(B, S, rope_w)
            else:
                kpe = None
            cache["lat"][L] = lat if cache["lat"][L] is None else torch.cat([cache["lat"][L], lat], 1)
            cache["kpe"][L] = kpe if cache["kpe"][L] is None else torch.cat([cache["kpe"][L], kpe], 1)
            rows_lat, rows_kpe = cache["lat"][src], cache["kpe"][src]
        else:
            rows_lat, rows_kpe = cache["lat"][src], cache["kpe"][src]
            assert rows_lat is not None and rows_lat.shape[1] >= past + S, \
                f"layer {L}: KV source {src} has no rows for this chunk (it must run first)"

        # DSA selection: full layers score + select once the visible context exceeds the budget
        # (fits-when-short rows select their whole visible set == dense); shared layers reuse the
        # nearest preceding full layer's selection, per forward (D1 §6).
        sel = None
        if lay.indexer != "shared" and past + S > cfg.idx_topk:
            keys = indexer_keys(xn, lay.idx_wk, lay.idx_k_norm, lay.idx_k_norm_b, cfg.index_norm_eps,
                                self.inv_freq, rope_w, positions)
            cache["ixk"][L] = keys if cache["ixk"][L] is None else torch.cat([cache["ixk"][L], keys], 1)
            sc = indexer_scores(xn, positions, qn_w=lay.idx_q_norm, qn_b=lay.idx_q_norm_b,
                                wq_b_w=lay.idx_wq_b, wp_w=lay.idx_proj, ss_s=lay.idx_ssmax_s,
                                ss_b=lay.idx_ssmax_b, eps=cfg.index_norm_eps, n_heads=cfg.idx_heads,
                                head_dim=cfg.idx_dim, inv_freq=self.inv_freq, rope_w=rope_w,
                                keys=cache["ixk"][L])
            sel = select_topk(sc + causal_mask(sc, offset=past), cfg.idx_topk)
            self.prev_sel = sel
        elif lay.indexer == "shared" and past + S > cfg.idx_topk:
            assert self.prev_sel is not None, f"layer {L}: shared indexer found no previous selection"
            sel = self.prev_sel

        # queries: straight from the block input (no q-LoRA), ssmax on the whole head, then the
        # trailing rope slice (absent on NoPE layers).
        q = self._lin(lay.q, xn).view(B, S, cfg.n_heads, cfg.qk_head)
        q = ssmax_apply(q, lay.ssmax_s, lay.ssmax_b, positions)
        w_uk, w_uv = self.folds[L]
        q_nope = q[..., :nope_w]
        q_pe = rotate_leading(q[..., nope_w:], self.inv_freq, positions, rope_w) if rope_w else q[..., nope_w:]
        o_lat = attend(q_nope, q_pe, rows_lat, rows_kpe, w_uk, w_uv, cfg.sm_scale, sel=sel, past_len=past)
        x = x + self._lin(lay.o, o_lat.reshape(B, S, cfg.n_heads * cfg.v_head))

        xn2 = rms_norm(x, lay.ffn_norm, cfg.eps)
        xr = xn2.reshape(B * S, cfg.dim)
        if lay.dense:                                        # D1 §10: dense layer 0, intermediate 24576
            # _lin, not silu_expert: the quantized dense weights are Exl3Linear on CUDA (node smoke
            # step 1b, 2026-11-06: dense layer 0 crashed on '.float()'). _lin's plain-tensor branch is
            # op-identical to silu_expert for fp32 xr, so the CPU bit-exact regime does not move.
            part = self._lin(lay.dense_down, F.silu(self._lin(lay.dense_gate, xr)) * self._lin(lay.dense_up, xr))
        else:
            part = self._moe(lay, xr)
        return x + part.view(B, S, cfg.dim), sel

    # -- full forward ------------------------------------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor, cache: dict, collect: Optional[list] = None,
                sel_out: Optional[list] = None) -> torch.Tensor:
        """tokens (1, S) long; cache from new_cache(cfg), mutated in place; returns logits (S, V).

        collect: per-layer residual streams (1,S,H) appended for diagnostics (the reference harness)."""

        cfg = self.cfg
        assert tokens.shape[0] == 1, "B == 1 in phase 2a (batching is the phase-2b engine)"
        self.prev_sel = None
        x = self.w.embed[tokens].float()
        S = tokens.shape[1]
        positions = torch.arange(cache["past"], cache["past"] + S, device=x.device)
        for L in range(cfg.n_layers):
            x, sel = self._layer(L, x, positions, cache)
            if sel_out is not None:
                sel_out.append((L, sel))
            if collect is not None:
                collect.append(x.clone())
        cache["past"] += S
        h = rms_norm(x, self.w.norm, cfg.eps)
        logits = self._lin(self.w.head, h) if self.w.head is not None else self._mat(self.w.head_bf16, h)
        return softcap(logits, cfg.softcap)[0]
