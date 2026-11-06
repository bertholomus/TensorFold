"""``cp4.Ops`` on CUDA: EXL3 linears and experts, the two Triton kernels, fp16 on the wire (DESIGN-CP4 §3).

Precision follows the served lanes rather than the fp32 CPU gates: activations enter EXL3 linears as fp16 (the kernels
take nothing else), accumulation is fp32, the residual stream and the router stay fp32. kv_a, wk and the indexer's head
weights stay the eager fp32 matmuls (they are small next to the rest); queries travel as the projections' own fp16
output, so nothing on the wire rounds them again.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tensorfold.cuda.exl3 import experts as exl3_experts
from tensorfold.cuda.exl3 import format as exl3_format
from tensorfold.cuda.exl3 import prefill as exl3_prefill
from tensorfold.cuda.exl3.linear import Exl3Linear

from . import cp4
from .cp4_kernels import fp8_linear, index_rows, index_scores, small_mat, sparse_mla

F16, F32 = torch.float16, torch.float32
SCORE_BYTES = int(os.environ.get("TF_ML4_CP_SCORE_MB") or 256) << 20      # one tile of a round's scores
MAX_PICKS = int(os.environ.get("TF_ML4_CP_MAX_PICKS") or 5120)            # picks per expert launch: 85 KB of scratch each
WS_KEEP = 6144 * 8192                                                     # q and o's decoded size; the dense layer's goes back
PROMPT_KERNEL = os.environ.get("TF_ML4_PROMPT_KERNEL") or "mma2"          # 1.45x mma on a layer's experts at 4,096 picks
M2_ROWS = int(os.environ.get("TF_ML4_M2_ROWS") or 128)                    # rows a weight tile is decoded for (64: exl3's default)


def trim_host() -> None:
    """Hand back what the load and the JIT left in malloc's arenas: on GB10 host pages are GPU memory."""

    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def embed_from_pack(w) -> torch.Tensor:
    """The embedding as a read-only map of its bytes in the pack: 2.3 GiB less GPU memory, the rows a round reads come
    through the page cache, which gives its pages back under pressure (DESIGN-CP4 §10)."""

    from .weights import LM, PackIndex

    index = PackIndex(w.model_dir).load()
    file, dtype, shape, nbytes, off = index.entries[f"{LM}.embed_tokens.weight"]
    start = index.file_at[file] + off
    assert dtype == "BF16" and start % 2 == 0, f"embedding: {dtype} at byte {start}"
    raw = torch.from_file(str(w.model_dir / file), shared=False, size=index.file_sizes[file], dtype=torch.uint8)
    return raw[start:start + nbytes].view(torch.bfloat16).view(*shape)


@dataclass
class Fp8Block:
    """A plain weight as the pack keeps it: e4m3 [N, K], and the fp32 grid of its 128 x 128 blocks."""

    w: torch.Tensor
    scale: torch.Tensor


@dataclass
class Exl3Part(Exl3Linear):
    """A block of a layer's K x N, in the layer's own words: tiles are addressed by strides, so a part is a pointer and the
    whole layer's strides. ``words`` is a flat view from the block's first tile, as long as the kernels' size check wants;
    their reads run past its end, inside the layer."""

    whole: tuple = (0, 0)

    @property
    def strides(self) -> tuple[int, int]:
        return self.whole


def exl3_part(m: Exl3Linear, k0: int, k1: int, c0: int, c1: int) -> Exl3Part:
    """Rows k0..k1 and columns c0..c1 of ``m`` (multiples of 128: its rotations work in 128-wide blocks)."""

    assert not (k0 % 128 or k1 % 128 or c0 % 128 or c1 % 128) and m.bias is None, f"part {k0, k1, c0, c1} of {m.k} x {m.n}"
    ks, nbs = m.strides
    at = (c0 // 128) * nbs + (k0 // 16) * ks
    words = m.words.view(-1)[at:at + (k1 - k0) * (c1 - c0) * m.k2 // 64]
    return Exl3Part(words, m.suh[k0:k1], m.svh[c0:c1], None, m.bits, m.codebook, k1 - k0, c1 - c0, m.layout, None, (ks, nbs))


def shared_linears(ex) -> tuple:
    """The shared expert (the last of ``ex``) as three Exl3Linears over its stored trellises."""

    e = ex.count - 1
    name = {exl3_experts.CB_3INST: "3inst", exl3_experts.CB_MCG: "mcg", exl3_experts.CB_MUL1: "mul1"}[ex.cb]

    def linear(t: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> Exl3Linear:
        return Exl3Linear(t.view(torch.int32), suh, svh, None, exl3_format.bits_of(t.shape), name, t.shape[0] * 16,
                          t.shape[1] * 16, "stored")

    return (linear(ex.keep[e], ex.suh_g[e], ex.svh_g[e]), linear(ex.keep[ex.count + e], ex.suh_u[e], ex.svh_u[e]),
            linear(ex.keep[2 * ex.count + e], ex.suh_d[e], ex.svh_d[e]))


def wq_b_from_pack(w) -> None:
    """The indexer's wq_b back to the pack's bytes, each layer checked bit for bit against the fp16 the load made: 1.03 GiB
    less on a rank, and a decode step reads half as much of it (DESIGN-CP4 §10)."""

    from tensorfold.cuda.exl3 import load as shared

    from .weights import LP, dequant_fp8_block

    sh = shared.Shards(w.model_dir)
    for i, lay in enumerate(w.layers):
        if lay.idx_wq_b is None:
            continue
        name = LP.format(L=i) + "self_attn.indexer.wq_b.weight"
        raw = sh.get(name).to(lay.idx_wq_b.device)
        scale = sh.get(name.replace(".weight", ".weight_scale_inv")).to(lay.idx_wq_b.device).contiguous()
        assert torch.equal(dequant_fp8_block(raw, scale), lay.idx_wq_b), f"layer {i}: wq_b's bytes are not the loaded weight"
        lay.idx_wq_b = Fp8Block(raw.contiguous(), scale)
    sh.close()


class CudaOps(cp4.Ops):
    wire = F16

    def __init__(self, device="cuda") -> None:
        self.device = device
        self.ws = exl3_prefill.Workspace()
        self.scratch: dict = {}
        self.lut: dict = {}

    def prepare(self, w) -> None:
        """kv_b as fp16, once, in place of the bf16 copy (bf16 fits fp16 down to its subnormals, 2^-14 here): the two
        absorbed matmuls read it every layer, and a per-call cast would cost a decode step about 1 GB of traffic."""

        for lay in w.layers:
            lay.kv_b = lay.kv_b.to(F16)
        if w.model_dir is not None and os.environ.get("TF_ML4_CP_EMBED_MAP", "1") != "0":
            host = embed_from_pack(w)
            probe = torch.tensor([0, 1, w.cfg.vocab // 2, w.cfg.vocab - 1])
            assert torch.equal(host[probe].to(w.embed.device), w.embed[probe.to(w.embed.device)]), "the embedding map is off"
            w.embed = host
        if w.model_dir is not None and os.environ.get("TF_ML4_CP_WQ_B_FP8", "1") != "0":
            wq_b_from_pack(w)
        trim_host()

    def embed(self, w, tokens: torch.Tensor) -> torch.Tensor:
        if w.embed.device.type == "cpu":
            return w.embed[tokens.cpu()].to(tokens.device).float()
        return w.embed[tokens].float()

    # -- linears ---------------------------------------------------------------------------------------------------

    def lin(self, mod, x: torch.Tensor) -> torch.Tensor:
        x = x.to(F16).contiguous()
        if isinstance(mod, torch.Tensor):
            return F.linear(x, mod.to(F16))
        m = x.shape[0]
        if m == 0:
            return torch.zeros((0, mod.n), dtype=F16, device=x.device)
        if m <= 128:
            return mod(x, out_dtype=F16)
        out = exl3_prefill.matmul(mod, x, torch.empty((m, mod.n), dtype=F16, device=x.device), self.ws)
        if self.ws.w.numel() > WS_KEEP:                          # one layer a round: its 288 MB can serve other transients
            self.ws.w = None
        return out

    def mat(self, mat: torch.Tensor | Fp8Block, x: torch.Tensor) -> torch.Tensor:
        if isinstance(mat, Fp8Block):                            # idx_wq_b, the one big plain matrix
            return fp8_linear(x.to(F16).contiguous(), mat.w, mat.scale)
        if mat.dtype == F16:
            return F.linear(x.to(F16), mat)
        if x.shape[0] in (1, 2, 4, 8):                           # a step: no fp32 copy of the weight
            return small_mat(x.float().contiguous(), mat)
        return x.float() @ mat.float().t()

    def mat32(self, mat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``mat`` in TF32: a deep prompt's rows on their way to fp8 or FP4 lose nothing they would keep."""

        torch.backends.cuda.matmul.allow_tf32 = True
        y = x.float() @ mat.float().t()
        torch.backends.cuda.matmul.allow_tf32 = False
        return y

    def router(self, gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.mat(gate, x)

    # -- tp.py's slices ----------------------------------------------------------------------------------------------

    def split_mat(self, mat, c0: int, c1: int):
        if isinstance(mat, Fp8Block):
            return Fp8Block(mat.w[c0:c1], mat.scale[c0 // 128:c1 // 128])
        if isinstance(mat, Exl3Linear):
            return exl3_part(mat, 0, mat.k, c0, c1)
        return mat[c0:c1]

    def split_in(self, mat, k0: int, k1: int):
        return exl3_part(mat, k0, k1, 0, mat.n)

    def split_ffn(self, lay, rank: int, world: int) -> tuple:
        gate, up, down = (lay.dense_gate, lay.dense_up, lay.dense_down) if lay.dense else shared_linears(lay.experts)
        if not lay.dense and lay.experts.width < gate.k:
            return gate, up, down                                # the load kept this rank's quarter
        w = gate.n // world
        c0, c1 = rank * w, (rank + 1) * w
        return exl3_part(gate, 0, gate.k, c0, c1), exl3_part(up, 0, up.k, c0, c1), exl3_part(down, c0, c1, 0, down.n)

    def lin32(self, mod, x: torch.Tensor) -> torch.Tensor:
        return mod(x.to(F16).contiguous(), out_dtype=F32)

    def ffn_part(self, t, x: torch.Tensor) -> torch.Tensor:
        x = x.to(F16).contiguous()
        act = F.silu(t.gate(x, out_dtype=F32)) * t.up(x, out_dtype=F32)
        return t.down(act.to(F16), out_dtype=F32)

    def moe_part(self, cp, lay, t, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ex, dev = lay.experts, x.device
        pick = self._lut(cp, lay, dev)[idx]
        routed = self._routed(ex, x.to(F16), pick, torch.where(pick < ex.count, w, torch.zeros_like(w)).float(), idx.shape[1])
        return routed + self.ffn_part(t, x)

    def split_moe(self, cp, lay, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ex, Q = lay.experts, x.shape[0]
        pick = torch.full((Q, idx.shape[1] + 1), ex.count - 1, dtype=torch.int32, device=x.device)
        pick[:, :-1] = idx
        wts = torch.ones((Q, idx.shape[1] + 1), dtype=F32, device=x.device)
        wts[:, :-1] = w
        return self._routed(ex, x, pick, wts, pick.shape[1])

    def logits(self, head, h: torch.Tensor) -> torch.Tensor:
        if isinstance(head, torch.Tensor):
            return h.float() @ head.float().t()
        return head(h.to(F16).contiguous(), out_dtype=F32)

    # -- selection ---------------------------------------------------------------------------------------------------

    def local_topk(self, qi: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, vis: torch.Tensor, n: int, k: int,
                   pos_of) -> tuple[torch.Tensor, torch.Tensor]:
        Q, dev = qi.shape[0], qi.device
        out_k = torch.full((Q, k), cp4.PAD_KEY, dtype=torch.int64, device=dev)
        out_r = torch.zeros((Q, k), dtype=torch.int32, device=dev)
        if n == 0:
            return out_k, out_r
        q16, w32 = qi.to(F16).contiguous(), wts.to(F32).contiguous()
        m = min(k, n)
        tile = max(64, SCORE_BYTES // (n * 4) // 64 * 64)
        for c0 in range(0, Q, tile):
            c1 = min(Q, c0 + tile)
            s = torch.empty((c1 - c0, n), dtype=F32, device=dev)  # the allocator's: the experts' transients reuse it
            index_scores(q16[c0:c1], w32[c0:c1], keys, vis[c0:c1], n, s)
            top = torch.topk(s, m, dim=1, sorted=False)
            key = cp4.score_keys(top.values, pos_of(top.indices))
            out_k[c0:c1, :m] = key.masked_fill(top.values == float("-inf"), cp4.PAD_KEY)
            out_r[c0:c1, :m] = top.indices.to(torch.int32)
        return out_k, out_r

    def local_topk_rows(self, qi: torch.Tensor, wts: torch.Tensor, keys: torch.Tensor, base: torch.Tensor,
                        vis: torch.Tensor, n: int, k: int, pos_of) -> tuple[torch.Tensor, torch.Tensor]:
        """A step's selection, no host read (a capture). ``n`` is at least ``k``: tp.py's score widths start there."""

        s = torch.empty((qi.shape[0], n), dtype=F32, device=qi.device)
        index_rows(qi.to(F16).contiguous(), wts.to(F32).contiguous(), keys, base, vis, n, s)
        top = torch.topk(s, k, dim=1, sorted=False)
        key = cp4.score_keys(top.values, pos_of(top.indices))
        return key.masked_fill(top.values == float("-inf"), cp4.PAD_KEY), top.indices.to(torch.int32)

    # -- attention ---------------------------------------------------------------------------------------------------

    def absorb_q(self, q_nope: torch.Tensor, wuk: torch.Tensor) -> torch.Tensor:
        """[Q, H, kv_lora] fp16 as a view of [H, Q, kv_lora] (the kernel takes the strides)."""

        return torch.bmm(q_nope.to(F16).transpose(0, 1), wuk).transpose(0, 1)

    def expand_o(self, o_lat: torch.Tensor, wuv: torch.Tensor) -> torch.Tensor:
        return torch.bmm(o_lat.transpose(0, 1), wuv.transpose(1, 2)).transpose(0, 1)

    def attend(self, q_lat: torch.Tensor, q_pe: torch.Tensor | None, pool: cp4.KvPool, src: int, rows: torch.Tensor,
               cnt: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        Q, H, C = q_lat.shape
        out = torch.empty((H, Q, C), dtype=F16, device=q_lat.device).transpose(0, 1)
        lse = torch.empty((Q, H), dtype=F32, device=q_lat.device)
        pe = None if q_pe is None else q_pe.to(F16).contiguous()
        sparse_mla(q_lat, pe, rows, cnt, pool.lat[src], sm_scale, out, lse)
        return out, lse

    # -- experts -----------------------------------------------------------------------------------------------------

    def _routed(self, ex, x: torch.Tensor, pick: torch.Tensor, w: torch.Tensor, slots: int,
                add: torch.Tensor | None = None) -> torch.Tensor:
        """One scratch for all layers (their experts have one shape); prompt sizes grow it, decode keeps its own."""

        R = x.shape[0]
        prompt = R >= exl3_experts.EXACT_ROWS
        key = (slots, prompt)
        s = self.scratch.get(key)
        if s is None or s.rows < R:
            self.scratch.pop(key, None)
            del s
            s = exl3_experts.Scratch(ex, rows=max(R, MAX_PICKS if prompt and slots == 1 else 64), slots=slots, device=x.device,
                                     prompt=prompt)
            self.scratch[key] = s
        return exl3_experts.routed(x.contiguous(), pick.contiguous(), w.contiguous(), ex, s, None, R,
                                   act_mode=exl3_experts.ACT_F32, add=add, kernel=PROMPT_KERNEL, m2_rows=M2_ROWS)

    def _routed_add(self, ex, x: torch.Tensor, pick: torch.Tensor, w: torch.Tensor, slots: int,
                    add: torch.Tensor) -> torch.Tensor:
        return self._routed(ex, x, pick, w, slots, add)

    def _flat(self, ex, x: torch.Tensor, pick: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """A prompt's picks, one per row, at most MAX_PICKS a launch."""

        out = [self._routed(ex, x[c:c + MAX_PICKS], pick[c:c + MAX_PICKS], w[c:c + MAX_PICKS], 1)
               for c in range(0, x.shape[0], MAX_PICKS)]
        return out[0] if len(out) == 1 else torch.cat(out)

    def _lut(self, cp, lay, dev) -> torch.Tensor:
        lut = self.lut.get(lay.idx)
        if lut is None:
            lut = torch.full((cp.cfg.n_routed,), lay.experts.count, dtype=torch.int32, device=dev)
            lut[torch.tensor(lay.expert_ids, device=dev)] = torch.arange(len(lay.expert_ids), dtype=torch.int32, device=dev)
            self.lut[lay.idx] = lut
        return lut

    def experts(self, cp, lay, rows: torch.Tensor, eids: torch.Tensor, wts: torch.Tensor,
                shared_from: int | None) -> torch.Tensor:
        ex, dev = lay.experts, rows.device
        n, k = eids.shape
        out = torch.zeros((n, cp.cfg.dim), dtype=F32, device=dev)
        if n == 0:
            return out
        rows = rows.to(F16)
        if k and n * k < exl3_experts.EXACT_ROWS:                # decode: rows as they are, the fused launches
            pick = torch.where(eids >= 0, self._lut(cp, lay, dev)[eids.clamp(min=0).long()], ex.count)
            out = self._routed(ex, rows, pick.to(torch.int32), wts, k)
        elif k:
            # a prompt: one scratch row per pick, not rows x slots (a received row carries 1-2 of its k picks here).
            # Slot by slot, so a row's picks add in slot order, whatever order the GPU runs the add in
            r_i, s_i = (eids >= 0).nonzero(as_tuple=True)
            lut = self._lut(cp, lay, dev)
            y = self._flat(ex, rows[r_i], lut[eids[r_i, s_i].long()].view(-1, 1), wts[r_i, s_i].view(-1, 1))
            for s in range(k):
                at = (s_i == s).nonzero().squeeze(1)
                out[r_i[at]] += y[at]
        if shared_from is not None:
            at = slice(shared_from, None)
            x = rows[at]
            pick = torch.full((x.shape[0], 1), ex.count - 1, dtype=torch.int32, device=dev)
            one = torch.ones((x.shape[0], 1), dtype=F32, device=dev)
            out[at] += self._flat(ex, x, pick, one)
        return out
