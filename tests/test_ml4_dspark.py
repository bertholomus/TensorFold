"""CPU gates for the draft head (cuda/dspark.py): its block and drafts against a plain dense reference on tiny random weights, whole and
split over ranks, and the served lane's steps with drafts: the same tokens as without, whatever the drafts were.

Run directly, no pytest, no numpy (the host-test venv):

    $ML4_HOST_TESTS/.venv/bin/python tests/test_ml4_dspark.py
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_ml4_cp4 as TC                                        # noqa: E402
import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from tensorfold.families.mistral_large4.cuda import cp4, dspark, tp   # noqa: E402
from tensorfold.families.mistral_large4.cuda.model import gptj_rotate, rms_norm, softcap, yarn_inv_freq   # noqa: E402

BF16 = torch.bfloat16
C = SimpleNamespace(dim=64, n_heads=8, kv_lora=32, rd=16, nope=16, qk_head=32, v_head=32, eps=1e-5, sm_scale=32 ** -0.5, vocab=256,
                    rope_theta=10000.0, rope_factor=2048.0, orig_len=512, n_layers=6, softcap=30.0)
HIDDEN, MK, NT, DL = 96, 16, 5, 2


class Dict:
    """``dspark.Source`` over a dict."""

    def __init__(self, d: dict) -> None:
        self.d = d

    def keys(self) -> list:
        return list(self.d)

    def shape(self, name: str) -> list:
        return list(self.d[name].shape)

    def get(self, name: str, rows: slice = slice(None), cols: slice = slice(None)) -> torch.Tensor:
        t = self.d[name]
        return t[rows, cols] if t.dim() > 1 else t[rows]


def weights(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)

    def r(*shape, s=None):
        return (torch.randn(*shape, generator=g) * (s or shape[-1] ** -0.5)).to(BF16)

    def n(size):
        return (1 + 0.1 * torch.randn(size, generator=g)).to(BF16)

    d, nv = C.dim, C.nope + C.v_head
    w = {"backbone.target_projection.weight": r(d, NT * d), "backbone.target_norm.weight": n(d), "backbone.final_norm.weight": n(d),
         "backbone.mask_embedding": r(2, d, s=1.0), "autoregressive_head.embedding.weight": r(C.vocab, MK, s=1.0),
         "autoregressive_head.joint_proj.weight": r(3 * MK, 2 * MK + d), "autoregressive_head.joint_proj.bias": r(3 * MK, s=0.1),
         "autoregressive_head.output.weight": r(C.vocab, MK, s=1.0)}
    for i in range(DL):
        p = f"backbone.layers.{i}."
        w |= {p + "attention_norm.weight": n(d), p + "ffn_norm.weight": n(d), p + "attention.kv_a_norm.weight": n(C.kv_lora),
              p + "attention.wq.weight": r(C.n_heads * C.qk_head, d), p + "attention.wkv_a_with_mqa.weight": r(C.kv_lora + C.rd, d),
              p + "attention.wkv_b.weight": r(C.n_heads * nv, C.kv_lora), p + "attention.wo.weight": r(d, C.n_heads * C.v_head),
              p + "feed_forward.w1.weight": r(HIDDEN, d), p + "feed_forward.w3.weight": r(HIDDEN, d), p + "feed_forward.w2.weight": r(d, HIDDEN)}
    return {f"mtp_dspark.{k}": v for k, v in w.items()}


class Lane(tp.Ml4Tp):
    """Only what the head takes from a lane, with a plain head and embedding."""

    def __init__(self, rank: int, world: int, comm, head: torch.Tensor, embed: torch.Tensor) -> None:
        self.cfg, self.rank, self.world, self.comm = C, rank, world, comm
        self.vl = C.vocab // world
        self.head = head[rank * self.vl:(rank + 1) * self.vl]
        self.ops, self.w = cp4.Ops(), SimpleNamespace(embed=embed)
        self.pool = SimpleNamespace(device=torch.device("cpu"))


def reference(w: dict, head: torch.Tensor, embed: torch.Tensor, taps: torch.Tensor, pos: torch.Tensor, anchor: int, p: int) -> tuple:
    """The head as a plain model: per head keys and values, every step spelled out, fp32 on the same weights."""

    def t(name):
        return w[f"mtp_dspark.{name}"].float()

    d, nv, Cl = C.dim, C.nope + C.v_head, C.kv_lora
    inv = yarn_inv_freq(C.rd, C.rope_theta, None, factor=C.rope_factor, original_max=C.orig_len)
    main = rms_norm(taps.float() @ t("backbone.target_projection.weight").T, t("backbone.target_norm.weight"), C.eps)
    block = torch.tensor([p, p + 1, p + 2])
    x = torch.cat([embed[anchor].float()[None], t("backbone.mask_embedding")])
    for i in range(DL):
        a = f"backbone.layers.{i}."

        def latents(rows, at):
            kv = rows @ t(a + "attention.wkv_a_with_mqa.weight").T
            return rms_norm(kv[:, :Cl], t(a + "attention.kv_a_norm.weight"), C.eps), gptj_rotate(kv[:, Cl:], inv, at)

        ctx_c, ctx_pe = latents(main, pos)
        xn = rms_norm(x, t(a + "attention_norm.weight"), C.eps)
        blk_c, blk_pe = latents(xn, block)
        lat, pe = torch.cat([ctx_c, blk_c]), torch.cat([ctx_pe, blk_pe])
        kvb = (lat @ t(a + "attention.wkv_b.weight").T).view(-1, C.n_heads, nv)
        q = (xn @ t(a + "attention.wq.weight").T).view(DEPTH, C.n_heads, C.qk_head)
        q_pe = gptj_rotate(q[..., C.nope:].reshape(-1, C.rd), inv, block.repeat_interleave(C.n_heads)).view(DEPTH, C.n_heads, C.rd)
        s = torch.einsum("qhd,khd->qhk", q[..., :C.nope], kvb[..., :C.nope]) + torch.einsum("qhd,kd->qhk", q_pe, pe)
        o = torch.einsum("qhk,khd->qhd", (s * C.sm_scale).softmax(-1), kvb[..., C.nope:]).reshape(DEPTH, -1)
        x = x + o @ t(a + "attention.wo.weight").T
        xn = rms_norm(x, t(a + "ffn_norm.weight"), C.eps)
        x = x + (F.silu(xn @ t(a + "feed_forward.w1.weight").T) * (xn @ t(a + "feed_forward.w3.weight").T)) @ t(a + "feed_forward.w2.weight").T
    h = rms_norm(x, t("backbone.final_norm.weight"), C.eps)
    base = softcap(h @ head.float().T, C.softcap)
    state, prev, out = torch.zeros(MK), anchor, []
    for i in range(DEPTH):
        joint = torch.cat([state, t("autoregressive_head.embedding.weight")[prev], h[i]]) @ t("autoregressive_head.joint_proj.weight").T
        gate, cand, bias = (joint + t("autoregressive_head.joint_proj.bias")).chunk(3)
        gate = torch.sigmoid(gate)
        state = gate * state + (1 - gate) * torch.tanh(cand)
        prev = int((base[i] + torch.tanh(bias) @ t("autoregressive_head.output.weight").T).argmax())
        out.append(prev)
    return h, out


DEPTH = dspark.DEPTH


def block_case(world: int, seed: int, n_ctx: int) -> None:
    w = weights(seed)
    g = torch.Generator().manual_seed(seed + 1)
    head = torch.randn(C.vocab, C.dim, generator=g).to(BF16)
    embed = torch.randn(C.vocab, C.dim, generator=g).to(BF16)
    taps = torch.randn(n_ctx, NT * C.dim, generator=g).to(BF16)
    pos = torch.arange(n_ctx)
    anchor = int(torch.randint(0, C.vocab, (1,), generator=g))
    want_h, want = reference(w, head, embed, taps, pos, anchor, n_ctx)

    def body(r, comm):
        lane = Lane(r, world, comm, head, embed)
        d = dspark.Drafter(lane, Dict(w), 8, taps=(1, 2, 3, 4))
        d.absorb(taps, pos, 5 * dspark.ROWS + pos % dspark.WINDOW)
        io = d._capture(1, True)
        slot = 5
        rows = torch.tensor([slot * dspark.ROWS + dspark.WINDOW + j for j in range(DEPTH)])
        x = io.x.clone()
        x[0] = embed[anchor].float()
        h = d.block(x, n_ctx + torch.arange(DEPTH), rows, torch.tensor([slot]), torch.tensor([n_ctx]))
        return h, d.pick(h, torch.tensor([anchor]))[0].tolist()

    got = TC.run_ranks(body, world) if world > 1 else [body(0, None)]
    for h, drafts in got:
        assert TC.cos(h, want_h) > 0.999, f"hidden rows off: cos {TC.cos(h, want_h)}"
        assert drafts == got[0][1], "ranks disagree on the drafts"
    agree = sum(a == b for a, b in zip(got[0][1], want))
    print(f"  world {world}, {n_ctx} context rows: cos {TC.cos(got[0][0], want_h):.6f}, drafts {got[0][1]} against {want}")
    assert agree >= 2, f"drafts {got[0][1]} != {want}"


def test_block():
    for world in (1, 2, 4):
        for seed, n_ctx in ((0, 9), (1, 40)):
            block_case(world, seed, n_ctx)


def test_window():
    """A prompt's taps through its rounds on every rank, short and past the window: the same rows in a slot as absorbing them in order."""

    W = 4
    old, dspark.WINDOW, dspark.KEYS, dspark.ROWS = (dspark.WINDOW, dspark.KEYS, dspark.ROWS), 32, 32 + DEPTH, 33 + DEPTH
    try:
        for prompt in (5, 9, 70):
            w = weights(3)
            g = torch.Generator().manual_seed(4)
            head, embed = torch.randn(C.vocab, C.dim, generator=g).to(BF16), torch.randn(C.vocab, C.dim, generator=g).to(BF16)
            taps = torch.randn(NT, prompt, C.dim, generator=g).to(BF16)

            def body(r, comm):
                lane = Lane(r, W, comm, head, embed)
                d = dspark.Drafter(lane, Dict(w), 8, taps=(1, 2, 3, 4))
                st = cp4.Stripe(W, 4)
                win = dspark.Window(st, r, prompt, 2, d)
                for start in range(0, prompt, 16):
                    p = torch.arange(start, min(start + 16, prompt))
                    own = p[st.rank(p) == r]
                    keep = win.round(own)
                    if keep is not None:
                        for t, L in enumerate((1, 2, 3, 4, C.n_layers)):
                            keep.put(L, taps[t, own].float())
                d.close(win)
                got = d.ring[:, 2 * dspark.ROWS:2 * dspark.ROWS + dspark.WINDOW].clone()
                pos = torch.arange(max(0, prompt - dspark.WINDOW), prompt)
                d.absorb(taps[:, pos].permute(1, 0, 2).reshape(pos.numel(), -1), pos, 3 * dspark.ROWS + pos % dspark.WINDOW)
                return got, d.ring[:, 3 * dspark.ROWS:3 * dspark.ROWS + dspark.WINDOW].clone()

            for got, want in TC.run_ranks(body, W):
                assert torch.equal(got, want), f"{prompt} rows: the window's rows differ"
            print(f"  {prompt} prompt rows: same ring")
    finally:
        dspark.WINDOW, dspark.KEYS, dspark.ROWS = old


class Oracle:
    """A head that knows the answers and gets some wrong on purpose: every ``bad``-th position it says something else."""

    def __init__(self, full: dict, bad: int) -> None:
        self.full, self.bad, self.calls = full, bad, 0

    def step(self, plan: list, drafts: bool = True, wait: bool = True) -> list:
        self.calls += 1
        out = []
        for slot, _first, kept, at, tok in plan:
            full, p = self.full[slot], at + kept
            assert full[p] == tok, f"slot {slot}: anchor {tok} at {p}, want {full[p]}"
            d = [full[q] if q < len(full) else 0 for q in range(p + 1, p + 1 + DEPTH)]
            out.append([(t + 1) % C.vocab if (p + j) % self.bad == 0 else t for j, t in enumerate(d)])
        return out if wait else torch.tensor(out)

    def close(self, w) -> None:
        pass


def test_engine_steps():
    """The lane's rows with drafts in them: four engines, the tokens of a lane without any, at every mix of right and wrong drafts."""

    import test_ml4_engine as TE

    from tensorfold.families.mistral_large4.cuda import dp4 as DP
    from tensorfold.families.mistral_large4.cuda import engine as EN

    def quiet(_s):
        return None

    W = 4
    _, w = TC.tiny(5)
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, w.cfg.vocab, (n,), generator=g).tolist() for n in (41, 13)]
    count = 12
    solo = EN.Ml4Engine(None, rank=0, world=1, device="cpu", weights=w, pool_tokens=4096, log=quiet)
    want = []
    for p in prompts:
        want.append([])
        solo.generate(p, count, stop_eos=False, on_tokens=TE.feed(want[-1]))
    old, cp4.BLOCK = cp4.BLOCK, 4
    was, EN.TP = EN.TP, True
    comms, link = TC.emu_comms(W), DP.EmuLink(W, 0)
    try:
        engines = [EN.Ml4Engine(None, rank=r, world=W, device="cpu", weights=TE.slice_weights(w, r, W), comm=comms[r],
                                link=link.fork(r), pool_tokens=256, log=quiet, fast=True) for r in range(W)]
        for bad in (1000, 2, 3):
            full = {15: prompts[0] + want[0], 14: prompts[1] + want[1]}
            for e in engines:
                e.draft, e.draft_rows, e.free_slots = Oracle(full, bad), DEPTH + 1, list(range(16))
            e = engines[0]
            e.concurrent = False
            ts = [threading.Thread(target=x.follow, daemon=True) for x in engines[1:]]
            [t.start() for t in ts]
            a, b = [], []
            s1 = e.admit(prompts[0], count, stop_eos=False, on_tokens=TE.feed(a))
            s2 = e.admit(prompts[1], count, stop_eos=False, on_tokens=TE.feed(b))
            assert (s1.slot, s2.slot) == (15, 14)
            steps = 0
            while not (s1.done and s2.done):
                assert e.run_round(), "a live stream got no round"
                steps += 1
            e.link.send(["stop"])
            [t.join(60) for t in ts]
            assert not any(t.is_alive() for t in ts), "a follower hung"
            assert a == want[0] and b == want[1], f"bad {bad}: {a} != {want[0]} or {b} != {want[1]}"
            assert all(x.runs == [[0, 256]] for x in e.extents), f"extents leaked: {[x.runs for x in e.extents]}"
            assert sorted(e.free_slots) == list(range(16)), "a slot leaked"
            print(f"  every {bad}th draft wrong: {steps} rounds, {e.draft.calls} draft steps")
    finally:
        cp4.BLOCK, EN.TP = old, was


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"ok   {t.__name__}")
        except BaseException:                                    # noqa: BLE001
            import traceback

            bad += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    sys.exit(1 if bad else 0)
