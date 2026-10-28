"""The prompt cache's scheduling on the host (CPU, no model): BeamEngine's own planner (_plan, _deliver and the entry
choices) drives a small Cache through prompts that share starts, re-sent prompts, agent-style growing prompts, salts,
scored prompts, cancels and a pool short of pages, then random traffic. Each step applies the plan the way _execute
does (releases, evictions, restores, reserves, then the step's writes, then saves) and checks:
- pages: every page free or held, a page's holder count its refs, no page written while another holder has it;
- contents: every page a sequence reads holds its own ids at those positions (a shared page holds the same start);
- rings and the drafter's context (tagged by id in the real tensors): a chunk after a restore finds the window before
  it, a sequence's first decode finds its drafter window, both copied by Cache._copy_rings / DFlash2.copy_context;
- the plan's encoding round-trips (what the other ranks run);
- a hit's prompt rows are the ones a fresh prefill would compute: every piece stays inside one chunk of the grid and
  has MIN_PIECE rows or more (rows come out the same in any such piece), or is a fresh prefill's own last chunk; an
  unscored prompt's pieces start on a page (where entries end), a scored prompt's on the grid (or the "pieces" knob's).
usage: python3 tools/beam_prompt_cache_check.py [STEPS]   (PC_CHECK_PASS at the end, else an AssertionError)"""

import random
import sys
import threading
from collections import deque
from types import SimpleNamespace

import torch

from tensorfold.families.beam.cuda import engine as E
from tensorfold.families.beam.cuda.cache import Cache
from tensorfold.families.beam.cuda.drafter import DFlash2

PAGE, CHUNK, SLOTS, ENTRIES, RING, DWIN = 256, 1024, 4, 4, 528, 4095
FINE = 256                  # entries off the grid from one page past the grid point (the engine's default is 768)


def tag(p: int, t: int) -> float:
    """Position p's id t as a ring or context tag (exact in bf16)."""
    return float((p * 31 + t) % 251)


class Drafter:
    """The drafter as the planner sees it: its window, its context tensors (an id tag a position) and its copy."""

    window = DWIN
    copy_context = DFlash2.copy_context

    def __init__(self, holders: int) -> None:
        self.ring = -(-(DWIN + 1 + 63) // 64) * 64
        self.ck = torch.full((1, holders, self.ring, 1, 1), -1.0)
        self.cv = torch.full((1, holders, self.ring, 1, 1), -1.0)


def engine(pages: int) -> E.BeamEngine:
    cfg = SimpleNamespace(window=512, kv_heads=1, full=[1], layers=3, sliding=lambda L: L != 1)
    eng = E.BeamEngine.__new__(E.BeamEngine)
    eng.cache = Cache(cfg, 1, pages=pages, page=PAGE, seqs=SLOTS, chunk=CHUNK, ring=RING, device="cpu",
                      entries=ENTRIES)
    eng.cache.k_ring.fill_(-1)
    eng.slots, eng.chunk, eng.entries, eng.limit = SLOTS, CHUNK, ENTRIES, pages * PAGE
    eng.fine, eng.test_pieces = FINE, 0
    eng.ent, eng._uses = [None] * ENTRIES, 0
    eng.drafter = Drafter(SLOTS + ENTRIES)
    eng.cv = threading.Condition()
    eng.waiting, eng.running, eng.releasing, eng._pending = deque(), [], [], []
    eng.debt, eng.decode_rows, eng.spec_k, eng.k_cap, eng.k_adapt, eng.lookup = 0.0, 16, 0, 0, (), False
    eng.eos = (-7,)
    return eng


class Sim:
    def __init__(self, pages: int, seed: int) -> None:
        self.eng = engine(pages)
        self.c = self.eng.cache
        self.content = {}                  # page -> {offset: id}
        self.rng = random.Random(seed)
        self.done: list[E.Request] = []
        self.hits = self.saves = self.evicts = self.steps = 0
        self.pieces = self.short = 0       # prompt pieces; pieces off the grid before a prompt's end

    def submit(self, prompt: list[int], max_tokens: int, salt=None, score: int = 0) -> E.Request:
        r = E.Request(list(prompt), max_tokens, None, False, need=len(prompt) + max_tokens, score=score, salt=salt)
        self.eng.waiting.append(r)
        return r

    def ids(self, r: E.Request, pos: int) -> int:
        return (r.prompt + r.out)[pos]

    def write(self, r: E.Request, p0: int, n: int) -> None:
        """A forward of r's positions [p0, p0 + n): its pages, its ring (the last RING), tagged by id."""
        c, slot = self.c, r.slot
        for p in range(p0, p0 + n):
            page = c.owned[slot][p // PAGE]
            assert c.refs[page] == 1, f"position {p} of slot {slot} written into page {page} that {c.refs[page]} hold"
            self.content.setdefault(page, {})[p % PAGE] = self.ids(r, p)
        keep = range(max(p0, p0 + n - RING), p0 + n)
        at = torch.tensor([slot * RING + p % RING for p in keep])
        val = torch.tensor([tag(p, self.ids(r, p)) for p in keep], dtype=torch.bfloat16)
        c.k_ring[0, at, 0, 0] = val
        c.v_ring[0, at, 0, 0] = val

    def ctx(self, entries) -> None:
        dr = self.eng.drafter
        by_slot = {r.slot: r for r in self.eng.running}
        for slot, first, _, n, _ in entries:
            r = by_slot[slot]
            at = torch.tensor([p % dr.ring for p in range(first, first + n)])
            val = torch.tensor([tag(p, self.ids(r, p)) for p in range(first, first + n)])
            dr.ck[0, slot, at, 0, 0] = val
            dr.cv[0, slot, at, 0, 0] = val

    def check_pages(self) -> None:
        c = self.c
        held: dict[int, int] = {}
        for h, pages in enumerate(c.owned):
            assert len(set(pages)) == len(pages), f"holder {h} holds a page twice"
            for p in pages:
                held[p] = held.get(p, 0) + 1
        assert len(set(c.free)) == len(c.free), "a page is free twice"
        for p in range(c.pages):
            assert c.refs[p] == held.get(p, 0), f"page {p}: refs {c.refs[p]}, holders {held.get(p, 0)}"
            assert (p in c.free) == (held.get(p, 0) == 0), f"page {p}: free {p in c.free}, holders {held.get(p, 0)}"
        kr = c.k_ring[0, :, 0, 0].tolist()
        dk = self.eng.drafter.ck[0, :, :, 0, 0].tolist()
        for i, x in enumerate(self.eng.ent):
            got = c.owned[SLOTS + i]
            if x is None:
                assert not got, f"entry {i} has no record but holds {len(got)} pages"
                continue
            assert len(got) == len(x.ids) // PAGE, f"entry {i}: {len(got)} pages for {len(x.ids)} ids"
            for k, page in enumerate(got):
                row = self.content.get(page, {})
                assert all(row.get(o) == x.ids[k * PAGE + o] for o in range(PAGE)), f"entry {i} page {k} changed"
            # its ring and drafter context end at its last id
            m = len(x.ids)
            for p in range(max(0, m - 511), m):
                assert kr[(SLOTS + i) * RING + p % RING] == tag(p, x.ids[p]), f"entry {i} ring at {p}"
            dr = self.eng.drafter
            for p in range(max(0, m - DWIN), m):
                assert dk[SLOTS + i][p % dr.ring] == tag(p, x.ids[p]), f"entry {i} drafter context at {p}"

    def check_reads(self, r: E.Request, upto: int) -> None:
        """r's pages hold its own ids at positions [0, upto)."""
        c = self.c
        for p in range(upto):
            page = c.owned[r.slot][p // PAGE]
            assert self.content.get(page, {}).get(p % PAGE) == self.ids(r, p), \
                f"slot {r.slot} position {p}: page {page} holds another id"

    def step(self) -> bool:
        eng, c = self.eng, self.c
        with eng.cv:
            plan = eng._plan()
        if plan is None:
            return False
        self.steps += 1
        back = E.Plan.decode(*plan.encode())
        for f in ("op", "kind", "releases", "reserves", "seqs", "ids", "evicts", "restores", "saves", "ctx"):
            a, b = getattr(plan, f), getattr(back, f)
            if isinstance(a, list):
                a, b = [tuple(v) if isinstance(v, (list, tuple)) else v for v in a], \
                       [tuple(v) if isinstance(v, (list, tuple)) else v for v in b]
            assert a == b, f"plan field {f} does not round-trip: {a} vs {b}"
        # _execute's order
        dr = eng.drafter
        for slot in plan.releases:
            c.release(slot)
        for e in plan.evicts:
            c.release(c.seqs + e)
        self.evicts += len(plan.evicts)
        for e, slot in plan.restores:
            c.restore(e, slot)
            dr.copy_context(c.seqs + e, slot)
            self.hits += 1
        for slot, n in plan.reserves:
            c.reserve(slot, n)
        by_slot = {r.slot: r for r in eng.running}
        prev = [x for x in plan.ctx if x[4] == E.CTX_PREV]
        if prev:
            self.ctx(prev)
        if plan.op == E.STEP:
            kr = c.k_ring[0, :, 0, 0].tolist()
            dk = dr.ck[0, :, :, 0, 0].tolist()
            for slot, p0, n in plan.seqs:
                r = by_slot[slot]
                assert all(c.owned[slot][k] >= 0 for k in range(-(-r.need // PAGE)))
                if plan.kind == E.PREFILL:
                    # a chunk reads the window before it from the ring: after a restore too
                    for p in range(max(0, p0 - 511), p0):
                        assert kr[slot * RING + p % RING] == tag(p, self.ids(r, p)), \
                            f"slot {slot} chunk at {p0}: ring at {p} is not its id"
                    # the rows a fresh prefill computes: inside one chunk of the grid, MIN_PIECE rows or more or the
                    # fresh prefill's own last chunk; unscored pieces start on a page, scored ones on the grid
                    g = (len(r.prompt) - 1) // CHUNK * CHUNK
                    assert p0 // CHUNK == (p0 + n - 1) // CHUNK, (p0, n, len(r.prompt))
                    assert n >= E.MIN_PIECE or (p0 == g and p0 + n == len(r.prompt)), (p0, n, len(r.prompt))
                    if r.score:
                        assert eng.test_pieces or p0 % CHUNK == 0, (p0, n, len(r.prompt))
                    else:
                        assert p0 % PAGE == 0, (p0, n, len(r.prompt))
                    self.pieces += 1
                    self.short += n < CHUNK and p0 + n != len(r.prompt)
                    self.check_reads(r, p0)
                else:
                    if len(r.out) == 1:            # its first decode: the drafter's window of its prompt is there
                        for p in range(max(0, p0 - DWIN), p0):
                            assert dk[slot][p % dr.ring] == tag(p, self.ids(r, p)), \
                                f"slot {slot} first decode at {p0}: drafter context at {p} is not its id"
                    self.check_reads(r, p0)
                self.write(r, p0, n)
            this = [x for x in plan.ctx if x[4] == E.CTX_THIS]
            if this:
                self.ctx(this)
            for slot, e, m in plan.saves:
                c.release(c.seqs + e)
                c.save(slot, e, m)
                dr.copy_context(slot, c.seqs + e)
                self.saves += 1
        self.check_pages()
        if plan.op != E.STEP:
            return True
        if plan.kind == E.PREFILL:
            tokens = [self.rng.randrange(1000, 2000)] if plan.samples else None
            scores = [] if plan.score else None
            eng.debt = 1.0
        else:
            tokens = [self.rng.randrange(1000, 2000) for _ in plan.samples]
            scores = None
            eng.debt -= 0.4
        finishing = list(eng.running)
        eng._deliver(plan, (tokens, scores, None))
        self.done += [r for r in finishing if r.done]
        return True

    def run(self, limit: int = 100000) -> None:
        for _ in range(limit):
            if not self.step() and not self.eng.waiting:
                return
        raise AssertionError("the requests did not finish")


def words(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(10, 900) for _ in range(n)]


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = random.Random(1)

    # 1. four prompts, then the same four again twice (rows.py's decode4): the first re-send resumes at its last chunk
    # grid point and keeps an entry at its last page boundary MIN_PIECE rows before its end; the second resumes there
    s = Sim(pages=200, seed=2)
    ps = [words(rng, 3 * CHUNK + 700 + 17 * i) for i in range(4)]
    for p in ps:
        s.submit(p, 16)
    s.run()
    again = [s.submit(p, 40) for p in ps]
    s.run()
    third = [s.submit(p, 40) for p in ps]
    s.run()
    for r, q, p in zip(again, third, ps):
        grid, fine = (len(p) - 1) // CHUNK * CHUNK, (len(p) - E.MIN_PIECE) // PAGE * PAGE
        assert r.cached == grid == r.stats["cached"] and q.cached == fine == q.stats["cached"], (r.cached, q.cached)
        assert fine > grid and len(p) - fine < 2 * PAGE, (len(p), grid, fine)
    assert s.hits == 8 and s.saves == 8, (s.hits, s.saves)
    print(f"1 rows.py's decode4: hits at {[r.cached for r in again]}, then {[q.cached for q in third]}, "
          f"{s.saves} saves, {s.evicts} evictions")

    # 2. an agent's growing prompt: each turn resumes from the last turn's entry and keeps its own (from the second
    # turn on, its last page boundary MIN_PIECE rows before its end when that is FINE past its grid point)
    s = Sim(pages=200, seed=3)
    p = words(rng, 2 * CHUNK + 100)
    cached = []
    for turn in range(5):
        r = s.submit(p, 30)
        s.run()
        cached.append(r.cached)
        p = p + r.out + words(rng, 700 + 300 * turn)
    assert cached == [0, 2048, 2560, 3584, 5120], cached
    print(f"2 agent turns: cached {cached}, {s.saves} saves, {s.evicts} evictions")

    # 3. salts and scored prompts: no hit across salts (None is a salt of its own); a scored prompt neither takes
    # nor keeps an entry
    s = Sim(pages=200, seed=4)
    p = words(rng, 2 * CHUNK + 50)
    for salt in ("a", "b", None):
        s.submit(p, 8, salt=salt)
        s.run()
    assert s.hits == 0 and s.saves == 3, (s.hits, s.saves)
    r = s.submit(p, 8, salt="a")
    s.run()
    assert r.cached == 2 * CHUNK and s.hits == 1, (r.cached, s.hits)
    q = s.submit(p, 8, salt="b", score=4)
    s.run()
    assert q.cached == 0 and s.hits == 1 and s.saves == 3, (q.cached, s.hits, s.saves)
    print("3 salts kept apart; a scored prompt neither takes nor keeps an entry")

    # 4. two identical prompts at once: one entry
    s = Sim(pages=200, seed=5)
    p = words(rng, 3 * CHUNK + 10)
    s.submit(p, 8)
    s.submit(p, 8)
    s.run()
    assert s.saves == 1 and sum(x is not None for x in s.eng.ent) == 1, s.saves
    print("4 two identical prompts at once: one entry")

    # 5. a pool short of pages: a long prompt evicts entries (least recently used first) and fits
    s = Sim(pages=64, seed=6)
    for i in range(3):
        s.submit(words(rng, 4 * CHUNK + 5 * i), 4)
        s.run()
    held = sum(len(s.c.owned[SLOTS + i]) for i in range(ENTRIES))
    big = s.submit(words(rng, 60 * PAGE - 40), 30)
    s.run()
    assert big.done and len(big.out) == 30 and s.evicts >= 2, (big.done, len(big.out), s.evicts)
    print(f"5 a {len(big.prompt)}-id prompt in a {64 * PAGE}-position pool: {s.evicts} evictions "
          f"(entries held {held} pages)")

    # 6. random traffic: shared starts, re-sends, growing prompts, salts, scored prompts, cancels
    s = Sim(pages=96, seed=7)
    bases = [words(rng, rng.randrange(CHUNK, 5 * CHUNK)) for _ in range(5)]
    live: list[E.Request] = []
    for _ in range(steps):
        if rng.random() < 0.08 and len(s.eng.waiting) < 6:
            b = rng.choice(bases)
            cut = rng.randrange(1, len(b))
            p = b[:cut] + words(rng, rng.randrange(0, 2 * CHUNK)) if rng.random() < 0.5 else list(b)
            if len(p) + 64 < s.eng.limit:
                live.append(s.submit(p, rng.randrange(1, 64), salt=rng.choice([None, None, "x"]),
                                     score=4 if rng.random() < 0.05 else 0))
        if rng.random() < 0.01 and s.eng.running:
            with s.eng.cv:
                rng.choice(s.eng.running).cancel = True
        s.step()
    s.run()
    for i in range(ENTRIES):                                  # every page comes back once the entries go
        if s.eng.ent[i] is not None:
            s.c.release(SLOTS + i)
            s.eng.ent[i] = None
    assert len(s.c.free) == s.c.pages and not any(s.c.refs), "pages lost"
    print(f"6 random traffic: {len(live)} requests, {s.steps} steps, {s.hits} hits, {s.saves} saves, "
          f"{s.evicts} evictions, {s.pieces} prompt pieces ({s.short} ending off the grid); every page back")

    # 7. the "pieces" knob (the row-invariance check): scored prompts cut at its multiples too, every piece MIN_PIECE
    # rows or more, unscored ones as before
    s = Sim(pages=96, seed=8)
    s.eng.test_pieces = 300
    for n in (2 * CHUNK + 77, CHUNK + 500, 900, 3 * CHUNK):
        s.submit(words(rng, n), 1, score=4)
        s.submit(words(rng, n), 8)
    s.run()
    assert len(s.done) == 8 and s.short >= 8, (len(s.done), s.short)
    print(f"7 the pieces knob: {s.pieces} prompt pieces, {s.short} ending off the grid")

    # 8. four prompts, then the same four re-sent one at a time (each admitted after the one before kept its longer
    # entry): each keeps it in the place of the entry it started from, so every re-send finds its own
    s = Sim(pages=200, seed=9)
    ps = [words(rng, 3 * CHUNK + 700 + 17 * i) for i in range(4)]
    for p in ps:
        s.submit(p, 16)
    s.run()
    again = []
    for p in ps:
        again.append(s.submit(p, 40))
        for _ in range(3):
            s.step()
    s.run()
    grids = [(len(p) - 1) // CHUNK * CHUNK for p in ps]
    assert [r.cached for r in again] == grids and s.saves == 8, ([r.cached for r in again], grids, s.saves)
    print(f"8 re-sent one at a time: hits at {[r.cached for r in again]}, {s.saves} saves, {s.evicts} evictions")
    print("PC_CHECK_PASS")


if __name__ == "__main__":
    main()
