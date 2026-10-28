"""One rank's KV stores: the full layers' keys and values in a pool of fixed-size pages (fp8 or bf16) that sequences
take pages from as they grow; each window layer's last ``ring`` positions per sequence slot (bf16); and one prompt
chunk's window buffer (the 511 positions before the chunk, then the chunk), reused by every window layer in turn.

The prompt cache's entries are holders after the sequence slots (``seqs + e``): an entry keeps a prompt's pages up to a
chunk boundary (held, not copied: a page goes back to the pool when its last holder lets it go) and a copy of the slot's
rings at that boundary, so a later prompt with the same start takes them and fills only the rest."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.families.beam.config import Config


class Cache:
    def __init__(self, cfg: Config, world: int, *, pages: int, page: int = 256, seqs: int, chunk: int,
                 ring: int = 528, full_dtype: torch.dtype = torch.float8_e4m3fn, device="cuda",
                 entries: int = 0) -> None:
        if page % 64:
            raise ValueError("pages hold whole 64-key blocks")
        if ring < cfg.window + 16:
            raise ValueError(f"a ring needs the window plus a verify window's rows, at least {cfg.window + 16}")
        self.cfg, self.page, self.ring, self.seqs, self.chunk = cfg, page, ring, seqs, chunk
        self.pages, self.entries = pages, entries
        hkv = cfg.kv_heads // world
        self.hkv = hkv
        self.full = {L: i for i, L in enumerate(cfg.full)}
        win = [L for L in range(cfg.layers) if cfg.sliding(L)]
        self.win = {L: i for i, L in enumerate(win)}
        self.k_full = torch.zeros((len(cfg.full), pages * page, hkv, 128), dtype=full_dtype, device=device)
        self.v_full = torch.zeros_like(self.k_full)
        self.k_ring = torch.zeros((len(win), (seqs + entries) * ring, hkv, 128), dtype=torch.bfloat16, device=device)
        self.v_ring = torch.zeros_like(self.k_ring)
        back = cfg.window - 1
        self.back = back
        self.k_buf = torch.zeros((back + chunk, hkv, 128), dtype=torch.bfloat16, device=device)
        self.v_buf = torch.zeros_like(self.k_buf)
        self.max_pages = pages
        self.table = torch.zeros((seqs, pages), dtype=torch.int32, device=device)
        self.owned: list[list[int]] = [[] for _ in range(seqs + entries)]    # a slot's or entry's pages in order
        self.refs = [0] * pages                                                 # a page's holders
        self.free = list(range(pages - 1, -1, -1))

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.k_full, self.v_full, self.k_ring, self.v_ring,
                                                           self.k_buf, self.v_buf, self.table))

    def capacity(self) -> int:
        """Positions the pool holds in all (one sequence's most)."""
        return self.pages * self.page

    def reserve(self, slot: int, length: int) -> None:
        """Pages for positions [0, length) of the sequence in ``slot`` (raises when the pool runs out)."""
        need = -(-length // self.page)
        have = self.owned[slot]
        if need > len(have):
            if need - len(have) > len(self.free):
                raise MemoryError(f"the KV pool has {len(self.free)} free pages; slot {slot} needs {need - len(have)}")
            new = [self.free.pop() for _ in range(need - len(have))]
            for p in new:
                self.refs[p] = 1
            self.table[slot, len(have):need] = torch.tensor(new, dtype=torch.int32)
            have.extend(new)

    def release(self, holder: int) -> None:
        """A slot (or entry: ``seqs + e``) lets its pages go; a page no one else holds goes back to the pool."""
        for p in reversed(self.owned[holder]):
            self.refs[p] -= 1
            if not self.refs[p]:
                self.free.append(p)
        self.owned[holder] = []

    def freed(self, holders) -> int:
        """Pages releasing all of ``holders`` would give back to the pool."""
        drop: dict[int, int] = {}
        for h in holders:
            for p in self.owned[h]:
                drop[p] = drop.get(p, 0) + 1
        return sum(1 for p, n in drop.items() if self.refs[p] == n)

    def save(self, slot: int, entry: int, length: int) -> None:
        """Entry ``entry`` (holding nothing) keeps positions [0, length) of ``slot``, whose last forward ended at
        ``length`` (a whole number of pages): the pages, held, and the rings, copied."""
        if length % self.page:
            raise ValueError(f"an entry ends on a page boundary, not at {length}")
        e = self.seqs + entry
        self.owned[e] = self.owned[slot][:length // self.page]
        for p in self.owned[e]:
            self.refs[p] += 1
        self._copy_rings(slot, e)

    def restore(self, entry: int, slot: int) -> None:
        """Slot ``slot`` (holding nothing) starts from entry ``entry``: its pages, held, and its rings, copied."""
        e = self.seqs + entry
        pages = list(self.owned[e])
        for p in pages:
            self.refs[p] += 1
        self.owned[slot] = pages
        if pages:
            self.table[slot, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
        self._copy_rings(e, slot)

    def _copy_rings(self, src: int, dst: int) -> None:
        a, b = src * self.ring, dst * self.ring
        self.k_ring[:, b:b + self.ring].copy_(self.k_ring[:, a:a + self.ring])
        self.v_ring[:, b:b + self.ring].copy_(self.v_ring[:, a:a + self.ring])

    def slots_full(self, slot: int, pos: torch.Tensor) -> torch.Tensor:
        """Pool slots of positions ``pos`` (host int64) in ``slot``'s pages."""
        pages = torch.tensor(self.owned[slot], dtype=torch.int64)
        return pages[pos // self.page] * self.page + pos % self.page

    def slots_ring(self, slot: int, pos: torch.Tensor) -> torch.Tensor:
        return slot * self.ring + pos % self.ring


@dataclass
class Batch:
    """One forward's rows: S sequences, each a run of consecutive positions (a prompt chunk, a decode row or a verify
    window). ``chunk``: one sequence's prompt chunk (window layers attend through the window buffer)."""

    pos: torch.Tensor             # [R] int32, device
    row0: torch.Tensor            # [S + 1] int32, device
    rows: int
    max_rows: int
    span: int                     # the widest full-layer key range of a row
    table: torch.Tensor           # [S, pages] int32 page tables of the batch's slots
    full_dst: torch.Tensor        # [R] int64 pool slots of the rows
    ring_start: torch.Tensor      # [S] int32 the slots' first ring slot
    ring_dst: torch.Tensor        # [R] int64 ring slots of the rows (decode / verify)
    chunk: bool = False
    p0: int = 0                   # a chunk's first position
    slot: int = 0                 # a chunk's sequence slot
    buf_from: torch.Tensor | None = None   # a chunk: ring slots of the window before it, and their buffer rows
    buf_to: torch.Tensor | None = None
    keep_rows: torch.Tensor | None = None  # a chunk: its rows that stay in the ring (the last ``ring``) ...
    keep_dst: torch.Tensor | None = None   # ... and their ring slots


def batch(cache: Cache, seqs: list[tuple[int, int, int]], device, *, chunk: bool = False) -> Batch:
    """seqs: (slot, first position, rows) per sequence; pages must already be reserved."""
    pos, row0, fdst, rdst = [], [0], [], []
    for slot, p0, n in seqs:
        p = torch.arange(p0, p0 + n, dtype=torch.int64)
        pos.append(p)
        row0.append(row0[-1] + n)
        fdst.append(cache.slots_full(slot, p))
        rdst.append(cache.slots_ring(slot, p))
    pos_h = torch.cat(pos)
    slots = [s for s, _, _ in seqs]
    b = Batch(pos=pos_h.to(torch.int32).to(device), row0=torch.tensor(row0, dtype=torch.int32, device=device),
              rows=row0[-1], max_rows=max(n for _, _, n in seqs), span=int(pos_h.max()) + 1,
              table=cache.table[slots] if len(set(slots)) == len(slots) else cache.table[slots],
              full_dst=torch.cat(fdst).to(device),
              ring_start=torch.tensor([s * cache.ring for s in slots], dtype=torch.int32, device=device),
              ring_dst=torch.cat(rdst).to(device), chunk=chunk)
    if chunk:
        if len(seqs) != 1:
            raise ValueError("a prompt chunk is one sequence's")
        slot, p0, n = seqs[0]
        if n > cache.chunk:
            raise ValueError(f"a chunk of {n} rows exceeds the window buffer's {cache.chunk}")
        b.p0, b.slot = p0, slot
        before = torch.arange(max(0, p0 - cache.back), p0, dtype=torch.int64)
        b.buf_from = cache.slots_ring(slot, before).to(device)
        b.buf_to = (before - (p0 - cache.back)).to(device)
        keep = torch.arange(max(0, n - cache.ring), n, dtype=torch.int64)
        b.keep_rows = (keep + cache.back).to(device)
        b.keep_dst = cache.slots_ring(slot, keep + p0).to(device)
    return b
