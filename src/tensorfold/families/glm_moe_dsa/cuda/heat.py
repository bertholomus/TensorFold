"""Pacing long prompts by GPU temperature (opt-in): TF_GLM_HEAT_PACE=HI:LO[:TOP] (``1``: 78:66; TOP: HI + 4).

At each prompt chunk's boundary every rank reads its GPU's temperature and the ranks exchange them (one tiny
all-gather, so all decide alike). When the hottest GPU is at HI C or more, or the last chunk's rise would take a GPU to
TOP C or more in the next one (deep chunks run longer), every rank waits there, its GPU idle, until the hottest is at
LO C or less; no rank waits inside a collective of the forward. A GB10 reads ~11 C cooler a second after its load stops
(the die drops to its heat sink), so LO sits more than that under HI: waiting for LO waits for the heat sink, and the
next chunk starts ~11 C over it. The chunks, kernels and outputs are the same, only later: a long prompt's wall time
grows by the pauses (``paused`` / ``pauses``, for reports). Off unless set."""

from __future__ import annotations

import ctypes
import os
import sys
import time


def _parse(v: str | None) -> tuple[int, int, int] | None:
    if not v or v == "0":
        return None
    if v == "1":
        v = "78:66"
    parts = [int(x) for x in v.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(f"TF_GLM_HEAT_PACE={v}: want HI:LO or HI:LO:TOP (degrees C)")
    hi, lo = parts[:2]
    top = parts[2] if len(parts) == 3 else hi + 4
    if not 0 < lo < hi <= top:
        raise ValueError(f"TF_GLM_HEAT_PACE={v}: want 0 < LO < HI <= TOP (degrees C)")
    return hi, lo, top


PACE = _parse(os.environ.get("TF_GLM_HEAT_PACE"))
paused = 0.0                                          # seconds the prompts waited for the GPUs to cool
pauses: list[tuple[float, float]] = []                # (start, end) of each wait, wall-clock seconds
hottest = 0                                           # the hottest GPU seen at a chunk boundary
_read = None
_last: list[int] | None = None                        # every GPU's temperature at the last boundary, under load
_rise: list[int] | None = None                        # every GPU's rise over the last chunk (two loaded readings)
_idle: list[int] | None = None                        # every GPU's temperature when the last pause ended
_cold: list[int] | None = None                        # every GPU's rise over a chunk run from a pause (from idle)


def _reader():
    """The GPU's temperature (C) through NVML: the only GPU, else the one at this process's CUDA device's PCI address."""

    import torch

    lib = ctypes.CDLL("libnvidia-ml.so.1")
    if lib.nvmlInit_v2() != 0:
        raise RuntimeError("TF_GLM_HEAT_PACE: NVML did not start")
    h = ctypes.c_void_p()
    n = ctypes.c_uint()
    lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
    if n.value == 1:
        rc = lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(h))
    else:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        bus = f"{p.pci_domain_id:08X}:{p.pci_bus_id:02X}:{p.pci_device_id:02X}.0".encode()
        rc = lib.nvmlDeviceGetHandleByPciBusId_v2(ctypes.c_char_p(bus), ctypes.byref(h))
    if rc != 0:
        raise RuntimeError(f"TF_GLM_HEAT_PACE: no NVML handle for this GPU ({rc})")

    def read() -> int:
        t = ctypes.c_uint()
        if lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(t)) != 0:      # NVML_TEMPERATURE_GPU
            raise RuntimeError("TF_GLM_HEAT_PACE: no GPU temperature")
        return int(t.value)

    return read


def temperature() -> int:
    global _read
    if _read is None:
        _read = _reader()
    return _read()


def _temps(comm, world: int) -> list[int]:
    """Every rank's GPU temperature, in rank order, the same list on every rank."""

    t = temperature()
    if comm is None or world == 1:
        return [t]
    import torch

    send = torch.full((1,), float(t), dtype=torch.float32, device="cuda")
    recv = torch.empty((world,), dtype=torch.float32, device="cuda")
    comm.all_gather(send, recv)
    return [int(x) for x in recv.tolist()]


def pace(comm=None, world: int = 1, rank: int = 0, *, first: bool = False) -> None:
    """At a prompt chunk's boundary, on every rank alike (``first``: the prompt's first chunk, no rise known yet).

    Two rises are kept per GPU: over a chunk that followed another (``_rise``, loaded to loaded) and over a chunk that
    followed a pause (``_cold``, idle to loaded: the ~11 C jump and the chunk's own heat). A pause lasts until the
    hottest GPU is at LO and, once ``_cold`` is known, until every GPU's idle reading plus its ``_cold`` is under TOP:
    deeper chunks run longer and heat more, so the pauses grow with them."""

    global paused, hottest, _last, _rise, _idle, _cold
    if PACE is None:
        return
    import torch

    hi, lo, top = PACE
    torch.cuda.synchronize()                           # the chunks so far have run (their heat is in)
    temps = _temps(comm, world)
    hottest = max(hottest, *temps)
    if first:                                          # the first reading may follow idle time: no rise from it
        _last = _rise = _idle = None
    elif _idle is not None:                            # this chunk ran from a pause
        _cold = [max(0, t - i) for t, i in zip(temps, _idle)]
    elif _last is not None:                            # a reading under load after one before it
        _rise = [max(0, t - p) for t, p in zip(temps, _last)]
    _last, _idle = (None if first else temps), None
    ahead = max(t + r for t, r in zip(temps, _rise)) if _rise is not None else max(temps)
    if max(temps) < hi and ahead < top:
        return

    def cool(ts: list[int]) -> bool:
        return max(ts) <= lo and (_cold is None or max(t + c for t, c in zip(ts, _cold)) < top)

    w0, t0, at = time.time(), time.monotonic(), temps
    while not cool(temps):
        time.sleep(1.0)
        temps = _temps(comm, world)
    dt = time.monotonic() - t0
    _last, _idle = None, temps
    paused += dt
    pauses.append((w0, w0 + dt))
    if rank == 0:
        print(f"[tensorfold] heat pace {time.strftime('%H:%M:%S', time.gmtime(w0))}Z: GPUs {at} C (rise {_rise}, from "
              f"idle {_cold}), waited {dt:.0f} s to {temps} C ({paused:.0f} s in all)", file=sys.stderr, flush=True)
