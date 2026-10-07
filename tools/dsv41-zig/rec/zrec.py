"""Record what the served DeepSeek-V4.1 engine runs, for the Zig port (pipeline/DESIGN-zig-port.md, milestone M1).

Every Triton launch (TensorFold 1.0's Recorder from tools/zig/triton_aot_manifest.py), every call into our CUDA
extensions and every ATen op, each under the phase the engine was in: a prompt chunk, a graph capture by rows and
context bucket, the drafter, sampling, a round's host work. A CUDA graph replay launches nothing from Python, so a
decode window's kernels are the ones its capture recorded.

TF_ZREC_DIR=<dir> turns it on (sitecustomize.py). It writes there when <dir>/DUMP appears (polled every 2 s) and at
exit: launches.json (Triton kernels, phases, grids, sites, and up to TF_ZREC_KEEP detailed launches a distinct shape;
extension calls are in its phases and log too), modules.json (the extension modules and their .so files), aten.json
(ATen ops by phase and signature, with call sites) and meta.json (versions, device, environment).

With CUDA_INJECTION64_PATH=.../ztrace.so the phases also go to that launch tracer. With TF_ZREC_LAYERS set (a list of
layer numbers, possibly empty), while <dir>/LAYERS exists every call of the LAYER_POINTS methods writes a line a tensor
it takes or returns to layers.jsonl (digest, shape, dtype, first and last row's first bytes), and the listed layers'
tensors (and Model.forward's, a round's inputs, logits and taps) are saved whole under <dir>/layers/: the Zig port's
layer gate (M3, and M4's decode rounds when they run eager: TF_DS_GRAPHS=0). A list of ints (a round's token ids,
positions, slots and extents) is taken as an int64 tensor.
"""

from __future__ import annotations

import atexit
from collections import Counter, defaultdict
import functools
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

OUT = Path(os.environ.get("TF_ZREC_DIR") or "/tmp/zrec")
KEEP = int(os.environ.get("TF_ZREC_KEEP") or 3)          # detailed entries kept a distinct launch or op signature
SIGS = int(os.environ.get("TF_ZREC_SIGS") or 256)        # distinct signatures kept an (phase, op) before folding
OPS_ON = os.environ.get("TF_ZREC_ATEN", "1") != "0"
FAMILY = "tensorfold.families.deepseek_v41"

_tls = threading.local()
_ztrace = None      # ztrace.so (CUDA_INJECTION64_PATH) when it traces this process: phases go to it too
_lock = threading.Lock()
_rec = None
_ops = None
_modules: dict[str, str | None] = {}


def _trace_phase(p: str) -> None:
    if _ztrace is not None:
        _ztrace.ztrace_phase(p.encode())


def phase() -> str:
    """This thread's innermost phase ("other" outside every patched method)."""

    st = getattr(_tls, "stack", None)
    return st[-1] if st else "other"


def _recorder_class():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from triton_aot_manifest import Recorder

    class Rec(Recorder):
        """The upstream Recorder with this thread's phase, always detailed, and the detail capped a distinct shape."""

        phase = property(lambda self: phase(), lambda self, v: None)
        detail = property(lambda self: True, lambda self, v: None)

        def __init__(self) -> None:
            super().__init__()
            self.seen: Counter = Counter()

        def _entry(self, row: dict) -> None:
            key = json.dumps([phase(), row.get("kind"), row.get("name"), row.get("hash"), row.get("grid"),
                              row.get("args"), row.get("kwargs")], sort_keys=True, default=str)
            with _lock:
                self.seen[key] += 1
                if self.seen[key] <= KEEP:
                    super()._entry(row)

    return Rec


def _meta(x: Any) -> Any:
    """An op argument as the port needs it: a tensor's dtype, shape, strides, device and whether it is a view at an
    offset; scalars by value."""

    import torch

    if isinstance(x, torch.Tensor):
        return ["T", str(x.dtype).replace("torch.", ""), list(x.shape), list(x.stride()), x.device.type,
                int(x.storage_offset())]
    if isinstance(x, (list, tuple)):
        return [_meta(y) for y in x]
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    if isinstance(x, float):
        return {"f": x}
    return str(x)


class Ops:
    """ATen ops by (phase, op, signature): counts, and the call sites of the first few of each."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.sites: dict[tuple, list] = {}
        self.folded: Counter = Counter()
        self.sigs: dict[tuple, int] = defaultdict(int)

    def add(self, func, args, kwargs) -> None:
        from triton_aot_manifest import call_site

        p, name = phase(), str(func)
        sig = json.dumps([_meta(list(args)), _meta(dict(kwargs))], default=str)
        key = (p, name, sig)
        with _lock:
            if key not in self.counts and self.sigs[(p, name)] >= SIGS:
                self.folded[(p, name)] += 1
                return
            if key not in self.counts:
                self.sigs[(p, name)] += 1
            self.counts[key] += 1
            n = self.counts[key]
        if n <= KEEP:
            site = call_site()
            with _lock:
                self.sites.setdefault(key, []).append(site)

    def dump(self) -> dict:
        with _lock:
            rows = [{"phase": p, "op": op, "sig": json.loads(sig), "count": n, "sites": self.sites.get((p, op, sig), [])}
                    for (p, op, sig), n in sorted(self.counts.items())]
            folded = [{"phase": p, "op": op, "count": n} for (p, op), n in sorted(self.folded.items())]
        return {"ops": rows, "folded": folded}


def _mode_class():
    from torch.utils._python_dispatch import TorchDispatchMode

    class OpLog(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            try:
                _ops.add(func, args, kwargs)
            except Exception as e:  # never break the op
                print(f"zrec: op not logged ({e!r})", file=sys.stderr, flush=True)
            return func(*args, **kwargs)

    return OpLog


_OpLog = None


def scoped(name, f):
    """`f` run under phase `name` (a string, or a function of self); the outermost one on a thread logs ATen ops."""

    @functools.wraps(f)
    def w(*a, **k):
        try:
            p = name(a[0]) if callable(name) else name
        except Exception:
            p = getattr(name, "__name__", "phase")
        st = _tls.__dict__.setdefault("stack", [])
        st.append(p)
        _trace_phase(p)
        mode = None
        if len(st) == 1 and OPS_ON and _OpLog is not None:
            mode = _OpLog()
            mode.__enter__()
        try:
            return f(*a, **k)
        finally:
            if mode is not None:
                mode.__exit__(None, None, None)
            st.pop()
            _trace_phase(st[-1] if st else "other")

    w._zrec = True
    return w


def _patch(module: str, owner: str | None, attr: str, name) -> None:
    import importlib

    try:
        m = importlib.import_module(module)
        target = getattr(m, owner) if owner else m
        f = getattr(target, attr)
        if getattr(f, "_zrec", False):
            return
        setattr(target, attr, scoped(name, f))
    except Exception as e:
        print(f"zrec: no phase for {module}.{owner}.{attr} ({e!r})", file=sys.stderr, flush=True)


def _rows_bucket(prefix: str):
    return lambda s: f"{prefix}:r{getattr(s, 'R', '?')}:b{getattr(s, 'bucket', '?')}"


PHASES = [
    # (module under FAMILY, class or None for a module function, method, phase)
    ("cuda.engine", "DsEngine", "__init__", "init"),
    ("cuda.weights", None, "load", "load"),
    ("cuda.weights", None, "attach_vl_bias", "load"),
    ("cuda.engine", "DsEngine", "warm", "warm"),
    ("cuda.engine", "DsEngine", "vision_warm", "vision_warm"),
    ("cuda.engine", "DsEngine", "prefill_steps", "prompt"),
    ("cuda.engine", "DsEngine", "_serial", "serial"),
    ("cuda.engine", "DsEngine", "_spec", "spec"),
    ("cuda.engine", "DsEngine", "_sample", "sample"),
    ("cuda.engine", "DsEngine", "follow", "follow"),
    ("cuda.graph", "GraphRunner", "warm", "graph_warm"),
    ("cuda.graph", "StaticDecoder", "capture", _rows_bucket("static")),
    ("cuda.graph", "StaticDecoder", "run", "static_run"),
    ("cuda.rounds", "RoundDecoder", "capture", _rows_bucket("round")),
    ("cuda.rounds", "RoundDecoder", "run", "round_run"),
    ("cuda.rounds", "RoundRunner", "forward", "round_forward"),
    ("cuda.dspark", "Drafter", "absorb", "absorb"),
    ("cuda.dspark", "Drafter", "absorb_many", "absorb"),
    ("cuda.dspark", "Drafter", "draft", "draft_eager"),
    ("cuda.dspark", "DraftGraph", "capture", "draft_capture"),
    ("cuda.dspark", "DraftGraph", "run", "draft_run"),
    ("cuda.dspark", "BatchDraftGraph", "capture", lambda s: f"draft_batch_capture:s{getattr(s, 'N', '?')}"),
    ("cuda.dspark", "BatchDraftGraph", "run", "draft_batch_run"),
    ("cuda.multi", "MultiDecoder", "warm", "multi_warm"),
    ("cuda.multi", "MultiDecoder", "_admit", "admit"),
    ("cuda.multi", "MultiDecoder", "_round", "round_host"),
    ("cuda.multi", "MultiDecoder", "_drafts", "drafts"),
    ("cuda.multi", "MultiDecoder", "_absorb_eager", "absorb"),
    ("cuda.multi", "MultiDecoder", "_copy_rows", "kept_copy"),
    ("cuda.multi", "MultiDecoder", "_snapshot", "kept_snapshot"),
    ("cuda.multi", "MultiDecoder", "_restore", "kept_restore"),
    ("cuda.multi", "MultiDecoder", "follow", "follow"),
    ("cuda.model", "Model", "forward", "forward"),
    ("cuda.model", "Engram", "rows", "engram_rows"),
    ("cuda.vision", "Tower", "encode", "vision"),
    ("cuda.vision", "Tower", "span_rows", "vision"),
]


# -- per-layer fixtures -------------------------------------------------------------------------------------------

LAYER_POINTS = [
    # (module under FAMILY, class, method[, positional arguments left out]): the prompt path's layer steps and the
    # gather between them; a decode round's inputs, logits and taps and its attention sublayers (eager rounds only)
    ("cuda.model", "Model", "forward"),
    ("cuda.model", "Model", "attention_k"),
    ("cuda.model", "Model", "moe"),
    ("cuda.model", "Comm", "gather"),
    ("cuda.rounds", "RoundRunner", "forward"),
    ("cuda.rounds", "RoundDecoder", "run"),
    ("cuda.rounds", "RoundDecoder", "_attention2", (4, 5)),     # cos, sin: whole RoPE tables
    # the drafter (M5): its batched pass (the drafts), its stages, each block's attention (with the block's ring
    # plane: the absorbed keys), the Markov loop, its absorbs of the target's taps
    ("cuda.dspark", "BatchDraftGraph", "run"),
    ("cuda.dspark", "BatchDraftGraph", "_stages_fused"),
    ("cuda.dspark", "BatchDraftGraph", "_attention"),
    ("cuda.markov", "Markov", "steps"),
    ("cuda.dspark", "Drafter", "absorb", (1, 2)),                # dc, sc: caches
    ("cuda.dspark", "Drafter", "absorb_many", (1, 2)),           # dpool, sc
    ("cuda.dspark", "Drafter", "draft", (1, 2)),                 # dc, sc
]
# saved whole whatever the layer (TF_ZREC_WHOLE=<where>,... overrides: a token recording keeps only the prompts)
WHOLE = tuple(x for x in os.environ["TF_ZREC_WHOLE"].split(",") if x) if "TF_ZREC_WHOLE" in os.environ else (
    "Model.forward", "RoundRunner.forward", "RoundDecoder.run", "BatchDraftGraph.run", "BatchDraftGraph._stages_fused",
    "Markov.steps", "Drafter.absorb", "Drafter.absorb_many", "Drafter.draft")
# TF_ZREC_ONLY=<where>,...: only these points are written (a token recording: the prompts ids and the rounds inputs)
ONLY = {x for x in os.environ.get("TF_ZREC_ONLY", "").split(",") if x}
LAYERS_ON = "TF_ZREC_LAYERS" in os.environ
LAYERS_FULL = {int(x) for x in os.environ.get("TF_ZREC_LAYERS", "").split(",") if x.strip().isdigit()}
FULL_MAX = 256 << 20     # bytes: a bigger tensor is never saved whole
DIGEST_MAX = 2 << 30     # bytes: a bigger one gets no digest either
_lay_lines: list[dict] = []
_lay = {"armed": False, "at": 0.0, "call": 0}


def _armed() -> bool:
    now = time.monotonic()
    if now - _lay["at"] > 0.2:
        _lay["at"] = now
        _lay["armed"] = (OUT / "LAYERS").exists()
    return _lay["armed"]


def _tensors(x, path: str, depth: int = 0):
    """(path, tensor) for x when it is a tensor, an int list (as int64) or an int (at depth > 0), or for each such item
    in x when it is a list or tuple (two levels deep: a list of windows, of (slot, taps, position) items)."""

    import torch

    if isinstance(x, torch.Tensor):
        yield path, x
    elif isinstance(x, int) and not isinstance(x, bool) and depth > 0:
        yield path, torch.tensor([x], dtype=torch.int64)
    elif isinstance(x, (list, tuple)) and x and all(isinstance(y, int) and not isinstance(y, bool) for y in x):
        yield path, torch.tensor(list(x), dtype=torch.int64)
    elif isinstance(x, (list, tuple)) and depth < 2:
        for i, y in enumerate(x):
            yield from _tensors(y, f"{path}.{i}", depth + 1)


def _layer_of(a) -> int | None:
    for v in a[1:3]:
        i = getattr(v, "idx", None)
        if isinstance(i, int):
            return i
    return None


def _put_layer(call: int, where: str, layer: int | None, path: str, t) -> None:
    import hashlib

    import torch

    line = {"phase": "/".join(getattr(_tls, "stack", None) or ["other"]), "call": call, "where": where, "layer": layer,
            "arg": path, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "stride": list(t.stride())}
    n = t.numel() * t.element_size()
    if n > DIGEST_MAX:
        line["skipped"] = n
    else:
        c = t.detach()
        if c.is_complex():
            c = torch.view_as_real(c)
        raw = c.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        row = len(raw) // t.shape[0] if t.dim() > 0 and t.shape[0] > 0 else len(raw)
        line.update(sha256=hashlib.sha256(raw).hexdigest(), head=raw[:64].hex(), tail=raw[len(raw) - row:][:64].hex())
        if n <= FULL_MAX and (layer in LAYERS_FULL or where in WHOLE):
            f = OUT / "layers" / f"{call:05d}.{where}.{path}.bin"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(raw)
            line["file"] = f.name
    with _lock:
        _lay_lines.append(line)


def _layer_point(where: str, f, skip: tuple = ()):
    """f, writing its tensors in and out while the LAYERS flag is up (synchronized first; never in a graph capture);
    positional arguments in `skip` (numbered from 1 after self) are left out."""

    @functools.wraps(f)
    def w(*a, **k):
        import torch

        if not _armed() or torch.cuda.is_current_stream_capturing() or (ONLY and where not in ONLY):
            return f(*a, **k)
        with _lock:
            _lay["call"] += 1
            call = _lay["call"]
        layer = _layer_of(a)
        ins = [(f"in{i}", v) for i, v in enumerate(a[1:], 1) if i not in skip] + [(f"in.{kk}", v) for kk, v in k.items()]

        def put(items, tag):
            try:
                torch.cuda.synchronize()
                for name, v in items:
                    for path, t in _tensors(v, tag + name):
                        _put_layer(call, where, layer, path, t)
            except Exception as e:  # never break the forward
                print(f"zrec: layer point {where} {tag} not taken ({e!r})", file=sys.stderr, flush=True)

        put(ins, "")
        out = f(*a, **k)
        put([("out", out)], "")
        put(ins, "post.")
        return out

    w._zrec_layer = True
    return w


def _patch_layers() -> None:
    import importlib

    for mod, owner, attr, *skip in LAYER_POINTS:
        try:
            target = getattr(importlib.import_module(f"{FAMILY}.{mod}"), owner)
            f = getattr(target, attr)
            if not getattr(f, "_zrec_layer", False):
                setattr(target, attr, _layer_point(f"{owner}.{attr}", f, skip[0] if skip else ()))
        except Exception as e:
            print(f"zrec: no layer point {mod}.{owner}.{attr} ({e!r})", file=sys.stderr, flush=True)


def _wrap_module(mod, label: str) -> None:
    names = tuple(n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n, None)))
    _rec.wrap(mod, names, label)
    with _lock:
        _modules[label] = getattr(mod, "__file__", None)


def _patch_loaders() -> None:
    """Every extension module comes back from a loader with its functions logged."""

    import torch.utils.cpp_extension as ce

    import tensorfold.cuda.build as tb

    for owner in (tb, ce):
        orig = owner.load
        if getattr(orig, "_zrec", False):
            continue

        def load(name, *a, _orig=orig, **k):
            m = _orig(name, *a, **k)
            _wrap_module(m, name)
            return m

        load._zrec = True
        owner.load = load


def _patch_sampling() -> None:
    import tensorfold.cuda.sampling as sm

    for attr in ("sample_rows", "sample_streams", "nucleus_rows"):
        if hasattr(sm, attr):
            setattr(sm, attr, scoped("sample", getattr(sm, attr)))


def meta() -> dict:
    import torch
    import triton

    out = {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
           "python": sys.version.split()[0], "argv": sys.argv, "pid": os.getpid(),
           "env": {k: v for k, v in os.environ.items() if k.startswith(("TF_", "TRITON_", "CUBLAS", "NCCL_", "TORCH_"))}}
    try:
        import tensorfold

        out["tensorfold"] = str(Path(tensorfold.__file__).resolve().parent)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            out["device"] = torch.cuda.get_device_name(0)
            out["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    return out


def dump() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if _ztrace is not None:
        _ztrace.ztrace_flush()
    _rec.dump(OUT / "launches.json.tmp")
    os.replace(OUT / "launches.json.tmp", OUT / "launches.json")
    with _lock:
        mods = dict(_modules)
    (OUT / "modules.json").write_text(json.dumps(mods, indent=1) + "\n")
    if _ops is not None:
        (OUT / "aten.json").write_text(json.dumps(_ops.dump(), indent=0, default=str) + "\n")
    (OUT / "meta.json").write_text(json.dumps(meta(), indent=1, default=str) + "\n")
    if LAYERS_ON:
        with _lock:
            lines = list(_lay_lines)
        (OUT / "layers.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (OUT / "DUMPED").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


def _watch() -> None:
    while True:
        time.sleep(2)
        flag = OUT / "DUMP"
        if flag.exists():
            try:
                dump()
                print(f"zrec: dumped to {OUT}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"zrec: dump failed ({e!r})", file=sys.stderr, flush=True)
            flag.unlink(missing_ok=True)


def install() -> None:
    """Patch Triton, the extension loaders, sampling and the engine's phases; start the dump watcher."""

    global _rec, _ops, _OpLog, _ztrace
    OUT.mkdir(parents=True, exist_ok=True)
    inj = os.environ.get("CUDA_INJECTION64_PATH")
    if inj and inj.endswith("ztrace.so"):
        import ctypes

        _ztrace = ctypes.CDLL(inj)
        _ztrace.ztrace_phase.argtypes = [ctypes.c_char_p]
    _rec = _recorder_class()().install()
    _ops = Ops()
    if OPS_ON:
        _OpLog = _mode_class()
    _patch_loaders()
    _patch_sampling()
    if LAYERS_ON:
        _patch_layers()
    for mod, owner, attr, name in PHASES:
        _patch(f"{FAMILY}.{mod}", owner, attr, name)
    threading.Thread(target=_watch, name="zrec-dump", daemon=True).start()
    atexit.register(dump)
    print(f"zrec: recording to {OUT} (ATen ops {'on' if OPS_ON else 'off'}, {KEEP} details a shape)",
          file=sys.stderr, flush=True)
