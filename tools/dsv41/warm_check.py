"""After the engine's warm-up: does any request still build a Triton kernel or capture a CUDA graph? (TP ranks, same
command on each.) Random prompt lengths, drafted and serial, a decode that crosses a context bucket, drafted == serial.

  python3 warm_check.py --rank R --model M [--context 262144] [--requests 40] [--out F]
"""

import argparse
import json
import random
import time

import torch


class Compiles:
    """Counts Triton in-memory cache misses (a build, or a load from the on-disk cache) by kernel name."""

    def __init__(self):
        self.names: list[str] = []
        self.detail: list[dict] = []
        import triton

        def hook(*args, **kw):
            fn = kw.get("fn")
            name = getattr(fn, "name", None) or getattr(getattr(fn, "jit_function", None), "__name__", None)
            self.names.append((name or str(kw.get("repr", "?"))[:60]).split(".")[-1])
            comp = kw.get("compile") or {}
            self.detail.append({"name": self.names[-1], "repr": str(kw.get("repr", ""))[-400:],
                                "constants": {str(k): str(v) for k, v in (comp.get("constants") or {}).items()}})
            return False

        if hasattr(getattr(triton, "knobs", None), "runtime") and hasattr(triton.knobs.runtime, "jit_cache_hook"):
            triton.knobs.runtime.jit_cache_hook = hook
        else:
            from triton.runtime.jit import JITFunction

            JITFunction.cache_hook = hook

    def take(self) -> list[str]:
        out, self.names = self.names, []
        return out

    def take_detail(self) -> list[dict]:
        out, self.detail = self.detail, []
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--master", required=True)
    p.add_argument("--port", type=int, default=29665)
    p.add_argument("--model", required=True)
    p.add_argument("--engram")
    p.add_argument("--context", type=int, default=262144)
    p.add_argument("--requests", type=int, default=40)
    p.add_argument("--out")
    a = p.parse_args()
    from pathlib import Path

    from tensorfold.families.deepseek_v41.cuda.engine import DsEngine

    t0 = time.perf_counter()
    eng = DsEngine(Path(a.model), rank=a.rank, world=a.world, master=a.master, port=a.port, drafts=3,
                   context=a.context, engram_dir=a.engram)
    init_s = time.perf_counter() - t0
    eng.request.stop_eos = False
    cc = Compiles()
    rng = random.Random(7)
    lengths = [rng.randint(1, 300) for _ in range(a.requests // 2)] + \
              [rng.randint(300, 9000) for _ in range(a.requests - a.requests // 2)]
    rng.shuffle(lengths)
    vocab = [rng.randint(1000, 100000) for _ in range(9000)]
    rows = []
    for i, n in enumerate(lengths):
        draft = i % 4 != 3
        cap0 = eng.runner.captures if eng.runner else 0
        torch.cuda.synchronize()
        tq = time.perf_counter()
        st = eng._run(vocab[:n], 8, None, False, lambda new: None, draft)
        torch.cuda.synchronize()
        rows.append({"len": n, "draft": draft, "seconds": round(time.perf_counter() - tq, 3),
                     "prefill_s": round(st["prefill_s"], 3), "compiles": cc.take(), "detail": cc.take_detail(),
                     "captures": (eng.runner.captures if eng.runner else 0) - cap0})
        if a.rank == 0:
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "detail"}), flush=True)
    # a decode that crosses the 1024 -> 2048 bucket, drafted and serial must agree
    same = {}
    for draft in (True, False):
        got: list = []
        cap0 = eng.runner.captures if eng.runner else 0
        st = eng._run(vocab[:1000], 96, None, False, got.extend, draft)
        same[draft] = got
        rows.append({"len": 1000, "decode": 96, "draft": draft, "tps": round(st["tokens_per_second"], 2),
                     "compiles": cc.take(), "captures": (eng.runner.captures if eng.runner else 0) - cap0})
        if a.rank == 0:
            print(json.dumps(rows[-1]), flush=True)
    summ = {"init_s": round(init_s, 1), "requests": len(rows),
            "requests_with_compiles": sum(bool(r["compiles"]) for r in rows),
            "compiled": sorted({c for r in rows for c in r["compiles"]}),
            "captures": sum(r["captures"] for r in rows),
            "drafted_equals_serial_across_bucket": same[True] == same[False]}
    if a.rank == 0:
        print(json.dumps({"summary": summ}), flush=True)
        if a.out:
            json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
