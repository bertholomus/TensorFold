"""The Engram I/O extension (engram_io.cpp, built by model._engram_io as the engine builds it) lets its process exit: its three reader
pools (127 threads each at the default TF_DS_ENGRAM_THREADS=128) and its AIO touch thread are stopped and joined at exit.
A thread left waiting on a condition variable that exit() destroys (the touch thread's touch_cv) would hold the process
in exit(): glibc's pthread_cond_destroy waits for the condition variable's waiters. A child process reads rows on every
path (reader pools 0, 1 and 2, the single-table call, AIO with and without a copy, touches), checks them against the
file's bytes, then exits normally, or is sent SIGTERM while its reads are idle (the engine's handler: KeyboardInterrupt,
then a normal exit). Either way it must end within 3 s, with status 0. Linux only (Linux AIO)."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the extension uses Linux AIO")

ROWS, RW, RS = 50_000, 256, 8              # a table's rows; weight and scale row bytes (264 B a row, as in the model's tables)
BASE_W = 3 * 4096 + 136                    # the weight rows' base (unaligned: rows straddle 4 KB blocks)
BASE_S = BASE_W + ROWS * RW + 72           # the scale rows', which end the file (the last row's block is short)
THREADS = 128                              # the default TF_DS_ENGRAM_THREADS: three pools of 127
EXIT_S = 3.0

CHILD = r"""
import json, os, signal, sys, time
import numpy as np
import torch
from tensorfold.families.deepseek_v41.cuda.model import ENGRAM_AIO_SLOTS, _engram_io

path = sys.argv[1]
rows, rw, rs, base_w, base_s, threads = (int(v) for v in sys.argv[2:8])
if sys.argv[8] == "sigterm":
    def _terminate(signum, frame):
        raise KeyboardInterrupt                  # the engine's handler (cli.py, cuda/http.py)
    signal.signal(signal.SIGTERM, _terminate)
io = _engram_io()
fd = os.open(path, os.O_RDONLY)
try:
    dfd = os.open(path, os.O_RDONLY | os.O_DIRECT)
except OSError as exc:
    print("RESULT " + json.dumps({"skip": f"O_DIRECT refused here: {exc}"}), flush=True)
    sys.exit(0)
if not io.aio_init(ENGRAM_AIO_SLOTS):
    print("RESULT " + json.dumps({"skip": "io_setup refused here"}), flush=True)
    sys.exit(0)
data = np.fromfile(path, dtype=np.uint8)
rng = np.random.default_rng(5)
none = torch.empty((0,), dtype=torch.uint8)
equal = of = 0
for m in (1, 6, 16, 256, 2048):
    ix = rng.integers(0, rows, size=m, dtype=np.int64)
    ix[0] = rows - 1
    idx = torch.from_numpy(ix)
    ww = torch.from_numpy(np.stack([data[base_w + i * rw:base_w + (i + 1) * rw] for i in ix]))
    ws = torch.from_numpy(np.stack([data[base_s + i * rs:base_s + (i + 1) * rs] for i in ix]))
    for call in ("gather_rows2", "gather_rows2_bg", 0, 1, 2, "aio"):
        ow, os_ = torch.empty((m, rw), dtype=torch.uint8), torch.empty((m, rs), dtype=torch.uint8)
        if call == "aio":
            if 2 * m > ENGRAM_AIO_SLOTS:
                continue
            # a touch through the touch thread, one submitted inline, then a read with its copy (a decode round's)
            io.aio_touch(dfd, base_w, rw, dfd, base_s, rs, idx[:1].clone())
            io.aio_start(dfd, base_w, rw, dfd, base_s, rs, idx[:1].clone(), none, none, 0)
            io.aio_wait(io.aio_start(dfd, base_w, rw, dfd, base_s, rs, idx, ow, os_, 1))
        elif isinstance(call, int):
            io.gather_rows2_at(call, fd, base_w, rw, fd, base_s, rs, idx, ow, os_, threads)
        else:
            getattr(io, call)(fd, base_w, rw, fd, base_s, rs, idx, ow, os_, threads)
        equal += int(torch.equal(ow, ww) and torch.equal(os_, ws))
        of += 1
    one = torch.empty((m, rs), dtype=torch.uint8)
    io.gather_rows(fd, base_s, rs, idx, one, threads)
    equal += int(torch.equal(one, ws))
    of += 1
tasks = len(os.listdir("/proc/self/task"))
print("RESULT " + json.dumps({"equal": equal, "of": of, "threads": tasks}), flush=True)
if sys.argv[8] == "sigterm":
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
"""


def _env() -> dict:
    env = dict(os.environ)
    src = Path(__file__).resolve().parents[1] / "src"
    if src.is_dir():
        env["PYTHONPATH"] = os.pathsep.join([str(src), *filter(None, [env.get("PYTHONPATH")])])
    return env


@pytest.fixture(scope="module")
def built():
    """The extension as the engine builds it, built once in a child (the children below load the cached build)."""

    probe = "from tensorfold.families.deepseek_v41.cuda.model import _engram_io; _engram_io()"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=1200, env=_env())
    if out.returncode != 0:
        pytest.skip("the extension does not build here: " + (out.stderr.strip().splitlines() or ["?"])[-1])


def _table(dirpath: Path) -> Path:
    """A file laid out as a layer's two Engram tables: weight rows at BASE_W, scale rows at BASE_S (random bytes)."""

    path = dirpath / "engram_tables.bin"
    np.random.default_rng(3).integers(0, 256, size=BASE_S + ROWS * RS, dtype=np.uint8).tofile(path)
    return path


def run_child(dirpath: Path, how: str):
    """(the child's reads, the seconds from the end of its reads (how="exit") or from SIGTERM (how="sigterm") to its
    exit, its status), or (reads, None, None) when it is still running EXIT_S seconds on (then killed)."""

    table = _table(dirpath)
    args = [str(table), ROWS, RW, RS, BASE_W, BASE_S, THREADS, how]
    with open(dirpath / f"child_{how}.err", "w") as err:
        p = subprocess.Popen([sys.executable, "-c", CHILD, *map(str, args)], stdout=subprocess.PIPE, stderr=err,
                             text=True, env=_env())
        late = threading.Timer(300, p.kill)              # no result in 5 min: killed (readline returns "")
        late.start()
        try:
            line = " "
            while line and not line.startswith("RESULT "):
                line = p.stdout.readline()
            late.cancel()
            if not line:
                p.wait()
                raise AssertionError(f"the child ended without its reads' result (status {p.returncode}): "
                                     + (dirpath / f"child_{how}.err").read_text()[-2000:])
            result = json.loads(line[len("RESULT "):])
            if "skip" in result:
                p.wait(timeout=60)
                pytest.skip(result["skip"])
            if how == "sigterm":
                time.sleep(0.3)                          # the reads idle
            t0 = time.monotonic()
            if how == "sigterm":
                p.send_signal(signal.SIGTERM)
            try:
                code = p.wait(timeout=EXIT_S)
            except subprocess.TimeoutExpired:
                return result, None, None
            return result, time.monotonic() - t0, code
        finally:
            late.cancel()
            if p.poll() is None:
                p.kill()
                p.wait()


def _check(result: dict, took, code, how: str) -> None:
    print(f"\n[engram exit, {how}] {result}; exit after {took if took is None else round(took, 3)} s, status {code}")
    # every path's rows are the file's bytes: 5 sizes x 7 calls, but no AIO read of 2048 rows (the ring holds 512)
    assert result["equal"] == result["of"] == 4 * 7 + 6, result
    assert result["threads"] >= 3 * (THREADS - 1) + 2, result            # the pools' threads, the touch thread, main
    assert took is not None, f"still running {EXIT_S} s after {'SIGTERM' if how == 'sigterm' else 'its reads'}"
    assert code == 0


def test_a_normal_exit_after_reads_on_every_path_ends_within_3_s(built, tmp_path):
    _check(*run_child(tmp_path, "exit"), "exit")


def test_sigterm_while_the_reads_are_idle_ends_within_3_s(built, tmp_path):
    _check(*run_child(tmp_path, "sigterm"), "sigterm")
