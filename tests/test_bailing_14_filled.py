"""The dense latent attention's filled walk (TF_LING_FILLED): a call sized for the whole cache, as a decode graph is,
equals one sized for the rows' chunks, as eager is, bit for bit, and both equal today's walk; partials past a row's last
chunk are never read. tests/bailing_kernel_checks.py runs it in a child process (Triton's interpreter on hosts without
CUDA, natively on a GPU).

Run: python -m pytest tests/test_bailing_14_filled.py -q
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")


def test_filled_walk_has_todays_and_eagers_bits():
    pytest.importorskip("triton")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = dict(os.environ)
    if dev == "cpu":
        env["TRITON_INTERPRET"] = "1"
    r = subprocess.run([sys.executable, str(REPO / "tests" / "bailing_kernel_checks.py"), "filled", dev], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout[-4000:] + r.stderr[-4000:]


def test_ling_walk_stays_off_until_the_gates():
    source = (REPO / "src" / "tensorfold" / "families" / "bailing_v3" / "cuda" / "forward.py").read_text()
    assert 'FILLED = os.environ.get("TF_LING_FILLED", "0") == "1"' in source and "filled=FILLED" in source
