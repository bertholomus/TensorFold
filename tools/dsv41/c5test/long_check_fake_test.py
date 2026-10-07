"""Acceptance tests for the C5 long_check client (run with /usr/bin/python3, stdlib only).

  (a) burst == solo (5/5), serial (draft off) == solo greedy, 4/4 needles recalled on the greedy replies, every burst
      request arrived before the first solo request, and the last request carries "draft": false
  (b) with the fake server's mismatch flag on, overlapping burst requests return different ids
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fake_server


def run_client(args, timeout=300):
    cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "long_check.py")] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError("long_check.py failed rc=%d\nstdout:\n%s\nstderr:\n%s"
                             % (r.returncode, r.stdout, r.stderr))
    return r.stdout


def test_basic():
    fs, httpd, base = fake_server.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "long_check.json")
            out = run_client(["--base", base, "--model", "m", "--lengths", "3000,3000,3000,3000",
                              "--tokens", "8", "--out", out_path])
            saved = json.load(open(out_path))
    finally:
        httpd.shutdown()
        httpd.server_close()
    summ = json.loads(out.strip().splitlines()[-1])
    assert summ["burst_equal_to_solo"] == 5, summ
    assert summ["of"] == 5, summ
    assert summ["drafted_equals_serial"] is True, summ
    assert summ["needles_found_burst"] == 4, summ
    assert summ["needles_found_solo"] == 4, summ
    comp = [r for r in fs.records if r["path"] == "/v1/completions"]
    assert len(comp) == 11, len(comp)  # 5 burst + 5 solo + 1 serial
    assert all(r["t"] < comp[5]["t"] for r in comp[:5]), "a burst request arrived after the first solo request"
    assert comp[-1]["body"].get("draft") is False, comp[-1]["body"]
    assert len(saved["replies"]) == 11 and all("token_ids" in r for r in saved["replies"]), "out file lacks token ids"


def test_mismatch():
    fs, httpd, base = fake_server.start()
    try:
        fs.mismatch = True
        with tempfile.TemporaryDirectory() as tmp:
            out = run_client(["--base", base, "--model", "m", "--lengths", "3000,3000,3000,3000",
                              "--tokens", "8", "--out", os.path.join(tmp, "long_check.json")])
    finally:
        httpd.shutdown()
        httpd.server_close()
    summ = json.loads(out.strip().splitlines()[-1])
    assert summ["burst_equal_to_solo"] < summ["of"], summ


def main():
    test_basic()
    print("long_check_fake_test: basic OK")
    test_mismatch()
    print("long_check_fake_test: mismatch OK")


if __name__ == "__main__":
    main()
