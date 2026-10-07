"""Acceptance tests for the C5 needle client (run with /usr/bin/python3, stdlib only).

  (a) needle.py without --shared sends exactly the same requests (path + body) as needle_orig.py (e5074cf's)
  (b) needle.py --shared builds one haystack per length, asks vault A/B/C in order, finds 3/3
  (c) every .py here and the two clients compile
"""

import glob
import json
import os
import py_compile
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENTS = os.path.dirname(HERE)                          # tools/dsv41: needle.py, long_check.py
sys.path.insert(0, HERE)

import fake_server


def run_client(script, args, timeout=300):
    cmd = [sys.executable, script] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError("%s failed rc=%d\nstdout:\n%s\nstderr:\n%s" % (script, r.returncode, r.stdout, r.stderr))
    return r.stdout


def test_parity():
    args = ["--model", "m", "--lengths", "1500,2500", "--depths", "0.1,0.9", "--seed", "7"]
    outs, recorded = [], []
    for script in (os.path.join(HERE, "needle_orig.py"), os.path.join(CLIENTS, "needle.py")):
        fs, httpd, base = fake_server.start()
        try:
            outs.append(run_client(script, ["--base", base] + args))
            recorded.append([(r["path"], r["body"]) for r in fs.records])
        finally:
            httpd.shutdown()
            httpd.server_close()
    assert recorded[0] == recorded[1], "recorded request lists differ"
    assert outs[0] == outs[1], "stdout differs"


def test_shared():
    fs, httpd, base = fake_server.start()
    try:
        out = run_client(os.path.join(CLIENTS, "needle.py"), ["--base", base, "--model", "m", "--shared",
                                       "--lengths", "2000", "--depths", "0.1,0.5,0.9", "--seed", "7"])
    finally:
        httpd.shutdown()
        httpd.server_close()
    objs = [json.loads(line) for line in out.strip().splitlines()]
    rows = [o for o in objs if "summary" not in o]
    assert len(rows) == 3, rows
    assert [r["label"] for r in rows] == ["A", "B", "C"], rows
    assert objs[-1]["summary"] == {"cases": 3, "found": 3}, objs[-1]
    assert all(r["shared"] is True and "cached_tokens" in r for r in rows)
    chats = [r for r in fs.records if r["path"] == "/v1/chat/completions"]
    assert len(chats) == 3, len(chats)
    prefixes = [c["body"]["messages"][0]["content"].split("\n\nWhat is")[0] for c in chats]
    assert prefixes[0] == prefixes[1] == prefixes[2], "chat requests do not share one haystack"
    for k, row in enumerate(rows):
        content = chats[k]["body"]["messages"][0]["content"]
        assert row["phrase"] in content, row
        assert content.find("vault %s is " % row["label"]) != -1, row
    assert (prefixes[0].find("vault A is ") < prefixes[0].find("vault B is ")
            < prefixes[0].find("vault C is ")), "needle depths out of order"


def test_compile():
    for path in sorted(glob.glob(os.path.join(HERE, "*.py"))) + \
                [os.path.join(CLIENTS, "needle.py"), os.path.join(CLIENTS, "long_check.py")]:
        py_compile.compile(path, doraise=True)


def main():
    test_parity()
    print("needle_fake_test: parity OK")
    test_shared()
    print("needle_fake_test: shared OK")
    test_compile()
    print("needle_fake_test: compile OK")


if __name__ == "__main__":
    main()
