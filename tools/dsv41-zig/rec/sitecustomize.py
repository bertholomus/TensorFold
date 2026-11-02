"""Turns on the Zig port's recorder (zrec.py) in every Python process of a recording container when TF_ZREC_DIR is set.

This folder goes first on PYTHONPATH, so this file shadows any other sitecustomize: that one is run after ours.
"""

import os
import sys


def _chain() -> None:
    """Run the next sitecustomize on sys.path (the image's own, if it has one)."""

    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        if not d or os.path.abspath(d) == here:
            continue
        p = os.path.join(d, "sitecustomize.py")
        if os.path.isfile(p):
            code = compile(open(p).read(), p, "exec")
            exec(code, {"__name__": "sitecustomize_next", "__file__": p})
            return


_chain()
if os.environ.get("TF_ZREC_DIR"):
    try:
        import zrec

        zrec.install()
    except Exception as e:  # the recorder never stops the server
        import traceback

        print(f"zrec: not installed ({e!r})", file=sys.stderr, flush=True)
        traceback.print_exc()
