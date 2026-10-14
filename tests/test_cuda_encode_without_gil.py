"""App._encode: a request's prompt is encoded through ``encode_batch`` (tokenizers releases the GIL there, not in
``encode``), with the same ids; a tokenizer without batches encodes as before; and with the real tokenizers library, a
long encode no longer stops another thread."""

import threading
import time
from types import SimpleNamespace

import pytest

from tensorfold.cuda.server import App


class Ids(SimpleNamespace):
    pass


class BatchTok:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append(("encode", text, add_special_tokens))
        return Ids(ids=[len(text)])

    def encode_batch(self, texts, add_special_tokens=False):
        self.calls.append(("batch", tuple(texts), add_special_tokens))
        return [Ids(ids=[len(t)]) for t in texts]


class PlainTok:
    def encode(self, text, add_special_tokens=False):
        return Ids(ids=[len(text), int(add_special_tokens)])


def test_a_prompt_is_encoded_as_a_batch_of_one():
    tok = BatchTok()
    assert App._encode(SimpleNamespace(tok=tok), "abcd", True) == [4]
    assert tok.calls == [("batch", ("abcd",), True)]


def test_a_tokenizer_without_batches_encodes_as_before():
    assert App._encode(SimpleNamespace(tok=PlainTok()), "abc", False) == [3, 0]


def test_a_long_encode_leaves_other_threads_running():
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers

    tok = tokenizers.Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    text = "a b " * 1_000_000
    stall, stop = [0.0], [False]

    def count():
        last = time.perf_counter()
        while not stop[0]:
            now = time.perf_counter()
            stall[0] = max(stall[0], now - last)
            last = now

    th = threading.Thread(target=count)
    th.start()
    time.sleep(0.05)
    stall[0] = 0.0
    t0 = time.perf_counter()
    ids = App._encode(SimpleNamespace(tok=tok), text, False)
    took = time.perf_counter() - t0
    stop[0] = True
    th.join()
    assert ids == tok.encode(text, add_special_tokens=False).ids
    assert took > 0.05 and stall[0] < took / 2      # the counter kept running through most of the encode
