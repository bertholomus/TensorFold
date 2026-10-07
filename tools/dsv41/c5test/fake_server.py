"""Fake OpenAI-style server standing in for the real TensorFold endpoint while testing the C5 clients (stdlib only).

Start one with fake_server.start(); it binds 127.0.0.1 on an ephemeral port inside a daemon thread and returns
(FakeServer, httpd, base_url). Every request is recorded as {"path", "body", "t"} on the FakeServer object.

  /tokenize             {"prompt": text}          -> one int per whitespace word, zlib.crc32(word) % 100000
        {"messages": [...]}                       -> same for the last user content + 5 fixed template ids;
                                                     remembers tuple(ids) -> that content
  /v1/chat/completions  content = the phrase of the vault label asked ("passphrase for vault X" -> "vault X is ..."),
        or of the single "passphrase is " needle when no label; usage prompt_tokens = word count, cached_tokens = 0
        the first time a text prefix up to the question is seen, else prompt_tokens - 10
  /v1/completions       token_ids deterministic in (prompt ids, temperature, seed) of length max_tokens, identical
        with or without "draft": false; text = the phrase of the content remembered for those ids; with
        server.mismatch = True, requests that overlap in time return different ids.
"""

import json
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TEMPLATE_IDS = [90001, 90002, 90003, 90004, 90005]


def word_tokens(text):
    return [zlib.crc32(w.encode()) % 100000 for w in text.split()]


def extract_phrase(content):
    # rfind: the label asked is in the trailing question, after the needles themselves
    idx = content.rfind("passphrase for vault ")
    if idx != -1:
        label = content[idx + len("passphrase for vault "):][:1]
        marker = "vault " + label + " is "
        j = content.find(marker)
        if j != -1:
            return content[j + len(marker):].split(".")[0].strip()
    idx = content.find("passphrase is ")
    if idx != -1:
        return content[idx + len("passphrase is "):].split(".")[0].strip()
    return ""


class FakeServer:
    def __init__(self):
        self.records = []  # {"path", "body", "t"} for every request, in arrival order
        self.prefixes = set()
        self.ids_to_content = {}
        self.mismatch = False
        self._lock = threading.Lock()
        self._inflight = set()
        self._overlapped = set()
        self._serial = 0

    def arrive(self, path, body):
        """Record one request and register it in flight; returns its serial for depart()."""
        with self._lock:
            self.records.append({"path": path, "body": body, "t": time.time()})
            self._serial += 1
            serial = self._serial
            self._inflight.add(serial)
            if len(self._inflight) > 1:  # everyone in flight during an overlap counts as burst
                self._overlapped.update(self._inflight)
        return serial

    def depart(self, serial):
        with self._lock:
            self._inflight.discard(serial)

    def tokenize(self, body):
        if "messages" in body:
            content = [m for m in body["messages"] if m.get("role") == "user"][-1]["content"]
            tokens = word_tokens(content) + TEMPLATE_IDS
            with self._lock:
                self.ids_to_content[tuple(tokens)] = content
        else:
            tokens = word_tokens(body["prompt"])
        return {"tokens": tokens, "count": len(tokens)}

    def _usage(self, content):
        n = len(content.split())
        prefix = content.split("\n\nWhat is")[0]
        with self._lock:
            if prefix in self.prefixes:
                cached = n - 10
            else:
                self.prefixes.add(prefix)
                cached = 0
        return n, cached

    def chat(self, body):
        content = [m for m in body["messages"] if m.get("role") == "user"][-1]["content"]
        n, cached = self._usage(content)
        return {"choices": [{"message": {"content": extract_phrase(content)}}],
                "usage": {"prompt_tokens": n, "prompt_tokens_details": {"cached_tokens": cached}}}

    def completions(self, body, serial):
        # in-flight bookkeeping (arrive/depart) is owned by the handler; it spans socket I/O so
        # concurrent requests genuinely overlap
        ids = body["prompt"]
        content = self.ids_to_content.get(tuple(ids), "")
        base = (sum(ids) + int(body.get("temperature", 1.0) * 10) + body.get("seed", 0)) % 1000
        token_ids = [(base + 7 * j) % 1000 for j in range(body.get("max_tokens", 16))]
        with self._lock:
            overlapped = serial in self._overlapped
        if self.mismatch and overlapped:
            token_ids = [(x + 137 * serial) % 1000 for x in token_ids]
        n, cached = self._usage(content)
        return {"choices": [{"text": extract_phrase(content)}],
                "usage": {"prompt_tokens": n, "prompt_tokens_details": {"cached_tokens": cached}},
                "tensorfold": {"token_ids": token_ids}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        fs = self.server.fs
        serial = fs.arrive(self.path, body)
        try:
            if self.path == "/tokenize":
                resp = fs.tokenize(body)
            elif self.path == "/v1/chat/completions":
                resp = fs.chat(body)
            elif self.path == "/v1/completions":
                resp = fs.completions(body, serial)
            else:
                resp = {}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            fs.depart(serial)


def start():
    """Start a fresh fake server; returns (FakeServer, httpd, base_url)."""
    fs = FakeServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.fs = fs
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return fs, httpd, "http://127.0.0.1:%d" % httpd.server_address[1]
