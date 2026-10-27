"""Freeze xgrammar's json_object verdicts on random byte strings as zig/src/core/grammar's differential corpus."""
import argparse
import random
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "zig/src/core/grammar/testdata/xgrammar_json_object.txt"
BLANKS = b" \n\r\t"
STOP = 256
BLANK_RUN = 32
# A raw UTF-8 surrogate (ED A0..BF): xgrammar 0.2.8 takes it, the Zig matcher refuses it on purpose.
SURROGATE = re.compile(rb"\xed[\xa0-\xbf]")


def matcher_factory():
    """A function that returns a fresh xgrammar matcher for json_object over a one-byte-per-token vocabulary."""
    import xgrammar as xgr

    vocab = [bytes([i]) for i in range(256)] + [b"<eos>"]
    info = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=len(vocab), stop_token_ids=[STOP])
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_json_schema('{"type": "object"}',
                                                                           max_whitespace_cnt=BLANK_RUN)
    return lambda: xgr.GrammarMatcher(compiled)


def verdict(new, data: bytes) -> str:
    """The first rejected byte's index, else c when the end of the reply is accepted, else p."""
    m = new()
    for i, b in enumerate(data):
        if not m.accept_token(b):
            return str(i)
    return "c" if m.accept_token(STOP) else "p"


def encode(data: bytes) -> str:
    """Printable bytes as themselves, a space as _, every other byte (and % and _) as %XX."""
    out = []
    for b in data:
        if b == 0x20:
            out.append("_")
        elif 0x21 <= b <= 0x7E and b not in (0x25, 0x5F):
            out.append(chr(b))
        else:
            out.append(f"%{b:02X}")
    return "".join(out)


class Gen:
    """Random JSON-like byte strings, mostly valid, with the near misses a byte matcher must refuse."""

    def __init__(self, rng: random.Random) -> None:
        self.r = rng

    def ws(self) -> bytes:
        """A blank run: usually short, sometimes at or just past the 32 limit."""
        k = self.r.random()
        n = 0 if k < 0.55 else self.r.randint(1, 3) if k < 0.88 else self.r.choice([31, 32, 33, 34]) if k < 0.96 \
            else self.r.randint(4, 30)
        return bytes(self.r.choice(BLANKS) if self.r.random() < 0.3 else 0x20 for _ in range(n))

    def utf8(self, cp: int) -> bytes:
        """Code point `cp` in UTF-8, surrogates excluded."""
        return chr(cp).encode("utf-8")

    def bad_utf8(self) -> bytes:
        """A byte run that is not strict UTF-8: overlong, too large, stray, truncated."""
        r = self.r
        return r.choice([
            bytes([r.choice([0xC0, 0xC1]), r.randint(0x80, 0xBF)]),
            bytes([0xE0, r.randint(0x80, 0x9F), r.randint(0x80, 0xBF)]),
            bytes([0xF0, r.randint(0x80, 0x8F), r.randint(0x80, 0xBF), r.randint(0x80, 0xBF)]),
            bytes([0xF4, r.randint(0x90, 0xBF), r.randint(0x80, 0xBF), r.randint(0x80, 0xBF)]),
            bytes([r.randint(0xF5, 0xFF), r.randint(0x80, 0xBF), r.randint(0x80, 0xBF)]),
            bytes([r.randint(0x80, 0xBF)]),
            bytes([r.randint(0xC2, 0xDF)]),
            bytes([r.randint(0xE1, 0xEC), r.randint(0x80, 0xBF)]),
            bytes([r.randint(0xF1, 0xF3), r.randint(0x80, 0xBF), r.randint(0x80, 0xBF)]),
            bytes([r.randint(0xC2, 0xDF), r.choice([0x22, 0x41, 0xC3])]),
        ])

    def escape(self) -> bytes:
        """A backslash escape: simple, \\u (lone surrogates and pairs too), or a bad one."""
        r, k = self.r, self.r.random()
        if k < 0.5:
            return b"\\" + bytes([r.choice(b'"\\/bfnrt')])
        if k < 0.8:
            return b"\\u" + "".join(r.choice("0123456789abcdefABCDEF") for _ in range(4)).encode()
        if k < 0.92:
            hi, lo = r.randint(0xD800, 0xDBFF), r.randint(0xDC00, 0xDFFF)
            return r.choice([f"\\u{hi:04X}\\u{lo:04x}", f"\\u{hi:04x}", f"\\u{lo:04X}"]).encode()
        return b"\\" + r.choice([b"x", b"a", b"U0041", b"u12", b"u12G4", b"'", b"0", b" ", b"\n"])

    def char(self) -> bytes:
        """One string character, or a near miss."""
        r, k = self.r, self.r.random()
        if k < 0.7:
            return bytes([r.choice([c for c in range(0x20, 0x80) if c not in (0x22, 0x5C)])])
        if k < 0.82:
            return self.escape()
        if k < 0.95:
            cp = r.choice([r.randint(0x80, 0x7FF), r.randint(0x800, 0xD7FF), r.randint(0xE000, 0xFFFF),
                           r.randint(0x10000, 0x10FFFF), r.choice([0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF])])
            return self.utf8(cp)
        if k < 0.98:
            return self.bad_utf8()
        return bytes([r.randint(0x00, 0x1F)])

    def string(self) -> bytes:
        """A quoted string, now and then left without its closing quote."""
        body = b"".join(self.char() for _ in range(self.r.choice([0, 0, 1, 1, 2, 3, 5, 8])))
        return b'"' + body + (b'"' if self.r.random() < 0.97 else b"")

    def number(self) -> bytes:
        """A JSON number, or a near miss such as 01, 1., .5 or +1."""
        r = self.r
        if r.random() < 0.8:
            text = r.choice(["", "-"]) + r.choice(["0", str(r.randint(1, 9)) + "".join(r.choice("0123456789")
                                                                                      for _ in range(r.randint(0, 4)))])
            if r.random() < 0.3:
                text += "." + "".join(r.choice("0123456789") for _ in range(r.randint(1, 3)))
            if r.random() < 0.25:
                text += r.choice("eE") + r.choice(["", "+", "-"]) + str(r.randint(0, 99))
            return text.encode()
        return r.choice([b"01", b"-01", b"1.", b".5", b"+1", b"-", b"1e", b"1e+", b"1.e5", b"--1", b"0x1", b"Infinity",
                         b"NaN", b"1.5.2", b"1e5e5", b"00", b"-.5", b"1E", b"0.", b"-0.e1"])

    def literal(self) -> bytes:
        """true, false or null, or a misspelling."""
        return self.r.choice([b"true", b"false", b"null", b"true", b"false", b"null", b"tru", b"True", b"nul", b"nulll",
                              b"fals", b"t", b"nill"])

    def value(self, depth: int) -> bytes:
        """Any value; containers only above the depth limit."""
        k = self.r.random()
        if depth < 4 and k < 0.22:
            return self.obj(depth + 1)
        if depth < 4 and k < 0.36:
            return self.arr(depth + 1)
        if k < 0.65:
            return self.string()
        if k < 0.88:
            return self.number()
        return self.literal()

    def obj(self, depth: int) -> bytes:
        """An object with zero to four members, with a missing colon, comma or key now and then."""
        r = self.r
        members = []
        for _ in range(r.choice([0, 1, 1, 2, 3, 4])):
            key = self.string() if r.random() < 0.95 else r.choice([b"a", b"1", b"null", b"'a'"])
            colon = b":" if r.random() < 0.97 else r.choice([b"", b"=", b";"])
            members.append(key + self.ws() + colon + self.ws() + self.value(depth))
        sep = b"," if r.random() < 0.97 else r.choice([b"", b";", b",,"])
        body = (self.ws() + sep + self.ws()).join(members)
        tail = b"," if members and r.random() < 0.03 else b""
        close = b"}" if r.random() < 0.96 else r.choice([b"]", b"", b"}}"])
        return b"{" + self.ws() + body + tail + self.ws() + close

    def arr(self, depth: int) -> bytes:
        """An array with zero to four items, with a missing or doubled comma now and then."""
        r = self.r
        items = [self.value(depth) for _ in range(r.choice([0, 1, 1, 2, 3, 4]))]
        sep = b"," if r.random() < 0.97 else r.choice([b"", b",,"])
        body = (self.ws() + sep + self.ws()).join(items)
        tail = b"," if items and r.random() < 0.03 else b""
        close = b"]" if r.random() < 0.96 else r.choice([b"}", b""])
        return b"[" + self.ws() + body + tail + self.ws() + close

    def doc(self) -> bytes:
        """A reply: an object mostly, else another root, blanks around it or something after it."""
        r, k = self.r, self.r.random()
        if k < 0.78:
            return self.obj(0)
        if k < 0.84:
            return self.value(0)
        if k < 0.92:
            return r.choice([b" ", b"\n", b"\t", b"\r\n"]) + self.obj(0) if r.random() < 0.5 else \
                self.obj(0) + r.choice([b" ", b"\n", b"\n\n", b"x", b"{}", b","])
        return b"{" + self.ws() + b'"' + b"a" * r.randint(0, 3) + b'"' + self.ws() + b":" + self.ws() + \
            self.value(0) + self.ws() + b"}"


def mutate(r: random.Random, data: bytes) -> bytes:
    """One random edit: insert, delete or replace a byte, or cut the string short."""
    if not data:
        return bytes([r.randint(0, 255)])
    i = r.randrange(len(data))
    k = r.random()
    if k < 0.3:
        return data[:i] + bytes([r.choice([r.randint(0, 255), r.choice(b'{}[]",: \n\\0e-.tu')])]) + data[i:]
    if k < 0.55:
        return data[:i] + data[i + 1:]
    if k < 0.8:
        return data[:i] + bytes([r.choice([r.randint(0, 255), r.choice(b'{}[]",: \n\\0e-.tu')])]) + data[i + 1:]
    return data[:i]


def main() -> int:
    """Generate, label with the installed xgrammar and write the corpus (or --check it against the frozen file)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--check", action="store_true", help="relabel the frozen file and report differences")
    args = p.parse_args()
    import xgrammar

    new = matcher_factory()
    version = getattr(xgrammar, "__version__", None) or __import__("importlib.metadata").metadata.version("xgrammar")
    if args.check:
        differ = 0
        for line in args.out.read_text().splitlines():
            if line.startswith("#"):
                continue
            want, _, payload = line.partition(" ")
            data = decode(payload)
            got = verdict(new, data)
            if got != want:
                differ += 1
                print(f"xgrammar {version}: {got}, frozen: {want}: {data!r}")
        print(f"{differ} differences")
        return 1 if differ else 0
    r = random.Random(args.seed)
    gen, seen, lines = Gen(r), set(), []
    while len(lines) < args.cases:
        data = gen.doc()
        if r.random() < 0.35:
            data = mutate(r, data)
        if data in seen or SURROGATE.search(data) or len(data) > 120:
            continue
        seen.add(data)
        lines.append(f"{verdict(new, data)} {encode(data)}".rstrip(" "))
    header = f"# xgrammar {version} json_object (max_whitespace_cnt={BLANK_RUN}) verdicts by tools/zig/xgrammar_json_corpus.py"
    args.out.write_text("\n".join([header, f"# seed {args.seed}, {len(lines)} cases: verdict, then the bytes"] + lines)
                        + "\n")
    print(f"{len(lines)} cases written to {args.out}")
    return 0


def decode(payload: str) -> bytes:
    """The inverse of encode."""
    out, i = bytearray(), 0
    while i < len(payload):
        c = payload[i]
        if c == "%":
            out.append(int(payload[i + 1:i + 3], 16))
            i += 3
        else:
            out.append(0x20 if c == "_" else ord(c))
            i += 1
    return bytes(out)


if __name__ == "__main__":
    sys.exit(main())
