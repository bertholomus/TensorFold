"""Freeze xgrammar's json_schema and guided_choice verdicts on random byte strings as zig/src/core/grammar's corpus."""
import argparse
import json
import random
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "zig/src/core/grammar/testdata/xgrammar_schema.txt"
STOP = 256
BLANK_RUN = 32
# A raw UTF-8 surrogate (ED A0..BF): xgrammar 0.2.8 takes it, the Zig matcher refuses it on purpose.
SURROGATE = re.compile(rb"\xed[\xa0-\xbf]")
BLANKS = b" \n\r\t"

ITEM = {"type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
        "required": ["name", "value"], "additionalProperties": False}
# Each schema is passed as Python 0.6 passed it: json.dumps(schema, ensure_ascii=False).
SCHEMAS = [
    # the lab's gate G3 (infra/v4-flash/window-2026-10b/win-gates.sh)
    {"type": "object", "additionalProperties": False, "required": ["action", "confidence", "pair"],
     "properties": {"action": {"type": "string", "enum": ["buy", "sell", "hold"]}, "confidence": {"type": "number"},
                    "pair": {"type": "string"}}},
    # OpenAI structured outputs guide: chain of thought, extraction, UI (recursive), moderation, $defs, linked list
    {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {
        "explanation": {"type": "string"}, "output": {"type": "string"}}, "required": ["explanation", "output"],
        "additionalProperties": False}}, "final_answer": {"type": "string"}}, "required": ["steps", "final_answer"],
     "additionalProperties": False},
    {"type": "object", "properties": {"title": {"type": "string"}, "authors": {"type": "array", "items": {"type": "string"}},
                                      "abstract": {"type": "string"}, "keywords": {"type": "array", "items": {"type": "string"}}},
     "required": ["title", "authors", "abstract", "keywords"], "additionalProperties": False},
    {"type": "object", "properties": {"type": {"type": "string", "enum": ["div", "button", "header", "section", "field", "form"]},
                                      "label": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#"}},
                                      "attributes": {"type": "array", "items": ITEM}},
     "required": ["type", "label", "children", "attributes"], "additionalProperties": False},
    {"type": "object", "properties": {"is_violating": {"type": "boolean"},
                                      "category": {"type": ["string", "null"], "enum": ["violence", "sexual", "self_harm", None]},
                                      "explanation_if_violating": {"type": ["string", "null"]}},
     "required": ["is_violating", "category", "explanation_if_violating"], "additionalProperties": False},
    {"type": "object", "properties": {"steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
                                      "final_answer": {"type": "string"}},
     "$defs": {"step": {"type": "object", "properties": {"explanation": {"type": "string"}, "output": {"type": "string"}},
                        "required": ["explanation", "output"], "additionalProperties": False}},
     "required": ["steps", "final_answer"], "additionalProperties": False},
    {"type": "object", "properties": {"linked_list": {"$ref": "#/$defs/linked_list_node"}},
     "$defs": {"linked_list_node": {"type": "object", "properties": {"value": {"type": "number"}, "next": {
         "anyOf": [{"$ref": "#/$defs/linked_list_node"}, {"type": "null"}]}}, "additionalProperties": False,
         "required": ["next", "value"]}}, "required": ["linked_list"], "additionalProperties": False},
    # Anthropic structured outputs guide (output_config.format), and Pydantic's shape for optional fields
    {"type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string", "format": "email"},
                                      "plan_interest": {"type": "string"}, "demo_requested": {"type": "boolean"}},
     "required": ["name", "email", "plan_interest", "demo_requested"], "additionalProperties": False},
    {"title": "Event", "type": "object", "properties": {
        "name": {"title": "Name", "type": "string"},
        "when": {"title": "When", "type": "string", "format": "date-time"},
        "id": {"title": "Id", "type": "string", "format": "uuid"},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Note"},
        "tags": {"items": {"type": "string"}, "title": "Tags", "type": "array"}},
     "required": ["name", "when", "id", "tags"], "additionalProperties": False},
    # optional properties around a required one, other keys, names that start each other
    {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}, "c": {"type": "boolean"},
                                      "d": {"type": "null"}}, "required": ["b"]},
    {"type": "object", "properties": {"a": {"type": "integer"}, "ab": {"type": "string"}, "abc": {"type": "boolean"}},
     "additionalProperties": {"type": "integer"}},
    {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "integer"}}, "required": ["y"],
     "additionalProperties": True},
    {"type": "object", "properties": {"k": {"type": "integer"}}, "additionalProperties": False},
    {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "integer"}}},
    {"type": "object", "properties": {"a\"b": {"type": "integer"}, "é": {"type": "integer"}, "\n/": {"type": "integer"}},
     "required": ["a\"b", "é", "\n/"]},
    {"type": "object"}, {"type": "array"}, {}, {"type": "object", "$id": "x"}, {"type": "object", "required": []},
    # arrays
    {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 4},
    {"type": "array", "items": {"type": "string"}, "maxItems": 0},
    {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}]},
    {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "items": {"type": "boolean"}, "minItems": 1,
     "maxItems": 4},
    {"type": "array", "prefixItems": [{"type": "integer"}], "items": {"type": "null"}, "minItems": 3},
    {"type": "array", "items": {"type": "array", "items": {"type": "object", "properties": {"v": {"type": "integer"}}}}},
    # strings, numbers and literals at the root
    {"type": "string"}, {"type": "string", "minLength": 2, "maxLength": 5}, {"type": "string", "minLength": 3},
    {"type": "string", "maxLength": 0}, {"type": "string", "format": "date"}, {"type": "string", "format": "time"},
    {"type": "string", "format": "ipv4"}, {"type": "string", "format": "email"}, {"type": "string", "format": "uuid"},
    {"type": "string", "format": "date-time"}, {"type": "string", "format": "unknown-format"},
    {"type": "integer"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"},
    {"enum": ["a", 1, 12, -3, 0.5, None, True, {"k": [1, "x"]}, "é\n\u007f/", 1e20]},
    {"const": {"z": 1, "a": [True, None, 2.25]}},
    {"type": ["integer", "number", "null"]},
    {"type": ["string", "array"], "items": {"type": "integer"}},
    {"type": ["object", "null"], "properties": {"a": {"type": "integer"}}, "required": ["a"]},
    {"anyOf": [{"type": "string", "format": "date"}, {"type": "integer"}, {"type": "null"},
               {"type": "array", "items": {"type": "boolean"}}]},
    {"oneOf": [{"enum": ["a", "b"]}, {"const": "c"}, {"type": "boolean"}]},
    {"allOf": [{"type": "object", "properties": {"q": {"type": "string", "minLength": 1}}, "required": ["q"]}]},
    {"type": "object", "properties": {"n": {"type": "object", "properties": {"n": {"type": "object", "properties": {
        "n": {"type": "array", "items": {"type": "object", "properties": {"leaf": {"type": "integer"}}}}}}}}}},
]
CHOICES = [["yes", "no"], ["a", "ab", "abc"], ["x\ny", "😀", "\u0001", "\u007f", "a\"b\\c"], ["positive", "negative", "neutral"]]


class Gen:
    """Replies to a schema: mostly valid, with blanks near the limits and the near misses a matcher must refuse."""

    def __init__(self, rng: random.Random, root) -> None:
        self.r, self.root = rng, root
        self.clean = True                                 # no deliberate near miss

    def odd(self, p: float) -> bool:
        """A near miss with chance p, unless this reply is clean."""
        return not self.clean and self.r.random() < p

    def ws(self, wide: bool = False) -> bytes:
        k = self.r.random()
        n = 0 if k < 0.6 else self.r.randint(1, 3) if k < 0.88 else self.r.choice([31, 32, 33] + ([63, 64, 65] if wide else []))
        return bytes(self.r.choice(BLANKS) if self.r.random() < 0.3 else 0x20 for _ in range(n))

    def resolve(self, s):
        while isinstance(s, dict) and "$ref" in s:
            uri, s = s["$ref"], self.root
            for part in uri[2:].split("/") if uri != "#" else []:
                if part:
                    s = s[part]
        return s

    def value(self, s, depth: int = 0) -> bytes:
        r = self.r
        s = self.resolve(s)
        if s is True or s == {} or depth > 6:
            return self.anything(depth)
        if "const" in s:
            return literal(s["const"])
        if "enum" in s:
            return literal(r.choice(s["enum"]))
        for word in ("anyOf", "oneOf"):
            if word in s:
                return self.value(r.choice(s[word]), depth)
        if "allOf" in s:
            return self.value(s["allOf"][0], depth)
        t = s.get("type")
        if isinstance(t, list):
            t = r.choice(t) if t else None
        if t is None:
            t = "object" if "properties" in s or "additionalProperties" in s else "array" if "items" in s or "prefixItems" in s else None
        if self.odd(0.04):
            return self.anything(depth)          # a value of another type
        if t == "object":
            return self.obj(s, depth)
        if t == "array":
            return self.arr(s, depth)
        if t == "string":
            return self.string(s)
        if t == "integer":
            return self.number(True)
        if t == "number":
            return self.number(False)
        if t == "boolean":
            return r.choice([b"true", b"false", b"tru", b"True"] if self.odd(0.5) else [b"true", b"false"])
        if t == "null":
            return r.choice([b"nul", b"None"]) if self.odd(0.3) else b"null"
        return self.anything(depth)

    def anything(self, depth: int) -> bytes:
        r = self.r
        k = r.random()
        if depth < 4 and k < 0.2:
            return self.obj({"type": "object", "additionalProperties": True}, depth)
        if depth < 4 and k < 0.35:
            return self.arr({"type": "array", "items": {}}, depth)
        if k < 0.6:
            return self.string({})
        if k < 0.85:
            return self.number(False)
        return r.choice([b"true", b"false", b"null"])

    def key(self, name: str) -> bytes:
        if self.odd(0.03):
            return b'"' + name.encode() + b'x"'
        return literal(name)

    def obj(self, s, depth: int) -> bytes:
        r = self.r
        props = list((s.get("properties") or {}).items())
        required = set(s.get("required") or [])
        plain = all(k in ("type", "title", "description") for k in s)
        addl = s.get("additionalProperties", plain)
        members = []
        for name, sub in props:
            if name in required or r.random() < 0.5:
                members.append((name, sub))
        if self.odd(0.08) and len(members) > 1:      # out of order
            i, j = r.sample(range(len(members)), 2)
            members[i], members[j] = members[j], members[i]
        if self.odd(0.06) and members:               # a required one left out
            members.pop(r.randrange(len(members)))
        extra = 0 if addl is False and not self.odd(0.1) else r.choice([0, 0, 1, 2])
        for _ in range(extra):
            name = r.choice(["zz", "a", "ab", "abcd", "", "\\u0061", "k" * r.randint(1, 4), "é"])
            members.append((name, addl if isinstance(addl, dict) else {}))
        parts = [self.key(n) + self.ws() + b":" + self.ws() + self.value(sub, depth + 1) for n, sub in members]
        body = (self.ws() + b"," + self.ws()).join(parts)
        return b"{" + self.ws() + body + self.ws() + b"}"

    def arr(self, s, depth: int) -> bytes:
        r = self.r
        prefix = s.get("prefixItems") or []
        items = s.get("items", "items" not in s and not prefix and all(k in ("type",) for k in s))
        lo = s.get("minItems", 0)
        hi = s.get("maxItems", lo + 4)
        n = r.randint(max(0, lo - 1), hi + 1) if self.odd(0.5) else r.randint(lo, hi)
        out = []
        for i in range(n):
            sub = prefix[i] if i < len(prefix) else items if isinstance(items, dict) else {} if items else {"type": "null"}
            out.append(self.value(sub, depth + 1))
        body = (self.ws() + b"," + self.ws()).join(out)
        return b"[" + self.ws(wide=bool(prefix)) + body + self.ws(wide=bool(prefix)) + b"]"

    def string(self, s) -> bytes:
        r = self.r
        fmt = s.get("format")
        if fmt in SAMPLES and r.random() < 0.8:
            text = r.choice(SAMPLES[fmt]) if self.odd(0.3) else self.formatted(fmt)
            if self.odd(0.2):
                i = r.randrange(len(text) + 1)
                text = text[:i] + r.choice(["", "x", "9", "-", ":", "."]) + text[i + 1:]
            return b'"' + text.encode() + b'"'
        lo, hi = s.get("minLength", 0), s.get("maxLength", 6)
        n = r.randint(max(0, lo - 1), hi + 1) if self.odd(0.5) else r.randint(lo, hi)
        plain = "minLength" not in s and "maxLength" not in s
        chars = []
        for _ in range(n):
            k = r.random()
            if k < 0.75:
                chars.append(bytes([r.choice([c for c in range(0x20, 0x7F) if c not in (0x22, 0x5C)])]))
            elif k < 0.85 and (plain or self.odd(0.5)):
                chars.append(r.choice([b"\\n", b"\\\"", b"\\u00e9", b"\\/"] + ([b"\\x"] if self.odd(0.5) else [])))
            elif k < 0.95:
                chars.append(chr(r.choice([0xE9, 0x4E2D, 0x1F600, 0x2028])).encode())
            elif not plain or self.odd(0.5):
                chars.append(bytes([r.choice([0x01, 0x09, 0x7F] + ([0x0A, 0x0D] if self.odd(0.5) else []))]))
        return b'"' + b"".join(chars) + b'"'

    def formatted(self, fmt: str) -> str:
        """A random text the format takes."""
        r = self.r
        d2 = lambda lo, hi: f"{r.randint(lo, hi):02d}"
        date = f"{r.randint(0, 9999):04d}-{d2(1, 12)}-{d2(1, 31)}"
        frac = "." + str(r.randint(0, 999)) if r.random() < 0.3 else ""
        zone = "Z" if r.random() < 0.5 else r.choice("+-") + d2(0, 23) + ":" + d2(0, 59)
        time = f"{d2(0, 23)}:{d2(0, 59)}:{d2(0, 60)}{frac}{zone}"
        hexs = lambda n: "".join(r.choice("0123456789abcdefABCDEF") for _ in range(n))
        word = lambda chars, n: "".join(r.choice(chars) for _ in range(n))
        alnum = "abcxyzABC019"
        if fmt == "date":
            return date
        if fmt == "time":
            return time
        if fmt == "date-time":
            return date + "T" + time
        if fmt == "uuid":
            return "-".join(hexs(n) for n in (8, 4, 4, 4, 12))
        if fmt == "ipv4":
            return ".".join(r.choice([str(r.randint(0, 255)), f"{r.randint(0, 99):02d}", f"{r.randint(0, 199):03d}"])
                            for _ in range(4))
        local = ".".join(word(alnum + "!#$%&'*+/=?^_`{|}~-", r.randint(1, 4)) for _ in range(r.randint(1, 2)))
        if r.random() < 0.15:
            local = '\\"' + word("ab ,@", r.randint(0, 3)) + '\\"'
        labels = [word(alnum, 1) + (word(alnum + "-", r.randint(0, 3)) + word(alnum, 1) if r.random() < 0.5 else "")]
        labels += [word(alnum, 1) + word(alnum + "-", r.randint(0, 2)) + word(alnum, 1) for _ in range(r.randint(0, 2))]
        return local + "@" + ".".join(labels)

    def number(self, integer: bool) -> bytes:
        r = self.r
        if not self.odd(0.15):
            text = r.choice(["", "-"]) + r.choice(["0", str(r.randint(1, 999))])
            if integer and text == "-0":
                text = "0"
            if not integer and r.random() < 0.4:
                text += "." + str(r.randint(0, 99))
            if not integer and r.random() < 0.2:
                text += r.choice("eE") + r.choice(["", "+", "-"]) + str(r.randint(0, 20))
            return text.encode()
        return r.choice([b"01", b"-0", b"1.", b".5", b"+1", b"1e5", b"1.5", b"-", b"2E-3"])


SAMPLES = {
    "date": ["2026-10-26", "1999-12-31", "2026-13-01", "2026-02-30", "26-10-26"],
    "time": ["08:47:00Z", "23:59:60.5+05:30", "24:00:00Z", "12:00:00"],
    "date-time": ["2026-10-26T08:47:00Z", "2026-10-26T08:47:00.123-07:00", "2026-10-26 08:47:00Z"],
    "uuid": ["123e4567-e89b-12d3-a456-426614174000", "123E4567-E89B-12D3-A456-42661417400", "zzze4567-e89b-12d3-a456-426614174000"],
    "ipv4": ["192.168.0.1", "255.255.255.255", "256.1.1.1", "0.01.001.9", "1.2.3"],
    "email": ["a.b+c@example.com", "a@b", "a@b.c", "\\\"x y\\\"@host.io", "a..b@x.io", "@x.io"],
}


def literal(v) -> bytes:
    """A value as xgrammar writes an enum or const (picojson): compact, \\u00XX for control bytes and DEL."""
    if isinstance(v, str):
        out = ['"']
        for ch in v:
            c = ord(ch)
            out.append({'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(
                ch, f"\\u{c:04x}" if c < 0x20 or c == 0x7F else ch))
        return ("".join(out) + '"').encode()
    if isinstance(v, bool) or v is None:
        return json.dumps(v).encode()
    if isinstance(v, int):
        return str(v).encode()
    if isinstance(v, float):
        return (("%.f" if abs(v) < 2 ** 53 and v == int(v) else "%.17g") % v).encode()
    if isinstance(v, list):
        return b"[" + b",".join(literal(x) for x in v) + b"]"
    return b"{" + b",".join(literal(k) + b":" + literal(x) for k, x in v.items()) + b"}"


def mutate(r: random.Random, data: bytes) -> bytes:
    """One random edit: insert, delete or replace a byte, or cut the string short."""
    if not data:
        return bytes([r.randint(0, 255)])
    i = r.randrange(len(data))
    pick = bytes([r.choice([r.randint(0, 255), r.choice(b'{}[]",: \n\\0e-.tunx')])])
    return r.choice([data[:i] + pick + data[i:], data[:i] + data[i + 1:], data[:i] + pick + data[i + 1:], data[:i]])


def compiler():
    import xgrammar as xgr

    vocab = [bytes([i]) for i in range(256)] + [b"<eos>"]
    info = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=len(vocab), stop_token_ids=[STOP])
    return xgr, xgr.GrammarCompiler(info, max_threads=1)


def verdict(xgr, compiled, data: bytes) -> str:
    """The first rejected byte's index, else c when the end of the reply is accepted, else p."""
    m = xgr.GrammarMatcher(compiled)
    for i, b in enumerate(data):
        if not m.accept_token(b):
            return str(i)
    return "c" if m.accept_token(STOP) else "p"


def encode(data: bytes) -> str:
    """Printable bytes as themselves, a space as _, every other byte (and % and _) as %XX."""
    return "".join("_" if b == 0x20 else chr(b) if 0x21 <= b <= 0x7E and b not in (0x25, 0x5F) else f"%{b:02X}"
                   for b in data)


def decode(payload: str) -> bytes:
    out, i = bytearray(), 0
    while i < len(payload):
        if payload[i] == "%":
            out.append(int(payload[i + 1:i + 3], 16))
            i += 3
        else:
            out.append(0x20 if payload[i] == "_" else ord(payload[i]))
            i += 1
    return bytes(out)


def sections():
    """(header, compile) for each schema and choice list, as Python 0.6 compiled them."""
    for s in SCHEMAS:
        text = json.dumps(s, ensure_ascii=False)
        yield "= schema " + text, s, lambda c, t=text: c.compile_json_schema(t, max_whitespace_cnt=BLANK_RUN)
    for ch in CHOICES:
        text = json.dumps(ch, ensure_ascii=False)
        rule = "root ::= " + " | ".join(json.dumps(v, ensure_ascii=False) for v in ch)
        yield "= choice " + text, ch, lambda c, g=rule: c.compile_grammar(g)


def main() -> int:
    """Generate, label with the installed xgrammar and write the corpus (or --check it against the frozen file)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=int, default=90, help="cases for each schema or choice list")
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--check", action="store_true", help="relabel the frozen file and report differences")
    args = p.parse_args()
    xgr, comp = compiler()
    version = __import__("importlib.metadata").metadata.version("xgrammar")
    if args.check:
        compiled, differ = None, 0
        builds = {h: build for h, _, build in sections()}
        for line in args.out.read_text().splitlines():
            if line.startswith("#"):
                continue
            if line.startswith("= "):
                compiled = builds[line](comp)
                continue
            want, _, payload = line.partition(" ")
            got = verdict(xgr, compiled, decode(payload))
            if got != want:
                differ += 1
                print(f"xgrammar {version}: {got}, frozen: {want}: {decode(payload)!r}")
        print(f"{differ} differences")
        return 1 if differ else 0
    r = random.Random(args.seed)
    lines = [f"# xgrammar {version} json_schema (max_whitespace_cnt={BLANK_RUN}) and guided_choice verdicts by "
             "tools/zig/xgrammar_schema_corpus.py",
             f"# seed {args.seed}: '= schema <json>' or '= choice <json>', then verdict and bytes per case"]
    total = 0
    for header, spec, build in sections():
        compiled = build(comp)
        lines.append(header)
        seen = set()
        gen = Gen(r, spec)
        tries = 0
        while len(seen) < args.cases and tries < args.cases * 50:
            tries += 1
            gen.clean = r.random() < 0.6
            if isinstance(spec, list):
                data = literal(r.choice(spec))[1:-1] if gen.odd(0.5) else r.choice(spec).encode()
            else:
                data = gen.value(spec)
            if r.random() < 0.35:
                data = mutate(r, data)
            if data in seen or SURROGATE.search(data) or len(data) > 400:
                continue
            seen.add(data)
            lines.append(f"{verdict(xgr, compiled, data)} {encode(data)}".rstrip(" "))
        total += len(seen)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"{total} cases in {len(SCHEMAS) + len(CHOICES)} sections written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
