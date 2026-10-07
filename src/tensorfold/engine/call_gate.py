"""A forced tool call opens a call; a thinking budget closes the think block, alike in every round kind."""

from __future__ import annotations

from typing import Any, Callable, Sequence

_ANSWER, _LEAD, _NAME, _DONE = range(4)


def call_format(rendered: str, name: str, openers: Sequence[str]) -> tuple[str, str, str] | None:
    """(opener, lead, tail) of the template's call to ``name``: the text before the name, and the mark that ends it."""

    at = rendered.rfind(name)
    starts = [(rendered.rfind(opener, 0, at), opener) for opener in openers]
    start, opener = max(starts, default=(-1, ""))
    if at < 0 or start < 0:
        return None
    end = at + len(name)
    return opener, rendered[start + len(opener):at], rendered[end:end + 1] if end < len(rendered) else ""


class CallGate:
    """Outside a think block the answer opens a call to an offered tool; each fix reads only earlier tokens."""

    # a breaking token becomes the opener (then the lead), or the rest of the offered name its prefix starts

    def __init__(self, opener: int, blank: Callable[[int], bool], *, think_open: int = -1, think_end: int = -1,
                 armed: bool = True, text: Callable[[int], str] | None = None,
                 encode: Callable[[str], list[int]] | None = None, lead: str = "", names: Sequence[str] = (),
                 tail: str = "") -> None:
        self.opener, self.blank = int(opener), blank
        self.think_open, self.think_end = int(think_open), int(think_end)
        self.text, self.encode = text, encode
        self.lead, self.names, self.tail = lead, [n for n in names if n], tail
        # (armed, phase, text of the lead or name so far); unarmed: the prompt left a think block open
        self.state: tuple[bool, int, str] = (bool(armed), _ANSWER, "")

    @classmethod
    def after_prompt(cls, prompt: Sequence[int], opener: int, blank: Callable[[int], bool], *, think_open: int = -1,
                     think_end: int = -1, **named: Any) -> "CallGate":
        """A gate held while ``prompt`` leaves a think block open (its last opener after its last end)."""

        last = {int(t): i for i, t in enumerate(prompt) if int(t) in (think_open, think_end)}
        held = min(think_open, think_end) >= 0 and last.get(think_open, -1) > last.get(think_end, -1)
        return cls(opener, blank, think_open=think_open, think_end=think_end, armed=not held, **named)

    @property
    def done(self) -> bool:
        return self.state[1] == _DONE

    @property
    def watching(self) -> bool:
        """Whether the next token can be cut (the one-token paths read it only then)."""

        armed, phase, _ = self.state
        return phase in (_LEAD, _NAME) or (phase == _ANSWER and armed)

    def _ends(self, char: str) -> bool:
        return char.isspace() or (char == self.tail if self.tail else char in "<>{}\"'(),")

    def _step(self, state: tuple[bool, int, str], token: int) -> tuple[tuple[bool, int, str], list[int] | None]:
        """(the state after ``token``, or the tokens that replace it when it breaks the call)."""

        armed, phase, seen = state
        if phase == _ANSWER:
            if not armed:
                return (token == self.think_end, _ANSWER, ""), None
            if token == self.think_open:
                return (False, _ANSWER, ""), None
            if token == self.opener:
                return (True, _LEAD, ""), None
            if self.blank(token):
                return state, None
            fix = self.encode(self.lead) if self.encode is not None and self.names and self.lead else []
            return state, [self.opener, *fix]
        if phase == _DONE or self.text is None or not self.names:
            return (armed, _DONE, ""), None
        written, owed = seen + self.text(token), ""       # owed: the lead this token was due to finish
        if phase == _LEAD:
            if self.lead.startswith(written):
                full = len(written) == len(self.lead)
                return (armed, _NAME if full else _LEAD, "" if full else written), None
            if not written.startswith(self.lead):
                return state, self.encode(self.lead[len(seen):])
            owed, seen, written = self.lead[len(seen):], "", written[len(self.lead):]
        end = next((i for i, c in enumerate(written) if self._ends(c)), len(written))
        if end == len(written) and any(n.startswith(written) for n in self.names):
            return (armed, _NAME, written), None
        if end < len(written) and written[:end] in self.names:
            return (armed, _DONE, ""), None
        name = next((n for n in self.names if n.startswith(seen)), None)
        fix = [] if name is None or self.encode is None else self.encode(owed + name[len(seen):] + self.tail)
        return ((armed, _DONE, ""), None) if not fix else (state, fix)

    def cut(self, tokens: Sequence[int]) -> tuple[int, list[int]] | None:
        """(index in the next committed ``tokens`` a fix replaces, the fix: its first token there, the rest forced)."""

        state = self.state
        for i, token in enumerate(int(t) for t in tokens):
            if state[1] == _DONE:
                return None
            state, fix = self._step(state, token)
            if fix:
                return i, fix
        return None

    def observe(self, token: int) -> None:
        """Follow a committed token (a fix's tokens included): the call's name ends the gate's work."""

        if not self.done:
            state, fix = self._step(self.state, int(token))
            self.state = state if fix is None else (state[0], _DONE, "")     # off script: stop constraining


class ThinkBudget:
    """The lane engine's thinking budget: the ``budget``-th token of a reply still thinking becomes ``close``."""

    def __init__(self, budget: int, close: Sequence[int], think_end: int) -> None:
        self.budget, self.close, self.think_end = int(budget), [int(t) for t in close], int(think_end)
        self.count, self.open = 0, True

    def cut(self, tokens: Sequence[int]) -> tuple[int, list[int]] | None:
        """(index in the next committed ``tokens`` the close replaces, the close), as ``CallGate.cut``."""

        if not self.open:
            return None
        for i, token in enumerate(tokens):
            if self.count + i + 1 >= self.budget:
                return i, list(self.close)
            if int(token) == self.think_end:
                return None
        return None

    def observe(self, token: int) -> None:
        self.count += 1
        self.open = self.open and int(token) != self.think_end


class ThinkLoop:
    """A reasoning loop guard: a reply still thinking whose reasoning stopped saying anything new gets ``close`` (the
    thinking budget's: a newline, </think>, a blank line) and goes on to its answer.

    "Nothing new": in ``windows`` windows of ``width`` reply tokens in a row, fewer than ``least`` of the window's token
    ``n``-grams were not seen in any earlier window (GitHub #9's word-8-gram novelty signal, on token ids). A reply
    looping on the same few lines (GitHub #6: "OK. / Writing. / Let me write." to the max_tokens cap) has ~0.002; a
    reasoning that moves on ~0.9. The decision is a function of the committed tokens alone."""

    def __init__(self, close: Sequence[int], think_end: int, width: int = 1024, n: int = 8, least: float = 0.02,
                 windows: int = 3) -> None:
        self.close, self.think_end = [int(t) for t in close], int(think_end)
        self.width, self.n, self.least, self.windows = int(width), int(n), float(least), int(windows)
        self.open = True
        self.seen: set = set()                   # every earlier window's n-grams
        self.tail: list[int] = []                # the n - 1 tokens before this window (n-grams across its edge)
        self.win: list[int] = []                 # this window's tokens
        self.dry = 0                             # windows in a row with less than ``least`` new
        self.fired = False

    def _novelty(self, toks: Sequence[int]) -> float:
        seq = [*self.tail, *toks]
        grams = [tuple(seq[j:j + self.n]) for j in range(len(seq) - self.n + 1)]
        return sum(1 for g in grams if g not in self.seen) / max(1, len(grams))

    def cut(self, tokens: Sequence[int]) -> tuple[int, list[int]] | None:
        """(index in the next committed ``tokens`` the close goes after, the close), as ``CallGate.cut``."""

        if not self.open:
            return None
        need = self.width - len(self.win)        # tokens until this window ends
        for i, token in enumerate(tokens):
            if int(token) == self.think_end:
                return None
            if i + 1 == need:
                if self.dry + 1 >= self.windows and self._novelty([*self.win, *tokens[:need]]) < self.least:
                    return i + 1, list(self.close)
                return None
        return None

    def observe(self, token: int) -> None:
        if not self.open:
            return
        if int(token) == self.think_end:
            self.open = False
            return
        self.win.append(int(token))
        if len(self.win) < self.width:
            return
        nov = self._novelty(self.win)
        seq = [*self.tail, *self.win]
        self.seen.update(tuple(seq[j:j + self.n]) for j in range(len(seq) - self.n + 1))
        self.tail, self.win = seq[-(self.n - 1):], []
        self.dry = self.dry + 1 if nov < self.least else 0
        if self.dry >= self.windows:                  # the cut's close follows: its </think> closes the gate
            self.fired = True


def generate_gated(generate: Callable[[list[int], int, Callable[[list[int]], bool]], Any], prompt: Sequence[int],
                   max_tokens: int, gates: Sequence[Any], on_tokens: Callable[[list[int]], bool]) -> Any:
    """Decode with ``gates``: at a cut go on from the prompt, reply and fix, or (a ``replay`` gate) run the prompt again."""

    reply: list[int] = []
    owed: list[int] = []                         # a replay's tokens already sent (a ``replay`` gate's cut)
    state = {"cut": False, "stopped": False, "replay": False}

    def take(new: list[int]) -> bool:
        if state["cut"] or state["stopped"]:
            return True                          # an engine that decodes on after a stop: the rest is not the reply
        if owed:                                 # a replay writes the tokens it sent before again first: not resent
            k = min(len(owed), len(new))
            if list(new[:k]) != owed[:k]:
                raise RuntimeError("a replay after a yield differs from the reply it sent")
            del owed[:k]
            new = list(new[k:])
            if not new:
                return False
        hits = [(hit, getattr(gate, "replay", False)) for gate in gates if (hit := gate.cut(new)) is not None]
        if hits:
            (at, fix), replay = min(hits, key=lambda h: (h[0][0], h[1]))   # the earliest; a fix before a replay
            new, state["cut"], state["replay"] = [*new[:at], *fix][:max_tokens - len(reply)], True, replay
        for token in new:
            for gate in gates:
                gate.observe(token)
        reply.extend(new)
        state["stopped"] = bool(on_tokens(new))
        return state["stopped"] or state["cut"]

    anchor = 0                                   # reply tokens in the current run's prompt
    runs = [generate(list(prompt), max_tokens, take)]
    while state["cut"] and not state["stopped"] and len(reply) < max_tokens:
        state["cut"] = False
        if state["replay"]:                      # from the same prompt again: its reply comes back token for token
            owed[:] = reply[anchor:]
        else:                                    # on from the reply and the fix
            anchor = len(reply)
        runs.append(generate([*prompt, *reply[:anchor]], max_tokens - anchor, take))
    stats = dict(runs[0] or {})
    for run in runs[1:]:
        for key, value in (run or {}).items():
            number = isinstance(value, (int, float)) and not isinstance(value, bool)
            summed = number and isinstance(stats.get(key), (int, float))
            stats[key] = stats[key] + value if summed else stats.get(key, value)     # times and rounds of every run
    return stats


__all__ = ["CallGate", "ThinkBudget", "ThinkLoop", "call_format", "generate_gated"]
