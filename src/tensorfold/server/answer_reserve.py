"""The answer reserve: a reply that thinks keeps part of max_tokens for its answer.

The server closes a think block the reply leaves open at max_tokens minus the reserve, where the reserve is
``ceil(ratio * max_tokens)``, at least ``minimum`` tokens and at most half of max_tokens (the half wins over the
floor): max_tokens 128 thinks for 64 and keeps 64 for the close and the answer. A request's smaller
``thinking_budget`` wins; a larger one, or one that asks for no limit, is held to the same cap. After that close, an
answer that would end before its first word (an end token, alone or after blanks) starts from ``LEAD`` instead, once
(``engine.call_gate.AnswerLead``), so a cut reply always answers. ``ratio`` 0 turns all of it off: the thinking budget
is then the request's (or ``--thinking-budget``) alone, as before the reserve."""

from __future__ import annotations

import math

# the words an answer that would be empty after the server's close starts from
LEAD = "Answer:"
# (ratio, minimum) where a family names none: off
DEFAULT = (0.0, 1024)


def reserve(max_tokens: int, ratio: float, minimum: int) -> int:
    """Reply tokens a thinking reply keeps for its close and answer (0: none)."""

    if ratio <= 0 or max_tokens < 2:
        return 0
    return min(max(math.ceil(round(ratio * max_tokens, 6)), int(minimum)), int(max_tokens) // 2)


def think_budget(asked: int, max_tokens: int, ratio: float, minimum: int) -> int:
    """The thinking budget a reply runs with: ``asked`` (0: none asked; below 0: no limit asked) held to max_tokens
    minus the reserve; ``asked`` itself where there is no reserve."""

    kept = reserve(max_tokens, ratio, minimum)
    if kept <= 0:
        return int(asked)
    cap = int(max_tokens) - kept
    return cap if asked <= 0 else min(int(asked), cap)


def note(ratio: float, minimum: int, max_tokens: int) -> str | None:
    """The startup line naming the reserve a server runs with (None: off)."""

    if ratio <= 0:
        return None
    return (f"[tensorfold] answer reserve: a reply still thinking at max_tokens minus {ratio:g} of it (at least "
            f"{minimum} tokens, at most half) has its think block closed, so its answer has room; a request without "
            f"max_tokens ({max_tokens}) thinks for up to {think_budget(0, max_tokens, ratio, minimum)} "
            "(--answer-reserve 0: off)")


def settings(family_default: tuple[float, int] | None, ratio: float | None, minimum: int | None) -> tuple[float, int]:
    """(ratio, minimum) a server runs with: the flags where given, else the family's, else off; ValueError if bad."""

    base_ratio, base_minimum = family_default or DEFAULT
    ratio = float(base_ratio if ratio is None else ratio)
    minimum = int(base_minimum if minimum is None else minimum)
    if not 0.0 <= ratio <= 0.5:
        raise ValueError(f"--answer-reserve {ratio}: a share of max_tokens from 0 (off) to 0.5")
    if minimum < 0:
        raise ValueError(f"--answer-reserve-min {minimum}: a token count, 0 or more")
    return ratio, minimum


__all__ = ["DEFAULT", "LEAD", "note", "reserve", "settings", "think_budget"]
