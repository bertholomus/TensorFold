"""Beam-1.0's request routes: the context check before a stream opens, and replies split by their channels.

The checkpoint's own template renders unchanged: the identity line, ``Reasoning effort: N`` from the request's
reasoning_effort (minimal 8 ... max; 512 when none is heard), tools as a developer message. A reply is a run of
Harmony-style messages after the prompt's ``<|start|>assistant``: a header up to ``<|message|>`` (``<|channel|>NAME``,
a recipient `` to=TOOL`` for a call, ``<|constrain|>xml``), then a body up to ``<|end|>``, ``<|return|>``, ``<|call|>``
or the next ``<|start|>``. Analysis bodies are the reasoning, final bodies the content, and a call's body holds
``<rfl:parameter name="K">`` + newline + value + ``\\n</rfl:parameter>`` per argument (Reflection's harmony_xml grammar,
tokenizer_config.json's response_template). Calls go into the answer as the shared parsers' markup,
``<tool_call><function=NAME><parameter=K>`` + value + ``</parameter></function></tool_call>``, so those parsers type the
values by the tool's schema, hide them from the streamed content and stream their arguments."""

from __future__ import annotations

import re
from typing import Any

from tensorfold.cuda.server import App, PreparedRequest, RequestError
from tensorfold.server.errors import CONTEXT_LIMIT

START, END, MESSAGE, CONSTRAIN = "<|start|>", "<|end|>", "<|message|>", "<|constrain|>"
RETURN, CALL = "<|return|>", "<|call|>"
_BODY_ENDS = (END, RETURN, CALL, START, CONSTRAIN)
_CHANNEL = re.compile(r"<\|channel\|>\s*([A-Za-z_]+)")
_RECIPIENT = re.compile(r"\bto=([^\s<]+)")
PARAM, PARAM_END = '<rfl:parameter name="', "\n</rfl:parameter>"


def _held(text: str) -> int:
    """How many characters at the end of ``text`` could begin a body's terminator (held while the reply streams)."""

    return max((k for tag in _BODY_ENDS for k in range(min(len(tag) - 1, len(text)), 0, -1)
                if text.endswith(tag[:k])), default=0)


def call_arguments(body: str, complete: bool) -> list[tuple[str, str]]:
    """A call body's (name, value) pairs: values whose close was written, and with ``complete`` the one the call
    ended in."""

    out: list[tuple[str, str]] = []
    at = body.find(PARAM)
    while at >= 0:
        k0 = at + len(PARAM)
        k1 = body.find('">', k0)
        if k1 < 0:
            break
        key, v0 = body[k0:k1], k1 + 2
        if body.startswith("\n", v0):
            v0 += 1
        nxt, close = body.find(PARAM, v0), body.find(PARAM_END, v0)
        if close >= 0 and (nxt < 0 or close < nxt):
            out.append((key, body[v0:close]))
            at = body.find(PARAM, close + len(PARAM_END))
            continue
        if nxt >= 0:                     # the next argument opened without a close: this value ends before it
            value = body[v0:nxt]
            out.append((key, value[:-1] if value.endswith("\n") else value))
            at = nxt
            continue
        if complete:
            value = body[v0:]
            out.append((key, value[:-1] if value.endswith("\n") else value))
        break
    return out


def split_channels(text: str, finished: bool) -> tuple[str, str]:
    """(reasoning, answer) of a Beam reply's text so far; the answer carries content and calls in order."""

    reasoning: list[str] = []
    answer: list[str] = []
    pos = 0
    while pos < len(text):
        m = text.find(MESSAGE, pos)
        if m < 0:                        # a header still being written, or text outside the format
            if finished and not text.startswith(START, pos) and "<|channel|>" not in text[pos:]:
                answer.append(text[pos:])
            break
        header = text[pos:m]
        b0 = m + len(MESSAGE)
        ends = [at for at in (text.find(t, b0) for t in _BODY_ENDS) if at >= 0]
        b1 = min(ends) if ends else len(text)
        closed = bool(ends) or finished
        body = text[b0:b1]
        if not closed:
            body = body[:len(body) - _held(body)]
        recipient = _RECIPIENT.search(header)
        channel = _CHANNEL.search(header)
        if recipient is not None:
            name = recipient.group(1)
            name = name[len("functions."):] if name.startswith("functions.") else name
            args = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in call_arguments(body, closed))
            answer.append(f"<tool_call>\n<function={name}>\n{args}" + ("</function>\n</tool_call>" if closed else ""))
        elif channel is not None and channel.group(1) == "final":
            answer.append(body)
        else:
            reasoning.append(body)
        pos = b1
        for t in (END, RETURN, CALL, CONSTRAIN):
            if text.startswith(t, pos):
                pos += len(t)
                break
    return "\n".join(r for r in reasoning if r) if len(reasoning) > 1 else "".join(reasoning), "".join(answer)


class BeamApp(App):
    def run(self, body: dict[str, Any], chat: bool, emit, *, prepared: PreparedRequest | None = None,
            cancelled=None) -> dict[str, Any]:
        """A completion may ask vLLM's ``prompt_logprobs`` (top-K a prompt position, the prompt's own token too):
        the engine scores its prompt chunks in the same prefill."""

        k = 0 if chat else body.get("prompt_logprobs") or 0
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 1024:
            raise RequestError("prompt_logprobs must be an integer from 0 to 1024")
        if k and body.get("stream"):
            raise RequestError("prompt_logprobs come with a whole (unstreamed) reply")
        self.engine.request.prompt_logprobs = k
        try:
            result = super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)
        finally:
            self.engine.request.prompt_logprobs = 0
        if k:
            stats = dict(result.get("stats") or {})
            result = {**result, "prompt_logprobs": stats.pop("prompt_logprobs", None), "stats": stats}
        return result

    def split_reply(self, raw: str, *, chat: bool, thinking: bool, finished: bool) -> tuple[str, str]:
        """A chat reply's channels, thinking or not (Beam always writes its analysis channel first)."""

        if not chat:
            return "", raw
        return split_channels(raw, finished)

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """The rendered prompt plus max_tokens against the KV pool's window, before a stream opens."""

        problem = self._check_fields(body)
        limit = getattr(self.engine, "limit", None)
        if problem or limit is None:
            return problem or super().check(body, prepared=prepared)
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        prompt = len(prepared.prompt)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        need = prompt + (int(asked) if asked else 1)
        if need <= limit:
            return super().check(body, prepared=prepared)
        detail = f"{prompt} prompt tokens plus max_tokens {int(asked)}" if asked else f"a {prompt}-token prompt"
        return (f"{CONTEXT_LIMIT} {limit} tokens: this request needs a {need}-token context ({detail}), which exceeds "
                f"the context window this server was started for; shorten the prompt or reply")
