"""Ling-3.1-flash's request routes: the context check before a stream opens, and each request's draft policy and
stop-at-EOS handed to the engine.

The checkpoint's own template renders unchanged (its ``detailed thinking on|off`` system line, the ``<think>`` or
``<think></think>`` generation prompt, think blocks kept in history), and the shared reply parsers read what Ling
writes: reasoning up to ``</think>`` (a finished tool call before it also ends the reasoning, as vLLM's ling3 parser
reads it) and GLM-style calls, ``<tool_call>NAME`` then ``<arg_key>K</arg_key><arg_value>V</arg_value>`` pairs,
values typed by the tool's schema. tests/test_bailing_12_serving.py holds both to the checkpoint's renders."""

from __future__ import annotations

from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest, RequestError
from tensorfold.server.errors import CONTEXT_LIMIT


class LingApp(App):
    reads_ignore_eos = True             # ``run`` hands it to the engine's request

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Validate the rendered prompt plus max_tokens against the engine context limit before streaming."""

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
        # OpenAI's wording, so prepare refuses it as context_length_exceeded (clients compact on it)
        return (f"{CONTEXT_LIMIT} {limit} tokens: this request needs a {need}-token context ({detail}), which exceeds "
                f"the context window this server was started for; shorten the prompt or reply"
                f"{self._restart(need, ' every rank')}")

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        model = str(body.get("model") or "")
        self.engine.request.policy = body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        return super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)
