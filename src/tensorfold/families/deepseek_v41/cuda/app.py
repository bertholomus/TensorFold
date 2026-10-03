"""OpenAI routes for the DeepSeek-V4.1 engine: TensorFold's CUDA App, with ignore_eos passed to the engine."""

from __future__ import annotations

from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest


class DsApp(App):
    reads_ignore_eos = True

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        return super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)
