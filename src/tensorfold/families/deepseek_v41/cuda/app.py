"""OpenAI routes for the DeepSeek-V4.1 engine: TensorFold's CUDA App, with ignore_eos passed to the engine, image
input (--vision) laid out as DeepSeek's encoding lays it out, and every reasoning_effort name mapped to DeepSeek's
numeric effort."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.server.errors import RequestError


# reasoning_effort -> the "Reasoning Effort: N" (1-100) line DeepSeek's encoding puts before the first turn when
# thinking. DeepSeek names low / high / max (50 / 75 / 100); minimal, medium and xhigh sit between them, so every
# OpenAI tier reaches the model as its own value (none or off: thinking off). No effort sent: high, as the reference.
EFFORT_VALUES = {"minimal": 25, "low": 50, "medium": 62, "high": 75, "xhigh": 88, "max": 100}


class DsApp(App):
    reads_ignore_eos = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        template = self.template
        template.efforts = frozenset(EFFORT_VALUES)     # the server passes every name through unchanged
        render = template.render

        def render_effort(messages, *, enable_thinking: bool, extra: dict[str, Any] | None = None, **kw: Any) -> str:
            extra = dict(extra or {})
            if enable_thinking:
                name = extra.get("reasoning_effort") or "high"
                extra["reasoning_effort"] = EFFORT_VALUES.get(name, name)
            return render(messages, enable_thinking=enable_thinking, extra=extra, **kw)

        template.render = render_effort

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        return super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)

    def _prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        """A chat with image_url parts (user turns and tool results): each image becomes DeepSeek's placeholder
        block, the prompt renders as text, and each placeholder token grows into the image's span."""

        vision = getattr(self.engine, "vision", None)
        messages = body.get("messages") if chat else None
        if vision is None or not isinstance(messages, list) or not vision.has_images(messages):
            return super()._prepare(body, chat)
        try:
            text_messages, pictures = vision.split(messages)
        except (ValueError, OSError) as exc:
            raise RequestError(f"image input: {exc}") from exc
        prepared = super()._prepare({**body, "messages": text_messages}, chat)
        try:
            vp = vision.expand(prepared.prompt, pictures)
        except ValueError as exc:
            raise RequestError(f"image input: {exc}") from exc
        return dataclasses.replace(prepared, prompt=vp.token_ids, vision=vp,
                                   sampling=self.sampling_for(body, vp.token_ids))
