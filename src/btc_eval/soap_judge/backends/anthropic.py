"""First-party Anthropic backend (the official ``anthropic`` SDK).

Runs the judge on Claude via api.anthropic.com. Defaults to adaptive thinking,
which suits the judging task, and **does not forward** ``temperature`` / ``top_p``
— current Opus models (4.8 / 4.7) reject sampling parameters with a 400, and
``temperature=0`` never guaranteed identical outputs anyway. Streaming is used so
large ``max_tokens`` values don't trip the SDK's non-streaming timeout guard.

Credentials come from ``ANTHROPIC_API_KEY`` (the SDK's default resolution).
Requires the ``anthropic`` SDK (``uv pip install '.[llm-anthropic]'``).

Recommended model id: ``claude-opus-4-8``.
"""

from __future__ import annotations

import logging

from btc_eval.soap_judge.backends import Message, register

log = logging.getLogger(__name__)


@register("anthropic")
class AnthropicBackend:
    """Judge backend backed by the first-party Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        thinking: str | None = "adaptive",
        **_,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' backend requires the 'anthropic' SDK. "
                "Install with:  uv pip install '.[llm-anthropic]'"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        # "adaptive" (default), "disabled", or None to omit the field entirely.
        self._thinking = thinking

    def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 8000,
    ) -> str:
        # temperature / top_p are intentionally dropped — Opus 4.8 / 4.7 return 400 on them.
        kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if self._thinking:
            kwargs["thinking"] = {"type": self._thinking}

        # Stream so large max_tokens values don't hit the SDK's non-streaming guard.
        with self._client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
        return next((b.text for b in message.content if b.type == "text"), "")
