"""OpenRouter backend (hosted models behind one API).

Ported from the original ``OpenRouterClient.generate_response`` minus the batch
plumbing and debug prints. Reads ``OPENROUTER_API_KEY`` from the environment;
an optional ``provider_name`` pins a specific upstream provider.

Requires ``requests`` (``uv pip install '.[llm-judge]'``).
"""

from __future__ import annotations

import json
import logging
import os

from btc_eval.soap_judge.backends import Message, call_with_retry, register

log = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# Reasoning models need more headroom; bump the token budget when the model id
# advertises one of these families (matches the original client's heuristic).
_THINKING_HINTS = ("z-ai", "glm", "kimi", "moonshot", "thinking")


@register("openrouter")
class OpenRouterBackend:
    """Hosted chat completions via OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        provider_name: str | None = None,
        site_url: str = "",
        site_name: str = "",
        timeout: float = 600.0,
        **_,
    ) -> None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openrouter' backend requires 'requests'. "
                "Install with:  uv pip install '.[llm-judge]'"
            ) from exc
        self._requests = requests
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            log.warning(
                "OPENROUTER_API_KEY not set; OpenRouter requests will fail until it is."
            )
        self.provider_name = provider_name
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }

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
        final: list[Message] = []
        if system:
            final.append({"role": "system", "content": system})
        final.extend(messages)

        req_tokens = max(max_tokens, 4000)
        if any(hint in model for hint in _THINKING_HINTS):
            req_tokens = max(req_tokens, 8000)

        payload: dict = {
            "model": model,
            "messages": final,
            "max_tokens": req_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self.provider_name:
            payload["provider"] = {
                "order": [self.provider_name],
                "allow_fallbacks": False,
            }

        def _post() -> dict:
            resp = self._requests.post(
                BASE_URL, headers=self.headers, data=json.dumps(payload),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        data = call_with_retry(_post, retry_on=(self._requests.RequestException,))
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            log.warning("Unexpected OpenRouter response shape: %s", data)
            return json.dumps(data)
