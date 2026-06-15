"""Local / self-hosted backend via the OpenAI-compatible chat API.

Targets Ollama's ``/v1/chat/completions`` endpoint by default, but works
unchanged against any OpenAI-compatible server (vLLM, LM Studio, llama.cpp
``server``, text-generation-webui) -- just point ``--host`` / ``$OLLAMA_HOST``
at it. This is the default judge backend: free, private, no API key required,
and reproducible-ish at ``temperature=0``.

Requires ``requests`` (``uv pip install '.[llm-judge]'``).
"""

from __future__ import annotations

import json
import os

from btc_eval.soap_judge.backends import Message, call_with_retry, register

DEFAULT_HOST = "http://localhost:11434"


@register("ollama")
class OllamaBackend:
    """OpenAI-compatible chat backend (Ollama and friends)."""

    def __init__(
        self,
        host: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        **_,
    ) -> None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - exercised via install hint
            raise ImportError(
                "The 'ollama' backend requires 'requests'. "
                "Install with:  uv pip install '.[llm-judge]'"
            ) from exc
        self._requests = requests
        base = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.url = f"{base}/v1/chat/completions"
        # Ollama ignores the key but the OpenAI client shape expects one.
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY") or "ollama"
        # Per-call HTTP timeout. Long judge generations on a busy GPU can exceed
        # the old 600s default; allow override via $OLLAMA_TIMEOUT (seconds).
        if timeout is None:
            timeout = float(os.environ.get("OLLAMA_TIMEOUT", "1800"))
        self.timeout = timeout

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
        payload = {
            "model": model,
            "messages": final,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        def _post() -> dict:
            resp = self._requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        data = call_with_retry(_post, retry_on=(self._requests.RequestException,))
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            # Surface an unexpected shape as text so the parser can log it.
            return json.dumps(data)
