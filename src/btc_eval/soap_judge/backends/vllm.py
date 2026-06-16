"""vLLM backend via its OpenAI-compatible server (``vllm serve``).

vLLM exposes the *same* ``/v1/chat/completions`` API as Ollama, so this reuses
the Ollama transport unchanged and only swaps the defaults: vLLM's port 8000 and
the ``$VLLM_HOST`` / ``$VLLM_API_KEY`` environment variables. Use it for
high-throughput judging — vLLM does continuous batching + PagedAttention, so feed
it many concurrent requests (large chunks, high ``--workers``).

No extra dependency beyond the ``ollama`` backend's ``requests`` (``.[llm-judge]``).
Set up the server with eval2026/judge/VLLM_SETUP.md.
"""

from __future__ import annotations

import os

from btc_eval.soap_judge.backends import register
from btc_eval.soap_judge.backends.ollama import OllamaBackend

DEFAULT_HOST = "http://localhost:8000"


@register("vllm")
class VLLMBackend(OllamaBackend):
    """OpenAI-compatible chat backend for a local vLLM server."""

    def __init__(
        self,
        host: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        **_,
    ) -> None:
        # vLLM ignores the key but the OpenAI client shape expects one ("EMPTY").
        super().__init__(
            host=host or os.environ.get("VLLM_HOST") or DEFAULT_HOST,
            api_key=api_key or os.environ.get("VLLM_API_KEY") or "EMPTY",
            timeout=timeout,
        )
