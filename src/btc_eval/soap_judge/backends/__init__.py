"""Pluggable LLM backends for the SOAP judge.

The pipeline depends on a single capability -- "send chat messages, get the
assistant's text back" -- captured by the :class:`LLMBackend` protocol. Each
concrete backend (Ollama, OpenRouter, Bedrock, first-party Anthropic, a test
mock) implements :meth:`LLMBackend.complete` and registers itself under a short
name via the :func:`register` decorator.

Selecting a backend at runtime::

    from btc_eval.soap_judge.backends import get_backend
    backend = get_backend("ollama", host="http://localhost:11434")
    text = backend.complete([{"role": "user", "content": prompt}], model="llama3.1:8b")

Adding a new backend is three steps:
    1. Create ``backends/<name>.py`` with an ``@register("<name>")`` class
       implementing ``complete(...) -> str``.
    2. Add ``<name>`` to the import line at the bottom of this module.
    3. Declare any heavy dependency as a ``pyproject`` optional-extra and import
       it lazily (inside ``__init__``/``complete``) so the registry stays
       import-light.

Heavy third-party imports (``requests``, ``boto3``, ``anthropic``) are
intentionally lazy: importing this package never requires them, so a user who
only runs the local Ollama backend installs nothing extra, and a missing
dependency surfaces as a clear error only when that backend is instantiated.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol, TypeVar, runtime_checkable

Message = dict[str, str]
"""A single chat message, ``{"role": "user"|"system"|"assistant", "content": str}``."""


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol every judge backend must satisfy.

    Implementations expose a ``name`` (the registry key) and a single
    :meth:`complete` method returning the assistant's text. Transport failures
    should raise; the pipeline catches per-task and records the failure rather
    than aborting the whole run.
    """

    name: str

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
        """Return the assistant's text for ``messages`` from ``model``."""
        ...


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    """Class decorator that registers a backend under ``name``."""

    def deco(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_backends() -> list[str]:
    """Return the sorted names of all registered backends."""
    return sorted(_REGISTRY)


def get_backend(name: str, **kwargs) -> LLMBackend:
    """Instantiate a registered backend by name.

    Extra keyword arguments are forwarded to the backend constructor; each
    backend ignores keys it does not use, so the CLI can pass a superset
    (``host``, ``region``, ``provider_name``, ``api_key`` ...).

    Raises:
        ValueError: if ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown LLM backend {name!r}. Available: {available_backends()}."
        )
    return _REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Shared retry helper (no third-party dependency)
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


def call_with_retry(
    fn: Callable[[], _T],
    *,
    attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Call ``fn`` with exponential backoff on the given exception types.

    Re-raises the last exception after ``attempts`` tries. ``sleep`` is
    injectable so tests can run without real delays.
    """
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203 - retry loop is intentional
            last = exc
            if i == attempts - 1:
                raise
            sleep(min(max_delay, base_delay * (2**i)))
    assert last is not None  # unreachable; satisfies type checker
    raise last


# Register the built-in backends. Module-level imports here are stdlib-only;
# each adapter defers its heavy third-party import, so this never fails on a
# missing optional dependency. Order: mock first (always usable), then the
# real backends.
from btc_eval.soap_judge.backends import (  # noqa: E402,F401
    anthropic,
    bedrock,
    mock,
    ollama,
    openrouter,
    vllm,
)

__all__ = [
    "LLMBackend",
    "Message",
    "register",
    "get_backend",
    "available_backends",
    "call_with_retry",
]
