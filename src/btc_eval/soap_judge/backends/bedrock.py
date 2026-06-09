"""AWS Bedrock backend (Anthropic Messages API).

Ported from the original ``BedrockClient``. Two deliberate changes from the
source:

* ``max_tokens`` is sent (the source commented it out; Bedrock's Anthropic
  provider rejects requests without it).
* No bespoke credential setup -- credentials come from the standard boto3
  chain (env vars, shared config/credentials, SSO, or an instance role).

Requires ``boto3`` (``uv pip install '.[llm-bedrock]'``).
"""

from __future__ import annotations

import json
import logging
import os

from btc_eval.soap_judge.backends import Message, call_with_retry, register

log = logging.getLogger(__name__)


@register("bedrock")
class BedrockBackend:
    """Anthropic-on-Bedrock chat backend."""

    def __init__(
        self,
        region: str | None = None,
        read_timeout: int = 3600,
        connect_timeout: int = 3600,
        **_,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import (
                ClientError,
                ConnectionClosedError,
                ConnectTimeoutError,
                EndpointConnectionError,
                ReadTimeoutError,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'bedrock' backend requires 'boto3'. "
                "Install with:  uv pip install '.[llm-bedrock]'"
            ) from exc
        self._retry_on = (
            ClientError,
            ReadTimeoutError,
            ConnectTimeoutError,
            EndpointConnectionError,
            ConnectionClosedError,
        )
        region = region or os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION"
        ) or "us-east-1"
        cfg = Config(
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
            retries={"max_attempts": 5, "mode": "adaptive"},
        )
        self._client = boto3.client("bedrock-runtime", region_name=region, config=cfg)

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
        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if system:
            body["system"] = system

        def _invoke() -> dict:
            resp = self._client.invoke_model(modelId=model, body=json.dumps(body))
            return json.loads(resp.get("body").read())

        data = call_with_retry(_invoke, retry_on=self._retry_on)
        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            log.warning("Unexpected Bedrock response shape: %s", data)
            return json.dumps(data)
