"""Parse and tally the LLM judge's raw text output.

The judge returns free-form text that should contain a single JSON object. These
helpers extract that JSON leniently (tolerating ``<reasoning>`` preambles and
``` ```json ``` fences), then flatten a judgment into the per-dialog scalar row.

Ported from the source pipeline's two parse steps, with one deliberate fix: the
source assumed the stored model output was a full OpenAI/OpenRouter *envelope*
(``json.loads(raw)["choices"][0]["message"]["content"]``) even though it had
already extracted the assistant text — so every note failed to parse. Here the
backend returns the assistant text directly, and these parsers treat
``raw_content`` as that text. There is no ``["choices"]`` unwrap.
"""

from __future__ import annotations

import json
import re
from typing import Any

from btc_eval.types import SOAP_ERROR_TYPES, SoapScores

_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL)
_FENCE_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_FENCE_RE = re.compile(r"```\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of one JSON object from assistant text.

    Strategy (first success wins): strip ``<reasoning>`` blocks, then try a
    ```` ```json ```` fence, a bare ```` ``` ```` fence, the substring from the
    first ``{`` to the last ``}``, and finally the whole string. Returns the
    parsed object, or ``None`` if nothing parses to a dict.
    """
    if not text or not text.strip():
        return None
    cleaned = _REASONING_RE.sub("", text).strip()

    candidates: list[str] = []
    for pattern in (_FENCE_JSON_RE, _FENCE_RE):
        m = pattern.search(cleaned)
        if m:
            candidates.append(m.group(1))
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(cleaned[first : last + 1])
    candidates.append(cleaned)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_claims(raw_text: str) -> dict | None:
    """Parse a stage-1 claim-extraction response into a ``soap_claims`` object."""
    return extract_json_object(raw_text)


def parse_judgment(raw_text: str) -> dict | None:
    """Parse a stage-2 judging response.

    Returns the judgment object only if it carries the ``subscores_1_to_5`` block
    that downstream scoring depends on; otherwise ``None`` (treated as a failure).
    """
    obj = extract_json_object(raw_text)
    if obj is None or "subscores_1_to_5" not in obj:
        return None
    return obj


def safe_int(x: Any, default: int = 0) -> int:
    """Coerce to int, falling back to ``default`` on ``None`` or bad values."""
    if x is None:
        return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    """Coerce to float, falling back to ``default`` on ``None`` or bad values."""
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _tally_error_types(judgment: dict) -> dict[str, int]:
    """Count per-claim error types from ``claim_judgments``.

    Each claim's ``error_types`` may be a list, a single string, or pipe-joined
    (e.g. ``"over_medicalization|wrong_section"``). The aggregate ``metrics``
    block may also carry explicit ``*_count`` values, which take precedence.
    """
    counts = {et: 0 for et in SOAP_ERROR_TYPES}
    for claim in judgment.get("claim_judgments", []) or []:
        error_types = claim.get("error_types", [])
        flat: list[str] = []
        if isinstance(error_types, list):
            for e in error_types:
                flat.extend(part.strip() for part in str(e).split("|"))
        elif isinstance(error_types, str):
            flat.extend(part.strip() for part in error_types.split("|"))
        for et in counts:
            if et in flat:
                counts[et] += 1
    return counts


def judgment_to_scores(judgment: dict) -> SoapScores:
    """Flatten a judgment JSON object into the per-dialog :class:`SoapScores`."""
    subs = judgment.get("subscores_1_to_5", {}) or {}
    metrics = judgment.get("metrics", {}) or {}
    rates = metrics.get("rates", {}) or {}
    claim_counts = metrics.get("claim_counts", {}) or {}
    coverage = metrics.get("coverage", {}) or {}
    conciseness = metrics.get("conciseness", {}) or {}

    tallied = _tally_error_types(judgment)

    return SoapScores(
        faithfulness=safe_int(subs.get("faithfulness_grounding")),
        structure=safe_int(subs.get("structure_formatting")),
        coverage=safe_int(subs.get("coverage_completeness")),
        conciseness=safe_int(subs.get("conciseness")),
        # Prefer an explicit aggregate count if the model supplied one, else tally.
        over_medicalization=safe_int(
            metrics.get("over_medicalization_count", tallied["over_medicalization"])
        ),
        under_medicalization=safe_int(
            metrics.get("under_medicalization_count", tallied["under_medicalization"])
        ),
        over_specific=safe_int(metrics.get("over_specific_count", tallied["over_specific"])),
        hallucination_rate=safe_float(rates.get("unsupported_rate")),
        contradiction_rate=safe_float(rates.get("contradiction_rate")),
        missed_claims=safe_int(claim_counts.get("not_in_transcript")),
        critical_omissions=safe_int(coverage.get("critical_omissions_count")),
        redundancy_count=safe_int(conciseness.get("redundancy_count")),
    )
