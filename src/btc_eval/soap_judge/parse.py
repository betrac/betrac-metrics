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

# Optional lenient-repair fallback. json-repair fixes truncated/glitched JSON
# (a judgment cut off mid-array, a stray character) that strict parsing drops.
# It is optional so the package still imports without it; declare it in the
# ``llm-judge`` extra so production installs include it.
try:
    from json_repair import repair_json as _repair_json
except Exception:  # pragma: no cover - optional dependency
    _repair_json = None

_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL)
_FENCE_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_FENCE_RE = re.compile(r"```\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_with_status(text: str) -> tuple[dict | None, str]:
    """Extract one JSON object from assistant text AND report how it parsed.

    Returns ``(obj, status)``. ``status`` is one of:

    - ``"strict"``      parsed as-is (the common case);
    - ``"repaired"``    only parsed after ``json-repair`` (truncated / malformed
      output whose content is still recoverable);
    - ``"empty"``       no text at all;
    - ``"unparseable"`` no dict recoverable even after repair.

    Strict strategy (first success wins): strip ``<reasoning>`` blocks, then try a
    ```` ```json ```` fence, a bare ```` ``` ```` fence, the substring from the
    first ``{`` to the last ``}``, and finally the whole string.
    """
    if not text or not text.strip():
        return None, "empty"
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
            return parsed, "strict"

    # Lenient fallback. Feed json-repair the text from the first ``{`` to the end
    # (so it can close a truncated object), then the whole cleaned string.
    if _repair_json is not None:
        repair_candidates = ([cleaned[first:]] if first != -1 else []) + [cleaned]
        for candidate in repair_candidates:
            try:
                parsed = json.loads(_repair_json(candidate))
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed:
                return parsed, "repaired"
    return None, "unparseable"


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of one JSON object (drops the status)."""
    return extract_json_with_status(text)[0]


def parse_claims_with_status(raw_text: str) -> tuple[dict | None, str]:
    """Parse a stage-1 claim-extraction response, reporting how it parsed."""
    return extract_json_with_status(raw_text)


def parse_claims(raw_text: str) -> dict | None:
    """Parse a stage-1 claim-extraction response into a ``soap_claims`` object."""
    return extract_json_with_status(raw_text)[0]


_LABEL_TO_COUNT_KEY = {
    "supported": "supported",
    "contradicted": "contradicted",
    "not-in-transcript": "not_in_transcript",
    "partial": "partial",
}


def _normalize_claim_label(value: Any) -> str:
    """Normalize minor formatting differences in claim labels."""
    label = str(value or "").strip().lower()
    label = re.sub(r"[\s_]+", "-", label)
    return label


def _is_true(value: Any) -> bool:
    """Accept JSON true and the string 'true'."""
    return value is True or str(value).strip().lower() == "true"


def recompute_judgment_metrics(judgment: dict) -> dict:
    """Recompute deterministic counts from the detailed judgment lists.

    The LLM may produce correct per-claim labels but inconsistent summary
    counts. The detailed lists are treated as the source of truth.
    """
    metrics = judgment.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        judgment["metrics"] = metrics

    # ---------------------------------------------------------------
    # Recompute claim_counts and rates from claim_judgments.
    # ---------------------------------------------------------------
    claim_judgments = judgment.get("claim_judgments")

    if isinstance(claim_judgments, list):
        valid_claims = [
            claim for claim in claim_judgments
            if isinstance(claim, dict)
        ]

        normalized_labels = [
            _normalize_claim_label(claim.get("label"))
            for claim in valid_claims
        ]

        # Only overwrite the model's summary when every item has one of
        # the four labels allowed by the output schema.
        labels_are_valid = (
            len(valid_claims) == len(claim_judgments)
            and all(label in _LABEL_TO_COUNT_KEY for label in normalized_labels)
        )

        if labels_are_valid:
            counts = {
                "total": len(valid_claims),
                "supported": 0,
                "contradicted": 0,
                "not_in_transcript": 0,
                "partial": 0,
                "who_said_mismatch": 0,
            }

            for claim, label in zip(valid_claims, normalized_labels):
                count_key = _LABEL_TO_COUNT_KEY[label]
                counts[count_key] += 1

                if _is_true(claim.get("who_said_mismatch")):
                    counts["who_said_mismatch"] += 1

            total = counts["total"]

            metrics["claim_counts"] = counts
            metrics["rates"] = {
                "unsupported_rate": (
                    counts["not_in_transcript"] / total if total else 0.0
                ),
                "contradiction_rate": (
                    counts["contradicted"] / total if total else 0.0
                ),
                "evidence_coverage_rate": (
                    counts["supported"] / total if total else 0.0
                ),
                "who_said_mismatch_rate": (
                    counts["who_said_mismatch"] / total if total else 0.0
                ),
            }

    # ---------------------------------------------------------------
    # Recompute coverage counts from coverage_checklist.
    # ---------------------------------------------------------------
    checklist = judgment.get("coverage_checklist")

    if (
        isinstance(checklist, list)
        and all(isinstance(item, dict) for item in checklist)
    ):
        checklist_total = len(checklist)
        checklist_yes = sum(
            _is_true(item.get("documented_in_note"))
            for item in checklist
        )
        critical_omissions = sum(
            not _is_true(item.get("documented_in_note"))
            and str(item.get("omission_severity", "")).strip().lower()
            == "critical"
            for item in checklist
        )

        coverage = metrics.get("coverage")
        if not isinstance(coverage, dict):
            coverage = {}
            metrics["coverage"] = coverage

        coverage.update({
            "checklist_total": checklist_total,
            "checklist_yes": checklist_yes,
            "coverage_rate": (
                checklist_yes / checklist_total
                if checklist_total
                else 0.0
            ),
            "critical_omissions_count": critical_omissions,
        })

    return judgment


def parse_judgment_with_status(raw_text: str) -> tuple[dict | None, str]:
    """Parse a stage-2 judging response, reporting how it parsed.

    ``status`` is ``"strict"`` / ``"repaired"`` when a judgment carrying the
    ``subscores_1_to_5`` block is recovered; ``"no_subscores"`` when JSON parsed
    but lacks that block; ``"empty"`` / ``"unparseable"`` otherwise. Only the
    first two are usable; the rest are failures.
    """
    obj, status = extract_json_with_status(raw_text)
    if obj is None:
        return None, status
    if "subscores_1_to_5" not in obj:
        return None, "no_subscores"
    recompute_judgment_metrics(obj)
    return obj, status


def parse_judgment(raw_text: str) -> dict | None:
    """Parse a stage-2 judging response (drops the status).

    Returns the judgment object only if it carries the ``subscores_1_to_5`` block
    that downstream scoring depends on; otherwise ``None`` (treated as a failure).
    """
    return parse_judgment_with_status(raw_text)[0]


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
    recompute_judgment_metrics(judgment)
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
