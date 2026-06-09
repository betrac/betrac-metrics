"""Deterministic, offline mock backend for tests and dry runs.

Returns valid ``soap_claims`` / ``soap_judgment`` JSON without any network
call. Scores are derived from a hash of the prompt, so they are stable across
runs but vary per input -- enough to exercise parsing, aggregation, and CSV
writing end to end.
"""

from __future__ import annotations

import hashlib
import json

from btc_eval.soap_judge.backends import Message, register


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


@register("mock")
class MockBackend:
    """Offline backend that fabricates schema-valid judge output."""

    def __init__(self, **_) -> None:  # accept and ignore CLI kwargs
        pass

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
        prompt = "\n".join(m.get("content", "") for m in messages)
        seed = _seed(prompt)
        # The judge template is the only one mentioning "soap_judgment".
        if "soap_judgment" in prompt:
            return json.dumps(self._judgment(seed))
        return json.dumps(self._claims(seed))

    @staticmethod
    def _claims(seed: int) -> dict:
        return {
            "doc_type": "soap_claims",
            "claims": [
                {
                    "claim_id": "C001",
                    "soap_section": "S",
                    "claim_type": "chief_complaint",
                    "polarity": "asserted",
                    "text": "patient reports chest pain",
                },
                {
                    "claim_id": "C002",
                    "soap_section": "A",
                    "claim_type": "diagnosis",
                    "polarity": "uncertain",
                    "text": "possible angina",
                },
            ],
        }

    @staticmethod
    def _judgment(seed: int) -> dict:
        faith = seed % 6
        structure = (seed // 6) % 6
        coverage = (seed // 36) % 6
        concise = (seed // 216) % 6
        over_med = seed % 3
        return {
            "doc_type": "soap_judgment",
            "subscores_0_to_5": {
                "faithfulness_grounding": faith,
                "structure_formatting": structure,
                "coverage_completeness": coverage,
                "conciseness": concise,
            },
            "metrics": {
                "claim_counts": {
                    "total": 2,
                    "supported": 1,
                    "contradicted": 0,
                    "not_in_transcript": 1,
                    "partial": 0,
                    "who_said_mismatch": 0,
                },
                "rates": {
                    "unsupported_rate": round((seed % 100) / 100, 2),
                    "contradiction_rate": round((seed % 50) / 100, 2),
                    "evidence_coverage_rate": 0.5,
                    "who_said_mismatch_rate": 0.0,
                },
                "coverage": {
                    "checklist_total": 4,
                    "checklist_yes": 3,
                    "coverage_rate": 0.75,
                    "critical_omissions_count": seed % 2,
                },
                "conciseness": {
                    "redundancy_count": seed % 4,
                    "low_value_supported_count": 0,
                },
            },
            "claim_judgments": [
                {
                    "claim_id": "C001",
                    "label": "Supported",
                    "error_types": [],
                },
                {
                    "claim_id": "C002",
                    "label": "Not-in-transcript",
                    "error_types": (
                        ["over_medicalization"] if over_med else ["missing_support"]
                    ),
                },
            ],
        }
