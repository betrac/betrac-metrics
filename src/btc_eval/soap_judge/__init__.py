"""LLM-as-a-Judge evaluation for SOAP notes.

Scores a generated SOAP note against the doctor-patient transcript it was
written from, using an LLM as a strict, evidence-bound judge across four
dimensions: faithfulness, structure, coverage, and conciseness.

The judge is pluggable: any backend implementing the :class:`LLMBackend`
protocol (Ollama, OpenRouter, Bedrock, first-party Anthropic, or a custom one)
can drive the pipeline. See :mod:`btc_eval.soap_judge.backends`.

Public entry points:
    run_extract  -- stage 1: extract atomic claims from each note
    run_judge    -- stage 2: judge each note vs its transcript + claims
    aggregate_soap -- mean/std across per-dialog scores
"""

from __future__ import annotations

from btc_eval.soap_judge.pipeline import aggregate_soap, run_extract, run_judge
from btc_eval.soap_judge.prompts import load_prompt

__all__ = [
    "run_extract",
    "run_judge",
    "aggregate_soap",
    "load_prompt",
]
