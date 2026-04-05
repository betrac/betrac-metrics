"""BTC Eval — Beyond Transcription Challenge Evaluation Harness."""

__version__ = "0.1.0"

from btc_eval.metrics.concept_f1 import extract_concept_ids, compute_concept_metrics
from btc_eval.io import load_summaries_jsonl, load_references_from_dir, print_compact_summary
from btc_eval.types import COMPETITION_ROUGE_TYPES, ROUGE_TYPES

__all__ = [
    "extract_concept_ids",
    "compute_concept_metrics",
    "load_summaries_jsonl",
    "load_references_from_dir",
    "print_compact_summary",
    "COMPETITION_ROUGE_TYPES",
    "ROUGE_TYPES",
]
