#!/usr/bin/env python3
"""Example: Concept F1 evaluation using btc_eval as a library.

For the full competition evaluation (Concept F1 + ROUGE-2/3), see
pipeline_scoring.py which provides a single score_predictions() function.

Run from the repository root:  uv run python examples/evaluate_example.py
"""

from btc_eval.io import load_summaries_jsonl
from btc_eval.matchers.open_medical import OpenMedicalMatcher
from btc_eval.metrics.concept_f1 import (
    aggregate_metrics,
    compute_concept_metrics,
    extract_concept_ids,
)

# Load data
predictions = load_summaries_jsonl("tests/data/sample_predictions.jsonl")
references = load_summaries_jsonl("tests/data/sample_references.jsonl")

# Initialize matcher (MeSH-only, no scispaCy required)
matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=False)
print(f"scispaCy active: {matcher.scispacy_active}")

# Evaluate each dialog
results = []
for dialog_id in sorted(set(predictions) & set(references)):
    ref_concepts = extract_concept_ids(references[dialog_id], matcher)
    hyp_concepts = extract_concept_ids(predictions[dialog_id], matcher)
    metrics = compute_concept_metrics(ref_concepts, hyp_concepts)
    results.append(metrics)
    print(f"{dialog_id}: F1={metrics.f1:.3f}  P={metrics.precision:.3f}  R={metrics.recall:.3f}")

# Aggregate
agg = aggregate_metrics(results)
print(f"\nMean F1: {agg.mean_f1:.3f} +/- {agg.std_f1:.3f}")
print(f"Mean Precision: {agg.mean_precision:.3f} +/- {agg.std_precision:.3f}")
print(f"Mean Recall: {agg.mean_recall:.3f} +/- {agg.std_recall:.3f}")
