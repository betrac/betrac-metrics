#!/usr/bin/env python3
"""Example: using btc_eval ROUGE metrics as a library.

For the full competition evaluation (Concept F1 + ROUGE-2/3), see
pipeline_scoring.py which provides a single score_predictions() function.

Run from the repository root:  uv run python examples/rouge_example.py
"""

from btc_eval.io import load_summaries_jsonl
from btc_eval.metrics.rouge import aggregate_rouge, compute_rouge, _get_scorer
from btc_eval.types import COMPETITION_ROUGE_TYPES

# Load data
predictions = load_summaries_jsonl("tests/data/sample_predictions.jsonl")
references = load_summaries_jsonl("tests/data/sample_references.jsonl")

# --- Competition metrics only (ROUGE-2/3) ---
print("=== Competition ROUGE (2/3) ===")
scorer = _get_scorer(use_stemmer=False, rouge_types=COMPETITION_ROUGE_TYPES)

results = []
for dialog_id in sorted(set(predictions) & set(references)):
    result = compute_rouge(
        references[dialog_id], predictions[dialog_id],
        scorer=scorer, rouge_types=COMPETITION_ROUGE_TYPES,
    )
    results.append(result)
    print(f"{dialog_id}: R-2={result.scores['rouge2'].fmeasure:.3f}  "
          f"R-3={result.scores['rouge3'].fmeasure:.3f}")

agg = aggregate_rouge(results, rouge_types=COMPETITION_ROUGE_TYPES)
print(f"\nMean ROUGE-2 F: {agg['rouge2_f_mean']:.3f} +/- {agg['rouge2_f_std']:.3f}")
print(f"Mean ROUGE-3 F: {agg['rouge3_f_mean']:.3f} +/- {agg['rouge3_f_std']:.3f}")

# --- All ROUGE types (1/2/3/4/L) ---
print("\n=== All ROUGE types ===")
scorer_all = _get_scorer(use_stemmer=True)

results_all = []
for dialog_id in sorted(set(predictions) & set(references)):
    result = compute_rouge(references[dialog_id], predictions[dialog_id], scorer=scorer_all)
    results_all.append(result)

agg_all = aggregate_rouge(results_all)
print(f"Mean ROUGE-1 F: {agg_all['rouge1_f_mean']:.3f}")
print(f"Mean ROUGE-2 F: {agg_all['rouge2_f_mean']:.3f}")
print(f"Mean ROUGE-L F: {agg_all['rougeL_f_mean']:.3f}")
