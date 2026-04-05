#!/usr/bin/env python3
"""Score predictions using the BeTraC 2026 competition metrics.

Drop this into your pipeline to evaluate after generating SOAP notes.
Requires: pip install "btc-eval[rouge,hf]"
For full concept coverage add scispaCy: pip install "btc-eval[all]"

Usage:
    python pipeline_scoring.py predictions.jsonl [--split validation]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from btc_eval.io import load_summaries_jsonl, load_references_from_hf, print_compact_summary
from btc_eval.matchers.open_medical import OpenMedicalMatcher
from btc_eval.metrics.concept_f1 import (
    aggregate_metrics,
    compute_concept_metrics,
    extract_concept_ids,
)
from btc_eval.metrics.rouge import aggregate_rouge, compute_rouge, _get_scorer
from btc_eval.types import COMPETITION_ROUGE_TYPES


def score_predictions(
    predictions: dict[str, str],
    references: dict[str, str],
    use_scispacy: bool = False,
) -> dict:
    """Score predictions against references using BeTraC competition metrics.

    Returns a dict with all competition metrics (concept F1/P/R, ROUGE-2/3 F/P/R).
    Can be called directly from your pipeline code.
    """
    shared_ids = sorted(set(predictions) & set(references))
    if not shared_ids:
        raise ValueError("No matching IDs between predictions and references")

    # Set up matcher and scorer
    matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=use_scispacy)
    scorer = _get_scorer(use_stemmer=False, rouge_types=COMPETITION_ROUGE_TYPES)

    # Evaluate each dialog
    concept_results = []
    rouge_results = []
    for dialog_id in shared_ids:
        ref_text = references[dialog_id]
        hyp_text = predictions[dialog_id]

        # Concept F1
        ref_concepts = extract_concept_ids(ref_text, matcher)
        hyp_concepts = extract_concept_ids(hyp_text, matcher)
        concept_results.append(compute_concept_metrics(ref_concepts, hyp_concepts))

        # ROUGE-2/3
        rouge_results.append(
            compute_rouge(ref_text, hyp_text, scorer=scorer, rouge_types=COMPETITION_ROUGE_TYPES)
        )

    # Aggregate
    concept_agg = aggregate_metrics(concept_results).to_dict()
    rouge_agg = aggregate_rouge(rouge_results, rouge_types=COMPETITION_ROUGE_TYPES)

    return {
        "num_dialogs": len(shared_ids),
        "concept_f1": concept_agg["mean_f1"],
        "concept_precision": concept_agg["mean_precision"],
        "concept_recall": concept_agg["mean_recall"],
        "rouge2_f": rouge_agg["rouge2_f_mean"],
        "rouge2_p": rouge_agg["rouge2_p_mean"],
        "rouge2_r": rouge_agg["rouge2_r_mean"],
        "rouge3_f": rouge_agg["rouge3_f_mean"],
        "rouge3_p": rouge_agg["rouge3_p_mean"],
        "rouge3_r": rouge_agg["rouge3_r_mean"],
    }


def main():
    parser = argparse.ArgumentParser(description="Score SOAP note predictions")
    parser.add_argument("predictions", help="Predictions JSONL file")
    parser.add_argument("--split", default="validation",
                        help="HF dataset split (default: validation)")
    parser.add_argument("--references", default=None,
                        help="Local references JSONL (omit to auto-load from HF)")
    parser.add_argument("--output", default=None, help="Write results JSON to this path")
    args = parser.parse_args()

    # Load predictions
    predictions = load_summaries_jsonl(args.predictions)
    print(f"Loaded {len(predictions)} predictions")

    # Load references
    if args.references:
        references = load_summaries_jsonl(args.references)
    else:
        print(f"Loading references from HuggingFace (split={args.split})...")
        references = load_references_from_hf(split=args.split)
    print(f"Loaded {len(references)} references")

    # Score
    results = score_predictions(predictions, references)

    # Display
    print_compact_summary(
        concept_agg={"mean_f1": results["concept_f1"],
                     "mean_precision": results["concept_precision"],
                     "mean_recall": results["concept_recall"]},
        rouge_agg={"rouge2_f_mean": results["rouge2_f"],
                   "rouge2_p_mean": results["rouge2_p"],
                   "rouge2_r_mean": results["rouge2_r"],
                   "rouge3_f_mean": results["rouge3_f"],
                   "rouge3_p_mean": results["rouge3_p"],
                   "rouge3_r_mean": results["rouge3_r"]},
        split=args.split,
        num_dialogs=results["num_dialogs"],
    )

    # Optionally save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
