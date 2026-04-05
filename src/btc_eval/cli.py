"""CLI entry point for btc-eval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from btc_eval.io import (
    build_reference_jsonl,
    load_references_from_dir,
    load_summaries_jsonl,
    print_compact_summary,
    print_results_table,
    print_rouge_table,
    write_results_csv,
    write_results_json,
    write_results_jsonl,
)
from btc_eval.metrics.concept_f1 import (
    aggregate_metrics,
    compute_concept_metrics,
    extract_concept_ids,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_paired_data(args: argparse.Namespace) -> tuple[dict, dict, list[str]] | None:
    """Load predictions and references, find shared IDs.

    Returns (predictions, references, shared_ids) or None on error.
    """
    predictions = load_summaries_jsonl(args.predictions)
    if not predictions:
        print("Error: no predictions loaded", file=sys.stderr)
        return None

    if getattr(args, "references", None):
        ref_format = getattr(args, "reference_format", "jsonl")
        if ref_format == "dir":
            references = load_references_from_dir(args.references)
        else:
            references = load_summaries_jsonl(args.references)
    else:
        use_cache = not getattr(args, "no_cache", False)
        references = _load_references_hf(getattr(args, "split", None), cache=use_cache)
        if references is None:
            return None

    pred_ids = set(predictions.keys())
    ref_ids = set(references.keys())
    pred_only = pred_ids - ref_ids
    ref_only = ref_ids - pred_ids
    allow_missing = getattr(args, "allow_missing", False)

    if pred_only:
        print(f"Warning: {len(pred_only)} prediction IDs have no reference", file=sys.stderr)

    if ref_only:
        if allow_missing:
            print(f"Warning: {len(ref_only)} reference IDs have no prediction (skipped)",
                  file=sys.stderr)
        else:
            missing = sorted(ref_only)
            print(f"Warning: {len(missing)} predictions missing, scored as zero: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
                  file=sys.stderr)
            for did in missing:
                predictions[did] = ""

    # In strict mode: all reference IDs; in allow-missing mode: only shared IDs
    if allow_missing:
        shared_ids = sorted(pred_ids & ref_ids)
    else:
        shared_ids = sorted(ref_ids)

    if not shared_ids:
        print("Error: no matching IDs between predictions and references", file=sys.stderr)
        return None

    # Warn about empty/whitespace summaries (excluding inserted empty predictions)
    empty_preds = [did for did in shared_ids
                   if did not in ref_only and not predictions[did].strip()]
    empty_refs = [did for did in shared_ids if not references[did].strip()]
    if empty_preds:
        print(
            f"Warning: {len(empty_preds)} predictions have empty/whitespace-only summaries: "
            f"{empty_preds[:5]}{'...' if len(empty_preds) > 5 else ''}",
            file=sys.stderr,
        )
    if empty_refs:
        print(
            f"Warning: {len(empty_refs)} references have empty/whitespace-only summaries: "
            f"{empty_refs[:5]}{'...' if len(empty_refs) > 5 else ''}",
            file=sys.stderr,
        )

    return predictions, references, shared_ids


def _load_references_hf(split: str | None, cache: bool = True) -> dict[str, str] | None:
    """Attempt to load references from HuggingFace (with local caching)."""
    if not split:
        print(
            "Error: --split is required when --references is not provided "
            "(needed to load from HuggingFace)",
            file=sys.stderr,
        )
        return None
    try:
        from btc_eval.io import _get_cache_path, load_references_from_hf

        cached = _get_cache_path("BeTraC/betrac-2026", split)
        if cache and cached.exists():
            print(f"Loading cached references from {cached}")
        else:
            print(f"Downloading references from HuggingFace (split={split})...")
            if cache:
                print(f"  (will be cached at {cached} for future runs)")
        return load_references_from_hf(split=split, cache=cache)
    except ImportError:
        print(
            "Error: 'datasets' library required for auto-loading references.\n"
            "Install with:  uv pip install '.[hf]'\n"
            "Or provide --references path to a local JSONL file.",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_evaluate_all(args: argparse.Namespace) -> int:
    """Run unified evaluation: Open Medical Concept F1 + ROUGE-2/3."""
    try:
        from btc_eval.metrics.rouge import aggregate_rouge, compute_rouge, _get_scorer
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    from btc_eval.matchers.open_medical import OpenMedicalMatcher
    from btc_eval.types import COMPETITION_ROUGE_TYPES, ConceptMetrics, RougeResult

    result = _load_paired_data(args)
    if result is None:
        return 1
    predictions, references, shared_ids = result

    # Initialize matchers/scorers
    use_scispacy = not args.no_scispacy
    matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=use_scispacy)
    matcher_label = "open-medical" if matcher.scispacy_active else "mesh-only"
    if use_scispacy and not matcher.scispacy_active:
        print(
            "Warning: scispaCy requested but not available, falling back to mesh-only",
            file=sys.stderr,
        )

    scorer = _get_scorer(
        use_stemmer=args.use_stemmer,
        rouge_types=COMPETITION_ROUGE_TYPES,
    )

    # Evaluate
    concept_metrics_list: list[ConceptMetrics] = []
    concept_dicts: list[dict] = []
    rouge_results: list[RougeResult] = []
    rouge_dicts: list[dict] = []
    hyp_word_counts: list[int] = []
    hyp_char_counts: list[int] = []

    for dialog_id in shared_ids:
        ref_text = references[dialog_id]
        hyp_text = predictions[dialog_id]
        hyp_word_counts.append(len(hyp_text.split()))
        hyp_char_counts.append(len(hyp_text.replace(" ", "").replace("\t", "").replace("\n", "")))

        # Concept F1
        ref_concepts = extract_concept_ids(ref_text, matcher)
        hyp_concepts = extract_concept_ids(hyp_text, matcher)
        cm = compute_concept_metrics(ref_concepts, hyp_concepts)
        concept_metrics_list.append(cm)
        row = cm.to_dict()
        row["id"] = dialog_id
        concept_dicts.append(row)

        # ROUGE-2/3
        rr = compute_rouge(
            ref_text,
            hyp_text,
            scorer=scorer,
            rouge_types=COMPETITION_ROUGE_TYPES,
        )
        rouge_results.append(rr)
        rrow = rr.to_dict()
        rrow["id"] = dialog_id
        rouge_dicts.append(rrow)

        if args.verbose:
            overlap = ref_concepts & hyp_concepts
            print(f"\n  {dialog_id}:")
            print(
                f"    Ref concepts ({len(ref_concepts)}): "
                f"{sorted(ref_concepts)[:10]}{'...' if len(ref_concepts) > 10 else ''}"
            )
            print(
                f"    Hyp concepts ({len(hyp_concepts)}): "
                f"{sorted(hyp_concepts)[:10]}{'...' if len(hyp_concepts) > 10 else ''}"
            )
            print(
                f"    Overlap ({len(overlap)}): "
                f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
            )
            print(f"    F1={cm.f1:.3f}  P={cm.precision:.3f}  R={cm.recall:.3f}")
            print(
                f"    ROUGE-2 F={rr.scores['rouge2'].fmeasure:.3f}  "
                f"ROUGE-3 F={rr.scores['rouge3'].fmeasure:.3f}"
            )

    # Aggregate
    concept_agg = aggregate_metrics(concept_metrics_list).to_dict()
    rouge_agg = aggregate_rouge(rouge_results, rouge_types=COMPETITION_ROUGE_TYPES)

    n = len(shared_ids)
    pred_stats = {
        "mean_pred_words": sum(hyp_word_counts) / n,
        "mean_pred_chars": sum(hyp_char_counts) / n,
    }

    # Confidence intervals for all three metrics
    ci_info: dict = {}
    if getattr(args, "bootstrap_ci", False) or getattr(args, "analytical_ci", False):
        from btc_eval.metrics.utils import bootstrap_ci, analytical_ci

        n = len(shared_ids)
        ci_metrics = {
            "concept_f1": [cm.f1 for cm in concept_metrics_list],
            "rouge2_f": [rr.scores["rouge2"].fmeasure for rr in rouge_results],
            "rouge3_f": [rr.scores["rouge3"].fmeasure for rr in rouge_results],
        }
        std_lookup = {
            "concept_f1": concept_agg["std_f1"],
            "rouge2_f": rouge_agg.get("rouge2_f_std", 0),
            "rouge3_f": rouge_agg.get("rouge3_f_std", 0),
        }

        for key, values in ci_metrics.items():
            if args.bootstrap_ci:
                lo, hi = bootstrap_ci(values, n_boot=args.bootstrap_n)
                ci_info[f"{key}_bci_lo"] = lo
                ci_info[f"{key}_bci_hi"] = hi
            if args.analytical_ci:
                ci_info[f"{key}_aci_half"] = analytical_ci(std_lookup[key], n)

    # Print compact summary (only show split when references came from HF)
    display_split = args.split if not args.references else None
    print_compact_summary(
        concept_agg,
        rouge_agg,
        team=args.team,
        system=getattr(args, "system", None),
        split=display_split,
        num_dialogs=len(shared_ids),
        ci_info=ci_info,
        matcher_label=matcher_label,
        pred_stats=pred_stats,
    )

    # Write output files
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "team": args.team or "",
            "system": getattr(args, "system", None) or "",
            "split": display_split or "",
            "num_dialogs": len(shared_ids),
            "matcher": matcher_label,
            "use_scispacy": matcher.scispacy_active,
            "use_stemmer": args.use_stemmer,
            "concept_f1": concept_agg["mean_f1"],
            "concept_precision": concept_agg["mean_precision"],
            "concept_recall": concept_agg["mean_recall"],
            "rouge2_f": rouge_agg.get("rouge2_f_mean", 0),
            "rouge2_p": rouge_agg.get("rouge2_p_mean", 0),
            "rouge2_r": rouge_agg.get("rouge2_r_mean", 0),
            "rouge3_f": rouge_agg.get("rouge3_f_mean", 0),
            "rouge3_p": rouge_agg.get("rouge3_p_mean", 0),
            "rouge3_r": rouge_agg.get("rouge3_r_mean", 0),
        }
        summary.update(pred_stats)
        summary.update(ci_info)
        write_results_json(summary, output_dir / "summary.json")
        write_results_jsonl(concept_dicts, output_dir / "concept_per_dialog.jsonl")
        write_results_jsonl(rouge_dicts, output_dir / "rouge_per_dialog.jsonl")
        print(f"\nResults saved to {output_dir}/")

    return 0


def cmd_concept_f1(args: argparse.Namespace) -> int:
    """Run medical concept evaluation. Returns 0 on success, 1 on error."""
    result = _load_paired_data(args)
    if result is None:
        return 1
    predictions, references, shared_ids = result

    # Initialize matcher
    use_scispacy = args.matcher == "open-medical"

    from btc_eval.matchers.open_medical import OpenMedicalMatcher

    matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=use_scispacy)
    print(f"Matcher: {'open-medical' if matcher.scispacy_active else 'mesh-only'}")
    print(f"Evaluating {len(shared_ids)} dialogs...")

    # Evaluate each dialog
    from btc_eval.types import ConceptMetrics as _CM

    per_dialog_metrics: list[_CM] = []
    per_dialog_dicts: list[dict] = []

    for dialog_id in shared_ids:
        ref_text = references[dialog_id]
        hyp_text = predictions[dialog_id]
        ref_concepts = extract_concept_ids(ref_text, matcher)
        hyp_concepts = extract_concept_ids(hyp_text, matcher)
        metrics = compute_concept_metrics(ref_concepts, hyp_concepts)
        per_dialog_metrics.append(metrics)

        row = metrics.to_dict()
        row["id"] = dialog_id
        row["ref_words"] = len(ref_text.split())
        row["hyp_words"] = len(hyp_text.split())
        per_dialog_dicts.append(row)

        if args.verbose:
            overlap = ref_concepts & hyp_concepts
            print(f"\n  {dialog_id}:")
            print(
                f"    Ref concepts ({len(ref_concepts)}): "
                f"{sorted(ref_concepts)[:10]}{'...' if len(ref_concepts) > 10 else ''}"
            )
            print(
                f"    Hyp concepts ({len(hyp_concepts)}): "
                f"{sorted(hyp_concepts)[:10]}{'...' if len(hyp_concepts) > 10 else ''}"
            )
            print(
                f"    Overlap ({len(overlap)}): "
                f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
            )
            print(f"    F1={metrics.f1:.3f}  P={metrics.precision:.3f}  R={metrics.recall:.3f}")

    # Aggregate
    agg = aggregate_metrics(per_dialog_metrics)
    agg_dict = agg.to_dict()

    # Add word-count statistics to aggregate
    ref_word_counts = [d["ref_words"] for d in per_dialog_dicts]
    hyp_word_counts = [d["hyp_words"] for d in per_dialog_dicts]
    n = len(ref_word_counts)
    mean_rw = sum(ref_word_counts) / n
    mean_hw = sum(hyp_word_counts) / n
    agg_dict["mean_ref_words"] = mean_rw
    agg_dict["std_ref_words"] = (sum((v - mean_rw) ** 2 for v in ref_word_counts) / n) ** 0.5
    agg_dict["mean_hyp_words"] = mean_hw
    agg_dict["std_hyp_words"] = (sum((v - mean_hw) ** 2 for v in hyp_word_counts) / n) ** 0.5

    # Print results
    print_results_table(per_dialog_dicts, agg_dict, summary_only=args.summary_only)

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_results_json(agg_dict, output_dir / "aggregate_metrics.json")
        write_results_jsonl(per_dialog_dicts, output_dir / "per_dialog_metrics.jsonl")
        write_results_csv(
            per_dialog_dicts,
            output_dir / "per_dialog_metrics.csv",
            fieldnames=[
                "id",
                "f1",
                "precision",
                "recall",
                "ref_count",
                "hyp_count",
                "overlap_count",
                "ref_words",
                "hyp_words",
            ],
        )
        print(f"\nResults saved to {output_dir}/")

    return 0


def cmd_build_references(args: argparse.Namespace) -> int:
    """Build a reference JSONL from a directory of .txt files. Returns 0 on success, 1 on error."""
    filter_ids = None
    if args.filter_from_jsonl:
        filter_ids = set()
        with open(args.filter_from_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    filter_ids.add(record["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Filtering to {len(filter_ids)} IDs from {args.filter_from_jsonl}")

    count = build_reference_jsonl(args.input_dir, args.output, filter_ids)
    print(f"Wrote {count} references to {args.output}")
    return 0 if count > 0 else 1


def cmd_download_references(args: argparse.Namespace) -> int:
    """Download references from HuggingFace and save as JSONL."""
    try:
        from btc_eval.io import _get_cache_path, _stream_references_from_hf, _write_cache
    except ImportError:
        print(
            "Error: 'datasets' library required.\n"
            "Install with:  uv pip install '.[hf]'",
            file=sys.stderr,
        )
        return 1

    split = args.split
    print(f"Downloading references from HuggingFace (split={split})...")
    summaries = _stream_references_from_hf("BeTraC/betrac-2026", split)
    if not summaries:
        print("Error: no references found", file=sys.stderr)
        return 1

    # Write to specified output or cache location
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for did, summary in sorted(summaries.items()):
                f.write(json.dumps({"id": did, "summary": summary}, ensure_ascii=False) + "\n")
        print(f"Wrote {len(summaries)} references to {output}")
    else:
        _write_cache(summaries, "BeTraC/betrac-2026", split)
        cached = _get_cache_path("BeTraC/betrac-2026", split)
        print(f"Cached {len(summaries)} references to {cached}")
        print(f"Future 'btc-eval evaluate --split {split}' runs will use the cache.")

    return 0


def cmd_rouge(args: argparse.Namespace) -> int:
    """Run ROUGE 1/2/3/4/L evaluation. Returns 0 on success, 1 on error."""
    try:
        from btc_eval.metrics.rouge import aggregate_rouge, compute_rouge, _get_scorer
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = _load_paired_data(args)
    if result is None:
        return 1
    predictions, references, shared_ids = result

    print(
        f"\nComputing ROUGE-1/2/3/4/L for {len(shared_ids)} dialogs "
        f"(stemmer={'on' if args.use_stemmer else 'off'})..."
    )

    scorer = _get_scorer(use_stemmer=args.use_stemmer)

    from btc_eval.types import RougeResult

    per_dialog_results: list[RougeResult] = []
    per_dialog_dicts: list[dict] = []

    for dialog_id in shared_ids:
        ref_text = references[dialog_id]
        hyp_text = predictions[dialog_id]
        result = compute_rouge(ref_text, hyp_text, scorer=scorer)
        per_dialog_results.append(result)

        row = result.to_dict()
        row["id"] = dialog_id
        row["ref_words"] = len(ref_text.split())
        row["hyp_words"] = len(hyp_text.split())
        per_dialog_dicts.append(row)

    # Aggregate
    agg = aggregate_rouge(per_dialog_results)

    # Add word-count statistics to aggregate
    ref_word_counts = [d["ref_words"] for d in per_dialog_dicts]
    hyp_word_counts = [d["hyp_words"] for d in per_dialog_dicts]
    n = len(ref_word_counts)
    mean_rw = sum(ref_word_counts) / n
    mean_hw = sum(hyp_word_counts) / n
    agg["mean_ref_words"] = mean_rw
    agg["std_ref_words"] = (sum((v - mean_rw) ** 2 for v in ref_word_counts) / n) ** 0.5
    agg["mean_hyp_words"] = mean_hw
    agg["std_hyp_words"] = (sum((v - mean_hw) ** 2 for v in hyp_word_counts) / n) ** 0.5

    # Print
    print_rouge_table(per_dialog_dicts, agg, summary_only=args.summary_only)

    # Save
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_results_json(agg, output_dir / "rouge_aggregate.json")
        write_results_jsonl(per_dialog_dicts, output_dir / "rouge_per_dialog.jsonl")

        from btc_eval.types import ROUGE_TYPES

        csv_fields = ["id"]
        for rt in ROUGE_TYPES:
            csv_fields.extend([f"{rt}_f", f"{rt}_p", f"{rt}_r"])
        csv_fields.extend(["ref_words", "hyp_words"])
        write_results_csv(
            per_dialog_dicts, output_dir / "rouge_per_dialog.csv", fieldnames=csv_fields
        )
        print(f"\nResults saved to {output_dir}/")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run batch comparison across prediction files. Returns 0 on success, 1 on error."""
    from btc_eval.compare import run_comparison

    return run_comparison(
        references_path=args.references,
        predictions_dir=args.predictions_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        skip_rouge=args.no_rouge,
        skip_concepts=args.no_concepts,
        open_medical=args.open_medical,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="btc-eval",
        description="BeTraC 2026 — Evaluation Harness",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- evaluate (unified competition metrics) ---
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run all competition metrics (Open Medical Concept F1 + ROUGE-2/3)",
    )
    evaluate_parser.add_argument(
        "--predictions",
        required=True,
        help="Predictions JSONL file (with 'id' and 'summary' fields)",
    )
    evaluate_parser.add_argument(
        "--references",
        default=None,
        help="References JSONL or directory (auto-loads from HuggingFace if omitted)",
    )
    evaluate_parser.add_argument(
        "--reference-format",
        choices=["jsonl", "dir"],
        default="jsonl",
        help="Format of the references input (default: jsonl)",
    )
    evaluate_parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split: train, validation, or test (default: validation)",
    )
    evaluate_parser.add_argument(
        "--team",
        default=None,
        help="Team identifier (included in output)",
    )
    evaluate_parser.add_argument(
        "--system",
        default=None,
        help="System or experiment name (included in output)",
    )
    evaluate_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing predictions instead of scoring them as zero",
    )
    evaluate_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download of references from HuggingFace (ignore local cache)",
    )
    evaluate_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for detailed results",
    )
    evaluate_parser.add_argument(
        "--use-stemmer",
        action="store_true",
        help="Use Porter stemmer for ROUGE tokenization",
    )
    evaluate_parser.add_argument(
        "--no-scispacy",
        action="store_true",
        help="Use MeSH-only matching (no scispaCy dependency required)",
    )
    evaluate_parser.add_argument(
        "--bootstrap-ci",
        action="store_true",
        help="Compute bootstrap 95%% confidence interval for Concept F1",
    )
    evaluate_parser.add_argument(
        "--analytical-ci",
        action="store_true",
        help="Compute analytical 95%% confidence interval for Concept F1",
    )
    evaluate_parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate_all)

    # --- concept-f1 ---
    eval_parser = subparsers.add_parser(
        "concept-f1",
        help="Run medical concept F1 evaluation",
    )
    eval_parser.add_argument(
        "--predictions",
        required=True,
        help="Predictions JSONL file (with 'id' and 'summary' fields)",
    )
    eval_parser.add_argument(
        "--references",
        required=True,
        help="References JSONL file or directory of .txt files",
    )
    eval_parser.add_argument(
        "--reference-format",
        choices=["jsonl", "dir"],
        default="jsonl",
        help="Format of the references input (default: jsonl)",
    )
    eval_parser.add_argument(
        "--matcher",
        choices=["open-medical", "mesh-only"],
        default="open-medical",
        help="Matcher: 'open-medical' (MeSH+scispaCy) or 'mesh-only' (default: open-medical)",
    )
    eval_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for results (optional)",
    )
    eval_parser.add_argument(
        "--summary-only",
        "-s",
        action="store_true",
        help="Print only aggregate statistics, skip per-dialog rows",
    )
    eval_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing predictions instead of scoring them as zero",
    )
    eval_parser.set_defaults(func=cmd_concept_f1)

    # --- build-references ---
    build_parser = subparsers.add_parser(
        "build-references",
        help="Build reference JSONL from .txt files",
    )
    build_parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing .txt reference files",
    )
    build_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path",
    )
    build_parser.add_argument(
        "--filter-from-jsonl",
        default=None,
        help="Only include IDs present in this JSONL file",
    )
    build_parser.set_defaults(func=cmd_build_references)

    # --- download-references ---
    dl_parser = subparsers.add_parser(
        "download-references",
        help="Download references from HuggingFace and cache locally",
    )
    dl_parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split: train, validation, or test (default: validation)",
    )
    dl_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSONL path (default: cache in ~/.cache/btc-eval/)",
    )
    dl_parser.set_defaults(func=cmd_download_references)

    # --- rouge ---
    rouge_parser = subparsers.add_parser(
        "rouge",
        help="Run ROUGE 1/2/3/4/L evaluation",
    )
    rouge_parser.add_argument(
        "--predictions",
        required=True,
        help="Predictions JSONL file (with 'id' and 'summary' fields)",
    )
    rouge_parser.add_argument(
        "--references",
        required=True,
        help="References JSONL file or directory of .txt files",
    )
    rouge_parser.add_argument(
        "--reference-format",
        choices=["jsonl", "dir"],
        default="jsonl",
        help="Format of the references input (default: jsonl)",
    )
    rouge_parser.add_argument(
        "--use-stemmer",
        action="store_true",
        help="Use Porter stemmer for ROUGE tokenization",
    )
    rouge_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for results (optional)",
    )
    rouge_parser.add_argument(
        "--summary-only",
        "-s",
        action="store_true",
        help="Print only aggregate statistics, skip per-dialog rows",
    )
    rouge_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing predictions instead of scoring them as zero",
    )
    rouge_parser.set_defaults(func=cmd_rouge)

    # --- compare (organizer use) ---
    compare_parser = subparsers.add_parser(
        "compare",
        help="Batch-compare ROUGE and concept F1 across prediction files (organizer use)",
    )
    compare_parser.add_argument(
        "--references",
        required=True,
        help="References JSONL file",
    )
    compare_parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory containing prediction JSONL files",
    )
    compare_parser.add_argument(
        "--manifest",
        default=None,
        help="Metadata manifest JSON (default: manifest.json in predictions-dir)",
    )
    compare_parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for CSV and LaTeX results",
    )
    compare_parser.add_argument(
        "--no-rouge",
        action="store_true",
        help="Skip ROUGE metrics",
    )
    compare_parser.add_argument(
        "--no-concepts",
        action="store_true",
        help="Skip MeSH concept F1 metrics",
    )
    compare_parser.add_argument(
        "--open-medical",
        action="store_true",
        help="Include open-medical (MeSH + scispaCy) concept F1 metrics",
    )
    compare_parser.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
