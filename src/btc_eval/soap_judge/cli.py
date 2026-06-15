"""CLI handler for the ``soap-judge`` subcommand.

Wired into the top-level parser in :mod:`btc_eval.cli`. Loads predictions (the
SOAP notes, as ``{"id","summary"}`` JSONL) and transcripts (``{"id","transcript"}``
JSONL), runs the two judging stages on the selected backend, and writes a
per-dialog CSV/JSONL plus an aggregate ``summary.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def add_soap_judge_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``soap-judge`` subparser and its arguments."""
    parser = subparsers.add_parser(
        "soap-judge",
        help="LLM-as-a-judge scoring of SOAP notes against transcripts",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Predictions JSONL (with 'id' and 'summary' fields; summary = the SOAP note)",
    )
    parser.add_argument(
        "--transcripts",
        required=True,
        help="Transcripts JSONL (with 'id' and 'transcript' fields)",
    )
    parser.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "openrouter", "bedrock", "anthropic", "mock"],
        help="LLM backend (default: ollama — local, no API key)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id for the chosen backend (e.g. 'llama3.1:8b', 'z-ai/glm-4.7')",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        metavar="i/N",
        help="Process only shard i of N (stable hash of dialog id). For SLURM "
        "array data-parallelism; aggregate per-shard outputs with 'soap-aggregate'.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="Max output tokens per LLM call (default: 16000). Reasoning models "
        "(gpt-oss, deepseek-r1, *-thinking) spend tokens on hidden reasoning, so "
        "the verbose claims JSON can truncate — use 20000+ for those.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Base URL for the ollama backend (default: $OLLAMA_HOST or localhost:11434)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region for the bedrock backend (default: $AWS_REGION or us-east-1)",
    )
    parser.add_argument(
        "--provider-name",
        default=None,
        help="Pin a specific upstream provider for the openrouter backend",
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="Directory of prompt overrides (extract_claims.txt / judge_soap_note.txt)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not write raw responses to a resume cache under the output dir",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run every LLM call even if a cached raw response exists",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for the resume cache (extract_raw/, judge_raw/). Defaults "
        "to the output dir. Point all shards at ONE shared dir so a completed "
        "dialog is reused no matter which shard re-processes it (the cache is "
        "keyed by dialog id, not shard).",
    )
    parser.set_defaults(func=cmd_soap_judge)


def _timing_stats(results: dict) -> dict:
    """Summarize per-call wall times for a stage (excludes cache-reused items)."""
    times = [
        r["elapsed_sec"]
        for r in results.values()
        if r.get("elapsed_sec") is not None and not r.get("cached")
    ]
    if not times:
        return {"llm_calls": 0}
    ordered = sorted(times)
    return {
        "llm_calls": len(times),
        "mean_sec": sum(times) / len(times),
        "median_sec": ordered[len(ordered) // 2],
        "min_sec": ordered[0],
        "max_sec": ordered[-1],
        "total_sec": sum(times),
    }


def _print_timing(extract: dict, judge: dict, wall_sec: float) -> None:
    """Print per-stage timing so runtime can be assessed at a glance."""
    print("\n  Timing (per LLM call, cache-reused excluded):")
    for label, t in (("extract", extract), ("judge", judge)):
        if t.get("llm_calls"):
            print(
                f"    {label:<8} {t['llm_calls']:>4} calls | "
                f"mean {t['mean_sec']:.1f}s | median {t['median_sec']:.1f}s | "
                f"min {t['min_sec']:.1f}s | max {t['max_sec']:.1f}s"
            )
        else:
            print(f"    {label:<8}   (all reused from cache)")
    print(f"    wall     {wall_sec:.1f}s ({wall_sec / 60:.1f} min) total")


def _print_soap_summary(summary: dict, n_eval: int, n_fail: int) -> None:
    """Print a compact aggregate summary to stdout."""
    print("\nSOAP LLM-Judge Results")
    print("=" * 60)
    print(f"Backend: {summary.get('backend')}   Model: {summary.get('model')}")
    print(f"Judged: {n_eval}   Failed to parse: {n_fail}")
    if n_eval:
        print("\n  Subscores (1-5, mean ± std):")
        for field in ("faithfulness", "structure", "coverage", "conciseness"):
            mean = summary.get(f"mean_{field}", 0.0)
            std = summary.get(f"std_{field}", 0.0)
            print(f"    {field:<14} {mean:5.2f} ± {std:.2f}")
        print("\n  Error rates / counts (mean):")
        print(f"    hallucination_rate   {summary.get('mean_hallucination_rate', 0.0):.3f}")
        print(f"    contradiction_rate   {summary.get('mean_contradiction_rate', 0.0):.3f}")
        print(f"    over_medicalization  {summary.get('mean_over_medicalization', 0.0):.2f}")
        print(f"    critical_omissions   {summary.get('mean_critical_omissions', 0.0):.2f}")
    print("=" * 60)


def cmd_soap_judge(args: argparse.Namespace) -> int:
    """Run the SOAP LLM-judge pipeline. Returns 0 on success, 1 on error."""
    from btc_eval.io import (
        load_summaries_jsonl,
        load_transcripts_jsonl,
        write_results_csv,
        write_results_json,
        write_results_jsonl,
    )
    from btc_eval.soap_judge import pipeline
    from btc_eval.soap_judge.backends import available_backends, get_backend
    from btc_eval.soap_judge.parse import judgment_to_scores
    from btc_eval.types import SOAP_CSV_COLUMNS

    predictions = load_summaries_jsonl(args.predictions)
    if not predictions:
        print("Error: no predictions loaded", file=sys.stderr)
        return 1
    if args.shard:
        try:
            idx, cnt = pipeline.parse_shard_spec(args.shard)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        keep = set(pipeline.select_shard(predictions.keys(), idx, cnt))
        predictions = {k: v for k, v in predictions.items() if k in keep}
        print(f"Shard {idx}/{cnt}: {len(predictions)} dialogs in this shard")
        if not predictions:
            print("Error: shard is empty", file=sys.stderr)
            return 1
    transcripts = load_transcripts_jsonl(args.transcripts)
    if not transcripts:
        print("Error: no transcripts loaded", file=sys.stderr)
        return 1

    try:
        backend = get_backend(
            args.backend,
            host=args.host,
            region=args.region,
            provider_name=args.provider_name,
        )
    except ValueError as exc:
        print(f"Error: {exc} (available: {available_backends()})", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resume cache: separate from the per-shard output so many shards can share one
    # id-keyed cache (robust resume regardless of how dialogs map to shards).
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir
    extract_raw = None if args.no_cache else cache_dir / "extract_raw"
    judge_raw = None if args.no_cache else cache_dir / "judge_raw"
    resume = not args.no_resume

    print(
        f"Backend: {args.backend} | model: {args.model} | "
        f"{len(predictions)} predictions, {len(transcripts)} transcripts"
    )

    wall_start = time.perf_counter()
    # Stage 1: extract atomic claims from each note. Stage 2: judge note vs
    # transcript using those claims (Amy & Andrew's two-stage design).
    claims = pipeline.run_extract(
        predictions, backend, args.model,
        workers=args.workers, max_tokens=args.max_tokens,
        raw_dir=extract_raw, prompt_dir=args.prompt_dir, resume=resume,
    )
    # Surface stage-1 parse failures: a failed extraction silently degrades the
    # judge to an empty claim set (so faithfulness/hallucination look clean).
    extract_failed = [cid for cid, r in claims.items() if r.get("status") != "parsed"]
    if extract_failed:
        print(
            f"Warning: {len(extract_failed)}/{len(claims)} claim extractions did "
            f"not parse — those notes are judged on an EMPTY claim set, which "
            f"inflates faithfulness. Raise --max-tokens (reasoning models need "
            f"20000+). First few: {extract_failed[:5]}"
            f"{'...' if len(extract_failed) > 5 else ''}",
            file=sys.stderr,
        )
    judged = pipeline.run_judge(
        predictions, transcripts, claims, backend, args.model,
        workers=args.workers, max_tokens=args.max_tokens,
        raw_dir=judge_raw, prompt_dir=args.prompt_dir, resume=resume,
    )

    per_dialog: list[dict] = []
    scores_list = []
    failures: list[dict] = []
    for dialog_id in sorted(judged):
        result = judged[dialog_id]
        judgment = result.get("judgment")
        if judgment is None:
            failures.append({
                "id": dialog_id,
                "status": result.get("status"),
                "error": result.get("error"),
                "raw_content": (result.get("raw_content") or "")[:2000],
            })
            continue
        scores = judgment_to_scores(judgment)
        scores_list.append(scores)
        row = scores.to_dict()
        row["id"] = dialog_id
        per_dialog.append(row)

    agg = pipeline.aggregate_soap(scores_list)
    extract_timing = _timing_stats(claims)
    judge_timing = _timing_stats(judged)
    wall_sec = time.perf_counter() - wall_start
    summary = {
        "backend": args.backend,
        "model": args.model,
        **agg,
        "num_failures": len(failures),
        "extract_parse_failures": len(extract_failed),
        "timing": {
            "wall_sec": wall_sec,
            "extract": extract_timing,
            "judge": judge_timing,
        },
    }

    write_results_csv(per_dialog, out_dir / "soap_eval_summary.csv",
                      fieldnames=list(SOAP_CSV_COLUMNS))
    write_results_jsonl(per_dialog, out_dir / "soap_judge_per_dialog.jsonl")
    write_results_json(summary, out_dir / "summary.json")
    if failures:
        write_results_jsonl(failures, out_dir / "failures.jsonl")

    _print_soap_summary(summary, n_eval=len(per_dialog), n_fail=len(failures))
    _print_timing(extract_timing, judge_timing, wall_sec)
    print(f"\nResults saved to {out_dir}/")
    return 0


def add_soap_aggregate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``soap-aggregate`` subparser (merge sharded per-dialog rows)."""
    parser = subparsers.add_parser(
        "soap-aggregate",
        help="Merge per-dialog soap-judge outputs (e.g. SLURM shards) and "
        "recompute the team aggregate",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="GLOB",
        help="Per-dialog JSONL files or globs, e.g. "
        "'out/team/shard_*/soap_judge_per_dialog.jsonl'",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for the merged CSV/JSONL + summary.json",
    )
    parser.set_defaults(func=cmd_soap_aggregate)


def cmd_soap_aggregate(args: argparse.Namespace) -> int:
    """Pool per-dialog rows from many shards and recompute the aggregate.

    Aggregates over the union of dialogs (deduped by id) — the correct way to
    combine shards, vs. averaging per-shard means (which breaks std and unequal
    shard sizes).
    """
    import glob
    import json
    import os

    from btc_eval.io import write_results_csv, write_results_json, write_results_jsonl
    from btc_eval.soap_judge.pipeline import aggregate_soap
    from btc_eval.types import SOAP_CSV_COLUMNS, SoapScores

    paths: list[str] = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        if matched:
            paths.extend(matched)
        elif os.path.exists(pattern):
            paths.append(pattern)
    paths = sorted(set(paths))
    if not paths:
        print(f"Error: no input files matched {args.inputs}", file=sys.stderr)
        return 1

    rows: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in record:
                    rows[record["id"]] = record  # dedup by id across shards

    fields = [c for c in SOAP_CSV_COLUMNS if c != "id"]
    valid_rows: list[dict] = []
    scores: list[SoapScores] = []
    skipped = 0
    for record in rows.values():
        try:
            scores.append(SoapScores(**{f: record[f] for f in fields}))
        except KeyError:
            skipped += 1
            continue
        valid_rows.append(record)
    if not scores:
        print("Error: no per-dialog rows with score fields found", file=sys.stderr)
        return 1

    agg = aggregate_soap(scores)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_results_jsonl(valid_rows, out_dir / "soap_judge_per_dialog.jsonl")
    write_results_csv(valid_rows, out_dir / "soap_eval_summary.csv",
                      fieldnames=list(SOAP_CSV_COLUMNS))
    write_results_json({"num_dialogs": len(scores), **agg}, out_dir / "summary.json")

    note = f" ({skipped} rows skipped: missing score fields)" if skipped else ""
    print(f"Aggregated {len(paths)} files -> {len(scores)} unique dialogs{note}")
    print(
        f"  mean: faithfulness {agg.get('mean_faithfulness', 0):.2f}, "
        f"structure {agg.get('mean_structure', 0):.2f}, "
        f"coverage {agg.get('mean_coverage', 0):.2f}, "
        f"conciseness {agg.get('mean_conciseness', 0):.2f}"
    )
    print(f"Results saved to {out_dir}/")
    return 0
