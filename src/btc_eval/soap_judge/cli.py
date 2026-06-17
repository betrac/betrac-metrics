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
        choices=["ollama", "vllm", "openrouter", "bedrock", "anthropic", "mock"],
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
        "--judge-max-tokens",
        type=int,
        default=None,
        help="Max output tokens for the JUDGE stage only (default: --max-tokens). "
        "The judgment is short (~5k), while its prompt is long; backends like vLLM "
        "reject prompt+max_tokens > max_model_len, so a smaller judge cap avoids "
        "that without truncating the (short) judgment.",
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
        "dialog is reused no matter which shard re-processes it (the cache is keyed "
        "by dialog id AND validated against --model, so a different model/precision "
        "never reuses another run's cached judgments).",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete the resume cache (extract_raw/ + judge_raw/) before running — "
        "for a clean re-run with the SAME model. A different model is already "
        "isolated automatically (the cache is model-keyed), so this is rarely needed.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="On resume, re-issue LLM calls whose cached response no longer parses "
        "(e.g. after raising --judge-max-tokens to fix truncations) instead of "
        "reusing the broken cached text.",
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
        if summary.get("n_excluded_empty_claims"):
            print(f"    (faithfulness/hallucination over {summary['n_faithfulness']}/{n_eval}; "
                  f"{summary['n_excluded_empty_claims']} excluded — extraction failed)")
        print("\n  Error rates / counts (mean):")
        print(f"    hallucination_rate   {summary.get('mean_hallucination_rate', 0.0):.3f}")
        print(f"    contradiction_rate   {summary.get('mean_contradiction_rate', 0.0):.3f}")
        print(f"    over_medicalization  {summary.get('mean_over_medicalization', 0.0):.2f}")
        print(f"    critical_omissions   {summary.get('mean_critical_omissions', 0.0):.2f}")
    print("=" * 60)


# --- Known-error reporting -------------------------------------------------
# Every dialog gets a status record; these are consolidated (here and in
# soap-aggregate) into a report of how often each known error affected a result
# and exactly which dialogs — so a silent failure (parse drop, empty-claims
# faithfulness inflation, LLM error) is always visible in the end result.

def _status_records(claims: dict, judged: dict) -> list[dict]:
    """One status record per judged dialog (written to soap_judge_status.jsonl)."""
    records = []
    for did in sorted(judged):
        jr = judged[did]
        er = claims.get(did, {})
        records.append({
            "id": did,
            "judge_status": jr.get("status"),       # parsed / parse_failed / llm_error
            "judge_parse": jr.get("parse_status"),  # strict / repaired / no_subscores / unparseable / empty / llm_error
            "extract_status": er.get("status"),     # parsed / parse_failed / llm_error
            "empty_claims": bool(jr.get("empty_claims")),
        })
    return records


def _buckets_from_status(records: list[dict]) -> tuple[dict, dict]:
    """Bucket per-dialog status into ``(errors, coverage)`` for the summary.

    ``errors`` maps a known-error kind -> ``{"count", "ids"}``; ``coverage``
    reports attempted / scored / failed. ``judge_repaired`` is informational (the
    dialog IS scored, recovered via json-repair); ``extract_failed_empty_claims``
    flags dialogs whose faithfulness is likely inflated (judged on empty claims).
    """
    from collections import defaultdict
    buckets: dict[str, list[str]] = defaultdict(list)
    scored = 0
    for r in records:
        did = r.get("id")
        jstatus = r.get("judge_status")
        jparse = r.get("judge_parse")
        if jstatus == "parsed":
            scored += 1
            if jparse == "repaired":
                buckets["judge_repaired"].append(did)
        elif jstatus == "llm_error":
            buckets["judge_llm_error"].append(did)
        else:  # parse_failed
            buckets[f"judge_{jparse or 'unparseable'}"].append(did)
        if r.get("empty_claims"):
            buckets["extract_failed_empty_claims"].append(did)
    errors = {k: {"count": len(v), "ids": v} for k, v in sorted(buckets.items())}
    coverage = {"attempted": len(records), "scored": scored,
                "failed": len(records) - scored}
    return errors, coverage


def _print_error_report(errors: dict, coverage: dict) -> None:
    """Print the known-error report (counts + a few affected ids)."""
    if coverage:
        print(f"\n  Coverage: {coverage.get('scored')}/{coverage.get('attempted')} "
              f"scored, {coverage.get('failed')} failed")
    if not errors:
        return
    print("  Known errors affecting results (kind: count [first ids]):")
    for kind, info in errors.items():
        ids = info.get("ids", [])
        shown = ", ".join(map(str, ids[:5])) + ("…" if len(ids) > 5 else "")
        flag = ("  ⚠ faithfulness likely inflated" if kind == "extract_failed_empty_claims"
                else "  (recovered via repair)" if kind == "judge_repaired" else "")
        print(f"    {kind:<28} {info.get('count'):>4}  [{shown}]{flag}")


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
    if args.clear_cache:
        import shutil
        for d in (extract_raw, judge_raw):
            if d and d.exists():
                shutil.rmtree(d, ignore_errors=True)
        print("Cleared resume cache (extract_raw/, judge_raw/)")

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
        retry_failed=args.retry_failed,
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
        workers=args.workers, max_tokens=(args.judge_max_tokens or args.max_tokens),
        raw_dir=judge_raw, prompt_dir=args.prompt_dir, resume=resume,
        retry_failed=args.retry_failed,
    )

    per_dialog: list[dict] = []
    scores_list = []
    empty_claims_list: list[bool] = []
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
        empty_claims_list.append(bool(result.get("empty_claims")))
        row = scores.to_dict()
        row["id"] = dialog_id
        per_dialog.append(row)

    # Exclude empty-claims dialogs (extraction failed) from faithfulness/hallucination.
    agg = pipeline.aggregate_soap(scores_list, empty_claims_list)
    status_records = _status_records(claims, judged)
    errors, coverage = _buckets_from_status(status_records)
    extract_timing = _timing_stats(claims)
    judge_timing = _timing_stats(judged)
    wall_sec = time.perf_counter() - wall_start
    summary = {
        "backend": args.backend,
        "model": args.model,
        **agg,
        "num_failures": len(failures),
        "extract_parse_failures": len(extract_failed),
        "coverage": coverage,   # attempted / scored / failed
        "errors": errors,       # per known-error kind: count + affected dialog ids
        "timing": {
            "wall_sec": wall_sec,
            "extract": extract_timing,
            "judge": judge_timing,
        },
    }

    write_results_csv(per_dialog, out_dir / "soap_eval_summary.csv",
                      fieldnames=list(SOAP_CSV_COLUMNS))
    write_results_jsonl(per_dialog, out_dir / "soap_judge_per_dialog.jsonl")
    # Per-dialog status for EVERY attempted dialog (incl. failures) — soap-aggregate
    # reads these to build the consolidated end-result error report.
    write_results_jsonl(status_records, out_dir / "soap_judge_status.jsonl")
    write_results_json(summary, out_dir / "summary.json")
    if failures:
        write_results_jsonl(failures, out_dir / "failures.jsonl")

    _print_soap_summary(summary, n_eval=len(per_dialog), n_fail=len(failures))
    _print_error_report(errors, coverage)
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

    # Consolidate per-shard status (a soap_judge_status.jsonl beside each input):
    # the END-result error report (how often each known error hit a result + which
    # dialogs) AND the empty_claims flag used to exclude extraction-failed dialogs
    # from faithfulness/hallucination. Without these files (older runs) we report
    # only the scored count and apply no exclusion.
    status_by_id: dict[str, dict] = {}
    for path in paths:
        status_path = os.path.join(os.path.dirname(path), "soap_judge_status.jsonl")
        if not os.path.exists(status_path):
            continue
        with open(status_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in rec:
                    status_by_id[rec["id"]] = rec

    empty_claims = [bool(status_by_id.get(r["id"], {}).get("empty_claims")) for r in valid_rows]
    agg = aggregate_soap(scores, empty_claims)

    if status_by_id:
        errors, coverage = _buckets_from_status(list(status_by_id.values()))
    else:
        errors = {}
        coverage = {"attempted": len(scores), "scored": len(scores), "failed": 0,
                    "note": "no soap_judge_status.jsonl found; failures not visible"}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_results_jsonl(valid_rows, out_dir / "soap_judge_per_dialog.jsonl")
    write_results_csv(valid_rows, out_dir / "soap_eval_summary.csv",
                      fieldnames=list(SOAP_CSV_COLUMNS))
    if status_by_id:
        write_results_jsonl(list(status_by_id.values()), out_dir / "soap_judge_status.jsonl")
    write_results_json(
        {"num_dialogs": len(scores), **agg, "coverage": coverage, "errors": errors},
        out_dir / "summary.json",
    )

    note = f" ({skipped} rows skipped: missing score fields)" if skipped else ""
    print(f"Aggregated {len(paths)} files -> {len(scores)} unique dialogs{note}")
    print(
        f"  mean: faithfulness {agg.get('mean_faithfulness', 0):.2f}, "
        f"structure {agg.get('mean_structure', 0):.2f}, "
        f"coverage {agg.get('mean_coverage', 0):.2f}, "
        f"conciseness {agg.get('mean_conciseness', 0):.2f}"
    )
    if agg.get("n_excluded_empty_claims"):
        print(f"  (faithfulness/hallucination over {agg['n_faithfulness']}/{len(scores)}; "
              f"{agg['n_excluded_empty_claims']} excluded — extraction failed)")
    _print_error_report(errors, coverage)
    print(f"Results saved to {out_dir}/")
    return 0
