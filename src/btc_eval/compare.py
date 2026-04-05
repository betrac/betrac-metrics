"""Batch comparison of ROUGE and concept-F1 across prediction files.

This module is intended for **organizer** use — comparing multiple systems
in bulk. Participants should use ``btc-eval evaluate`` for single-system scoring.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from btc_eval.io import load_summaries_jsonl
from btc_eval.metrics.concept_f1 import (
    aggregate_metrics,
    compute_concept_metrics,
    extract_concept_ids,
)


# ---------------------------------------------------------------------------
# Manifest / filename parsing
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load a JSON manifest file describing prediction files.

    Each entry must have a ``"file"`` key (filename relative to the predictions
    directory).  All other keys are treated as label columns.
    """
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)
    for entry in entries:
        if "file" not in entry:
            raise ValueError(f"Manifest entry missing 'file' key: {entry}")
    return entries


def parse_filename(path: Path) -> dict[str, str]:
    """Extract audio format, pipeline, and model from a prediction filename.

    Fallback when no manifest is available.
    Expected: summaries-DD-{N}-{audio}-{pipeline}-{model}.jsonl
    """
    stem = path.stem
    m = re.match(r"summaries-DD-\d+-(opus|flac)-(Cascade|E2E)-(.*)", stem)
    if not m:
        return {"Audio": "?", "Pipeline": "?", "Model": stem}
    return {"Audio": m.group(1), "Pipeline": m.group(2), "Model": m.group(3)}


def discover_predictions(pred_dir: Path, manifest_path: Path | None) -> list[dict]:
    """Return a list of dicts with ``"_path"`` and label columns.

    If *manifest_path* exists, load it; otherwise glob for ``*.jsonl`` and
    fall back to filename parsing.
    """
    # Try manifest
    if manifest_path is None:
        manifest_path = pred_dir / "manifest.json"
    if manifest_path.is_file():
        entries = load_manifest(manifest_path)
        for entry in entries:
            entry["_path"] = pred_dir / entry.pop("file")
        return entries

    # Fallback: glob + filename parsing
    pred_files = sorted(pred_dir.glob("*.jsonl"))
    entries = []
    for pf in pred_files:
        entry = parse_filename(pf)
        entry["_path"] = pf
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def compute_word_counts(
    references: dict[str, str], predictions: dict[str, str], shared_ids: list[str]
) -> tuple[float, float]:
    """Return mean word counts for reference and hypothesis."""
    ref_wc = [len(references[did].split()) for did in shared_ids]
    hyp_wc = [len(predictions[did].split()) for did in shared_ids]
    n = len(shared_ids)
    return sum(ref_wc) / n, sum(hyp_wc) / n


def run_rouge(
    references: dict[str, str], predictions: dict[str, str], shared_ids: list[str]
) -> dict[str, float]:
    """Compute aggregate ROUGE scores. Returns dict with R-2/3/L F-means."""
    from btc_eval.metrics.rouge import aggregate_rouge, compute_rouge, _get_scorer

    scorer = _get_scorer(use_stemmer=False)
    results = []
    for did in shared_ids:
        results.append(compute_rouge(references[did], predictions[did], scorer=scorer))
    agg = aggregate_rouge(results)
    return {
        "R-2": agg["rouge2_f_mean"],
        "R-3": agg["rouge3_f_mean"],
        "R-L": agg["rougeL_f_mean"],
    }


def run_concept_f1(
    references: dict[str, str],
    predictions: dict[str, str],
    shared_ids: list[str],
    use_scispacy: bool = False,
) -> dict[str, float]:
    """Compute aggregate concept F1. Returns dict with F1, Precision, Recall."""
    from btc_eval.matchers.open_medical import OpenMedicalMatcher

    matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=use_scispacy)
    per_dialog = []
    for did in shared_ids:
        ref_concepts = extract_concept_ids(references[did], matcher)
        hyp_concepts = extract_concept_ids(predictions[did], matcher)
        per_dialog.append(compute_concept_metrics(ref_concepts, hyp_concepts))
    agg = aggregate_metrics(per_dialog)
    return {"F1": agg.mean_f1, "Prec": agg.mean_precision, "Rec": agg.mean_recall}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

LATEX_HEADER_MAP = {
    "Mesh F1": "Mesh F1",
    "Mesh Prec": "Mesh P",
    "Mesh Rec": "Mesh R",
    "Open F1": "Open F1",
    "Open Prec": "Open P",
    "Open Rec": "Open R",
    "RefWds": "Ref Wds",
    "HypWds": "Hyp Wds",
}


def _fmt_val(val) -> str:
    """Format a value for CSV/LaTeX output."""
    if isinstance(val, float):
        if val > 100:
            return f"{val:.1f}"
        return f"{val:.4f}"
    return str(val)


def write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    """Write comparison table as CSV (rounded to 4 decimal places)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt_val(row.get(col, "")) for col in columns])
    print(f"  CSV:   {path}")


def write_latex(rows: list[dict], columns: list[str], label_columns: list[str], path: Path) -> None:
    """Write comparison table as a LaTeX tabular (booktabs)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    n_text = len(label_columns)
    align = "l" * n_text + "r" * (len(columns) - n_text)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Evaluation comparison across note generation configurations.}\n")
        f.write("\\label{tab:comparison}\n")
        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write("\\toprule\n")

        headers = [LATEX_HEADER_MAP.get(c, c) for c in columns]
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")

        for row in rows:
            cells = [_fmt_val(row.get(col, "")) for col in columns]
            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  LaTeX: {path}")


def print_markdown(rows: list[dict], columns: list[str]) -> None:
    """Print a markdown comparison table to stdout."""
    widths = {}
    for col in columns:
        w = len(col)
        for row in rows:
            val = row.get(col, "")
            if isinstance(val, float):
                w = max(w, len(f"{val:.3f}"))
            else:
                w = max(w, len(str(val)))
        widths[col] = w

    header = "| " + " | ".join(f"{col:<{widths[col]}}" for col in columns) + " |"
    separator = "|-" + "-|-".join("-" * widths[col] for col in columns) + "-|"
    print(header)
    print(separator)

    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            w = widths[col]
            if isinstance(val, float):
                cells.append(f"{val:{w}.3f}")
            else:
                cells.append(f"{str(val):<{w}}")
        print("| " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------


def run_comparison(
    references_path: str | Path,
    predictions_dir: str | Path,
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    skip_rouge: bool = False,
    skip_concepts: bool = False,
    open_medical: bool = False,
) -> int:
    """Run batch comparison. Returns 0 on success, 1 on error."""
    pred_dir = Path(predictions_dir)
    entries = discover_predictions(
        pred_dir,
        Path(manifest_path) if manifest_path else None,
    )
    if not entries:
        print(f"Error: no prediction files found in {pred_dir}", file=sys.stderr)
        return 1

    print(f"Loading references from {references_path}...")
    references = load_summaries_jsonl(references_path)
    print(f"  {len(references)} references loaded")
    print(f"Found {len(entries)} prediction files\n")

    # Determine label columns (everything except _path and metric columns)
    label_columns = [k for k in entries[0] if k != "_path"]

    # Build full column list
    columns = list(label_columns)
    if not skip_rouge:
        columns.extend(["R-2", "R-3", "R-L"])
    if not skip_concepts:
        columns.extend(["Mesh F1", "Mesh Prec", "Mesh Rec"])
    if open_medical:
        columns.extend(["Open F1", "Open Prec", "Open Rec"])
    columns.extend(["RefWds", "HypWds"])

    # Evaluate each prediction file
    rows: list[dict] = []
    for entry in entries:
        pred_path = entry["_path"]
        labels = {k: v for k, v in entry.items() if k != "_path"}
        print(f"Evaluating {pred_path.name} ...", end="", flush=True)

        predictions = load_summaries_jsonl(pred_path)
        shared_ids = sorted(set(predictions) & set(references))
        if not shared_ids:
            print(" SKIP (no matching IDs)")
            continue

        row: dict = dict(labels)

        ref_wds, hyp_wds = compute_word_counts(references, predictions, shared_ids)
        row["RefWds"] = int(round(ref_wds))
        row["HypWds"] = int(round(hyp_wds))

        if not skip_rouge:
            rouge = run_rouge(references, predictions, shared_ids)
            row.update(rouge)
            print(f" R-L={rouge['R-L']:.3f}", end="", flush=True)

        if not skip_concepts:
            mesh = run_concept_f1(references, predictions, shared_ids, use_scispacy=False)
            row["Mesh F1"] = mesh["F1"]
            row["Mesh Prec"] = mesh["Prec"]
            row["Mesh Rec"] = mesh["Rec"]
            print(f" MeshF1={mesh['F1']:.3f}", end="", flush=True)

        if open_medical:
            om = run_concept_f1(references, predictions, shared_ids, use_scispacy=True)
            row["Open F1"] = om["F1"]
            row["Open Prec"] = om["Prec"]
            row["Open Rec"] = om["Rec"]
            print(f" OpenF1={om['F1']:.3f}", end="", flush=True)

        print()
        rows.append(row)

    # Sort by Mesh F1 descending (if available), else by R-L
    sort_key = "Mesh F1" if "Mesh F1" in columns else "R-L"
    rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)

    # Print markdown table
    print("\n" + "=" * 80)
    print_markdown(rows, columns)
    print("=" * 80)

    # Write files
    if output_dir:
        out = Path(output_dir)
        print(f"\nWriting results to {out}/")
        write_csv(rows, columns, out / "comparison.csv")
        write_latex(rows, columns, label_columns, out / "comparison.tex")

    return 0
