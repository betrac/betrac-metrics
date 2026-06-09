"""I/O utilities for loading summaries and writing results."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def load_summaries_jsonl(path: str | Path) -> dict[str, str]:
    """Load a JSONL file with 'id' and 'summary' fields.

    Returns a dict mapping dialog ID to summary text.
    Lines missing 'id' or 'summary' fields are skipped with a warning.
    """
    summaries: dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        log.error("File not found: %s", path)
        return summaries
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Line %d in %s is not valid JSON, skipping", line_num, path)
                continue
            dialog_id = record.get("id")
            summary = record.get("summary")
            if dialog_id is None or summary is None:
                log.warning("Line %d in %s missing 'id' or 'summary', skipping", line_num, path)
                continue
            if dialog_id in summaries:
                log.warning("Duplicate ID '%s' at line %d in %s (overwriting previous)",
                            dialog_id, line_num, path)
            summaries[dialog_id] = summary
    return summaries


def load_transcripts_jsonl(path: str | Path) -> dict[str, str]:
    """Load a JSONL file with 'id' and 'transcript' fields.

    Returns a dict mapping dialog ID to transcript text. Used by the SOAP judge,
    which scores a note against the doctor-patient transcript it was written from.
    Lines missing 'id' or 'transcript' are skipped with a warning.
    """
    transcripts: dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        log.error("File not found: %s", path)
        return transcripts
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Line %d in %s is not valid JSON, skipping", line_num, path)
                continue
            dialog_id = record.get("id")
            transcript = record.get("transcript")
            if dialog_id is None or transcript is None:
                log.warning("Line %d in %s missing 'id' or 'transcript', skipping",
                            line_num, path)
                continue
            if dialog_id in transcripts:
                log.warning("Duplicate ID '%s' at line %d in %s (overwriting previous)",
                            dialog_id, line_num, path)
            transcripts[dialog_id] = transcript
    return transcripts


def load_references_from_dir(
    dir_path: str | Path,
    filter_ids: set[str] | None = None,
) -> dict[str, str]:
    """Load .txt reference files from a directory.

    Each file's stem is the dialog ID, full text is the summary.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        log.error("Directory not found: %s", dir_path)
        return {}
    summaries: dict[str, str] = {}
    for txt_file in sorted(dir_path.glob("*.txt")):
        dialog_id = txt_file.stem
        if filter_ids and dialog_id not in filter_ids:
            continue
        summaries[dialog_id] = txt_file.read_text(encoding="utf-8").strip()
    return summaries


_CACHE_DIR = Path.home() / ".cache" / "btc-eval"


def load_references_from_hf(
    dataset: str = "BeTraC/betrac-2026",
    split: str = "validation",
    cache: bool = True,
) -> dict[str, str]:
    """Load reference SOAP notes from HuggingFace dataset.

    On first call, streams the dataset and caches the extracted references
    as a JSONL file in ~/.cache/btc-eval/. Subsequent calls load from cache.
    Use cache=False to force re-download.

    Requires the 'datasets' library: pip install 'btc-eval[hf]'
    """
    # Check cache first
    if cache:
        cached = _get_cache_path(dataset, split)
        if cached.exists():
            log.info("Loading cached references from %s", cached)
            return load_summaries_jsonl(cached)

    # Download from HuggingFace (check datasets is available first)
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise ImportError(
            "'datasets' library is required for HuggingFace loading. "
            "Install with:  uv pip install '.[hf]'"
        )

    summaries = _stream_references_from_hf(dataset, split)

    # Write cache
    if cache and summaries:
        _write_cache(summaries, dataset, split)

    return summaries


def _get_cache_path(dataset: str, split: str) -> Path:
    """Return the cache file path for a dataset/split combination."""
    safe_name = dataset.replace("/", "_")
    return _CACHE_DIR / f"references-{safe_name}-{split}.jsonl"


def _write_cache(summaries: dict[str, str], dataset: str, split: str) -> None:
    """Write references to the local cache as JSONL."""
    cached = _get_cache_path(dataset, split)
    cached.parent.mkdir(parents=True, exist_ok=True)
    with open(cached, "w", encoding="utf-8") as f:
        for dialog_id, summary in sorted(summaries.items()):
            f.write(json.dumps({"id": dialog_id, "summary": summary}, ensure_ascii=False) + "\n")
    log.info("Cached %d references to %s", len(summaries), cached)


def _stream_references_from_hf(dataset: str, split: str) -> dict[str, str]:
    """Stream references from HuggingFace WebDataset (no caching)."""
    import ast

    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    summaries: dict[str, str] = {}

    for i, item in enumerate(ds):
        # Extract sample ID from the json metadata field
        sample_id = f"sample_{i:04d}"
        meta_raw = item.get("json", "")
        if isinstance(meta_raw, dict):
            sample_id = str(meta_raw.get("id", sample_id))
        elif isinstance(meta_raw, (str, bytes, bytearray)):
            if isinstance(meta_raw, (bytes, bytearray)):
                meta_raw = meta_raw.decode("utf-8")
            if meta_raw:
                try:
                    meta = json.loads(meta_raw)
                    sample_id = str(meta.get("id", sample_id))
                except json.JSONDecodeError:
                    try:
                        meta = ast.literal_eval(meta_raw)
                        sample_id = str(meta.get("id", sample_id))
                    except (ValueError, SyntaxError):
                        pass

        # Extract SOAP note text
        soap = item.get("soap.txt", "")
        if not soap:
            log.warning("Item %d (%s) has no soap.txt, skipping", i, sample_id)
            continue
        if isinstance(soap, (bytes, bytearray)):
            soap = soap.decode("utf-8")

        summaries[sample_id] = soap

    log.info("Loaded %d references from %s [%s]", len(summaries), dataset, split)
    return summaries


def _get_transcript_cache_path(dataset: str, split: str) -> Path:
    """Return the cache file path for a dataset/split's reference transcripts."""
    safe_name = dataset.replace("/", "_")
    return _CACHE_DIR / f"transcripts-{safe_name}-{split}.jsonl"


def _stream_transcripts_from_hf(dataset: str, split: str) -> dict[str, str]:
    """Stream ground-truth reference transcripts from the HuggingFace WebDataset.

    The value is the dataset's ``transcript.txt`` — the labeled DOCTOR/PATIENT
    reference dialogue, NOT an ASR hypothesis. Keyed by the sample's canonical
    UUID (from the ``json`` metadata), the same id scheme as the references.
    Use ``scripts/map_ids.py`` to align ``dialog_*`` prediction ids if needed.
    """
    import ast

    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    transcripts: dict[str, str] = {}

    for i, item in enumerate(ds):
        sample_id = f"sample_{i:04d}"
        meta_raw = item.get("json", "")
        if isinstance(meta_raw, dict):
            sample_id = str(meta_raw.get("id", sample_id))
        elif isinstance(meta_raw, (str, bytes, bytearray)):
            if isinstance(meta_raw, (bytes, bytearray)):
                meta_raw = meta_raw.decode("utf-8")
            if meta_raw:
                try:
                    sample_id = str(json.loads(meta_raw).get("id", sample_id))
                except json.JSONDecodeError:
                    try:
                        sample_id = str(ast.literal_eval(meta_raw).get("id", sample_id))
                    except (ValueError, SyntaxError):
                        pass

        transcript = item.get("transcript.txt", "")
        if not transcript:
            log.warning("Item %d (%s) has no transcript.txt, skipping", i, sample_id)
            continue
        if isinstance(transcript, (bytes, bytearray)):
            transcript = transcript.decode("utf-8")

        transcripts[sample_id] = transcript

    log.info("Loaded %d transcripts from %s [%s]", len(transcripts), dataset, split)
    return transcripts


def build_reference_jsonl(
    input_dir: str | Path,
    output_path: str | Path,
    filter_ids: set[str] | None = None,
) -> int:
    """Read .txt files from input_dir and write a JSONL file.

    Returns the number of references written.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for txt_file in sorted(input_dir.glob("*.txt")):
            dialog_id = txt_file.stem
            if filter_ids and dialog_id not in filter_ids:
                continue
            summary = txt_file.read_text(encoding="utf-8").strip()
            record = {"id": dialog_id, "summary": summary}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_results_json(results: dict, path: str | Path) -> None:
    """Write evaluation results as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def write_results_csv(
    rows: list[dict],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> None:
    """Write per-dialog results as a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_jsonl(rows: list[dict], path: str | Path) -> None:
    """Write per-dialog results as a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def print_compact_summary(
    concept_agg: dict,
    rouge_agg: dict,
    team: str | None = None,
    system: str | None = None,
    split: str | None = None,
    num_dialogs: int = 0,
    ci_info: dict | None = None,
    matcher_label: str | None = None,
    pred_stats: dict | None = None,
    file=None,
) -> None:
    """Print a compact summary of competition metrics."""
    if file is None:
        file = sys.stdout

    print("\nBeTraC 2026 Evaluation Results", file=file)
    print("=" * 70, file=file)
    if team:
        print(f"Team: {team}", file=file)
    if system:
        print(f"System: {system}", file=file)
    if split:
        print(f"Split: {split}", file=file)
    print(f"Dialogs: {num_dialogs}", file=file)
    if matcher_label:
        print(f"Matcher: {matcher_label}", file=file)

    metrics = [
        (
            "concept_f1",
            "Concept F1",
            concept_agg.get("mean_f1", 0),
            concept_agg.get("mean_precision", 0),
            concept_agg.get("mean_recall", 0),
        ),
        (
            "rouge2_f",
            "ROUGE-2 F1",
            rouge_agg.get("rouge2_f_mean", 0),
            rouge_agg.get("rouge2_p_mean", 0),
            rouge_agg.get("rouge2_r_mean", 0),
        ),
        (
            "rouge3_f",
            "ROUGE-3 F1",
            rouge_agg.get("rouge3_f_mean", 0),
            rouge_agg.get("rouge3_p_mean", 0),
            rouge_agg.get("rouge3_r_mean", 0),
        ),
    ]
    first = True
    for key, label, f1v, pv, rv in metrics:
        line = f"  {label}:  {f1v:.4f}"
        if ci_info:
            bci_lo_key = f"{key}_bci_lo"
            if bci_lo_key in ci_info:
                line += f" [{ci_info[bci_lo_key]:.4f}, {ci_info[f'{key}_bci_hi']:.4f}]"
            aci_key = f"{key}_aci_half"
            if aci_key in ci_info:
                line += f" \u00b1{ci_info[aci_key]:.4f}"
        line += f"  (P={pv:.4f}, R={rv:.4f})"
        if first:
            print(f"\n{line}", file=file)
            first = False
        else:
            print(line, file=file)

    if pred_stats:
        print(f"\n  Avg prediction length: "
              f"{pred_stats['mean_pred_words']:.1f} words, "
              f"{pred_stats['mean_pred_chars']:.1f} non-whitespace chars",
              file=file)

    print("=" * 70, file=file)


def print_results_table(
    per_dialog: list[dict],
    aggregate: dict,
    file=None,
    summary_only: bool = False,
) -> None:
    """Print a formatted results table."""
    if file is None:
        file = sys.stdout

    print("\n" + "=" * 80, file=file)
    print("MEDICAL CONCEPT EVALUATION RESULTS", file=file)
    print("=" * 80, file=file)

    print(
        f"\n{'':25} {'F1':>8} {'Prec':>8} {'Recall':>8}"
        f" {'#Ref':>5} {'#Hyp':>5} {'#Olap':>5}"
        f" {'RefWds':>7} {'HypWds':>7}",
        file=file,
    )
    print("-" * 80, file=file)

    if not summary_only:
        for d in per_dialog:
            print(
                f"{d['id']:<25} {d['f1']:>8.4f} {d['precision']:>8.4f} "
                f"{d['recall']:>8.4f} {d['ref_count']:>5} {d['hyp_count']:>5} "
                f"{d['overlap_count']:>5}"
                f" {d.get('ref_words', 0):>7} {d.get('hyp_words', 0):>7}",
                file=file,
            )
        print("-" * 80, file=file)

    mean_rw = aggregate.get("mean_ref_words", 0)
    mean_hw = aggregate.get("mean_hyp_words", 0)
    std_rw = aggregate.get("std_ref_words", 0)
    std_hw = aggregate.get("std_hyp_words", 0)

    print(
        f"{'MEAN':<25} {aggregate['mean_f1']:>8.4f} "
        f"{aggregate['mean_precision']:>8.4f} "
        f"{aggregate['mean_recall']:>8.4f}"
        f" {'':>5} {'':>5} {'':>5}"
        f" {mean_rw:>7.1f} {mean_hw:>7.1f}",
        file=file,
    )
    print(
        f"{'STD':<25} {aggregate['std_f1']:>8.4f} "
        f"{aggregate['std_precision']:>8.4f} "
        f"{aggregate['std_recall']:>8.4f}"
        f" {'':>5} {'':>5} {'':>5}"
        f" {std_rw:>7.1f} {std_hw:>7.1f}",
        file=file,
    )
    print(f"\nEvaluated {aggregate['num_dialogs']} dialogs", file=file)
    print("=" * 80, file=file)


def print_rouge_table(
    per_dialog: list[dict],
    aggregate: dict,
    rouge_types: tuple[str, ...] = ("rouge1", "rouge2", "rouge3", "rouge4", "rougeL"),
    file=None,
    summary_only: bool = False,
) -> None:
    """Print a formatted ROUGE results table (F-scores only for compactness)."""
    if file is None:
        file = sys.stdout

    print("\n" + "=" * 100, file=file)
    print("ROUGE EVALUATION RESULTS (F-measure)", file=file)
    print("=" * 100, file=file)

    # Header
    header = f"\n{'':25}"
    for rt in rouge_types:
        label = rt.replace("rouge", "R-")
        header += f" {label:>10}"
    header += f" {'RefWds':>8} {'HypWds':>8}"
    print(header, file=file)
    print("-" * 100, file=file)

    if not summary_only:
        # Per-dialog rows
        for d in per_dialog:
            row = f"{d['id']:<25}"
            for rt in rouge_types:
                row += f" {d.get(f'{rt}_f', 0):>10.4f}"
            row += f" {d.get('ref_words', 0):>8} {d.get('hyp_words', 0):>8}"
            print(row, file=file)
        print("-" * 100, file=file)

    # Aggregate
    mean_rw = aggregate.get("mean_ref_words", 0)
    mean_hw = aggregate.get("mean_hyp_words", 0)
    std_rw = aggregate.get("std_ref_words", 0)
    std_hw = aggregate.get("std_hyp_words", 0)

    mean_row = f"{'MEAN':<25}"
    std_row = f"{'STD':<25}"
    for rt in rouge_types:
        mean_row += f" {aggregate.get(f'{rt}_f_mean', 0):>10.4f}"
        std_row += f" {aggregate.get(f'{rt}_f_std', 0):>10.4f}"
    mean_row += f" {mean_rw:>8.1f} {mean_hw:>8.1f}"
    std_row += f" {std_rw:>8.1f} {std_hw:>8.1f}"
    print(mean_row, file=file)
    print(std_row, file=file)

    # Also show precision/recall summary
    print("\nDetailed (Precision / Recall):", file=file)
    for rt in rouge_types:
        label = rt.replace("rouge", "R-")
        p_mean = aggregate.get(f"{rt}_p_mean", 0)
        r_mean = aggregate.get(f"{rt}_r_mean", 0)
        print(f"  {label:<8} P={p_mean:.4f}  R={r_mean:.4f}", file=file)

    print(f"\nEvaluated {aggregate.get('num_dialogs', 0)} dialogs", file=file)
    print("=" * 100, file=file)
