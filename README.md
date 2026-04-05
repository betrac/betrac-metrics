# BeTraC 2026 — Evaluation Metrics

Official evaluation tooling for the [BeTraC 2026](https://betrac.github.io) challenge (IEEE SLT).
Computes all competition metrics for scoring SOAP note submissions against reference notes.

**No UMLS license required.** Similar tools (e.g., QuickUMLS) require a UMLS Metathesaurus
license. This tooling uses open MeSH vocabulary and optional scispaCy NER instead, so
participants can install and run it without any license agreements.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
  # or: brew install uv
  ```

## Quick Start (Participants)

```bash
# Clone and install with all evaluation dependencies
git clone https://github.com/betrac/betrac-metrics.git
cd betrac-metrics
uv venv
uv pip install ".[all]"

# Download the scispaCy biomedical NER model (needed for full concept matching)
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

# Or use the Makefile shortcut: make setup

# Evaluate your predictions against validation references (auto-downloaded from HuggingFace)
uv run btc-eval evaluate \
    --predictions my_predictions.jsonl \
    --split validation \
    --team my-team-name \
    --bootstrap-ci
```

### Example Output

```
BeTraC 2026 Evaluation Results
==================================================
Team: my-team-name
Split: validation
Dialogs: 400
Matcher: open-medical

  Concept F1:  0.6904 [0.6512, 0.7296]  (P=0.7211, R=0.6623)
  ROUGE-2 F1:  0.1872 [0.1654, 0.2090]  (P=0.2542, R=0.1493)
  ROUGE-3 F1:  0.0482 [0.0371, 0.0593]  (P=0.0627, R=0.0390)
==================================================
```

The `[lo, hi]` brackets are 95% bootstrap confidence intervals (shown with `--bootstrap-ci`).
The `Split:` line appears when references are loaded from HuggingFace.

## Competition Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **Open Medical Concept F1** | Primary (ranking) | MeSH keyword + scispaCy NER concept matching |
| **ROUGE-2 F** | Secondary | Bigram overlap F-measure |
| **ROUGE-3 F** | Secondary | Trigram overlap F-measure |

All metrics report Precision and Recall alongside F-measure.

## Installation

The package is installed from source (not yet on PyPI):

```bash
git clone https://github.com/betrac/betrac-metrics.git
cd betrac-metrics
uv venv

# Everything (recommended)
uv pip install ".[all]"
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

Individual extras if you want a lighter install:

```bash
uv pip install ".[rouge]"              # ROUGE scoring only
uv pip install ".[hf]"                 # HuggingFace auto-loading for references
uv pip install ".[scispacy]"           # scispaCy NER for broader concept coverage
uv pip install ".[rouge,hf]"           # ROUGE + HuggingFace (no scispaCy)
```

The scispaCy model download is always a separate step — it's not on PyPI:

```bash
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

Without the scispaCy model, `btc-eval evaluate` falls back to MeSH-only matching
(~100 curated terms). The output will show `Matcher: mesh-only` so you know.

## Input Format

### Prediction JSONL

Each line must be a JSON object with at minimum `id` and `summary` fields.
Other fields (e.g., timing, model info) are ignored. Invalid JSON lines are
skipped with a warning — check stderr if your dialog count looks low.

```json
{"id": "16504082-504f-41a8-8838-f826d932a31b", "summary": "S: Patient reports chest pain...\nO: BP 140/90...\nA: ACS suspected...\nP: ECG, troponin..."}
{"id": "a3b2c1d0-1234-5678-9abc-def012345678", "summary": "S: Fever and cough for 3 days...\nO: Temp 101.5F...\nA: URI...\nP: Rest, fluids..."}
```

This format is directly compatible with the output of [betrac-2026-baseline](https://github.com/betrac/betrac-2026-baseline) and [betrac-2026-cascade](https://github.com/betrac/betrac-2026-cascade).

### Reference Format

References can be provided as:
- **JSONL** (same format as predictions): `--references refs.jsonl`
- **Directory of .txt files**: `--references /path/to/dir/ --reference-format dir`
  (filename stem = dialog ID, file content = SOAP note; non-.txt files are ignored)
- **Auto-loaded from HuggingFace**: omit `--references` and specify `--split`

### ID Matching

Only dialog IDs present in **both** predictions and references are evaluated.
Mismatched IDs produce warnings but don't block evaluation.

## How Scoring Works

### Concept F1 (Primary Metric)

1. **Extract** medical concept CUIs from both reference and prediction texts
2. **Compare** concept sets using set overlap:
   - Precision = |overlap| / |predicted concepts|
   - Recall = |overlap| / |reference concepts|
   - F1 = harmonic mean of Precision and Recall
3. **Aggregate** per-dialog scores via macro-averaging

**Matchers:**
- **MeSH-only** (`--no-scispacy`): ~100 curated clinical terms, no extra dependencies
- **Open Medical** (default): MeSH + scispaCy biomedical NER for broader coverage

### ROUGE (Secondary Metrics)

Standard ROUGE n-gram overlap against reference SOAP notes.
The competition uses ROUGE-2 and ROUGE-3 F-measure.

## CLI Reference

All commands accept a global `--verbose` flag for detailed output (place it before the subcommand: `uv run btc-eval --verbose <command> ...`).

### `btc-eval evaluate` — Unified competition scoring (recommended)

```bash
uv run btc-eval evaluate \
    --predictions FILE \
    [--references FILE] \
    [--reference-format jsonl|dir] \
    [--split validation|train|test] \
    [--team TEAM_NAME] \
    [--output DIR] \
    [--no-scispacy] \
    [--use-stemmer] \
    [--allow-missing] \
    [--bootstrap-ci] [--analytical-ci] [--bootstrap-n N]
```

Runs Open Medical Concept F1 + ROUGE-2/3 in a single pass.
When `--references` is omitted, auto-loads from HuggingFace (requires `btc-eval[hf]`).
References are cached locally after the first download (~2s vs ~45s on subsequent runs).
Use `--no-cache` to force a fresh download.

By default, predictions must cover all reference IDs — missing predictions are
scored as zero. Use `--allow-missing` to skip missing IDs instead (useful during development).

Use `--bootstrap-ci` and/or `--analytical-ci` to add 95% confidence intervals
to all three metrics. Bootstrap CI is shown as `[lo, hi]`, analytical as `±half-width`.

**Output files** (when `--output` is specified):
- `summary.json` — compact results with team, split, all metrics, evaluation metadata
- `concept_per_dialog.jsonl` — per-dialog concept F1/P/R
- `rouge_per_dialog.jsonl` — per-dialog ROUGE-2/3 scores

### `btc-eval concept-f1` — Concept evaluation only

```bash
uv run btc-eval concept-f1 \
    --predictions FILE --references FILE \
    [--matcher open-medical|mesh-only] \
    [--reference-format jsonl|dir] \
    [--output DIR] [--summary-only]
```

### `btc-eval rouge` — ROUGE evaluation only

```bash
uv run btc-eval rouge \
    --predictions FILE --references FILE \
    [--reference-format jsonl|dir] \
    [--use-stemmer] \
    [--output DIR] [--summary-only]
```

Computes ROUGE-1/2/3/4/L (all five types).

### `btc-eval compare` — Batch comparison (organizers)

```bash
uv run btc-eval compare \
    --references FILE --predictions-dir DIR \
    [--manifest FILE] [--output-dir DIR] \
    [--no-rouge] [--no-concepts] [--open-medical]
```

Evaluates multiple prediction files and produces comparison tables (CSV, LaTeX, markdown).

### `btc-eval build-references` — Convert .txt to JSONL

```bash
uv run btc-eval build-references \
    --input-dir DIR --output FILE \
    [--filter-from-jsonl FILE]
```

### `btc-eval download-references` — Download references from HuggingFace

```bash
# Cache locally (default: ~/.cache/btc-eval/)
uv run btc-eval download-references --split validation

# Or save to a specific file
uv run btc-eval download-references --split validation --output refs.jsonl
```

Downloads reference SOAP notes from the HuggingFace dataset without the audio data.
The `evaluate` command does this automatically, but this is useful to pre-download
references or save them to a specific location.

## Python API

```python
from btc_eval.io import load_summaries_jsonl
from btc_eval.matchers.open_medical import OpenMedicalMatcher
from btc_eval.metrics.concept_f1 import extract_concept_ids, compute_concept_metrics
from btc_eval.metrics.rouge import compute_rouge, _get_scorer
from btc_eval.types import COMPETITION_ROUGE_TYPES

# Load data
predictions = load_summaries_jsonl("predictions.jsonl")
references = load_summaries_jsonl("references.jsonl")

# Concept F1
matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=False)
for did in sorted(set(predictions) & set(references)):
    ref = extract_concept_ids(references[did], matcher)
    hyp = extract_concept_ids(predictions[did], matcher)
    m = compute_concept_metrics(ref, hyp)
    print(f"{did}: F1={m.f1:.3f}")

# ROUGE-2/3
scorer = _get_scorer(use_stemmer=True, rouge_types=COMPETITION_ROUGE_TYPES)
for did in sorted(set(predictions) & set(references)):
    result = compute_rouge(
        references[did], predictions[did],
        scorer=scorer, rouge_types=COMPETITION_ROUGE_TYPES,
    )
    print(f"{did}: ROUGE-2 F={result.scores['rouge2'].fmeasure:.3f}")
```

See [examples/pipeline_scoring.py](examples/pipeline_scoring.py) for a standalone script
with a `score_predictions()` function you can copy into your project.

## Troubleshooting

### "No matching IDs between predictions and references"

Your prediction IDs don't match the reference IDs. Ensure your predictions use the
same IDs as the references. If loading from HuggingFace, verify you're using the
correct split. Use `uv run btc-eval --verbose evaluate ...` to see per-dialog details.

### Scores lower than expected

By default, any reference dialog without a matching prediction is scored as zero
(F1=0, ROUGE=0) and included in the aggregate. If you're only evaluating a subset,
use `--allow-missing` to skip those dialogs.

### Low or zero scores

- Check for empty summaries: the tool warns about empty/whitespace-only predictions on stderr, but still evaluates them (resulting in zero scores)
- Verify you're using the right matcher: `--no-scispacy` uses only ~100 MeSH terms
- The output shows `Matcher: mesh-only` or `Matcher: open-medical` so you know which was used

### Invalid JSON in prediction files

Invalid JSON lines are skipped with a warning on stderr. Evaluation proceeds with
the valid lines. If your dialog count is unexpectedly low, check stderr for
"is not valid JSON" messages.

### Import errors

| Error | Fix |
|-------|-----|
| `rouge-score is required` | `uv pip install ".[rouge]"` |
| `datasets library is required` | `uv pip install ".[hf]"` |
| `spaCy not installed` | `uv pip install ".[scispacy]"` |
| `scispaCy model not found` | `uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz` |

### `ConfigValidationError` when loading scispaCy

If you see `ConfigValidationError: 'True' is not <class 'bool'>`, this is a known
incompatibility between `en_core_sci_md` v0.5.4 (trained with spaCy 3.7) and spaCy 3.8+.
The evaluation tool auto-patches the model config on first run. If that fails, reinstall
the model:

```bash
uv pip install --force-reinstall https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

### Stale cached references

When using `--split` without `--references`, references are cached locally in
`~/.cache/btc-eval/` after the first download. If the HuggingFace dataset is
updated (e.g., reference notes are corrected), your cached copy will be outdated.

To fix: re-download with `--no-cache` or delete the cache file:

```bash
# Force fresh download
uv run btc-eval evaluate --predictions p.jsonl --split validation --no-cache

# Or delete the cache manually
rm ~/.cache/btc-eval/references-BeTraC_betrac-2026-validation.jsonl
```

### `FutureWarning` from spaCy

If you see a warning like `FutureWarning: Possible set union at position 6328` from
`spacy/language.py`, this is a known upstream issue in spaCy's regex handling.
It does not affect evaluation results and can be safely ignored.

### scispaCy silently falls back to MeSH-only

If the scispaCy model isn't installed, the tool continues with MeSH-only matching
and shows `Matcher: mesh-only` in the output. Install the model for full coverage.

## Development

```bash
# Create venv and install dev dependencies
uv venv
uv pip install -e ".[dev,rouge,hf]"

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

### Project Structure

```
src/btc_eval/
├── __init__.py              # Package exports
├── cli.py                   # CLI (evaluate, concept-f1, rouge, compare, build-references)
├── compare.py               # Batch comparison (organizer use)
├── io.py                    # I/O: JSONL, directory, HuggingFace loading
├── types.py                 # Data types (ConceptMetrics, RougeResult, etc.)
├── matchers/
│   ├── __init__.py          # Matcher protocol
│   ├── mock.py              # Mock matcher for testing
│   └── open_medical.py      # MeSH + scispaCy matcher
└── metrics/
    ├── concept_f1.py        # Concept F1/Precision/Recall
    ├── rouge.py             # ROUGE evaluation
    └── utils.py             # Bootstrap and analytical confidence intervals
```

## Related Repositories

| Repository | Description |
|------------|-------------|
| [betrac-2026-baseline](https://github.com/betrac/betrac-2026-baseline) | 3B end-to-end baseline (Qwen2.5-Omni) |
| [betrac-2026-cascade](https://github.com/betrac/betrac-2026-cascade) | Cascade reference topline (Whisper + Qwen3, not eligible) |
| [BeTraC/betrac-2026](https://huggingface.co/datasets/BeTraC/betrac-2026) | Dataset on HuggingFace |

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

The [Synth-DoPaCo](https://huggingface.co/datasets/BeTraC/betrac-2026) dataset
used in BeTraC 2026 was created by the Play-Your-Part team during the
[JSALT 2025](https://www.clsp.jhu.edu/workshops/) workshop at Johns Hopkins
University hosted at Brno University.
