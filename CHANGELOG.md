# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] — 2026-04-04

### Added

- `btc-eval evaluate` — unified competition scoring command
  - Open Medical Concept F1 + ROUGE-2/3 in a single pass
  - `--team` and `--split` flags for labeling results
  - Auto-loads references from HuggingFace when `--references` is omitted
  - Bootstrap and analytical confidence intervals (`--bootstrap-ci`, `--analytical-ci`)
  - `--verbose` mode for per-dialog concept breakdowns
  - Compact summary output with matcher label
  - `summary.json` output with evaluation metadata
- `btc-eval concept-f1` — standalone concept F1 evaluation
- `btc-eval rouge` — standalone ROUGE-1/2/3/4/L evaluation
- `btc-eval compare` — batch comparison across prediction files (organizer use)
- `btc-eval build-references` — convert .txt directory to JSONL
- Open Medical Matcher: MeSH vocabulary (~100 curated terms) + optional scispaCy NER
- Configurable ROUGE types via `COMPETITION_ROUGE_TYPES`
- HuggingFace dataset auto-loading (streaming)
- Graceful error handling for malformed JSONL and empty summaries
- 79 tests across 8 test files

[0.1.0]: https://github.com/betrac/betrac-metrics/releases/tag/v0.1.0
