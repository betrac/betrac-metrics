# Contributing to BeTraC Metrics

Thank you for your interest in contributing to the BeTraC 2026 evaluation toolkit.

## Development Setup

```bash
git clone https://github.com/betrac/betrac-metrics.git
cd betrac-metrics

# Install all dependencies (including dev, ROUGE, scispaCy)
make setup

# Or manually
uv pip install -e ".[dev,rouge,hf]"
```

## Running Tests

```bash
# Full test suite
pytest

# Specific test file
pytest tests/test_evaluate.py

# With verbose output
pytest -v
```

All tests should pass before submitting changes. The test suite currently has
75+ tests covering metrics, matchers, I/O, and CLI integration.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
(line length: 100 characters).

```bash
# Check for issues
ruff check src/ tests/

# Auto-format
ruff format src/ tests/
```

## Making Changes

1. Create a branch for your changes
2. Write tests first (test-driven development preferred)
3. Implement the change
4. Verify: `pytest && ruff check src/ tests/`
5. Submit a pull request

## Project Structure

```
src/btc_eval/
├── cli.py           # CLI commands (evaluate, concept-f1, rouge, compare, build-references)
├── io.py            # I/O: JSONL, directory, HuggingFace loading
├── types.py         # Data types and constants
├── compare.py       # Batch comparison (organizer use)
├── matchers/        # Concept extraction (MeSH + scispaCy)
└── metrics/         # Metric computation (concept F1, ROUGE, CIs)
```

## Metric Definitions

If modifying how metrics are computed, please ensure:

- **Concept F1**: Set-overlap based, uses epsilon (1e-6) for empty sets
- **ROUGE**: Uses the `rouge-score` library; competition types are ROUGE-2 and ROUGE-3
- **Aggregation**: Macro-averaged across dialogs (each dialog weighted equally)

## Reporting Issues

File issues at [github.com/betrac/betrac-metrics/issues](https://github.com/betrac/betrac-metrics/issues)
or contact [betrac@googlegroups.com](mailto:betrac@googlegroups.com).

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0, the same license as the project.
