"""Tests for the unified evaluate command."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ROUGE is required for the evaluate command
pytest.importorskip("rouge_score")

DATA_DIR = Path(__file__).parent / "data"


class TestEvaluateCommand:
    def _run_evaluate(self, *extra_args, tmp_path=None):
        cmd = [
            sys.executable,
            "-m",
            "btc_eval.cli",
            "evaluate",
            "--predictions",
            str(DATA_DIR / "sample_predictions.jsonl"),
            "--references",
            str(DATA_DIR / "sample_references.jsonl"),
            "--no-scispacy",
        ]
        cmd.extend(extra_args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

    def test_compact_output(self, tmp_path):
        result = self._run_evaluate()
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "BeTraC 2026 Evaluation Results" in result.stdout
        assert "Concept F1:" in result.stdout
        assert "ROUGE-2" in result.stdout
        assert "ROUGE-3" in result.stdout

    def test_no_rouge1_rouge4_rougeL(self, tmp_path):
        result = self._run_evaluate()
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Compact output should NOT contain non-competition ROUGE types
        assert "ROUGE-1" not in result.stdout
        assert "ROUGE-4" not in result.stdout
        assert "ROUGE-L" not in result.stdout

    def test_team_in_output(self, tmp_path):
        result = self._run_evaluate("--team", "awesome-lab")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Team: awesome-lab" in result.stdout

    def test_split_not_shown_with_local_references(self, tmp_path):
        result = self._run_evaluate("--split", "validation")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Split should NOT appear when --references is provided (local file)
        assert "Split:" not in result.stdout

    def test_output_files(self, tmp_path):
        result = self._run_evaluate(
            "--team",
            "test-team",
            "--split",
            "validation",
            "--output",
            str(tmp_path / "results"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Check summary.json
        summary_path = tmp_path / "results" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["team"] == "test-team"
        assert summary["split"] == ""  # local references, no split
        assert summary["num_dialogs"] == 5
        assert "concept_f1" in summary
        assert "concept_precision" in summary
        assert "concept_recall" in summary
        assert "rouge2_f" in summary
        assert "rouge2_p" in summary
        assert "rouge2_r" in summary
        assert "rouge3_f" in summary
        assert "rouge3_p" in summary
        assert "rouge3_r" in summary
        # Metadata fields
        assert summary["matcher"] == "mesh-only"
        assert summary["use_scispacy"] is False
        assert "use_stemmer" in summary

        # Check per-dialog files
        concept_per = tmp_path / "results" / "concept_per_dialog.jsonl"
        assert concept_per.exists()
        lines = concept_per.read_text().strip().split("\n")
        assert len(lines) == 5

        rouge_per = tmp_path / "results" / "rouge_per_dialog.jsonl"
        assert rouge_per.exists()
        lines = rouge_per.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_matcher_label_in_output(self, tmp_path):
        result = self._run_evaluate()
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Matcher: mesh-only" in result.stdout

    def test_dialogs_count(self, tmp_path):
        result = self._run_evaluate()
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Dialogs: 5" in result.stdout

    def test_bootstrap_ci(self, tmp_path):
        result = self._run_evaluate("--bootstrap-ci")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Bootstrap CI should show brackets [lo, hi]
        assert "[" in result.stdout
        assert "]" in result.stdout

    def test_analytical_ci(self, tmp_path):
        result = self._run_evaluate("--analytical-ci")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Analytical CI should show ±
        assert "±" in result.stdout

    def test_ci_in_output_json(self, tmp_path):
        result = self._run_evaluate(
            "--bootstrap-ci",
            "--analytical-ci",
            "--output",
            str(tmp_path / "results"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        summary = json.loads((tmp_path / "results" / "summary.json").read_text())
        # Concept F1 CIs
        assert "concept_f1_bci_lo" in summary
        assert "concept_f1_bci_hi" in summary
        assert "concept_f1_aci_half" in summary
        # ROUGE CIs
        assert "rouge2_f_bci_lo" in summary
        assert "rouge3_f_bci_lo" in summary


class TestStrictScoring:
    """Tests for missing prediction handling (strict mode vs --allow-missing)."""

    @pytest.fixture
    def partial_predictions(self, tmp_path):
        """Create predictions JSONL with only 3 of the 5 sample IDs."""
        src = DATA_DIR / "sample_predictions.jsonl"
        lines = src.read_text().strip().split("\n")
        # Keep only first 3 lines
        out = tmp_path / "partial.jsonl"
        out.write_text("\n".join(lines[:3]) + "\n")
        return out

    def _run(self, predictions, *extra_args):
        cmd = [
            sys.executable, "-m", "btc_eval.cli",
            "evaluate",
            "--predictions", str(predictions),
            "--references", str(DATA_DIR / "sample_references.jsonl"),
            "--no-scispacy",
        ]
        cmd.extend(extra_args)
        return subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )

    def test_strict_scores_missing_as_zero(self, partial_predictions, tmp_path):
        """Default: missing predictions scored as zero, all 5 dialogs evaluated."""
        result = self._run(
            partial_predictions,
            "--output", str(tmp_path / "results"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "scored as zero" in result.stderr
        assert "Dialogs: 5" in result.stdout

        summary = json.loads((tmp_path / "results" / "summary.json").read_text())
        assert summary["num_dialogs"] == 5
        # With 2 zero-scored dialogs, F1 must be lower than with 3 dialogs only
        assert summary["concept_f1"] < 0.9

    def test_allow_missing_skips(self, partial_predictions, tmp_path):
        """With --allow-missing: only 3 shared dialogs evaluated."""
        result = self._run(
            partial_predictions,
            "--allow-missing",
            "--output", str(tmp_path / "results"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "skipped" in result.stderr
        assert "Dialogs: 3" in result.stdout

        summary = json.loads((tmp_path / "results" / "summary.json").read_text())
        assert summary["num_dialogs"] == 3

    def test_duplicate_id_warning(self, tmp_path):
        """Duplicate IDs in predictions emit a warning."""
        dupes = tmp_path / "dupes.jsonl"
        src = DATA_DIR / "sample_predictions.jsonl"
        lines = src.read_text().strip().split("\n")
        # Append the first line again (duplicate conv_001)
        dupes.write_text("\n".join(lines) + "\n" + lines[0] + "\n")
        result = self._run(dupes, "--allow-missing")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Duplicate" in result.stderr
