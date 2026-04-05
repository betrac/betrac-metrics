"""Integration tests for the CLI."""

import json
import subprocess
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


class TestEvaluateCLI:
    def test_mesh_only_evaluation(self, tmp_path):
        """End-to-end test: evaluate with mesh-only matcher."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "btc_eval.cli",
                "concept-f1",
                "--predictions",
                str(DATA_DIR / "sample_predictions.jsonl"),
                "--references",
                str(DATA_DIR / "sample_references.jsonl"),
                "--matcher",
                "mesh-only",
                "--output",
                str(tmp_path / "results"),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "MEDICAL CONCEPT EVALUATION RESULTS" in result.stdout
        assert "Evaluated 5 dialogs" in result.stdout

        # Check output files
        agg = json.loads((tmp_path / "results" / "aggregate_metrics.json").read_text())
        assert "mean_f1" in agg
        assert agg["num_dialogs"] == 5
        assert agg["mean_f1"] > 0  # non-zero metrics

        # Per-dialog JSONL
        per_dialog = tmp_path / "results" / "per_dialog_metrics.jsonl"
        lines = per_dialog.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_dir_reference_format(self, tmp_path):
        """Test with references as a directory of .txt files."""
        # Create reference dir from JSONL
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        with open(DATA_DIR / "sample_references.jsonl") as f:
            for line in f:
                record = json.loads(line)
                (ref_dir / f"{record['id']}.txt").write_text(record["summary"])

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "btc_eval.cli",
                "concept-f1",
                "--predictions",
                str(DATA_DIR / "sample_predictions.jsonl"),
                "--references",
                str(ref_dir),
                "--reference-format",
                "dir",
                "--matcher",
                "mesh-only",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Evaluated 5 dialogs" in result.stdout


class TestBuildReferencesCLI:
    def test_basic(self, tmp_path):
        # Create source .txt files
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "d001.txt").write_text("Summary A")
        (src_dir / "d002.txt").write_text("Summary B")

        out = tmp_path / "refs.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "btc_eval.cli",
                "build-references",
                "--input-dir",
                str(src_dir),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Wrote 2 references" in result.stdout

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
