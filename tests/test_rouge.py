"""Tests for ROUGE evaluation metrics."""

import subprocess
import sys
from pathlib import Path

import pytest

from btc_eval.types import ROUGE_TYPES

# Skip all tests if rouge-score not installed
pytest.importorskip("rouge_score")

from btc_eval.metrics.rouge import aggregate_rouge, compute_rouge, _get_scorer


DATA_DIR = Path(__file__).parent / "data"


class TestComputeRouge:
    @pytest.fixture(autouse=True)
    def setup_scorer(self):
        self.scorer = _get_scorer(use_stemmer=False)

    def test_identical_texts(self):
        text = "The patient has chest pain and shortness of breath."
        result = compute_rouge(text, text, scorer=self.scorer)
        for rt in ROUGE_TYPES:
            assert result.scores[rt].fmeasure == pytest.approx(1.0)
            assert result.scores[rt].precision == pytest.approx(1.0)
            assert result.scores[rt].recall == pytest.approx(1.0)

    def test_completely_different(self):
        ref = "Alpha beta gamma delta epsilon"
        hyp = "One two three four five"
        result = compute_rouge(ref, hyp, scorer=self.scorer)
        for rt in ROUGE_TYPES:
            assert result.scores[rt].fmeasure == pytest.approx(0.0)

    def test_partial_overlap(self):
        ref = "The patient has chest pain and shortness of breath"
        hyp = "The patient reports chest pain with dyspnea"
        result = compute_rouge(ref, hyp, scorer=self.scorer)
        # ROUGE-1 should have some overlap (shared words: the, patient, chest, pain)
        assert result.scores["rouge1"].fmeasure > 0.3
        assert result.scores["rouge1"].fmeasure < 1.0
        # ROUGE-L should also have overlap
        assert result.scores["rougeL"].fmeasure > 0.2

    def test_rouge_types_present(self):
        result = compute_rouge("hello world", "hello world", scorer=self.scorer)
        assert set(result.scores.keys()) == set(ROUGE_TYPES)

    def test_to_dict(self):
        result = compute_rouge("chest pain", "chest pain", scorer=self.scorer)
        d = result.to_dict()
        assert "rouge1_f" in d
        assert "rouge2_f" in d
        assert "rouge3_f" in d
        assert "rouge4_f" in d
        assert "rougeL_f" in d
        assert "rouge1_p" in d
        assert "rouge1_r" in d

    def test_f_scores_dict(self):
        result = compute_rouge("test text here", "test text here", scorer=self.scorer)
        f_dict = result.f_scores_dict()
        assert len(f_dict) == len(ROUGE_TYPES)
        for rt in ROUGE_TYPES:
            assert rt in f_dict

    def test_empty_hypothesis(self):
        result = compute_rouge("some reference text", "", scorer=self.scorer)
        for rt in ROUGE_TYPES:
            assert result.scores[rt].fmeasure == 0.0

    def test_use_stemmer(self):
        scorer_stemmed = _get_scorer(use_stemmer=True)
        ref = "The patients were running quickly"
        hyp = "The patient runs quick"
        result_no_stem = compute_rouge(ref, hyp, scorer=self.scorer)
        result_stemmed = compute_rouge(ref, hyp, scorer=scorer_stemmed)
        # Stemming should generally increase overlap
        assert result_stemmed.scores["rouge1"].fmeasure >= result_no_stem.scores["rouge1"].fmeasure


class TestAggregateRouge:
    def test_single_result(self):
        scorer = _get_scorer()
        result = compute_rouge("chest pain noted", "chest pain noted", scorer=scorer)
        agg = aggregate_rouge([result])
        assert agg["num_dialogs"] == 1
        assert agg["rouge1_f_mean"] == pytest.approx(1.0)
        assert agg["rouge1_f_std"] == pytest.approx(0.0)

    def test_multiple_results(self):
        scorer = _get_scorer()
        r1 = compute_rouge("chest pain", "chest pain", scorer=scorer)
        r2 = compute_rouge("alpha beta", "gamma delta", scorer=scorer)
        agg = aggregate_rouge([r1, r2])
        assert agg["num_dialogs"] == 2
        assert agg["rouge1_f_mean"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert agg["rouge1_f_std"] > 0

    def test_empty(self):
        agg = aggregate_rouge([])
        assert agg["num_dialogs"] == 0

    def test_all_keys_present(self):
        scorer = _get_scorer()
        result = compute_rouge("test", "test", scorer=scorer)
        agg = aggregate_rouge([result])
        for rt in ROUGE_TYPES:
            for suffix in ("f", "p", "r"):
                assert f"{rt}_{suffix}_mean" in agg
                assert f"{rt}_{suffix}_std" in agg


class TestConfigurableRougeTypes:
    def test_compute_rouge_subset(self):
        from btc_eval.types import COMPETITION_ROUGE_TYPES

        scorer = _get_scorer(use_stemmer=False, rouge_types=COMPETITION_ROUGE_TYPES)
        result = compute_rouge(
            "The patient has chest pain and shortness of breath.",
            "The patient has chest pain and shortness of breath.",
            scorer=scorer,
            rouge_types=COMPETITION_ROUGE_TYPES,
        )
        assert set(result.scores.keys()) == set(COMPETITION_ROUGE_TYPES)
        for rt in COMPETITION_ROUGE_TYPES:
            assert result.scores[rt].fmeasure == pytest.approx(1.0)

    def test_aggregate_rouge_subset(self):
        from btc_eval.types import COMPETITION_ROUGE_TYPES

        scorer = _get_scorer(use_stemmer=False, rouge_types=COMPETITION_ROUGE_TYPES)
        r1 = compute_rouge(
            "chest pain", "chest pain", scorer=scorer, rouge_types=COMPETITION_ROUGE_TYPES
        )
        r2 = compute_rouge(
            "alpha beta", "gamma delta", scorer=scorer, rouge_types=COMPETITION_ROUGE_TYPES
        )
        agg = aggregate_rouge([r1, r2], rouge_types=COMPETITION_ROUGE_TYPES)
        assert agg["num_dialogs"] == 2
        for rt in COMPETITION_ROUGE_TYPES:
            assert f"{rt}_f_mean" in agg
        # Should NOT have rouge1 or rouge4
        assert "rouge1_f_mean" not in agg
        assert "rouge4_f_mean" not in agg
        assert "rougeL_f_mean" not in agg

    def test_competition_rouge_types_constant(self):
        from btc_eval.types import COMPETITION_ROUGE_TYPES

        assert COMPETITION_ROUGE_TYPES == ("rouge2", "rouge3")


class TestRougeCLI:
    def test_basic_evaluation(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "btc_eval.cli",
                "rouge",
                "--predictions",
                str(DATA_DIR / "sample_predictions.jsonl"),
                "--references",
                str(DATA_DIR / "sample_references.jsonl"),
                "--output",
                str(tmp_path / "results"),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "ROUGE EVALUATION RESULTS" in result.stdout
        assert "Evaluated 5 dialogs" in result.stdout

        # Check output files
        import json

        agg = json.loads((tmp_path / "results" / "rouge_aggregate.json").read_text())
        assert agg["num_dialogs"] == 5
        assert agg["rouge1_f_mean"] > 0

        per_dialog = tmp_path / "results" / "rouge_per_dialog.jsonl"
        lines = per_dialog.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_with_stemmer(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "btc_eval.cli",
                "rouge",
                "--predictions",
                str(DATA_DIR / "sample_predictions.jsonl"),
                "--references",
                str(DATA_DIR / "sample_references.jsonl"),
                "--use-stemmer",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "stemmer=on" in result.stdout
