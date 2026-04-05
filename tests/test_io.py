"""Tests for I/O utilities."""

import io
from unittest.mock import patch, MagicMock

from btc_eval.io import (
    build_reference_jsonl,
    load_references_from_dir,
    load_references_from_hf,
    load_summaries_jsonl,
    print_compact_summary,
)


class TestLoadSummariesJsonl:
    def test_basic(self, sample_predictions_path):
        summaries = load_summaries_jsonl(sample_predictions_path)
        assert len(summaries) == 5
        assert "conv_001" in summaries
        assert "chest pain" in summaries["conv_001"].lower()

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        assert load_summaries_jsonl(empty) == {}

    def test_skips_bad_lines(self, tmp_path):
        f = tmp_path / "partial.jsonl"
        f.write_text(
            '{"id": "a", "summary": "good"}\n'
            '{"id": "b"}\n'  # missing summary
            '{"id": "c", "summary": "also good"}\n'
        )
        result = load_summaries_jsonl(f)
        assert len(result) == 2
        assert "b" not in result

    def test_warns_on_duplicate_ids(self, tmp_path, caplog):
        f = tmp_path / "dupes.jsonl"
        f.write_text(
            '{"id": "a", "summary": "first"}\n'
            '{"id": "b", "summary": "good"}\n'
            '{"id": "a", "summary": "second"}\n'
        )
        import logging
        with caplog.at_level(logging.WARNING):
            result = load_summaries_jsonl(f)
        # Last occurrence wins
        assert result["a"] == "second"
        assert result["b"] == "good"
        assert len(result) == 2
        # Warning emitted
        assert any("Duplicate" in msg for msg in caplog.messages)

    def test_skips_malformed_json(self, tmp_path):
        f = tmp_path / "malformed.jsonl"
        f.write_text(
            '{"id": "a", "summary": "good"}\n'
            "this is not json\n"
            '{"id": "c", "summary": "also good"}\n'
        )
        result = load_summaries_jsonl(f)
        assert len(result) == 2
        assert "a" in result
        assert "c" in result


class TestLoadReferencesFromDir:
    def test_basic(self, tmp_path):
        (tmp_path / "dialog_001.txt").write_text("Summary one")
        (tmp_path / "dialog_002.txt").write_text("Summary two")
        result = load_references_from_dir(tmp_path)
        assert len(result) == 2
        assert result["dialog_001"] == "Summary one"

    def test_filter(self, tmp_path):
        (tmp_path / "dialog_001.txt").write_text("Summary one")
        (tmp_path / "dialog_002.txt").write_text("Summary two")
        result = load_references_from_dir(tmp_path, filter_ids={"dialog_001"})
        assert len(result) == 1


class TestPrintCompactSummary:
    def test_contains_all_metrics(self):
        concept_agg = {
            "mean_f1": 0.6904,
            "mean_precision": 0.7211,
            "mean_recall": 0.6623,
        }
        rouge_agg = {
            "rouge2_f_mean": 0.1872,
            "rouge2_p_mean": 0.2542,
            "rouge2_r_mean": 0.1493,
            "rouge3_f_mean": 0.0482,
            "rouge3_p_mean": 0.0627,
            "rouge3_r_mean": 0.0390,
        }
        buf = io.StringIO()
        print_compact_summary(
            concept_agg,
            rouge_agg,
            team="test-team",
            split="validation",
            num_dialogs=400,
            file=buf,
        )
        output = buf.getvalue()
        assert "BeTraC 2026" in output
        assert "Team: test-team" in output
        assert "Split: validation" in output
        assert "Dialogs: 400" in output
        assert "Concept F1:" in output
        assert "0.6904" in output
        assert "ROUGE-2" in output
        assert "0.1872" in output
        assert "ROUGE-3" in output
        assert "0.0482" in output

    def test_no_team(self):
        concept_agg = {"mean_f1": 0.5, "mean_precision": 0.5, "mean_recall": 0.5}
        rouge_agg = {
            "rouge2_f_mean": 0.1,
            "rouge2_p_mean": 0.1,
            "rouge2_r_mean": 0.1,
            "rouge3_f_mean": 0.1,
            "rouge3_p_mean": 0.1,
            "rouge3_r_mean": 0.1,
        }
        buf = io.StringIO()
        print_compact_summary(concept_agg, rouge_agg, num_dialogs=5, file=buf)
        output = buf.getvalue()
        assert "Team:" not in output
        assert "Concept F1:" in output

    def test_matcher_label(self):
        concept_agg = {"mean_f1": 0.5, "mean_precision": 0.5, "mean_recall": 0.5}
        rouge_agg = {
            "rouge2_f_mean": 0.1,
            "rouge2_p_mean": 0.1,
            "rouge2_r_mean": 0.1,
            "rouge3_f_mean": 0.1,
            "rouge3_p_mean": 0.1,
            "rouge3_r_mean": 0.1,
        }
        buf = io.StringIO()
        print_compact_summary(
            concept_agg,
            rouge_agg,
            num_dialogs=5,
            matcher_label="mesh-only",
            file=buf,
        )
        output = buf.getvalue()
        assert "Matcher: mesh-only" in output


class TestLoadReferencesFromHF:
    def _mock_datasets(self, mock_data):
        """Patch the datasets module so load_references_from_hf can import it."""
        mock_module = MagicMock()
        mock_module.load_dataset.return_value = mock_data
        return patch.dict("sys.modules", {"datasets": mock_module})

    def test_loads_from_hf(self):
        """Test loading references from a mocked HuggingFace dataset."""
        mock_data = [
            {
                "json": '{"id": "conv_001"}',
                "soap.txt": "S: Patient reports headache.\nO: BP 120/80.",
            },
            {
                "json": {"id": "conv_002"},
                "soap.txt": b"S: Chest pain.\nO: ECG normal.",
            },
        ]
        with self._mock_datasets(mock_data):
            refs = load_references_from_hf(split="validation", cache=False)

        assert len(refs) == 2
        assert "conv_001" in refs
        assert "conv_002" in refs
        assert "headache" in refs["conv_001"]
        assert "Chest pain" in refs["conv_002"]

    def test_handles_bytes_soap(self):
        """Test that bytes-encoded soap.txt is decoded."""
        mock_data = [
            {
                "json": {"id": "conv_003"},
                "soap.txt": b"S: Fever.\nO: Temp 101F.",
            },
        ]
        with self._mock_datasets(mock_data):
            refs = load_references_from_hf(split="validation", cache=False)

        assert refs["conv_003"] == "S: Fever.\nO: Temp 101F."

    def test_skips_missing_soap(self):
        """Test that items without soap.txt are skipped."""
        mock_data = [
            {"json": {"id": "conv_001"}, "soap.txt": "S: Good."},
            {"json": {"id": "conv_002"}},  # no soap.txt
        ]
        with self._mock_datasets(mock_data):
            refs = load_references_from_hf(split="validation", cache=False)

        assert len(refs) == 1
        assert "conv_001" in refs


class TestBuildReferenceJsonl:
    def test_roundtrip(self, tmp_path):
        # Create source txt files
        src = tmp_path / "src"
        src.mkdir()
        (src / "d001.txt").write_text("Summary A")
        (src / "d002.txt").write_text("Summary B")

        # Build JSONL
        out = tmp_path / "out.jsonl"
        count = build_reference_jsonl(src, out)
        assert count == 2

        # Read back
        summaries = load_summaries_jsonl(out)
        assert summaries["d001"] == "Summary A"
        assert summaries["d002"] == "Summary B"
