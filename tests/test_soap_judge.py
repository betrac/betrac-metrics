"""Tests for the SOAP LLM-judge package (offline — MockBackend, no network)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from btc_eval.soap_judge import pipeline
from btc_eval.soap_judge.backends import available_backends, get_backend
from btc_eval.soap_judge.backends.mock import MockBackend
from btc_eval.soap_judge.parse import (
    extract_json_object,
    judgment_to_scores,
    parse_judgment,
    safe_float,
    safe_int,
)
from btc_eval.soap_judge.prompts import load_prompt
from btc_eval.types import SOAP_CSV_COLUMNS


# ---------------------------------------------------------------------------
# Parser units
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_json_fence(self):
        text = 'Here is the result:\n```json\n{"a": 1, "b": 2}\n```\nDone.'
        assert extract_json_object(text) == {"a": 1, "b": 2}

    def test_bare_fence(self):
        assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_reasoning_block_stripped(self):
        text = '<reasoning>thinking hard {ignore: this}</reasoning>\n{"a": 5}'
        assert extract_json_object(text) == {"a": 5}

    def test_prose_around_object(self):
        text = 'Sure! {"x": 10} hope that helps'
        assert extract_json_object(text) == {"x": 10}

    def test_garbage_returns_none(self):
        assert extract_json_object("no json here at all") is None
        assert extract_json_object("") is None

    def test_not_a_dict_returns_none(self):
        assert extract_json_object("[1, 2, 3]") is None


class TestParseJudgment:
    def test_requires_subscores(self):
        # Object without subscores_1_to_5 is treated as a failure.
        assert parse_judgment('{"doc_type": "soap_judgment"}') is None

    def test_valid_judgment(self):
        obj = parse_judgment('{"subscores_1_to_5": {"faithfulness_grounding": 4}}')
        assert obj is not None
        assert obj["subscores_1_to_5"]["faithfulness_grounding"] == 4


class TestSafeCoerce:
    def test_safe_int(self):
        assert safe_int(None) == 0
        assert safe_int("3") == 3
        assert safe_int("bad", default=-1) == -1

    def test_safe_float(self):
        assert safe_float(None) == 0.0
        assert safe_float("0.5") == 0.5
        assert safe_float(None, default=1.0) == 1.0


class TestJudgmentToScores:
    def test_pipe_joined_error_types_are_tallied(self):
        judgment = {
            "subscores_1_to_5": {
                "faithfulness_grounding": 4,
                "structure_formatting": 5,
                "coverage_completeness": 3,
                "conciseness": 2,
            },
            "metrics": {
                "rates": {"unsupported_rate": 0.25, "contradiction_rate": 0.1},
                "claim_counts": {"not_in_transcript": 2},
                "coverage": {"critical_omissions_count": 1},
                "conciseness": {"redundancy_count": 3},
            },
            "claim_judgments": [
                {"claim_id": "C001", "error_types": ["over_medicalization|wrong_section"]},
                {"claim_id": "C002", "error_types": "over_specific"},
                {"claim_id": "C003", "error_types": ["over_medicalization"]},
            ],
        }
        scores = judgment_to_scores(judgment)
        assert scores.faithfulness == 4
        assert scores.conciseness == 2
        assert scores.over_medicalization == 2  # tallied across pipe + list forms
        assert scores.over_specific == 1
        assert scores.hallucination_rate == 0.25
        assert scores.missed_claims == 2
        assert scores.critical_omissions == 1
        assert scores.redundancy_count == 3

    def test_explicit_count_overrides_tally(self):
        judgment = {
            "subscores_1_to_5": {},
            "metrics": {"over_medicalization_count": 9},
            "claim_judgments": [{"error_types": ["over_medicalization"]}],
        }
        assert judgment_to_scores(judgment).over_medicalization == 9


# ---------------------------------------------------------------------------
# Backend registry + prompts
# ---------------------------------------------------------------------------


def test_registry_has_builtin_backends():
    for name in ("mock", "ollama", "openrouter", "bedrock", "anthropic"):
        assert name in available_backends()


def test_get_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("does-not-exist")


def test_mock_backend_returns_valid_json():
    backend = MockBackend()
    extract_prompt = load_prompt("extract_claims").format(soap_note="S: cough.")
    claims = json.loads(backend.complete([{"role": "user", "content": extract_prompt}], "m"))
    assert claims["doc_type"] == "soap_claims"

    judge_prompt = load_prompt("judge_soap_note").format(
        transcript="DOCTOR: hi", soap_note="S: cough", claims_json="{}"
    )
    judgment = json.loads(backend.complete([{"role": "user", "content": judge_prompt}], "m"))
    assert judgment["doc_type"] == "soap_judgment"
    assert "subscores_1_to_5" in judgment


def test_prompt_override(tmp_path):
    override = tmp_path / "extract_claims.txt"
    override.write_text("CUSTOM {soap_note}", encoding="utf-8")
    assert load_prompt("extract_claims", str(tmp_path)) == "CUSTOM {soap_note}"
    # Falls back to the packaged default when the dir has no override.
    assert "atomic" in load_prompt("extract_claims", str(tmp_path / "empty")).lower()


# ---------------------------------------------------------------------------
# End-to-end pipeline (MockBackend)
# ---------------------------------------------------------------------------


PREDICTIONS = {
    "d001": "S: Patient reports chest pain.\nA: Possible angina.\nP: ECG ordered.",
    "d002": "S: Cough for three days.\nO: Temp 38.1.\nA: URI.\nP: Rest and fluids.",
}
TRANSCRIPTS = {
    "d001": "DOCTOR: What brings you in?\nPATIENT: Chest pain since morning.",
    "d002": "DOCTOR: How long the cough?\nPATIENT: About three days, with fever.",
}


def test_pipeline_end_to_end():
    backend = MockBackend()
    claims = pipeline.run_extract(PREDICTIONS, backend, "mock", workers=2)
    assert set(claims) == set(PREDICTIONS)
    assert all(c["status"] == "parsed" for c in claims.values())

    judged = pipeline.run_judge(PREDICTIONS, TRANSCRIPTS, claims, backend, "mock", workers=2)
    assert set(judged) == set(PREDICTIONS)
    scores = [judgment_to_scores(j["judgment"]) for j in judged.values()]
    agg = pipeline.aggregate_soap(scores)
    assert agg["num_dialogs"] == 2
    assert "mean_faithfulness" in agg
    assert "std_faithfulness" in agg
    assert 0.0 <= agg["mean_faithfulness"] <= 5.0


def test_judge_skips_dialogs_without_transcript():
    backend = MockBackend()
    claims = pipeline.run_extract(PREDICTIONS, backend, "mock", workers=2)
    judged = pipeline.run_judge(
        PREDICTIONS, {"d001": TRANSCRIPTS["d001"]}, claims, backend, "mock", workers=2
    )
    assert set(judged) == {"d001"}


class _CountingBackend:
    """Wraps MockBackend and counts complete() calls (for resume tests)."""

    def __init__(self):
        self._mock = MockBackend()
        self.calls = 0

    def complete(self, messages, model, **kwargs):
        self.calls += 1
        return self._mock.complete(messages, model, **kwargs)


def test_resume_skips_cached_calls(tmp_path):
    backend = _CountingBackend()
    raw_dir = tmp_path / "extract_raw"
    pipeline.run_extract(PREDICTIONS, backend, "mock", workers=1, raw_dir=raw_dir)
    assert backend.calls == 2
    assert (raw_dir / "d001.json").exists()

    # Second run with resume=True reuses the cache — no new LLM calls.
    backend2 = _CountingBackend()
    pipeline.run_extract(PREDICTIONS, backend2, "mock", workers=1, raw_dir=raw_dir)
    assert backend2.calls == 0


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, mapping: dict, field: str):
    with open(path, "w", encoding="utf-8") as f:
        for did, text in mapping.items():
            f.write(json.dumps({"id": did, field: text}) + "\n")


# ---------------------------------------------------------------------------
# Sharding + aggregation (SLURM data-parallelism)
# ---------------------------------------------------------------------------


class TestSharding:
    def test_parse_shard_spec(self):
        from btc_eval.soap_judge.pipeline import parse_shard_spec

        assert parse_shard_spec("3/16") == (3, 16)
        assert parse_shard_spec("0/1") == (0, 1)
        for bad in ["16/16", "-1/4", "5/4", "abc", "4", "1/0"]:
            with pytest.raises(ValueError):
                parse_shard_spec(bad)

    def test_select_shard_is_a_partition(self):
        from btc_eval.soap_judge.pipeline import select_shard

        keys = [f"dialog_{i:04d}" for i in range(200)]
        n = 7
        shards = [select_shard(keys, i, n) for i in range(n)]
        # Disjoint and complete.
        assert sum(len(s) for s in shards) == len(keys)
        assert set().union(*[set(s) for s in shards]) == set(keys)
        # Deterministic.
        assert select_shard(keys, 2, n) == select_shard(keys, 2, n)
        # Reasonably balanced (no shard wildly off for 200/7).
        assert all(10 < len(s) < 60 for s in shards)


class TestShardThenAggregateCLI:
    def test_end_to_end(self, tmp_path):
        preds = tmp_path / "preds.jsonl"
        trans = tmp_path / "trans.jsonl"
        notes = {f"d{i:02d}": f"S: c{i}.\nO: o{i}.\nA: a{i}.\nP: p{i}." for i in range(20)}
        tr = {k: f"DOCTOR: hi {k}\nPATIENT: sx {k}" for k in notes}
        _write_jsonl(preds, notes, "summary")
        _write_jsonl(trans, tr, "transcript")
        cwd = str(Path(__file__).parent.parent)

        def run(*args):
            return subprocess.run(
                [sys.executable, "-m", "btc_eval.cli", *args],
                capture_output=True, text=True, cwd=cwd,
            )

        n_shards = 4
        for s in range(n_shards):
            r = run("soap-judge", "--backend", "mock", "--model", "mock",
                    "--predictions", str(preds), "--transcripts", str(trans),
                    "--shard", f"{s}/{n_shards}", "--output", str(tmp_path / f"shard_{s}"))
            assert r.returncode == 0, r.stderr

        # Shards together cover every dialog exactly once.
        seen = set()
        for s in range(n_shards):
            for line in (tmp_path / f"shard_{s}" / "soap_judge_per_dialog.jsonl").read_text().splitlines():
                seen.add(json.loads(line)["id"])
        assert seen == set(notes)

        # Aggregate recomputes over the union.
        r = run("soap-aggregate",
                "--inputs", str(tmp_path / "shard_*/soap_judge_per_dialog.jsonl"),
                "--output", str(tmp_path / "agg"))
        assert r.returncode == 0, r.stderr
        summary = json.loads((tmp_path / "agg" / "summary.json").read_text())
        assert summary["num_dialogs"] == 20
        merged = (tmp_path / "agg" / "soap_eval_summary.csv").read_text().splitlines()
        assert merged[0].split(",") == list(SOAP_CSV_COLUMNS)
        assert len(merged) == 21  # header + 20


class TestSoapJudgeCLI:
    def test_end_to_end_mock(self, tmp_path):
        preds = tmp_path / "preds.jsonl"
        trans = tmp_path / "trans.jsonl"
        out = tmp_path / "out"
        _write_jsonl(preds, PREDICTIONS, "summary")
        _write_jsonl(trans, TRANSCRIPTS, "transcript")

        result = subprocess.run(
            [
                sys.executable, "-m", "btc_eval.cli", "soap-judge",
                "--backend", "mock",
                "--model", "mock",
                "--predictions", str(preds),
                "--transcripts", str(trans),
                "--output", str(out),
                "--workers", "2",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "SOAP LLM-Judge Results" in result.stdout

        # CSV exists with the 13-column header in order.
        csv_lines = (out / "soap_eval_summary.csv").read_text().strip().split("\n")
        assert csv_lines[0].split(",") == list(SOAP_CSV_COLUMNS)
        assert len(csv_lines) == 1 + len(PREDICTIONS)

        # Aggregate summary.
        summary = json.loads((out / "summary.json").read_text())
        assert summary["num_dialogs"] == 2
        assert summary["backend"] == "mock"
        assert summary["num_failures"] == 0
        assert summary["extract_parse_failures"] == 0
        # Timing instrumentation is recorded per stage.
        assert summary["timing"]["extract"]["llm_calls"] == 2
        assert summary["timing"]["judge"]["llm_calls"] == 2
        assert summary["timing"]["wall_sec"] >= 0

        # Per-dialog JSONL.
        per_dialog = (out / "soap_judge_per_dialog.jsonl").read_text().strip().split("\n")
        assert len(per_dialog) == 2


# ---------------------------------------------------------------------------
# Lenient repair, model-keyed cache, and the known-error report
# ---------------------------------------------------------------------------


class TestRepairFallback:
    def test_strict_status(self):
        from btc_eval.soap_judge.parse import parse_judgment_with_status
        obj, status = parse_judgment_with_status('{"subscores_1_to_5": {"faithfulness_grounding": 3}}')
        assert obj is not None and status == "strict"

    def test_truncated_recovers_as_repaired(self):
        from btc_eval.soap_judge.parse import _repair_json, parse_judgment_with_status
        if _repair_json is None:
            pytest.skip("json-repair not installed")
        truncated = ('```json\n{"subscores_1_to_5": {"faithfulness_grounding": 2, '
                     '"coverage_completeness": 3}, "claim_judgments": [{"claim": "x", "verd')
        obj, status = parse_judgment_with_status(truncated)
        assert obj is not None and status == "repaired"
        assert obj["subscores_1_to_5"]["faithfulness_grounding"] == 2

    def test_unparseable_and_empty_status(self):
        from btc_eval.soap_judge.parse import parse_judgment_with_status
        assert parse_judgment_with_status("not json")[1] == "unparseable"
        assert parse_judgment_with_status("")[1] == "empty"


class TestCacheModelKey:
    def test_model_mismatch_is_a_miss(self, tmp_path):
        from btc_eval.soap_judge.pipeline import _load_cached_raw, _write_cached_raw
        _write_cached_raw(tmp_path, "d1", "model-A", '{"subscores_1_to_5": {}}')
        assert _load_cached_raw(tmp_path, "d1", "model-A") is not None  # same model -> hit
        assert _load_cached_raw(tmp_path, "d1", "model-B") is None      # different -> miss (re-issue)


class TestErrorReport:
    def test_buckets_classify_failures(self):
        from btc_eval.soap_judge.cli import _buckets_from_status
        records = [
            {"id": "a", "judge_status": "parsed", "judge_parse": "strict", "empty_claims": False},
            {"id": "b", "judge_status": "parsed", "judge_parse": "repaired", "empty_claims": False},
            {"id": "c", "judge_status": "parse_failed", "judge_parse": "unparseable", "empty_claims": False},
            {"id": "e", "judge_status": "llm_error", "judge_parse": "llm_error", "empty_claims": False},
            {"id": "f", "judge_status": "parsed", "judge_parse": "strict", "empty_claims": True},
        ]
        errors, coverage = _buckets_from_status(records)
        assert coverage == {"attempted": 5, "scored": 3, "failed": 2}
        assert errors["judge_unparseable"]["ids"] == ["c"]
        assert errors["judge_llm_error"]["ids"] == ["e"]
        assert errors["judge_repaired"]["ids"] == ["b"]               # recovered, still scored
        assert errors["extract_failed_empty_claims"]["ids"] == ["f"]  # faithfulness-inflated, still scored


class TestEmptyClaimsExclusion:
    @staticmethod
    def _score(faith, hall):
        from btc_eval.types import SoapScores
        return SoapScores(
            faithfulness=faith, structure=3, coverage=3, conciseness=3,
            over_medicalization=0, under_medicalization=0, over_specific=0,
            hallucination_rate=hall, contradiction_rate=0.0,
            missed_claims=0, critical_omissions=0, redundancy_count=0,
        )

    def test_faithfulness_and_hallucination_exclude_empty_claims(self):
        from btc_eval.soap_judge.pipeline import aggregate_soap
        # two real dialogs + one extract-failed dialog with INFLATED faithfulness (5) / clean hall (0)
        scores = [self._score(2, 0.5), self._score(2, 0.5), self._score(5, 0.0)]
        agg = aggregate_soap(scores, [False, False, True])
        assert agg["num_dialogs"] == 3                       # structure/coverage/concise use all 3
        assert agg["mean_structure"] == 3.0
        assert agg["mean_faithfulness"] == 2.0               # NOT (2+2+5)/3 = 3.0
        assert agg["mean_hallucination_rate"] == 0.5         # NOT (0.5+0.5+0)/3
        assert agg["n_faithfulness"] == 2
        assert agg["n_excluded_empty_claims"] == 1

    def test_no_exclusion_is_backward_compatible(self):
        from btc_eval.soap_judge.pipeline import aggregate_soap
        scores = [self._score(2, 0.5), self._score(2, 0.5), self._score(5, 0.0)]
        agg = aggregate_soap(scores)                          # no empty_claims arg
        assert agg["mean_faithfulness"] == 3.0                # all 3 counted
        assert "n_excluded_empty_claims" not in agg
