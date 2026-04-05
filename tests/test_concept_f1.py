"""Tests for concept F1 metrics."""

from btc_eval.metrics.concept_f1 import (
    aggregate_metrics,
    compute_concept_metrics,
    extract_concept_ids,
)
from btc_eval.types import ConceptMetrics


class TestExtractConceptIds:
    def test_basic(self, mock_matcher):
        cuis = extract_concept_ids("Patient has diabetes and hypertension", mock_matcher)
        assert "C0011849" in cuis  # diabetes
        assert "C0020538" in cuis  # hypertension

    def test_empty_text(self, mock_matcher):
        cuis = extract_concept_ids("", mock_matcher)
        assert cuis == set()

    def test_no_medical_terms(self, mock_matcher):
        cuis = extract_concept_ids("The weather is nice today", mock_matcher)
        assert cuis == set()


class TestComputeConceptMetrics:
    def test_perfect_match(self):
        ref = {"A", "B", "C"}
        hyp = {"A", "B", "C"}
        m = compute_concept_metrics(ref, hyp)
        assert m.f1 > 0.99
        assert m.precision > 0.99
        assert m.recall > 0.99
        assert m.overlap_count == 3

    def test_no_overlap(self):
        ref = {"A", "B"}
        hyp = {"C", "D"}
        m = compute_concept_metrics(ref, hyp)
        assert m.f1 < 0.01
        assert m.overlap_count == 0

    def test_partial_overlap(self):
        ref = {"A", "B", "C"}
        hyp = {"B", "C", "D"}
        m = compute_concept_metrics(ref, hyp)
        # overlap = {B, C}, |overlap|=2, |hyp|=3, |ref|=3
        assert abs(m.precision - 2 / 3) < 0.01
        assert abs(m.recall - 2 / 3) < 0.01
        assert m.overlap_count == 2

    def test_empty_hypothesis(self):
        ref = {"A", "B"}
        hyp = set()
        m = compute_concept_metrics(ref, hyp)
        assert m.f1 < 0.01
        assert m.hyp_count == 0

    def test_empty_reference(self):
        ref = set()
        hyp = {"A", "B"}
        m = compute_concept_metrics(ref, hyp)
        assert m.f1 < 0.01
        assert m.ref_count == 0


class TestAggregateMetrics:
    def test_single_dialog(self):
        m = ConceptMetrics(
            f1=0.8, precision=0.7, recall=0.9, ref_count=10, hyp_count=10, overlap_count=7
        )
        agg = aggregate_metrics([m])
        assert agg.mean_f1 == 0.8
        assert agg.std_f1 == 0.0
        assert agg.num_dialogs == 1

    def test_multiple_dialogs(self):
        metrics = [
            ConceptMetrics(
                f1=0.8, precision=0.7, recall=0.9, ref_count=10, hyp_count=10, overlap_count=7
            ),
            ConceptMetrics(
                f1=0.6, precision=0.5, recall=0.7, ref_count=10, hyp_count=10, overlap_count=5
            ),
        ]
        agg = aggregate_metrics(metrics)
        assert abs(agg.mean_f1 - 0.7) < 0.01
        assert agg.std_f1 > 0
        assert agg.num_dialogs == 2

    def test_empty(self):
        agg = aggregate_metrics([])
        assert agg.num_dialogs == 0
        assert agg.mean_f1 == 0
