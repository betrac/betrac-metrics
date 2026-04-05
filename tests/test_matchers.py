"""Tests for matchers."""

from btc_eval.matchers import Matcher
from btc_eval.matchers.mock import MockMatcher
from btc_eval.matchers.open_medical import OpenMedicalMatcher


class TestMockMatcher:
    def test_implements_protocol(self):
        assert isinstance(MockMatcher(), Matcher)

    def test_empty_text(self, mock_matcher):
        assert mock_matcher.match("") == []

    def test_no_concepts(self, mock_matcher):
        assert mock_matcher.match("hello world") == []

    def test_single_concept(self, mock_matcher):
        results = mock_matcher.match("Patient has chest pain")
        assert len(results) == 1
        cuis = {c["cui"] for c in results[0]}
        assert "C0008031" in cuis  # chest pain
        assert "C0030705" in cuis  # patient

    def test_multiple_concepts(self, mock_matcher):
        results = mock_matcher.match("diabetes managed with metformin")
        cuis = {c["cui"] for c in results[0]}
        assert "C0011849" in cuis  # diabetes
        assert "C0025598" in cuis  # metformin

    def test_case_insensitive(self, mock_matcher):
        results = mock_matcher.match("CHEST PAIN")
        assert len(results) == 1

    def test_no_partial_match(self, mock_matcher):
        # "dia" should not match "diabetes"
        assert mock_matcher.match("dia") == []


class TestOpenMedicalMatcher:
    def test_implements_protocol(self):
        matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=False)
        assert isinstance(matcher, Matcher)

    def test_mesh_only(self):
        matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=False)
        results = matcher.match("Patient has chest pain and hypertension")
        assert len(results) == 1
        cuis = {c["cui"] for c in results[0]}
        assert "D002637" in cuis  # chest pain
        assert "D006973" in cuis  # hypertension

    def test_empty_text(self):
        matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=False)
        assert matcher.match("") == []

    def test_expanded_vocabulary(self):
        """Verify the expanded vocabulary has more than the original 32 terms."""
        matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=False)
        assert len(matcher.mesh_concepts) > 50

    def test_new_terms(self):
        """Test concepts added in the expanded vocabulary."""
        matcher = OpenMedicalMatcher(use_mesh=True, use_scispacy=False)
        results = matcher.match("Patient reports nausea, headache, and back pain")
        cuis = {c["cui"] for c in results[0]}
        assert "D009325" in cuis  # nausea
        assert "D006261" in cuis  # headache
        assert "D001416" in cuis  # back pain
