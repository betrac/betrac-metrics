"""Shared test fixtures."""

from pathlib import Path

import pytest

from btc_eval.matchers.mock import MockMatcher


@pytest.fixture
def mock_matcher():
    return MockMatcher()


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_predictions_path(data_dir):
    return data_dir / "sample_predictions.jsonl"


@pytest.fixture
def sample_references_path(data_dir):
    return data_dir / "sample_references.jsonl"
