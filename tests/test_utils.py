"""Tests for confidence interval utilities."""

import pytest

from btc_eval.metrics.utils import analytical_ci, bootstrap_ci


class TestBootstrapCI:
    def test_constant_values(self):
        """All-same values should have a zero-width CI."""
        lo, hi = bootstrap_ci([0.5, 0.5, 0.5, 0.5, 0.5])
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_returns_ordered(self):
        """lo should be <= hi."""
        lo, hi = bootstrap_ci([0.1, 0.5, 0.9, 0.3, 0.7])
        assert lo <= hi

    def test_mean_within_ci(self):
        """The sample mean should fall within the CI."""
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        lo, hi = bootstrap_ci(values)
        mean = sum(values) / len(values)
        assert lo <= mean <= hi

    def test_reproducible_with_seed(self):
        """Same seed should give same result."""
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        r1 = bootstrap_ci(values, seed=42)
        r2 = bootstrap_ci(values, seed=42)
        assert r1 == r2

    def test_different_seeds(self):
        """Different seeds may give different results."""
        values = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8]
        r1 = bootstrap_ci(values, seed=42)
        r2 = bootstrap_ci(values, seed=99)
        # They could theoretically be equal, but almost certainly won't be
        assert r1 != r2


class TestAnalyticalCI:
    def test_zero_std(self):
        """Zero std should give zero half-width."""
        half = analytical_ci(0.0, 10)
        assert half == pytest.approx(0.0)

    def test_positive_half_width(self):
        """Non-zero std should give positive half-width."""
        half = analytical_ci(0.1, 10)
        assert half > 0

    def test_larger_n_smaller_ci(self):
        """More samples should give a narrower CI."""
        half_small = analytical_ci(0.1, 10)
        half_large = analytical_ci(0.1, 100)
        assert half_large < half_small

    def test_known_value(self):
        """Check against hand-computed value: z * std / sqrt(n)."""
        # z_0.975 = 1.959964, std=1.0, n=100 -> 1.959964 * 1.0 / 10 = 0.1959964
        half = analytical_ci(1.0, 100)
        assert half == pytest.approx(0.1959964, rel=1e-4)
