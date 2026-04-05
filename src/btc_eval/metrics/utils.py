"""Confidence interval utilities for metric aggregation."""

from __future__ import annotations

import math

import numpy as np

_Z95 = 1.959964  # scipy.stats.norm.ppf(0.975), i.e. z for 95% CI


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return (lo, hi) bootstrap percentile CI for the mean of *values*.

    Resamples *n_boot* times with replacement and returns the
    ``alpha/2`` and ``1 - alpha/2`` percentiles of the bootstrap distribution
    of the mean, giving a two-sided ``(1 - alpha) * 100``% confidence interval.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    samples = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.percentile(samples, 100.0 * alpha / 2))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2)))
    return lo, hi


def analytical_ci(std: float, n: int) -> float:
    """Return the half-width of a 95% CI under a normal approximation.

    Computes ``z_0.975 * std / sqrt(n)`` which gives the half-width of the
    standard 95% confidence interval for the mean.
    """
    return _Z95 * std / math.sqrt(n)
