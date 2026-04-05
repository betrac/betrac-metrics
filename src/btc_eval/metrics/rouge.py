"""ROUGE evaluation metrics (1/2/3/4/L).

Requires the optional ``rouge-score`` dependency:
    uv pip install "btc-eval[rouge]"
"""

from __future__ import annotations

from btc_eval.types import ROUGE_TYPES, RougeResult, RougeScores


def _get_scorer(use_stemmer: bool = False, rouge_types: tuple[str, ...] | None = None):
    """Lazily import and create a RougeScorer."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for ROUGE evaluation. "
            "Install with:  uv pip install 'btc-eval[rouge]'  "
            "or:  pip install rouge-score"
        )
    types = rouge_types if rouge_types is not None else ROUGE_TYPES
    return rouge_scorer.RougeScorer(list(types), use_stemmer=use_stemmer)


def compute_rouge(
    reference: str,
    hypothesis: str,
    scorer=None,
    use_stemmer: bool = False,
    rouge_types: tuple[str, ...] | None = None,
) -> RougeResult:
    """Compute ROUGE scores for a single reference–hypothesis pair.

    Parameters
    ----------
    reference : str
        Reference summary text.
    hypothesis : str
        Hypothesis (predicted) summary text.
    scorer : RougeScorer or None
        Pre-created scorer for reuse across calls. If None, creates one.
    use_stemmer : bool
        Whether to use Porter stemmer (only used if scorer is None).
    rouge_types : tuple of str or None
        Which ROUGE types to compute. Defaults to all (ROUGE_TYPES).

    Returns
    -------
    RougeResult with scores for each requested ROUGE type.
    """
    types = rouge_types if rouge_types is not None else ROUGE_TYPES
    if scorer is None:
        scorer = _get_scorer(use_stemmer, rouge_types=types)

    raw = scorer.score(reference, hypothesis)
    scores = {}
    for rouge_type in types:
        s = raw[rouge_type]
        scores[rouge_type] = RougeScores(
            precision=s.precision,
            recall=s.recall,
            fmeasure=s.fmeasure,
        )
    return RougeResult(scores=scores)


def aggregate_rouge(
    results: list[RougeResult],
    rouge_types: tuple[str, ...] | None = None,
) -> dict:
    """Compute mean and std of ROUGE scores across dialogs.

    Returns a flat dict like:
        {"rouge2_f_mean": 0.45, "rouge2_f_std": 0.12, ...}
    """
    types = rouge_types if rouge_types is not None else ROUGE_TYPES
    n = len(results)
    if n == 0:
        return {"num_dialogs": 0}

    agg: dict = {"num_dialogs": n}

    for rouge_type in types:
        for metric in ("fmeasure", "precision", "recall"):
            suffix = {"fmeasure": "f", "precision": "p", "recall": "r"}[metric]
            values = [getattr(r.scores[rouge_type], metric) for r in results]
            mean = sum(values) / n
            std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
            agg[f"{rouge_type}_{suffix}_mean"] = mean
            agg[f"{rouge_type}_{suffix}_std"] = std

    return agg
