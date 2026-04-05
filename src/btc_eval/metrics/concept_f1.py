"""Medical concept F1/Precision/Recall evaluation.

Extracts medical concept identifiers from reference and hypothesis texts,
then computes set-overlap metrics.
"""

from __future__ import annotations

from btc_eval.matchers import Matcher
from btc_eval.types import AggregateMetrics, ConceptMetrics


def extract_concept_ids(text: str, matcher: Matcher) -> set[str]:
    """Extract a set of concept CUIs from text using the given matcher."""
    results = matcher.match(text)
    cuis: set[str] = set()
    for concept_group in results:
        for concept in concept_group:
            cuis.add(concept["cui"])
    return cuis


def compute_concept_metrics(
    ref_concepts: set[str],
    hyp_concepts: set[str],
    eps: float = 1e-6,
) -> ConceptMetrics:
    """Compute concept-level precision, recall, and F1.

    Args:
        ref_concepts: Set of concept CUIs from the reference summary.
        hyp_concepts: Set of concept CUIs from the hypothesis (predicted) summary.
        eps: Small value added to denominators to avoid division by zero when
            concept sets are empty.
    """
    overlap = ref_concepts & hyp_concepts
    p = len(overlap) / (len(hyp_concepts) + eps)
    r = len(overlap) / (len(ref_concepts) + eps)
    f1 = 2 * p * r / (p + r + eps)
    return ConceptMetrics(
        f1=f1,
        precision=p,
        recall=r,
        ref_count=len(ref_concepts),
        hyp_count=len(hyp_concepts),
        overlap_count=len(overlap),
    )


def aggregate_metrics(per_dialog: list[ConceptMetrics]) -> AggregateMetrics:
    """Compute mean and std across a list of per-dialog metrics."""
    n = len(per_dialog)
    if n == 0:
        return AggregateMetrics(0, 0, 0, 0, 0, 0, 0)

    def _mean_std(values: list[float]) -> tuple[float, float]:
        mean = sum(values) / n
        std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
        return mean, std

    f1s = [m.f1 for m in per_dialog]
    precs = [m.precision for m in per_dialog]
    recs = [m.recall for m in per_dialog]

    mf1, sf1 = _mean_std(f1s)
    mp, sp = _mean_std(precs)
    mr, sr = _mean_std(recs)

    return AggregateMetrics(
        mean_f1=mf1,
        std_f1=sf1,
        mean_precision=mp,
        std_precision=sp,
        mean_recall=mr,
        std_recall=sr,
        num_dialogs=n,
    )
