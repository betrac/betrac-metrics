"""Data types for evaluation results."""

from dataclasses import dataclass, field


@dataclass
class ConceptMetrics:
    """Metrics for a single dialog's concept-based evaluation."""

    f1: float
    precision: float
    recall: float
    ref_count: int
    hyp_count: int
    overlap_count: int

    def to_dict(self) -> dict:
        return {
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "ref_count": self.ref_count,
            "hyp_count": self.hyp_count,
            "overlap_count": self.overlap_count,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all dialogs."""

    mean_f1: float
    std_f1: float
    mean_precision: float
    std_precision: float
    mean_recall: float
    std_recall: float
    num_dialogs: int

    def to_dict(self) -> dict:
        return {
            "mean_f1": self.mean_f1,
            "std_f1": self.std_f1,
            "mean_precision": self.mean_precision,
            "std_precision": self.std_precision,
            "mean_recall": self.mean_recall,
            "std_recall": self.std_recall,
            "num_dialogs": self.num_dialogs,
        }


# ROUGE types used by metrics/rouge.py

ROUGE_TYPES = ("rouge1", "rouge2", "rouge3", "rouge4", "rougeL")

# Competition metrics: only ROUGE-2 and ROUGE-3
COMPETITION_ROUGE_TYPES = ("rouge2", "rouge3")


@dataclass
class RougeScores:
    """Precision, recall, fmeasure for a single ROUGE type."""

    precision: float
    recall: float
    fmeasure: float


@dataclass
class RougeResult:
    """ROUGE scores for a single dialog across all ROUGE types."""

    scores: dict[str, RougeScores] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {}
        for rouge_type, s in self.scores.items():
            out[f"{rouge_type}_f"] = s.fmeasure
            out[f"{rouge_type}_p"] = s.precision
            out[f"{rouge_type}_r"] = s.recall
        return out

    def f_scores_dict(self) -> dict[str, float]:
        """Return only the F-measure for each ROUGE type."""
        return {rt: s.fmeasure for rt, s in self.scores.items()}
