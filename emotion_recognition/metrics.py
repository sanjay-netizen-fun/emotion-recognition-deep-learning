"""Lightweight classification metrics (NumPy only).

Kept dependency-free so evaluation reports can be generated without
scikit-learn, and unit-tested without TensorFlow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .labels import EMOTIONS


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    """Return a ``(C, C)`` matrix with rows = true class, columns = predicted."""
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)

    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred differ in length: {true.shape} vs {pred.shape}")
    if true.size and (
        true.min() < 0
        or pred.min() < 0
        or true.max() >= num_classes
        or pred.max() >= num_classes
    ):
        raise ValueError(f"labels must be in [0, {num_classes - 1}]")

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


@dataclass
class ClassificationMetrics:
    """Aggregate and per-class metrics for a multiclass problem."""

    labels: tuple[str, ...]
    matrix: np.ndarray
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    support: int = 0
    per_class: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "support": self.support,
            "accuracy": round(self.accuracy, 6),
            "balanced_accuracy": round(self.balanced_accuracy, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "macro_f1": round(self.macro_f1, 6),
            "weighted_f1": round(self.weighted_f1, 6),
            "per_class": [
                {key: (round(value, 6) if isinstance(value, float) else value) for key, value in row.items()}
                for row in self.per_class
            ],
            "confusion_matrix": self.matrix.tolist(),
            "labels": list(self.labels),
        }

    def report(self) -> str:
        """Human-readable classification report."""
        header = f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
        lines = [header, "-" * len(header)]
        for row in self.per_class:
            lines.append(
                f"{row['label']:<12}{row['precision']:>10.3f}{row['recall']:>10.3f}"
                f"{row['f1']:>10.3f}{row['support']:>10d}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'accuracy':<12}{'':>10}{'':>10}{self.accuracy:>10.3f}{self.support:>10d}"
        )
        lines.append(
            f"{'macro avg':<12}{self.macro_precision:>10.3f}{self.macro_recall:>10.3f}"
            f"{self.macro_f1:>10.3f}{self.support:>10d}"
        )
        lines.append(
            f"{'weighted':<12}{'':>10}{'':>10}{self.weighted_f1:>10.3f}{self.support:>10d}"
        )
        return "\n".join(lines)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str] = EMOTIONS,
) -> ClassificationMetrics:
    """Compute accuracy, per-class precision/recall/F1 and their averages."""
    names = tuple(labels)
    num_classes = len(names)
    matrix = confusion_matrix(y_true, y_pred, num_classes)

    support_total = int(matrix.sum())
    true_positive = np.diag(matrix).astype(float)
    predicted_total = matrix.sum(axis=0).astype(float)
    support = matrix.sum(axis=1).astype(float)

    precision = np.array([_safe_div(true_positive[i], predicted_total[i]) for i in range(num_classes)])
    recall = np.array([_safe_div(true_positive[i], support[i]) for i in range(num_classes)])
    f1 = np.array(
        [
            _safe_div(2 * precision[i] * recall[i], precision[i] + recall[i])
            for i in range(num_classes)
        ]
    )

    per_class = [
        {
            "label": names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    ]

    present = support > 0
    return ClassificationMetrics(
        labels=names,
        matrix=matrix,
        accuracy=_safe_div(true_positive.sum(), support_total),
        balanced_accuracy=float(recall[present].mean()) if present.any() else 0.0,
        macro_precision=float(precision[present].mean()) if present.any() else 0.0,
        macro_recall=float(recall[present].mean()) if present.any() else 0.0,
        macro_f1=float(f1[present].mean()) if present.any() else 0.0,
        weighted_f1=_safe_div(float((f1 * support).sum()), float(support.sum())),
        support=support_total,
        per_class=per_class,
    )


def top_k_accuracy(scores: np.ndarray, y_true: Sequence[int], k: int = 2) -> float:
    """Fraction of samples whose true class is in the model's top-k predictions."""
    probabilities = np.asarray(scores)
    true = np.asarray(y_true, dtype=np.int64)
    if probabilities.ndim != 2:
        raise ValueError(f"expected (N, C) scores, got shape {probabilities.shape}")
    if probabilities.shape[0] != true.shape[0]:
        raise ValueError("scores and labels differ in length")
    if not 1 <= k <= probabilities.shape[1]:
        raise ValueError(f"k must be between 1 and {probabilities.shape[1]}")

    top = np.argsort(probabilities, axis=1)[:, -k:]
    hits = (top == true[:, None]).any(axis=1)
    return float(hits.mean()) if hits.size else 0.0
