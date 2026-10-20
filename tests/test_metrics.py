import numpy as np
import pytest

from emotion_recognition.labels import EMOTIONS
from emotion_recognition.metrics import (
    classification_metrics,
    confusion_matrix,
    top_k_accuracy,
)


def test_confusion_matrix_counts_pairs():
    matrix = confusion_matrix([0, 0, 1, 2, 2], [0, 1, 1, 2, 0], num_classes=3)
    assert matrix.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]


def test_confusion_matrix_validation():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0], num_classes=2)
    with pytest.raises(ValueError):
        confusion_matrix([0, 3], [0, 1], num_classes=2)


def test_perfect_predictions_score_one():
    truth = list(range(7))
    metrics = classification_metrics(truth, truth, EMOTIONS)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.support == 7
    assert metrics.matrix.trace() == 7


def test_known_two_class_example():
    # 3 true positives for class 0, 1 false negative, 1 false positive.
    metrics = classification_metrics([0, 0, 0, 1], [0, 0, 1, 1], labels=("A", "B"))
    row = {item["label"]: item for item in metrics.per_class}
    assert row["A"]["precision"] == pytest.approx(1.0)
    assert row["A"]["recall"] == pytest.approx(2 / 3)
    assert row["B"]["precision"] == pytest.approx(0.5)
    assert row["B"]["recall"] == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(0.75)


def test_missing_class_does_not_break_averages():
    metrics = classification_metrics([0, 0, 1], [0, 1, 1], labels=("A", "B", "C"))
    row = {item["label"]: item for item in metrics.per_class}
    assert row["C"]["support"] == 0
    assert row["C"]["f1"] == 0.0
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_report_contains_labels_and_averages():
    metrics = classification_metrics([0, 1, 2], [0, 2, 2], EMOTIONS[:3])
    report = metrics.report()
    assert "Angry" in report and "Fear" in report
    assert "accuracy" in report
    assert "macro avg" in report


def test_to_dict_is_json_friendly():
    payload = classification_metrics([0, 1], [0, 0], EMOTIONS).to_dict()
    assert payload["support"] == 2
    assert len(payload["confusion_matrix"]) == 7
    assert payload["labels"] == list(EMOTIONS)
    assert isinstance(payload["per_class"][0]["precision"], float)


def test_top_k_accuracy():
    scores = np.array(
        [
            [0.1, 0.6, 0.3],
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )
    assert top_k_accuracy(scores, [1, 0, 0], k=1) == pytest.approx(2 / 3)
    assert top_k_accuracy(scores, [1, 0, 0], k=2) == pytest.approx(2 / 3)
    assert top_k_accuracy(scores, [0, 0, 0], k=2) == pytest.approx(1 / 3)


def test_top_k_validation():
    with pytest.raises(ValueError):
        top_k_accuracy(np.zeros((2, 3)), [0, 1], k=4)
    with pytest.raises(ValueError):
        top_k_accuracy(np.zeros((2, 3)), [0], k=1)
