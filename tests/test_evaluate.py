import json

import numpy as np
import pytest

import evaluate
from emotion_recognition.labels import EMOTIONS


class StubRecognizer:
    """Every face is classified as Happy (index 3)."""

    probabilities = np.array([0.05, 0.05, 0.05, 0.6, 0.1, 0.1, 0.05], dtype=np.float32)

    def __init__(self, *args, **kwargs):
        self.labels = EMOTIONS

    def predict_batch(self, faces):
        return np.tile(self.probabilities, (len(faces), 1))


@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(evaluate, "EmotionRecognizer", StubRecognizer)
    monkeypatch.setattr(evaluate, "default_model_path", lambda: "stub.keras")


def test_parse_args_defaults():
    args = evaluate.parse_args(["--data", "images"])
    assert args.data.name == "images"
    assert args.split is None
    assert args.detect is False
    assert args.batch_size == 64


def test_format_confusion_matrix():
    matrix = np.array([[2, 1], [0, 3]])
    text = evaluate.format_confusion(matrix, ("Happy", "Sad"))
    assert "Hap" in text and "Sad" in text
    assert "true\\pred" in text


def test_most_confused_pairs_ignores_diagonal():
    matrix = np.zeros((3, 3), dtype=int)
    np.fill_diagonal(matrix, 5)
    matrix[0, 1] = 2
    matrix[2, 0] = 4
    pairs = evaluate.most_confused_pairs(matrix, ("A", "B", "C"))
    assert pairs[0] == {"true": "C", "predicted": "A", "count": 4}
    assert all(pair["true"] != pair["predicted"] for pair in pairs)


def test_load_gray_image(tmp_path):
    from PIL import Image

    path = tmp_path / "colour.png"
    Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8), mode="RGB").save(path)
    image = evaluate.load_gray_image(path)
    assert image.shape == (20, 30)


def test_load_gray_image_returns_none_for_garbage(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"nope")
    assert evaluate.load_gray_image(path) is None


def test_evaluate_cli_writes_reports(dataset_root, tmp_path, stub_model):
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "predictions.csv"

    exit_code = evaluate.main(
        [
            "--data",
            str(dataset_root / "train"),
            "--json",
            str(json_path),
            "--csv",
            str(csv_path),
            "--errors",
            "0",
        ]
    )

    assert exit_code == 0
    report = json.loads(json_path.read_text())
    assert report["metrics"]["support"] == 9
    assert report["metrics"]["accuracy"] == pytest.approx(4 / 9, abs=1e-4)
    assert report["top2_accuracy"] == pytest.approx(6 / 9, abs=1e-4)
    assert len(csv_path.read_text().strip().splitlines()) == 10


def test_evaluate_cli_threshold_failure(dataset_root, stub_model, capsys):
    exit_code = evaluate.main(
        [
            "--data",
            str(dataset_root / "train"),
            "--limit",
            "4",
            "--min-accuracy",
            "0.99",
            "--errors",
            "0",
        ]
    )
    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().err


def test_evaluate_cli_bad_dataset(tmp_path, stub_model):
    assert evaluate.main(["--data", str(tmp_path / "missing")]) == 2


def test_evaluate_cli_bad_split(dataset_root, stub_model):
    assert evaluate.main(["--data", str(dataset_root), "--split", "test"]) == 0
    assert evaluate.main(["--data", str(dataset_root / "train"), "--split", "test"]) == 2
