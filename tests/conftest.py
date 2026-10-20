"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeModel:
    """Stand-in for a Keras model: returns the same distribution for every input."""

    def __init__(self, probabilities: list[float] | None = None) -> None:
        self.probabilities = np.asarray(
            probabilities or [0.05, 0.05, 0.05, 0.6, 0.1, 0.1, 0.05], dtype=np.float32
        )
        self.calls = 0

    def __call__(self, batch, training: bool = False):
        assert training is False
        self.calls += 1
        return np.tile(self.probabilities, (len(batch), 1))


class StubDetector:
    """Detector that returns pre-scripted boxes."""

    def __init__(self, boxes: list[list[int]] | None = None) -> None:
        self.boxes = np.asarray(boxes if boxes is not None else [], dtype=np.int32).reshape(-1, 4)

    def detect(self, _image) -> np.ndarray:
        return self.boxes


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def gray_face() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def bgr_image() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(80, 64, 3), dtype=np.uint8)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Create a small train/test dataset with case-insensitive class folders."""
    layout = {
        "train/angry": 3,
        "train/Happy": 4,
        "train/Neutral": 2,
        "test/Angry": 1,
        "test/happy": 2,
    }
    from PIL import Image

    for relative, count in layout.items():
        folder = tmp_path / relative
        folder.mkdir(parents=True)
        for index in range(count):
            array = np.full((48, 48), 40 * index % 255, dtype=np.uint8)
            Image.fromarray(array, mode="L").save(folder / f"{index}.png")

    # Non-image and unknown-class files should be ignored by the scanner.
    (tmp_path / "train" / "angry" / "notes.txt").write_text("ignore me")
    (tmp_path / "train" / "NotAnEmotion").mkdir()
    (tmp_path / "train" / "NotAnEmotion" / "x.png").write_bytes(b"not really an image")
    return tmp_path
