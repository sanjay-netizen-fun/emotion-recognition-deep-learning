"""Model loading and inference.

Keras is imported lazily so that evaluation/reporting utilities (and their
tests) can run in environments where TensorFlow is not installed.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .detection import FaceDetector, crop_box
from .labels import EMOTIONS
from .preprocessing import INPUT_SIZE, preprocess_faces, to_gray

DEFAULT_MODEL_FILENAME = "emotiondetector.keras"
MODEL_PATH_ENV = "EMOTION_MODEL_PATH"
REPO_ROOT = Path(__file__).resolve().parent.parent


def default_model_path() -> Path:
    """Path to the trained model, overridable with ``EMOTION_MODEL_PATH``."""
    override = os.environ.get(MODEL_PATH_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / DEFAULT_MODEL_FILENAME


@lru_cache(maxsize=2)
def load_keras_model(model_path: str):
    """Load a Keras model from a ``.keras`` file (cached per path).

    Falls back to the ``emotiondetector.json`` + weights pair used by the
    original desktop app if the archive cannot be loaded directly.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"model file not found: {path}. "
            f"Set {MODEL_PATH_ENV} to point at a trained .keras model."
        )

    import keras  # imported here to keep this module import-light

    try:
        # compile=False keeps inference-only loads fast and avoids restoring
        # optimizer state that inference never needs.
        return keras.models.load_model(path, compile=False)
    except Exception as primary_error:  # noqa: BLE001 - fall back deliberately
        json_path = path.with_suffix(".json")
        if not json_path.is_file():
            raise RuntimeError(f"failed to load model from {path}") from primary_error
        with open(json_path, encoding="utf-8") as handle:
            model = keras.models.model_from_json(handle.read())
        model.load_weights(path)
        return model


@dataclass
class FacePrediction:
    """Prediction for a single detected face."""

    box: tuple[int, int, int, int]
    emotion: str
    confidence: float  # probability of the winning class, in [0, 1]
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def percent(self) -> float:
        return self.confidence * 100.0

    def as_dict(self, digits: int = 4) -> dict:
        return {
            "box": [int(value) for value in self.box],
            "emotion": self.emotion,
            "confidence": round(float(self.confidence), digits),
            "percent": round(float(self.percent), 2),
            "scores": {name: round(float(value), digits) for name, value in self.scores.items()},
        }


@dataclass
class FrameAnalysis:
    """Result of analysing one image or video frame."""

    width: int
    height: int
    faces: list[FacePrediction] = field(default_factory=list)
    fallback: bool = False  # True when no face was found and the whole frame was used
    elapsed_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.faces)

    def as_dict(self, digits: int = 4) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "count": self.count,
            "fallback": bool(self.fallback),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "faces": [face.as_dict(digits=digits) for face in self.faces],
        }


class EmotionRecognizer:
    """Detect faces and classify their expressions.

    The model expects ``(N, 48, 48, 1)`` grayscale, 0-1 normalized inputs and
    outputs a softmax distribution over :data:`emotion_recognition.labels.EMOTIONS`.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        model=None,
        labels: Sequence[str] = EMOTIONS,
        warmup: bool = True,
    ) -> None:
        if model is None:
            path = Path(model_path) if model_path is not None else default_model_path()
            model = load_keras_model(str(path))
            self.model_path: str | None = str(path)
        else:
            self.model_path = None

        self.model = model
        self.labels = tuple(labels)
        self._lock = threading.Lock()

        if warmup:
            self._warmup()

    @property
    def input_size(self) -> tuple[int, int]:
        return INPUT_SIZE

    def _warmup(self) -> None:
        """Run one dummy batch so the first real request is not the slow one."""
        try:
            self.predict_batch([np.zeros((*INPUT_SIZE[::-1],), dtype=np.uint8)])
        except Exception:  # noqa: BLE001 - warmup must never break construction
            pass

    def predict_batch(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        """Return an ``(N, 7)`` array of class probabilities."""
        batch = preprocess_faces(list(faces), size=self.input_size)
        if batch.shape[0] == 0:
            return np.empty((0, len(self.labels)), dtype=np.float32)

        with self._lock:
            try:
                output = self.model(batch, training=False)
            except TypeError:  # non-Keras callables (tests, other backends)
                output = self.model(batch)

        if isinstance(output, (list, tuple)):
            output = output[0]
        if hasattr(output, "numpy"):
            output = output.numpy()

        scores = np.asarray(output, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[1] != len(self.labels):
            raise ValueError(
                f"model returned shape {scores.shape}, expected (N, {len(self.labels)})"
            )
        return scores

    def predict_face(self, face: np.ndarray) -> FacePrediction:
        """Classify a single pre-cropped face image."""
        scores = self.predict_batch([face])[0]
        return self._to_prediction(scores, box=(0, 0, 0, 0))

    def analyze_frame(
        self,
        image: np.ndarray,
        *,
        detector: FaceDetector | None = None,
        max_faces: int = 5,
        fallback_to_frame: bool = True,
    ) -> FrameAnalysis:
        """Detect faces in an image/frame and classify each one.

        When no face is found and ``fallback_to_frame`` is true, the whole frame
        is classified as a single face. This mirrors what the desktop app does
        and keeps small, tightly cropped images (such as dataset samples)
        usable.
        """
        if image is None or np.asarray(image).size == 0:
            raise ValueError("image is empty")

        started = time.perf_counter()
        frame = np.asarray(image)
        height, width = frame.shape[:2]

        boxes = (
            detector.detect(frame)
            if detector is not None
            else np.empty((0, 4), dtype=np.int32)
        )

        fallback = False
        if len(boxes) == 0:
            if not fallback_to_frame:
                return FrameAnalysis(
                    width=width,
                    height=height,
                    faces=[],
                    fallback=False,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            boxes = np.array([[0, 0, width, height]], dtype=np.int32)
            fallback = True

        # Keep the largest faces, then restore a stable left-to-right order.
        areas = boxes[:, 2] * boxes[:, 3]
        if max_faces and len(boxes) > max_faces:
            keep = np.argsort(areas)[::-1][:max_faces]
            boxes = boxes[np.sort(keep)]

        crops = [crop_box(frame, box) for box in boxes]
        scores = self.predict_batch(crops)

        faces = [
            self._to_prediction(score_row, box=tuple(int(v) for v in box))
            for box, score_row in zip(boxes, scores, strict=True)
        ]

        return FrameAnalysis(
            width=width,
            height=height,
            faces=faces,
            fallback=fallback,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _to_prediction(self, scores: np.ndarray, box: tuple[int, int, int, int]) -> FacePrediction:
        index = int(np.argmax(scores))
        return FacePrediction(
            box=box,
            emotion=self.labels[index],
            confidence=float(scores[index]),
            scores={name: float(value) for name, value in zip(self.labels, scores, strict=True)},
        )


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes (JPEG/PNG/WebP/...) into a BGR array.

    Raises:
        ValueError: if the buffer is not a decodable image.
    """
    import cv2

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image data")
    return image


__all__ = [
    "DEFAULT_MODEL_FILENAME",
    "EmotionRecognizer",
    "FacePrediction",
    "FrameAnalysis",
    "default_model_path",
    "decode_image",
    "load_keras_model",
    "to_gray",
]
