"""Haar-cascade face detection helpers.

OpenCV's ``opencv-python`` wheels bundle the cascade XML files under
``cv2.data.haarcascades``. Some distributions (for example OpenCV 5 headless
wheels) ship the Python package without that data directory, so the lookup here
also checks a few well-known locations and raises a readable error instead of
failing with ``cv2.error`` deep inside ``detectMultiScale``.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import to_gray

CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


def _candidate_directories() -> list[Path]:
    candidates: list[Path] = []

    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        candidates.append(Path(data_dir))

    cv2_dir = getattr(cv2, "__file__", None)
    if cv2_dir:
        candidates.append(Path(cv2_dir).parent / "data")

    if hasattr(sys, "_MEIPASS"):  # PyInstaller one-file builds
        candidates.append(Path(sys._MEIPASS))

    return candidates


def cascade_path(filename: str = CASCADE_FILENAME) -> Path:
    """Locate a bundled Haar cascade XML file.

    Raises:
        FileNotFoundError: if no cascade with that name can be found.
    """
    for directory in _candidate_directories():
        candidate = directory / filename
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(directory) for directory in _candidate_directories())
    raise FileNotFoundError(
        f"could not find {filename}; searched {searched or '(no OpenCV data directory)'}. "
        "Install a regular 'opencv-python' wheel (4.x includes the cascade data) "
        "or pass an explicit cascade path."
    )


class FaceDetector:
    """Thin, deterministic wrapper around OpenCV's frontal-face cascade."""

    def __init__(
        self,
        cascade_file: str | os.PathLike[str] | None = None,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_neighbors < 0:
            raise ValueError("min_neighbors must be non-negative")

        path = Path(cascade_file) if cascade_file is not None else cascade_path()
        self.cascade_file = path
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = (int(min_size[0]), int(min_size[1]))

        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise ValueError(f"failed to load Haar cascade from {path}")
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray | object) -> np.ndarray:
        """Detect faces and return an ``(N, 4)`` int array of ``x, y, w, h`` boxes.

        Boxes are ordered left-to-right so that repeated runs label faces in the
        same order.
        """
        gray = to_gray(image)
        with self._lock:
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )

        if len(faces) == 0:
            return np.empty((0, 4), dtype=np.int32)

        boxes = np.asarray(faces, dtype=np.int32).reshape(-1, 4)
        # Sort by left edge, then top edge, for stable output ordering.
        order = np.lexsort((boxes[:, 1], boxes[:, 0]))
        return boxes[order]

    def __call__(self, image: np.ndarray | object) -> np.ndarray:
        return self.detect(image)


def crop_box(image: np.ndarray, box: tuple[int, int, int, int] | np.ndarray) -> np.ndarray:
    """Crop a box from an image, clipped to the image bounds."""
    height, width = image.shape[:2]
    x, y, w, h = (int(value) for value in box)
    x0 = max(0, min(x, width - 1))
    y0 = max(0, min(y, height - 1))
    x1 = max(x0 + 1, min(x + w, width))
    y1 = max(y0 + 1, min(y + h, height))
    return image[y0:y1, x0:x1]
