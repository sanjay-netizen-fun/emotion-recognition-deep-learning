"""Image preprocessing shared by the desktop app, evaluation and web demo.

Training (``trainmodel.ipynb``) loads each dataset image with
``keras.utils.load_img(..., color_mode="grayscale")``, converts it to an array
and reshapes it to ``(N, 48, 48, 1)`` before dividing by 255. These helpers
reproduce exactly that pipeline for single faces and batches.
"""

from __future__ import annotations

import cv2
import numpy as np

INPUT_SIZE: tuple[int, int] = (48, 48)


def _as_array(image: np.ndarray | object) -> np.ndarray:
    """Convert PIL images / array-likes to a numpy array, rejecting empties."""
    if image is None:
        raise ValueError("image is None")
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image is empty")
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image array, got shape {array.shape}")
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValueError(f"unsupported channel count in shape {array.shape}")
    return array


def to_gray(image: np.ndarray | object) -> np.ndarray:
    """Return a single-channel grayscale image.

    Accepts grayscale, BGR and BGRA arrays (OpenCV convention), as well as PIL
    images. Channel order is assumed to be OpenCV's BGR for 3/4 channel inputs.

    Integer inputs are converted to ``uint8``. Floating point inputs keep their
    dtype (and therefore their precision), so arrays that were already scaled to
    ``[0, 1]`` survive a second preprocessing pass unchanged. This makes
    :func:`preprocess_face` idempotent.
    """
    array = _as_array(image)

    if array.ndim == 2:
        gray = array
    elif array.shape[2] == 1:
        gray = array[:, :, 0]
    elif array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    if np.issubdtype(gray.dtype, np.floating):
        return np.ascontiguousarray(gray.astype(np.float32, copy=False))

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(gray)


def _to_unit_range(gray: np.ndarray) -> np.ndarray:
    """Scale an image to ``[0, 1]`` as float32.

    Images already in ``[0, 1]`` are left untouched, so applying the whole
    preprocessing pipeline twice produces the same result as applying it once.
    """
    values = gray.astype(np.float32, copy=False)
    if values.size and values.max() <= 1.0 and values.min() >= 0.0:
        return values
    return values / 255.0


def preprocess_face(
    face: np.ndarray | object,
    size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Turn one face image into a ``(48, 48, 1)`` normalized float32 array."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid target size {size!r}")

    gray = to_gray(face)
    if gray.shape[:2] != (height, width):
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    return _to_unit_range(gray).reshape(height, width, 1)


def preprocess_faces(
    faces: list[np.ndarray] | tuple[np.ndarray, ...],
    size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Stack several faces into a ``(N, 48, 48, 1)`` batch.

    An empty input yields an empty batch instead of raising, which keeps batch
    inference code branch-free.
    """
    width, height = size
    if len(faces) == 0:
        return np.empty((0, height, width, 1), dtype=np.float32)

    batch = np.stack([preprocess_face(face, size=size) for face in faces], axis=0)
    return batch.astype(np.float32, copy=False)
