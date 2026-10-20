"""Model definition and dataset loading for training.

``build_model`` mirrors the architecture stored in ``emotiondetector.json`` so
that ``train.py`` reproduces the shipped model from scratch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .dataset import DatasetSample
from .labels import NUM_CLASSES
from .preprocessing import INPUT_SIZE, preprocess_face


def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.4):
    """Build the convolutional network used by ``emotiondetector.keras``.

    Architecture: four ``Conv2D`` blocks (128/256/512/512 filters, 3x3, valid
    padding) each followed by max pooling and dropout, then a classifier head
    with 512 and 256 ReLU units and a softmax output.
    """
    from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
    from keras.models import Sequential

    width, height = INPUT_SIZE
    return Sequential(
        [
            Input(shape=(height, width, 1)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
            Conv2D(256, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
            Conv2D(512, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
            Conv2D(512, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(dropout),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ],
        name="emotion_cnn",
    )


def load_sample_images(
    samples: Sequence[DatasetSample],
    *,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[DatasetSample]]:
    """Read samples into ``(X, y, kept_samples)`` arrays.

    Images that cannot be decoded are skipped and reported in the returned list
    only if they were readable, so ``X.shape >= y.shape`` always holds.
    """
    import cv2

    try:
        from tqdm import tqdm

        iterator: Iterable = tqdm(samples, desc="loading", unit="img", disable=not progress)
    except ImportError:  # pragma: no cover - tqdm is optional
        iterator = samples

    features: list[np.ndarray] = []
    labels: list[int] = []
    kept: list[DatasetSample] = []

    for sample in iterator:
        image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        features.append(preprocess_face(image))
        labels.append(sample.index)
        kept.append(sample)

    if not features:
        raise ValueError("no readable images found")

    return (
        np.stack(features, axis=0),
        np.asarray(labels, dtype=np.int64),
        kept,
    )


def one_hot(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """One-hot encode integer labels (matches ``to_categorical``)."""
    from keras.utils import to_categorical

    return to_categorical(labels, num_classes=num_classes)
