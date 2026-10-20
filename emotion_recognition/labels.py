"""Emotion class definitions.

The training notebook builds labels with ``sklearn.preprocessing.LabelEncoder``
fitted on the dataset folder names. ``LabelEncoder`` sorts classes
alphabetically, which for the FER-style folders used here gives exactly the
order below. Keeping that order in one place avoids the classic
"model index != display label" bug.
"""

from __future__ import annotations

EMOTIONS: tuple[str, ...] = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
)

NUM_CLASSES = len(EMOTIONS)

EMOTION_INDEX: dict[str, int] = {name: i for i, name in enumerate(EMOTIONS)}

# Lower-cased folder name -> canonical label, for ImageFolder-style datasets.
FOLDER_TO_LABEL: dict[str, str] = {name.lower(): name for name in EMOTIONS}


def label_for_index(index: int) -> str:
    """Return the canonical emotion name for a model output index."""
    try:
        position = int(index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"emotion index {index!r} is not an integer") from exc

    if not 0 <= position < NUM_CLASSES:
        raise ValueError(f"emotion index {index!r} is out of range for {NUM_CLASSES} classes")
    return EMOTIONS[position]


def index_for_label(label: str) -> int:
    """Return the model output index for an emotion name or folder name."""
    key = str(label).strip().lower()
    if key not in FOLDER_TO_LABEL:
        raise ValueError(
            f"unknown emotion label {label!r}; expected one of {', '.join(EMOTIONS)}"
        )
    return EMOTION_INDEX[FOLDER_TO_LABEL[key]]
