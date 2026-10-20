"""Dataset discovery for evaluation.

Supports the layout used by ``trainmodel.ipynb``::

    images/
      train/<emotion>/*.jpg
      test/<emotion>/*.jpg

as well as a flat, single-split layout::

    some_folder/<emotion>/*.jpg

Class folders are matched case-insensitively, so ``angry`` and ``Angry`` both
map to the canonical ``Angry`` label.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .labels import FOLDER_TO_LABEL, index_for_label

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pgm", ".tif", ".tiff"}
)
SPLIT_NAMES = ("train", "test")


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    label: str
    index: int


def _class_directories(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        child
        for child in directory.iterdir()
        if child.is_dir() and child.name.lower() in FOLDER_TO_LABEL
    )


def available_splits(root: str | Path) -> list[str]:
    """Return the split names found under ``root`` (empty for flat layouts)."""
    directory = Path(root)
    return [name for name in SPLIT_NAMES if _class_directories(directory / name)]


def resolve_split_dir(root: str | Path, split: str | None = None) -> Path:
    """Resolve ``root`` to a directory containing class sub-folders.

    Args:
        root: dataset root, or a directory with ``train``/``test`` children.
        split: ``"train"`` or ``"test"`` when ``root`` uses the split layout.
    """
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {directory}")

    splits = available_splits(directory)
    if splits:
        if split is None:
            if len(splits) == 1:
                return directory / splits[0]
            raise ValueError(
                f"{directory} contains splits {splits}; pass --split "
                f"{'|'.join(splits)} to choose one"
            )
        if split not in splits:
            raise ValueError(f"split {split!r} not found under {directory} (found {splits})")
        return directory / split

    if _class_directories(directory):
        if split is not None:
            raise ValueError(
                f"{directory} is a single split; --split {split} is not applicable"
            )
        return directory

    expected = ", ".join(sorted(FOLDER_TO_LABEL.values()))
    raise ValueError(
        f"no recognized emotion folders in {directory}; expected sub-folders named "
        f"one of: {expected}"
    )


def scan_dataset(
    root: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[DatasetSample]:
    """Collect image samples and their labels.

    Args:
        root: dataset root (split layout or flat layout).
        split: split to use when ``root`` has ``train``/``test`` children.
        limit: keep at most this many samples (sampled per class, round-robin).
        shuffle: shuffle samples deterministically using ``seed``.
        seed: random seed for shuffling and balanced limiting.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")

    directory = resolve_split_dir(root, split)
    rng = random.Random(seed)

    per_class: list[list[DatasetSample]] = []
    for class_dir in _class_directories(directory):
        label = FOLDER_TO_LABEL[class_dir.name.lower()]
        index = index_for_label(label)
        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if files:
            per_class.append([DatasetSample(path=path, label=label, index=index) for path in files])

    if not per_class:
        raise ValueError(f"no images found under {directory}")

    if limit is not None:
        per_class = _balanced_limit(per_class, limit, rng)

    samples = [sample for group in per_class for sample in group]
    if shuffle:
        rng.shuffle(samples)
    else:
        samples.sort(key=lambda sample: (sample.index, str(sample.path)))
    return samples


def _balanced_limit(
    per_class: Iterable[list[DatasetSample]],
    limit: int,
    rng: random.Random,
) -> list[list[DatasetSample]]:
    """Take up to ``limit`` samples, spread as evenly as possible across classes."""
    groups = []
    for group in per_class:
        group = list(group)
        rng.shuffle(group)
        groups.append(group)

    counts = {index: 0 for index in range(len(groups))}
    selected: list[list[DatasetSample]] = [[] for _ in groups]
    remaining = limit

    while remaining > 0:
        progressed = False
        for index, group in enumerate(groups):
            if remaining == 0:
                break
            if counts[index] < len(group):
                selected[index].append(group[counts[index]])
                counts[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return [group for group in selected if group]
