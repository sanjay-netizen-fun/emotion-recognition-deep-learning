from collections import Counter

import pytest

from emotion_recognition.dataset import (
    available_splits,
    resolve_split_dir,
    scan_dataset,
)
from emotion_recognition.labels import index_for_label


def test_available_splits(dataset_root):
    assert available_splits(dataset_root) == ["train", "test"]


def test_scan_train_split_case_insensitive(dataset_root):
    samples = scan_dataset(dataset_root, split="train")
    counts = Counter(sample.label for sample in samples)
    assert counts == {"Angry": 3, "Happy": 4, "Neutral": 2}
    assert all(sample.index == index_for_label(sample.label) for sample in samples)


def test_scan_requires_split_when_both_exist(dataset_root):
    with pytest.raises(ValueError, match="contains splits"):
        scan_dataset(dataset_root)


def test_scan_unknown_split(dataset_root):
    with pytest.raises(ValueError, match="not found"):
        scan_dataset(dataset_root, split="validation")


def test_scan_flat_layout(dataset_root):
    samples = scan_dataset(dataset_root / "test")
    assert Counter(sample.label for sample in samples) == {"Angry": 1, "Happy": 2}


def test_scan_ignores_non_images_and_unknown_folders(dataset_root):
    samples = scan_dataset(dataset_root, split="train")
    assert all(sample.path.suffix == ".png" for sample in samples)
    assert all("NotAnEmotion" not in str(sample.path) for sample in samples)


def test_limit_is_balanced_across_classes(dataset_root):
    samples = scan_dataset(dataset_root, split="train", limit=6)
    counts = Counter(sample.label for sample in samples)
    assert len(samples) == 6
    assert max(counts.values()) - min(counts.values()) <= 1


def test_limit_larger_than_dataset(dataset_root):
    samples = scan_dataset(dataset_root, split="train", limit=100)
    assert len(samples) == 9


def test_shuffle_is_deterministic(dataset_root):
    first = [sample.path for sample in scan_dataset(dataset_root, split="train", shuffle=True, seed=7)]
    second = [sample.path for sample in scan_dataset(dataset_root, split="train", shuffle=True, seed=7)]
    third = [sample.path for sample in scan_dataset(dataset_root, split="train", shuffle=True, seed=8)]
    assert first == second
    assert first != third


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_dataset(tmp_path / "nope")


def test_no_class_folders(tmp_path):
    (tmp_path / "random").mkdir()
    with pytest.raises(ValueError, match="no recognized emotion folders"):
        scan_dataset(tmp_path)


def test_invalid_limit(dataset_root):
    with pytest.raises(ValueError):
        scan_dataset(dataset_root, split="train", limit=0)


def test_split_on_flat_layout_is_rejected(dataset_root):
    with pytest.raises(ValueError, match="single split"):
        resolve_split_dir(dataset_root / "test", "train")


def test_validation_split_is_recognized(tmp_path):
    import numpy as np
    from PIL import Image

    folder = tmp_path / "images" / "validation" / "Happy"
    folder.mkdir(parents=True)
    Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L").save(folder / "0.png")

    assert available_splits(tmp_path / "images") == ["validation"]
    assert [sample.label for sample in scan_dataset(tmp_path / "images", split="validation")] == ["Happy"]
    # A single split can also be resolved without passing --split.
    assert [sample.label for sample in scan_dataset(tmp_path / "images")] == ["Happy"]

    sad = tmp_path / "images" / "test" / "Sad"
    sad.mkdir(parents=True)
    Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L").save(sad / "0.png")

    assert available_splits(tmp_path / "images") == ["validation", "test"]
    with pytest.raises(ValueError, match="contains splits"):
        scan_dataset(tmp_path / "images")
