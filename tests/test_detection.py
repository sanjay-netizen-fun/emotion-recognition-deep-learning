import cv2
import numpy as np
import pytest

from emotion_recognition.detection import (
    CASCADE_FILENAME,
    FaceDetector,
    cascade_path,
    crop_box,
)


def _cascade_available() -> bool:
    try:
        cascade_path()
    except FileNotFoundError:
        return False
    return True


needs_cascade = pytest.mark.skipif(
    not _cascade_available(), reason="OpenCV cascade data not available"
)


@needs_cascade
def test_cascade_path_exists():
    path = cascade_path()
    assert path.name == CASCADE_FILENAME
    assert path.is_file()


def test_missing_cascade_raises_readable_error():
    with pytest.raises(FileNotFoundError, match="could not find"):
        cascade_path("does_not_exist.xml")


def test_detector_validates_parameters():
    if not _cascade_available():
        pytest.skip("OpenCV cascade data not available")
    with pytest.raises(ValueError):
        FaceDetector(scale_factor=1.0)
    with pytest.raises(ValueError):
        FaceDetector(min_neighbors=-1)
    with pytest.raises(ValueError):
        FaceDetector("missing.xml")


@needs_cascade
def test_detector_returns_empty_boxes_for_blank_image():
    detector = FaceDetector()
    boxes = detector.detect(np.zeros((240, 320), dtype=np.uint8))
    assert boxes.shape == (0, 4)


@needs_cascade
def test_detector_accepts_colour_images():
    detector = FaceDetector()
    boxes = detector.detect(np.zeros((120, 160, 3), dtype=np.uint8))
    assert boxes.shape == (0, 4)


@needs_cascade
def test_detection_is_deterministic_on_real_face():
    # Use a synthetic "face-like" pattern: the cascade may or may not fire, but
    # two runs must agree and the shape must be valid either way.
    image = np.full((300, 300), 160, dtype=np.uint8)
    cv2.circle(image, (150, 150), 70, 200, -1)
    cv2.circle(image, (125, 130), 8, 40, -1)
    cv2.circle(image, (175, 130), 8, 40, -1)
    cv2.ellipse(image, (150, 175), (30, 15), 0, 0, 180, 40, 3)
    first = FaceDetector().detect(image)
    second = FaceDetector().detect(image)
    assert first.shape[1:] == (4,)
    assert np.array_equal(first, second)


def test_crop_box_clips_to_bounds():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert crop_box(image, (-5, -5, 8, 8)).shape == (3, 3)
    assert crop_box(image, (8, 8, 10, 10)).shape == (2, 2)
    assert crop_box(image, (0, 0, 10, 10)).shape == (10, 10)


def test_crop_box_never_returns_empty():
    image = np.zeros((5, 5), dtype=np.uint8)
    assert crop_box(image, (20, 20, 4, 4)).size >= 1
