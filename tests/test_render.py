import numpy as np
import pytest

from emotion_recognition.labels import EMOTIONS
from emotion_recognition.model import FacePrediction, FrameAnalysis
from webapp.render import EMOTION_COLORS, annotate_frame, encode_jpeg


def make_analysis(box=(4, 4, 20, 20), emotion="Happy") -> FrameAnalysis:
    scores = {name: 0.0 for name in EMOTIONS}
    scores[emotion] = 0.9
    face = FacePrediction(box=box, emotion=emotion, confidence=0.9, scores=scores)
    return FrameAnalysis(width=48, height=48, faces=[face], fallback=False, elapsed_ms=1.0)


def test_palette_covers_all_emotions():
    assert set(EMOTION_COLORS) == set(EMOTIONS)


def test_annotate_frame_keeps_shape_and_does_not_mutate():
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    original = image.copy()
    annotated = annotate_frame(image, make_analysis(), scale=1)
    assert annotated.shape == image.shape
    assert np.array_equal(image, original)
    assert not np.array_equal(annotated, original)


def test_annotate_frame_upscales_small_images():
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    annotated = annotate_frame(image, make_analysis(), scale=3)
    assert annotated.shape == (144, 144, 3)


def test_annotate_frame_handles_no_faces():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    annotated = annotate_frame(image, FrameAnalysis(width=20, height=20), scale=1)
    assert np.array_equal(annotated, image)


def test_annotate_frame_rejects_bad_scale():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        annotate_frame(image, FrameAnalysis(width=20, height=20), scale=0)


def test_encode_jpeg_produces_jpeg_bytes():
    data = encode_jpeg(np.zeros((16, 16, 3), dtype=np.uint8))
    assert data[:2] == b"\xff\xd8"
    assert data[-2:] == b"\xff\xd9"
