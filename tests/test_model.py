import numpy as np
import pytest

from emotion_recognition.labels import EMOTIONS
from emotion_recognition.model import (
    EmotionRecognizer,
    FrameAnalysis,
    decode_image,
    default_model_path,
    load_keras_model,
)
from tests.conftest import FakeModel, StubDetector


def make_recognizer(probabilities=None) -> EmotionRecognizer:
    return EmotionRecognizer(model=FakeModel(probabilities), warmup=False)


def test_predict_face_returns_named_scores(gray_face):
    recognizer = make_recognizer([0.05, 0.05, 0.05, 0.6, 0.1, 0.1, 0.05])
    prediction = recognizer.predict_face(gray_face)

    assert prediction.emotion == "Happy"
    assert prediction.confidence == pytest.approx(0.6)
    assert prediction.percent == pytest.approx(60.0)
    assert set(prediction.scores) == set(EMOTIONS)
    assert sum(prediction.scores.values()) == pytest.approx(1.0)


def test_predict_batch_empty_batch():
    recognizer = make_recognizer()
    result = recognizer.predict_batch([])
    assert result.shape == (0, len(EMOTIONS))


def test_analyze_frame_uses_detected_faces(gray_face):
    recognizer = make_recognizer()
    detector = StubDetector([[0, 0, 20, 20], [30, 30, 20, 20]])
    analysis = recognizer.analyze_frame(gray_face, detector=detector)

    assert isinstance(analysis, FrameAnalysis)
    assert analysis.count == 2
    assert analysis.fallback is False
    assert [face.box for face in analysis.faces] == [(0, 0, 20, 20), (30, 30, 20, 20)]
    assert analysis.width == 64 and analysis.height == 64


def test_analyze_frame_limits_to_largest_faces(gray_face):
    recognizer = make_recognizer()
    detector = StubDetector([[0, 0, 10, 10], [0, 0, 40, 40], [0, 0, 20, 20]])
    analysis = recognizer.analyze_frame(gray_face, detector=detector, max_faces=2)

    areas = sorted(face.box[2] * face.box[3] for face in analysis.faces)
    assert areas == [400, 1600]


def test_analyze_frame_falls_back_to_whole_frame(gray_face):
    recognizer = make_recognizer()
    analysis = recognizer.analyze_frame(gray_face, detector=StubDetector(), fallback_to_frame=True)

    assert analysis.count == 1
    assert analysis.fallback is True
    assert analysis.faces[0].box == (0, 0, 64, 64)


def test_analyze_frame_can_refuse_fallback(gray_face):
    recognizer = make_recognizer()
    analysis = recognizer.analyze_frame(gray_face, detector=StubDetector(), fallback_to_frame=False)

    assert analysis.count == 0
    assert analysis.fallback is False
    assert analysis.as_dict()["faces"] == []


def test_analyze_frame_without_detector_uses_full_frame(gray_face):
    recognizer = make_recognizer()
    analysis = recognizer.analyze_frame(gray_face, detector=None)
    assert analysis.count == 1
    assert analysis.fallback is True


def test_analysis_serializes(gray_face):
    recognizer = make_recognizer()
    payload = recognizer.analyze_frame(gray_face).as_dict()

    assert payload["count"] == 1
    assert payload["width"] == 64
    assert isinstance(payload["elapsed_ms"], float)
    face = payload["faces"][0]
    assert face["emotion"] == "Happy"
    assert len(face["scores"]) == 7
    assert 0.0 <= face["confidence"] <= 1.0


def test_empty_image_raises():
    recognizer = make_recognizer()
    with pytest.raises(ValueError):
        recognizer.analyze_frame(np.array([]))


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="model file not found"):
        load_keras_model(str(tmp_path / "nope.keras"))


def test_default_model_path_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom.keras"
    monkeypatch.setenv("EMOTION_MODEL_PATH", str(target))
    assert default_model_path() == target.resolve()


def test_default_model_path_points_at_repo_model():
    path = default_model_path()
    assert path.name == "emotiondetector.keras"


def test_decode_image_roundtrip(tmp_path):
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8), mode="RGB").save(path)

    decoded = decode_image(path.read_bytes())
    assert decoded.shape == (32, 48, 3)


def test_decode_image_rejects_garbage():
    with pytest.raises(ValueError):
        decode_image(b"definitely not an image")
