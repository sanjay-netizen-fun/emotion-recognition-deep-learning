"""End-to-end checks against the real trained model and dataset.

These tests are marked ``slow`` (they load TensorFlow). The dataset folder is
git-ignored, so they skip automatically on machines without it::

    pytest -m "not slow"      # fast unit tests only
    pytest -m slow            # model/dataset integration tests
"""

import base64
import io
from pathlib import Path

import pytest

pytest.importorskip("keras")

from emotion_recognition.labels import EMOTIONS  # noqa: E402
from emotion_recognition.model import (  # noqa: E402
    EmotionRecognizer,
    default_model_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = REPO_ROOT / "images" / "validation"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not default_model_path().is_file(), reason="trained model not available"),
    pytest.mark.skipif(not VALIDATION_DIR.is_dir(), reason="validation dataset not available"),
]


@pytest.fixture(scope="module")
def recognizer():
    return EmotionRecognizer(default_model_path())


def test_model_predicts_expected_emotion(recognizer):
    import cv2

    image = cv2.imread(str(VALIDATION_DIR / "Happy" / "0.jpg"), cv2.IMREAD_GRAYSCALE)
    analysis = recognizer.analyze_frame(image)

    assert analysis.count == 1
    assert analysis.faces[0].emotion == "Happy"
    assert analysis.faces[0].confidence > 0.5


def test_model_accuracy_on_small_sample(recognizer):
    import cv2

    correct = 0
    total = 0
    for emotion in EMOTIONS:
        folder = VALIDATION_DIR / emotion
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.jpg"))[:4]
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            prediction = recognizer.analyze_frame(image).faces[0].emotion
            correct += prediction == emotion
            total += 1

    assert total >= 20
    assert correct / total >= 0.5


def test_web_api_end_to_end():
    flask = pytest.importorskip("flask")
    assert flask is not None
    import cv2

    from tests.conftest import FakeModel  # noqa: F401 - ensures conftest import works
    from webapp.app import create_app

    image = cv2.imread(str(VALIDATION_DIR / "Happy" / "0.jpg"), cv2.IMREAD_GRAYSCALE)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok

    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as client:
        assert client.get("/api/health").status_code == 200
        assert client.post("/api/warmup").status_code == 200

        response = client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(buffer.tobytes()), "face.jpg"), "detect": "0", "annotate": "1"},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["faces"][0]["emotion"] == "Happy"
    assert payload["annotated_image"].startswith("data:image/jpeg;base64,")

    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    assert encoded
