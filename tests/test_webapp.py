import base64
import io

import numpy as np
import pytest

pytest.importorskip("flask")

from emotion_recognition.model import EmotionRecognizer  # noqa: E402
from tests.conftest import FakeModel  # noqa: E402
from webapp.app import create_app  # noqa: E402


def jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    import cv2

    image = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def client():
    recognizer = EmotionRecognizer(model=FakeModel(), warmup=False)
    app = create_app(recognizer=recognizer)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_index_serves_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Emotion" in response.data
    assert b"/static/app.js" in response.data


def test_static_assets(client):
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/favicon.svg").status_code == 200


def test_classes(client):
    response = client.get("/api/classes")
    assert response.status_code == 200
    assert response.get_json()["classes"] == [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprise",
    ]


def test_health(client):
    payload = client.get("/api/health").get_json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["classes"][3] == "Happy"
    assert isinstance(payload["detection_available"], bool)
    assert payload["uptime_seconds"] >= 0


def test_warmup(client):
    payload = client.post("/api/warmup").get_json()
    assert payload["status"] == "ready"
    assert payload["model_loaded"] is True
    assert payload["classes"][0] == "Angry"


def test_analyze_multipart(client):
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(jpeg_bytes()), "face.jpg"), "detect": "0"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["faces"][0]["emotion"] == "Happy"
    assert payload["faces"][0]["percent"] == pytest.approx(60.0)
    assert payload["fallback"] is True
    assert payload["width"] == 64 and payload["height"] == 64


def test_analyze_base64_json(client):
    encoded = base64.b64encode(jpeg_bytes(48, 48)).decode("ascii")
    response = client.post(
        "/api/analyze",
        json={"image": f"data:image/jpeg;base64,{encoded}", "detect": False},
    )
    assert response.status_code == 200
    assert response.get_json()["count"] == 1


def test_analyze_annotate(client):
    response = client.post(
        "/api/analyze",
        data={
            "image": (io.BytesIO(jpeg_bytes()), "face.jpg"),
            "detect": "0",
            "annotate": "1",
        },
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    assert payload["annotated_image"].startswith("data:image/jpeg;base64,")


def test_analyze_rejects_missing_image(client):
    response = client.post("/api/analyze", json={})
    assert response.status_code == 400
    assert "no image" in response.get_json()["error"]


def test_analyze_rejects_bad_base64(client):
    response = client.post("/api/analyze", json={"image": "data:image/jpeg;base64,@@@@"})
    assert response.status_code == 400
    assert "base64" in response.get_json()["error"]


def test_analyze_rejects_undecodable_bytes(client):
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(b"not an image"), "x.jpg"), "detect": "0"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_analyze_validates_max_faces(client):
    response = client.post(
        "/api/analyze",
        data={
            "image": (io.BytesIO(jpeg_bytes()), "face.jpg"),
            "detect": "0",
            "max_faces": "99",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "between" in response.get_json()["error"]


def test_analyze_with_detection(client):
    health = client.get("/api/health").get_json()
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(jpeg_bytes()), "face.jpg"), "detect": "1"},
        content_type="multipart/form-data",
    )

    if not health["detection_available"]:
        assert response.status_code == 503
        assert "detection is unavailable" in response.get_json()["error"]
        return

    assert response.status_code == 200
    payload = response.get_json()
    # A flat gray image contains no face, and fallback is disabled while detection is on.
    assert payload["count"] == 0
    assert payload["fallback"] is False


def test_unknown_api_route_returns_json(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.get_json()["error"] == "endpoint not found"


def test_truthy_parsing():
    from webapp.app import _parse_int, _truthy

    assert _truthy(None) is True
    assert _truthy("false") is False
    assert _truthy("0") is False
    assert _truthy(False) is False
    assert _parse_int(None, 5, 1, 10) == 5
    with pytest.raises(ValueError):
        _parse_int("abc", 5, 1, 10)
