#!/usr/bin/env python3
"""Flask web demo for the emotion recognition model.

Quick start::

    pip install -r requirements-web.txt
    python webapp/app.py                 # http://localhost:8000

The server binds to ``0.0.0.0`` so it also works from containers and remote
preview environments. Frames are processed in memory and never written to disk.

API
---
``GET  /api/health``    service + model information
``GET  /api/classes``   supported emotion labels
``POST /api/analyze``   image in, predictions out

``/api/analyze`` accepts either a multipart field named ``image`` or JSON
``{"image": "data:image/jpeg;base64,..."}``. Optional fields: ``detect``
(boolean, default true), ``max_faces`` (1-10, default 5), ``annotate``
(boolean, default false; adds a base64 JPEG preview).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import logging
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from emotion_recognition import __version__  # noqa: E402
from emotion_recognition.detection import FaceDetector  # noqa: E402
from emotion_recognition.labels import EMOTIONS  # noqa: E402
from emotion_recognition.model import (  # noqa: E402
    EmotionRecognizer,
    decode_image,
    default_model_path,
)
from webapp.render import annotate_frame, encode_jpeg  # noqa: E402

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB per frame
MAX_FACES_LIMIT = 10
STATIC_DIR = Path(__file__).resolve().parent / "static"

log = logging.getLogger("emotion-webapp")


def _truthy(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if value is None or value == "":
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected an integer, got {value!r}") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"value must be between {minimum} and {maximum}")
    return number


def _decode_base64_image(raw: str) -> bytes:
    payload = raw.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    payload = "".join(payload.split())
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image is not valid base64 data") from exc


def _extract_image() -> bytes:
    """Pull image bytes out of a multipart upload or a JSON body."""
    if "image" in request.files:
        data = request.files["image"].read()
    else:
        payload = request.get_json(silent=True) or {}
        raw = payload.get("image")
        if not isinstance(raw, str) or not raw:
            raise ValueError("no image provided; send a multipart 'image' field or JSON {'image': <base64>}")
        data = _decode_base64_image(raw)

    if not data:
        raise ValueError("image is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    return data


def _request_options() -> tuple[bool, int, bool]:
    source = request.form if request.form else (request.get_json(silent=True) or {})
    detect = _truthy(source.get("detect"), default=True)
    max_faces = _parse_int(source.get("max_faces"), default=5, minimum=1, maximum=MAX_FACES_LIMIT)
    annotate = _truthy(source.get("annotate"), default=False)
    return detect, max_faces, annotate


def create_app(
    recognizer: EmotionRecognizer | None = None,
    *,
    static_folder: str | os.PathLike[str] | None = None,
) -> Flask:
    """Application factory.

    Args:
        recognizer: pre-built recognizer (used by tests to avoid loading Keras).
        static_folder: override for the static asset directory.
    """
    app = Flask(
        __name__,
        static_folder=str(static_folder or STATIC_DIR),
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["JSON_SORT_KEYS"] = False
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.extensions["emotion_recognizer"] = recognizer
    app.extensions["emotion_detector"] = None
    app.extensions["started_at"] = time.time()
    app.extensions["recognizer_lock"] = threading.Lock()

    def get_recognizer() -> EmotionRecognizer:
        existing = app.extensions.get("emotion_recognizer")
        if existing is not None:
            return existing
        with app.extensions["recognizer_lock"]:
            if app.extensions["emotion_recognizer"] is None:
                log.info("loading model from %s", default_model_path())
                app.extensions["emotion_recognizer"] = EmotionRecognizer()
            return app.extensions["emotion_recognizer"]

    def get_detector() -> FaceDetector | None:
        detector = app.extensions.get("emotion_detector")
        if detector is None:
            try:
                detector = FaceDetector()
            except (FileNotFoundError, ValueError) as exc:
                log.warning("face detection unavailable: %s", exc)
                return None
            app.extensions["emotion_detector"] = detector
        return detector

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/favicon.svg")
    def favicon():
        return send_from_directory(app.static_folder, "favicon.svg")

    @app.post("/api/warmup")
    def warmup():
        """Load the model ahead of the first frame (idempotent)."""
        started = time.perf_counter()
        try:
            recognizer_instance = get_recognizer()
        except (FileNotFoundError, RuntimeError) as exc:
            log.exception("model load failed")
            return jsonify({"error": f"model unavailable: {exc}"}), 503
        return jsonify(
            {
                "status": "ready",
                "model_loaded": True,
                "model": str(default_model_path()),
                "classes": list(getattr(recognizer_instance, "labels", EMOTIONS)),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            }
        )

    @app.get("/api/health")
    def health():
        loaded = app.extensions.get("emotion_recognizer") is not None
        return jsonify(
            {
                "status": "ok",
                "version": __version__,
                "model": str(default_model_path()),
                "model_loaded": loaded,
                "classes": list(EMOTIONS),
                "detection_available": get_detector() is not None,
                "uptime_seconds": round(time.time() - app.extensions["started_at"], 1),
            }
        )

    @app.get("/api/classes")
    def classes():
        return jsonify({"classes": list(EMOTIONS)})

    @app.post("/api/analyze")
    def analyze():
        started = time.perf_counter()
        try:
            data = _extract_image()
            detect, max_faces, annotate = _request_options()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            image = decode_image(data)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        detector = get_detector() if detect else None
        if detect and detector is None:
            return (
                jsonify(
                    {
                        "error": "face detection is unavailable on this server "
                        "(OpenCV cascade data not found); retry with detect=false"
                    }
                ),
                503,
            )

        try:
            model = get_recognizer()
        except (FileNotFoundError, RuntimeError) as exc:
            log.exception("model load failed")
            return jsonify({"error": f"model unavailable: {exc}"}), 503

        analysis = model.analyze_frame(
            image,
            detector=detector,
            max_faces=max_faces,
            fallback_to_frame=not detect,
        )

        payload = analysis.as_dict()
        payload["model"] = str(default_model_path())
        payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 2)

        if annotate:
            preview = annotate_frame(image, analysis, scale=1)
            payload["annotated_image"] = (
                "data:image/jpeg;base64,"
                + base64.b64encode(encode_jpeg(preview)).decode("ascii")
            )

        return jsonify(payload)

    @app.errorhandler(413)
    def too_large(_error):
        return (
            jsonify({"error": f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}),
            413,
        )

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "endpoint not found"}), 404
        return send_from_directory(app.static_folder, "index.html"), 404

    @app.errorhandler(500)
    def server_error(error):  # pragma: no cover - defensive
        log.exception("unhandled error", exc_info=error)
        return jsonify({"error": "internal server error"}), 500

    @app.after_request
    def no_cache_api(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the emotion recognition web demo.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--debug", action="store_true", help="enable Flask debug reloader")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    app = create_app()
    print(f"Emotion Recognition web demo on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
