"""Annotation helpers for the web demo.

Draws detected faces and their predicted emotions onto a copy of the frame so
API consumers can get a ready-to-display image (``annotate=true``).
"""

from __future__ import annotations

import cv2
import numpy as np

from emotion_recognition.model import FrameAnalysis

# BGR equivalents of the web palette (#ef4444, #84cc16, #a855f7, #fbbf24,
# #94a3b8, #3b82f6, #f97316).
EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "Angry": (68, 68, 239),
    "Disgust": (22, 204, 132),
    "Fear": (247, 85, 168),
    "Happy": (36, 191, 251),
    "Neutral": (184, 163, 148),
    "Sad": (246, 130, 59),
    "Surprise": (22, 115, 249),
}


def annotate_frame(image: np.ndarray, analysis: FrameAnalysis, *, scale: int = 3) -> np.ndarray:
    """Return a copy of ``image`` with boxes, labels and a confidence bar.

    Args:
        image: BGR frame the analysis was computed on.
        analysis: result of :meth:`EmotionRecognizer.analyze_frame`.
        scale: upsample factor for small frames (48x48 dataset images), applied
            before drawing so text stays legible. ``1`` keeps the original size.
    """
    if scale < 1:
        raise ValueError("scale must be at least 1")

    canvas = image.copy()
    if scale > 1:
        canvas = cv2.resize(
            canvas,
            (canvas.shape[1] * scale, canvas.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    for face in analysis.faces:
        x, y, w, h = (int(value * scale) for value in face.box)
        color = EMOTION_COLORS.get(face.emotion, (255, 255, 255))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        label = f"{face.emotion} {face.percent:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(0.8, w / 220.0))
        thickness = 1 if w < 160 else 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Label above the box when possible, otherwise inside its top edge.
        box_top = y - text_height - baseline - 8
        if box_top < 0:
            box_top = y
        box_right = min(canvas.shape[1], x + text_width + 12)
        cv2.rectangle(canvas, (x, box_top), (box_right, box_top + text_height + baseline + 8), color, -1)
        cv2.putText(
            canvas,
            label,
            (x + 6, box_top + text_height + 3),
            font,
            font_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )

    return canvas


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR image as JPEG bytes."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("could not encode image as JPEG")
    return buffer.tobytes()
