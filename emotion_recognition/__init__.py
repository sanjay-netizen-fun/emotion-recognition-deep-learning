"""Shared core for the Emotion Recognition project.

The package is intentionally import-light: TensorFlow/Keras is only imported
when a model is actually loaded, so utilities such as ``emotion_recognition.metrics``
can be used without the deep learning stack installed.
"""

from .detection import FaceDetector, cascade_path
from .labels import EMOTION_INDEX, EMOTIONS, index_for_label, label_for_index
from .model import (
    DEFAULT_MODEL_FILENAME,
    EmotionRecognizer,
    FacePrediction,
    FrameAnalysis,
    default_model_path,
    load_keras_model,
)
from .preprocessing import INPUT_SIZE, preprocess_face, preprocess_faces, to_gray

__all__ = [
    "EMOTIONS",
    "EMOTION_INDEX",
    "label_for_index",
    "index_for_label",
    "DEFAULT_MODEL_FILENAME",
    "EmotionRecognizer",
    "FacePrediction",
    "FrameAnalysis",
    "default_model_path",
    "load_keras_model",
    "INPUT_SIZE",
    "preprocess_face",
    "preprocess_faces",
    "to_gray",
    "FaceDetector",
    "cascade_path",
]

__version__ = "1.1.0"
