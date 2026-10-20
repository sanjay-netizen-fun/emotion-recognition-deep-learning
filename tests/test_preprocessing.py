import numpy as np
import pytest

from emotion_recognition.preprocessing import (
    INPUT_SIZE,
    preprocess_face,
    preprocess_faces,
    to_gray,
)


def test_input_size_matches_training():
    assert INPUT_SIZE == (48, 48)


def test_to_gray_handles_grayscale_and_colour(bgr_image):
    assert to_gray(bgr_image).shape == (80, 64)

    bgra = np.dstack([bgr_image, np.full((80, 64), 255, dtype=np.uint8)])
    assert to_gray(bgra).shape == (80, 64)

    single_channel = bgr_image[:, :, :1]
    assert to_gray(single_channel).shape == (80, 64)

    assert to_gray(bgr_image.astype(np.float32)).dtype == np.float32


def test_preprocess_face_shape_dtype_and_range(gray_face):
    processed = preprocess_face(gray_face)
    assert processed.shape == (48, 48, 1)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min() and processed.max() <= 1.0


def test_preprocess_face_scales_like_training():
    image = np.arange(48 * 48, dtype=np.uint16).reshape(48, 48) % 256
    image = image.astype(np.uint8)
    processed = preprocess_face(image)
    expected = image.astype(np.float32) / 255.0
    assert np.allclose(processed[..., 0], expected, atol=1e-6)


def test_preprocess_face_is_idempotent(gray_face):
    once = preprocess_face(gray_face)
    twice = preprocess_face(once)
    assert np.allclose(once, twice)


def test_preprocess_face_does_not_darken_normalized_input(gray_face):
    once = preprocess_face(gray_face)
    assert preprocess_face(once).mean() == pytest.approx(once.mean())


def test_preprocess_face_accepts_pil_images(gray_face):
    from PIL import Image

    result = preprocess_face(Image.fromarray(gray_face, mode="L"))
    assert result.shape == (48, 48, 1)


def test_preprocess_faces_empty_batch_has_fixed_shape():
    batch = preprocess_faces([])
    assert batch.shape == (0, 48, 48, 1)
    assert batch.dtype == np.float32


def test_preprocess_faces_batches(gray_face):
    batch = preprocess_faces([gray_face, gray_face])
    assert batch.shape == (2, 48, 48, 1)


@pytest.mark.parametrize("bad", [None, np.array([]), np.zeros((4, 4, 2), dtype=np.uint8)])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        preprocess_face(bad)


def test_invalid_target_size_raises(gray_face):
    with pytest.raises(ValueError):
        preprocess_face(gray_face, size=(0, 48))
