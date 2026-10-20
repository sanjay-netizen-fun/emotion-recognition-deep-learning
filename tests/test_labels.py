import pytest

from emotion_recognition.labels import (
    EMOTION_INDEX,
    EMOTIONS,
    FOLDER_TO_LABEL,
    NUM_CLASSES,
    index_for_label,
    label_for_index,
)


def test_class_order_matches_label_encoder():
    # LabelEncoder sorts alphabetically; the trained model relies on that order.
    assert EMOTIONS == tuple(sorted(EMOTIONS, key=str.lower))
    assert NUM_CLASSES == 7
    assert list(EMOTION_INDEX.values()) == list(range(NUM_CLASSES))


def test_label_for_index_roundtrip():
    for index, name in enumerate(EMOTIONS):
        assert label_for_index(index) == name
        assert index_for_label(name) == index


def test_index_for_label_is_case_insensitive():
    assert index_for_label("happy") == EMOTION_INDEX["Happy"]
    assert index_for_label("  ANGRY ") == EMOTION_INDEX["Angry"]
    assert FOLDER_TO_LABEL["neutral"] == "Neutral"


@pytest.mark.parametrize("index", [-1, 7, 99])
def test_label_for_index_rejects_out_of_range(index):
    with pytest.raises(ValueError):
        label_for_index(index)


def test_index_for_label_rejects_unknown():
    with pytest.raises(ValueError, match="unknown emotion label"):
        index_for_label("confused")
