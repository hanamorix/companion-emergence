"""ExtractedItem.normalize: emotion-strength cleanup at the ingest boundary."""

from __future__ import annotations

import logging
import math

import pytest

from brain.ingest.types import ExtractedItem
from brain.memory.store import Memory

_NON_FINITE = [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity"]


@pytest.mark.parametrize("raw", _NON_FINITE, ids=repr)
def test_normalize_drops_non_finite_emotion_strength(raw, caplog) -> None:
    """#466: NaN answers False to `<= 0.0`, so it got past the drop check, and +inf
    was clamped to 10.0 as if it were a real strength. Both are now rejected and logged."""
    item = ExtractedItem(text="x", emotions={"joy": raw, "love": 4.0})
    with caplog.at_level(logging.WARNING, logger="brain.ingest.types"):
        item.normalize()

    assert item.emotions == {"love": 4.0}
    assert "joy" in caplog.text


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf")], ids=repr)
def test_create_new_drops_non_finite_emotion_and_keeps_score_finite(raw, caplog) -> None:
    """#466: create_new sums emotions into the score, so one non-finite strength made
    the whole memory's score (and importance) non-finite."""
    with caplog.at_level(logging.WARNING, logger="brain.memory.store"):
        mem = Memory.create_new(
            content="x", memory_type="conversation", domain="us",
            emotions={"joy": raw, "love": 4.0},
        )

    assert mem.emotions == {"love": 4.0}
    assert mem.score == 4.0
    assert math.isfinite(mem.importance)
    assert math.isfinite(mem.peak_emotion_intensity)
    assert "joy" in caplog.text and "conversation" in caplog.text
