"""test_recall.py — recall-touch intensity + handle_recall_touch."""

from __future__ import annotations

import pytest

from brain.grief import recall


def test_touch_intensity_fresh_loss() -> None:
    result = recall.compute_touch_intensity(
        grave_emotion_max=0.8,
        salience_at_drop=0.5,
        lived_days_since_loss=0.0,
    )
    assert result == pytest.approx(2.0, abs=0.01)


def test_touch_intensity_recency_decay() -> None:
    fresh = recall.compute_touch_intensity(
        grave_emotion_max=1.0, salience_at_drop=1.0, lived_days_since_loss=0.0
    )
    aged = recall.compute_touch_intensity(
        grave_emotion_max=1.0, salience_at_drop=1.0, lived_days_since_loss=14.0
    )
    assert aged == pytest.approx(fresh * 0.5, abs=0.01)


def test_touch_intensity_clamped_at_10() -> None:
    result = recall.compute_touch_intensity(
        grave_emotion_max=3.0, salience_at_drop=3.0, lived_days_since_loss=0.0
    )
    assert result == 10.0


def test_touch_intensity_old_loss_low() -> None:
    result = recall.compute_touch_intensity(
        grave_emotion_max=0.9, salience_at_drop=0.9, lived_days_since_loss=60.0
    )
    assert result < 0.1
