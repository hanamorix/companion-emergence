"""test_prompt.py — render_grief_block + helpers."""

from __future__ import annotations

from brain.grief import prompt


def test_weight_bucket_heavy() -> None:
    assert prompt.weight_bucket(emotion_max_normalised=0.85) == "heavy"


def test_weight_bucket_medium() -> None:
    assert prompt.weight_bucket(emotion_max_normalised=0.5) == "medium"


def test_weight_bucket_light() -> None:
    assert prompt.weight_bucket(emotion_max_normalised=0.2) == "light"


def test_weight_bucket_heavy_boundary() -> None:
    # exactly at WEIGHT_HEAVY normalised threshold (7.0 -> 0.7) -> heavy
    assert prompt.weight_bucket(emotion_max_normalised=0.7) == "heavy"


def test_weight_bucket_medium_boundary() -> None:
    # exactly at WEIGHT_MEDIUM normalised threshold (3.0 -> 0.3) -> medium
    assert prompt.weight_bucket(emotion_max_normalised=0.3) == "medium"
