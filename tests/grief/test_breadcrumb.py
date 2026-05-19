"""test_breadcrumb.py — intensity formulas + content phrases + write path."""

from __future__ import annotations

import pytest

from brain.grief import breadcrumb


def test_compute_drop_intensity_high_emotion_high_salience() -> None:
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=0.9, salience_at_drop=0.7)
    assert result == pytest.approx(4.41, abs=0.01)


def test_compute_drop_intensity_low_inputs_under_floor() -> None:
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=0.2, salience_at_drop=0.3)
    assert result == pytest.approx(0.42, abs=0.01)


def test_compute_drop_intensity_clamped_at_10() -> None:
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=2.0, salience_at_drop=2.0)
    assert result == 10.0


def test_compute_drop_intensity_clamped_at_zero() -> None:
    result = breadcrumb.compute_drop_intensity(emotion_at_ingest_max=-0.5, salience_at_drop=0.7)
    assert result == 0.0


def test_compute_arc_close_intensity_heavy_member() -> None:
    result = breadcrumb.compute_arc_close_intensity(arc_max_member_emotion=0.8)
    assert result == pytest.approx(5.6, abs=0.01)


def test_compute_arc_close_intensity_under_floor() -> None:
    result = breadcrumb.compute_arc_close_intensity(arc_max_member_emotion=0.2)
    assert result == pytest.approx(1.4, abs=0.01)
