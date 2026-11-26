"""Tests for brain.felt_time.weather_shift — the rolling baseline (the shift detector went in #785)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.felt_time.weather_shift import Baseline, update_baseline


def test_baseline_mean_sigma_from_seven_day_samples():
    # 168 hourly samples spanning 7 days, mean ~5.0, σ ~1.0
    start = datetime(2026, 5, 10, 0, 0, tzinfo=UTC)
    samples = [(start + timedelta(hours=h), 5.0 + (h % 3 - 1)) for h in range(168)]
    b = update_baseline(Baseline.empty(), samples)
    assert 4.5 < b.mean < 5.5
    assert 0.5 < b.sigma < 1.5
    assert b.window_start <= samples[0][0]
    assert b.window_end >= samples[-1][0]
