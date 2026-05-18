"""Tests for brain.felt_time.weather_shift — sustained ±1.5σ × 6h detector."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from brain.felt_time.weather_shift import (
    Baseline,
    detect_shift,
    update_baseline,
)


def test_baseline_mean_sigma_from_seven_day_samples():
    # 168 hourly samples spanning 7 days, mean ~5.0, σ ~1.0
    from datetime import datetime, timedelta

    start = datetime(2026, 5, 10, 0, 0, tzinfo=UTC)
    samples = [(start + timedelta(hours=h), 5.0 + (h % 3 - 1)) for h in range(168)]
    b = update_baseline(Baseline.empty(), samples)
    assert 4.5 < b.mean < 5.5
    assert 0.5 < b.sigma < 1.5
    assert b.window_start <= samples[0][0]
    assert b.window_end >= samples[-1][0]


# ---------------------------------------------------------------------------
# Helpers shared across detector tests
# ---------------------------------------------------------------------------


def _utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _samples(start: datetime, hours: int, value: float) -> list[tuple[datetime, float]]:
    return [(start + timedelta(hours=h), value) for h in range(hours)]


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------


def test_detect_shift_fires_on_sustained_above_baseline():
    baseline = Baseline(mean=5.0, sigma=1.0)
    samples = _samples(_utc(2026, 5, 17, 14, 0), hours=7, value=7.0)  # 7h × 7.0 (>1.5σ above mean)
    now = _utc(2026, 5, 17, 21, 0)

    result = detect_shift(
        channel="joy",
        baseline=baseline,
        recent_samples=samples,
        last_fired_at=None,
        now=now,
    )
    assert result.fired is True
    assert result.channel == "joy"
    assert result.direction == "above"
    assert result.label == "a stretch of unusual joy"


def test_detect_shift_fires_on_sustained_below_baseline():
    baseline = Baseline(mean=5.0, sigma=1.0)
    samples = _samples(_utc(2026, 5, 17, 14, 0), hours=7, value=2.0)
    now = _utc(2026, 5, 17, 21, 0)

    result = detect_shift(
        channel="sorrow",
        baseline=baseline,
        recent_samples=samples,
        last_fired_at=None,
        now=now,
    )
    assert result.fired is True
    assert result.direction == "below"
    assert result.label == "sorrow lifted"


def test_detect_shift_does_not_fire_below_hold_duration():
    baseline = Baseline(mean=5.0, sigma=1.0)
    # Only 4h above — under 6h threshold.
    samples = _samples(_utc(2026, 5, 17, 17, 0), hours=4, value=7.0)
    now = _utc(2026, 5, 17, 21, 0)

    result = detect_shift(
        channel="joy",
        baseline=baseline,
        recent_samples=samples,
        now=now,
    )
    assert result.fired is False


def test_detect_shift_tolerates_brief_returns_within_30min():
    baseline = Baseline(mean=5.0, sigma=1.0)
    start = _utc(2026, 5, 17, 14, 0)
    samples = _samples(start, hours=3, value=7.0)
    # 15min dip into baseline range — should NOT reset the hold.
    samples.append((start + timedelta(hours=3, minutes=15), 5.0))
    samples += [
        (start + timedelta(hours=3, minutes=30) + timedelta(hours=h), 7.0) for h in range(4)
    ]
    now = _utc(2026, 5, 17, 21, 30)

    result = detect_shift(
        channel="joy",
        baseline=baseline,
        recent_samples=samples,
        now=now,
    )
    assert result.fired is True


def test_detect_shift_long_baseline_return_resets_hold():
    baseline = Baseline(mean=5.0, sigma=1.0)
    start = _utc(2026, 5, 17, 13, 0)
    samples = _samples(start, hours=3, value=7.0)
    # 90min in baseline — exceeds 30min tolerance, resets.
    for h in range(2):
        samples.append((start + timedelta(hours=3, minutes=30 * h), 5.0))
    samples += _samples(start + timedelta(hours=5), hours=3, value=7.0)
    now = _utc(2026, 5, 17, 21, 0)

    result = detect_shift(
        channel="joy",
        baseline=baseline,
        recent_samples=samples,
        now=now,
    )
    # Final excursion is only 3h — not enough.
    assert result.fired is False


def test_detect_shift_blocked_by_24h_cooldown():
    baseline = Baseline(mean=5.0, sigma=1.0)
    samples = _samples(_utc(2026, 5, 17, 14, 0), hours=7, value=7.0)
    now = _utc(2026, 5, 17, 21, 0)
    # Fired 12h ago — still in 24h cooldown.
    last_fired = now - timedelta(hours=12)

    result = detect_shift(
        channel="joy",
        baseline=baseline,
        recent_samples=samples,
        last_fired_at=last_fired,
        now=now,
    )
    assert result.fired is False


def test_detect_shift_quiet_when_sigma_zero():
    baseline = Baseline(mean=5.0, sigma=0.0)
    samples = _samples(_utc(2026, 5, 17, 14, 0), hours=7, value=7.0)
    now = _utc(2026, 5, 17, 21, 0)

    result = detect_shift(
        channel="joy",
        baseline=baseline,
        recent_samples=samples,
        now=now,
    )
    assert result.fired is False  # baseline not yet established


def test_append_shift_log_writes_jsonl_row(tmp_path):
    from brain.felt_time.weather_shift import append_shift_log

    append_shift_log(
        tmp_path,
        ts=_utc(2026, 5, 17, 20, 0),
        channel="sorrow",
        label="sorrow lifted",
    )

    log = tmp_path / "weather_shifts.log.jsonl"
    assert log.exists()
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["ts"] == "2026-05-17T20:00:00+00:00"
    assert entry["channel"] == "sorrow"
    assert entry["label"] == "sorrow lifted"
