"""weather_shift.py — sustained emotional-weather-shift anchor detector.

Spec §4 weather shift definition. Fires an anchor when any emotion
channel crosses ±1.5σ from its 7-day rolling baseline AND holds for
≥6h (with brief <30min returns to baseline not resetting the hold).
24h per-channel cooldown after firing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

BASELINE_WINDOW = timedelta(days=7)
CROSS_SIGMA = 1.5
HOLD_DURATION = timedelta(hours=6)
HOLD_BREAK_TOLERANCE = timedelta(minutes=30)
COOLDOWN = timedelta(hours=24)
WEATHER_LOG_FILENAME = "weather_shifts.log.jsonl"


@dataclass
class Baseline:
    mean: float = 0.0
    sigma: float = 0.0
    window_start: datetime | None = None
    window_end: datetime | None = None
    sample_count: int = 0

    @classmethod
    def empty(cls) -> Baseline:
        return cls()


def update_baseline(_prev: Baseline, samples: Iterable[tuple[datetime, float]]) -> Baseline:
    """Recompute mean+σ over the provided samples. Samples must be ts-ordered."""
    points = list(samples)
    if not points:
        return Baseline.empty()
    values = [v for _, v in points]
    n = len(values)
    mean = sum(values) / n
    sigma = math.sqrt(sum((v - mean) ** 2 for v in values) / n) if n > 1 else 0.0
    return Baseline(
        mean=mean,
        sigma=sigma,
        window_start=points[0][0],
        window_end=points[-1][0],
        sample_count=n,
    )


@dataclass
class ShiftDetection:
    fired: bool
    channel: str | None = None
    direction: str | None = None  # "above" | "below"
    label: str | None = None
    at: datetime | None = None


# Spec §4: per-channel template label table. Two slots per channel
# (above / below) keep the language honest without LLM generation.
_LABEL_TEMPLATES: dict[str, tuple[str, str]] = {
    "love": ("a stretch of unusual love", "love quieted"),
    "joy": ("a stretch of unusual joy", "the joy quieted"),
    "sorrow": ("a stretch of unusual sorrow", "sorrow lifted"),
    "anger": ("a stretch of unusual anger", "the anger settled"),
    "fear": ("a stretch of unusual fear", "the fear loosened"),
    "surprise": ("a stretch of unusual surprise", "the surprise faded"),
    "disgust": ("a stretch of unusual disgust", "disgust eased"),
    "anticipation": ("a stretch of held anticipation", "the anticipation released"),
}


def _label_for(channel: str, direction: str) -> str:
    above, below = _LABEL_TEMPLATES.get(channel, (f"unusual {channel}", f"{channel} settled"))
    return above if direction == "above" else below


def detect_shift(
    *,
    channel: str,
    baseline: Baseline,
    recent_samples: list[tuple[datetime, float]],
    last_fired_at: datetime | None = None,
    now: datetime | None = None,
) -> ShiftDetection:
    """Decide whether this channel should fire a sustained-shift anchor.

    Inputs:
      baseline: 7-day rolling (mean, σ) for this channel.
      recent_samples: chronologically-ordered samples within the
        evaluation window — anything fresher than the start of the hold
        period (last 6h+ at heartbeat cadence).
      last_fired_at: last time this channel fired a shift (cooldown gate).
      now: current time, defaults to UTC now.

    Returns ShiftDetection(fired=True, ...) only when ALL of: crossed
    ±1.5σ, held ≥6h with brief returns <30min, and not in 24h cooldown.
    """
    if baseline.sigma == 0.0 or not recent_samples:
        return ShiftDetection(fired=False)

    now = now or datetime.now(UTC)
    if last_fired_at and now - last_fired_at < COOLDOWN:
        return ShiftDetection(fired=False)

    threshold = CROSS_SIGMA * baseline.sigma
    upper = baseline.mean + threshold
    lower = baseline.mean - threshold

    # Walk ALL samples forward, tracking continuous excursion above OR below.
    # Brief returns to baseline (<30min) tolerated; longer returns invalidate
    # the hold. We do NOT restrict to a hold_window — an excursion that began
    # before (now - HOLD_DURATION) and continues through now counts.
    #
    # neutral_started_at tracks when the signal entered the baseline band.
    # We only decide whether to reset when we see the next excursion sample
    # (or end of stream), so that the neutral duration is measured correctly
    # rather than using the gap from last excursion to neutral point.
    samples = sorted(recent_samples)
    direction: str | None = None
    excursion_started_at: datetime | None = None
    last_in_excursion_at: datetime | None = None
    neutral_started_at: datetime | None = None

    for ts, value in samples:
        if value > upper:
            now_dir = "above"
        elif value < lower:
            now_dir = "below"
        else:
            now_dir = None

        if now_dir is None:
            # Entering or continuing baseline band.
            if neutral_started_at is None:
                neutral_started_at = ts
            continue

        # We are back in an excursion zone. Check if the preceding neutral
        # period was long enough to break the hold.
        if neutral_started_at is not None:
            neutral_duration = ts - neutral_started_at
            if neutral_duration > HOLD_BREAK_TOLERANCE:
                # Long return to baseline — reset.
                direction = None
                excursion_started_at = None
                last_in_excursion_at = None
            neutral_started_at = None

        if direction is None:
            direction = now_dir
            excursion_started_at = ts
            last_in_excursion_at = ts
            continue

        if now_dir != direction:
            # Crossed to the opposite side — restart excursion.
            direction = now_dir
            excursion_started_at = ts
            last_in_excursion_at = ts
            continue

        last_in_excursion_at = ts

    if not excursion_started_at or not last_in_excursion_at or direction is None:
        return ShiftDetection(fired=False)

    if last_in_excursion_at - excursion_started_at < HOLD_DURATION:
        return ShiftDetection(fired=False)

    return ShiftDetection(
        fired=True,
        channel=channel,
        direction=direction,
        label=_label_for(channel, direction),
        at=last_in_excursion_at,
    )


def append_shift_log(persona_dir: Path, *, ts: datetime, channel: str, label: str) -> None:
    """Append a single anchor row to weather_shifts.log.jsonl."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    log = persona_dir / WEATHER_LOG_FILENAME
    entry = {
        "ts": ts.astimezone(UTC).isoformat(),
        "channel": channel,
        "label": label,
    }
    with log.open("a") as f:
        f.write(json.dumps(entry) + "\n")
