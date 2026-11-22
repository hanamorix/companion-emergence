"""weather_shift.py — rolling mean and σ of a sampled signal.

`update_baseline` feeds the felt-time intensity drivers and the
chat-activity baseline (`supervisor._derive_intensity_drivers`). The
module once also held a sustained-shift anchor detector and its log
writer; that anchor never fired and was removed in #785. The module name
is kept so the supervisor's imports and the saved
`FeltTimeState.weather_baselines` field keep their meaning.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

BASELINE_WINDOW = timedelta(days=7)


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
