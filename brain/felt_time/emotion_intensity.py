"""emotion_intensity.py — derive emotional_intensity driver from memory baseline.

Computes the max positive sigma-deviation across emotion channels, using the
same memory rows already loaded for body-strain computation in the supervisor.

Returns a float in [0, 1]:
  - 0.0 when fewer than 10 samples exist per channel, or when the per-channel
    max is at or below the rolling baseline mean.
  - Up to 1.0 when the peak emotion deviates far above baseline (clips at 3 sigma).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.memory.store import Memory

_MIN_SAMPLES = 10
_SIGMA_SCALE = 3.0  # deviation at which intensity saturates to 1.0


def compute(memories: list[Memory]) -> float:
    """Return emotional_intensity in [0.0, 1.0] from per-channel sigma-deviation.

    Uses the same memory rows already loaded by _derive_intensity_drivers.
    The current channel value is the per-channel max (mirrors aggregate_state
    max-pool behaviour without requiring vocabulary registration).

    Returns 0.0 when:
      - memories is empty or no memories.db exists
      - all channels have fewer than _MIN_SAMPLES samples
      - all current values are at or below their channel baseline mean
    """
    per_ch: dict[str, list[float]] = defaultdict(list)
    for mem in memories:
        if mem.created_at is None:
            continue
        for ch, val in mem.emotions.items():
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval > 0.0:
                per_ch[ch].append(fval)

    devs: list[float] = []
    for _ch, vals in per_ch.items():
        n = len(vals)
        if n < _MIN_SAMPLES:
            continue
        mean = sum(vals) / n
        sigma = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
        if sigma <= 0.0:
            continue
        cur = max(vals)  # max-pool mirrors aggregate_state behaviour
        dev = (cur - mean) / max(sigma, 0.1)
        if dev > 0.0:
            devs.append(dev)

    return min(1.0, max(devs) / _SIGMA_SCALE) if devs else 0.0
