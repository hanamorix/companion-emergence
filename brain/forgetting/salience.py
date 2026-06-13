"""salience.py — composite scoring from 5 inputs per spec §3.

Pure function. Reads MemoryStore, HebbianMatrix, FeltTimeState; computes
a [0, 1] salience score. Defaults are intuition-driven (tunable in a
future spec; tracked in spec §7 deferred).

Composite formula:
    salience = 0.30·emotion + 0.20·hebbian + 0.20·recall + 0.20·soul + 0.10·freshness
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from brain.felt_time.state import FeltTimeState
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore

DEFAULT_WEIGHTS: dict[str, float] = {
    "emotion": 0.30,
    "hebbian": 0.20,
    "recall": 0.20,
    "soul": 0.20,
    "freshness": 0.10,
}

# Normalisation denominators (per spec §3).
_EMOTION_DENOMINATOR = 10.0
_HEBBIAN_DENOMINATOR = 20.0
_RECALL_DENOMINATOR = 10.0
_FRESHNESS_LIVED_HOURS_HORIZON = 2160.0  # 90 lived-days — soft landing past the 30-day recency grace

# v0.0.33 Track 3 — peak blend. The FIELD never decays; its salience
# contribution lingers over a long lived-time horizon so forgetting stays
# reachable (spec constraint: sticky, never immortal). Calibration per the
# plan's table: peak-5 fades ~day 46, loses ~day 127.
_PEAK_LAMBDA = 1.5
_PEAK_LOG_DENOMINATOR = math.log1p(10.0)
_PEAK_LINGER_LIVED_HOURS_HORIZON = 4320.0  # 180 lived-days


@dataclass(frozen=True)
class SalienceInputs:
    emotion: float
    hebbian: float
    recall: float
    soul: float
    freshness: float


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _emotion_input(memory: Memory) -> float:
    if not memory.emotions:
        return 0.0
    max_intensity = max(memory.emotions.values())
    return _clamp(float(max_intensity) / _EMOTION_DENOMINATOR)


def _hebbian_input(memory: Memory, hebbian: HebbianMatrix) -> float:
    try:
        count = hebbian.activation_count(memory.id)
    except (AttributeError, KeyError):
        count = 0
    return _clamp(float(count) / _HEBBIAN_DENOMINATOR)


def _recall_input(memory: Memory) -> float:
    return _clamp(float(memory.recall_count) / _RECALL_DENOMINATOR)


def _soul_input(memory: Memory, soul_linked_ids: Iterable[str]) -> float:
    return 1.0 if memory.id in soul_linked_ids else 0.0


def _lived_hours_since(anchor: datetime, felt_time_state: FeltTimeState | None) -> float | None:
    """Lived hours elapsed since `anchor`, or None for the cold-start
    treat-as-fresh cases (no felt-time state / no lived age / no usable rate).
    Extracted from _freshness_input (v0.0.33) so the peak linger shares one
    lived-time approximation."""
    if felt_time_state is None:
        return None
    now = datetime.now(UTC)
    wall_delta_s = (now - anchor).total_seconds()
    if felt_time_state.lived_age_hours <= 0.0:
        return None
    if felt_time_state.last_tick_ts is None:
        return wall_delta_s / 3600.0
    wall_clock_since_first_tick_s = (
        now - datetime.fromisoformat(felt_time_state.last_tick_ts)
    ).total_seconds()
    if wall_clock_since_first_tick_s <= 0:
        return None
    rate_lived_per_wall = felt_time_state.lived_age_hours / max(
        wall_clock_since_first_tick_s / 3600.0, 1e-6
    )
    return (wall_delta_s / 3600.0) * rate_lived_per_wall


def _freshness_input(memory: Memory, felt_time_state: FeltTimeState | None) -> float:
    # Anchor on last access if the memory was ever recalled, else on creation.
    # A freshly-created memory has last_accessed_at=None; falling back to
    # created_at is what lets recency protect a recent memory (it used to
    # return 0.0 here — the structural defect that deleted days-old memories).
    anchor = memory.last_accessed_at or memory.created_at
    if anchor is None:
        return 0.0
    lived = _lived_hours_since(anchor, felt_time_state)
    if lived is None:
        return 1.0  # cold-start cases — treat as fresh (spec §7)
    return 1.0 - _clamp(lived / _FRESHNESS_LIVED_HOURS_HORIZON)


def _peak_input(memory: Memory, felt_time_state: FeltTimeState | None) -> float:
    """Salience residue of having mattered: log-scaled peak intensity with a
    long linear linger. Survives emotion decay + noise-floor deletion by
    reading the monotone peak field, not the live dict."""
    peak = getattr(memory, "peak_emotion_intensity", 0.0) or 0.0
    if peak <= 0.0:
        return 0.0
    anchor = memory.last_accessed_at or memory.created_at
    lived = _lived_hours_since(anchor, felt_time_state)
    linger = 1.0 if lived is None else 1.0 - _clamp(lived / _PEAK_LINGER_LIVED_HOURS_HORIZON)
    return _clamp(_PEAK_LAMBDA * math.log1p(peak) / _PEAK_LOG_DENOMINATOR * linger)


def score(
    memory: Memory,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    felt_time_state: FeltTimeState | None,
    soul_linked_ids: Iterable[str],
    weights: dict[str, float] | None = None,
) -> float:
    """Composite salience in [0, 1]. See module docstring."""
    w = weights or DEFAULT_WEIGHTS
    inputs = SalienceInputs(
        emotion=max(_emotion_input(memory), _peak_input(memory, felt_time_state)),
        hebbian=_hebbian_input(memory, hebbian),
        recall=_recall_input(memory),
        soul=_soul_input(memory, soul_linked_ids),
        freshness=_freshness_input(memory, felt_time_state),
    )
    return _clamp(
        w["emotion"] * inputs.emotion
        + w["hebbian"] * inputs.hebbian
        + w["recall"] * inputs.recall
        + w["soul"] * inputs.soul
        + w["freshness"] * inputs.freshness
    )


def compute_inputs(
    memory: Memory,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    felt_time_state: FeltTimeState | None,
    soul_linked_ids: Iterable[str],
) -> SalienceInputs:
    """Return the per-input breakdown — used by the graveyard writer to
    record salience_inputs_at_drop for audit."""
    return SalienceInputs(
        emotion=max(_emotion_input(memory), _peak_input(memory, felt_time_state)),
        hebbian=_hebbian_input(memory, hebbian),
        recall=_recall_input(memory),
        soul=_soul_input(memory, soul_linked_ids),
        freshness=_freshness_input(memory, felt_time_state),
    )
