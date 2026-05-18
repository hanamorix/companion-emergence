"""salience.py — composite scoring from 5 inputs per spec §3.

Pure function. Reads MemoryStore, HebbianMatrix, FeltTimeState; computes
a [0, 1] salience score. Defaults are intuition-driven (tunable in a
future spec; tracked in spec §7 deferred).

Composite formula:
    salience = 0.30·emotion + 0.20·hebbian + 0.20·recall + 0.20·soul + 0.10·freshness
"""

from __future__ import annotations

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
_FRESHNESS_LIVED_HOURS_HORIZON = 720.0  # 30 lived-days


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


def _freshness_input(memory: Memory, felt_time_state: FeltTimeState | None) -> float:
    if memory.last_accessed_at is None:
        return 0.0  # never accessed = no freshness signal
    if felt_time_state is None:
        return 1.0  # cold-start before felt-time exists = treat as fresh (spec §7)
    # Approximation per spec §3 + §7 deferred — wall-clock delta × current
    # lived-age ratio. Exact per-memory lived_age_at column deferred.
    now = datetime.now(UTC)
    wall_delta_s = (now - memory.last_accessed_at).total_seconds()
    # If FeltTimeState has lived_age=0 (cold start), there's no rate to
    # apply yet — treat as 1.0 (fresh) so we don't fade everything on the
    # first pass.
    if felt_time_state.lived_age_hours <= 0.0:
        return 1.0  # cold-start: no lived age yet — treat as fresh
    if felt_time_state.last_tick_ts is None:
        # lived_age known but no tick history — approximate rate=1 (wall=lived).
        lived_hours_since_access = wall_delta_s / 3600.0
    else:
        wall_clock_since_first_tick_s = (
            now - datetime.fromisoformat(felt_time_state.last_tick_ts)
        ).total_seconds()
        if wall_clock_since_first_tick_s <= 0:
            return 1.0
        # rate = lived_hours per wall_hour since first tick (rough)
        # For a tighter approximation we'd integrate, but spec §7 accepts this.
        rate_lived_per_wall = felt_time_state.lived_age_hours / max(
            wall_clock_since_first_tick_s / 3600.0, 1e-6
        )
        lived_hours_since_access = (wall_delta_s / 3600.0) * rate_lived_per_wall
    return 1.0 - _clamp(lived_hours_since_access / _FRESHNESS_LIVED_HOURS_HORIZON)


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
        emotion=_emotion_input(memory),
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
        emotion=_emotion_input(memory),
        hebbian=_hebbian_input(memory, hebbian),
        recall=_recall_input(memory),
        soul=_soul_input(memory, soul_linked_ids),
        freshness=_freshness_input(memory, felt_time_state),
    )
