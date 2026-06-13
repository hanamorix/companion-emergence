"""Persisted SelfModelState — current_gap + bounded gap_history.

Spec §4, §9 (R-state + R-cadence Task 3).

Mirrors brain/felt_time/state.py load-or-recover pattern and uses
brain/health/attempt_heal.py::save_with_backup for atomic persistence.

State file: self_model_state.json in persona_dir.

§7 tripwire: when a new current_gap displaces an existing open/acknowledged
gap, the displaced gap MUST survive in gap_history AND a WARN log is emitted.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from brain.health.attempt_heal import save_with_backup
from brain.self_model.gap import Gap

logger = logging.getLogger(__name__)

_STATE_FILENAME = "self_model_state.json"

# History is bounded — keeps the last N completed gaps.
_GAP_HISTORY_CAP = 20

# Statuses considered "unresolved" for the §7 tripwire warning.
_UNRESOLVED_STATUSES = frozenset({"open", "acknowledged"})


# ── Gap serialisation ─────────────────────────────────────────────────────────


def _gap_to_dict(gap: Gap) -> dict:
    return {
        "per_channel": gap.per_channel,
        "magnitude": gap.magnitude,
        "unnamed_pressure": gap.unnamed_pressure,
        "note": gap.note,
        "status": gap.status,
        "first_seen_ts": gap.first_seen_ts,
        "last_seen_ts": gap.last_seen_ts,
        "sustained_ticks": gap.sustained_ticks,
        "channel_cooldowns": gap.channel_cooldowns,
    }


def _gap_from_dict(d: dict) -> Gap:
    return Gap(
        per_channel=d.get("per_channel") or {},
        magnitude=float(d.get("magnitude", 0.0)),
        unnamed_pressure=float(d.get("unnamed_pressure", 0.0)),
        note=d.get("note"),
        status=d.get("status", "open"),
        first_seen_ts=d.get("first_seen_ts"),
        last_seen_ts=d.get("last_seen_ts"),
        sustained_ticks=int(d.get("sustained_ticks", 0)),
        channel_cooldowns=d.get("channel_cooldowns") or {},
    )


# ── State dataclass ───────────────────────────────────────────────────────────


@dataclass
class SelfModelState:
    """Persisted self-model state.

    current_gap: the Gap being reflected on right now (None if no active gap).
    gap_history: completed gaps (resolved / dismissed / displaced), newest last.
                 Capped at _GAP_HISTORY_CAP entries (oldest dropped first).
    """

    current_gap: Gap | None = None
    gap_history: list[Gap] = field(default_factory=list)


# ── Persistence ───────────────────────────────────────────────────────────────


def _state_path(persona_dir: Path) -> Path:
    return persona_dir / _STATE_FILENAME


def load_or_recover(persona_dir: Path) -> tuple[SelfModelState, bool]:
    """Load persisted state.

    Returns (state, recovered):
    - Missing file  → (fresh state, False)   — normal first-run
    - Corrupt JSON  → (fresh state, True)    — R-state recovery
    - Valid file    → (loaded state, False)
    """
    path = _state_path(persona_dir)
    if not path.exists():
        return SelfModelState(), False

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SelfModelState(), True

    if not isinstance(raw, dict):
        return SelfModelState(), True

    try:
        current_gap: Gap | None = None
        raw_gap = raw.get("current_gap")
        if raw_gap is not None:
            current_gap = _gap_from_dict(raw_gap)

        gap_history: list[Gap] = []
        for item in raw.get("gap_history") or []:
            gap_history.append(_gap_from_dict(item))

        return SelfModelState(current_gap=current_gap, gap_history=gap_history), False
    except Exception:
        return SelfModelState(), True


def save(persona_dir: Path, state: SelfModelState) -> None:
    """Atomically persist state via save_with_backup."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_gap": _gap_to_dict(state.current_gap) if state.current_gap is not None else None,
        "gap_history": [_gap_to_dict(g) for g in state.gap_history],
    }
    save_with_backup(_state_path(persona_dir), payload)


# ── Gap transition helper ─────────────────────────────────────────────────────


def push_gap(state: SelfModelState, new_gap: Gap) -> SelfModelState:
    """Set new_gap as current_gap, displacing any existing one into gap_history.

    §7 tripwire: if the displaced gap is open/acknowledged (unresolved), a WARN
    log is emitted.  The displaced gap MUST survive in history regardless.

    gap_history is capped at _GAP_HISTORY_CAP (oldest dropped when full).
    """
    new_history = list(state.gap_history)

    if state.current_gap is not None:
        displaced = state.current_gap
        if displaced.status in _UNRESOLVED_STATUSES:
            logger.warning(
                "self_model: displacing unresolved gap (status=%r, magnitude=%.3f) "
                "into history — this gap was never resolved or dismissed.",
                displaced.status,
                displaced.magnitude,
            )
        # Always preserve the displaced gap in history.
        new_history.append(displaced)

    # Enforce the history cap.
    if len(new_history) > _GAP_HISTORY_CAP:
        new_history = new_history[-_GAP_HISTORY_CAP:]

    return SelfModelState(current_gap=new_gap, gap_history=new_history)
