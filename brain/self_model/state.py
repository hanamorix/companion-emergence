"""Persisted SelfModelState — current_gap + bounded gap_history.

Spec §4, §9 (R-state + R-cadence Task 3).

Mirrors brain/felt_time/state.py load-or-recover pattern and uses
brain/health/json_io.py::write_json for atomic persistence.

State file: self_model_state.json in persona_dir.

§7 tripwire: when a new current_gap displaces an existing open/acknowledged
gap, the displaced gap MUST survive in gap_history AND a WARN log is emitted.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from brain.health.json_io import read_json_result, write_json
from brain.self_model.gap import UNSET, Gap
from brain.utils.file_lock import file_lock
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

_STATE_FILENAME = "self_model_state.json"

# History is bounded — keeps the last N completed gaps.
_GAP_HISTORY_CAP = 20

# Statuses considered "unresolved" for the §7 tripwire warning.
_UNRESOLVED_STATUSES = frozenset({"open", "acknowledged"})


# ── Gap serialisation ─────────────────────────────────────────────────────────


def _gap_to_dict(gap: Gap) -> dict:
    out = {
        "per_channel": gap.per_channel,
        "magnitude": gap.magnitude,
        "unnamed_pressure": gap.unnamed_pressure,
        "note": gap.note,
        "status": gap.status,
        "first_seen_ts": gap.first_seen_ts,
        "last_seen_ts": gap.last_seen_ts,
        "sustained_ticks": gap.sustained_ticks,
        "channel_cooldowns": gap.channel_cooldowns,
        "directions": gap.directions,
    }
    # A gap that never carried the field stays without the key, so it reads back as UNSET.
    if gap.reset_seen is not UNSET:
        out["reset_seen"] = gap.reset_seen
    return out


def _directions_from(value: object) -> dict[str, int]:
    """`{str: int}` from a saved value; an empty dict for a missing key or a malformed value."""
    if not isinstance(value, dict):
        return {}
    try:
        return {str(k): int(v) for k, v in value.items()}
    except (TypeError, ValueError):
        return {}


def _non_finite(value: object) -> bool:
    """True when a JSON value holds NaN or ±Infinity anywhere (json.loads accepts them; old saves could write them)."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_non_finite(v) for v in value.values())
    if isinstance(value, list):
        return any(_non_finite(v) for v in value)
    return False


def _gap_from_dict(d: dict) -> Gap:
    # A NaN never equals itself, so a gap holding one never matches its own re-load (the tick's snapshot check,
    # #558) and cannot be saved (write_json refuses NaN): it is malformed.
    if _non_finite(d):
        raise ValueError("a non-finite number (NaN or Infinity)")
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
        directions=_directions_from(d.get("directions")),
        reset_seen=d["reset_seen"] if "reset_seen" in d else UNSET,
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
    - Valid file with malformed gaps → (the gaps that parse, True): a bad
      current_gap loads as None and a bad history entry is left out, so one
      bad entry no longer costs the whole history (#558).
    """
    path = _state_path(persona_dir)
    # An unreadable, corrupt or non-object file is the module's default (None) and its evidence is logged there.
    res = read_json_result(path, default=lambda: None, expect=dict)
    if res.outcome == "missing":
        return SelfModelState(), False
    if res.outcome == "defaulted":
        return SelfModelState(), True
    raw = res.data

    problems: list[str] = []
    current_gap: Gap | None = None
    raw_gap = raw.get("current_gap")
    if raw_gap is not None:
        try:
            current_gap = _gap_from_dict(raw_gap)
        except Exception as exc:  # noqa: BLE001 -- a malformed current gap loads as None
            problems.append(f"dropped a malformed current_gap ({exc})")

    gap_history: list[Gap] = []
    raw_history = raw.get("gap_history") or []
    if not isinstance(raw_history, list):
        problems.append(f"dropped gap_history: expected a list, got {type(raw_history).__name__}")
        raw_history = []
    bad: list[str] = []
    for item in raw_history:
        try:
            gap_history.append(_gap_from_dict(item))
        except Exception as exc:  # noqa: BLE001 -- a malformed entry is left out, the rest load
            bad.append(str(exc))
    if bad:
        problems.append(f"dropped {len(bad)} of {len(raw_history)} gap_history entries ({'; '.join(bad)})")

    state = SelfModelState(current_gap=current_gap, gap_history=gap_history)
    if problems:
        return _recovered(path, state, "; ".join(problems))
    return state, False


def _recovered(path: Path, state: SelfModelState, why: str) -> tuple[SelfModelState, bool]:
    """Keep what parsed, and say so: every caller discards the flag (#367)."""
    # Callers load this every chat turn until the supervisor saves over it.
    warn_throttled(
        logger, f"self_model_state:{path}",
        "self-model state %s: %s; kept %d gap_history entries", path, why, len(state.gap_history),
        exc_info=False,
    )
    return state, True


@contextmanager
def locked(persona_dir: Path) -> Iterator[None]:
    """Hold the state file's cross-process lock for a whole load → change → save (#558).

    Every writer takes it around its read-modify-write: the reconcile tool, the supervisor tick (around a re-load
    that checks its snapshot still holds, and its save) and the one-time repair. Readers that only render it (the
    chat prompt, tool recruitment) do not: ``save`` replaces the file atomically. Not re-entrant: never load-and-save
    under it through a helper that takes it again.
    """
    with file_lock(_state_path(persona_dir)):
        yield


def save(persona_dir: Path, state: SelfModelState) -> None:
    """Atomically persist state via write_json."""
    payload = {
        "current_gap": _gap_to_dict(state.current_gap) if state.current_gap is not None else None,
        "gap_history": [_gap_to_dict(g) for g in state.gap_history],
    }
    write_json(_state_path(persona_dir), payload, backups=3)


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
