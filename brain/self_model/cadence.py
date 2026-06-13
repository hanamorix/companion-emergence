"""Persisted, self-pacing wall-clock cadence for self-model reflection.

Mirrors brain/soul/cadence.py exactly in shape and intent.

The supervisor's other cadences use process-relative time.monotonic(), which
resets on every restart and does not advance during system sleep. This module
persists the NEXT-DUE time as a wall-clock datetime so the cadence survives
restarts and sleep, and self-paces by outcome:

  - clean (drained, no failures)    → normal 6-hour interval (R-cadence)
  - backlog (candidates pending)    → 30-min catch-up to drain fast
  - model-call failures (e.g. 429)  → escalating backoff (30m → 1h → 2h …)
                                       capped at normal_interval; resets on
                                       first clean pass

State file: self_model_cadence_state.json in persona_dir.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

_STATE_FILENAME = "self_model_cadence_state.json"

# Normal interval between reflection ticks (R-cadence).
_BASE_INTERVAL_HOURS = 6.0
_BASE_INTERVAL_S = _BASE_INTERVAL_HOURS * 3600.0

# Backlog remaining: re-run in 30 min to drain fast.
_CATCHUP_INTERVAL_S = 30 * 60.0

# Backoff base: 30 min, then doubles (1h, 2h, …) capped at normal interval.
_BACKOFF_BASE_S = 30 * 60.0


@dataclass(frozen=True)
class SelfModelCadenceState:
    """Persisted cadence state for self-model reflection.

    next_reflection_at: wall-clock due time (None → due now / never run).
    consecutive_failures: count of consecutive model-call failures.
    """

    next_reflection_at: datetime | None  # None => due now
    consecutive_failures: int


def _state_path(persona_dir: Path) -> Path:
    return persona_dir / _STATE_FILENAME


def _parse_ts(raw: object) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def load(persona_dir: Path) -> SelfModelCadenceState:
    """Load persisted cadence. Missing/corrupt → due-now, zero failures.

    Fail-safe toward running, not toward silence.
    """
    try:
        data = json.loads(_state_path(persona_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SelfModelCadenceState(next_reflection_at=None, consecutive_failures=0)
    if not isinstance(data, dict):
        return SelfModelCadenceState(next_reflection_at=None, consecutive_failures=0)
    cf = data.get("consecutive_failures", 0)
    if not isinstance(cf, int) or cf < 0:
        cf = 0
    return SelfModelCadenceState(
        next_reflection_at=_parse_ts(data.get("next_reflection_at")),
        consecutive_failures=cf,
    )


def save(persona_dir: Path, state: SelfModelCadenceState) -> None:
    """Atomically persist the cadence (temp file + rename)."""
    path = _state_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_reflection_at": (
            state.next_reflection_at.isoformat()
            if state.next_reflection_at is not None
            else None
        ),
        "consecutive_failures": state.consecutive_failures,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def is_due(state: SelfModelCadenceState, *, now: datetime) -> bool:
    """True when the self-model reflection should run now."""
    return state.next_reflection_at is None or now >= state.next_reflection_at


def compute_next_state(
    state: SelfModelCadenceState,
    *,
    outcome: str,
    now: datetime,
    normal_interval_s: float = _BASE_INTERVAL_S,
) -> SelfModelCadenceState:
    """Advance cadence state based on tick outcome.

    outcome values:
      "clean"   → normal interval, reset consecutive_failures.
      "backlog" → short catch-up interval, reset consecutive_failures.
      "failure" → escalating backoff, increment consecutive_failures.

    Any unrecognised outcome is treated as "clean".
    """
    if outcome == "failure":
        cf = state.consecutive_failures + 1
        backoff = _BACKOFF_BASE_S * (2 ** (cf - 1))
        interval = min(normal_interval_s, backoff)
        return SelfModelCadenceState(
            next_reflection_at=now + timedelta(seconds=interval),
            consecutive_failures=cf,
        )
    if outcome == "backlog":
        interval = min(_CATCHUP_INTERVAL_S, normal_interval_s)
        return SelfModelCadenceState(
            next_reflection_at=now + timedelta(seconds=interval),
            consecutive_failures=0,
        )
    # "clean" or unknown
    return SelfModelCadenceState(
        next_reflection_at=now + timedelta(seconds=normal_interval_s),
        consecutive_failures=0,
    )
