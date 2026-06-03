"""Persisted, self-pacing cadence for autonomous soul review.

The supervisor's other cadences time off process-relative ``time.monotonic()``,
which resets on every restart and does not advance during system sleep — so a
6-hour soul-review interval rarely elapses on a desktop app that isn't running
continuously, and crystallization candidates pile up. (A user hit exactly this:
a multi-day Claude session-limit (429) outage left a backlog the autonomous
review never drained.)

This module persists the NEXT-due time as a wall-clock timestamp so the cadence
survives restarts and sleep, and self-paces by outcome:

  - backlog remains (calls working) -> short catch-up interval, drain fast
  - model-call failures (e.g. 429)  -> escalating backoff, recover fast
                                        post-limit without hammering the API
  - clean (drained, no failures)    -> normal interval

Only the soul-review cadence is persisted here; forgetting + narrative stay on
their existing supervisor cadence (decoupling is safe — forgetting already
exempts under-review soul-linked memories regardless of pass order).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

_STATE_FILENAME = "soul_review_state.json"

# Backlog remaining (calls working): re-run in 30 min to drain fast.
_CATCHUP_INTERVAL_S = 30 * 60.0
# First failure backs off 30 min, then doubles (1h, 2h, …) capped at the normal
# interval — so a sustained outage settles to the normal cadence (no hammering)
# while the first success after the limit lifts resets it immediately.
_BACKOFF_BASE_S = 30 * 60.0


@dataclass(frozen=True)
class ReviewCadenceState:
    next_review_at: datetime | None  # None => due now (never run / reset)
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


def load_cadence_state(persona_dir: Path) -> ReviewCadenceState:
    """Load persisted cadence. Missing/corrupt → due-now, zero failures.

    Self-healing: a bad state file must never wedge the review (fail-safe
    toward running, not toward silence).
    """
    try:
        data = json.loads(_state_path(persona_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ReviewCadenceState(next_review_at=None, consecutive_failures=0)
    if not isinstance(data, dict):
        return ReviewCadenceState(next_review_at=None, consecutive_failures=0)
    cf = data.get("consecutive_failures", 0)
    if not isinstance(cf, int) or cf < 0:
        cf = 0
    return ReviewCadenceState(
        next_review_at=_parse_ts(data.get("next_review_at")),
        consecutive_failures=cf,
    )


def save_cadence_state(persona_dir: Path, state: ReviewCadenceState) -> None:
    """Atomically persist the cadence (temp file + rename)."""
    path = _state_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_review_at": (
            state.next_review_at.isoformat() if state.next_review_at is not None else None
        ),
        "consecutive_failures": state.consecutive_failures,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def is_due(state: ReviewCadenceState, *, now: datetime) -> bool:
    """True when the soul review should run now (never-run, or wall-clock due)."""
    return state.next_review_at is None or now >= state.next_review_at


def compute_next_state(
    *,
    now: datetime,
    model_failures: int,
    eligible_pending: int,
    normal_interval_s: float,
    prev_failures: int,
) -> ReviewCadenceState:
    """Next-due time + failure count, paced by the tick's outcome.

    - ``model_failures > 0`` → escalating backoff (30m, 1h, 2h, … capped at
      ``normal_interval_s``); ``consecutive_failures`` increments. The first
      clean pass resets it, so recovery after a 429 lifts is fast.
    - else ``eligible_pending > 0`` → 30-min catch-up to drain the backlog.
    - else → ``normal_interval_s``.
    """
    if model_failures > 0:
        cf = prev_failures + 1
        backoff = _BACKOFF_BASE_S * (2 ** (cf - 1))
        interval = min(normal_interval_s, backoff)
        return ReviewCadenceState(
            next_review_at=now + timedelta(seconds=interval), consecutive_failures=cf
        )
    if eligible_pending > 0:
        interval = min(_CATCHUP_INTERVAL_S, normal_interval_s)
        return ReviewCadenceState(
            next_review_at=now + timedelta(seconds=interval), consecutive_failures=0
        )
    return ReviewCadenceState(
        next_review_at=now + timedelta(seconds=normal_interval_s), consecutive_failures=0
    )
