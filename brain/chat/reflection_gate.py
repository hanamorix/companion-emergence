"""Reflection debounce: fire background pass-2 only on salience-significant turns,
and not more than once per time window per kind. TIME-based (cross-session and
cross-restart safe) — persists last_reflection_ts per kind in reflection_state.json.
Fails open (reflect now) on any state error, logged at WARNING so a persistent
failure (which would silently disable the debounce) is visible."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.health.json_io import Recovery, read_json, write_json
from brain.utils.log_throttle import count_throttled

log = logging.getLogger(__name__)

_SALIENCE_THRESHOLD = 0.30
_MIN_SECONDS_BETWEEN = 90.0  # debounce window (tunable); ~replaces the old 4-turn window
# #368: suppressing is routine (most short turns), so it logs at INFO at most
# hourly per reason and kind, with a running count, and at DEBUG in between.
_SUPPRESS_LOG_INTERVAL_S = 3600.0


def _load(persona_dir: Path) -> dict:
    # RAISE: a corrupt or unreadable file raises, and the caller fails open; a missing one is {}.
    return read_json(persona_dir / "reflection_state.json", default=dict, recover=Recovery.RAISE)


def _atomic_write(persona_dir: Path, state: dict) -> None:
    write_json(persona_dir / "reflection_state.json", state, indent=None, fsync=False)


def should_reflect(signal, persona_dir: Path, *, kind: str, now: datetime | None = None) -> bool:
    if signal.score < _SALIENCE_THRESHOLD:
        count_throttled(
            log, f"reflection_gate:low_salience:{kind}", logging.INFO,
            "reflection_gate: suppressed kind=%s, low salience %.2f < %.2f",
            kind, signal.score, _SALIENCE_THRESHOLD, interval_s=_SUPPRESS_LOG_INTERVAL_S,
        )
        return False
    now = now or datetime.now(UTC)
    try:
        state = _load(persona_dir)
        last_iso = state.get(kind, {}).get("last_reflection_ts")
        if last_iso is not None:
            last = datetime.fromisoformat(last_iso)
            elapsed = (now - last).total_seconds()
            if elapsed < _MIN_SECONDS_BETWEEN:
                count_throttled(
                    log, f"reflection_gate:too_soon:{kind}", logging.INFO,
                    "reflection_gate: suppressed kind=%s, too soon %.0fs < %.0fs",
                    kind, elapsed, _MIN_SECONDS_BETWEEN, interval_s=_SUPPRESS_LOG_INTERVAL_S,
                )
                return False
        state.setdefault(kind, {})["last_reflection_ts"] = now.isoformat()
        _atomic_write(persona_dir, state)
        return True
    except Exception:  # noqa: BLE001 — fail open: a state bug must not silence her reflection
        log.warning("reflection_gate state error; reflecting (fail-open)", exc_info=True)
        return True
