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

log = logging.getLogger(__name__)

_SALIENCE_THRESHOLD = 0.30
_MIN_SECONDS_BETWEEN = 90.0  # debounce window (tunable); ~replaces the old 4-turn window


def _load(persona_dir: Path) -> dict:
    # RAISE: a corrupt or unreadable file raises, and the caller fails open; a missing one is {}.
    return read_json(persona_dir / "reflection_state.json", default=dict, recover=Recovery.RAISE)


def _atomic_write(persona_dir: Path, state: dict) -> None:
    write_json(persona_dir / "reflection_state.json", state, indent=None, fsync=False)


def should_reflect(signal, persona_dir: Path, *, kind: str, now: datetime | None = None) -> bool:
    if signal.score < _SALIENCE_THRESHOLD:
        return False
    now = now or datetime.now(UTC)
    try:
        state = _load(persona_dir)
        last_iso = state.get(kind, {}).get("last_reflection_ts")
        if last_iso is not None:
            last = datetime.fromisoformat(last_iso)
            if (now - last).total_seconds() < _MIN_SECONDS_BETWEEN:
                return False
        state.setdefault(kind, {})["last_reflection_ts"] = now.isoformat()
        _atomic_write(persona_dir, state)
        return True
    except Exception:  # noqa: BLE001 — fail open: a state bug must not silence her reflection
        log.warning("reflection_gate state error; reflecting (fail-open)", exc_info=True)
        return True
