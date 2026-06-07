"""Reflection debounce: fire background pass-2 only on salience-significant turns,
and not more than once per time window per kind. TIME-based (cross-session and
cross-restart safe) — persists last_reflection_ts per kind in reflection_state.json.
Fails open (reflect now) on any state error, logged at WARNING so a persistent
failure (which would silently disable the debounce) is visible."""
from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

_SALIENCE_THRESHOLD = 0.30
_MIN_SECONDS_BETWEEN = 90.0  # debounce window (tunable); ~replaces the old 4-turn window


def _load(persona_dir: Path) -> dict:
    p = persona_dir / "reflection_state.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))  # may raise → caller fails open


def _atomic_write(persona_dir: Path, state: dict) -> None:
    persona_dir.mkdir(parents=True, exist_ok=True)
    tmp = persona_dir / "reflection_state.json.tmp"
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, persona_dir / "reflection_state.json")  # atomic on POSIX/Windows


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
