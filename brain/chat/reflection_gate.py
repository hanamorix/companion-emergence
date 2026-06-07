"""Reflection debounce: fire the background pass-2 passes only on salience-
significant turns, and not more than once per debounce window per kind.
Persists a per-kind cursor in reflection_state.json. Fails open (reflect now)
on any state error, and records the turn_index when it fires (caller passes a
monotonic per-session turn index)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_SALIENCE_THRESHOLD = 0.30
_MIN_TURNS_BETWEEN = 4  # debounce window in turns


def _load(persona_dir: Path) -> dict:
    p = persona_dir / "reflection_state.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())  # may raise → caller treats as fail-open


def should_reflect(signal, persona_dir: Path, *, kind: str, turn_index: int) -> bool:
    if signal.score < _SALIENCE_THRESHOLD:
        return False
    try:
        state = _load(persona_dir)
        last = state.get(kind, {}).get("last_turn_index")
        if last is not None and (turn_index - last) < _MIN_TURNS_BETWEEN:
            return False
        state.setdefault(kind, {})["last_turn_index"] = turn_index
        persona_dir.mkdir(parents=True, exist_ok=True)
        (persona_dir / "reflection_state.json").write_text(json.dumps(state))
        return True
    except Exception:  # noqa: BLE001 — fail open: a state bug must not silence her reflection
        log.debug("reflection_gate state error; reflecting (fail-open)", exc_info=True)
        return True
