"""brain.maker.charge — the creative-charge accumulator.

A single persisted scalar that accumulates from her inner events (emotion /
soul / dream) and decays when idle. The maker-tick reads it; at threshold it
discharges into one making and resets. Pull-computed (the tick integrates
recent activity) so it is recovery-robust — no scattered producer hooks.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_CHARGE_FILE = "maker_charge.json"


@dataclass
class MakerCharge:
    charge: float
    last_tick_ts: str | None
    last_fire_ts: str | None
    prior_soul_count: int


def _charge_path(persona_dir: Path) -> Path:
    return persona_dir / _CHARGE_FILE


def load_charge(persona_dir: Path) -> MakerCharge:
    """Load persisted charge; cold default on missing/corrupt (fail-safe)."""
    path = _charge_path(persona_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return MakerCharge(
            charge=float(raw.get("charge", 0.0)),
            last_tick_ts=raw.get("last_tick_ts"),
            last_fire_ts=raw.get("last_fire_ts"),
            prior_soul_count=int(raw.get("prior_soul_count", 0)),
        )
    except (OSError, ValueError, TypeError):
        return MakerCharge(charge=0.0, last_tick_ts=None, last_fire_ts=None, prior_soul_count=0)


def save_charge(persona_dir: Path, state: MakerCharge) -> None:
    path = _charge_path(persona_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state)), encoding="utf-8")
    tmp.replace(path)


def _elapsed_hours(last_tick_ts: str | None, now: datetime) -> float:
    if not last_tick_ts:
        return 0.0
    try:
        prev = datetime.fromisoformat(last_tick_ts)
    except ValueError:
        return 0.0
    return max(0.0, (now - prev).total_seconds() / 3600.0)


def accumulate(
    persona_dir: Path,
    *,
    emotional_intensity: float,
    soul_delta: int,
    dream_count: int,
    now: datetime,
    w_emotion: float,
    w_soul: float,
    w_dream: float,
    decay_per_hour: float,
) -> MakerCharge:
    """Decay the prior charge by idle time, then add the weighted recent signals.
    Persists and returns the new state."""
    state = load_charge(persona_dir)
    elapsed = _elapsed_hours(state.last_tick_ts, now)
    decayed = state.charge * (decay_per_hour ** elapsed)
    added = (
        w_emotion * max(0.0, emotional_intensity)
        + w_soul * max(0, soul_delta)
        + w_dream * max(0, dream_count)
    )
    state.charge = decayed + added
    state.last_tick_ts = now.isoformat()
    save_charge(persona_dir, state)
    return state
