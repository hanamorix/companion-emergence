"""MCP tool callables for felt_time_now + pressure_since.

These are pure functions that the dispatcher (brain/tools/dispatch.py)
calls. Read-only; no LLM cost; no side effects.

Spec §4 MCP tools.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.felt_time import FeltTime
from brain.felt_time.state import ANCHOR_TYPES


def _serialise_state(state) -> dict[str, Any]:
    now = datetime.now(UTC)
    anchors = {}
    for k, a in state.anchors.items():
        anchor_dt = datetime.fromisoformat(a.ts)
        anchors[k] = {
            "ts": a.ts,
            "label": a.label,
            "hours_ago": round((now - anchor_dt).total_seconds() / 3600.0, 1),
        }
    return {
        "lived_age_hours": round(state.lived_age_hours, 2),
        "anchors": anchors,
        "pressure_since_last_anchor": asdict(state.pressure),
    }


def felt_time_now(*, persona_dir: Path) -> dict[str, Any]:
    """Return the full felt-time state.

    Use this when you want to introspect: "how long has it actually
    been since..." or "what's my current sense of accumulated time?"
    Cold start returns lived_age_hours = 0.0 and anchors = {}.
    """
    ft = FeltTime(persona_dir=persona_dir)
    return _serialise_state(ft.get_state())


def pressure_since(*, arguments: dict[str, Any], persona_dir: Path) -> dict[str, Any]:
    """Return the pressure vector since the latest anchor of a given type.

    anchor_type ∈ {"dream", "growth", "soul", "weather_shift"}.
    """
    anchor_type = arguments.get("anchor_type")
    if anchor_type not in ANCHOR_TYPES:
        raise ValueError(f"anchor_type must be one of {ANCHOR_TYPES}, got {anchor_type!r}")
    ft = FeltTime(persona_dir=persona_dir)
    return asdict(ft.get_state().pressure)
