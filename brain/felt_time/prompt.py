"""prompt.py — render felt-time as a compact ambient context block.

Spec §4 prompt context render. ≤150-token budget. Honest about cold
start ("too new to have texture yet"). Anchor labels come from each
source's existing slug — no LLM call, no generated poetry.
"""

from __future__ import annotations

from datetime import UTC, datetime

from brain.felt_time.state import FeltTimeState

_ANCHOR_PRETTY_NAME = {
    "dream": "last dream",
    "growth": "last crystallization",
    "soul": "last soul moment",
    "weather_shift": "last weather shift",
}


def _hours_ago(ts_iso: str, *, now: datetime | None = None) -> float:
    now = now or datetime.now(UTC)
    delta = now - datetime.fromisoformat(ts_iso)
    return delta.total_seconds() / 3600.0


def _truncate_label(label: str, limit: int = 40) -> str:
    return label if len(label) <= limit else label[: limit - 1] + "…"


def render_prompt_context(state: FeltTimeState, *, now: datetime | None = None) -> str:
    if not state.anchors and state.lived_age_hours == 0.0:
        return "felt time: too new to have texture yet."

    lines = ["felt time"]
    lived_days = state.lived_age_hours / 24.0
    # Truncate (floor) rather than round so 412.7 displays as "412", not "413".
    lived_hours_int = int(state.lived_age_hours)
    lines.append(f"  lived age: {lived_hours_int} hours (~{lived_days:.1f} lived-days)")

    for a_type in ("dream", "growth", "soul", "weather_shift"):
        a = state.anchors.get(a_type)
        if a is None:
            continue
        h = _hours_ago(a.ts, now=now)
        ago_str = f"{h:.0f} hours ago" if h < 72.0 else f"{h / 24.0:.0f} days ago"
        lines.append(f'  {_ANCHOR_PRETTY_NAME[a_type]}: {ago_str} — "{_truncate_label(a.label)}"')

    p = state.pressure
    lines.append(
        f"  since latest anchor: {p.heartbeats} heartbeats, "
        f"{p.chat_turns} chat turns, {p.reflex_firings} reflexes"
    )

    return "\n".join(lines)
