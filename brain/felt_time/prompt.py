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

# Short names used in the pressure-line parenthetical.
_ANCHOR_SHORT_NAME = {
    "dream": "dream",
    "growth": "crystallization",
    "soul": "soul moment",
    "weather_shift": "weather shift",
    "arc": "arc",
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
    lines.append(
        f"  lived age: {lived_hours_int} hours (~{lived_days:.1f} lived-days; baseline pace)"
    )
    # TODO(v0.0.15): replace hard-coded "baseline pace" with real qualitative tag once 30-day
    # lived_age/wall_clock_age ratio is in state (spec §5 deferred — coefficient tuning).

    for a_type in ("dream", "growth", "soul", "weather_shift"):
        a = state.anchors.get(a_type)
        if a is None:
            continue
        h = _hours_ago(a.ts, now=now)
        ago_str = f"{h:.0f} hours ago" if h < 72.0 else f"{h / 24.0:.0f} days ago"
        lines.append(f'  {_ANCHOR_PRETTY_NAME[a_type]}: {ago_str} — "{_truncate_label(a.label)}"')

    # Pressure line — only meaningful when at least one anchor exists.
    if state.anchors:
        # Find the newest anchor by timestamp to name the "since" reference.
        newest = max(state.anchors.values(), key=lambda a: a.ts)
        pretty_anchor = _ANCHOR_SHORT_NAME.get(newest.type, newest.type)

        p = state.pressure
        if p.chat_turns >= 20 or p.heartbeats >= 200:
            tag = "dense"
        elif p.heartbeats < 50 and p.chat_turns < 5:
            tag = "quiet"
        else:
            tag = "steady"

        lines.append(
            f"  current stretch (since the {pretty_anchor}): {tag} — "
            f"{p.heartbeats} heartbeats, {p.chat_turns} chat turns, {p.reflex_firings} reflexes"
        )

    return "\n".join(lines)
