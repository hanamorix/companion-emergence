"""prompt.py — render felt-time as a compact ambient context block.

Spec §4 prompt context render. ≤150-token budget. Honest about cold
start ("too new to have texture yet"). Anchor labels come from each
source's existing slug — no LLM call, no generated poetry.
"""

from __future__ import annotations

from datetime import UTC, datetime

from brain.felt_time.state import FeltTimeState, HorizonBucket

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


_HORIZON_CONTRAST_THRESHOLD = 0.30
_HORIZON_CURRENT_DAYS: dict[str, float] = {"week": 7.0, "month": 30.0}


def _horizon_contrast_line(
    horizon_pressure: dict[str, HorizonBucket],
    *,
    now: datetime,
) -> str | None:
    """Return a qualitative contrast line or None if nothing meaningful."""
    candidates: list[tuple[float, str]] = []

    for key in ("week", "month"):
        bucket = horizon_pressure.get(key)
        if bucket is None:
            continue
        prev_turns = bucket.prev_counters.chat_turns
        if prev_turns == 0:
            continue

        try:
            if bucket.period_start_ts:
                start = datetime.fromisoformat(bucket.period_start_ts)
                current_days = max((now - start).total_seconds() / 86400.0, 1.0)
            else:
                current_days = _HORIZON_CURRENT_DAYS[key]
        except ValueError:
            current_days = _HORIZON_CURRENT_DAYS[key]

        prev_days = _HORIZON_CURRENT_DAYS[key]
        current_rate = bucket.counters.chat_turns / current_days
        prev_rate = prev_turns / prev_days

        if prev_rate == 0:
            continue

        ratio = current_rate / prev_rate
        if ratio > 1 + _HORIZON_CONTRAST_THRESHOLD:
            candidates.append((ratio - 1.0, "this stretch has been denser than it was recently"))
        elif ratio < 1 - _HORIZON_CONTRAST_THRESHOLD:
            candidates.append((1.0 - ratio, "things have been quieter lately than they were"))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _open_arc_lines(arc_anchors: list, *, now: datetime) -> list[str]:
    """Return rendered lines for open arc threads and a recently-closed summary."""
    if not arc_anchors:
        return []

    # For each arc title, the most recent event determines its state.
    latest_per_title: dict[str, object] = {}
    for a in arc_anchors:
        existing = latest_per_title.get(a.label)
        if existing is None or a.ts > existing.ts:
            latest_per_title[a.label] = a

    open_arcs = sorted(
        [a for a in latest_per_title.values() if a.event_type == "arc_opened"],
        key=lambda a: a.ts,
        reverse=True,
    )[:2]

    lines: list[str] = []
    if open_arcs:
        parts = []
        for a in open_arcs:
            h = _hours_ago(a.ts, now=now)
            ago = f"{h:.0f} hours ago" if h < 72.0 else f"{h / 24.0:.0f} days ago"
            parts.append(f'"{_truncate_label(a.label)}" since {ago}')
        if len(open_arcs) == 1:
            lines.append(f"  open thread: {parts[0]}")
        else:
            lines.append(f"  open threads ({len(open_arcs)}): {'; '.join(parts)}")

    # Recently-closed (within 30 days).
    closed_recent = [
        a
        for a in latest_per_title.values()
        if a.event_type == "arc_closed" and _hours_ago(a.ts, now=now) <= 30 * 24.0
    ]
    if closed_recent:
        n = len(closed_recent)
        lines.append(f"  {'one thread' if n == 1 else f'{n} threads'} closed recently")

    return lines


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

    # Horizon contrast — qualitative cross-scale temporal texture.
    if state.horizon_pressure:
        _now = now or datetime.now(UTC)
        contrast = _horizon_contrast_line(state.horizon_pressure, now=_now)
        if contrast:
            lines.append(f"  {contrast}")

    # Arc threads — open and recently-closed.
    if state.arc_anchors:
        _now_for_arcs = now or datetime.now(UTC)
        lines.extend(_open_arc_lines(state.arc_anchors, now=_now_for_arcs))

    return "\n".join(lines)
