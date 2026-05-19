"""Ambient 'current arc' block for chat prompt builder.

Spec §6. ≤80 token budget. Renders the most-recently-extended open arc
as "current"; lists one-line digests of up to two other open arcs;
overflow becomes `+ N more`.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.narrative_memory.arc import Arc
from brain.narrative_memory.state import load_or_recover

# Cap of "also open" digests rendered before overflow
_ALSO_OPEN_CAP = 2


def render_current_arc_block(persona_dir: Path) -> str:
    """Returns the arcs block, or 'arcs: still forming …' on cold start.

    No-cost read of arcs_state.json; never falls through to log replay.
    """
    state = load_or_recover(persona_dir)
    if not state.open:
        return "arcs\n  still forming — no anchors have seeded threads yet"

    sorted_arcs = sorted(
        state.open.values(),
        key=lambda a: a.last_extended_at_iso,
        reverse=True,
    )
    current = sorted_arcs[0]
    others = sorted_arcs[1:]

    lines = ["arcs"]
    lines.append(f"  current: \"{current.title}\" — {_describe_arc(current)}")

    if others:
        shown = others[:_ALSO_OPEN_CAP]
        rest = others[_ALSO_OPEN_CAP:]
        digests = ", ".join(f"\"{a.title}\" ({len(a.members)} memories)" for a in shown)
        suffix = f", + {len(rest)} more" if rest else ""
        lines.append(f"  also open: {digests}{suffix}")

    return "\n".join(lines)


def _describe_arc(arc: Arc) -> str:
    """e.g. 'opened 4 lived-days ago, 8 memories, last extended 3 lived-hours ago'.

    Wall-clock approximation in v1 — the prompt renderer does NOT have
    FeltTimeState available, so 'lived-X-ago' is rendered from the
    wall-clock delta. This matches the pragmatism the ambient block needs:
    it's a hint, not a precise read-out.
    """
    opened_hrs = _hours_ago(arc.opened_at_iso)
    extended_hrs = _hours_ago(arc.last_extended_at_iso)
    return (
        f"opened {_render_hours(opened_hrs)} ago, "
        f"{len(arc.members)} memories, "
        f"last extended {_render_hours(extended_hrs)} ago"
    )


def _hours_ago(iso: str) -> float:
    try:
        ts = datetime.fromisoformat(iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(UTC) - ts).total_seconds() / 3600.0)


def _render_hours(hours: float) -> str:
    if hours >= 48.0:
        return f"{hours / 24.0:.0f} lived-days"
    return f"{hours:.0f} lived-hours"
