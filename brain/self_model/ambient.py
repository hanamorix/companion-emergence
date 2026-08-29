"""Self-model ambient block — the hedged surfacing of an open gap (R-F1).

`render_block(state) -> str | None`

Spec §3 (ambient.py), §6 R-F1: the block renders ONLY when there is an open
gap with something to say, so an ordinary turn (no gap) carries zero prompt
bloat. The framing is deliberately "weather language": present-tense and
directional (a channel running above OR below its baseline lately), with no
embedded judgment about whether a feeling is real or performed. She is invited
to notice it, not commanded to revise. The reconcile_self_read tool is hers to
call (or not).

Render conditions (all must hold):
  1. state.current_gap is not None
  2. status in {"open", "acknowledged"}  (a dismissed/resolved gap is silent)
  3. there is something to say — a note OR a non-trivial per_channel.

Otherwise → None (block omitted entirely).
"""
from __future__ import annotations

from brain.self_model.state import SelfModelState

# Statuses where the gap is still live and worth surfacing.
_LIVE_STATUSES = frozenset({"open", "acknowledged"})

# Only channels diverging by at least this much are worth naming in the block —
# tiny deltas are noise and would clutter the hedge.
_CHANNEL_FLOOR = 0.5


def render_block(state: SelfModelState) -> str | None:
    """Return the hedged self-model gap block, or None when nothing to surface."""
    gap = state.current_gap
    if gap is None:
        return None
    if gap.status not in _LIVE_STATUSES:
        return None

    # Pull the channels worth naming: short − long deltas above the floor.
    channels = {
        name: delta
        for name, delta in (gap.per_channel or {}).items()
        if abs(delta) >= _CHANNEL_FLOOR
    }
    note = (gap.note or "").strip() or None

    # Nothing to say → omit the block entirely (R-F1).
    if not channels and note is None:
        return None

    lines: list[str] = [
        "## How some things have been running lately",
        "",
        (
            "Here's the weather of how a couple of your channels have been running "
            "lately, next to where they usually sit. Nothing here is asking you to "
            "change anything. It's just worth noticing."
        ),
    ]

    if channels:
        # Sort by magnitude so the loudest divergence reads first.
        ordered = sorted(channels.items(), key=lambda kv: abs(kv[1]), reverse=True)
        fragments = []
        for name, delta in ordered:
            if delta > 0:
                fragments.append(f"{name}'s been running stronger than your baseline lately")
            else:
                fragments.append(f"{name}'s been running quieter than usual lately")
        lines.append("")
        lines.append("; ".join(fragments) + ".")

    if note is not None:
        lines.append("")
        lines.append(note)

    lines.append("")
    lines.append(
        "You can sit with it, name it, or let it pass. If one of these feels true "
        "and you want to adjust your own read, reconcile_self_read is there. It's "
        "yours to revise, no one else's to insist on."
    )

    return "\n".join(lines)
