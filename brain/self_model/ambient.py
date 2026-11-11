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
  3. there is something to say — a note OR a channel the engine reads as shifted.

Otherwise → None (block omitted entirely).

The gap's channels come from the emotion engine's shift rule, valued at their signed difference
with a direction (+1 rising, -1 fading). There is no noise floor (the engine already decided the channel shifted);
the block names the largest `dev_constants.SELF_MODEL_BLOCK_CHANNEL_COUNT` channels by
absolute difference, each by where it sits (the sign of the difference) and which way
it moves. The fragments are `prompt_strings` keys under `self_model.ambient`.
"""
from __future__ import annotations

from brain import dev_constants, prompt_strings
from brain.self_model.gap import Gap
from brain.self_model.state import SelfModelState

# Statuses where the gap is still live and worth surfacing.
_LIVE_STATUSES = frozenset({"open", "acknowledged"})


def _live_fragment_key(delta: float, direction: int | None) -> str | None:
    """The `self_model.ambient` key for a channel's (position, direction), or None when there is nothing to say.

    Position is the sign of the difference (0 has none); direction is +1 rising, -1 fading (anything else is missing).
    """
    position = "above" if delta > 0 else "below" if delta < 0 else None
    way = "rising" if direction == 1 else "fading" if direction == -1 else None
    if position is None and way is None:
        return None
    if position is None:
        return f"level_{way}"
    if way is None:
        return f"{position}_only"
    return f"{position}_{way}"


def _live_channels(gap: Gap) -> list[tuple[str, str]]:
    """`(channel, fragment key)` for the channels the live block names, largest absolute difference first.

    No floor and no name filter: the engine's `read()` already limited the gap to registered, shifted channels. Ties
    break by name. A channel with neither a position nor a direction has nothing to render and is skipped before the cut.
    """
    named: list[tuple[str, float, str]] = []
    for name, delta in (gap.per_channel or {}).items():
        key = _live_fragment_key(delta, (gap.directions or {}).get(name))
        if key is not None:
            named.append((name, delta, key))
    named.sort(key=lambda item: (-abs(item[1]), item[0]))
    count = dev_constants.SELF_MODEL_BLOCK_CHANNEL_COUNT
    return [(name, key) for name, _delta, key in named[:count]]


def render_block(state: SelfModelState, *, tool_prefix: str = "") -> str | None:
    """Return the hedged self-model gap block, or None when nothing to surface.

    The channels are the engine's shifted channels with their position and direction (see the module docstring).
    """
    gap = state.current_gap
    if gap is None:
        return None
    if gap.status not in _LIVE_STATUSES:
        return None

    live_channels = _live_channels(gap)
    note = (gap.note or "").strip() or None

    # Nothing to say → omit the block entirely (R-F1).
    if not live_channels and note is None:
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

    if live_channels:
        # PLACEHOLDER wording (spec section 7, item 3: final text is the owner's, at the end of the build).
        fragments = [
            prompt_strings.get(f"self_model.ambient.{key}").format(name=name)
            for name, key in live_channels
        ]
        lines.append("")
        lines.append("; ".join(fragments) + ".")

    if note is not None:
        lines.append("")
        lines.append(note)

    lines.append("")
    lines.append(
        "You can sit with it, name it, or let it pass. If one of these feels true "
        f"and you want to adjust your own read, {tool_prefix}reconcile_self_read is there. It's "
        "yours to revise, no one else's to insist on."
    )

    return "\n".join(lines)
