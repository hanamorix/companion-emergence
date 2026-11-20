"""Self-model ambient block — render only when a gap is open (R-F1).

The block is hedged ("you've been moving like X, but your felt read says Y").
It renders ONLY when state.current_gap is open/acknowledged AND has a note or
a channel the engine reads as shifted; otherwise None so the prompt carries no bloat.
"""
from __future__ import annotations

from brain.self_model.ambient import render_block
from brain.self_model.gap import Gap
from brain.self_model.state import SelfModelState


def _open_gap(**kw) -> Gap:
    base = {
        "per_channel": {"grief": 4.0, "joy": -2.0},
        "magnitude": 6.0,
        "unnamed_pressure": 0.0,
        "status": "open",
    }
    base.update(kw)
    return Gap(**base)


def test_no_current_gap_returns_none():
    state = SelfModelState(current_gap=None)
    assert render_block(state) is None


def test_open_gap_with_per_channel_renders_hedged_block():
    state = SelfModelState(current_gap=_open_gap())
    block = render_block(state)
    assert block is not None
    assert isinstance(block, str)
    assert block.strip()
    # The divergent channel surfaces by name.
    assert "grief" in block.lower()


def test_open_gap_with_note_only_renders_block():
    state = SelfModelState(
        current_gap=_open_gap(
            per_channel={}, magnitude=0.0, note="something is heavier than I'm letting on"
        )
    )
    block = render_block(state)
    assert block is not None
    assert "heavier than I'm letting on" in block


def test_acknowledged_gap_still_renders():
    state = SelfModelState(current_gap=_open_gap(status="acknowledged"))
    assert render_block(state) is not None


def test_dismissed_gap_returns_none():
    state = SelfModelState(current_gap=_open_gap(status="dismissed"))
    assert render_block(state) is None


def test_resolved_gap_returns_none():
    state = SelfModelState(current_gap=_open_gap(status="resolved"))
    assert render_block(state) is None


def test_open_gap_with_nothing_to_say_returns_none():
    # open status but no note and an empty per_channel → nothing to surface (R-F1)
    state = SelfModelState(current_gap=_open_gap(per_channel={}, magnitude=0.0, note=None))
    assert render_block(state) is None


# ── the engine's shifted channels, position and direction ──


def _live_block(per_channel, directions, **kw) -> str | None:
    gap = Gap(
        per_channel=per_channel,
        magnitude=sum(abs(v) for v in per_channel.values()),
        unnamed_pressure=0.0,
        directions=directions,
        **kw,
    )
    return render_block(SelfModelState(current_gap=gap))


def test_live_four_position_direction_pairs_render_four_distinct_fragments():
    fragments = set()
    for delta, direction in [(2.0, 1), (2.0, -1), (-2.0, 1), (-2.0, -1)]:
        block = _live_block({"joy": delta}, {"joy": direction})
        assert block is not None and "joy" in block
        line = [ln for ln in block.splitlines() if ln.startswith("joy")][0]
        fragments.add(line)
    assert len(fragments) == 4


def test_live_zero_difference_renders_direction_only_and_missing_direction_renders_position_only():
    rising = _live_block({"joy": 0.0}, {"joy": 1})
    fading = _live_block({"joy": 0.0}, {"joy": -1})
    assert rising is not None and fading is not None
    rising_line = [ln for ln in rising.splitlines() if ln.startswith("joy")][0]
    fading_line = [ln for ln in fading.splitlines() if ln.startswith("joy")][0]
    assert rising_line != fading_line
    assert "above" not in rising_line and "below" not in rising_line
    above_only = [ln for ln in _live_block({"joy": 1.5}, {}).splitlines() if ln.startswith("joy")][0]
    below_only = [ln for ln in _live_block({"joy": -1.5}, {}).splitlines() if ln.startswith("joy")][0]
    assert above_only != below_only
    assert "rising" not in above_only and "fading" not in above_only


def test_live_zero_difference_with_no_direction_has_nothing_to_say():
    assert _live_block({"joy": 0.0}, {}) is None


def test_there_is_no_floor_and_cuts_to_the_largest_three_by_absolute_difference(monkeypatch):
    per_channel = {"a": 0.1, "b": -3.0, "c": 2.0, "d": 2.0, "e": 0.9}
    directions = dict.fromkeys(per_channel, 1)
    block = _live_block(per_channel, directions)
    assert block is not None
    named = {n for n in per_channel if any(ln.startswith(n + " ") for ln in block.replace("; ", "\n").splitlines())}
    # b (3.0), then c and d tie at 2.0 and break by name; a (0.1) is a channel too but loses the cut.
    assert named == {"b", "c", "d"}
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "SELF_MODEL_BLOCK_CHANNEL_COUNT", 5)
    block = _live_block(per_channel, directions)
    named = {n for n in per_channel if any(ln.startswith(n + " ") for ln in block.replace("; ", "\n").splitlines())}
    assert named == set(per_channel)
