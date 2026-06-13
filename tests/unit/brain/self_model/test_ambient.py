"""Self-model ambient block — render only when a gap is open (R-F1).

The block is hedged ("you've been moving like X, but your felt read says Y").
It renders ONLY when state.current_gap is open/acknowledged AND has a note or
non-trivial per_channel; otherwise None so the prompt carries no bloat.
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


def test_open_gap_with_only_subfloor_channels_returns_none():
    # deltas below the surfacing floor are noise — omit the block
    state = SelfModelState(
        current_gap=_open_gap(per_channel={"grief": 0.1}, magnitude=0.1, note=None)
    )
    assert render_block(state) is None
