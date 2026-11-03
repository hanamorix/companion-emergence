"""Pins for the self-model gap's 0.5 floor and candidate set, before the readers move to the engine.

A channel's gap is kept when ``short - long`` is at least the floor (so exactly the floor stays and a hair below
goes), and a channel present only in the long read is still a candidate. The ambient block names a channel at
exactly its own floor. Deleted with the old branch when the reader switch-over lands.
"""

from __future__ import annotations

import pytest

from brain.self_model.ambient import render_block
from brain.self_model.derived import DerivedRead
from brain.self_model.gap import Gap, compute_gap
from brain.self_model.state import SelfModelState

_REASON = (
    "pins the old self-model gap and block branch (0.5 floor with >=, long-only channel is a candidate); "
    "passes on the current code by design and is deleted with the branch at phase C"
)


def _read(channels: dict[str, float]) -> DerivedRead:
    return DerivedRead(channels=channels, unnamed_pressure=0.0, sources={})


def _open_state(per_channel: dict[str, float]) -> SelfModelState:
    gap = Gap(per_channel=per_channel, magnitude=sum(abs(v) for v in per_channel.values()), unnamed_pressure=0.0)
    return SelfModelState(current_gap=gap)


@pytest.mark.no_base_failure(reason=_REASON)
def test_gap_keeps_a_delta_of_exactly_the_floor_and_drops_one_just_below():
    # 2**-10 is exact in binary, so the two deltas are exactly 0.5 and 0.5 - 2**-10.
    at_floor = compute_gap(_read({"joy": 1.0}), _read({"joy": 0.5}))
    just_below = compute_gap(_read({"joy": 1.0}), _read({"joy": 0.5009765625}))

    assert at_floor.per_channel == {"joy": 0.5}
    assert just_below.per_channel == {}


@pytest.mark.no_base_failure(reason=_REASON)
def test_gap_treats_a_channel_only_in_the_long_read_as_a_candidate():
    gap = compute_gap(_read({}), _read({"joy": 3.0}))

    assert gap.per_channel == {"joy": -3.0}


@pytest.mark.no_base_failure(reason=_REASON)
def test_block_names_a_channel_at_exactly_the_floor_and_not_one_just_below():
    block = render_block(_open_state({"grief": 0.5}))
    below = render_block(_open_state({"grief": 0.4990234375}))

    assert block is not None
    assert "grief" in block
    assert below is None
