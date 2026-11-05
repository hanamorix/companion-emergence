"""`fold_guarded` (Engine 4, #579): a fold step that raises leaves the state exactly as it was, on every branch.

The restore is what keeps a half-folded event from being saved. The event-content branch (`_SKIPPABLE`) is the common one, and
nothing else in the suite fails if its restore is removed, so each type it names is pinned here, with the neighbour branch (any
other exception) beside it.
"""

from __future__ import annotations

import copy
import logging
from datetime import UTC, datetime

import pytest

from brain.emotion import engine_utils as eu

LOGGER = "brain.emotion.engine_utils"
SAVED = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _state() -> eu.EngineState:
    return eu.EngineState(
        saved_at=SAVED,
        last_update=SAVED,
        active_day="2026-06-01",
        persona_start_at=SAVED,
        repeats={"m1": eu.RepeatEntry(SAVED, 1)},
        emotions={
            "joy": eu.EmotionState(
                short=eu.Average(4.0, 1.0), long=eu.Average(4.0, 1.0), last_entry_day="2026-06-01"
            )
        },
        queue_line=3,
    )


def _half_fold_then(error: BaseException, state: eu.EngineState):
    """A step that changes top-level fields, the nested averages, the repeat list and the snapshot list, and then raises `error`."""

    def step() -> None:
        state.queue_line = 99
        state.active_day = "2026-06-09"
        joy = state.emotions["joy"]
        joy.short.avg, joy.long.weight = 9.5, 7.0
        joy.snapshot.append(eu.SnapshotDay("2026-06-08", 1.0, 1.0, True, None))
        state.repeats["m2"] = eu.RepeatEntry(SAVED, 2)
        state.emotions["grief"] = eu.EmotionState(
            short=eu.Average(1.0, 1.0), long=eu.Average(1.0, 1.0), last_entry_day=None
        )
        raise error

    return step


@pytest.mark.no_base_failure(
    reason="the restore already exists on the base; this pins it against removal"
)
@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad value"),
        OverflowError("too big"),
        OSError("no such time"),
        TypeError("wrong type"),
    ],
)
def test_a_skippable_error_puts_the_state_back_exactly_in_place_and_returns_false(
    error: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    state = _state()
    before = copy.deepcopy(state)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert (
            eu.fold_guarded(state, _half_fold_then(error, state), "a test event", key="t") is False
        )
    assert state == before
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        f"emotion engine: a test event skipped ({type(error).__name__}: {error})"
    ]


@pytest.mark.no_base_failure(
    reason="the restore already exists on the base; this pins the neighbour branch beside the skippable one"
)
def test_an_unexpected_error_also_puts_the_state_back_and_returns_false() -> None:
    state = _state()
    before = copy.deepcopy(state)
    assert (
        eu.fold_guarded(
            state, _half_fold_then(RuntimeError("a bug"), state), "a test event", key="t-unexpected"
        )
        is False
    )
    assert state == before


@pytest.mark.no_base_failure(
    reason="the control for the two restore tests: a step that does not raise keeps its changes; passes on the base by design"
)
def test_a_step_that_succeeds_keeps_its_changes_and_returns_true() -> None:
    state = _state()

    def step() -> None:
        state.queue_line = 99

    assert eu.fold_guarded(state, step, "a test event", key="t-ok") is True
    assert state.queue_line == 99
