"""Check 25 (Engine 4, #579): end-to-end simulations of the shift rule through the real drain, `close_days`, `shift_for` and `read()`.

Hand-built histories in a tmp persona directory, one queued showing per day (weight 1, a new memory id each time, so no repeat
factor) and one `note_active` per active day. Nothing here unit-tests `run_tracker`: the flags are what the whole path gives.
Every expected flag is worked out by hand FIRST and written as a literal; the derivation is in the comment above each case.

How to read the derivations. A day "closes" when the first drain of a later day runs, so the flag read after the drain of
day d is the flag of the snapshot up to day d - 1 (written `closed(c)` below for "after day c closed"). The tracker only looks
at the SIGN of each daily move of an average: an average moves toward the day's value, so a value above the average is an up
move, below it a down move, and exactly equal a zero move (the delta form makes a steady feeling's moves exactly zero). A
single day against a run (a "blip", a zero move or an off day included) is counted into the run's length; a second one in a
row ends it, and the two reverse days are the first two days of the next run (a zero among them starts none). A feeling reads
as shifted when its short run in progress is LONGER than the longest finished run of the long average in the last 30 days
(equal does not count), the start gate is open (the last snapshot day ends 30 days or more after the persona's start), and a
fall was felt on more of its days than not.

Days are numbered from 0; `D(n)` is local noon of day n. A long run is finished by two days against it.
"""

from __future__ import annotations

import copy
import logging
from datetime import date, timedelta
from pathlib import Path

import pytest

from brain.emotion import engine_files, engine_rebuild
from brain.emotion.engine_utils import (
    Average,
    EmotionState,
    EngineState,
    Event,
    RunRecord,
    SnapshotDay,
    close_days,
)
from brain.memory.store import MemoryStore
from tests.unit.brain.emotion._engine_support import (
    DAY0,
    change,
    install,
    local_day,
    local_noon,
    make_persona,
    queue,
    row,
    save,
    save_as_backup,
    showing,
    state_of,
)


def D(n: int):
    return local_noon(local_day(n))


def _day(n: int) -> str:
    return local_day(n).isoformat()


class Sim:
    """One persona, one engine; `run` plays a timeline day by day and returns the shift after each closed day."""

    def __init__(self, root: Path, *, start_day: int = -60, state: EngineState | None = None) -> None:
        self.persona = make_persona(root / "persona")
        self.engine = install(self.persona, D(400))
        self.counter = 0
        if state is None:
            state = EngineState(saved_at=D(start_day), persona_start_at=D(start_day))
        save(self.persona, state)

    def entry(self, day: int, emotion: str, value: float, hours: float = 0.0) -> None:
        self.counter += 1
        queue(self.persona, showing(f"sim-{self.counter}", D(day) + timedelta(hours=hours), {emotion: value}))

    def drain(self, day: int, hours: float = 1.0) -> None:
        self.engine.note_active(D(day) + timedelta(hours=hours))

    def shift(self, emotion: str = "joy") -> int:
        reading = self.engine.read().get(emotion)
        return 0 if reading is None else reading.shift

    def run(
        self,
        values: dict[int, float],
        *,
        first: int,
        last: int,
        off: frozenset[int] = frozenset(),
        emotion: str = "joy",
    ) -> dict[int, int]:
        """Play every day from the earliest of `values` (and `first`) to `last` plus one more drain; `{c: shift after day c
        closed}` for each c from `first` to `last`."""
        seen: dict[int, int] = {}
        for day in range(min([first, *values]), last + 2):
            if day in off:
                continue
            if day in values:
                self.entry(day, emotion, values[day])
            self.drain(day)
            snapshot = state_of(self.persona).emotions.get(emotion)
            if snapshot is not None and snapshot.snapshot:
                closed = (date.fromisoformat(snapshot.snapshot[-1].day) - DAY0).days
                seen[closed] = self.shift(emotion)
        return {c: flag for c, flag in seen.items() if first <= c <= last}


def _steady(until: int, value: float) -> dict[int, float]:
    return dict.fromkeys(range(until + 1), value)


@pytest.fixture
def sim(tmp_path: Path) -> Sim:
    return Sim(tmp_path)


# --- (a) steady feeling ---------------------------------------------------------------------------------------------


def test_a_steady_feeling_is_never_flagged_and_both_averages_move_by_exactly_zero(tmp_path: Path) -> None:
    """(a) One entry a day at 6.0, 6.3, 7.7 and 10.0 for 120 days, 24 h apart with a jitter of at most 2 h around noon.

    Fed the value it already holds, an average does not move at all (delta form): every daily move after the first day of a
    feeling is exactly 0.0, so no run ever starts and nothing is flagged.
    """
    simulation = Sim(tmp_path)
    jitter = [0.0, 1.5, -2.0, 0.5, 2.0, -1.0, 1.0]
    values = {"joy": 6.0, "grief": 6.3, "love": 7.7, "desire": 10.0}
    for day in range(120):
        for number, (name, value) in enumerate(values.items()):
            simulation.entry(day, name, value, hours=jitter[(day + number) % len(jitter)])
    simulation.drain(120)  # one drain folds the whole history in time order
    state = state_of(simulation.persona)
    for name in values:
        days = state.emotions[name].snapshot
        assert len(days) == 30
        assert all(a.short == b.short and a.long == b.long for a, b in zip(days, days[1:], strict=False)), name
        assert days[-1].short == values[name] and days[-1].long == values[name]
    assert {name: reading.shift for name, reading in simulation.engine.read().items()} == dict.fromkeys(values, 0)


def test_a_steady_feeling_read_every_day_is_never_flagged(sim: Sim) -> None:
    flags = sim.run(_steady(45, 6.3), first=0, last=44)
    assert set(flags.values()) == {0} and len(flags) == 45


# --- (b) steady climb after a finished long run ------------------------------------------------------------------------


def _pulse_then_climb() -> dict[int, float]:
    """Days 0-32 flat 2.0; 33-36 at 8.0; 37-38 at 0.0; 39-46 at 9.0; 47-48 at 0.0."""
    values = _steady(32, 2.0)
    values.update(dict.fromkeys(range(33, 37), 8.0))
    values.update(dict.fromkeys(range(37, 39), 0.0))
    values.update(dict.fromkeys(range(39, 47), 9.0))
    values.update(dict.fromkeys(range(47, 49), 0.0))
    return values


def test_a_climb_is_flagged_at_the_close_where_its_run_first_exceeds_the_longest_finished_long_run_and_cleared_after_two_days_against(
    sim: Sim,
) -> None:
    """(b) The long average rises on days 33-36 (8.0 against a flat 2.0: 4 up days) and falls on 37-38 (two against: the run is
    finished, length 4). The short average does the same, so its current run on day 38 is a DOWN run (37, 38: length 2).
    Day 39 (9.0, above the short average) is a blip against that down run (length 3); day 40 is the second up day, which ends
    the down run and makes days 39 and 40 the first two days of an up run (length 2). Days 41, 42, 43 make it 3, 4, 5.
    Bar 4: equal does not count, so closed(42) is still 0 and closed(43) is the first flag, +1. It stays through 46 and through
    day 47 (the first day against, a blip), and day 48 (the second) ends the run: closed(48) is 0 again.
    """
    flags = sim.run(_pulse_then_climb(), first=33, last=48)
    assert {c: flag for c, flag in flags.items() if c <= 42} == dict.fromkeys(range(33, 43), 0)
    assert [flags[c] for c in range(43, 48)] == [1, 1, 1, 1, 1]
    assert flags[48] == 0


# --- (c) zig-zag ------------------------------------------------------------------------------------------------------


def test_a_zig_zag_is_one_run_whose_length_grows_by_one_a_day(sim: Sim) -> None:
    """(c) After the same finished long run (bar 4, finished on day 38), the value alternates 9.0 (odd offset) and 0.0.

    The short average's down run (37, 38: length 2) meets day 39 (9.0, up: a blip, length 3); day 40 (0.0, down) is with the
    run again, which clears the blip (length 4); day 41 is a blip again (length 5), and so on: ONE run whose length grows by one
    every day, never two against in a row. Bar 4: closed(40) is 0 (length 4: equal), closed(41) is the first flag (5), a fall
    felt on every one of its days, so -1; and it stays -1 on every later day while the bar is in the window.
    """
    values = _steady(32, 2.0)
    values.update(dict.fromkeys(range(33, 37), 8.0))
    values.update(dict.fromkeys(range(37, 39), 0.0))
    for day in range(39, 60):
        values[day] = 9.0 if (day - 39) % 2 == 0 else 0.0
    flags = sim.run(values, first=38, last=56)
    assert [flags[c] for c in range(38, 41)] == [0, 0, 0]
    assert [flags[c] for c in range(41, 57)] == [-1] * 16


# --- (d) stuck at 10 and at 0 --------------------------------------------------------------------------------------------


def _long_run_bar_snapshot(*, short_days: list[float]) -> list[SnapshotDay]:
    """Snapshot days 0 to 27 of a feeling: the long average has ONE finished run of length 4 (up on days 10-13, down on 14-15,
    flat otherwise); the short average is flat 5.0 until day 21 and then follows `short_days` (days 22 to 27). Every day felt."""
    long_values = [3.0] * 10 + [3.1, 3.2, 3.3, 3.4] + [3.3, 3.2] + [3.2] * 12
    short_values = [5.0] * 22 + short_days
    return [SnapshotDay(_day(n), short_values[n], long_values[n], True, None) for n in range(28)]


def _state_with_snapshot(snapshot: list[SnapshotDay], *, start_day: int = -60) -> EngineState:
    last = snapshot[-1]
    return EngineState(
        saved_at=D(27),
        last_update=D(27),
        active_day=_day(28),
        persona_start_at=D(start_day),
        emotions={"joy": EmotionState(short=Average(last.short, 2.0), long=Average(last.long, 20.0), last_entry_day=_day(27), snapshot=snapshot)},
    )


@pytest.mark.parametrize(
    ("pinned", "climb"),
    [(10.0, [6.0, 7.0, 8.0, 9.0, 9.5, 10.0]), (0.0, [4.0, 3.0, 2.0, 1.0, 0.5, 0.0])],
    ids=["rise pinned at 10", "fall pinned at 0"],
)
def test_a_run_pinned_at_a_limit_ends_on_the_second_flat_day_and_starts_none(tmp_path: Path, pinned: float, climb: list[float]) -> None:
    """(d) The short average moves six days in one direction (days 22-27, length 6, felt on all) and reaches the limit on
    day 27: bar 4 is beaten, flag +1 (rise) or -1 (fall). Entries at the limit then feed it the value it holds: exact zero
    moves. The first flat day (28) is a blip: the flag stays. The second (29) ends the run and a zero starts none: 0, and it
    stays 0 on every later flat day.
    """
    simulation = Sim(tmp_path, state=_state_with_snapshot(_long_run_bar_snapshot(short_days=climb)))
    direction = 1 if pinned > 5.0 else -1
    assert simulation.shift() == direction  # closed(27): the run of 6 beats the bar of 4
    flags = simulation.run(dict.fromkeys(range(28, 36), pinned), first=28, last=34)
    assert [flags[c] for c in (28, 29, 30, 31, 32, 33, 34)] == [direction, 0, 0, 0, 0, 0, 0]


# --- (e) falls ----------------------------------------------------------------------------------------------------------------


def _falling_base() -> EngineState:
    return _state_with_snapshot(_long_run_bar_snapshot(short_days=[5.0] * 6))


def test_a_fall_through_quiet_day_zeros_alone_does_not_count_but_the_same_fall_felt_does(tmp_path: Path) -> None:
    """(e) Bar 4. Active days with no entry close with a zero entry, so the short average falls (a zero is below it) with the day
    NOT felt: six such days make a fall of length 6 felt on 0 of 6, which does not count (0). The control: the same six days
    with an entry of 0.0 each (also below the average, and felt) make a fall of 6 felt on 6: -1.
    """
    quiet = Sim(tmp_path / "quiet", state=_falling_base())
    flags = quiet.run({}, first=28, last=33)
    assert set(flags.values()) == {0}
    felt = Sim(tmp_path / "felt", state=_falling_base())
    flags = felt.run(dict.fromkeys(range(28, 34), 0.0), first=28, last=33)
    assert flags[33] == -1 and [flags[c] for c in (28, 29, 30)] == [0, 0, 0]  # 4 days is not longer than the bar of 4


def test_a_fall_felt_on_exactly_half_its_days_does_not_count_and_on_more_than_half_does(tmp_path: Path) -> None:
    """(e) Six falling days, entries (0.0) on days 28, 30, 32 and none on 29, 31, 33 (the quiet days fall by a zero): felt on 3
    of 6 is exactly half, which is not "more than half": 0. Entries on 28, 29, 30, 31 and none on 32, 33: 4 of 6: -1.
    """
    half = Sim(tmp_path / "half", state=_falling_base())
    flags = half.run({28: 0.0, 30: 0.0, 32: 0.0}, first=28, last=33)
    assert flags[33] == 0
    most = Sim(tmp_path / "most", state=_falling_base())
    flags = most.run(dict.fromkeys(range(28, 32), 0.0), first=28, last=33)
    assert flags[33] == -1


# --- (f) quiet rules ------------------------------------------------------------------------------------------------------------


def test_the_start_gate_a_young_persona_is_gated_and_the_same_history_with_an_old_start_is_not(tmp_path: Path) -> None:
    """(f) The history of (d): the short run of 6 beats the bar of 4 at closed(27). The gate: the last snapshot day (27) ends at
    local midnight, the start of day 28; the shift reads 0 unless that is 30 days or more after the persona's start (a noon). A
    start at noon of day -4 is 31.5 days before it, day -3 is 30.5: open, +1. Day -2 is 29.5 and day -1 is 28.5: closed, 0.
    """
    climb = [6.0, 7.0, 8.0, 9.0, 9.5, 10.0]
    outcome = {}
    for start in (-4, -3, -2, -1):
        simulation = Sim(tmp_path / f"start{start}", state=_state_with_snapshot(_long_run_bar_snapshot(short_days=climb), start_day=start))
        outcome[start] = simulation.shift()
    assert outcome == {-4: 1, -3: 1, -2: 0, -1: 0}


def test_a_feeling_with_no_finished_long_run_is_quiet(tmp_path: Path) -> None:
    """(f) The same short climb, but the long average only ever rises (no run finished in the window): quiet, 0."""
    snapshot = _long_run_bar_snapshot(short_days=[6.0, 7.0, 8.0, 9.0, 9.5, 10.0])
    control = Sim(tmp_path / "control", state=_state_with_snapshot(snapshot))
    assert control.shift() == 1  # with the finished long run of (d) the same short climb is a shift
    snapshot = [day._replace(long=3.0 + 0.01 * n) for n, day in enumerate(snapshot)]  # up every day: one run, never finished
    assert Sim(tmp_path / "quiet", state=_state_with_snapshot(snapshot)).shift() == 0


def test_a_persona_whose_first_rows_are_uncounted_opens_its_gate_thirty_days_after_the_oldest_of_them(tmp_path: Path) -> None:
    """(f), the G8 case through the commit fold: the persona's first rows are of a type that counts for nothing. The start-only
    event sets `persona_start_at` to the oldest of them (day 0 noon). The snapshot of (d) with its last day 29 ends at the start
    of day 30, 29.5 days after that noon: gated, 0. With its last day 30 it ends 30.5 days after it: open, +1.
    """
    persona = make_persona(tmp_path / "persona")
    install(persona, D(400))
    store = MemoryStore(persona / "memories.db")
    try:
        store.create(row(D(0), memory_type="journal_entry", emotions={}))
        store.create(row(D(5), memory_type="journal_entry", emotions={}))
    finally:
        store.close()
    state = state_of(persona)
    assert state.persona_start_at == D(0)
    climb = [6.0, 7.0, 8.0, 9.0, 9.5, 10.0]
    for last_day, expected in ((29, 0), (30, 1)):
        shift = last_day - 27  # the snapshot of (d) is shifted later by this many days
        snapshot = [d._replace(day=_day(n + shift)) for n, d in enumerate(_long_run_bar_snapshot(short_days=climb))]
        grafted = EngineState(
            saved_at=D(last_day),
            last_update=D(last_day),
            active_day=_day(last_day + 1),
            persona_start_at=state.persona_start_at,
            emotions={"joy": EmotionState(short=Average(10.0, 2.0), long=Average(3.2, 20.0), last_entry_day=_day(last_day), snapshot=snapshot)},
        )
        save(persona, grafted)
        assert install(persona, D(400)).read()["joy"].shift == expected


def _first_rebuild(persona: Path, now) -> None:  # noqa: ANN001
    """Run the first rebuild on `persona` at `now` the way the engine's start-up does, through `engine_rebuild.rebuild`.

    The two callbacks are the ones `catch_up` takes: first mode folds the stored values and does not call the fade, and the
    weight of every row here is 1, so neither callback changes a number.
    """
    engine_rebuild.rebuild(
        persona,
        "first",
        now,
        faded_emotions_of=lambda memory: dict(memory.emotions),
        modifier_of=lambda memory: 1.0,
    )


@pytest.mark.parametrize(("oldest_day", "expected"), [(-62, 1), (18, 0)], ids=["oldest memory 90 days old", "oldest memory 10 days old"])
def test_the_start_gate_at_the_first_rebuild_counts_from_the_oldest_memory_not_from_the_oldest_entry(
    tmp_path: Path, oldest_day: int, expected: int
) -> None:
    """(f), the rebuild half of the G8 case. A persona that already has a database is rebuilt at noon of day 28 + 3 h. Its rows
    in the 30-day window are on its last 5 days only (days 23 to 27, one conversation row each), so the engine's own oldest entry
    is day 23 and the persona would read as new. Its oldest row of all is a journal entry (it counts for nothing) dated either 90
    or 10 days before day 28.

    The rebuild takes `persona_start_at` from that oldest row (a plain query, outside the window): day -62 noon and day 18 noon.
    The gate then compares the end of the last snapshot day with it. The snapshot of (d) is put on the rebuilt state (the
    rebuild itself builds only days 23 to 27, too few for any run) with its last day 27: its end is the start of day 28, 89.5
    days after the first start and 9.5 days after the second. Open, +1 for the old persona; gated, 0 for the other.
    """
    now = D(28) + timedelta(hours=3)
    rows = [row(D(oldest_day), memory_type="journal_entry", emotions={}, mid="oldest")]
    rows += [row(D(day), emotions={"joy": 5.0 + day % 3}, mid=f"recent-{day}") for day in range(23, 28)]
    persona = make_persona(tmp_path / "persona", rows)
    _first_rebuild(persona, now)
    rebuilt = state_of(persona)
    assert rebuilt.persona_start_at == D(oldest_day)  # not D(23): the oldest memory, not the oldest entry in the window
    assert {entry.day for entry in rebuilt.emotions["joy"].snapshot} <= {_day(day) for day in range(23, 28)}
    climb = [6.0, 7.0, 8.0, 9.0, 9.5, 10.0]
    rebuilt.emotions = {
        "joy": EmotionState(
            short=Average(10.0, 2.0),
            long=Average(3.2, 20.0),
            last_entry_day=_day(27),
            snapshot=_long_run_bar_snapshot(short_days=climb),
        )
    }
    rebuilt.active_day = _day(28)
    save(persona, rebuilt)
    assert install(persona, now).read()["joy"].shift == expected


# --- (h) validity ------------------------------------------------------------------------------------------------------------


def _hundred_twenty_day_persona(root: Path) -> Sim:
    simulation = Sim(root)
    for day in range(120):
        simulation.entry(day, "joy", 6.0 if day < 100 else 6.0 + 0.5 * (day - 99), hours=0.5)
        simulation.entry(day, "grief", 3.0)
    simulation.drain(120)
    return simulation


def test_a_clean_hundred_and_twenty_day_run_logs_no_bad_day(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        simulation = _hundred_twenty_day_persona(tmp_path)
        simulation.engine.read()
        engine_files.read_state(simulation.persona)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "snapshot" in r.getMessage()] == []
    assert len(state_of(simulation.persona).emotions["joy"].snapshot) == 30


def _edit_state(persona: Path, edit) -> None:  # noqa: ANN001
    import json

    path = engine_files.state_path(persona)
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    "damage",
    ["non-finite", "out of order", "run starts after its day"],
)
def test_a_damaged_snapshot_day_is_dropped_and_the_shift_is_the_one_worked_out_without_it(tmp_path: Path, damage: str) -> None:
    """(h) The damaged day is dropped as if missing, and `read()` then equals a read of the same state with that day removed."""
    damaged = _hundred_twenty_day_persona(tmp_path / "damaged")
    removed = _hundred_twenty_day_persona(tmp_path / "removed")
    victim = 7

    def damage_it(data: dict) -> None:
        days = data["emotions"]["joy"]["snapshot"]
        if damage == "non-finite":
            days[victim][1] = float("nan")
        elif damage == "out of order":
            days[victim], days[victim + 1] = days[victim + 1], days[victim]
        else:
            days[victim][4] = ["2999-01-01", 1, days[victim][0]]

    def remove_it(data: dict) -> None:
        del data["emotions"]["joy"]["snapshot"][victim]
        if damage == "out of order":
            del data["emotions"]["joy"]["snapshot"][victim]  # both swapped days are lost: the second one is out of order

    _edit_state(damaged.persona, damage_it)
    _edit_state(removed.persona, remove_it)
    loaded = engine_files.read_state(damaged.persona)
    assert loaded.state is not None and loaded.problem is None
    days = [d.day for d in loaded.state.emotions["joy"].snapshot]
    assert len(days) == len(set(days)) and days == sorted(days) and len(days) < 30
    assert {n: r.shift for n, r in install(damaged.persona, D(400)).read().items()} == {
        n: r.shift for n, r in install(removed.persona, D(400)).read().items()
    }


def test_an_average_above_ten_is_held_to_ten_and_the_day_kept(tmp_path: Path) -> None:
    simulation = _hundred_twenty_day_persona(tmp_path)
    _edit_state(simulation.persona, lambda data: data["emotions"]["joy"]["snapshot"][3].__setitem__(1, 11.0))
    loaded = engine_files.read_state(simulation.persona)
    days = loaded.state.emotions["joy"].snapshot
    assert len(days) == 30 and days[3].short == 10.0


@pytest.mark.parametrize("value", [0.0, 2.0, 5.999, 6.0])
def test_an_entry_at_or_below_an_average_never_raises_it(tmp_path: Path, value: float) -> None:
    simulation = Sim(tmp_path)
    for day in range(3):
        simulation.entry(day, "joy", 6.0)
    simulation.drain(3)
    before = state_of(simulation.persona).emotions["joy"]
    simulation.entry(3, "joy", value, hours=2.0)
    simulation.drain(3, hours=3.0)
    after = state_of(simulation.persona).emotions["joy"]
    assert after.short.avg <= before.short.avg and after.long.avg <= before.long.avg
    if value == 6.0:
        assert (after.short.avg, after.long.avg) == (6.0, 6.0)  # exactly: the delta form moves nothing


# --- (j) a long run that began before the window ------------------------------------------------------------------------------


def _seeded_snapshot(*, run_start_offset: int, short_moves: int = 29) -> list[SnapshotDay]:
    """30 snapshot days (0 to 29). The long average's run began `run_start_offset` days BEFORE day 0, is recorded on the oldest
    day as RunRecord(start, +1, last with-direction day = day 0), rises on days 1-10 (so its last with-direction day is 10), and is
    finished by two days against (11, 12). The short average rises on days 1 to `short_moves`; every day felt."""
    long_values = [3.0 + 0.1 * n for n in range(11)] + [3.9, 3.8] + [3.8] * 17
    short_values = [2.0 + 0.1 * min(n, short_moves) for n in range(30)]
    start = local_day(-run_start_offset).isoformat()
    return [
        SnapshotDay(_day(n), short_values[n], long_values[n], True, RunRecord(start, 1, _day(0)) if n == 0 else None)
        for n in range(30)
    ]


def _window_state(snapshot: list[SnapshotDay]) -> EngineState:
    return EngineState(
        saved_at=D(29),
        last_update=D(29),
        active_day=_day(30),
        persona_start_at=D(-100),
        emotions={"joy": EmotionState(short=Average(snapshot[-1].short, 2.0), long=Average(snapshot[-1].long, 20.0), last_entry_day=_day(29), snapshot=snapshot)},
    )


def test_a_long_run_that_began_before_the_window_keeps_its_true_length(tmp_path: Path) -> None:
    """(j) The long run's last with-direction day is window day 10 and it began 34 days before day 0 (44 days before day 10): its
    true length is 45, though only 11 of its days are in the window. The short run of 29 moves does not beat it: 0. A run that began 9
    days before day 0 (19 before day 10) has true length 20: 29 beats it: +1.
    """
    bar_45 = Sim(tmp_path / "bar45", state=_window_state(_seeded_snapshot(run_start_offset=34)))
    bar_20 = Sim(tmp_path / "bar20", state=_window_state(_seeded_snapshot(run_start_offset=9)))
    assert (bar_45.shift(), bar_20.shift()) == (0, 1)


def _edge_snapshot(long_values: list[float], short_values: list[float], first_record: RunRecord) -> list[SnapshotDay]:
    """30 snapshot days from per-day values; only the oldest day carries a run record (the seed), every day is felt."""
    return [
        SnapshotDay(_day(n), short_values[n], long_values[n], True, first_record if n == 0 else None) for n in range(30)
    ]


def _short_rising_for(moves: int) -> list[float]:
    """A short average flat, then rising for its last `moves` days (days 30 - moves to 29): a run still in progress at the end."""
    return [2.0 + 0.1 * max(0, n - (29 - moves)) for n in range(30)]


def test_a_blip_on_the_oldest_day_with_the_next_day_also_against_starts_the_new_run_on_that_day_with_length_two(tmp_path: Path) -> None:
    """(j), the accepted edge ER. The long run (true length 4 up to day -1) has its first reverse day on the OLDEST snapshot day
    (day 0), so that day has no earlier day to read a move from and is read as a move the other way. Day 1 is against again: the old
    run ends (finished length 3), and the two reverse days are the first two days of a DOWN run: it starts on day 0 with length 2, grows
    to 9 on day 8 (down on days 1-8), and is finished by two days up (9 and 10): finished length 9. The bar is 9. A short run of 9 moves
    does not beat it (equal): 0; a short run of 10 does: +1. (Had the run started on day 1 with length 1, 9 would beat a bar of 8.)
    """
    long_values = [5.0 - 0.1 * n for n in range(9)] + [4.3, 4.4] + [4.4] * 19
    record = RunRecord(local_day(-3).isoformat(), 1, local_day(-1).isoformat())  # last with-direction day -1, so day 0 is the pending day
    equal = Sim(tmp_path / "equal", state=_window_state(_edge_snapshot(long_values, _short_rising_for(9), record)))
    beats = Sim(tmp_path / "beats", state=_window_state(_edge_snapshot(long_values, _short_rising_for(10), record)))
    assert (equal.shift(), beats.shift()) == (0, 1)


def test_a_zero_move_blip_with_a_day_before_it_starts_the_new_run_on_the_next_day_with_length_one(tmp_path: Path) -> None:
    """(j) The long run is seeded on day 0 (start day -1, last with-direction day 0, true length 2). Day 1 is a ZERO move (a blip, kind
    read from the day's own move, which exists because day 0 is before it); day 2 is down: the old run ends (finished length 2: the
    length held, 3, less the blip) and, the first reverse day being a zero, the new down run starts on day 2 with length 1. It grows
    to 7 on day 8 and is finished by two up days (9, 10): finished length 7; the bar is 7. A short run of 7 moves: equal, 0; of 8: +1.
    """
    long_values = [5.0, 5.0] + [5.0 - 0.1 * (n - 1) for n in range(2, 9)] + [4.4, 4.5] + [4.5] * 19
    record = RunRecord(local_day(-1).isoformat(), 1, local_day(0).isoformat())
    equal = Sim(tmp_path / "equal", state=_window_state(_edge_snapshot(long_values, _short_rising_for(7), record)))
    beats = Sim(tmp_path / "beats", state=_window_state(_edge_snapshot(long_values, _short_rising_for(8), record)))
    assert (equal.shift(), beats.shift()) == (0, 1)


# --- (k) off days ------------------------------------------------------------------------------------------------------------


def _climb_values(through: int) -> dict[int, float]:
    """The timeline of (b) with the 9.0 climb running through day `through`, then two days at 0.0."""
    values = _steady(32, 2.0)
    values.update(dict.fromkeys(range(33, 37), 8.0))
    values.update(dict.fromkeys(range(37, 39), 0.0))
    values.update(dict.fromkeys(range(39, through + 1), 9.0))
    values.update(dict.fromkeys(range(through + 1, through + 3), 0.0))
    return values


def test_one_off_day_inside_a_run_carries_it_on_and_adds_one_to_its_length(sim: Sim) -> None:
    """(k) The climb of (b) (bar 4; up run starting on day 39) with day 41 an OFF day (no drain, no entry): the gap is a zero move, a
    blip, counted into the length: day 40 is length 2, the off day makes 3, day 42 makes 4 (equal: 0), day 43 makes 5: +1, as without
    the off day. Had the off day been ignored the flag would come a day later, had it ended the run it would not come on 43 at all.
    The snapshot holds no day 41, so the flags are labelled by the last closed day.
    """
    flags = sim.run(_climb_values(46), first=40, last=48, off=frozenset({41}))
    assert flags == {40: 0, 42: 0, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 0}


def test_two_off_days_in_a_row_end_the_run_before_the_first_and_start_none(sim: Sim) -> None:
    """(k) Days 41 and 42 off: two zero moves in a row end the up run (it finishes before the first of them) and a zero starts no new
    one. Entries resume on day 43 (9.0, above the average: up): a NEW run starts there with length 1 (44: 2, 45: 3, 46: 4: equal, 0) and
    day 47 makes 5: +1. Closed(43) to closed(46) are 0, closed(47) is the first flag; the two days at 0.0 (51, 52) end it again.
    """
    flags = sim.run(_climb_values(50), first=40, last=52, off=frozenset({41, 42}))
    assert flags == {40: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 1, 48: 1, 49: 1, 50: 1, 51: 1, 52: 0}


# --- (l) the run record is carried by `close_days` past the 30-day window ----------------------------------------------------------


def _long_run_then_short_climb(*, up_days: int) -> dict[int, float]:
    """Day 0 at 1.0, days 1..up_days at 10.0 (up_days up moves of the long average), two days at 0.0, then 10.0 again for 17 days.

    Every repeat of 10.0 after the first is a real move, because the averages are still below 10.0. The two days at 0.0 finish the
    long run (finished length `up_days`); the short average is then a down run of 2, a blip, and a new up run that reaches 17.
    """
    values = {0: 1.0}
    values.update(dict.fromkeys(range(1, up_days + 1), 10.0))
    values.update(dict.fromkeys(range(up_days + 1, up_days + 3), 0.0))
    values.update(dict.fromkeys(range(up_days + 3, up_days + 20), 10.0))
    return values


@pytest.mark.no_base_failure(reason="the carry already exists on the base; this pins it against a `run = None` mutation of `close_days`")
def test_a_thirty_one_day_long_run_is_read_with_its_true_length_after_the_window_has_slid_past_most_of_it(sim: Sim) -> None:
    """(l) The long average rises on days 1 to 31 (a run of 31 up days that starts on day 1), falls on 32 and 33 (finished: length 31),
    then rises again. The short average's run in progress after day 50 closed is 17 days long (a down run of 2, a blip on day 34, and an
    up run that starts on day 34 and reaches day 50: 17 days). The snapshot after day 50 closed holds days 21 to 50, so only 11 days
    of the long run (21 to 31) are in the window, and 10 of its moves (22 to 31: the oldest snapshot day has no earlier day to move from): the bar is 31 only because `close_days` carried the run's record forward on
    every snapshot day since day 1. 17 does not beat 31: 0. Had the record not been carried, the window alone would give a finished run
    of 10, and 17 would beat it: a false +1.
    """
    flags = sim.run(_long_run_then_short_climb(up_days=31), first=50, last=50)
    assert flags == {50: 0}
    oldest = state_of(sim.persona).emotions["joy"].snapshot[0]
    assert oldest.day == _day(21) and oldest.run == RunRecord(_day(1), 1, _day(21))  # day 21's record: the run began on day 1, before the window


@pytest.mark.no_base_failure(reason="the control for the case above: the same path with a shorter true run flags on the base too")
def test_the_same_path_with_a_sixteen_day_long_run_does_flag_so_the_zero_above_is_the_bar(sim: Sim) -> None:
    """(l) Control. A long run of 16 up days (days 1 to 16), finished by days 17 and 18, and the same 17-day short run after day 35
    closed. The window (days 6 to 35) again holds 11 days of the long run (6 to 16; 10 of its moves), the true bar is 16, and 17 beats it: +1. So the 0
    of the 31-day case comes from the bar, not from a closed gate or a quiet feeling.
    """
    flags = sim.run(_long_run_then_short_climb(up_days=16), first=35, last=35)
    assert flags == {35: 1}


def _close_state(snapshot: list[SnapshotDay], long_avg: float, *, active: int) -> EngineState:
    """A state whose open day is `active`, with `snapshot` kept for joy and an entry already on the open day (no quiet-day zero)."""
    return EngineState(
        saved_at=D(active),
        last_update=D(active),
        active_day=_day(active),
        persona_start_at=D(-100),
        emotions={
            "joy": EmotionState(
                short=Average(long_avg, 1.0), long=Average(long_avg, 1.0), last_entry_day=_day(active), snapshot=snapshot
            )
        },
    )


def _written_record(state: EngineState, closing: int) -> RunRecord | None:
    snapshot = state.emotions["joy"].snapshot
    assert snapshot[-1].day == _day(closing)
    return snapshot[-1].run


@pytest.mark.no_base_failure(reason="the pending-day kind already exists on the base; this pins it against forcing it to other")
def test_a_zero_move_on_the_carried_pending_day_then_a_reversal_starts_the_new_run_on_the_reversing_day() -> None:
    """(l) The seed's pending day (a day after the run's last with-direction day, so not yet decided) takes its kind from the stored
    snapshot: day 1's long average equals day 0's, a zero move. A down move on day 2 ends the run (finished) and, the first reverse
    day being a zero, the new down run starts on day 2 with length 1: the record written on day 2 is (day 2, down, day 2).
    """
    snapshot = [
        SnapshotDay(_day(0), 5.0, 5.0, True, None),
        SnapshotDay(_day(1), 5.0, 5.0, True, RunRecord(_day(-4), 1, _day(0))),
    ]
    state = _close_state(snapshot, 4.9, active=2)
    close_days(state, D(3))
    assert _written_record(state, 2) == RunRecord(_day(2), -1, _day(2))


@pytest.mark.no_base_failure(reason="the control for the case above: a pending day that moved the other way starts the new run on it")
def test_a_down_move_on_the_carried_pending_day_then_another_starts_the_new_run_on_the_pending_day() -> None:
    """(l) Control. Day 1's long average is below day 0's (a move the other way), and day 2 moves down again: both reverse days are
    down, so the new down run starts on the pending day (day 1) with length 2: the record is (day 1, down, day 2).
    """
    snapshot = [
        SnapshotDay(_day(0), 5.0, 5.0, True, None),
        SnapshotDay(_day(1), 4.9, 4.9, True, RunRecord(_day(-4), 1, _day(0))),
    ]
    state = _close_state(snapshot, 4.8, active=2)
    close_days(state, D(3))
    assert _written_record(state, 2) == RunRecord(_day(1), -1, _day(2))


@pytest.mark.no_base_failure(reason="the pending-day kind of an off day already exists on the base; this pins it against forcing it to other")
def test_an_off_day_then_a_reversal_in_one_close_starts_the_new_run_on_the_reversing_day() -> None:
    """(l) The run is seeded on day 0 with its last with-direction day day 0, the app was off on day 1 (a zero move, the run's first
    reverse day), and day 2 moves down in the same close. The zero first day starts no run: the new down run starts on day 2 with
    length 1 and the record is (day 2, down, day 2).
    """
    snapshot = [SnapshotDay(_day(0), 5.0, 5.0, True, RunRecord(_day(-4), 1, _day(0)))]
    state = _close_state(snapshot, 4.9, active=2)
    close_days(state, D(3))
    assert _written_record(state, 2) == RunRecord(_day(2), -1, _day(2))


# --- (i) catch-up ---------------------------------------------------------------------------------------------------------


_RUN_DAYS = 40
_COPY_DAY = 20


def _joy_on(day: int) -> float:
    return 2.0 + (day * 3) % 7  # 2 to 8, a different number most days


def _grief_change_on(day: int) -> float:
    return 1.0 + (day * 2) % 5  # 1 to 5


def _fold_view(state: EngineState) -> dict:
    """Everything the fold decides, in plain numbers (no save or reset times, which differ by design)."""
    return {
        name: {
            "short": (emotion.short.avg, emotion.short.weight),
            "long": (emotion.long.avg, emotion.long.weight),
            "last_entry_day": emotion.last_entry_day,
            "snapshot": [(d.day, d.short, d.long, d.felt, d.run) for d in emotion.snapshot],
        }
        for name, emotion in sorted(state.emotions.items())
    } | {"_": {"active_day": state.active_day, "last_update": state.last_update, "start": state.persona_start_at, "repeats": state.repeats}}


def _assert_same_fold(got: dict, want: dict) -> None:
    assert got.keys() == want.keys()
    for name in want:
        if name == "_":
            assert got[name] == want[name]
            continue
        for key in ("short", "long"):
            assert got[name][key] == pytest.approx(want[name][key], abs=1e-9), (name, key)
        assert got[name]["last_entry_day"] == want[name]["last_entry_day"], name
        assert len(got[name]["snapshot"]) == len(want[name]["snapshot"]), name
        for mine, theirs in zip(got[name]["snapshot"], want[name]["snapshot"], strict=True):
            assert mine[0] == theirs[0] and mine[3:] == theirs[3:], (name, mine, theirs)
            assert mine[1:3] == pytest.approx(theirs[1:3], abs=1e-9), (name, mine, theirs)


@pytest.mark.no_base_failure(
    reason="Check 25 (i) is a whole-run equality pin of `catch_up`, which Engine 4 built and main already has; it is listed under "
    "S6 only because the change log it replays is written only live. It passes on the base by design; the check below that it "
    "can fail (a missing change line) was run by hand."
)
def test_a_state_copy_from_day_twenty_caught_up_from_the_database_and_the_change_log_equals_the_uninterrupted_run(tmp_path: Path) -> None:
    """(i) A 40-day run, one committed row and one chat change on every day (a new memory id each, so no repeat factor; no
    showing, no climax, no access-only fold), drained at the end of each day, is the uninterrupted run. The state it held
    after day 20 is copied: it becomes the ONLY readable state (a backup copy) of a second persona with the same database, the
    change lines of days 21 to 38 in its change log, and only the last turn's change (day 39) in its queue.

    The history is built so that equality holds by construction (E5.19: the catch-up inherits the rebuild's limits): every row
    after day 20 is in the database, every change but the last is in the log, the replay window's upper bound is the oldest
    queued line (exclusive), so the last change folds once through the drain and no line twice. One drain then gives the
    uninterrupted run's averages, weights and snapshot days to 1e-9, each queued event once.

    The change lines are written with `engine_files.append_change`, the writer a live engine uses for its own change log; the
    uninterrupted run is not live here, so it writes none, and the second persona's log is what a live run would have left.
    """
    rows = [row(D(day), emotions={"joy": _joy_on(day)}, mid=f"day-{day}") for day in range(_RUN_DAYS)]
    uninterrupted = make_persona(tmp_path / "uninterrupted")
    store = MemoryStore(uninterrupted / "memories.db")
    copied: EngineState | None = None
    try:
        for day in range(_RUN_DAYS):
            live = install(uninterrupted, D(day) + timedelta(hours=1))
            store.create(rows[day])
            live.fold_change({"grief": _grief_change_on(day)}, D(day) + timedelta(hours=2), "pass2")
            live.note_active(D(day) + timedelta(hours=3))
            if day == _COPY_DAY:
                copied = copy.deepcopy(state_of(uninterrupted))
    finally:
        store.close()
    assert copied is not None
    last = D(_RUN_DAYS - 1) + timedelta(hours=3)
    want = _fold_view(state_of(uninterrupted))
    assert len(want["joy"]["snapshot"]) == 30 and want["grief"]["snapshot"]  # a real 30-day snapshot, not an empty comparison

    crashed = make_persona(tmp_path / "crashed", rows)
    save_as_backup(crashed, copied)
    for day in range(_COPY_DAY + 1, _RUN_DAYS - 1):
        assert engine_files.append_change(
            crashed, Event(kind="change", at=D(day) + timedelta(hours=2), emotions={"grief": _grief_change_on(day)}, source="pass2")
        )
    queue(crashed, change(D(_RUN_DAYS - 1) + timedelta(hours=2), {"grief": _grief_change_on(_RUN_DAYS - 1)}))
    recovered = install(crashed, last)
    recovered.note_active(last)
    assert recovered.status().caught_up
    _assert_same_fold(_fold_view(state_of(crashed)), want)
    assert engine_files.read_queue(crashed, from_line=0).events == []  # the one queued line was folded once and the file emptied
