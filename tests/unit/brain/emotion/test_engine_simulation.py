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

import logging
from datetime import date, timedelta
from pathlib import Path

import pytest

from brain.emotion import engine_files
from brain.emotion.engine_utils import Average, EmotionState, EngineState, RunRecord, SnapshotDay
from tests.unit.brain.emotion._engine_support import (
    DAY0,
    install,
    local_day,
    local_noon,
    make_persona,
    queue,
    save,
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
    snapshot = [day._replace(long=3.0 + 0.01 * n) for n, day in enumerate(snapshot)]  # up every day: one run, never finished
    assert Sim(tmp_path, state=_state_with_snapshot(snapshot)).shift() == 0


def test_a_persona_whose_first_rows_are_uncounted_opens_its_gate_thirty_days_after_the_oldest_of_them(tmp_path: Path) -> None:
    """(f), the G8 case through the commit fold: the persona's first rows are of a type that counts for nothing. The start-only
    event sets `persona_start_at` to the oldest of them (day 0 noon). The snapshot of (d) with its last day 29 ends at the start
    of day 30, 29.5 days after that noon: gated, 0. With its last day 30 it ends 30.5 days after it: open, +1.
    """
    from brain.memory.store import MemoryStore
    from tests.unit.brain.emotion._engine_support import row

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
