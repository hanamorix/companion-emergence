"""Engine 6, G1: the one-time x10 fix of small feelings (`nudge_scale_fix` and `MemoryStore.scale_up_small_nudge_rows`).

A fixture database is built with `MemoryStore.create(fold=False)` (the engine is not told). A damaged row is made with a direct
`sqlite3` connection in the test, which is not product code. Nothing here touches a real persona.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files, nudge_scale_fix
from brain.memory.store import Memory, MemoryStore
from tests.unit.brain.emotion._engine_support import refuse_the_switch_stat

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _memory(memory_id: str, memory_type: str, emotions: dict[str, float], *, importance: float = 3.0) -> Memory:
    numbers = [float(v) for v in emotions.values()]
    return Memory(
        id=memory_id,
        content=f"content of {memory_id}",
        memory_type=memory_type,
        domain="us",
        created_at=NOW,
        emotions=dict(emotions),
        tags=[],
        importance=importance,
        score=sum(numbers),
        peak_emotion_intensity=max(numbers, default=0.0),
    )


def _persona(tmp_path: Path, *rows: Memory) -> Path:
    store = MemoryStore(tmp_path / "memories.db")
    try:
        for memory in rows:
            store.create(memory, fold=False)
    finally:
        store.close()
    return tmp_path


def _raw(persona: Path, sql: str, *args: object) -> None:
    connection = sqlite3.connect(persona / "memories.db")
    try:
        connection.execute(sql, args)
        connection.commit()
    finally:
        connection.close()


def _read(persona: Path, memory_id: str) -> sqlite3.Row:
    connection = sqlite3.connect(persona / "memories.db")
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            "SELECT emotions_json, score, peak_emotion_intensity, importance FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
    finally:
        connection.close()


def _marker(persona: Path) -> int:
    connection = sqlite3.connect(persona / "memories.db")
    try:
        return connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()


def _feelings(persona: Path, memory_id: str) -> dict[str, float]:
    return json.loads(_read(persona, memory_id)["emotions_json"])


def _unchanged(persona: Path, memory_id: str, emotions: dict[str, float]) -> bool:
    row = _read(persona, memory_id)
    return json.loads(row["emotions_json"]) == emotions and row["peak_emotion_intensity"] == max(emotions.values(), default=0.0)


# --- the selector and the three columns ---------------------------------------------------------


def test_rows_of_the_three_types_have_feelings_score_and_peak_multiplied_and_importance_left_alone(tmp_path: Path) -> None:
    persona = _persona(
        tmp_path,
        _memory("reach", "initiate_outbound", {"tenderness": 0.15, "longing": 0.1}, importance=4.0),
        _memory("made", "making", {"pride": 0.12}),
        _memory("peer", "kindled_peer", {"warmth": 0.5, "trust": 0.25}),
    )
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 3
    # hand-worked: 0.15 x 10 = 1.5, 0.1 x 10 = 1.0, score 0.25 x 10 = 2.5, peak 0.15 x 10 = 1.5
    reach = _read(persona, "reach")
    assert json.loads(reach["emotions_json"]) == pytest.approx({"tenderness": 1.5, "longing": 1.0})
    assert reach["score"] == pytest.approx(2.5)
    assert reach["peak_emotion_intensity"] == pytest.approx(1.5)
    assert reach["importance"] == 4.0  # the spec names three columns: importance is deliberately untouched
    assert _feelings(persona, "made") == pytest.approx({"pride": 1.2})
    assert _feelings(persona, "peer") == pytest.approx({"warmth": 5.0, "trust": 2.5})


def test_a_row_of_another_type_is_not_multiplied_however_small(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("talk", "conversation", {"joy": 0.2}), _memory("dream", "dream", {"awe": 0.3}))
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 0
    assert _unchanged(persona, "talk", {"joy": 0.2}) and _unchanged(persona, "dream", {"awe": 0.3})
    assert _marker(persona) == dev_constants.NUDGE_SCALE_FIX_MARKER  # it ran, and found nothing to do


def test_a_row_already_on_the_ten_scale_is_not_multiplied(tmp_path: Path) -> None:
    persona = _persona(
        tmp_path,
        _memory("big", "making", {"pride": 4.0, "joy": 0.3}),  # peak 4, one small value: not a small row
        _memory("one", "kindled_peer", {"warmth": 1.0}),  # a value of exactly 1 is not below the threshold
    )
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 0
    assert _unchanged(persona, "big", {"pride": 4.0, "joy": 0.3})
    assert _unchanged(persona, "one", {"warmth": 1.0})


def test_a_row_whose_stored_peak_is_small_while_a_feeling_is_not_is_skipped_with_a_warning_and_not_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    persona = _persona(tmp_path, _memory("odd", "making", {"pride": 0.4}), _memory("fine", "making", {"pride": 0.3}))
    _raw(persona, "UPDATE memories SET emotions_json = ? WHERE id = ?", json.dumps({"pride": 3.0}), "odd")  # peak column stays 0.4
    caplog.set_level(logging.WARNING)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _feelings(persona, "odd") == {"pride": 3.0}
    assert _feelings(persona, "fine") == pytest.approx({"pride": 3.0})
    assert any("odd" in r.getMessage() and "skipped" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("bad_json", "label"),
    [
        ("not json at all", "unparseable"),
        (json.dumps(["tenderness", 0.2]), "not a mapping"),
        (json.dumps({"tenderness": "0.2"}), "a text value"),
        (json.dumps({"tenderness": True}), "a boolean value"),
        (json.dumps({"tenderness": None}), "a null value"),
        # json.loads accepts NaN and Infinity. The range test used to reject them without naming them; it must keep doing so
        # (C4 of the Q11b record). These cases pass on the code before Q11b by design: the behaviour is unchanged.
        pytest.param(
            '{"tenderness": NaN}',
            "a NaN value",
            marks=pytest.mark.no_base_failure(reason="retained behaviour: the old range test rejected NaN too; this pins that the Q11b rewrite keeps it"),
        ),
        pytest.param(
            '{"tenderness": Infinity}',
            "an infinite value",
            marks=pytest.mark.no_base_failure(reason="retained behaviour: the old range test rejected Infinity too; this pins that the Q11b rewrite keeps it"),
        ),
        pytest.param(
            '{"tenderness": -Infinity}',
            "a negative infinite value",
            marks=pytest.mark.no_base_failure(reason="retained behaviour: the old range test rejected -Infinity too; this pins that the Q11b rewrite keeps it"),
        ),
        # Review F1/F2: a whole-number feeling too big for a float, and a finite one whose product would overflow to -Infinity.
        # json.loads accepts both. The row stays skipped, as before Q11b (then for the range, now for the product).
        pytest.param(
            '{"tenderness": 1' + "0" * 400 + "}",
            "a whole number too large for a float",
            id="huge positive int",
            marks=pytest.mark.no_base_failure(reason="retained behaviour: the old range test rejected a huge int too; this pins that the Q11b rewrite does not raise on it"),
        ),
        pytest.param(
            '{"tenderness": -1' + "0" * 400 + "}",
            "a negative whole number too large for a float",
            id="huge negative int",
            marks=pytest.mark.no_base_failure(reason="retained behaviour: the old range test rejected a negative huge int too; this pins that the Q11b rewrite does not raise on it"),
        ),
        pytest.param(
            '{"tenderness": -1e308}',
            "a negative number whose product would overflow",
            id="negative overflow",
            marks=pytest.mark.no_base_failure(reason="retained behaviour: the row stays skipped (before: a negative; now: ten times it is not a finite number, which a multiply would store as -Infinity)"),
        ),
    ],
)
def test_a_damaged_row_is_skipped_and_not_counted_and_does_not_stop_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, bad_json: str, label: str
) -> None:
    persona = _persona(tmp_path, _memory("bad", "initiate_outbound", {"tenderness": 0.2}), _memory("good", "initiate_outbound", {"tenderness": 0.2}))
    _raw(persona, "UPDATE memories SET emotions_json = ? WHERE id = ?", bad_json, "bad")
    caplog.set_level(logging.WARNING)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1, label
    assert _read(persona, "bad")["emotions_json"] == bad_json
    assert _read(persona, "bad")["peak_emotion_intensity"] == 0.2  # not multiplied
    assert _feelings(persona, "good") == pytest.approx({"tenderness": 2.0})
    assert any("bad" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert _marker(persona) == dev_constants.NUDGE_SCALE_FIX_MARKER  # one damaged row does not stop the fix for good


@pytest.mark.parametrize(
    "damage",
    ["UPDATE memories SET score = 1e308 WHERE id = ?", "UPDATE memories SET peak_emotion_intensity = -1e308 WHERE id = ?"],
    ids=["score too large", "peak too negative"],
)
def test_a_row_whose_score_or_peak_times_ten_is_not_finite_is_skipped_with_a_warning_and_not_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, damage: str
) -> None:
    """Review F1 of Q11b: ten times these columns is an infinity, which the multiply must not store."""
    persona = _persona(tmp_path, _memory("big", "making", {"pride": 0.2}), _memory("fine", "making", {"pride": 0.3}))
    _raw(persona, damage, "big")
    before = tuple(_read(persona, "big"))
    caplog.set_level(logging.WARNING)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert tuple(_read(persona, "big")) == before  # nothing of the row was written
    assert _feelings(persona, "fine") == pytest.approx({"pride": 3.0})
    assert any("big" in r.getMessage() and "skipped" in r.getMessage() for r in caplog.records)


def test_a_row_with_a_score_that_is_not_a_number_is_skipped(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("bad", "making", {"pride": 0.2}))
    _raw(persona, "UPDATE memories SET score = ? WHERE id = ?", "lots", "bad")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 0
    assert _feelings(persona, "bad") == {"pride": 0.2}


def test_a_row_with_no_feelings_is_left_alone_and_not_counted(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    persona = _persona(tmp_path, _memory("empty", "kindled_peer", {}), _memory("some", "kindled_peer", {"warmth": 0.4}))
    caplog.set_level(logging.WARNING)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _feelings(persona, "empty") == {}
    empty = _read(persona, "empty")
    assert empty["score"] == 0.0 and empty["peak_emotion_intensity"] == 0.0  # nothing to multiply: every column as it was
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]  # a no-op, not a skipped row


@pytest.mark.parametrize("memory_type", ["kindled_peer", "making", "initiate_outbound"])
def test_a_row_with_a_negative_member_is_multiplied_whole_like_any_other_small_row(tmp_path: Path, memory_type: str) -> None:
    """Planning B18 Q11b (owner R13, one threshold for all three types): a negative member is multiplied with the rest of its row.

    The engine holds a stored negative at 0.0 before any fold, so the negative changes nothing it reads; the positive member is put
    on the 0-10 scale, which the old skip of the whole row left at a tenth of its size."""
    persona = _persona(tmp_path, _memory("mixed", memory_type, {"warmth": 0.4, "fear": -0.3}, importance=4.0))
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    # hand-worked: 0.4 x 10 = 4.0 and -0.3 x 10 = -3.0; score (0.4 - 0.3) x 10 = 1.0; peak 0.4 x 10 = 4.0
    row = _read(persona, "mixed")
    assert json.loads(row["emotions_json"]) == pytest.approx({"warmth": 4.0, "fear": -3.0})
    assert row["score"] == pytest.approx(1.0)
    assert row["peak_emotion_intensity"] == pytest.approx(4.0)
    assert row["importance"] == 4.0


def test_a_row_whose_feelings_are_all_negative_is_multiplied_whole_too(tmp_path: Path) -> None:
    """Its stored peak is negative, which is below the threshold, so the selector picks it like any small row."""
    persona = _persona(tmp_path, _memory("cold", "kindled_peer", {"fear": -0.3, "anger": -0.2}), _memory("warm", "kindled_peer", {"warmth": 0.5}))
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 2
    # hand-worked: -0.3 x 10 = -3.0, -0.2 x 10 = -2.0; score -0.5 x 10 = -5.0; peak -0.2 x 10 = -2.0
    row = _read(persona, "cold")
    assert json.loads(row["emotions_json"]) == pytest.approx({"fear": -3.0, "anger": -2.0})
    assert row["score"] == pytest.approx(-5.0)
    assert row["peak_emotion_intensity"] == pytest.approx(-2.0)
    assert _feelings(persona, "warm") == pytest.approx({"warmth": 5.0})


# --- the transaction: all or nothing, marker included -----------------------------------------


def test_a_row_count_mismatch_rolls_everything_back_marker_included(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}), _memory("b", "making", {"pride": 0.3}))
    # An update that touches no row (RAISE(IGNORE) skips it): the count the fix expects is not the count it got.
    _raw(persona, "CREATE TRIGGER skip_b BEFORE UPDATE ON memories WHEN OLD.id = 'b' BEGIN SELECT RAISE(IGNORE); END")
    with pytest.raises(RuntimeError, match="nothing was changed"):
        nudge_scale_fix.scale_up_small_feelings(persona)
    assert _unchanged(persona, "a", {"pride": 0.2}) and _unchanged(persona, "b", {"pride": 0.3})
    assert _marker(persona) == 0


def test_a_failure_part_way_through_the_updates_rolls_the_earlier_rows_back(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}), _memory("b", "making", {"pride": 0.3}))
    _raw(persona, "CREATE TRIGGER fail_b BEFORE UPDATE ON memories WHEN OLD.id = 'b' BEGIN SELECT RAISE(ABORT, 'refused'); END")
    with pytest.raises(sqlite3.DatabaseError, match="refused"):
        nudge_scale_fix.scale_up_small_feelings(persona)
    assert _unchanged(persona, "a", {"pride": 0.2}) and _unchanged(persona, "b", {"pride": 0.3})
    assert _marker(persona) == 0
    # the failure left no transaction open: with the cause gone the next run completes
    _raw(persona, "DROP TRIGGER fail_b")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 2


def test_the_marker_is_set_in_the_same_transaction_and_survives_a_reopen(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    assert _marker(persona) == 0
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _marker(persona) == dev_constants.NUDGE_SCALE_FIX_MARKER
    store = MemoryStore(persona / "memories.db")  # a fresh open of the file sees it
    try:
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == dev_constants.NUDGE_SCALE_FIX_MARKER
    finally:
        store.close()


def test_a_second_run_multiplies_nothing(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    # a row written small afterwards (an old writer) is not multiplied: the marker says the fix has run
    store = MemoryStore(persona / "memories.db")
    try:
        store.create(_memory("late", "making", {"pride": 0.2}), fold=False)
    finally:
        store.close()
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 0
    assert _feelings(persona, "a") == pytest.approx({"pride": 2.0})  # not 20
    assert _feelings(persona, "late") == {"pride": 0.2}


def test_the_method_refuses_to_run_inside_an_open_transaction(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    store = MemoryStore(persona / "memories.db")
    try:
        store._conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeError, match="open transaction"):
            store.scale_up_small_nudge_rows()
        store._conn.rollback()
    finally:
        store.close()
    assert _unchanged(persona, "a", {"pride": 0.2})


# --- the constants are read from dev_constants -------------------------------------------------


def test_the_factor_threshold_types_and_marker_come_from_dev_constants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(
        tmp_path,
        _memory("made", "making", {"pride": 0.2}),
        _memory("talk", "conversation", {"joy": 0.2}),
        _memory("mid", "making", {"pride": 1.5}),
    )
    monkeypatch.setattr(dev_constants, "NUDGE_SCALE_FIX_FACTOR", 3.0)
    monkeypatch.setattr(dev_constants, "NUDGE_SCALE_FIX_MEMORY_TYPES", ("making", "conversation"))
    monkeypatch.setattr(dev_constants, "NUDGE_SCALE_FIX_THRESHOLD", 2.0)
    monkeypatch.setattr(dev_constants, "NUDGE_SCALE_FIX_MARKER", 7)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 3
    assert _feelings(persona, "made") == pytest.approx({"pride": 0.6})  # the factor
    assert _feelings(persona, "talk") == pytest.approx({"joy": 0.6})  # the type list
    assert _feelings(persona, "mid") == pytest.approx({"pride": 4.5})  # the threshold: 1.5 is below 2.0
    assert _marker(persona) == 7  # the marker


def test_the_log_names_the_count_and_every_multiplied_id_so_the_rows_can_be_found_by_hand(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    persona = _persona(tmp_path, *[_memory(f"row-{n}", "making", {"pride": 0.2}) for n in range(5)], _memory("other", "conversation", {"joy": 0.2}))
    caplog.set_level(logging.INFO, logger="brain.memory.store")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 5
    (line,) = [r.getMessage() for r in caplog.records if "multiplied the feelings" in r.getMessage()]
    assert "feelings of 5 rows" in line
    assert all(f"row-{n}" in line for n in range(5)) and "other" not in line


class _FailingOn:
    """A stand-in for the store's connection that passes everything through and raises on one statement or on `commit`."""

    def __init__(self, real: sqlite3.Connection, *, statement: str | None = None, commit: bool = False) -> None:
        self._real, self._statement, self._commit = real, statement, commit

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def execute(self, sql: str, *args: object):
        if self._statement is not None and sql.startswith(self._statement):
            raise sqlite3.OperationalError("injected")
        return self._real.execute(sql, *args)

    def commit(self) -> None:
        if self._commit:
            raise sqlite3.OperationalError("injected commit failure")
        self._real.commit()


@pytest.mark.parametrize("fault", [{"statement": "PRAGMA user_version ="}, {"commit": True}], ids=["marker write fails", "commit fails"])
def test_the_updates_and_the_marker_are_one_transaction_a_failure_of_either_leaves_neither(tmp_path: Path, fault: dict) -> None:
    """If the marker were written after the commit, a crash between them would leave multiplied rows and no marker, and the next
    start would multiply again. The marker write is the last statement before the commit: failing it, or the commit, rolls the rows back."""
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    store = MemoryStore(persona / "memories.db")
    try:
        store._conn = _FailingOn(store._conn, **fault)
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            store.scale_up_small_nudge_rows()
    finally:
        store._conn = store._conn._real
        store.close()
    assert _unchanged(persona, "a", {"pride": 0.2})
    assert _marker(persona) == 0


# --- when the fix runs ----------------------------------------------------------------------


def test_it_is_skipped_on_a_persona_whose_switch_exists(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    assert engine_files.write_switch(persona, NOW)
    assert nudge_scale_fix.scale_up_small_feelings(persona) is None
    assert _unchanged(persona, "a", {"pride": 0.2})
    assert _marker(persona) == 0


def test_it_is_skipped_when_the_switch_cannot_be_inspected_and_the_store_is_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    opened: list[object] = []
    real = nudge_scale_fix.MemoryStore
    monkeypatch.setattr(nudge_scale_fix, "MemoryStore", lambda *a, **k: opened.append(a) or real(*a, **k))
    refuse_the_switch_stat(monkeypatch)
    assert nudge_scale_fix.scale_up_small_feelings(persona) is None
    assert opened == []
    assert _unchanged(persona, "a", {"pride": 0.2}) and _marker(persona) == 0


def test_it_does_nothing_and_creates_nothing_when_there_is_no_database(tmp_path: Path) -> None:
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_it_opens_the_store_with_the_default_integrity_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write that runs once in a persona's life is not made into a file that fails its check (decision D-12)."""
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    calls: list[dict] = []
    real = nudge_scale_fix.MemoryStore

    def spy(*args: object, **kwargs: object) -> MemoryStore:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(nudge_scale_fix, "MemoryStore", spy)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert calls == [{}]  # no `integrity_check=False`
