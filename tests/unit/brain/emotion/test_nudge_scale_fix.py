"""Engine 6, G1 and C2: the one-time x10 fix of small feelings (`nudge_scale_fix` and `MemoryStore.scale_up_small_nudge_rows`).

A fixture database is built with `MemoryStore.create(fold=False)` (the engine is not told). A damaged row is made with a direct
`sqlite3` connection in the test, which is not product code. Nothing here touches a real persona.

C2 (Planning B29): a negative feeling or peak is held to 0.0 and then multiplied; the score is multiplied as stored; a row with no
feelings is multiplied in its peak and score; no row carries a marker; the fix's progress is `PRAGMA user_version`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files, nudge_scale_fix
from brain.emotion.engine_utils import EngineState
from brain.memory.store import Memory, MemoryStore
from tests.unit.brain.emotion._engine_support import make_live, ruin_the_state

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _memory(
    memory_id: str,
    memory_type: str,
    emotions: dict[str, float],
    *,
    importance: float = 3.0,
    metadata: dict | None = None,
    score: float | None = None,
    peak: float | None = None,
) -> Memory:
    numbers = [float(v) for v in emotions.values()]
    return Memory.from_stored_row(
        id=memory_id,
        content=f"content of {memory_id}",
        memory_type=memory_type,
        domain="us",
        created_at=NOW,
        emotions=dict(emotions),
        tags=[],
        importance=importance,
        score=sum(numbers) if score is None else score,
        metadata=dict(metadata or {}),
        peak_emotion_intensity=max(numbers, default=0.0) if peak is None else peak,
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


def _metadata_text(persona: Path, memory_id: str) -> str:
    connection = sqlite3.connect(persona / "memories.db")
    try:
        return connection.execute("SELECT metadata_json FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
    finally:
        connection.close()


def _user_version(persona: Path) -> int:
    """The fix's progress note, read through the store's own method (the only reader the product has)."""
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        return store.x10_progress()
    finally:
        store.close()


def _raw_user_version(persona: Path) -> int:
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
        pytest.param("[" * 200000 + "]" * 200000, "nested past the parser's limit", id="deeply nested"),
        (json.dumps(["tenderness", 0.2]), "not a mapping"),
        (json.dumps({"tenderness": "0.2"}), "a text value"),
        (json.dumps({"tenderness": True}), "a boolean value"),
        (json.dumps({"tenderness": None}), "a null value"),
        # json.loads accepts NaN and Infinity. `_hold` cannot hold a NaN, so the row stays unchanged and logged (Planning B29 C2 item 1).
        pytest.param(
            '{"tenderness": NaN}',
            "a NaN value",
            marks=pytest.mark.no_base_failure(reason="retained behaviour: the old range test rejected NaN too; this pins that the hold keeps it"),
        ),
        pytest.param(
            '{"tenderness": Infinity}',
            "an infinite value",
            marks=pytest.mark.no_base_failure(reason="retained behaviour: the old range test rejected Infinity too; this pins that the hold keeps it"),
        ),
        # A whole number too large for a float: json.loads accepts it, and its comparison with the threshold must not raise.
        pytest.param(
            '{"tenderness": 1' + "0" * 400 + "}",
            "a whole number too large for a float",
            id="huge positive int",
            marks=pytest.mark.no_base_failure(reason="retained behaviour: the old range test rejected a huge int too; this pins that the hold does not raise on it"),
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


@pytest.mark.parametrize(
    "damage",
    ["UPDATE memories SET score = 1e308 WHERE id = ?"],
    ids=["score too large"],
)
def test_a_row_whose_score_times_ten_is_not_finite_is_skipped_with_a_warning_and_not_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, damage: str
) -> None:
    """Review F1 of Q11b: ten times this column is an infinity, which the multiply must not store."""
    persona = _persona(tmp_path, _memory("big", "making", {"pride": 0.2}), _memory("fine", "making", {"pride": 0.3}))
    _raw(persona, damage, "big")
    before = tuple(_read(persona, "big"))
    caplog.set_level(logging.WARNING)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert tuple(_read(persona, "big")) == before  # nothing of the row was written
    assert _feelings(persona, "fine") == pytest.approx({"pride": 3.0})
    assert any("big" in r.getMessage() and "skipped" in r.getMessage() for r in caplog.records)
    assert _user_version(persona) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED  # the pass finished


def test_a_row_with_a_score_that_is_not_a_number_is_skipped(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("bad", "making", {"pride": 0.2}))
    _raw(persona, "UPDATE memories SET score = ? WHERE id = ?", "lots", "bad")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 0
    assert _feelings(persona, "bad") == {"pride": 0.2}


# --- C2 item 1 and Q15: negatives held to 0.0 then x10, peak held and x10, score as stored, no skip for no feelings -----------


@pytest.mark.parametrize("memory_type", ["kindled_peer", "making", "initiate_outbound"])
def test_a_negative_member_is_held_to_zero_and_then_multiplied_with_the_rest_of_its_row(tmp_path: Path, memory_type: str) -> None:
    """Planning B29 C2 item 1 (owner 2026-10-06: negatives are converted to 0): in a row the negative member becomes 0.0, the
    positive one reaches the 0-10 scale, the score is ten times what is stored, the peak is ten times the held peak."""
    persona = _persona(tmp_path, _memory("mixed", memory_type, {"warmth": 0.4, "fear": -0.3}, importance=4.0))
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    # hand-worked: 0.4 x 10 = 4.0 and max(-0.3, 0) x 10 = 0.0; the stored score (0.4 - 0.3 = 0.1) x 10 = 1.0; peak 0.4 x 10 = 4.0
    row = _read(persona, "mixed")
    assert json.loads(row["emotions_json"]) == pytest.approx({"warmth": 4.0, "fear": 0.0})
    assert row["score"] == pytest.approx(1.0)
    assert row["peak_emotion_intensity"] == pytest.approx(4.0)
    assert row["importance"] == 4.0


def test_a_row_whose_feelings_are_all_negative_has_them_and_its_peak_held_to_zero_and_its_score_multiplied_as_stored(tmp_path: Path) -> None:
    """Its stored peak is negative, which is below the threshold, so the selector picks it like any small row."""
    persona = _persona(tmp_path, _memory("cold", "kindled_peer", {"fear": -0.3, "anger": -0.2}), _memory("warm", "kindled_peer", {"warmth": 0.5}))
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 2
    # hand-worked: both feelings held to 0.0 then x10 = 0.0; the stored peak -0.2 held to 0.0 then x10 = 0.0; the stored score -0.5 x 10 = -5.0
    row = _read(persona, "cold")
    assert json.loads(row["emotions_json"]) == {"fear": 0.0, "anger": 0.0}
    assert row["score"] == pytest.approx(-5.0)
    assert row["peak_emotion_intensity"] == 0.0
    assert _feelings(persona, "warm") == pytest.approx({"warmth": 5.0})


@pytest.mark.parametrize(
    "stored",
    ['{"tenderness": -Infinity}', '{"tenderness": -1e308}', '{"tenderness": -1' + "0" * 400 + "}"],
    ids=["negative infinity", "huge negative float", "huge negative int"],
)
def test_a_huge_negative_feeling_is_held_to_zero_and_the_row_is_multiplied(tmp_path: Path, stored: str) -> None:
    """The hold takes any negative to 0.0, including the ones whose product used to overflow: the row is multiplied, not skipped."""
    persona = _persona(tmp_path, _memory("neg", "initiate_outbound", {"tenderness": 0.2, "other": 0.1}))
    _raw(persona, "UPDATE memories SET emotions_json = ? WHERE id = ?", stored.replace("}", ', "other": 0.1}'), "neg")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _feelings(persona, "neg") == pytest.approx({"tenderness": 0.0, "other": 1.0})


def test_a_negative_peak_column_is_held_to_zero_before_it_is_multiplied(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("neg", "making", {"pride": 0.2}))
    _raw(persona, "UPDATE memories SET peak_emotion_intensity = -1e308 WHERE id = ?", "neg")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _read(persona, "neg")["peak_emotion_intensity"] == 0.0


def test_the_score_is_multiplied_as_stored_not_recomputed(tmp_path: Path) -> None:
    """Planning B29 C2 item 1 (owner 2026-10-22: "emotion values, score and peak x10"): the score is a creation-time snapshot."""
    persona = _persona(tmp_path, _memory("snap", "making", {"pride": 0.2}, score=0.7))
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _read(persona, "snap")["score"] == pytest.approx(7.0)  # 0.7 x 10, not the 2.0 of the new feelings
    assert _user_version(persona) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED


def test_a_decayed_row_with_no_feelings_still_has_its_peak_and_score_multiplied(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Planning B27 Q15: a decayed nudge row can have empty emotions while its peak (it survives the decay) and score are still on the old scale."""
    persona = _persona(
        tmp_path,
        _memory("decayed", "kindled_peer", {}, score=0.3, peak=0.3),
        _memory("blank", "kindled_peer", {}),  # nothing at all: multiplied too, 0 x 10 = 0
    )
    caplog.set_level(logging.WARNING)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 2
    decayed = _read(persona, "decayed")
    assert json.loads(decayed["emotions_json"]) == {}
    assert decayed["score"] == pytest.approx(3.0) and decayed["peak_emotion_intensity"] == pytest.approx(3.0)
    blank = _read(persona, "blank")
    assert blank["score"] == 0.0 and blank["peak_emotion_intensity"] == 0.0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]  # not a skipped row


# --- the transaction: all or nothing, progress note included -----------------------------------


@pytest.mark.no_base_failure(reason="retained behaviour: a damaged column the fix does not touch never stopped the fix on the base either; this keeps it so now that no value check reads the row back")
def test_a_damaged_column_the_fix_does_not_touch_does_not_stop_start_up(tmp_path: Path) -> None:
    """A row with broken tags or timestamps is multiplied like any other: the fix reads and writes only its own three columns."""
    persona = _persona(tmp_path, _memory("odd", "making", {"pride": 0.2}), _memory("fine", "making", {"pride": 0.3}))
    _raw(persona, "UPDATE memories SET tags_json = ?, created_at = ?, last_accessed_at = ? WHERE id = ?", "not json", "yesterday", "garbage", "odd")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 2
    assert _feelings(persona, "odd") == pytest.approx({"pride": 2.0})
    assert _user_version(persona) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED


def test_a_failure_part_way_through_the_updates_rolls_the_earlier_rows_back(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}), _memory("b", "making", {"pride": 0.3}))
    _raw(persona, "CREATE TRIGGER fail_b BEFORE UPDATE ON memories WHEN OLD.id = 'b' BEGIN SELECT RAISE(ABORT, 'refused'); END")
    with pytest.raises(sqlite3.DatabaseError, match="refused"):
        nudge_scale_fix.scale_up_small_feelings(persona)
    assert _unchanged(persona, "a", {"pride": 0.2}) and _unchanged(persona, "b", {"pride": 0.3})
    assert _user_version(persona) == 0
    # the failure left no transaction open: with the cause gone the next run completes
    _raw(persona, "DROP TRIGGER fail_b")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 2


def test_no_row_carries_a_marker_and_the_progress_note_is_the_version_number(tmp_path: Path) -> None:
    """Owner S77 ruled out per-row markers; S78: "yes use the version number for a half finished repair"."""
    persona = _persona(
        tmp_path,
        _memory("a", "making", {"pride": 0.2}, metadata={"keep": "me", "n": 3}),
        _memory("b", "kindled_peer", {"warmth": 0.4}),
        _memory("big", "making", {"pride": 4.0}),
        _memory("talk", "conversation", {"joy": 0.2}, metadata={"keep": "too"}),
    )
    texts = {memory_id: _metadata_text(persona, memory_id) for memory_id in ("a", "b", "big", "talk")}
    store = MemoryStore(persona / "memories.db")
    try:
        assert store.x10_progress() == 0  # a fresh database: nothing done
        assert store.scale_up_small_nudge_rows() == 2
        assert store.x10_progress() == dev_constants.NUDGE_SCALE_FIX_PROGRESS_ROWS_DONE  # 1, set in the rows' own transaction
    finally:
        store.close()
    assert {memory_id: _metadata_text(persona, memory_id) for memory_id in texts} == texts  # metadata is as it was, byte for byte
    assert _raw_user_version(persona) == 1  # the same number, read straight from the file


def test_a_second_run_skips_the_rows_because_the_progress_note_says_they_are_done(tmp_path: Path) -> None:
    """The retry case. A row written between the two runs (a writer on old code) is missed: the owner's accepted edge (S77)."""
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    store = MemoryStore(persona / "memories.db")
    try:
        assert store.scale_up_small_nudge_rows() == 1
        store.create(_memory("late", "making", {"pride": 0.2}), fold=False)
        assert store.scale_up_small_nudge_rows() == 0  # the number is 1: skipped
    finally:
        store.close()
    assert _feelings(persona, "a") == pytest.approx({"pride": 2.0})  # not 20
    assert _feelings(persona, "late") == {"pride": 0.2}  # the accepted edge


def test_a_database_whose_progress_note_is_already_set_has_nothing_multiplied(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    _raw(persona, "PRAGMA user_version = 1")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 0
    assert _unchanged(persona, "a", {"pride": 0.2})


def test_two_starters_at_once_multiply_each_row_exactly_once(tmp_path: Path) -> None:
    """The bridge and a CLI command can start together: the database lock serialises them and the second reads the number the first set."""
    persona = _persona(tmp_path, *[_memory(f"r{n}", "making", {"pride": 0.2}) for n in range(8)])
    barrier = threading.Barrier(2)
    counts: list[int | None] = []
    errors: list[BaseException] = []

    def start() -> None:
        try:
            barrier.wait(10)
            counts.append(nudge_scale_fix.scale_up_small_feelings(persona))
        except BaseException as exc:  # noqa: BLE001 - reported below, from the main thread
            errors.append(exc)

    threads = [threading.Thread(target=start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors and sorted(c or 0 for c in counts) == [0, 8]
    assert _user_version(persona) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED  # one starter set it, the other read it
    assert all(_feelings(persona, f"r{n}") == pytest.approx({"pride": 2.0}) for n in range(8))  # once, not 20


@pytest.mark.parametrize("stored", ["", "null", "{}"], ids=["empty text", "json null", "empty mapping"])
def test_a_row_with_no_metadata_is_multiplied_like_any_other_and_its_metadata_is_left_as_stored(tmp_path: Path, stored: str) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    _raw(persona, "UPDATE memories SET metadata_json = ? WHERE id = ?", stored, "a")
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _metadata_text(persona, "a") == stored
    assert _feelings(persona, "a") == pytest.approx({"pride": 2.0})


@pytest.mark.parametrize(
    "stored",
    ["not json {", "[1, 2]", '"text"', "7", "[" * 200000 + "]" * 200000],
    ids=["unparseable", "a list", "a string", "a number", "nested past the parser's limit"],
)
def test_a_row_whose_metadata_cannot_be_read_as_a_mapping_is_not_multiplied_and_is_logged_and_the_count_accounts_for_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, stored: str
) -> None:
    """Planning B29 C2 item 3 (P4): detected with `_metadata_or_none`; the returned count does not include it."""
    persona = _persona(tmp_path, _memory("odd", "making", {"pride": 0.2}), _memory("fine", "making", {"pride": 0.3}))
    _raw(persona, "UPDATE memories SET metadata_json = ? WHERE id = ?", stored, "odd")
    caplog.set_level(logging.WARNING)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _metadata_text(persona, "odd") == stored
    assert _feelings(persona, "odd") == {"pride": 0.2}
    assert _feelings(persona, "fine") == pytest.approx({"pride": 3.0})
    assert any("odd" in r.getMessage() and "skipped" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert _user_version(persona) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED  # the skipped row does not hold the note back


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


def test_the_factor_threshold_and_types_come_from_dev_constants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(
        tmp_path,
        _memory("made", "making", {"pride": 0.2}),
        _memory("talk", "conversation", {"joy": 0.2}),
        _memory("mid", "making", {"pride": 1.5}),
    )
    monkeypatch.setattr(dev_constants, "NUDGE_SCALE_FIX_FACTOR", 3.0)
    monkeypatch.setattr(dev_constants, "NUDGE_SCALE_FIX_MEMORY_TYPES", ("making", "conversation"))
    monkeypatch.setattr(dev_constants, "NUDGE_SCALE_FIX_THRESHOLD", 2.0)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 3
    assert _feelings(persona, "made") == pytest.approx({"pride": 0.6})  # the factor
    assert _feelings(persona, "talk") == pytest.approx({"joy": 0.6})  # the type list
    assert _feelings(persona, "mid") == pytest.approx({"pride": 4.5})  # the threshold: 1.5 is below 2.0
    assert _user_version(persona) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED


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


@pytest.mark.parametrize("fault", [{"statement": "UPDATE memories SET"}, {"commit": True}], ids=["update fails", "commit fails"])
def test_the_values_and_the_progress_note_are_one_write_a_failure_of_the_update_or_the_commit_leaves_neither(tmp_path: Path, fault: dict) -> None:
    """The number is set in the same transaction as the multiplied values, so a crash cannot leave a multiplied row with number 0
    (the next start would multiply it again): failing the update, or the commit, leaves the row as it was and the number at 0."""
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
    assert _user_version(persona) == 0


# --- when the fix runs ----------------------------------------------------------------------


def test_it_is_skipped_on_a_persona_whose_state_holds_the_switch_over_time(tmp_path: Path) -> None:
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    make_live(persona, NOW)
    assert nudge_scale_fix.scale_up_small_feelings(persona) is None
    assert _unchanged(persona, "a", {"pride": 0.2})
    assert _user_version(persona) == 0


def test_it_runs_on_a_persona_with_a_state_that_holds_no_switch_over_time(tmp_path: Path) -> None:
    """A shadow state (what the engine wrote before the switch-over) is not a switch-over."""
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    engine_files.save_state(persona, EngineState(saved_at=NOW))
    assert engine_files.switch_over_state(engine_files.read_state(persona)) is False  # the rule the fix asks
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _user_version(persona) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED  # no queue file: the queue step is done all the same


def test_it_runs_when_the_state_cannot_be_read_and_the_progress_note_decides(tmp_path: Path) -> None:
    """Planning B31 SW4: an unreadable state cannot say whether the persona was switched, so the database's own record
    decides: the steps read the progress number themselves. Unstarted (0) runs; finished (2) multiplies nothing again."""
    persona = _persona(tmp_path, _memory("a", "making", {"pride": 0.2}))
    engine_files.save_state(persona, EngineState(saved_at=NOW))
    ruin_the_state(persona)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert _user_version(persona) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 0  # the same start again: nothing is multiplied twice


def test_with_no_database_both_halves_are_skipped_and_logged_and_nothing_is_created(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Planning B29 C2 item 4: there is nothing to multiply and nowhere to record progress. A queue entry that waits is left too."""
    queue = tmp_path / "pending_candidates.jsonl"
    queue.write_bytes(b'{"id": "q", "memory_type": "making", "emotions": {"joy": 0.2}, "score": 0.2}\n')
    caplog.set_level(logging.WARNING)
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) is None
    assert [p.name for p in tmp_path.iterdir()] == ["pending_candidates.jsonl"]
    assert queue.read_bytes() == b'{"id": "q", "memory_type": "making", "emotions": {"joy": 0.2}, "score": 0.2}\n'
    assert any("no memories.db" in r.getMessage() and "skipped" in r.getMessage() for r in caplog.records)


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
