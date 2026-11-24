"""The check on the feeling names in `emotion_state.json` at bridge start-up (#801 section 7, acceptance criterion 8).

`engine_utils.merge_emotion_states` (the merge of two saved feelings that land on one name) and
`engine_state_names.check_state_names` / `run_state_names_pass` (the pass), on synthetic personas in tmp. The pass is the owner's, so
the tests set the bridge-owner flag (the registry is restored after each test by the autouse fixture in tests/conftest.py).
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files, engine_state_names, vocabulary, vocabulary_access
from brain.emotion.engine_utils import (
    Average,
    EmotionState,
    EngineState,
    RepeatEntry,
    RunRecord,
    SnapshotDay,
    merge_emotion_states,
)
from tests.unit.brain.emotion._engine_support import NOW, OtherHolder, save_as_backup

FILE = "emotion_vocabulary.json"
D = [(date(2026, 5, 1) + timedelta(days=n)).isoformat() for n in range(60)]


def _emotion(short: tuple[float, float], long: tuple[float, float], day: str | None = None, snapshot: list[SnapshotDay] | None = None) -> EmotionState:
    return EmotionState(short=Average(*short), long=Average(*long), last_entry_day=day, snapshot=list(snapshot or []))


def _snap(day: str, short: float, long: float, felt: bool = False, run: RunRecord | None = None) -> SnapshotDay:
    return SnapshotDay(day, short, long, felt, run)


# ---------------------------------------------------------------------------------------------
# The merge of two saved feelings
# ---------------------------------------------------------------------------------------------


def test_the_averages_are_the_weighted_means_with_the_weights_added() -> None:
    merged = merge_emotion_states(_emotion((4.0, 1.0), (2.0, 1.0)), _emotion((8.0, 3.0), (6.0, 1.0)))
    assert merged.short == Average(7.0, 4.0)
    assert merged.long == Average(4.0, 2.0)


def test_two_weights_of_zero_keep_the_first_and_a_first_weight_of_zero_takes_the_second() -> None:
    both_zero = merge_emotion_states(_emotion((4.0, 0.0), (2.0, 0.0)), _emotion((8.0, 0.0), (6.0, 0.0)))
    assert both_zero.short == Average(4.0, 0.0)
    assert both_zero.long == Average(2.0, 0.0)
    first_zero = merge_emotion_states(_emotion((4.0, 0.0), (2.0, 0.0)), _emotion((8.0, 2.0), (6.0, 2.0)))
    assert first_zero.short == Average(8.0, 2.0)
    assert first_zero.long == Average(6.0, 2.0)


def test_the_later_last_entry_day_is_kept_and_a_missing_one_loses() -> None:
    assert merge_emotion_states(_emotion((1.0, 1.0), (1.0, 1.0), D[3]), _emotion((1.0, 1.0), (1.0, 1.0), D[9])).last_entry_day == D[9]
    assert merge_emotion_states(_emotion((1.0, 1.0), (1.0, 1.0), D[9]), _emotion((1.0, 1.0), (1.0, 1.0), D[3])).last_entry_day == D[9]
    assert merge_emotion_states(_emotion((1.0, 1.0), (1.0, 1.0), None), _emotion((1.0, 1.0), (1.0, 1.0), D[3])).last_entry_day == D[3]
    assert merge_emotion_states(_emotion((1.0, 1.0), (1.0, 1.0), None), _emotion((1.0, 1.0), (1.0, 1.0), None)).last_entry_day is None


def test_a_snapshot_day_in_both_is_averaged_and_felt_if_either_and_the_runs_are_recomputed() -> None:
    first = _emotion((1.0, 1.0), (1.0, 1.0), snapshot=[_snap(D[0], 1.0, 1.0), _snap(D[1], 2.0, 2.0, felt=True), _snap(D[2], 3.0, 3.0)])
    second = _emotion((1.0, 1.0), (1.0, 1.0), snapshot=[_snap(D[1], 6.0, 4.0), _snap(D[3], 5.0, 5.0, felt=True)])
    merged = merge_emotion_states(first, second)
    assert [(d.day, d.short, d.long, d.felt) for d in merged.snapshot] == [
        (D[0], 1.0, 1.0, False),
        (D[1], 4.0, 3.0, True),
        (D[2], 3.0, 3.0, False),
        (D[3], 5.0, 5.0, True),
    ]
    # D1 is +2 on D0 (a run starts), D2 is a zero move (the first day against it), D3 is +2 (it carries on to D3).
    assert [d.run for d in merged.snapshot] == [
        None,
        RunRecord(D[1], 1, D[1]),
        RunRecord(D[1], 1, D[1]),
        RunRecord(D[1], 1, D[3]),
    ]


def test_a_calendar_day_between_two_snapshot_days_is_a_zero_move_in_the_recomputed_run() -> None:
    first = _emotion((1.0, 1.0), (1.0, 1.0), snapshot=[_snap(D[0], 1.0, 1.0), _snap(D[1], 2.0, 2.0)])
    second = _emotion((1.0, 1.0), (1.0, 1.0), snapshot=[_snap(D[4], 3.0, 3.0)])
    merged = merge_emotion_states(first, second)
    assert [d.day for d in merged.snapshot] == [D[0], D[1], D[4]]
    # D1 starts a rising run; D2 and D3 (off days) are two zero moves against it, which end it; D4 starts a new run.
    assert [d.run for d in merged.snapshot] == [None, RunRecord(D[1], 1, D[1]), RunRecord(D[4], 1, D[4])]


def test_the_oldest_kept_day_keeps_the_run_it_was_stored_with_the_first_ones_before_the_second_ones() -> None:
    stored = RunRecord(D[0], 1, D[2])
    other = RunRecord(D[1], -1, D[2])
    first = _emotion((1.0, 1.0), (1.0, 1.0), snapshot=[_snap(D[2], 1.0, 1.0, run=stored)])
    second = _emotion((1.0, 1.0), (1.0, 1.0), snapshot=[_snap(D[2], 1.0, 1.0, run=other), _snap(D[3], 2.0, 2.0)])
    merged = merge_emotion_states(first, second)
    assert merged.snapshot[0].run == stored
    assert merged.snapshot[1].run == RunRecord(D[0], 1, D[3])  # the carried run goes on from the stored record
    only_second = merge_emotion_states(_emotion((1.0, 1.0), (1.0, 1.0), snapshot=[_snap(D[2], 1.0, 1.0)]), second)
    assert only_second.snapshot[0].run == other


def test_the_window_is_applied_to_the_merged_snapshot_from_its_latest_day() -> None:
    window = int(dev_constants.EMOTION_LONG_HALF_LIFE_D)
    first = _emotion((1.0, 1.0), (1.0, 1.0), snapshot=[_snap(D[0], 1.0, 1.0), _snap(D[window - 1], 1.0, 1.0)])
    second = _emotion((1.0, 1.0), (1.0, 1.0), snapshot=[_snap(D[window], 1.0, 1.0)])
    merged = merge_emotion_states(first, second)
    # D[window] is the latest day: D[0] is `window` days before it and goes; D[window - 1] stays.
    assert [d.day for d in merged.snapshot] == [D[window - 1], D[window]]


def test_the_merge_changes_neither_input() -> None:
    first = _emotion((4.0, 1.0), (2.0, 1.0), D[1], [_snap(D[0], 1.0, 1.0), _snap(D[1], 2.0, 2.0)])
    second = _emotion((8.0, 3.0), (6.0, 1.0), D[2], [_snap(D[1], 4.0, 4.0)])
    before = (repr(first), repr(second))
    merge_emotion_states(first, second)
    assert (repr(first), repr(second)) == before


# ---------------------------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _owner(monkeypatch: pytest.MonkeyPatch) -> None:
    vocabulary.set_bridge_owner(True)
    vocabulary_access._ADD_RETRY_AFTER.clear()
    yield
    vocabulary_access._ADD_RETRY_AFTER.clear()


def _persona(tmp_path: Path, emotions: dict[str, EmotionState], entries: list[dict] | None = None, **fields: object) -> Path:
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / FILE).write_text(json.dumps({"version": 1, "emotions": entries or []}), encoding="utf-8")
    base = {
        "saved_at": NOW,
        "last_update": NOW,
        "active_day": D[5],
        "persona_start_at": NOW - timedelta(days=60),
        "queue_line": 7,
        "rows_through": 5,
        "repeats": {"mem-1": RepeatEntry(NOW, 2)},
        "switch_over_at": NOW - timedelta(days=30),
    }
    base.update(fields)
    assert engine_files.save_state(persona, EngineState(emotions=emotions, **base), verify=True)
    return persona


def _entry(name: str) -> dict:
    return {"name": name, "description": "a real description", "category": "persona_extension", "decay_half_life_days": 45.0, "intensity_clamp": 10.0}


def _state(persona: Path) -> EngineState:
    loaded = engine_files.read_state(persona)
    assert loaded.state is not None, loaded.problem
    return loaded.state


def _bytes(persona: Path) -> bytes:
    return engine_files.state_path(persona).read_bytes()


def test_a_saved_state_with_joy_and_Joy_ends_with_one_joy_key_holding_the_weighted_means(tmp_path: Path) -> None:
    persona = _persona(tmp_path, {"joy": _emotion((4.0, 1.0), (2.0, 1.0), D[2]), "Joy": _emotion((8.0, 3.0), (6.0, 1.0), D[4])})
    result = engine_state_names.check_state_names(persona)
    assert (result.renamed, result.merged, result.dropped, result.written) == (1, 1, 0, True)
    emotions = _state(persona).emotions
    assert list(emotions) == ["joy"]
    assert emotions["joy"].short == Average(7.0, 4.0)
    assert emotions["joy"].long == Average(4.0, 2.0)
    assert emotions["joy"].last_entry_day == D[4]


def test_a_key_the_gate_refuses_is_dropped_and_the_others_stay(tmp_path: Path) -> None:
    persona = _persona(
        tmp_path,
        {"joy": _emotion((4.0, 1.0), (2.0, 1.0)), "---": _emotion((1.0, 1.0), (1.0, 1.0)), "x" * 101: _emotion((1.0, 1.0), (1.0, 1.0))},
    )
    result = engine_state_names.check_state_names(persona)
    assert (result.dropped, result.written) == (2, True)
    assert list(_state(persona).emotions) == ["joy"]


def test_a_new_name_that_passes_the_gate_is_added_to_her_vocabulary_and_kept_under_its_tidied_form(tmp_path: Path) -> None:
    persona = _persona(tmp_path, {"Quiet  Dread": _emotion((3.0, 1.0), (3.0, 1.0))})
    assert vocabulary.get("quiet_dread") is None
    result = engine_state_names.check_state_names(persona)
    assert (result.renamed, result.written) == (1, True)
    assert list(_state(persona).emotions) == ["quiet_dread"]
    assert vocabulary.get("quiet_dread") is not None
    names = [e["name"] for e in json.loads((persona / FILE).read_text(encoding="utf-8"))["emotions"]]
    assert names == ["quiet_dread"]


def test_a_key_matching_an_entry_with_a_non_tidied_spelling_is_kept_under_the_tidied_form_of_that_spelling(tmp_path: Path) -> None:
    persona = _persona(tmp_path, {"comfort_seeking": _emotion((3.0, 1.0), (3.0, 1.0))}, entries=[_entry("Comfort Seeking")])
    from brain.emotion.persona_loader import load_persona_vocabulary

    load_persona_vocabulary(persona / FILE)
    before = _bytes(persona)
    result = engine_state_names.check_state_names(persona)
    assert result == engine_state_names.StateNamesResult()
    assert _bytes(persona) == before  # nothing changed, so nothing was written
    assert list(_state(persona).emotions) == ["comfort_seeking"]
    names = [e["name"] for e in json.loads((persona / FILE).read_text(encoding="utf-8"))["emotions"]]
    assert names == ["Comfort Seeking"]  # her file keeps its own spelling


def test_a_key_spelled_another_way_is_saved_under_the_tidied_form_of_the_entry_spelling(tmp_path: Path) -> None:
    persona = _persona(tmp_path, {"comfort seeking": _emotion((3.0, 1.0), (3.0, 1.0))}, entries=[_entry("Comfort Seeking")])
    from brain.emotion.persona_loader import load_persona_vocabulary

    load_persona_vocabulary(persona / FILE)
    result = engine_state_names.check_state_names(persona)
    assert (result.renamed, result.merged, result.written) == (1, 0, True)
    assert list(_state(persona).emotions) == ["comfort_seeking"]
    names = [e["name"] for e in json.loads((persona / FILE).read_text(encoding="utf-8"))["emotions"]]
    assert names == ["Comfort Seeking"]


def test_a_state_with_nothing_to_change_is_not_rewritten_and_makes_no_backup(tmp_path: Path) -> None:
    persona = _persona(tmp_path, {"joy": _emotion((4.0, 1.0), (2.0, 1.0)), "grief": _emotion((1.0, 1.0), (1.0, 1.0))})
    before = _bytes(persona)
    assert engine_state_names.check_state_names(persona) == engine_state_names.StateNamesResult()
    assert _bytes(persona) == before
    assert not engine_files.state_path(persona).with_name(engine_files.state_path(persona).name + ".bak1").exists()


def test_a_write_carries_every_other_field_unchanged_saved_at_included(tmp_path: Path) -> None:
    persona = _persona(tmp_path, {"joy": _emotion((4.0, 1.0), (2.0, 1.0)), "Joy": _emotion((8.0, 3.0), (6.0, 1.0))})
    before = _state(persona)
    engine_state_names.check_state_names(persona)
    after = _state(persona)
    for field in ("saved_at", "last_update", "active_day", "persona_start_at", "switch_over_at", "queue_line", "rows_through", "reset_at"):
        assert getattr(after, field) == getattr(before, field), field
    assert after.repeats == before.repeats
    assert after.repeats["mem-1"].count == 2


def test_a_state_read_from_a_backup_copy_is_left_alone(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / FILE).write_text('{"version": 1, "emotions": []}', encoding="utf-8")
    state = EngineState(saved_at=NOW, last_update=NOW, active_day=D[5], emotions={"Joy": _emotion((4.0, 1.0), (2.0, 1.0))})
    save_as_backup(persona, state)
    before = _bytes(persona)
    with caplog.at_level(logging.WARNING):
        assert engine_state_names.check_state_names(persona) == engine_state_names.StateNamesResult()
    assert _bytes(persona) == before
    assert any("older copy" in r.getMessage() for r in caplog.records)


def test_no_state_and_an_unreadable_state_are_left_alone(tmp_path: Path) -> None:
    persona = tmp_path / "persona"
    persona.mkdir()
    assert engine_state_names.check_state_names(persona) == engine_state_names.StateNamesResult()
    assert not engine_files.state_path(persona).exists()
    engine_files.state_path(persona).parent.mkdir(parents=True, exist_ok=True)
    engine_files.state_path(persona).write_bytes(b"\x00 not json")
    assert engine_state_names.check_state_names(persona) == engine_state_names.StateNamesResult()
    assert _bytes(persona) == b"\x00 not json"


def test_a_name_the_writer_could_not_add_is_kept_under_its_tidied_form(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path, {"Odd  Name": _emotion((3.0, 1.0), (3.0, 1.0))})
    monkeypatch.setattr(vocabulary_access, "names_used", lambda _dir, names: dict.fromkeys(names, "odd_name"))  # `cannot_write` answers the tidied form
    result = engine_state_names.check_state_names(persona)
    assert (result.renamed, result.written) == (1, True)
    assert list(_state(persona).emotions) == ["odd_name"]


def test_the_pass_asks_for_all_the_keys_in_one_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path, {"joy": _emotion((4.0, 1.0), (2.0, 1.0)), "grief": _emotion((1.0, 1.0), (1.0, 1.0)), "Quiet Dread": _emotion((1.0, 1.0), (1.0, 1.0))})
    calls: list[list[str]] = []
    real = vocabulary_access.names_used
    monkeypatch.setattr(vocabulary_access, "names_used", lambda d, names: (calls.append(list(names)), real(d, names))[1])
    engine_state_names.check_state_names(persona)
    assert calls == [["joy", "grief", "Quiet Dread"]]


def test_a_lock_wait_that_runs_out_leaves_the_state_alone_and_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    persona = _persona(tmp_path, {"Joy": _emotion((4.0, 1.0), (2.0, 1.0))})
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.05)
    before = _bytes(persona)
    holder = OtherHolder(persona)
    try:
        with caplog.at_level(logging.WARNING):
            assert engine_state_names.run_state_names_pass(persona) == engine_state_names.StateNamesResult()
    finally:
        holder.release()
    assert _bytes(persona) == before
    assert any("lock did not come" in r.getMessage() for r in caplog.records)


def test_a_save_that_fails_its_read_back_is_logged_and_reported_as_not_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    persona = _persona(tmp_path, {"Joy": _emotion((4.0, 1.0), (2.0, 1.0))})
    calls: list[bool] = []

    def failing(_persona_dir: Path, _state: EngineState, *, verify: bool = False) -> bool:
        calls.append(verify)
        return False

    monkeypatch.setattr(engine_files, "save_state", failing)
    with caplog.at_level(logging.ERROR):
        result = engine_state_names.run_state_names_pass(persona)
    assert calls == [True]  # the write is always a verified one
    assert result.written is False
    assert any("could not be saved" in r.getMessage() for r in caplog.records)


def test_a_raise_inside_the_pass_is_logged_and_the_state_is_left_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    persona = _persona(tmp_path, {"Joy": _emotion((4.0, 1.0), (2.0, 1.0))})
    before = _bytes(persona)

    def boom(*_args: object, **_kwargs: object) -> dict:
        raise RuntimeError("the writer broke")

    monkeypatch.setattr(vocabulary_access, "names_used", boom)
    with caplog.at_level(logging.ERROR):
        assert engine_state_names.run_state_names_pass(persona) == engine_state_names.StateNamesResult()
    assert _bytes(persona) == before
    assert any("the pass failed" in r.getMessage() for r in caplog.records)


def test_the_state_lock_is_held_while_the_names_are_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path, {"Joy": _emotion((4.0, 1.0), (2.0, 1.0))})
    held: list[bool] = []
    real = vocabulary_access.names_used

    def spy(persona_dir: Path, names: list[str]) -> dict:
        monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.01)
        try:
            with engine_files.state_lock(persona_dir):
                held.append(False)
        except OSError:
            held.append(True)  # a second taker times out: this call is inside the pass's lock hold
        return real(persona_dir, names)

    monkeypatch.setattr(vocabulary_access, "names_used", spy)
    engine_state_names.check_state_names(persona)
    assert held == [True]


# ---------------------------------------------------------------------------------------------
# "Her vocabulary has loaded" (the bridge's start-up helper)
# ---------------------------------------------------------------------------------------------


def test_the_start_up_load_answers_true_for_a_loaded_vocabulary_and_for_a_fresh_persona(tmp_path: Path) -> None:
    from brain.bridge.server import ensure_persona_vocabulary_loaded
    from brain.memory.store import MemoryStore

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    healthy = tmp_path / "healthy"
    healthy.mkdir()
    (healthy / FILE).write_text(json.dumps({"version": 1, "emotions": [_entry("quiet_dread")]}), encoding="utf-8")
    store = MemoryStore(":memory:")
    try:
        assert ensure_persona_vocabulary_loaded(fresh, store=store) is True
        assert ensure_persona_vocabulary_loaded(healthy, store=store) is True
    finally:
        store.close()


def test_the_start_up_load_answers_false_when_the_load_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.bridge.server import ensure_persona_vocabulary_loaded
    from brain.emotion import persona_loader
    from brain.memory.store import MemoryStore

    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / FILE).write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")

    def broken(*_args: object, **_kwargs: object) -> tuple:
        raise OSError("unreadable")

    monkeypatch.setattr(persona_loader, "load_persona_vocabulary_with_anomaly", broken)
    store = MemoryStore(":memory:")
    try:
        assert ensure_persona_vocabulary_loaded(persona, store=store) is False
    finally:
        store.close()


def test_the_start_up_load_answers_false_for_a_corrupt_file_that_was_reset_and_true_once_it_was_rebuilt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.bridge.server import ensure_persona_vocabulary_loaded
    from brain.emotion import persona_loader
    from brain.health.anomaly import BrainAnomaly
    from brain.memory.store import MemoryStore

    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / FILE).write_text(json.dumps({"version": 1, "emotions": []}), encoding="utf-8")

    def answering(action: str):
        anomaly = BrainAnomaly(timestamp="2026-11-24T00:00:00Z", file=str(persona / FILE), kind="json_parse_error", action=action, quarantine_path=None, likely_cause="test", detail="test")
        return lambda *_a, **_k: (0, anomaly)

    store = MemoryStore(":memory:")
    try:
        monkeypatch.setattr(persona_loader, "load_persona_vocabulary_with_anomaly", answering("reset_to_default"))
        assert ensure_persona_vocabulary_loaded(persona, store=store) is False
        monkeypatch.setattr(persona_loader, "load_persona_vocabulary_with_anomaly", answering("reconstructed_from_memories"))
        assert ensure_persona_vocabulary_loaded(persona, store=store) is True
    finally:
        store.close()
