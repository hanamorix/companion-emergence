"""The emotion engine's queue and change log (Engine 2, #577).

The queue (`state/emotion_pending.json`) is a JSON list of events waiting to be folded; the change log
(`state/emotion_changes/YYYY-MM-DD.jsonl`) is a crash-recovery record of chat-derived changes. These are the
file-level cases: the per-event check, the queue file's read/append/write/clear, the change log's append, list,
prune and tolerant reader. Cases that need a hook or a drain (fold order, early-out, wiring) belong to the slices that
build them and extend this file.
"""

from __future__ import annotations

import copy
import json
import logging
import pathlib
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files, engine_utils
from brain.emotion.engine_utils import EngineState, Event, day_of, fold_event
from brain.health import json_io

LOGGER = "brain.emotion.engine_files"
NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)
AT = "2026-06-10T10:00:00Z"


def _errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _raw(which: str = "commit", **over: object) -> dict:
    base: dict[str, dict] = {
        "commit": {
            "kind": "commit",
            "at": AT,
            "memory_id": "m1",
            "emotions": {"joy": 6.2},
            "modifier": 1.0,
            "climax": 0.0,
        },
        "assigned": {"kind": "assigned", "at": AT, "memory_id": "m2", "emotions": {"joy": 3.0}, "modifier": 0.55},
        "showing": {"kind": "showing", "at": AT, "memory_id": "m3", "emotions": {"grief": 1.5}, "modifier": 1.0},
        "change": {"kind": "change", "at": AT, "source": "pass2", "emotions": {"joy": -2.0}},
        "start": {"kind": "start", "at": AT},
    }
    out = dict(base[which])
    out.update(over)
    return out


def _drop(raw: dict, key: str) -> dict:
    out = dict(raw)
    del out[key]
    return out


def _queue_file(persona_dir: Path) -> Path:
    path = engine_files.queue_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_raw_queue(persona_dir: Path, content: object) -> Path:
    path = _queue_file(persona_dir)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


def _change(at: datetime, **emotions: float) -> Event:
    return Event(kind="change", at=at, emotions=emotions or {"joy": 1.0}, source="pass2")


def _tree(root: Path) -> dict[str, bytes | None]:
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None) for p in sorted(root.rglob("*"))
    }


# --- the per-event check: what is valid ----------------------------------------------------------


@pytest.mark.parametrize("kind", ["commit", "assigned", "showing", "change", "start"])
def test_every_kind_round_trips_through_its_json_form(kind: str) -> None:
    raw = _raw(kind)
    event = engine_files.event_from_json(raw)
    assert event.kind == kind
    assert event.at == datetime(2026, 6, 10, 10, 0, tzinfo=UTC)
    assert engine_files.event_to_json(event) == raw


def test_the_event_fields_come_back_as_given() -> None:
    commit = engine_files.event_from_json(_raw("commit", modifier=0.55, climax=8.5))
    assert (commit.memory_id, dict(commit.emotions), commit.modifier, commit.climax) == (
        "m1",
        {"joy": 6.2},
        0.55,
        8.5,
    )
    change = engine_files.event_from_json(_raw("change"))
    assert (change.source, dict(change.emotions)) == ("pass2", {"joy": -2.0})


def test_a_change_event_has_no_modifier_in_its_json_and_a_stored_one_is_ignored() -> None:
    assert "modifier" not in engine_files.event_to_json(_change(NOW))
    event = engine_files.event_from_json(_raw("change", modifier=0.2))
    assert event.modifier == 1.0
    assert "modifier" not in engine_files.event_to_json(event)


def test_a_missing_modifier_is_one_and_a_missing_climax_is_zero() -> None:
    event = engine_files.event_from_json(_drop(_drop(_raw("commit"), "modifier"), "climax"))
    assert (event.modifier, event.climax) == (1.0, 0.0)
    assert engine_files.event_from_json(_drop(_raw("showing"), "modifier")).modifier == 1.0


def test_a_showing_or_change_with_no_emotions_key_is_valid_and_empty() -> None:
    assert dict(engine_files.event_from_json(_drop(_raw("showing"), "emotions")).emotions) == {}
    assert dict(engine_files.event_from_json(_drop(_raw("change"), "emotions")).emotions) == {}


def test_a_change_without_a_source_is_valid() -> None:
    assert engine_files.event_from_json(_drop(_raw("change"), "source")).source is None


def test_naive_and_offset_times_are_read_as_utc_instants() -> None:
    naive = engine_files.event_from_json(_raw("start", at="2026-06-10T10:00:00"))
    offset = engine_files.event_from_json(_raw("start", at="2026-06-10T12:00:00+02:00"))
    assert naive.at == offset.at == datetime(2026, 6, 10, 10, 0, tzinfo=UTC)


def test_event_to_json_writes_utc_z_times_for_any_zone() -> None:
    zone = datetime(2026, 6, 10, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    assert engine_files.event_to_json(Event(kind="start", at=zone))["at"] == "2026-06-10T10:00:00Z"


# --- the per-event check: what is invalid ------------------------------------------------------

_INVALID = {
    "not an object": ("an event", "not an object"),
    "a list": (["commit"], "not an object"),
    "unknown kind": (_raw("commit", kind="dream"), "kind"),
    "missing kind": (_drop(_raw("commit"), "kind"), "kind"),
    "kind not text": (_raw("commit", kind=3), "kind"),
    "missing at": (_drop(_raw("commit"), "at"), "at"),
    "at null": (_raw("commit", at=None), "at"),
    "at a number": (_raw("commit", at=1717000000), "at"),
    "at not a time": (_raw("commit", at="soon"), "at"),
    "start without at": (_drop(_raw("start"), "at"), "at"),
    "commit without memory_id": (_drop(_raw("commit"), "memory_id"), "memory_id"),
    "assigned without memory_id": (_drop(_raw("assigned"), "memory_id"), "memory_id"),
    "showing without memory_id": (_drop(_raw("showing"), "memory_id"), "memory_id"),
    "empty memory_id": (_raw("showing", memory_id=""), "memory_id"),
    "memory_id a list": (_raw("commit", memory_id=["a"]), "memory_id"),
    "memory_id a dict": (_raw("commit", memory_id={"a": 1}), "memory_id"),
    "memory_id a number": (_raw("commit", memory_id=7), "memory_id"),
    "commit with no emotions": (_raw("commit", emotions={}), "emotion"),
    "commit without the emotions key": (_drop(_raw("commit"), "emotions"), "emotion"),
    "assigned with no emotions": (_raw("assigned", emotions={}), "emotion"),
    "emotions not a mapping": (_raw("showing", emotions=[["joy", 1]]), "emotion"),
    "emotions a string": (_raw("change", emotions="joy"), "emotion"),
    "non-finite amount (nan)": (_raw("commit", emotions={"joy": float("nan")}), "joy"),
    "non-finite amount (inf)": (_raw("change", emotions={"joy": float("inf")}), "joy"),
    "amount a string": (_raw("showing", emotions={"joy": "6"}), "joy"),
    "amount a bool": (_raw("commit", emotions={"joy": True}), "joy"),
    "amount null": (_raw("commit", emotions={"joy": None}), "joy"),
    "amount too large for a float": (_raw("commit", emotions={"joy": 10**400}), "joy"),
    "one bad amount among good ones": (_raw("commit", emotions={"joy": 1.0, "grief": float("nan")}), "grief"),
    "modifier nan": (_raw("commit", modifier=float("nan")), "modifier"),
    "modifier a string": (_raw("showing", modifier="1"), "modifier"),
    "modifier a bool": (_raw("assigned", modifier=True), "modifier"),
    "climax inf": (_raw("commit", climax=float("inf")), "climax"),
    "climax a string": (_raw("commit", climax="9"), "climax"),
    "source not text": (_raw("change", source=5), "source"),
}


@pytest.mark.parametrize("label", sorted(_INVALID))
def test_an_invalid_event_is_refused_with_the_reason(label: str) -> None:
    raw, word = _INVALID[label]
    with pytest.raises(engine_files.InvalidEventError) as caught:
        engine_files.event_from_json(raw)
    assert word in str(caught.value)


@pytest.mark.parametrize("error", [OverflowError("date value out of range"), OSError("pre-1970 on Windows")])
def test_an_at_that_cannot_be_turned_into_a_local_day_is_invalid(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = engine_utils.to_local

    def failing(dt: datetime) -> datetime:
        if dt.year < 1970:
            raise error
        return real(dt)

    monkeypatch.setattr(engine_utils, "to_local", failing)
    with pytest.raises(engine_files.InvalidEventError, match="day"):
        engine_files.event_from_json(_raw("commit", at="1969-12-31T12:00:00Z"))
    assert engine_files.event_from_json(_raw("commit")).kind == "commit"  # a normal time still passes


_WRONG = [None, "x", [], {}, True, 10**400, -1, float("nan"), float("inf"), 0, 2.5]


def test_every_event_the_check_accepts_folds_without_raising() -> None:
    """The check and the fold agree: nothing that passes can fail every drain for ever (E6.10)."""
    accepted = refused = 0
    for kind in ("commit", "assigned", "showing", "change", "start"):
        base = _raw(kind)
        fields: list[tuple[str, ...]] = [(key,) for key in base]
        if "emotions" in base:
            fields += [("emotions", name) for name in base["emotions"]]
        for path in fields:
            for wrong in _WRONG:
                raw = copy.deepcopy(base)
                node = raw
                for step in path[:-1]:
                    node = node[step]
                node[path[-1]] = wrong
                try:
                    event = engine_files.event_from_json(raw)
                except engine_files.InvalidEventError:
                    refused += 1
                    continue
                fold_event(EngineState(), event, NOW)
                accepted += 1
    assert accepted > 20 and refused > 100


# --- the queue file ----------------------------------------------------------------------------


def test_a_missing_queue_is_an_empty_queue(tmp_path: Path) -> None:
    assert engine_files.read_queue(tmp_path) == []


def test_the_queue_read_drops_each_invalid_event_with_a_reason_and_keeps_the_rest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    labels = sorted(_INVALID)
    good = [_raw("commit", memory_id="g1"), _raw("change"), _raw("start"), _raw("showing", memory_id="g2")]
    events: list[object] = []
    for index, label in enumerate(labels):
        events.append(good[index % len(good)])
        events.append(_INVALID[label][0])
    _write_raw_queue(tmp_path, events)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        read = engine_files.read_queue(tmp_path)
    assert len(read) == len(labels)
    assert [e.kind for e in read] == [good[i % len(good)]["kind"] for i in range(len(labels))]
    messages = _errors(caplog)
    assert len(messages) == len(labels)
    for index, message in enumerate(messages):
        assert str(engine_files.queue_path(tmp_path)) in message
        assert _INVALID[labels[index]][1] in message  # says why


def test_the_queue_read_keeps_file_order(tmp_path: Path) -> None:
    _write_raw_queue(tmp_path, [_raw("commit", memory_id="b", at="2026-06-10T11:00:00Z"), _raw("commit", memory_id="a")])
    assert [e.memory_id for e in engine_files.read_queue(tmp_path)] == ["b", "a"]


@pytest.mark.parametrize(
    "content",
    ["{torn", "", "   ", "null", '{"kind": "commit"}', '"text"', "42", "\xff\xfe"],
)
def test_an_unparseable_or_non_list_queue_is_discarded_with_a_line_and_not_rewritten(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str
) -> None:
    """Accepted edge E3 (E6.18): the queue file is discarded and no rebuild follows."""
    path = _write_raw_queue(tmp_path, content)
    before = path.read_bytes()
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.read_queue(tmp_path) == []
    messages = _errors(caplog)
    assert len(messages) == 1 and str(path) in messages[0] and "discard" in messages[0]
    assert path.read_bytes() == before  # the read itself writes nothing; the drain removes the file


def test_an_io_fault_on_the_queue_raises_and_discards_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_raw_queue(tmp_path, [_raw("commit")])
    before = path.read_bytes()
    real = json_io._read_bytes

    def failing(p: Path):
        if p == path:
            raise PermissionError("share violation")
        return real(p)

    monkeypatch.setattr(json_io, "_read_bytes", failing)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        with pytest.raises(OSError):
            engine_files.read_queue(tmp_path)
        with pytest.raises(OSError):
            engine_files.append_event(tmp_path, engine_files.event_from_json(_raw("start")))
    assert not [m for m in _errors(caplog) if "discard" in m]  # an I/O fault is not a discard
    monkeypatch.undo()
    assert path.read_bytes() == before


def test_append_adds_to_the_end_and_creates_the_state_folder(tmp_path: Path) -> None:
    assert not (tmp_path / "state").exists()
    first = engine_files.event_from_json(_raw("commit", memory_id="a"))
    second = engine_files.event_from_json(_raw("change"))
    engine_files.append_event(tmp_path, first)
    engine_files.append_event(tmp_path, second)
    assert json.loads(engine_files.queue_path(tmp_path).read_text(encoding="utf-8")) == [
        engine_files.event_to_json(first),
        engine_files.event_to_json(second),
    ]
    assert engine_files.read_queue(tmp_path) == [first, second]


def test_append_keeps_the_raw_events_already_there_even_an_invalid_one(tmp_path: Path) -> None:
    existing = [_raw("commit", memory_id="a"), {"kind": "dream", "at": AT}, "junk"]
    _write_raw_queue(tmp_path, existing)
    engine_files.append_event(tmp_path, engine_files.event_from_json(_raw("start")))
    stored = json.loads(engine_files.queue_path(tmp_path).read_text(encoding="utf-8"))
    assert stored[:3] == existing and stored[3] == _raw("start") and len(stored) == 4


def test_append_over_an_unparseable_queue_replaces_it_with_the_one_event(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_raw_queue(tmp_path, "{torn")
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        engine_files.append_event(tmp_path, engine_files.event_from_json(_raw("start")))
    assert json.loads(engine_files.queue_path(tmp_path).read_text(encoding="utf-8")) == [_raw("start")]
    assert any("discard" in m for m in _errors(caplog))


def test_a_non_finite_event_cannot_be_appended_and_the_queue_is_untouched(tmp_path: Path) -> None:
    engine_files.append_event(tmp_path, engine_files.event_from_json(_raw("start")))
    before = _tree(tmp_path)
    bad = Event(kind="commit", at=NOW, memory_id="m", emotions={"joy": float("nan")})
    with pytest.raises(ValueError):
        engine_files.append_event(tmp_path, bad)
    assert _tree(tmp_path) == before


def test_writing_an_empty_queue_removes_the_file_and_a_missing_one_is_fine(tmp_path: Path) -> None:
    engine_files.write_queue(tmp_path, [engine_files.event_from_json(_raw("start"))])
    assert engine_files.queue_path(tmp_path).exists()
    engine_files.write_queue(tmp_path, [])
    assert not engine_files.queue_path(tmp_path).exists()
    engine_files.write_queue(tmp_path, [])  # nothing there: not an error


def test_the_queue_is_written_compact_with_no_backups(tmp_path: Path) -> None:
    events = [engine_files.event_from_json(_raw(kind)) for kind in ("commit", "change")]
    engine_files.write_queue(tmp_path, events)
    engine_files.write_queue(tmp_path, events)
    engine_files.append_event(tmp_path, events[0])
    names = sorted(p.name for p in (tmp_path / "state").iterdir())
    assert names == ["emotion_pending.json"]  # no .bak, no temp file
    text = engine_files.queue_path(tmp_path).read_text(encoding="utf-8")
    assert text.count("\n") == 1


def test_clear_queue_removes_the_file_and_tolerates_none(tmp_path: Path) -> None:
    engine_files.clear_queue(tmp_path)
    _write_raw_queue(tmp_path, "{torn")
    engine_files.clear_queue(tmp_path)
    assert not engine_files.queue_path(tmp_path).exists()


def test_no_queue_or_change_log_function_takes_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The callers hold the one lock (it is not re-entrant): none of these may take it a second time."""

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("a function of the queue or the change log took the lock")

    monkeypatch.setattr(engine_files, "state_lock", forbidden)
    monkeypatch.setattr(engine_files, "file_lock", forbidden, raising=False)
    engine_files.append_event(tmp_path, engine_files.event_from_json(_raw("start")))
    assert len(engine_files.read_queue(tmp_path)) == 1
    engine_files.write_queue(tmp_path, [engine_files.event_from_json(_raw("start"))])
    engine_files.clear_queue(tmp_path)
    assert engine_files.append_change(tmp_path, _change(NOW)) is True
    engine_files.list_change_files(tmp_path)
    engine_files.read_changes(tmp_path, NOW)
    engine_files.prune_changes(tmp_path, "2026-06-10")


# --- the change log: append --------------------------------------------------------------------


def _change_file(persona_dir: Path, day: str) -> Path:
    return engine_files.changes_dir(persona_dir) / f"{day}.jsonl"


def test_a_change_is_appended_as_one_line_to_the_file_of_its_local_day(tmp_path: Path) -> None:
    event = _change(NOW)
    assert engine_files.append_change(tmp_path, event) is True
    path = _change_file(tmp_path, day_of(NOW))
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [engine_files.event_to_json(event)]
    assert engine_files.append_change(tmp_path, _change(NOW + timedelta(minutes=5), grief=2.0)) is True
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_a_change_after_local_midnight_goes_to_the_next_days_file(tmp_path: Path) -> None:
    day = date(2026, 6, 10)
    before = datetime.combine(day, datetime.max.time().replace(microsecond=0)).astimezone(UTC)
    after = before + timedelta(seconds=2)
    engine_files.append_change(tmp_path, _change(before))
    engine_files.append_change(tmp_path, _change(after))
    assert engine_files.list_change_files(tmp_path) == ["2026-06-10.jsonl", "2026-06-11.jsonl"]


def test_an_append_that_reports_failure_returns_false_and_raises_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_files, "append_jsonl", lambda *a, **k: None)
    assert engine_files.append_change(tmp_path, _change(NOW)) is False


def test_the_append_asks_for_a_held_lock_and_logging_not_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def spy(path: Path, record: object, **kwargs: object):
        seen.update(kwargs)
        return 10

    monkeypatch.setattr(engine_files, "append_jsonl", spy)
    assert engine_files.append_change(tmp_path, _change(NOW)) is True
    assert seen == {"lock": "held", "on_error": "log"}


def test_an_unwritable_change_log_logs_and_returns_false(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "emotion_changes").write_text("a file where the folder should be", encoding="utf-8")
    with caplog.at_level(logging.DEBUG):
        assert engine_files.append_change(tmp_path, _change(NOW)) is False
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_a_change_that_cannot_be_encoded_or_dated_is_logged_and_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.append_change(tmp_path, _change(NOW, joy=float("nan"))) is False
    assert len(_errors(caplog)) == 1
    caplog.clear()

    def failing(dt: datetime) -> datetime:
        raise OverflowError("date value out of range")

    monkeypatch.setattr(engine_utils, "to_local", failing)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.append_change(tmp_path, _change(NOW)) is False
    assert len(_errors(caplog)) == 1
    assert not (tmp_path / "state" / "emotion_changes").exists()


def test_only_a_change_event_can_be_appended_to_the_log(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        engine_files.append_change(tmp_path, engine_files.event_from_json(_raw("commit")))


# --- the change log: list and prune ------------------------------------------------------------


def _touch_days(persona_dir: Path, *days: str) -> None:
    folder = engine_files.changes_dir(persona_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for day in days:
        (folder / f"{day}.jsonl").write_text("{}\n", encoding="utf-8")


def test_list_returns_names_of_valid_day_files_only(tmp_path: Path) -> None:
    _touch_days(tmp_path, "2026-06-02", "2026-06-01")
    folder = engine_files.changes_dir(tmp_path)
    for odd in ("notes.txt", "2026-06-03.jsonl.bak", "2026-13-45.jsonl", "x.jsonl", "2026-6-3.jsonl", "20260603.jsonl"):
        (folder / odd).write_text("x", encoding="utf-8")
    (folder / "2026-06-04.jsonl").mkdir()  # a directory named like a day
    assert engine_files.list_change_files(tmp_path) == ["2026-06-01.jsonl", "2026-06-02.jsonl"]


def test_list_of_a_missing_folder_is_empty_and_of_an_unlistable_one_is_empty_with_a_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert engine_files.list_change_files(tmp_path) == []
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "emotion_changes").write_text("a file", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.list_change_files(tmp_path) == []
    assert len(_errors(caplog)) == 1


def test_prune_deletes_whole_files_before_the_boundary_and_keeps_the_boundary_day(tmp_path: Path) -> None:
    _touch_days(tmp_path, "2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13", "2026-06-10")
    # A = 2026-06-10: the boundary label is 30 calendar days earlier, 2026-05-11.
    deleted = engine_files.prune_changes(tmp_path, "2026-06-10")
    assert deleted == ["2026-05-10.jsonl"]
    assert engine_files.list_change_files(tmp_path) == [
        "2026-05-11.jsonl",
        "2026-05-12.jsonl",
        "2026-05-13.jsonl",
        "2026-06-10.jsonl",
    ]


def test_prune_reads_the_boundary_from_the_dev_constant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch_days(tmp_path, "2026-06-01", "2026-06-02", "2026-06-09", "2026-06-10")
    monkeypatch.setattr(dev_constants, "EMOTION_LONG_HALF_LIFE_D", 8.0)
    assert engine_files.prune_changes(tmp_path, "2026-06-10") == ["2026-06-01.jsonl"]  # boundary 2026-06-02


def test_prune_deletes_nothing_when_the_active_day_was_unset(tmp_path: Path) -> None:
    _touch_days(tmp_path, "2020-01-01", "2026-06-10")
    assert engine_files.prune_changes(tmp_path, None) == []
    assert len(engine_files.list_change_files(tmp_path)) == 2


def test_prune_leaves_foreign_files_and_day_named_directories_alone(tmp_path: Path) -> None:
    _touch_days(tmp_path, "2020-01-01")
    folder = engine_files.changes_dir(tmp_path)
    (folder / "README.txt").write_text("keep", encoding="utf-8")
    (folder / "2020-01-02.jsonl.bak").write_text("keep", encoding="utf-8")
    (folder / "2020-13-45.jsonl").write_text("keep", encoding="utf-8")
    (folder / "2020-01-03.jsonl").mkdir()
    (folder / "2020-01-03.jsonl" / "inner").write_text("keep", encoding="utf-8")
    assert engine_files.prune_changes(tmp_path, "2026-06-10") == ["2020-01-01.jsonl"]
    assert sorted(p.name for p in folder.iterdir()) == [
        "2020-01-02.jsonl.bak",
        "2020-01-03.jsonl",
        "2020-13-45.jsonl",
        "README.txt",
    ]


def test_prune_with_no_folder_or_a_bad_anchor_deletes_nothing_and_raises_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert engine_files.prune_changes(tmp_path, "2026-06-10") == []
    _touch_days(tmp_path, "2020-01-01")
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.prune_changes(tmp_path, "banana") == []
    assert len(_errors(caplog)) == 1
    assert engine_files.list_change_files(tmp_path) == ["2020-01-01.jsonl"]


def test_a_failed_delete_is_logged_the_others_still_go_and_the_next_close_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _touch_days(tmp_path, "2026-01-01", "2026-01-02", "2026-01-03")
    real_unlink = pathlib.Path.unlink

    def refuse_the_second(self: Path, *args: object, **kwargs: object):
        if self.name == "2026-01-02.jsonl":
            raise PermissionError("in use")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", refuse_the_second)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        deleted = engine_files.prune_changes(tmp_path, "2026-06-10")
    assert deleted == ["2026-01-01.jsonl", "2026-01-03.jsonl"]
    assert engine_files.list_change_files(tmp_path) == ["2026-01-02.jsonl"]
    assert len(_errors(caplog)) == 1 and "2026-01-02.jsonl" in _errors(caplog)[0]
    monkeypatch.undo()
    assert engine_files.prune_changes(tmp_path, "2026-06-10") == ["2026-01-02.jsonl"]


# --- the change log: the tolerant reader -------------------------------------------------------


def _line(at: datetime, **emotions: float) -> str:
    return json.dumps(engine_files.event_to_json(_change(at, **emotions)))


def _write_day(persona_dir: Path, day: str, *lines: str) -> Path:
    folder = engine_files.changes_dir(persona_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{day}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _local_noon(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time().replace(hour=12)).astimezone(UTC)


def test_the_reader_folds_lines_inside_the_window_in_date_then_line_order(tmp_path: Path) -> None:
    d = date(2026, 6, 10)
    t = [_local_noon(d + timedelta(days=i)) for i in range(3)]
    _write_day(tmp_path, str(d + timedelta(days=1)), _line(t[1], joy=2.0), _line(t[1] + timedelta(hours=1), joy=3.0))
    _write_day(tmp_path, str(d), _line(t[0], joy=1.0))
    _write_day(tmp_path, str(d + timedelta(days=2)), _line(t[2], joy=4.0))
    got = engine_files.read_changes(tmp_path, t[0])
    assert [e.emotions["joy"] for e in got] == [1.0, 2.0, 3.0, 4.0]
    assert all(e.kind == "change" for e in got)


def test_the_reader_starts_at_the_day_before_the_window_and_drops_lines_before_it(tmp_path: Path) -> None:
    d = date(2026, 6, 10)
    since = _local_noon(d)
    # A file named two days early is not even opened: its line is dated inside the window, so reading it would show.
    _write_day(tmp_path, str(d - timedelta(days=2)), _line(since + timedelta(hours=1), joy=9.0))
    _write_day(
        tmp_path,
        str(d - timedelta(days=1)),
        _line(since - timedelta(hours=14), joy=1.0),  # the slack day: read, but before `since`
        _line(since + timedelta(minutes=1), joy=2.0),  # a line inside the window sitting in the slack day's file
    )
    got = engine_files.read_changes(tmp_path, since)
    assert [e.emotions["joy"] for e in got] == [2.0]


def test_the_reader_includes_a_line_dated_exactly_at_the_start_of_the_window(tmp_path: Path) -> None:
    since = _local_noon(date(2026, 6, 10))
    _write_day(tmp_path, "2026-06-10", _line(since, joy=5.0))
    assert [e.emotions["joy"] for e in engine_files.read_changes(tmp_path, since)] == [5.0]


def test_the_reader_adds_nothing_and_raises_nothing_for_a_missing_folder_or_a_gap(tmp_path: Path) -> None:
    since = _local_noon(date(2026, 6, 10))
    assert engine_files.read_changes(tmp_path, since) == []
    _write_day(tmp_path, "2026-06-10", _line(since, joy=1.0))
    _write_day(tmp_path, "2026-06-13", _line(since + timedelta(days=3), joy=2.0))  # 11 and 12 missing
    assert [e.emotions["joy"] for e in engine_files.read_changes(tmp_path, since)] == [1.0, 2.0]


def test_the_reader_ignores_files_that_are_not_day_files(tmp_path: Path) -> None:
    since = _local_noon(date(2026, 6, 10))
    folder = engine_files.changes_dir(tmp_path)
    _write_day(tmp_path, "2026-06-10", _line(since, joy=1.0))
    (folder / "notes.jsonl").write_text(_line(since, joy=7.0) + "\n", encoding="utf-8")
    (folder / "2026-06-11.jsonl.bak").write_text(_line(since, joy=7.0) + "\n", encoding="utf-8")
    (folder / "2026-13-45.jsonl").write_text(_line(since, joy=7.0) + "\n", encoding="utf-8")
    (folder / "2026-06-12.jsonl").mkdir()
    assert [e.emotions["joy"] for e in engine_files.read_changes(tmp_path, since)] == [1.0]


def test_the_reader_skips_a_torn_line_a_bad_line_and_a_line_that_is_not_a_change(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    since = _local_noon(date(2026, 6, 10))
    good = _line(since, joy=1.0)
    path = _write_day(
        tmp_path,
        "2026-06-10",
        good,
        "{not json",
        json.dumps(_raw("change", emotions={"joy": float("nan")})),  # fails the event check
        json.dumps(_raw("commit")),  # a valid event, but not a change line
        json.dumps([1, 2]),
        _line(since + timedelta(minutes=1), joy=2.0),
    )
    with path.open("ab") as fh:
        fh.write(b'{"kind": "change", "at": "2026-06-10T')  # torn last line, no newline
    before = path.read_bytes()
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        got = engine_files.read_changes(tmp_path, since)
    assert [e.emotions["joy"] for e in got] == [1.0, 2.0]
    assert path.read_bytes() == before  # the reader repairs nothing
    assert len(_errors(caplog)) == 2  # the invalid event and the commit line


def test_the_reader_treats_an_unlistable_folder_as_missing_with_a_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "emotion_changes").write_text("a file", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.read_changes(tmp_path, NOW) == []
    assert len(_errors(caplog)) == 1


def test_the_reader_uses_what_it_could_read_of_a_file_that_fails_midway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    since = _local_noon(date(2026, 6, 10))
    _write_day(tmp_path, "2026-06-10", _line(since, joy=1.0))
    _write_day(tmp_path, "2026-06-11", _line(since + timedelta(days=1), joy=2.0))
    real = engine_files.iter_jsonl

    def failing_first(path: Path, **kwargs: object):
        if path.name == "2026-06-10.jsonl":
            raise PermissionError("locked")
        return real(path, **kwargs)

    monkeypatch.setattr(engine_files, "iter_jsonl", failing_first)
    assert [e.emotions["joy"] for e in engine_files.read_changes(tmp_path, since)] == [2.0]


def test_a_change_written_then_read_round_trips(tmp_path: Path) -> None:
    event = _change(NOW, joy=-2.5, grief=0.25)
    engine_files.append_change(tmp_path, event)
    assert engine_files.read_changes(tmp_path, NOW - timedelta(days=1)) == [event]


def test_the_first_append_creates_only_the_folder_and_the_day_file(tmp_path: Path) -> None:
    engine_files.append_change(tmp_path, _change(NOW))
    assert {p.name for p in (tmp_path / "state").rglob("*")} == {"emotion_changes", f"{day_of(NOW)}.jsonl"}
