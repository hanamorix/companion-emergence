"""The emotion engine's queue and change log (Engine 2, #577).

The queue (`state/emotion_pending.jsonl`) is an append-only file of events waiting to be folded, one JSON event per line;
the change log (`state/emotion_changes/YYYY-MM-DD.jsonl`) is a crash-recovery record of chat-derived changes. These are
the file-level cases: the per-event check, the queue file's append, read by position and emptying (a bad line is dropped on
its own), the accessor each file operation goes through, the change log's append, list, prune and tolerant reader. The hook
cases (Engine 3, #578: what a memory write queues, the start-only event, the guard, the lock) are at the end. Cases that need a
drain (fold order, early-out, wiring) belong to the slice that builds it and extend this file.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import io
import json
import logging
import math
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine as engine_module
from brain.emotion import engine_files, engine_utils
from brain.emotion.engine_utils import EngineState, Event, day_of, fold_event, source_modifier
from brain.health import json_io
from brain.memory.store import Memory, MemoryStore
from brain.utils import file_lock, file_ops

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


def _put_queue_file(persona_dir: Path, content: object) -> Path:
    """Write the queue file as bytes (so no platform rewrites a newline): bytes as given, text as UTF-8, a list as one JSON line each."""
    path = _queue_file(persona_dir)
    if isinstance(content, list):
        content = "".join(json.dumps(item) + "\n" for item in content)
    path.write_bytes(content if isinstance(content, bytes) else str(content).encode("utf-8"))
    return path


def _change(at: datetime, **emotions: float) -> Event:
    return Event(kind="change", at=at, emotions=emotions or {"joy": 1.0}, source="pass2")


def _read(persona_dir: Path, from_line: int = 0) -> list[Event]:
    """The events `read_queue` returns from line `from_line` (the position the old one-argument reader had implicitly: 0)."""
    return engine_files.read_queue(persona_dir, from_line=from_line).events


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


_WRONG = [None, "x", [], {}, True, 10**400, -1, float("nan"), float("inf"), 0, 2.5, 1e308, -1e308, 10**300]


def test_every_event_the_check_accepts_folds_and_saves(tmp_path: Path) -> None:
    """The check and the fold agree: nothing that passes can fail every drain for ever (design doc section 2.13), not even at the save."""
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
                state = EngineState()
                fold_event(state, event, NOW)
                state.saved_at = NOW
                engine_files.save_state(tmp_path, state)  # strict: a non-finite number anywhere raises
                accepted += 1
    assert accepted > 20 and refused > 100


# --- a modifier outside the range its producer can make is held to the range -------------------


def _memory(memory_type: str, **metadata: object) -> Memory:
    return Memory.create_new(content="c", memory_type=memory_type, domain="us", emotions={"joy": 5.0}, tags=[], metadata=metadata)


def _produced_range() -> tuple[float, float]:
    """The smallest and largest modifier `source_modifier` makes over the kinds and the extreme inputs, and a check that all lie between."""
    outputs = [source_modifier(_memory("conversation"), 0.0), source_modifier(_memory("making"), 0.0)]
    peer = _memory("kindled_peer", peer_id="p")
    outputs += [source_modifier(peer, trust) for trust in (-5, 0, 0.3, 0.5, 1, 9, math.nan, math.inf, -math.inf)]
    research = [{}, {"engagement_rating": "abc"}] + [
        {"engagement_rating": rating} for rating in (-3, 0, 5, "8", 10, 15, math.nan, math.inf, -math.inf)
    ]
    outputs += [source_modifier(_memory("research", **metadata), 0.0) for metadata in research]
    return min(outputs), max(outputs)


def test_the_held_range_is_the_range_source_modifier_produces() -> None:
    low, high = _produced_range()
    assert (low, high) == (pytest.approx(0.1), pytest.approx(1.0))  # both ends are reached, so neither bound is looser
    assert engine_files.event_from_json(_raw("showing", modifier=-1e308)).modifier == low
    assert engine_files.event_from_json(_raw("showing", modifier=1e308)).modifier == high


@pytest.mark.parametrize("kind", ["commit", "assigned", "showing"])
@pytest.mark.parametrize("value", [1e308, -1e308, 2.0, 0.0, -1.0, 0.05, pytest.param(10**300, id="int-1e300")])
def test_a_finite_modifier_outside_the_range_is_held_to_the_nearest_bound(kind: str, value: float) -> None:
    low, high = _produced_range()
    expected = low if value < low else high
    event = engine_files.event_from_json(_raw(kind, modifier=value))
    twin = engine_files.event_from_json(_raw(kind, modifier=0.5))
    assert event.modifier == expected
    assert dataclasses.replace(event, modifier=0.5) == twin  # nothing else about the event changed


@pytest.mark.parametrize("kind", ["commit", "assigned", "showing"])
def test_a_hold_writes_one_warning_saying_what_was_found_and_where_it_went(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    low, high = _produced_range()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        engine_files.event_from_json(_raw(kind, memory_id="mem-77", modifier=1e308))
    records = [r for r in caplog.records if r.name == LOGGER]
    assert [r.levelno for r in records] == [logging.WARNING]
    text = records[0].getMessage()
    for part in (kind, "mem-77", repr(1e308), f"{low} to {high}", f"held to {high}"):
        assert part in text


@pytest.mark.no_base_failure(reason="pins what the hold must leave alone: an in-range or absent modifier, and a change or start event's stored one, are unchanged and quiet; the old code passes it")
def test_an_in_range_modifier_is_untouched_and_quiet(caplog: pytest.LogCaptureFixture) -> None:
    low, high = _produced_range()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        for kind in ("commit", "assigned", "showing"):
            for value in (low, 0.55, 0.9999, high):
                assert engine_files.event_from_json(_raw(kind, modifier=value)).modifier == value
            assert engine_files.event_from_json(_drop(_raw(kind), "modifier")).modifier == 1.0
        assert engine_files.event_from_json(_raw("change", modifier=1e308)).modifier == 1.0  # a change does not read it
        assert engine_files.event_from_json(_raw("start", modifier=1e308)).kind == "start"
    assert [r for r in caplog.records if r.name == LOGGER] == []


@pytest.mark.no_base_failure(reason="pins the refusal that already exists and must stay when the hold is added: a modifier that is not a finite number is refused, not held; the old code passes it")
@pytest.mark.parametrize("kind", ["commit", "assigned", "showing"])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), "1", True, pytest.param(10**400, id="int-1e400")]
)
def test_a_modifier_that_is_not_a_finite_number_is_still_refused(kind: str, value: object) -> None:
    with pytest.raises(engine_files.InvalidEventError, match="modifier"):
        engine_files.event_from_json(_raw(kind, modifier=value))


def test_the_queue_read_still_drops_a_non_finite_modifier_and_keeps_the_neighbours(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _put_queue_file(tmp_path, [_raw("commit", memory_id="a"), _raw("showing", modifier=float("inf")), _raw("assigned", memory_id="c")])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        events = _read(tmp_path)
    assert [e.memory_id for e in events] == ["a", "c"]
    errors = _errors(caplog)
    assert len(errors) == 1 and "modifier" in errors[0]


def test_a_huge_queued_modifier_is_held_then_folds_and_saves(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The overflow this closes: an event with modifier about 1e308 once folded to an infinite average and made every save fail."""
    low, high = _produced_range()
    raw = [
        _raw("commit", memory_id="a"),
        _raw("commit", memory_id="big", modifier=1e308, emotions={"joy": 9.0}),
        _raw("assigned", memory_id="big2", modifier=-1e308, emotions={"grief": 4.0}),
        _raw("change"),
    ]
    queue = _put_queue_file(tmp_path, raw)
    before = _tree(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        events = _read(tmp_path)
    assert _tree(tmp_path) == before  # the hold is read-side: the file is not rewritten, nothing is created
    assert queue.read_bytes() == before[str(queue.relative_to(tmp_path))]
    assert [e.memory_id for e in events] == ["a", "big", "big2", None]
    assert [e.modifier for e in events[1:3]] == [high, low]
    assert _errors(caplog) == []
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2

    state = EngineState()
    for event in events:
        fold_event(state, event, NOW)
    state.saved_at = NOW
    state.active_day = day_of(NOW)  # the drain sets the day; a state with `last_update` and no `active_day` is not valid
    engine_files.save_state(tmp_path, state)  # strict serialization: raises on any non-finite number
    loaded = engine_files.read_state(tmp_path)
    assert loaded.source == "main" and loaded.state is not None
    for emotion in loaded.state.emotions.values():
        for average in (emotion.short, emotion.long):
            assert math.isfinite(average.avg) and math.isfinite(average.weight)
    assert set(loaded.state.emotions) >= {"joy", "grief"}


# --- the queue file ----------------------------------------------------------------------------


def _event(i: int) -> Event:
    """The i-th of a rotation of the five kinds, with its own memory id and time."""
    kind = ("commit", "assigned", "showing", "change", "start")[i % 5]
    over: dict[str, object] = {"at": f"2026-06-10T10:{i % 60:02d}:00Z"}
    if kind in ("commit", "assigned", "showing"):
        over["memory_id"] = f"mem-{i}"
    return engine_files.event_from_json(_raw(kind, **over))


def _jline(raw: object) -> bytes:
    return json.dumps(raw).encode("utf-8") + b"\n"


def test_a_missing_queue_is_an_empty_queue(tmp_path: Path) -> None:
    assert _read(tmp_path) == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("content", [b"", b"\n\n  \n", b"\r\n\r\n", b"\n"], ids=["empty", "blank", "crlf-blank", "newline"])
def test_an_empty_or_blank_queue_file_is_an_empty_queue_with_no_log_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: bytes
) -> None:
    path = _put_queue_file(tmp_path, content)
    before = _tree(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert _read(tmp_path) == []
    assert [r for r in caplog.records if r.name == LOGGER] == []
    assert _tree(tmp_path) == before and path.read_bytes() == content


def test_appended_events_are_one_line_each_and_read_back_in_order(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    events = [_event(i) for i in range(40)]
    with caplog.at_level(logging.DEBUG):
        for event in events:
            assert engine_files.append_event(tmp_path, event) is True
        assert _read(tmp_path) == events
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    data = engine_files.queue_path(tmp_path).read_bytes()
    assert data.endswith(b"\n")
    lines = data.split(b"\n")[:-1]
    assert [json.loads(line) for line in lines] == [engine_files.event_to_json(e) for e in events]
    assert engine_files.queue_path(tmp_path).name == "emotion_pending.jsonl"
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == ["emotion_pending.jsonl"]  # no .bak, temp or .lock


def test_an_append_never_touches_what_is_already_in_the_file(tmp_path: Path) -> None:
    path = engine_files.queue_path(tmp_path)
    before = b""
    inode = None
    for i in range(200):
        assert engine_files.append_event(tmp_path, _event(i)) is True
        after = path.read_bytes()
        assert after.startswith(before) and len(after) > len(before)
        if inode is not None:
            assert path.stat().st_ino == inode  # the file is not replaced
        inode = path.stat().st_ino
        before = after


def _cpu_of_appends(persona_dir: Path, count: int) -> float:
    events = [_event(i) for i in range(count)]
    start = time.process_time()
    for event in events:
        engine_files.append_event(persona_dir, event)
    return time.process_time() - start


def test_the_cost_of_appends_does_not_grow_with_the_depth_of_the_queue(tmp_path: Path) -> None:
    """Ten times the appends cost about ten times the time (the old whole-file rewrite cost about a hundred times)."""
    small = _cpu_of_appends(tmp_path / "small", 200)
    large = _cpu_of_appends(tmp_path / "large", 2000)
    assert large < 30 * max(small, 0.02)


def test_the_queue_read_keeps_file_order_not_time_order(tmp_path: Path) -> None:
    _put_queue_file(tmp_path, [_raw("commit", memory_id="b", at="2026-06-10T11:00:00Z"), _raw("commit", memory_id="a")])
    assert [e.memory_id for e in _read(tmp_path)] == ["b", "a"]


def test_a_valid_event_with_non_ascii_text_reads_back_equal(tmp_path: Path) -> None:
    event = engine_files.event_from_json(_raw("commit", memory_id="café-日本"))
    assert engine_files.append_event(tmp_path, event) is True
    assert _read(tmp_path) == [event]


def test_a_torn_last_line_is_neither_read_counted_nor_logged_and_the_next_append_makes_it_a_dropped_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    good = [_event(i) for i in range(3)]
    for event in good:
        engine_files.append_event(tmp_path, event)
    path = engine_files.queue_path(tmp_path)
    with path.open("ab") as fh:
        fh.write(b'{"kind": "commit", "at": "2026-06-10T10:0')  # a crash cut this append short: no newline
    before = path.read_bytes()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        result = engine_files.read_queue(tmp_path, from_line=0)
    assert result.events == good and result.end_line == 3  # the torn line is not a line yet
    assert [r for r in caplog.records if r.name == LOGGER] == []  # and the reader says nothing about it
    assert path.read_bytes() == before  # the read writes nothing
    caplog.clear()

    fourth = _event(3)
    assert engine_files.append_event(tmp_path, fourth) is True
    assert path.read_bytes().startswith(before)  # the three good lines and the torn one are byte-identical
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        result = engine_files.read_queue(tmp_path, from_line=0)
    assert result.events == [*good, fourth] and result.end_line == 5
    messages = _errors(caplog)
    assert len(messages) == 1 and "line 4" in messages[0] and "malformed JSON" in messages[0]  # now a complete, bad line


@pytest.mark.parametrize("label", sorted(_INVALID))
def test_each_invalid_event_line_is_dropped_alone_with_its_line_number_and_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, label: str
) -> None:
    raw, word = _INVALID[label]
    path = _put_queue_file(tmp_path, [_raw("commit", memory_id="a"), raw, _raw("assigned", memory_id="c")])
    before = _tree(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        events = _read(tmp_path)
    assert [e.memory_id for e in events] == ["a", "c"]
    messages = _errors(caplog)
    assert len(messages) == 1
    assert str(path) in messages[0] and "line 2" in messages[0]
    if label in ("not an object", "a list"):
        word = "not a JSON object"  # a line that is valid JSON but not an object is refused before the event check
    assert word in messages[0].split("dropped (", 1)[1]  # the reason, not the path or the frame
    assert _tree(tmp_path) == before  # the read writes nothing


_BAD_LINES = {
    "malformed": (b"{not json", "malformed JSON"),
    "an integer": (b"42", "not a JSON object"),
    "a list": (b"[1]", "not a JSON object"),
    "a string": (b'"text"', "not a JSON object"),
    "null": (b"null", "not a JSON object"),
    "invalid utf-8": (b"\xff\xfe\xfa{", "invalid UTF-8"),
    "a very long line": (b"x" * 2_000_000, "malformed JSON"),
    "deeply nested": (b"[" * 100_000, "malformed JSON"),
    "an integer of 5,000 digits": (b"1" * 5000, "malformed JSON"),  # a plain ValueError, which the line parser lets through
}


@pytest.mark.parametrize("label", sorted(_BAD_LINES))
def test_a_line_that_is_not_an_event_is_dropped_alone_with_its_line_number_and_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, label: str
) -> None:
    bad, word = _BAD_LINES[label]
    path = _put_queue_file(tmp_path, _jline(_raw("commit", memory_id="a")) + bad + b"\n" + _jline(_raw("assigned", memory_id="c")))
    before = _tree(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        events = _read(tmp_path)
    assert [e.memory_id for e in events] == ["a", "c"]
    messages = _errors(caplog)
    assert len(messages) == 1
    assert str(path) in messages[0] and "line 2" in messages[0] and word in messages[0]
    assert len(messages[0]) < 1000  # the content of the line is not echoed
    assert _tree(tmp_path) == before


def test_crlf_endings_and_a_byte_order_mark_are_tolerated_and_later_lines_are_numbered_physically(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first, second = _raw("commit", memory_id="a"), _raw("assigned", memory_id="c")
    data = b"\xef\xbb\xbf" + json.dumps(first).encode() + b"\r\n" + json.dumps(second).encode() + b"\r\n"
    path = _put_queue_file(tmp_path, data)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert [e.memory_id for e in _read(tmp_path)] == ["a", "c"]
    assert [r for r in caplog.records if r.name == LOGGER] == []
    _put_queue_file(tmp_path, data + b"junk\r\n")
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert [e.memory_id for e in _read(tmp_path)] == ["a", "c"]
    messages = _errors(caplog)
    assert len(messages) == 1 and str(path) in messages[0] and "line 3" in messages[0]


def test_blank_lines_count_in_the_line_numbers(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _put_queue_file(tmp_path, _jline(_raw("commit", memory_id="a")) + b"\n\n" + b"{oops\n")
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert [e.memory_id for e in _read(tmp_path)] == ["a"]
    messages = _errors(caplog)
    assert len(messages) == 1 and "line 4" in messages[0]


def test_a_line_of_zero_bytes_is_dropped_like_a_torn_line_and_the_next_line_is_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """design doc section 6: an unclean machine stop can leave a zeroed record; it is dropped with its own line, the rest is read."""
    first, second = _raw("commit", memory_id="a"), _raw("assigned", memory_id="c")
    path = _put_queue_file(tmp_path, _jline(first) + b"\x00" * 64 + b"\n" + _jline(second))
    before = _tree(tmp_path)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        events = _read(tmp_path)
    assert [e.memory_id for e in events] == ["a", "c"]
    messages = _errors(caplog)
    assert len(messages) == 1 and str(path) in messages[0] and "line 2" in messages[0]
    assert _tree(tmp_path) == before


def test_a_zero_filled_tail_with_no_newline_is_not_read_and_the_next_append_makes_it_one_dropped_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = engine_files.event_from_json(_raw("commit", memory_id="a"))
    second = engine_files.event_from_json(_raw("assigned", memory_id="c"))
    path = _put_queue_file(tmp_path, _jline(_raw("commit", memory_id="a")) + b"\x00" * 64)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert _read(tmp_path) == [first]
    assert _errors(caplog) == []  # no newline yet: a torn tail, not a line
    caplog.clear()
    assert engine_files.append_event(tmp_path, second) is True
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert _read(tmp_path) == [first, second]
    messages = _errors(caplog)
    assert len(messages) == 1 and str(path) in messages[0] and "line 2" in messages[0]


def test_a_record_cut_short_by_a_zero_fill_is_dropped_and_zeros_in_front_of_a_record_drop_the_whole_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    keep = [_raw("commit", memory_id="a"), _raw("showing", memory_id="z")]
    cut = json.dumps(_raw("assigned", memory_id="cut")).encode()[:20] + b"\x00" * 40 + b"\n"
    glued = b"\x00" * 16 + _jline(_raw("assigned", memory_id="glued"))
    path = _put_queue_file(tmp_path, _jline(keep[0]) + cut + glued + _jline(keep[1]))
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        events = _read(tmp_path)
    assert [e.memory_id for e in events] == ["a", "z"]  # the glued record is NOT returned: a damaged line is dropped whole
    messages = _errors(caplog)
    assert len(messages) == 2 and all(str(path) in m for m in messages)
    assert "line 2" in messages[0] and "line 3" in messages[1]


def test_a_read_fault_raises_and_drops_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = engine_files.queue_path(tmp_path)
    path.mkdir(parents=True)  # a directory where the file should be: opening it is an OSError on every platform
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        with pytest.raises(OSError):
            _read(tmp_path)
    assert not [m for m in _errors(caplog) if "dropped" in m or "discard" in m]
    assert path.is_dir()


def test_an_append_that_cannot_reach_the_folder_is_logged_and_returns_false(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "state").write_bytes(b"a file where the folder should be")
    with caplog.at_level(logging.DEBUG):
        assert engine_files.append_event(tmp_path, _event(0)) is False
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert (tmp_path / "state").read_bytes() == b"a file where the folder should be"


def test_an_append_with_a_directory_in_the_files_place_is_logged_and_returns_false(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = engine_files.queue_path(tmp_path)
    path.mkdir(parents=True)
    with caplog.at_level(logging.DEBUG):
        assert engine_files.append_event(tmp_path, _event(0)) is False
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert path.is_dir()


def test_an_event_that_cannot_be_encoded_is_logged_once_and_the_queue_is_untouched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert engine_files.append_event(tmp_path, _event(0)) is True
    before = _tree(tmp_path)
    bad = Event(kind="commit", at=NOW, memory_id="m", emotions={"joy": float("nan")})
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.append_event(tmp_path, bad) is False
    assert len(_errors(caplog)) == 1
    assert _tree(tmp_path) == before


def test_the_append_asks_append_jsonl_for_a_held_lock_and_logging_with_no_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """design doc section 2.9: no fsync per append; the caller holds the lock, so the append takes none of its own."""
    seen: dict[str, object] = {}
    result: list[int | None] = [10]

    def spy(path: Path, record: object, **kwargs: object):
        seen.update(kwargs)
        seen["path"] = path
        return result[0]

    monkeypatch.setattr(engine_files, "append_jsonl", spy)
    assert engine_files.append_event(tmp_path, _event(0)) is True
    assert seen == {"lock": "held", "on_error": "log", "path": engine_files.queue_path(tmp_path)}
    result[0] = None  # append_jsonl's way of saying it logged and wrote nothing
    assert engine_files.append_event(tmp_path, _event(0)) is False


def test_an_oserror_from_the_append_is_logged_and_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing(*args: object, **kwargs: object):
        raise OSError("disk full")

    monkeypatch.setattr(engine_files, "append_jsonl", failing)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.append_event(tmp_path, _event(0)) is False
    assert len(_errors(caplog)) == 1 and "disk full" in _errors(caplog)[0]


# --- the queue: read by position, emptied (S2, S14, S17, S21) ------------------------------------


def _queue_of(persona_dir: Path, count: int) -> list[Event]:
    events = [_event(i) for i in range(count)]
    for event in events:
        assert engine_files.append_event(persona_dir, event) is True
    return events


def test_read_queue_returns_the_lines_after_the_position_and_the_line_count(tmp_path: Path) -> None:
    events = _queue_of(tmp_path, 5)
    whole = engine_files.read_queue(tmp_path, from_line=0)
    assert isinstance(whole, engine_files.QueueRead)
    assert (whole.events, whole.end_line) == (events, 5)
    after_two = engine_files.read_queue(tmp_path, from_line=2)
    assert (after_two.events, after_two.end_line) == (events[2:], 5)
    at_end = engine_files.read_queue(tmp_path, from_line=5)
    assert (at_end.events, at_end.end_line) == ([], 5)


def test_a_bad_line_before_the_position_is_not_read_or_logged_and_one_after_it_is_logged_with_its_number(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _put_queue_file(tmp_path, _jline(_raw("commit", memory_id="a")) + b"{oops\n" + _jline(_raw("commit", memory_id="c")) + b"{bad\n")
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        result = engine_files.read_queue(tmp_path, from_line=2)
    assert [e.memory_id for e in result.events] == ["c"] and result.end_line == 4
    messages = _errors(caplog)
    assert len(messages) == 1 and str(path) in messages[0] and "line 4" in messages[0]


def test_a_position_past_the_end_means_the_queue_was_emptied_and_it_is_read_again_from_line_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _queue_of(tmp_path, 2)
    calls: list[tuple[Path, int]] = []
    real = engine_files.read_jsonl_from

    def spy(path: Path, start_line: int = 0):
        calls.append((Path(path), start_line))
        return real(path, start_line)

    monkeypatch.setattr(engine_files, "read_jsonl_from", spy)
    result = engine_files.read_queue(tmp_path, from_line=7)
    assert (result.events, result.end_line) == (events, 2)
    queue = engine_files.queue_path(tmp_path)
    assert calls == [(queue, 7), (queue, 0)]
    calls.clear()
    assert engine_files.read_queue(tmp_path, from_line=2).events == []  # at the end is not past it: one read
    assert calls == [(queue, 2)]


@pytest.mark.parametrize("from_line", [0, 3, 10**400])
def test_a_missing_queue_is_no_events_and_end_line_zero_for_any_position(tmp_path: Path, from_line: int) -> None:
    result = engine_files.read_queue(tmp_path, from_line=from_line)
    assert (result.events, result.end_line) == ([], 0)
    assert list(tmp_path.iterdir()) == []


def test_a_huge_position_reads_the_whole_queue_from_line_zero(tmp_path: Path) -> None:
    events = _queue_of(tmp_path, 3)
    assert engine_files.read_queue(tmp_path, from_line=10**400).events == events


def test_a_negative_position_is_read_as_zero_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """design doc section 2.9: a position outside its range is held to the range. `tmp_path` is unique, so the throttle cannot hide the line."""
    events = _queue_of(tmp_path, 3)
    starts: list[int] = []
    real = engine_files.read_jsonl_from

    def spy(path: Path, start_line: int = 0):
        starts.append(start_line)
        return real(path, start_line)

    monkeypatch.setattr(engine_files, "read_jsonl_from", spy)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        result = engine_files.read_queue(tmp_path, from_line=-4)
    assert result == engine_files.read_queue(tmp_path, from_line=0)
    assert result.events == events and result.end_line == 3
    assert all(start >= 0 for start in starts)
    messages = _errors(caplog)
    assert len(messages) == 1 and "-4" in messages[0] and str(engine_files.queue_path(tmp_path)) in messages[0]


def test_empty_queue_leaves_an_empty_file_returns_true_and_the_next_append_starts_afresh(tmp_path: Path) -> None:
    _queue_of(tmp_path, 3)
    path = engine_files.queue_path(tmp_path)
    assert engine_files.empty_queue(tmp_path) is True
    assert path.is_file() and path.read_bytes() == b""  # emptied, never deleted
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]  # no .bak, temp or .lock beside it
    result = engine_files.read_queue(tmp_path, from_line=0)
    assert (result.events, result.end_line) == ([], 0)
    fresh = _event(7)
    assert engine_files.append_event(tmp_path, fresh) is True
    assert path.read_bytes() == _jline(engine_files.event_to_json(fresh))


def test_empty_queue_asks_write_jsonl_for_an_empty_file_with_no_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Path, object, dict[str, object]]] = []
    real = engine_files.write_jsonl

    def spy(path: Path, records: object, **kwargs: object):
        seen.append((Path(path), records, kwargs))
        return real(path, records, **kwargs)

    monkeypatch.setattr(engine_files, "write_jsonl", spy)
    assert engine_files.empty_queue(tmp_path) is True
    assert seen == [(engine_files.queue_path(tmp_path), [], {"fsync": False})]


def test_a_failed_emptying_returns_false_logs_raises_nothing_and_leaves_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _queue_of(tmp_path, 3)
    queue = engine_files.queue_path(tmp_path)
    before = queue.read_bytes()

    def failing(*args: object, **kwargs: object):
        raise OSError("disk full")

    monkeypatch.setattr(engine_files, "write_jsonl", failing)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert engine_files.empty_queue(tmp_path) is False
    assert any(r.levelno >= logging.WARNING and "disk full" in r.getMessage() for r in caplog.records)
    assert queue.read_bytes() == before  # the queue is as it was


def test_emptying_a_queue_path_that_is_a_folder_returns_false_and_keeps_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(dev_constants, "FILE_RETRY_SLEEP_S", 0)  # a replace onto a folder is a retried PermissionError on Windows and macOS
    queue = engine_files.queue_path(tmp_path)
    queue.mkdir(parents=True)
    (queue / "inner").write_bytes(b"keep")
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert engine_files.empty_queue(tmp_path) is False
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert queue.is_dir() and (queue / "inner").read_bytes() == b"keep"


def test_the_position_rule_in_one_sequence(tmp_path: Path) -> None:
    """Append 3, read to the end, empty, append 2: the stored position 3 is past the end of the new queue, so it is read from line 0."""
    first = _queue_of(tmp_path, 3)
    read = engine_files.read_queue(tmp_path, from_line=0)
    assert (read.events, read.end_line) == (first, 3)
    assert engine_files.empty_queue(tmp_path) is True
    later = [_event(10), _event(11)]
    for event in later:
        assert engine_files.append_event(tmp_path, event) is True
    again = engine_files.read_queue(tmp_path, from_line=read.end_line)
    assert (again.events, again.end_line) == (later, 2)


# --- every file operation goes through its accessor (design doc section 2.9) -------------------------------

_RAW_FILE_ATTRS = frozenset(
    "open unlink iterdir stat lstat exists is_file is_dir mkdir rmdir read_text write_text read_bytes write_bytes "
    "touch rename scandir listdir walk glob rglob".split()
)


def test_engine_files_makes_no_raw_file_call_and_takes_the_lock_in_one_place() -> None:
    tree = ast.parse(Path(engine_files.__file__).read_text(encoding="utf-8"))
    raw: list[str] = []
    locked_within_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raw += [f"import {a.name}" for a in node.names if a.name.split(".")[0] in {"os", "shutil", "glob"}]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {"os", "shutil", "glob"}:
            raw.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"open", "file_lock"}:
                raw.append(f"line {node.lineno}: {func.id}(...)")
            elif isinstance(func, ast.Attribute):
                base = func.value.id if isinstance(func.value, ast.Name) else None
                if func.attr in _RAW_FILE_ATTRS and base != "file_ops":
                    raw.append(f"line {node.lineno}: .{func.attr}(...)")
                if func.attr == "locked_within":
                    locked_within_calls += 1
                if base == "file_lock" and func.attr == "file_lock":
                    raw.append(f"line {node.lineno}: file_lock.file_lock(...)")
    assert raw == []
    assert locked_within_calls == 1  # state_lock is the one bounded acquisition


def test_the_queue_read_goes_through_read_jsonl_from_with_the_queue_path_and_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _queue_of(tmp_path, 3)
    calls: list[tuple[Path, int]] = []
    real = engine_files.read_jsonl_from

    def spy(path: Path, start_line: int = 0):
        calls.append((Path(path), start_line))
        return real(path, start_line)

    monkeypatch.setattr(engine_files, "read_jsonl_from", spy)
    assert len(_read(tmp_path, from_line=1)) == 2
    assert calls == [(engine_files.queue_path(tmp_path), 1)]


def test_no_queue_or_change_log_function_takes_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The callers hold the one lock (it is not re-entrant): none of these may take it a second time."""

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("a function of the queue or the change log took the lock")

    monkeypatch.setattr(engine_files, "state_lock", forbidden)
    monkeypatch.setattr(file_lock, "locked_within", forbidden)
    monkeypatch.setattr(file_lock, "file_lock", forbidden)
    monkeypatch.setattr(json_io, "file_lock", forbidden)  # json_io imports the lock by name: its writers must not take it either
    assert engine_files.append_event(tmp_path, engine_files.event_from_json(_raw("start"))) is True
    assert len(_read(tmp_path)) == 1
    assert engine_files.empty_queue(tmp_path) is True
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


@pytest.mark.parametrize("error", [OverflowError("date value out of range"), OSError("pre-1970 on Windows")])
def test_a_change_that_cannot_be_encoded_or_dated_is_logged_and_returns_false(
    error: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.append_change(tmp_path, _change(NOW, joy=float("nan"))) is False
    assert len(_errors(caplog)) == 1
    caplog.clear()

    def failing(dt: datetime) -> datetime:
        raise error

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


def test_the_listing_goes_through_list_dir_and_stats_each_matching_name_through_stat_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch_days(tmp_path, "2026-06-02", "2026-06-01")
    folder = engine_files.changes_dir(tmp_path)
    (folder / "notes.txt").write_text("x", encoding="utf-8")
    (folder / "2026-13-45.jsonl").write_text("x", encoding="utf-8")
    (folder / "2026-06-04.jsonl").mkdir()  # matches the pattern, so it is asked about, and refused as not a regular file
    listed: list[Path] = []
    statted: list[Path] = []
    real_list, real_stat = file_ops.list_dir, file_ops.stat_path

    def list_spy(path: str | Path) -> list[str]:
        listed.append(Path(path))
        return real_list(path)

    def stat_spy(path: str | Path):
        statted.append(Path(path))
        return real_stat(path)

    monkeypatch.setattr(file_ops, "list_dir", list_spy)
    monkeypatch.setattr(file_ops, "stat_path", stat_spy)
    assert engine_files.list_change_files(tmp_path) == ["2026-06-01.jsonl", "2026-06-02.jsonl"]
    assert listed == [folder]
    assert sorted(statted) == [folder / "2026-06-01.jsonl", folder / "2026-06-02.jsonl", folder / "2026-06-04.jsonl"]


def test_an_entry_whose_stat_fails_or_finds_nothing_is_skipped_and_the_others_are_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch_days(tmp_path, "2026-06-01", "2026-06-02", "2026-06-03")
    real_stat = file_ops.stat_path

    def flaky(path: str | Path):
        if Path(path).name == "2026-06-01.jsonl":
            raise PermissionError("denied")
        if Path(path).name == "2026-06-02.jsonl":
            return None  # vanished between the listing and the stat
        return real_stat(path)

    monkeypatch.setattr(file_ops, "stat_path", flaky)
    assert engine_files.list_change_files(tmp_path) == ["2026-06-03.jsonl"]


def test_a_listing_error_other_than_missing_is_logged_and_read_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refuse(path: str | Path) -> list[str]:
        raise PermissionError("denied")

    monkeypatch.setattr(file_ops, "list_dir", refuse)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.list_change_files(tmp_path) == []
    assert len(_errors(caplog)) == 1


def test_prune_deletes_through_delete_file_and_never_offers_it_a_folder_named_like_a_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch_days(tmp_path, "2020-01-01", "2026-06-10")
    folder = engine_files.changes_dir(tmp_path)
    (folder / "2020-01-03.jsonl").mkdir()
    asked: list[Path] = []
    real_delete = file_ops.delete_file

    def spy(path: str | Path) -> None:
        asked.append(Path(path))
        real_delete(path)

    monkeypatch.setattr(file_ops, "delete_file", spy)
    assert engine_files.prune_changes(tmp_path, "2026-06-10") == ["2020-01-01.jsonl"]
    assert asked == [folder / "2020-01-01.jsonl"]
    assert (folder / "2020-01-03.jsonl").is_dir()


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
    real_delete = file_ops.delete_file

    def refuse_the_second(path: str | Path) -> None:
        if Path(path).name == "2026-01-02.jsonl":
            raise PermissionError("in use")
        real_delete(path)

    monkeypatch.setattr(file_ops, "delete_file", refuse_the_second)
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


def test_the_reader_takes_a_naive_start_as_utc_and_refuses_nothing_else(tmp_path: Path) -> None:
    since = _local_noon(date(2026, 6, 10))
    _write_day(tmp_path, "2026-06-10", _line(since, joy=1.0))
    naive = since.replace(tzinfo=None)
    assert [e.emotions["joy"] for e in engine_files.read_changes(tmp_path, naive)] == [1.0]
    with pytest.raises(ValueError):
        engine_files.read_changes(tmp_path, "soon")  # type: ignore[arg-type]


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


def _kind_line(kind: object, at: datetime) -> str:
    return json.dumps({"kind": kind, "at": at.isoformat(), "emotions": {"joy": 9.0}})


@pytest.mark.parametrize("kind", [[], {}, [1], [[]], {"a": 1}], ids=["list", "dict", "list-of-one", "nested", "dict-of-one"])
def test_the_reader_skips_a_line_whose_kind_cannot_be_looked_up_and_reads_the_rest(
    kind: object, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unhashable `kind` (valid JSON) used to raise TypeError out of the membership test and abort the whole read."""
    since = _local_noon(date(2026, 6, 10))
    path = _write_day(
        tmp_path,
        "2026-06-10",
        _line(since, joy=1.0),
        _kind_line(kind, since),
        _line(since + timedelta(minutes=1), joy=2.0),
    )
    before = path.read_bytes()
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        got = engine_files.read_changes(tmp_path, since)
    assert [e.emotions["joy"] for e in got] == [1.0, 2.0]
    assert len(_errors(caplog)) == 1 and "unknown kind" in _errors(caplog)[0]
    assert path.read_bytes() == before


@pytest.mark.no_base_failure(reason="a number, a bool, null or an unknown text kind is hashable, so it already reaches event_from_json's own kind check and is dropped at base; this pins that the whole set of non-text kinds is dropped one line at a time")
@pytest.mark.parametrize("kind", [5, 1.5, True, None, "x", "Change"], ids=["int", "float", "bool", "null", "text", "case"])
def test_the_reader_skips_a_line_whose_kind_is_not_a_known_text_and_reads_the_rest(
    kind: object, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    since = _local_noon(date(2026, 6, 10))
    _write_day(
        tmp_path,
        "2026-06-10",
        _line(since, joy=1.0),
        _kind_line(kind, since),
        _line(since + timedelta(minutes=1), joy=2.0),
    )
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        got = engine_files.read_changes(tmp_path, since)
    assert [e.emotions["joy"] for e in got] == [1.0, 2.0]
    assert len(_errors(caplog)) == 1


@pytest.mark.no_base_failure(reason="read_queue already checks the kind as text first (event_from_json); this pins that the queue and the change log answer the same lines the same way")
@pytest.mark.parametrize("kind", [[], {}, 5, None, "x"], ids=["list", "dict", "int", "null", "text"])
def test_the_queue_skips_a_line_whose_kind_is_not_a_known_text_and_reads_the_rest(
    kind: object, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = engine_files.queue_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([json.dumps(_raw("commit")), _kind_line(kind, NOW), json.dumps(_raw("commit", memory_id="m2"))]) + "\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        got = engine_files.read_queue(tmp_path, from_line=0)
    assert [e.memory_id for e in got.events] == ["m1", "m2"] and got.end_line == 3
    assert len(_errors(caplog)) == 1


@pytest.mark.no_base_failure(reason="guards the new hold line against firing in the change-log reader: a commit line there is dropped as not a change with one error line and no 'held' warning; the old code has no hold line to misfire")
def test_a_commit_line_in_the_change_log_is_dropped_as_not_a_change_without_a_hold_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    since = _local_noon(date(2026, 6, 10))
    _write_day(tmp_path, "2026-06-10", json.dumps(_raw("commit", modifier=1e308)), _line(since, joy=1.0))
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        got = engine_files.read_changes(tmp_path, since)
    assert [e.emotions["joy"] for e in got] == [1.0]
    assert [r.levelno for r in caplog.records if r.name == LOGGER] == [logging.ERROR]
    assert "not a change" in caplog.records[-1].getMessage()


def test_the_reader_treats_an_unlistable_folder_as_missing_with_a_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "emotion_changes").write_text("a file", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert engine_files.read_changes(tmp_path, NOW) == []
    assert len(_errors(caplog)) == 1


def _reader_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The WARNING lines of the shared JSONL readers (`brain.health.*`): which of them logs is the reader's own business."""
    return [r.getMessage() for r in caplog.records if r.name.startswith("brain.health.") and r.levelno == logging.WARNING]


def _unloadable_digits() -> str:
    return '{"x": ' + "1" * (sys.get_int_max_str_digits() + 1) + "}"  # the parser refuses it: ValueError


def _unloadable_nesting() -> str:
    return '{"x": ' + "[" * 200_000 + "]" * 200_000 + "}"  # past the parser's recursion limit: RecursionError


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(
            _unloadable_digits,
            marks=pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="the interpreter's integer digit limit is off"),
        ),
        _unloadable_nesting,
    ],
    ids=["digits", "nesting"],
)
def test_the_reader_skips_a_line_the_json_parser_cannot_load_with_a_warning_naming_file_and_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, bad
) -> None:
    since = _local_noon(date(2026, 6, 10))
    unloadable = bad()
    with pytest.raises((ValueError, RecursionError)) as parser_error:  # the apparatus: this line really is one the parser refuses
        json.loads(unloadable)
    assert not isinstance(parser_error.value, json.JSONDecodeError)
    path = _write_day(tmp_path, "2026-06-10", _line(since, joy=1.0), unloadable, _line(since + timedelta(minutes=1), joy=2.0))
    before = path.read_bytes()
    with caplog.at_level(logging.WARNING):
        got = engine_files.read_changes(tmp_path, since)
    assert [e.emotions["joy"] for e in got] == [1.0, 2.0]
    assert path.read_bytes() == before  # the reader repairs nothing
    named = [m for m in _reader_warnings(caplog) if str(path) in m and "line 2" in m]
    assert len(named) == 1


@pytest.mark.no_base_failure(reason="pins what the switch to the raw reader must keep: iter_jsonl warned about a malformed line the same way, naming the file and the line")
def test_a_malformed_change_line_is_skipped_with_a_warning_naming_file_and_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    since = _local_noon(date(2026, 6, 10))
    path = _write_day(tmp_path, "2026-06-10", _line(since, joy=1.0), "{not json", _line(since, joy=2.0))
    with caplog.at_level(logging.WARNING):
        got = engine_files.read_changes(tmp_path, since)
    assert [e.emotions["joy"] for e in got] == [1.0, 2.0]
    assert len([m for m in _reader_warnings(caplog) if str(path) in m and "line 2" in m]) == 1


@pytest.mark.no_base_failure(reason="pins the I/O tolerance design doc section 2.10 requires, which iter_jsonl had too: a day file that cannot be opened adds nothing, with a warning naming it, and the other days are still read")
def test_an_unopenable_day_file_adds_nothing_and_the_other_days_are_still_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    since = _local_noon(date(2026, 6, 10))
    _write_day(tmp_path, "2026-06-10", _line(since, joy=1.0))
    broken = _write_day(tmp_path, "2026-06-11", _line(since + timedelta(days=1), joy=8.0))
    _write_day(tmp_path, "2026-06-12", _line(since + timedelta(days=2), joy=3.0))
    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        if Path(path).name == broken.name:
            raise PermissionError(13, "denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(json_io, "open", fake_open, raising=False)
    with caplog.at_level(logging.WARNING):
        got = engine_files.read_changes(tmp_path, since)
    assert [e.emotions["joy"] for e in got] == [1.0, 3.0]
    assert any(broken.name in m and "(io:" in m for m in _reader_warnings(caplog))


@pytest.mark.no_base_failure(reason="pins the I/O tolerance design doc section 2.10 requires, which iter_jsonl had too: a read fault in the middle of a day file ends that file's lines, keeps what was read, warns naming the file, and the next day is still read")
def test_a_read_fault_mid_file_ends_that_file_keeps_what_was_read_and_reads_the_next_day(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    since = _local_noon(date(2026, 6, 10))
    faulty = _write_day(tmp_path, "2026-06-10", _line(since, joy=1.0), _line(since + timedelta(minutes=1), joy=9.0))
    _write_day(tmp_path, "2026-06-11", _line(since + timedelta(days=1), joy=3.0))
    first_line = (_line(since, joy=1.0) + "\n").encode("utf-8")  # ends in a newline, so the line is whole before the fault

    class _FaultAfterOneLine(io.RawIOBase):
        def __init__(self) -> None:
            self._served = False

        def readable(self) -> bool:
            return True

        def readinto(self, buffer) -> int:
            if self._served:
                raise OSError(5, "input/output error")
            self._served = True
            buffer[: len(first_line)] = first_line
            return len(first_line)

    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        if Path(path).name == faulty.name:
            return io.BufferedReader(_FaultAfterOneLine())
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(json_io, "open", fake_open, raising=False)
    with caplog.at_level(logging.WARNING):
        got = engine_files.read_changes(tmp_path, since)
    assert [e.emotions["joy"] for e in got] == [1.0, 3.0]
    assert any(faulty.name in m and "(io:" in m for m in _reader_warnings(caplog))


def test_a_change_written_then_read_round_trips(tmp_path: Path) -> None:
    event = _change(NOW, joy=-2.5, grief=0.25)
    engine_files.append_change(tmp_path, event)
    assert engine_files.read_changes(tmp_path, NOW - timedelta(days=1)) == [event]


def test_the_first_append_creates_only_the_folder_and_the_day_file(tmp_path: Path) -> None:
    engine_files.append_change(tmp_path, _change(NOW))
    assert {p.name for p in (tmp_path / "state").rglob("*")} == {"emotion_changes", f"{day_of(NOW)}.jsonl"}


# ================================================================================================
# The hooks (Engine 3, #578): memory writes queue events; a hook appends, never folds, never raises
# ================================================================================================

HOOK_LOGGER = "brain.emotion.engine"
HOOK_AT = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
EngineCls = engine_module.EmotionEngine


def _hook_mem(
    memory_type: str = "conversation",
    *,
    emotions: dict | None = None,
    tags: list | None = None,
    metadata: dict | None = None,
    created_at: datetime | None = None,
    mid: str | None = None,
) -> Memory:
    emo = {"joy": 6.0} if emotions is None else dict(emotions)
    numbers = [v for v in emo.values() if isinstance(v, (int, float)) and math.isfinite(v)]
    return Memory(
        id=mid or str(uuid.uuid4()),
        content="a hook test memory",
        memory_type=memory_type,
        domain="us",
        created_at=created_at or HOOK_AT,
        emotions=emo,
        tags=list(tags or []),
        importance=1.0,
        score=float(sum(numbers)),
        metadata=dict(metadata or {}),
        peak_emotion_intensity=max(numbers, default=0.0),
    )


@pytest.fixture
def hook_store(tmp_path: Path) -> Iterator[MemoryStore]:
    store = MemoryStore(tmp_path / "memories.db")
    yield store
    store.close()


def _events(persona_dir: Path) -> list[Event]:
    return _read(persona_dir)


def _lines(persona_dir: Path) -> list[dict]:
    path = engine_files.queue_path(persona_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _start_state(persona_dir: Path, start: datetime | None) -> None:
    engine_files.save_state(persona_dir, EngineState(saved_at=NOW, persona_start_at=start))


class _Probe:
    """Call-through spies. Each call records its arguments and whether the engine lock was held (the "held flag").

    `state_lock` is wrapped so `held` is True for the body of the `with`. A spy that is never called proves nothing, so a
    test that relies on one asserts `n(name) >= 1`.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self.held = False
        self.lock_calls = 0
        self.calls: dict[str, list[tuple[tuple, bool]]] = {}
        real_lock = engine_files.state_lock
        probe = self

        @contextlib.contextmanager
        def lock(persona_dir: Path) -> Iterator[bool]:
            probe.lock_calls += 1
            with real_lock(persona_dir) as got:
                probe.held = True
                try:
                    yield got
                finally:
                    probe.held = False

        monkeypatch.setattr(engine_files, "state_lock", lock)

    def wrap(self, owner: object, name: str, key: str | None = None) -> None:
        real = getattr(owner, name)
        label = key or name

        def spy(*args: object, **kwargs: object) -> object:
            self.calls.setdefault(label, []).append((args, self.held))
            return real(*args, **kwargs)

        self._monkeypatch.setattr(owner, name, spy)

    def n(self, name: str) -> int:
        return len(self.calls.get(name, []))

    def held_at(self, name: str) -> list[bool]:
        return [held for _args, held in self.calls.get(name, [])]


# --- C1: a counted create appends one commit event; nothing is folded -----------------------------


def test_a_counted_create_appends_one_commit_event_and_folds_nothing(tmp_path: Path, hook_store: MemoryStore) -> None:
    mem = _hook_mem(emotions={"joy": 6.0, "climax": 8.0})
    hook_store.create(mem)
    events = _events(tmp_path)
    assert len(events) == 1
    event = events[0]
    assert event.kind == "commit" and event.memory_id == mem.id and event.at == mem.created_at
    assert dict(event.emotions) == {"joy": 6.0, "climax": 8.0}
    assert event.modifier == 1.0 and event.climax == 8.0
    assert not engine_files.state_path(tmp_path).exists()  # appended, not folded
    before = engine_files.queue_path(tmp_path).read_bytes()
    hook_store.create(_hook_mem(emotions={"grief": 2.0}))
    after = engine_files.queue_path(tmp_path).read_bytes()
    assert after.startswith(before) and len(_events(tmp_path)) == 2
    assert not engine_files.state_path(tmp_path).exists()


def test_create_with_fold_false_writes_no_queue_and_no_state_folder(tmp_path: Path, hook_store: MemoryStore) -> None:
    hook_store.create(_hook_mem(), fold=False)
    assert not engine_files.queue_path(tmp_path).exists()
    assert not (tmp_path / "state").exists()
    hook_store.create(_hook_mem())  # the positive control: the same store queues when fold is on
    assert len(_events(tmp_path)) == 1


# --- C1b: faded versus stored --------------------------------------------------------------------


def test_commit_and_assigned_carry_faded_emotions_and_the_stored_climax(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EngineCls, "faded_value", lambda self, memory, emotion: memory.emotions[emotion] / 4)
    hook_store.create(_hook_mem(emotions={"joy": 8.0, "climax": 8.0}))
    empty = _hook_mem(emotions={})
    hook_store.create(empty, fold=False)
    hook_store.update(empty.id, emotions={"joy": 8.0, "climax": 8.0})
    commit, assigned = _events(tmp_path)
    assert commit.kind == "commit" and dict(commit.emotions) == {"joy": 2.0, "climax": 2.0}
    assert commit.climax == 8.0
    assert assigned.kind == "assigned" and dict(assigned.emotions) == {"joy": 2.0, "climax": 2.0}
    assert "climax" not in _lines(tmp_path)[1]


def test_a_faded_value_that_raises_drops_that_name_with_a_line_and_queues_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def faded(self: object, memory: Memory, emotion: str) -> float:
        if emotion == "grief":
            raise RuntimeError("no half-life")
        return memory.emotions[emotion]

    monkeypatch.setattr(EngineCls, "faded_value", faded)
    engine = engine_module.engine_for(tmp_path)
    with caplog.at_level(logging.WARNING, logger=HOOK_LOGGER):
        engine.on_memory_committed(_hook_mem(emotions={"joy": 6.0, "grief": 3.0}))
    assert [dict(e.emotions) for e in _events(tmp_path)] == [{"joy": 6.0}]
    assert any("grief" in r.getMessage() for r in caplog.records)
    # every name failing leaves nothing to fold: this hook appends the start-only event instead (B02.6; the spec makes it a direct apply, Engine 4 moves it)
    monkeypatch.setattr(EngineCls, "faded_value", lambda self, memory, emotion: 1 / 0)
    engine.on_memory_committed(_hook_mem(emotions={"joy": 6.0}))
    assert [e.kind for e in _events(tmp_path)] == ["commit", "start"]


def test_a_nan_value_is_dropped_and_a_row_left_with_nothing_queues_the_start_only_event(tmp_path: Path) -> None:
    engine = engine_module.engine_for(tmp_path)
    engine.on_memory_committed(_hook_mem(emotions={"joy": math.nan, "grief": 3.0}))
    engine.on_memory_committed(_hook_mem(emotions={"joy": math.nan}))
    first, second = _events(tmp_path)
    assert first.kind == "commit" and dict(first.emotions) == {"grief": 3.0}
    assert second.kind == "start" and second.at == HOOK_AT


def test_a_non_text_emotion_key_does_not_abort_the_hook_and_the_callers_data_is_untouched(tmp_path: Path) -> None:
    mem = _hook_mem(emotions={"Joy": 4.0, "joy": 6.0, 7: 1.0, "climax": 8.0})
    before = copy.deepcopy(mem)
    engine_module.engine_for(tmp_path).on_memory_committed(mem)
    event = _events(tmp_path)[0]
    assert dict(event.emotions) == {"joy": 5.0, "climax": 8.0}  # two spellings of one name are averaged
    assert event.climax == 8.0
    assert mem == before


def test_the_hooks_leave_the_callers_memory_and_update_argument_alone(tmp_path: Path, hook_store: MemoryStore) -> None:
    engine = engine_module.engine_for(tmp_path)
    mem = _hook_mem(emotions={"Joy": 4.0, "joy": 6.0, "Grief": 2.0})
    before = copy.deepcopy(mem)
    engine.on_memory_committed(mem)
    engine.on_emotions_assigned(mem)
    engine.on_memory_shown(mem, None)
    assert mem == before
    empty = _hook_mem(emotions={})
    hook_store.create(empty, fold=False)
    given = {"Joy": 4.0, "joy": 6.0}
    given_before = dict(given)
    hook_store.update(empty.id, emotions=given)
    assert given == given_before
    assert [e.kind for e in _events(tmp_path)][-1] == "assigned"


# --- C2: which types count ----------------------------------------------------------------------

_COUNTED_ROWS = {
    "conversation type": {"memory_type": "conversation"},
    "conversation tag": {"memory_type": "heartbeat", "tags": ["conversation"]},
    "conversation summary": {"memory_type": "heartbeat", "metadata": {"source_summary": "conversation:abc"}},
    "add_memory": {"memory_type": "meta", "metadata": {"tags": [], "importance": 5}},
    "making": {"memory_type": "making"},
    "reach-out delivered": {"memory_type": "initiate_outbound", "metadata": {"initiate_state": "delivered"}},
    "note": {"memory_type": "initiate_outbound", "metadata": {"initiate_state": "note"}},
    "peer": {"memory_type": "kindled_peer", "metadata": {"peer_id": "p1"}},
    "research": {"memory_type": "research"},
}

_UNCOUNTED_ROWS = {
    "monologue": {"memory_type": "monologue_inner"},
    "monologue reflex": {"memory_type": "monologue_reflex"},
    "emotion only": {"memory_type": "emotion_only"},
    "journal": {"memory_type": "journal"},
    "climax journal": {"memory_type": "climax_journal"},
    "grief breadcrumb": {"memory_type": "grief_breadcrumb"},
    "dream": {"memory_type": "dream"},
    "heartbeat": {"memory_type": "heartbeat"},
    "reflex": {"memory_type": "reflex"},
    "file write": {"memory_type": "file_write"},
    "self model resolved": {"memory_type": "self_model_resolved"},
    "reach-out pending": {"memory_type": "initiate_outbound", "metadata": {"initiate_state": "pending"}},
    "reach-out unknown state": {"memory_type": "initiate_outbound", "metadata": {"initiate_state": "sent-ish"}},
    "reach-out non-text state": {"memory_type": "initiate_outbound", "metadata": {"initiate_state": 3}},
    "conversation tag with a session id": {"memory_type": "heartbeat", "tags": ["conversation:abc"]},
}


@pytest.mark.parametrize("label", sorted(_COUNTED_ROWS))
def test_a_counted_type_queues_a_commit_event(tmp_path: Path, hook_store: MemoryStore, label: str) -> None:
    hook_store.create(_hook_mem(**_COUNTED_ROWS[label]))
    events = _events(tmp_path)
    assert [e.kind for e in events] == ["commit"]


@pytest.mark.parametrize("label", sorted(_UNCOUNTED_ROWS))
def test_an_uncounted_type_queues_no_commit_and_asks_for_no_modifier_or_trust(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    _start_state(tmp_path, HOOK_AT - timedelta(days=30))  # an earlier start: no start event is wanted either
    probe = _Probe(monkeypatch)
    probe.wrap(engine_module, "source_modifier")
    probe.wrap(engine_files, "peer_trust")
    hook_store.create(_hook_mem(**_UNCOUNTED_ROWS[label]))
    assert _events(tmp_path) == []
    assert probe.n("source_modifier") == 0 and probe.n("peer_trust") == 0
    # the positive control: a counted peer row on the same store queues, and then both spies are called
    hook_store.create(_hook_mem(**_COUNTED_ROWS["peer"]))
    assert [e.kind for e in _events(tmp_path)] == ["commit"]
    assert probe.n("source_modifier") >= 1 and probe.n("peer_trust") >= 1


# --- C3: modifiers (read from the raw line: the reader holds a modifier into range) ---------------


def _raw_modifier(persona_dir: Path) -> float:
    return _lines(persona_dir)[-1]["modifier"]


def test_modifiers_of_each_source(tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    hook_store.create(_hook_mem())
    assert _raw_modifier(tmp_path) == 1.0
    hook_store.create(_hook_mem(**_COUNTED_ROWS["peer"]))  # no relationship record: the 0.0 stand-in
    assert _raw_modifier(tmp_path) == pytest.approx(0.1)
    hook_store.create(_hook_mem(**_COUNTED_ROWS["research"]))  # no rating
    assert _raw_modifier(tmp_path) == pytest.approx(0.1)
    seen: list[tuple] = []

    def trust(persona_dir: Path, peer_id: str) -> float:
        seen.append((persona_dir, peer_id))
        return 1.0 if len(seen) == 1 else 0.5

    monkeypatch.setattr(engine_files, "peer_trust", trust)
    hook_store.create(_hook_mem(**_COUNTED_ROWS["peer"]))
    assert _raw_modifier(tmp_path) == pytest.approx(1.0)
    hook_store.create(_hook_mem(**_COUNTED_ROWS["peer"]))
    assert _raw_modifier(tmp_path) == pytest.approx(0.55)
    assert seen == [(tmp_path.resolve(), "p1")] * 2


def test_a_peer_row_without_a_peer_id_does_not_ask_for_trust(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _Probe(monkeypatch)
    probe.wrap(engine_files, "peer_trust")
    hook_store.create(_hook_mem("kindled_peer", metadata={}))
    assert probe.n("peer_trust") == 0 and _raw_modifier(tmp_path) == pytest.approx(0.1)
    hook_store.create(_hook_mem("kindled_peer", metadata={"peer_id": "p1"}))
    assert probe.n("peer_trust") == 1  # the control


# --- C4: the start-only event -------------------------------------------------------------------


def _uncounted(created_at: datetime = HOOK_AT) -> Memory:
    return _hook_mem("heartbeat", created_at=created_at)


def test_an_uncounted_row_queues_a_start_event_when_no_start_is_stored(tmp_path: Path, hook_store: MemoryStore) -> None:
    hook_store.create(_uncounted())
    events = _events(tmp_path)
    assert [(e.kind, e.at) for e in events] == [("start", HOOK_AT)]
    assert not engine_files.state_path(tmp_path).exists()


def test_an_uncounted_row_queues_a_start_event_when_the_state_is_unreadable(tmp_path: Path, hook_store: MemoryStore) -> None:
    path = engine_files.state_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{ this is not json")
    hook_store.create(_uncounted())
    assert [e.kind for e in _events(tmp_path)] == ["start"]


def test_the_start_event_follows_the_stored_start(tmp_path: Path, hook_store: MemoryStore) -> None:
    _start_state(tmp_path, HOOK_AT + timedelta(days=1))  # stored start later than the row
    hook_store.create(_uncounted())
    assert [e.kind for e in _events(tmp_path)] == ["start"]
    _start_state(tmp_path, HOOK_AT - timedelta(days=1))  # stored start earlier: nothing
    hook_store.create(_uncounted())
    assert len(_events(tmp_path)) == 1
    _start_state(tmp_path, HOOK_AT)  # equal: the rule is "later than", so nothing
    hook_store.create(_uncounted())
    assert len(_events(tmp_path)) == 1


def test_the_start_check_reads_naive_and_offset_times_as_instants(tmp_path: Path, hook_store: MemoryStore) -> None:
    _start_state(tmp_path, HOOK_AT)  # 09:00 UTC
    hook_store.create(_uncounted(datetime(2026, 6, 1, 9, 0)))  # naive: read as UTC, equal
    hook_store.create(_uncounted(datetime(2026, 6, 1, 14, 0, tzinfo=timezone(timedelta(hours=5)))))  # 09:00 UTC, equal
    assert _events(tmp_path) == []
    hook_store.create(_uncounted(datetime(2026, 6, 1, 13, 59, tzinfo=timezone(timedelta(hours=5)))))  # 08:59 UTC
    assert [e.kind for e in _events(tmp_path)] == ["start"]


def test_a_backup_state_supplies_the_stored_start(tmp_path: Path, hook_store: MemoryStore) -> None:
    _start_state(tmp_path, HOOK_AT)
    _start_state(tmp_path, HOOK_AT)  # a second save makes the first a backup copy
    engine_files.state_path(tmp_path).write_bytes(b"not json")  # the main file is damaged, the backup is readable
    hook_store.create(_uncounted(HOOK_AT + timedelta(days=1)))  # later than the backup's start: nothing
    assert _events(tmp_path) == []
    hook_store.create(_uncounted(HOOK_AT - timedelta(days=1)))
    assert [e.kind for e in _events(tmp_path)] == ["start"]


def test_the_start_check_takes_no_lock_but_an_append_takes_it_once(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_state(tmp_path, HOOK_AT - timedelta(days=1))
    probe = _Probe(monkeypatch)
    hook_store.create(_uncounted())
    assert probe.lock_calls == 0 and _events(tmp_path) == []
    hook_store.create(_hook_mem())  # a commit event: one lock
    assert probe.lock_calls == 1
    hook_store.create(_uncounted(HOOK_AT - timedelta(days=9)))  # a start event: one more
    assert probe.lock_calls == 2
    assert [e.kind for e in _events(tmp_path)] == ["commit", "start"]


def test_the_state_is_read_through_the_cache_not_on_every_create(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_state(tmp_path, HOOK_AT - timedelta(days=1))
    probe = _Probe(monkeypatch)
    probe.wrap(engine_files, "read_state")
    hook_store.create(_uncounted())
    hook_store.create(_uncounted())
    assert probe.n("read_state") == 1
    _start_state(tmp_path, HOOK_AT - timedelta(days=2))  # the file is replaced: the stat key changes
    hook_store.create(_uncounted())
    assert probe.n("read_state") == 2


def test_a_counted_row_with_no_emotions_queues_the_start_event_and_moves_persona_start_at(
    tmp_path: Path, hook_store: MemoryStore
) -> None:
    mem = _hook_mem(emotions={})
    hook_store.create(mem)
    events = _events(tmp_path)
    assert [(e.kind, e.at) for e in events] == [("start", mem.created_at)]
    state = EngineState()
    assert state.persona_start_at is None
    for event in events:
        fold_event(state, event, NOW)
    assert state.persona_start_at == mem.created_at


# --- C5: update -> assigned ---------------------------------------------------------------------


def _row_without_emotions(store: MemoryStore, memory_type: str = "conversation", **kw: object) -> Memory:
    mem = _hook_mem(memory_type, emotions={}, **kw)  # type: ignore[arg-type]
    store.create(mem, fold=False)
    return mem


def test_an_emotions_write_onto_an_empty_counted_row_queues_one_assigned_event(
    tmp_path: Path, hook_store: MemoryStore
) -> None:
    mem = _row_without_emotions(hook_store)
    hook_store.update(mem.id, emotions={"joy": 3.0})
    events = _events(tmp_path)
    assert len(events) == 1
    event = events[0]
    assert event.kind == "assigned" and event.memory_id == mem.id and event.at == mem.created_at
    assert dict(event.emotions) == {"joy": 3.0} and event.modifier == 1.0
    assert "climax" not in _lines(tmp_path)[0]


def test_an_update_that_must_not_fire_fires_nothing(tmp_path: Path, hook_store: MemoryStore) -> None:
    filled = _hook_mem()
    hook_store.create(filled, fold=False)
    hook_store.update(filled.id, emotions={"grief": 4.0})  # filled to filled
    empty = _row_without_emotions(hook_store)
    hook_store.update(empty.id, content="reworded")  # content only
    hook_store.update(empty.id, emotions={"joy": 3.0}, fold=False)  # fold=False
    uncounted = _row_without_emotions(hook_store, "heartbeat")
    hook_store.update(uncounted.id, emotions={"joy": 3.0})  # an uncounted row: no assigned, no start event
    assert _events(tmp_path) == []
    control = _row_without_emotions(hook_store)
    hook_store.update(control.id, emotions={"joy": 3.0})
    assert [e.kind for e in _events(tmp_path)] == ["assigned"]


def test_fold_is_not_a_column_and_an_unknown_field_still_raises(hook_store: MemoryStore) -> None:
    mem = _row_without_emotions(hook_store)
    hook_store.update(mem.id, fold=False, content="ok")
    assert hook_store.get(mem.id, bump=False).content == "ok"
    with pytest.raises(ValueError, match="Unknown update field"):
        hook_store.update(mem.id, bogus=1)


def test_update_asks_the_database_once_for_the_row_and_the_hook_adds_no_second_select(
    tmp_path: Path, hook_store: MemoryStore
) -> None:
    def selects(fold: bool) -> int:
        mem = _row_without_emotions(hook_store)
        seen: list[str] = []
        hook_store._conn.set_trace_callback(seen.append)
        try:
            hook_store.update(mem.id, fold=fold, emotions={"joy": 3.0})
        finally:
            hook_store._conn.set_trace_callback(None)
        return sum(1 for s in seen if "SELECT" in s.upper() and "FROM MEMORIES" in s.upper())

    assert selects(False) == 1
    assert selects(True) == 1
    assert len(_events(tmp_path)) == 1  # the fold=True update did queue (the count is not of a silent path)


def test_a_combined_update_is_judged_on_the_row_as_it_was(tmp_path: Path, hook_store: MemoryStore) -> None:
    counted = _row_without_emotions(hook_store, "conversation")
    hook_store.update(counted.id, memory_type="dream", emotions={"joy": 3.0})  # was counted: fires
    uncounted = _row_without_emotions(hook_store, "heartbeat")
    hook_store.update(uncounted.id, memory_type="conversation", emotions={"joy": 3.0})  # was uncounted: silent
    assert [e.memory_id for e in _events(tmp_path)] == [counted.id]


def test_a_pending_reach_out_flipped_to_delivered_without_emotions_queues_nothing(
    tmp_path: Path, hook_store: MemoryStore
) -> None:
    pending = _hook_mem("initiate_outbound", metadata={"initiate_state": "pending"})
    hook_store.create(pending, fold=False)
    hook_store.update(pending.id, metadata={"initiate_state": "delivered"})
    assert _events(tmp_path) == []
    control = _row_without_emotions(hook_store)
    hook_store.update(control.id, emotions={"joy": 3.0})
    assert len(_events(tmp_path)) == 1


# --- the hook is after the commit ---------------------------------------------------------------


def test_a_duplicate_id_create_queues_nothing(tmp_path: Path, hook_store: MemoryStore) -> None:
    mem = _hook_mem()
    hook_store.create(mem)
    with pytest.raises(sqlite3.IntegrityError):
        hook_store.create(mem)
    assert len(_events(tmp_path)) == 1


def test_the_hooks_run_after_the_commit_and_after_the_reembed(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def readable_elsewhere(mid: str) -> tuple | None:
        other = sqlite3.connect(tmp_path / "memories.db")
        try:
            return other.execute("SELECT id, embedding FROM memories WHERE id = ?", (mid,)).fetchone()
        finally:
            other.close()

    def committed(self: object, memory: Memory) -> None:
        seen["create_in_transaction"] = hook_store._conn.in_transaction
        seen["create_row"] = readable_elsewhere(memory.id)

    def assigned(self: object, memory: Memory) -> None:
        seen["update_in_transaction"] = hook_store._conn.in_transaction
        seen["update_row"] = readable_elsewhere(memory.id)

    monkeypatch.setattr(EngineCls, "on_memory_committed", committed)
    monkeypatch.setattr(EngineCls, "on_emotions_assigned", assigned)
    mem = _hook_mem(emotions={})
    hook_store.create(mem)
    assert seen["create_in_transaction"] is False and seen["create_row"] is not None
    hook_store.update(mem.id, content="new words", emotions={"joy": 3.0})
    assert seen["update_in_transaction"] is False
    assert seen["update_row"] is not None and seen["update_row"][1] is not None  # re-embedded before the hook


# --- C5b: methods that stay silent ---------------------------------------------------------------


def test_update_emotions_batch_and_set_active_clear_snapshot_fire_no_hook(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _Probe(monkeypatch)
    probe.wrap(engine_module, "engine_for")
    mem = _row_without_emotions(hook_store)
    hook_store.update_emotions_batch([(mem.id, {"joy": 3.0})])
    hook_store.set_active_clear_snapshot(mem.id)
    assert not engine_files.queue_path(tmp_path).exists() and probe.n("engine_for") == 0
    other = _row_without_emotions(hook_store)
    hook_store.update(other.id, emotions={"joy": 3.0})  # the control: update does ask for the engine and queue
    assert probe.n("engine_for") >= 1 and len(_events(tmp_path)) == 1


# --- C6: a hook never raises into the write ------------------------------------------------------


def _one_error(caplog: pytest.LogCaptureFixture, cause: BaseException) -> None:
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info is not None and errors[0].exc_info[1] is cause


def test_a_failure_inside_a_hook_is_logged_once_and_the_row_stays(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    boom = RuntimeError("boom")

    def raiser(*args: object, **kwargs: object) -> object:
        raise boom

    for owner, name in ((engine_module, "resolve_emotions"), (engine_files, "append_event")):
        with monkeypatch.context() as patch, caplog.at_level(logging.ERROR):
            caplog.clear()
            patch.setattr(owner, name, raiser)
            mem = _hook_mem()
            assert hook_store.create(mem) == mem.id
            assert hook_store.get(mem.id, bump=False) is not None
            _one_error(caplog, boom)
    with monkeypatch.context() as patch, caplog.at_level(logging.ERROR):
        caplog.clear()
        patch.setattr("brain.memory.store.replace", raiser)
        empty = _row_without_emotions(hook_store)
        hook_store.update(empty.id, emotions={"joy": 3.0})  # returns
        assert hook_store.get(empty.id, bump=False).emotions == {"joy": 3.0}
        _one_error(caplog, boom)
    with monkeypatch.context() as patch, caplog.at_level(logging.ERROR):
        caplog.clear()
        patch.setattr(engine_files, "append_event", raiser)
        engine = engine_module.engine_for(tmp_path)
        assert engine.on_memory_shown(_hook_mem(), None) is None
        _one_error(caplog, boom)
        caplog.clear()
        assert engine.fold_change({"joy": 1.0}, None, "test") is None
        _one_error(caplog, boom)
        caplog.clear()
        assert engine.on_emotions_assigned(_hook_mem()) is None
        _one_error(caplog, boom)


def test_a_queue_that_cannot_be_written_is_logged_by_the_append_and_the_row_stays(
    tmp_path: Path, hook_store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    engine_files.queue_path(tmp_path).mkdir(parents=True)  # a directory where the file should be
    mem = _hook_mem()
    with caplog.at_level(logging.WARNING):
        assert hook_store.create(mem) == mem.id
    assert hook_store.get(mem.id, bump=False) is not None
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_an_unwritable_state_folder_logs_one_line_without_a_traceback_not_one_per_write(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def denied(persona_dir: Path) -> object:
        raise PermissionError("denied")

    monkeypatch.setattr(engine_files, "state_lock", denied)
    with caplog.at_level(logging.DEBUG):
        mem = _hook_mem()
        assert hook_store.create(mem) == mem.id
        first = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(first) == 1 and first[0].exc_info is None and "denied" in first[0].getMessage()
        caplog.clear()
        assert hook_store.create(_hook_mem()) is not None
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []  # throttled to DEBUG
    assert hook_store.get(mem.id, bump=False) is not None


# --- C7: an in-memory store is a silent no-op ------------------------------------------------------


def test_an_in_memory_store_never_asks_for_an_engine_and_logs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    probe = _Probe(monkeypatch)
    probe.wrap(engine_module, "engine_for")
    store = MemoryStore(":memory:")
    try:
        with caplog.at_level(logging.WARNING):
            mem = _hook_mem(emotions={})
            store.create(mem)
            store.update(mem.id, emotions={"joy": 3.0})
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
        assert probe.n("engine_for") == 0
        assert list(tmp_path.iterdir()) == []
    finally:
        store.close()
    (tmp_path / "p").mkdir()
    file_backed = MemoryStore(tmp_path / "p" / "memories.db")
    try:
        file_backed.create(_hook_mem())  # the control: a file-backed store does ask, once
        assert probe.n("engine_for") == 1
    finally:
        file_backed.close()


# --- C8: locking and data safety ----------------------------------------------------------------


def test_every_hook_appends_while_the_lock_is_held(tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _Probe(monkeypatch)
    probe.wrap(engine_files, "append_event")
    hook_store.create(_hook_mem())  # commit
    empty = _row_without_emotions(hook_store)
    hook_store.update(empty.id, emotions={"joy": 3.0})  # assigned
    engine = engine_module.engine_for(tmp_path)
    engine.on_memory_shown(_hook_mem(), None)  # showing
    engine.fold_change({"joy": 1.0}, None, "test")  # change
    hook_store.create(_uncounted())  # start (no state file)
    kinds = {args[1].kind for args, _held in probe.calls["append_event"]}
    assert kinds == {"commit", "assigned", "showing", "change", "start"}
    assert all(probe.held_at("append_event"))


def test_the_hook_resolves_everything_before_it_takes_the_lock_and_touches_no_database_inside_it(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _Probe(monkeypatch)
    for owner, name in (
        (engine_module, "source_modifier"),
        (engine_files, "peer_trust"),
        (EngineCls, "faded_value"),
        (engine_module, "resolve_emotions"),
        (MemoryStore, "get"),
    ):
        probe.wrap(owner, name)
    hook_store.create(_hook_mem(**_COUNTED_ROWS["peer"]))
    for name in ("source_modifier", "peer_trust", "faded_value", "resolve_emotions"):
        assert probe.n(name) >= 1, name
        assert not any(probe.held_at(name)), name
    assert not any(probe.held_at("get"))  # never read from memories.db while the lock is held
    assert probe.lock_calls == 1


def test_an_existing_queued_event_is_kept_when_another_is_appended(tmp_path: Path, hook_store: MemoryStore) -> None:
    hook_store.create(_hook_mem(emotions={"joy": 1.0}))
    first = engine_files.queue_path(tmp_path).read_bytes().splitlines()[0]
    hook_store.create(_hook_mem(emotions={"joy": 2.0}))
    lines = engine_files.queue_path(tmp_path).read_bytes().splitlines()
    assert lines[0] == first and len(lines) == 2


# --- C9: showing and change ---------------------------------------------------------------------


def test_a_showing_carries_the_faded_value_at_the_given_or_the_entry_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EngineCls, "faded_value", lambda self, memory, emotion: memory.emotions[emotion] / 2)
    engine = EngineCls(tmp_path, now=lambda: NOW)
    mem = _hook_mem(emotions={"joy": 8.0})
    shown_at = NOW - timedelta(hours=1)
    engine.on_memory_shown(mem, shown_at)
    engine.on_memory_shown(mem, None)
    first, second = _events(tmp_path)
    assert (first.kind, first.at, dict(first.emotions)) == ("showing", shown_at, {"joy": 4.0})
    assert second.at == NOW
    engine.on_memory_shown(_hook_mem("heartbeat"), None)  # uncounted
    engine.on_memory_shown(_hook_mem(emotions={}), None)  # nothing to fold
    assert len(_events(tmp_path)) == 2


def test_fold_change_queues_one_change_event_and_writes_no_change_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(EngineCls, "is_live", lambda self: True)
    engine = EngineCls(tmp_path, now=lambda: NOW)
    engine.fold_change({"joy": -2.0, "grief": 1.0}, None, "pass2")
    engine.fold_change({"joy": 1.0}, NOW - timedelta(hours=2), "chat")
    first, second = _events(tmp_path)
    assert (first.kind, first.at, first.source, dict(first.emotions)) == ("change", NOW, "pass2", {"joy": -2.0, "grief": 1.0})
    assert second.at == NOW - timedelta(hours=2) and second.source == "chat"
    assert not engine_files.changes_dir(tmp_path).exists()  # the change-log line is the switch-over's step
    engine.fold_change({}, None, "x")
    engine.fold_change({"joy": math.nan}, None, "x")
    assert len(_events(tmp_path)) == 2


# --- C10 --------------------------------------------------------------------------------------


def test_peer_trust_is_zero_and_does_no_io(tmp_path: Path) -> None:
    assert engine_files.peer_trust(tmp_path, "anyone") == 0.0
    assert engine_files.peer_trust(tmp_path / "no" / "such", "") == 0.0
    assert list(tmp_path.iterdir()) == []


# --- C12(c): the store reaches the stub through the module attribute ----------------------------


def test_the_store_calls_out_through_engine_for(tmp_path: Path, hook_store: MemoryStore, emotion_engine_stub) -> None:
    assert engine_module.engine_for(tmp_path / "anywhere") is emotion_engine_stub
    mem = _hook_mem(emotions={})
    hook_store.create(mem)
    hook_store.update(mem.id, emotions={"joy": 3.0})
    hook_store.update(mem.id, content="again", fold=False)
    names = [call[0] for call in emotion_engine_stub.recorded_calls]
    assert names == ["on_memory_committed", "on_emotions_assigned"]
    assert emotion_engine_stub.recorded_calls[1][1].emotions == {"joy": 3.0}
    assert not engine_files.queue_path(tmp_path).exists()  # the stub queued nothing


# --- the bounded engine lock (Engine 3 delta): every hook passes the owner's wait; a timeout is not a disk fault ----


class _WaitRecorder:
    """A call-through recorder around `file_lock.locked_within`, patched where `engine_files.state_lock` looks it up.

    It records each call's path and wait, and whether its context was open when `append_event` ran.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[Path, float]] = []
        self.open = False
        self.append_open: list[bool] = []
        real = file_lock.locked_within
        real_append = engine_files.append_event
        rec = self

        @contextlib.contextmanager
        def locked_within(path: Path, seconds: float, **kwargs: object) -> Iterator[None]:
            rec.calls.append((path, seconds))
            with real(path, seconds, **kwargs):
                rec.open = True
                try:
                    yield
                finally:
                    rec.open = False

        def append(*args: object, **kwargs: object) -> object:
            rec.append_open.append(rec.open)
            return real_append(*args, **kwargs)

        monkeypatch.setattr(file_lock, "locked_within", locked_within)
        monkeypatch.setattr(engine_files, "append_event", append)


@pytest.mark.parametrize("kind", ["commit", "assigned", "showing", "change", "start"])
def test_every_hook_takes_the_engine_lock_once_with_the_owners_wait_and_appends_inside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 7.25)
    recorder = _WaitRecorder(monkeypatch)
    engine = engine_module.engine_for(tmp_path)
    fire = {
        "commit": lambda: engine.on_memory_committed(_hook_mem()),
        "assigned": lambda: engine.on_emotions_assigned(_hook_mem()),
        "showing": lambda: engine.on_memory_shown(_hook_mem(), None),
        "change": lambda: engine.fold_change({"joy": 1.0}, None, "test"),
        "start": lambda: engine.on_memory_committed(_uncounted()),
    }
    fire[kind]()
    assert recorder.calls == [(engine_files.state_path(tmp_path), 7.25)]
    assert recorder.append_open == [True]
    assert [e.kind for e in _events(tmp_path)] == [kind]


def test_a_held_engine_lock_does_not_hang_a_write_and_the_row_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, held_lock
) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.2)
    db = tmp_path / "memories.db"
    mem = _hook_mem()
    outcome: dict[str, object] = {}

    def worker() -> None:
        store = MemoryStore(db)  # opened in this thread: the connection is bound to the thread that makes it
        try:
            outcome["id"] = store.create(mem)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            store.close()

    thread = threading.Thread(target=worker, name="create-under-held-lock", daemon=True)
    started = False
    with caplog.at_level(logging.DEBUG, logger=HOOK_LOGGER):
        try:
            with held_lock(engine_files.state_path(tmp_path)):
                thread.start()
                started = True
                thread.join(5)
                finished = not thread.is_alive()
        finally:
            if started:
                thread.join(5)  # a regression that hangs is released by the holder's exit, never left running
    assert finished, "create hung behind a held engine lock"
    assert outcome == {"id": mem.id}
    assert _lines(tmp_path) == []  # the event was not recorded
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "gave up on the engine lock" in errors[0] and "on_memory_committed" in errors[0] and "not recorded" in errors[0]
    reader = MemoryStore(db)
    try:
        assert reader.get(mem.id, bump=False) is not None  # the row is committed
        later = _hook_mem()
        reader.create(later)  # the positive control: with the holder gone the next create appends its line
    finally:
        reader.close()
    assert [e.memory_id for e in _events(tmp_path)] == [later.id]


def test_a_lock_wait_that_runs_out_is_logged_as_that_not_as_a_disk_fault(
    tmp_path: Path, hook_store: MemoryStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def timed_out(persona_dir: Path) -> object:
        raise file_lock.LockTimeoutError("could not lock the state file within 5 s")

    def disk_fault(persona_dir: Path) -> object:
        raise OSError("input/output error")

    monkeypatch.setattr(engine_files, "state_lock", timed_out)
    with caplog.at_level(logging.DEBUG, logger=HOOK_LOGGER):
        mem = _hook_mem()
        assert hook_store.create(mem) == mem.id
        first = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(first) == 1 and first[0].exc_info is None
        wording = first[0].getMessage()
        assert "gave up on the engine lock" in wording and "on_memory_committed" in wording and "not recorded" in wording
        assert "could not queue" not in wording
        caplog.clear()
        hook_store.create(_hook_mem())
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []  # throttled to DEBUG
        assert hook_store.get(mem.id, bump=False) is not None
        assert _lines(tmp_path) == []
        # the generic OSError line keeps its own wording and its own throttle: the timeout's did not swallow it
        monkeypatch.setattr(engine_files, "state_lock", disk_fault)
        caplog.clear()
        hook_store.create(_hook_mem())
        other = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(other) == 1 and other[0].exc_info is None
        assert "could not queue its event" in other[0].getMessage()
        assert "gave up on the engine lock" not in other[0].getMessage()
