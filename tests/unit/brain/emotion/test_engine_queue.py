"""The emotion engine's queue and change log (Engine 2, #577).

The queue (`state/emotion_pending.jsonl`) is an append-only file of events waiting to be folded, one JSON event per line;
the change log (`state/emotion_changes/YYYY-MM-DD.jsonl`) is a crash-recovery record of chat-derived changes. These are
the file-level cases: the per-event check, the queue file's append, read by position and emptying (a bad line is dropped on
its own), the accessor each file operation goes through, the change log's append, list, prune and tolerant reader. Cases that
need a hook or a drain (fold order, early-out, wiring) belong to the slices that build them and extend this file.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import logging
import math
import time
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files, engine_utils
from brain.emotion.engine_utils import EngineState, Event, day_of, fold_event, source_modifier
from brain.memory.store import Memory
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
    """The check and the fold agree: nothing that passes can fail every drain for ever (E6.10), not even at the save."""
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
    """B04.2: an unclean machine stop can leave a zeroed record; it is dropped with its own line, the rest is read."""
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
    """B04.2: no fsync per append; the caller holds the lock, so the append takes none of its own."""
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
    """P12b: a position outside its range is held to the range. `tmp_path` is unique, so the throttle cannot hide the line."""
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


def test_clear_queue_no_longer_exists() -> None:
    assert not hasattr(engine_files, "clear_queue")


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


# --- every file operation goes through its accessor (P18, S1, S37) -------------------------------

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


def test_a_change_written_then_read_round_trips(tmp_path: Path) -> None:
    event = _change(NOW, joy=-2.5, grief=0.25)
    engine_files.append_change(tmp_path, event)
    assert engine_files.read_changes(tmp_path, NOW - timedelta(days=1)) == [event]


def test_the_first_append_creates_only_the_folder_and_the_day_file(tmp_path: Path) -> None:
    engine_files.append_change(tmp_path, _change(NOW))
    assert {p.name for p in (tmp_path / "state").rglob("*")} == {"emotion_changes", f"{day_of(NOW)}.jsonl"}
