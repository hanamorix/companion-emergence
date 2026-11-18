"""Hana's 11-18 review of #633, point 5 (owner S112): the rebuild notice, `engine_files.settle_rebuild_notice` and its reader.

`rebuild_notice.json` in the persona folder says why a switch-over did not complete, for as long as the persona is NOT switched over.
These tests cover the one writer's five steps in order (the healthy path's one stat, a switched persona's delete, the no-reason
return, the `since` rule, the write), the one-line reason, the reader's refusals, and that no failure of the writer ever raises.
Personas are tmp directories; nothing here touches a real persona.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files
from brain.utils import file_ops
from tests.unit.brain.emotion._engine_support import (
    NOW,
    make_live,
    make_persona,
    ruin_the_state,
    save,
    simple_state,
)


def _record(persona: Path) -> dict:
    return json.loads(engine_files.rebuild_notice_path(persona).read_text(encoding="utf-8"))


def _put(persona: Path, at: str, reason: str = "earlier failure") -> None:
    engine_files.rebuild_notice_path(persona).write_text(json.dumps({"at": at, "reason": reason}), encoding="utf-8")


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    return make_persona(tmp_path / "nell")


def test_the_path_is_in_the_persona_folder_by_the_constant(persona: Path) -> None:
    assert engine_files.rebuild_notice_path(persona) == persona / dev_constants.REBUILD_NOTICE_FILENAME == persona / "rebuild_notice.json"


def test_a_reason_on_an_unswitched_persona_is_written_and_read_back(persona: Path) -> None:
    engine_files.settle_rebuild_notice(persona, "the first rebuild failed (RebuildCheckError: forced)")
    written = _record(persona)
    assert set(written) == {"at", "reason"} and written["reason"] == "the first rebuild failed (RebuildCheckError: forced)"
    assert engine_files.read_rebuild_notice(persona) == written
    assert datetime.fromisoformat(written["at"].replace("Z", "+00:00")) > datetime.now(UTC) - timedelta(minutes=1)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_the_notice_is_written_with_mode_0600(persona: Path) -> None:
    engine_files.settle_rebuild_notice(persona, "why")
    assert engine_files.rebuild_notice_path(persona).stat().st_mode & 0o777 == 0o600


def test_a_multi_line_reason_is_made_one_line(persona: Path) -> None:
    engine_files.settle_rebuild_notice(persona, "first line\n  second\tline\r\n\nthird")
    assert _record(persona)["reason"] == "first line second line third"


def test_a_later_reason_replaces_an_earlier_one(persona: Path) -> None:
    engine_files.settle_rebuild_notice(persona, "one")
    engine_files.settle_rebuild_notice(persona, "two")
    assert _record(persona)["reason"] == "two"


def test_a_switched_persona_gets_no_notice(persona: Path) -> None:
    make_live(persona, NOW)
    engine_files.settle_rebuild_notice(persona, "a roll-forward failed")
    assert not engine_files.rebuild_notice_path(persona).exists()


def test_a_switched_persona_has_an_old_notice_deleted_by_a_reason_or_by_none(persona: Path) -> None:
    make_live(persona, NOW)
    for reason in ("a roll-forward failed", None):
        _put(persona, "2020-01-01T00:00:00Z")
        engine_files.settle_rebuild_notice(persona, reason)
        assert not engine_files.rebuild_notice_path(persona).exists()


@pytest.mark.no_base_failure(reason="a pin of the rule's other half: on base there is no notice to keep, so the file is trivially still there")
def test_no_reason_on_an_unswitched_persona_keeps_the_notice(persona: Path) -> None:
    _put(persona, "2020-01-01T00:00:00Z")
    before = engine_files.rebuild_notice_path(persona).read_bytes()
    engine_files.settle_rebuild_notice(persona, None)
    assert engine_files.rebuild_notice_path(persona).read_bytes() == before


def test_an_unreadable_state_is_not_switched_so_the_notice_is_written(persona: Path) -> None:
    save(persona, simple_state(saved_at=NOW))
    ruin_the_state(persona)
    engine_files.settle_rebuild_notice(persona, "the state cannot be read")
    assert _record(persona)["reason"] == "the state cannot be read"


def test_the_healthy_path_costs_one_stat_and_reads_no_state(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stats: list[str] = []
    real = file_ops.stat_path
    monkeypatch.setattr(engine_files.file_ops, "stat_path", lambda p: (stats.append(Path(p).name), real(p))[1])
    monkeypatch.setattr(engine_files, "read_state", lambda *a, **k: pytest.fail("the healthy path read the state"))
    engine_files.settle_rebuild_notice(persona, None)
    assert stats == [dev_constants.REBUILD_NOTICE_FILENAME]


def test_a_notice_newer_than_since_stands_and_an_older_one_is_replaced(persona: Path) -> None:
    since = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
    _put(persona, "2026-11-18T12:00:05Z", "the child's own, exact reason")
    engine_files.settle_rebuild_notice(persona, "the update of your memories stopped (exit code 1)", since=since)
    assert _record(persona)["reason"] == "the child's own, exact reason"
    _put(persona, "2026-11-18T11:59:59Z", "an earlier attempt's")
    engine_files.settle_rebuild_notice(persona, "the update of your memories stopped (exit code 1)", since=since)
    assert _record(persona)["reason"] == "the update of your memories stopped (exit code 1)"


def test_a_notice_with_no_usable_time_or_that_cannot_be_read_is_replaced_under_since(persona: Path) -> None:
    since = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
    _put(persona, "not a time")
    engine_files.settle_rebuild_notice(persona, "new", since=since)
    assert _record(persona)["reason"] == "new"
    engine_files.rebuild_notice_path(persona).write_bytes(b"\x00 not json")
    engine_files.settle_rebuild_notice(persona, "newer", since=since)
    assert _record(persona)["reason"] == "newer"


def test_the_writer_never_creates_a_persona_folder(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    engine_files.settle_rebuild_notice(missing, "why")
    assert not missing.exists()


@pytest.mark.parametrize("content", [None, b"\x00 not json", b"[1, 2]", b'{"at": "x"}', b'{"reason": ""}', b'{"reason": 5}'])
def test_the_reader_answers_none_for_a_missing_unreadable_or_reasonless_record(persona: Path, content: bytes | None) -> None:
    if content is not None:
        engine_files.rebuild_notice_path(persona).write_bytes(content)
    assert engine_files.read_rebuild_notice(persona) is None


def test_the_reader_answers_none_when_the_read_raises(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _put(persona, "2020-01-01T00:00:00Z")

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("read blew up")

    monkeypatch.setattr(engine_files, "read_json", boom)
    assert engine_files.read_rebuild_notice(persona) is None


@pytest.mark.parametrize("failing", ["write", "delete", "state", "stat"])
def test_a_failure_of_the_writer_never_raises_and_is_logged_without_a_traceback(
    persona: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    warned: list[dict] = []
    monkeypatch.setattr(engine_files, "warn_throttled", lambda lg, key, msg, *a, **kw: warned.append(kw))

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    if failing == "write":
        monkeypatch.setattr(engine_files, "write_json", boom)
        engine_files.settle_rebuild_notice(persona, "why")
    elif failing == "delete":
        make_live(persona, NOW)
        _put(persona, "2020-01-01T00:00:00Z")
        monkeypatch.setattr(engine_files.file_ops, "delete_file", boom)
        engine_files.settle_rebuild_notice(persona, None)
    elif failing == "state":
        monkeypatch.setattr(engine_files, "read_state", boom)
        engine_files.settle_rebuild_notice(persona, "why")
    else:
        monkeypatch.setattr(engine_files.file_ops, "stat_path", boom)
        engine_files.settle_rebuild_notice(persona, "why")
    assert warned and all(kw == {"exc_info": False} for kw in warned)


def test_a_stat_that_raises_is_read_as_a_notice_that_exists_so_a_switched_persona_is_still_cleared(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_live(persona, NOW)
    _put(persona, "2020-01-01T00:00:00Z")
    real = file_ops.stat_path
    calls: list[int] = []

    def flaky(path: str | Path):
        if not calls:
            calls.append(1)
            raise OSError("transient")
        return real(path)

    monkeypatch.setattr(engine_files.file_ops, "stat_path", flaky)
    engine_files.settle_rebuild_notice(persona, None)
    assert not engine_files.rebuild_notice_path(persona).exists()


def test_the_record_shape_is_two_keys() -> None:
    assert engine_files.rebuild_notice_record(at="2026-11-18T12:00:00Z", reason="why") == {"at": "2026-11-18T12:00:00Z", "reason": "why"}
