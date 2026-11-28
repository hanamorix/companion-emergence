"""The stats line at the end of a finished conversation file (#650 spec section 5, acceptance A15)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.ingest import session_stats
from brain.ingest.buffer import append_archive, last_user_turn_ts, read_archive, turn_identity
from brain.ingest.session_stats import STATS_ROW_TYPE, append_session_stats, build_session_stats

_SID = "sess_stats"


def _local(day: int, hour: int, minute: int, month: int = 10) -> datetime:
    """A wall-clock time in the machine's own zone, as an aware datetime (so the expected local text is the same on every zone)."""
    return datetime(2026, month, day, hour, minute).astimezone()


def _row(speaker: str, when: datetime, text: str | None = None) -> dict:
    return {
        "session_id": _SID,
        "speaker": speaker,
        "text": text or f"{speaker} at {when.isoformat()}",
        "ts": when.astimezone(UTC).isoformat(timespec="seconds"),
    }


def _users(*times: datetime) -> list[dict]:
    return [_row("user", t) for t in times]


def test_a_sitting_ends_at_a_gap_of_thirty_minutes_and_the_average_counts_a_one_message_sitting_as_zero() -> None:
    t0 = _local(1, 9, 0)
    rows = [
        _row("user", t0),
        _row("assistant", t0 + timedelta(minutes=29)),  # same sitting: 29 minutes
        _row("user", t0 + timedelta(minutes=59)),  # a gap of exactly 30 minutes starts a second sitting
        _row("assistant", t0 + timedelta(minutes=89)),  # 30 again: a third, of one message (0 minutes)
    ]
    stats = build_session_stats(rows)
    assert stats is not None
    # sittings: [0, 29] = 29 min, [59] = 0, [89] = 0 -> mean 9.666... -> 9.7
    assert stats["avg_sitting_minutes"] == 9.7


def test_the_row_is_tagged_first_and_has_no_speaker_ts_or_count_field() -> None:
    stats = build_session_stats(_users(_local(1, 9, 0), _local(1, 9, 5)))
    assert stats is not None
    assert list(stats)[0] == "type" and stats["type"] == STATS_ROW_TYPE == "session_stats"
    assert set(stats) == {"type", "avg_sitting_minutes", "busiest"}


def test_busiest_lists_the_users_top_stretches_per_local_day_most_messages_first() -> None:
    day1 = [
        _local(1, 8, 0), _local(1, 8, 10),  # a stretch of 2: 08:00-08:10
        _local(1, 12, 0), _local(1, 12, 10), _local(1, 12, 20), _local(1, 12, 25),  # 4: 12:00-12:25
        _local(1, 15, 0),  # 1: 15:00-15:00
        _local(1, 18, 0), _local(1, 18, 5), _local(1, 18, 9),  # 3: 18:00-18:09
        _local(1, 21, 22), _local(1, 21, 40), _local(1, 22, 5), _local(1, 22, 30), _local(1, 22, 59),  # 5: 21:22-22:59
    ]
    day2 = [_local(2, 10, 0), _local(2, 10, 3)]
    rows = _users(*day1, *day2)
    # assistant rows split nothing and are not part of any stretch
    rows += [_row("assistant", _local(1, 12, 12)), _row("assistant", _local(1, 12, 40))]
    stats = build_session_stats(rows)
    assert stats is not None
    assert dev_constants.BODY_STATS_BUSIEST_COUNT == 3
    assert stats["busiest"] == {
        _local(1, 0, 0).date().isoformat(): ["21:22-22:59", "12:00-12:25", "18:00-18:09"],
        _local(2, 0, 0).date().isoformat(): ["10:00-10:03"],
    }


def test_a_stretch_across_local_midnight_is_listed_under_the_day_it_started() -> None:
    rows = _users(_local(1, 23, 40), _local(1, 23, 55), _local(2, 0, 10), _local(2, 0, 20))
    stats = build_session_stats(rows)
    assert stats is not None
    assert stats["busiest"] == {_local(1, 0, 0).date().isoformat(): ["23:40-00:20"]}


def test_each_turn_counts_once_and_summary_stats_and_undated_rows_are_not_turns() -> None:
    t0 = _local(1, 9, 0)
    first, second = _row("user", t0), _row("assistant", t0 + timedelta(minutes=10))
    rows = [
        {"speaker": "summary", "text": "[folded]", "ts": (t0 - timedelta(days=9)).astimezone(UTC).isoformat()},
        {"type": STATS_ROW_TYPE, "avg_sitting_minutes": 999.0, "busiest": {}},
        {"speaker": "user", "text": "no time"},
        first,
        second,
        dict(first),  # the same turn in the archive and in the buffer (the carried tail)
    ]
    assert turn_identity(rows[3]) == turn_identity(rows[5])
    stats = build_session_stats(rows)
    assert stats is not None
    assert stats["avg_sitting_minutes"] == 10.0
    assert build_session_stats(rows[:3]) is None, "no dated turn, no row"
    assert build_session_stats([]) is None


def test_append_writes_one_row_last_from_the_archive_and_the_live_rows(tmp_path: Path) -> None:
    t0 = _local(1, 9, 0)
    archived = [_row("user", t0), _row("assistant", t0 + timedelta(minutes=2))]
    append_archive(tmp_path, _SID, archived)
    live = [archived[1], _row("user", t0 + timedelta(minutes=4))]  # one row is in both: counted once

    assert append_session_stats(tmp_path, _SID, live) is True

    archive = read_archive(tmp_path, _SID)
    assert archive[:2] == archived and len(archive) == 3
    assert archive[-1]["type"] == STATS_ROW_TYPE
    assert archive[-1]["avg_sitting_minutes"] == 4.0


def test_an_empty_buffer_gets_no_row_and_neither_does_a_file_with_no_dated_turn(tmp_path: Path) -> None:
    append_archive(tmp_path, _SID, [_row("user", _local(1, 9, 0))])
    assert append_session_stats(tmp_path, _SID, []) is False
    assert append_session_stats(tmp_path, "sess_other", [{"speaker": "summary", "text": "x"}]) is False
    assert [r.get("type") for r in read_archive(tmp_path, _SID)] == [None]
    assert read_archive(tmp_path, "sess_other") == []


def test_a_file_that_already_ends_with_its_stats_row_gets_no_second_one(tmp_path: Path) -> None:
    live = _users(_local(1, 9, 0), _local(1, 9, 5))
    assert append_session_stats(tmp_path, _SID, live) is True
    assert append_session_stats(tmp_path, _SID, live) is False  # the retry of a close whose buffer removal failed
    assert [r.get("type") for r in read_archive(tmp_path, _SID)].count(STATS_ROW_TYPE) == 1
    # a turn archived after the row makes it no longer last: the finished file gets a new row after that turn
    append_archive(tmp_path, _SID, _users(_local(1, 11, 0)))
    assert append_session_stats(tmp_path, _SID, live) is True
    assert [r.get("type") for r in read_archive(tmp_path, _SID)].count(STATS_ROW_TYPE) == 2


def test_a_write_failure_is_logged_with_the_session_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(session_stats, "append_archive", broken)
    with caplog.at_level(logging.ERROR, logger="brain.ingest.session_stats"):
        assert append_session_stats(tmp_path, _SID, _users(_local(1, 9, 0))) is False
    assert any(_SID in rec.getMessage() for rec in caplog.records)


def test_readers_that_walk_archive_rows_skip_the_stats_row(tmp_path: Path) -> None:
    """A15: with a stats row last in an archive, every reader of archive rows returns what it returned without it."""
    from brain.body.energy_inputs import _window_rows
    from brain.ingest.extraction_backfill import _newest_user_ts, _rows_of
    from brain.initiate.user_pattern import _archived_user_turn_times

    now = datetime.now(UTC)
    rows = [
        _row("user", now - timedelta(days=2)),
        _row("assistant", now - timedelta(days=2) + timedelta(minutes=1)),
        _row("user", now - timedelta(days=1)),
    ]
    append_archive(tmp_path, _SID, rows)
    cutoff = now - timedelta(days=30)
    before = (
        last_user_turn_ts(tmp_path),
        _rows_of(tmp_path, _SID, now),
        _newest_user_ts(tmp_path),
        sorted(_window_rows(tmp_path, cutoff)),
        _archived_user_turn_times(tmp_path, cutoff, set()),
    )

    assert append_session_stats(tmp_path, _SID, rows) is True
    assert read_archive(tmp_path, _SID)[-1]["type"] == STATS_ROW_TYPE

    after = (
        last_user_turn_ts(tmp_path),
        _rows_of(tmp_path, _SID, now),
        _newest_user_ts(tmp_path),
        sorted(_window_rows(tmp_path, cutoff)),
        _archived_user_turn_times(tmp_path, cutoff, set()),
    )
    assert before == after
    assert before[0] is not None and len(before[1]) == 3 and len(before[3]) == 3 and len(before[4]) == 2
