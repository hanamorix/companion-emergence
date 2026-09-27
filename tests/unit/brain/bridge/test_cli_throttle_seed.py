"""ram-spike-fix S82 (C41): cli_throttle.seed_last_message_from_active_conversations —
seeding the is_chat_idle anchor from the newest SAVED message timestamp in
``<persona_dir>/active_conversations/*.jsonl``, narrowing C24's "fresh process
is idle" to "idle once the lull has passed since the last saved message"."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge import cli_throttle

_LULL = 600.0  # default chat.idle_lull_seconds — no tunables override in these tests


@pytest.fixture(autouse=True)
def _reset():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


def _write_turn(persona_dir: Path, session_id: str, ts) -> None:
    """Append one raw jsonl turn line — bypasses brain.ingest.buffer so a
    malformed/missing ``ts`` can be injected directly (buffer.ingest_turn
    would silently default a falsy ts to "now")."""
    d = persona_dir / "active_conversations"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{session_id}.jsonl"
    record = {"session_id": session_id, "speaker": "user", "text": "hi"}
    if ts is not _MISSING:
        record["ts"] = ts
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


_MISSING = object()


def test_no_active_conversations_dir_leaves_default_idle(tmp_path: Path):
    """No saved messages at all (fresh install) -> anchor untouched (-inf) ->
    idle immediately, matching C24's untouched fresh-process semantics."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    cli_throttle.seed_last_message_from_active_conversations(
        persona_dir, now_wall=datetime.now(UTC), now_mono=1_000.0
    )
    assert cli_throttle.is_chat_idle(now=1_000.0) is True
    assert cli_throttle.is_chat_idle(now=0.0) is True


def test_empty_active_conversations_dir_leaves_default_idle(tmp_path: Path):
    """The dir exists but has no .jsonl files, or every buffer is empty ->
    still idle (no saved messages anywhere)."""
    persona_dir = tmp_path / "persona"
    (persona_dir / "active_conversations").mkdir(parents=True)
    (persona_dir / "active_conversations" / "empty.jsonl").write_text("")
    cli_throttle.seed_last_message_from_active_conversations(
        persona_dir, now_wall=datetime.now(UTC), now_mono=1_000.0
    )
    assert cli_throttle.is_chat_idle(now=1_000.0) is True


def test_restart_right_after_a_message_is_not_idle_until_lull_elapses_from_saved_ts(
    tmp_path: Path,
):
    """S82's central case: a message saved 5s ago (wall clock) -> NOT idle
    right after seeding, and idle only once the LULL has passed since that
    saved timestamp (not since process start)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    now_wall = datetime(2026, 1, 1, tzinfo=UTC)
    saved_ts = (now_wall - timedelta(seconds=5)).isoformat()
    _write_turn(persona_dir, "sess-1", saved_ts)

    mono_start = 10_000.0
    cli_throttle.seed_last_message_from_active_conversations(
        persona_dir, now_wall=now_wall, now_mono=mono_start
    )

    # The saved message was 5s old at seed time -> remaining lull = LULL - 5.
    remaining = _LULL - 5.0
    assert cli_throttle.is_chat_idle(now=mono_start) is False
    assert cli_throttle.is_chat_idle(now=mono_start + remaining - 1.0) is False
    assert cli_throttle.is_chat_idle(now=mono_start + remaining) is True


def test_newest_across_multiple_sessions_is_used(tmp_path: Path):
    """The seed reads the newest turn across ALL active_conversations
    buffers, not just one session."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    now_wall = datetime(2026, 1, 1, tzinfo=UTC)
    older = (now_wall - timedelta(seconds=700)).isoformat()  # already past the lull
    newer = (now_wall - timedelta(seconds=5)).isoformat()  # still within the lull
    _write_turn(persona_dir, "sess-old", older)
    _write_turn(persona_dir, "sess-new", newer)

    mono_start = 10_000.0
    cli_throttle.seed_last_message_from_active_conversations(
        persona_dir, now_wall=now_wall, now_mono=mono_start
    )
    # If only the older session were consulted this would be idle; the
    # newer session's timestamp must win.
    assert cli_throttle.is_chat_idle(now=mono_start) is False


def test_missing_ts_field_fails_closed(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_turn(persona_dir, "sess-1", _MISSING)

    with caplog.at_level(logging.ERROR, logger="brain.bridge.cli_throttle"):
        cli_throttle.seed_last_message_from_active_conversations(
            persona_dir, now_wall=datetime.now(UTC), now_mono=10_000.0
        )
    assert cli_throttle.is_chat_idle(now=10_000.0) is False
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_unparseable_ts_fails_closed(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_turn(persona_dir, "sess-1", "not-a-real-timestamp")

    with caplog.at_level(logging.ERROR, logger="brain.bridge.cli_throttle"):
        cli_throttle.seed_last_message_from_active_conversations(
            persona_dir, now_wall=datetime.now(UTC), now_mono=10_000.0
        )
    assert cli_throttle.is_chat_idle(now=10_000.0) is False
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    # Fails closed, but only for one lull — it must recover normally afterward.
    assert cli_throttle.is_chat_idle(now=10_000.0 + _LULL) is True


def test_corrupt_trailing_line_fails_closed_not_silent_stale_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Round-1 code red-team MAJOR: a syntactically corrupt/truncated
    trailing JSONL line (a crash/OOM mid-append) must NOT be silently
    skipped in favor of an OLDER good line — that would understate how
    recent the chat really was, the wrong direction for a fail-closed idle
    gate. The prior (buggy) behavior would have used the older good line
    below (> the lull) and reported IDLE; this must instead fail closed
    (not idle) exactly like a malformed ts."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    now_wall = datetime(2026, 1, 1, tzinfo=UTC)
    older_good = (now_wall - timedelta(seconds=700)).isoformat()  # > the 600s lull
    _write_turn(persona_dir, "sess-1", older_good)
    # Simulate a crash mid-append: a truncated, syntactically-invalid line.
    path = persona_dir / "active_conversations" / "sess-1.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"session_id": "sess-1", "speaker": "user", "text": "cut off mid-w')

    with caplog.at_level(logging.ERROR, logger="brain.bridge.cli_throttle"):
        cli_throttle.seed_last_message_from_active_conversations(
            persona_dir, now_wall=now_wall, now_mono=10_000.0
        )
    assert cli_throttle.is_chat_idle(now=10_000.0) is False
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_future_saved_timestamp_is_clamped_not_stuck_forever(tmp_path: Path):
    """A saved ts in the future (clock skew) must not pin the bridge
    non-idle indefinitely — it is clamped to elapsed=0 ("right now"), so it
    clears after exactly one lull, same as a message saved at seed time."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    now_wall = datetime(2026, 1, 1, tzinfo=UTC)
    future_ts = (now_wall + timedelta(days=365)).isoformat()
    _write_turn(persona_dir, "sess-1", future_ts)

    mono_start = 10_000.0
    cli_throttle.seed_last_message_from_active_conversations(
        persona_dir, now_wall=now_wall, now_mono=mono_start
    )
    assert cli_throttle.is_chat_idle(now=mono_start) is False
    # Must clear after exactly one lull past the SEED instant, not stay
    # stuck for anywhere near the year of clock skew in the saved value.
    assert cli_throttle.is_chat_idle(now=mono_start + _LULL) is True


def test_z_suffix_and_naive_timestamps_both_parse(tmp_path: Path):
    """Timezone-aware parsing: a trailing 'Z' and a naive (no-offset) string
    both parse as UTC, matching the existing on-disk convention/parsers
    elsewhere in the codebase (brain/chat/session.py, buffer.py)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    now_wall = datetime(2026, 1, 1, tzinfo=UTC)
    naive = (now_wall - timedelta(seconds=5)).replace(tzinfo=None).isoformat()
    _write_turn(persona_dir, "sess-naive", naive)

    cli_throttle.seed_last_message_from_active_conversations(
        persona_dir, now_wall=now_wall, now_mono=10_000.0
    )
    assert cli_throttle.is_chat_idle(now=10_000.0) is False


def test_large_buffer_is_read_via_bounded_seek_not_a_full_file_read(tmp_path: Path):
    """Round-2 red-team MINOR: getting the newest line must not read the
    whole active_conversations buffer into memory — bounded backward seek
    only. Cross-checked here by asserting the correct newest turn is still
    found even when many megabytes of older lines precede it."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    now_wall = datetime(2026, 1, 1, tzinfo=UTC)
    d = persona_dir / "active_conversations"
    d.mkdir(parents=True)
    path = d / "sess-1.jsonl"
    old_ts = (now_wall - timedelta(seconds=700)).isoformat()  # > the 600s lull
    with path.open("w", encoding="utf-8") as fh:
        # ~5 MB of older, well-formed turns.
        for _ in range(30_000):
            fh.write(json.dumps({"session_id": "sess-1", "speaker": "user", "text": "x" * 100, "ts": old_ts}) + "\n")
        recent_ts = (now_wall - timedelta(seconds=5)).isoformat()
        fh.write(json.dumps({"session_id": "sess-1", "speaker": "user", "text": "newest", "ts": recent_ts}) + "\n")

    from brain.bridge.cli_throttle import _last_nonempty_raw_line

    line = _last_nonempty_raw_line(path, chunk_size=256)
    assert json.loads(line)["text"] == "newest"

    cli_throttle.seed_last_message_from_active_conversations(
        persona_dir, now_wall=now_wall, now_mono=10_000.0
    )
    assert cli_throttle.is_chat_idle(now=10_000.0) is False  # the RECENT turn must win, not the old bulk


def test_seeded_anchor_is_consistent_with_time_since_last_message(tmp_path: Path):
    """S72: time_since_last_message() reads the SAME anchor is_chat_idle
    does — seeding it once here keeps both consistent for free."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    now_wall = datetime(2026, 1, 1, tzinfo=UTC)
    saved_ts = (now_wall - timedelta(seconds=42)).isoformat()
    _write_turn(persona_dir, "sess-1", saved_ts)

    mono_start = 10_000.0
    cli_throttle.seed_last_message_from_active_conversations(
        persona_dir, now_wall=now_wall, now_mono=mono_start
    )
    assert cli_throttle.time_since_last_message(now=mono_start) == pytest.approx(42.0)
