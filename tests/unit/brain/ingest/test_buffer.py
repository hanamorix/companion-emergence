"""Tests for brain.ingest.buffer — BUFFER + CLOSE stage."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.ingest.buffer import (
    delete_backoff,
    delete_cursor,
    delete_session_buffer,
    ingest_turn,
    list_active_sessions,
    read_backoff,
    read_cursor,
    read_session,
    read_session_after,
    session_silence_minutes,
    write_backoff,
    write_cursor,
)


def test_ingest_turn_appends_to_session_file(tmp_path: Path) -> None:
    """ingest_turn writes a JSONL record to active_conversations/<session_id>.jsonl."""
    session_id = ingest_turn(
        tmp_path, {"session_id": "sess_abc", "speaker": "Hana", "text": "hello"}
    )
    assert session_id == "sess_abc"
    buf_file = tmp_path / "active_conversations" / "sess_abc.jsonl"
    assert buf_file.exists()
    line = buf_file.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["session_id"] == "sess_abc"
    assert record["speaker"] == "Hana"
    assert record["text"] == "hello"
    assert "ts" in record


def test_ingest_turn_auto_generates_session_id(tmp_path: Path) -> None:
    """When session_id is absent, ingest_turn generates one and returns it."""
    session_id = ingest_turn(tmp_path, {"speaker": "Nell", "text": "hi"})
    assert session_id.startswith("sess_")
    buf_file = tmp_path / "active_conversations" / f"{session_id}.jsonl"
    assert buf_file.exists()


def test_ingest_turn_multiple_turns_append(tmp_path: Path) -> None:
    """Multiple ingest_turn calls to the same session append multiple lines."""
    for text in ("first", "second", "third"):
        ingest_turn(tmp_path, {"session_id": "sess_multi", "speaker": "user", "text": text})
    buf_file = tmp_path / "active_conversations" / "sess_multi.jsonl"
    lines = [ln for ln in buf_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3
    texts = [json.loads(ln)["text"] for ln in lines]
    assert texts == ["first", "second", "third"]


def test_list_active_sessions_returns_existing_buffer_ids(tmp_path: Path) -> None:
    """list_active_sessions returns only session_ids that have buffer files."""
    ingest_turn(tmp_path, {"session_id": "sess_a", "speaker": "Hana", "text": "a"})
    ingest_turn(tmp_path, {"session_id": "sess_b", "speaker": "Nell", "text": "b"})
    sessions = list_active_sessions(tmp_path)
    assert set(sessions) == {"sess_a", "sess_b"}


def test_list_active_sessions_empty_when_no_dir(tmp_path: Path) -> None:
    """list_active_sessions returns [] when the active_conversations dir doesn't exist."""
    result = list_active_sessions(tmp_path / "nonexistent")
    assert result == []


def test_read_session_reads_turns_and_skips_malformed(tmp_path: Path) -> None:
    """read_session parses valid lines and skips corrupt ones."""
    buf_dir = tmp_path / "active_conversations"
    buf_dir.mkdir(parents=True)
    buf_file = buf_dir / "sess_test.jsonl"
    buf_file.write_text(
        '{"session_id": "sess_test", "speaker": "A", "text": "ok", "ts": "2026-04-25T10:00:00+00:00"}\n'
        "NOT_VALID_JSON\n"
        '{"session_id": "sess_test", "speaker": "B", "text": "also ok", "ts": "2026-04-25T10:01:00+00:00"}\n',
        encoding="utf-8",
    )
    turns = read_session(tmp_path, "sess_test")
    assert len(turns) == 2
    assert turns[0]["speaker"] == "A"
    assert turns[1]["speaker"] == "B"


def test_session_silence_minutes_computes_from_last_turn(tmp_path: Path) -> None:
    """session_silence_minutes returns minutes since the last turn's ts."""
    past = (datetime.now(UTC) - timedelta(minutes=7)).isoformat(timespec="seconds")
    turns = [
        {"ts": (datetime.now(UTC) - timedelta(minutes=20)).isoformat(timespec="seconds")},
        {"ts": past},
    ]
    minutes = session_silence_minutes(turns)
    # Should be approximately 7 minutes — allow 1-minute tolerance for test timing.
    assert 6.0 <= minutes <= 8.0


def test_session_silence_minutes_returns_zero_for_empty(tmp_path: Path) -> None:
    """session_silence_minutes returns 0.0 when turns list is empty."""
    assert session_silence_minutes([]) == 0.0


def test_delete_session_buffer_is_idempotent(tmp_path: Path) -> None:
    """delete_session_buffer is a no-op when the file doesn't exist."""
    # Should not raise — file was never created.
    delete_session_buffer(tmp_path, "nonexistent_session")
    # Also works when file exists.
    ingest_turn(tmp_path, {"session_id": "sess_del", "speaker": "x", "text": "y"})
    delete_session_buffer(tmp_path, "sess_del")
    assert not (tmp_path / "active_conversations" / "sess_del.jsonl").exists()
    # Second call — still no error.
    delete_session_buffer(tmp_path, "sess_del")


# ---- I-2 follow-up audit: session_id validation ----


def test_ingest_turn_rejects_path_traversal_session_id(tmp_path: Path) -> None:
    """A session_id containing '..' or '/' must raise ValueError, not write
    outside the persona's active_conversations dir.

    Reproduces I-2 from the 2026-05-05 follow-up audit. The HTTP bridge
    constrains session_id at the request-model layer, but the function's
    contract advertises 'Optional: session_id' and previously accepted any
    string straight into a filename interpolation."""
    import pytest as _pytest

    for evil in [
        "../../etc/passwd",
        "../escape",
        "a/b",
        "..",
        "x" * 65,  # too long
        "no spaces here",  # space disallowed
        "weird:char",  # colon disallowed (also Windows-illegal)
    ]:
        with _pytest.raises(ValueError, match="invalid session_id"):
            ingest_turn(tmp_path, {"session_id": evil, "speaker": "u", "text": "x"})
    # Empty string is fine — the `or` fallback in ingest_turn synthesizes a
    # fresh sess_<8hex> id. The audit's concern was that "" would land at
    # ".jsonl"; that no longer happens.
    # Persona dir is unchanged — no escaped writes
    parent_files = list(tmp_path.parent.iterdir())
    leaked = [p for p in parent_files if "passwd" in p.name or "escape" in p.name]
    assert leaked == [], f"unexpected leak: {leaked}"


def test_ingest_turn_accepts_uuid_and_sess_prefix_session_ids(tmp_path: Path) -> None:
    """Both UUID4 (with hyphens) and the sess_<8hex> fallback must be accepted."""
    import uuid as _uuid

    sid_uuid = str(_uuid.uuid4())
    sid_sess = "sess_abcd1234"
    ingest_turn(tmp_path, {"session_id": sid_uuid, "speaker": "u", "text": "a"})
    ingest_turn(tmp_path, {"session_id": sid_sess, "speaker": "u", "text": "b"})
    assert (tmp_path / "active_conversations" / f"{sid_uuid}.jsonl").exists()
    assert (tmp_path / "active_conversations" / f"{sid_sess}.jsonl").exists()


# ---------------------------------------------------------------------------
# image_shas — multimodal turns
# ---------------------------------------------------------------------------


def test_ingest_turn_records_image_shas(tmp_path):
    from brain.ingest.buffer import ingest_turn, read_session

    sid = ingest_turn(
        tmp_path,
        {
            "session_id": "sess_abc12345",
            "speaker": "user",
            "text": "look at this",
            "image_shas": ["a" * 64, "b" * 64],
        },
    )
    rec = read_session(tmp_path, sid)[0]
    assert rec["image_shas"] == ["a" * 64, "b" * 64]


def test_ingest_turn_omits_image_shas_when_absent(tmp_path):
    from brain.ingest.buffer import ingest_turn, read_session

    sid = ingest_turn(
        tmp_path,
        {"session_id": "sess_def67890", "speaker": "user", "text": "hi"},
    )
    rec = read_session(tmp_path, sid)[0]
    assert "image_shas" not in rec


def test_ingest_turn_omits_image_shas_when_empty_list(tmp_path):
    from brain.ingest.buffer import ingest_turn, read_session

    sid = ingest_turn(
        tmp_path,
        {
            "session_id": "sess_e0e0e0e0",
            "speaker": "user",
            "text": "hi",
            "image_shas": [],
        },
    )
    rec = read_session(tmp_path, sid)[0]
    assert "image_shas" not in rec


def test_ingest_turn_image_shas_accepts_tuple(tmp_path):
    """Caller passing a tuple — record stores a list."""
    from brain.ingest.buffer import ingest_turn, read_session

    sid = ingest_turn(
        tmp_path,
        {
            "session_id": "sess_aabbccdd",
            "speaker": "user",
            "text": "hi",
            "image_shas": ("a" * 64, "b" * 64),
        },
    )
    rec = read_session(tmp_path, sid)[0]
    assert rec["image_shas"] == ["a" * 64, "b" * 64]


# ---------------------------------------------------------------------------
# cursor sidecar + read_session_after — full-session-context plan, Task 1
# ---------------------------------------------------------------------------


def test_write_and_read_cursor_roundtrip(tmp_path: Path) -> None:
    ingest_turn(tmp_path, {"session_id": "sess_abc", "speaker": "user", "text": "hi"})
    write_cursor(tmp_path, "sess_abc", "2026-05-10T20:00:00+00:00")
    assert read_cursor(tmp_path, "sess_abc") == "2026-05-10T20:00:00+00:00"


def test_read_cursor_missing_returns_none(tmp_path: Path) -> None:
    assert read_cursor(tmp_path, "sess_abc") is None


def test_read_cursor_malformed_returns_none(tmp_path: Path) -> None:
    (tmp_path / "active_conversations").mkdir(parents=True)
    (tmp_path / "active_conversations" / "sess_abc.cursor").write_text(
        "not-a-timestamp", encoding="utf-8"
    )
    assert read_cursor(tmp_path, "sess_abc") is None


def test_write_cursor_rejects_malformed_ts(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_cursor(tmp_path, "sess_abc", "garbage")


def test_delete_cursor_is_idempotent(tmp_path: Path) -> None:
    delete_cursor(tmp_path, "sess_abc")
    write_cursor(tmp_path, "sess_abc", "2026-05-10T20:00:00+00:00")
    delete_cursor(tmp_path, "sess_abc")
    assert read_cursor(tmp_path, "sess_abc") is None


def test_read_session_after_returns_only_post_cursor_turns(tmp_path: Path) -> None:
    sid = "sess_abc"
    ingest_turn(
        tmp_path,
        {"session_id": sid, "speaker": "user", "text": "a", "ts": "2026-05-10T20:00:00+00:00"},
    )
    ingest_turn(
        tmp_path,
        {"session_id": sid, "speaker": "assistant", "text": "b", "ts": "2026-05-10T20:00:05+00:00"},
    )
    ingest_turn(
        tmp_path,
        {"session_id": sid, "speaker": "user", "text": "c", "ts": "2026-05-10T20:01:00+00:00"},
    )
    out = read_session_after(tmp_path, sid, "2026-05-10T20:00:30+00:00")
    assert [t["text"] for t in out] == ["c"]


def test_read_session_after_none_cursor_returns_all(tmp_path: Path) -> None:
    sid = "sess_abc"
    ingest_turn(
        tmp_path,
        {"session_id": sid, "speaker": "user", "text": "a", "ts": "2026-05-10T20:00:00+00:00"},
    )
    out = read_session_after(tmp_path, sid, None)
    assert len(out) == 1


def test_read_session_after_malformed_cursor_returns_all(tmp_path: Path) -> None:
    sid = "sess_abc"
    ingest_turn(
        tmp_path,
        {"session_id": sid, "speaker": "user", "text": "a", "ts": "2026-05-10T20:00:00+00:00"},
    )
    out = read_session_after(tmp_path, sid, "not-a-ts")
    assert len(out) == 1


# ---------------------------------------------------------------------------
# F-011 — backoff sidecar primitives
# ---------------------------------------------------------------------------


def test_read_backoff_missing_returns_none(tmp_path: Path) -> None:
    assert read_backoff(tmp_path, "sess_abc") is None


def test_read_backoff_malformed_returns_none(tmp_path: Path) -> None:
    (tmp_path / "active_conversations").mkdir(parents=True)
    backoff_file = tmp_path / "active_conversations" / "sess_abc.backoff"
    # Not JSON.
    backoff_file.write_text("not json at all", encoding="utf-8")
    assert read_backoff(tmp_path, "sess_abc") is None
    # JSON but wrong shape (list, not dict).
    backoff_file.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_backoff(tmp_path, "sess_abc") is None
    # JSON dict but missing required keys.
    backoff_file.write_text('{"failures": 2}', encoding="utf-8")
    assert read_backoff(tmp_path, "sess_abc") is None
    # Bad ts inside an otherwise well-shaped dict.
    backoff_file.write_text('{"failures": 2, "first_failure_at": "garbage"}', encoding="utf-8")
    assert read_backoff(tmp_path, "sess_abc") is None
    # Empty file.
    backoff_file.write_text("", encoding="utf-8")
    assert read_backoff(tmp_path, "sess_abc") is None


def test_write_and_read_backoff_roundtrip(tmp_path: Path) -> None:
    write_backoff(tmp_path, "sess_abc", failures=2, first_failure_at="2026-05-10T20:00:00+00:00")
    state = read_backoff(tmp_path, "sess_abc")
    assert state == {
        "failures": 2,
        "first_failure_at": "2026-05-10T20:00:00+00:00",
    }


def test_delete_backoff_is_idempotent(tmp_path: Path) -> None:
    # No file present — must not raise.
    delete_backoff(tmp_path, "sess_abc")
    write_backoff(tmp_path, "sess_abc", failures=1, first_failure_at="2026-05-10T20:00:00+00:00")
    delete_backoff(tmp_path, "sess_abc")
    assert read_backoff(tmp_path, "sess_abc") is None
    # Second delete on missing file — still no-op.
    delete_backoff(tmp_path, "sess_abc")


def test_write_backoff_rejects_malformed_ts(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_backoff(tmp_path, "sess_abc", failures=1, first_failure_at="garbage")
