"""Tests for brain.ingest.buffer — BUFFER + CLOSE stage."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.ingest import buffer as buffer_mod
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
from brain.utils import file_ops


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


def test_delete_session_buffer_retries_transient_permission_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Windows CI regression (WinError 32): unlink of a freshly-written buffer
    can hit a transient PermissionError (antivirus/indexer or a not-yet-GC'd
    reader handle — Windows can't delete open files). delete_session_buffer
    must retry briefly instead of failing close_session with ingest_failed."""
    from pathlib import Path as _PathCls


    ingest_turn(tmp_path, {"session_id": "sess_retry", "speaker": "x", "text": "y"})
    path = tmp_path / "active_conversations" / "sess_retry.jsonl"
    assert path.exists()

    real_unlink = _PathCls.unlink
    attempts = {"n": 0}

    def flaky_unlink(self, missing_ok=False):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PermissionError(32, "The process cannot access the file")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(_PathCls, "unlink", flaky_unlink)
    monkeypatch.setattr(file_ops.time, "sleep", lambda _s: None)

    delete_session_buffer(tmp_path, "sess_retry")
    assert attempts["n"] == 3
    assert not path.exists()


# ---- I-2 follow-up audit: session_id validation ----


def test_ingest_turn_rejects_path_traversal_session_id(tmp_path: Path) -> None:
    """A session_id containing '..' or '/' must raise ValueError, not write
    outside the persona's active_conversations dir.

    Reproduces I-2 from the 2026-05-05 follow-up audit. The HTTP bridge
    constrains session_id at the request-model layer, but the function's
    contract advertises 'Optional: session_id' and previously accepted any
    string straight into a filename interpolation."""
    import pytest as _pytest

    # Snapshot the shared pytest basetemp BEFORE the evil loop. A real traversal
    # would create a NEW entry here; comparing before/after is robust against
    # unrelated tmp dirs whose names merely contain "escape"/"passwd" (e.g.
    # security tests named *_escape_*), which a substring scan false-flagged.
    before = set(tmp_path.parent.iterdir())

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
    # Persona dir is unchanged — no escaped writes landed in the parent.
    after = set(tmp_path.parent.iterdir())
    assert after == before, f"unexpected leak: {sorted(after - before)}"


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


# ── #198: rewrite_session_atomic survives a transient os.replace refusal ─────
# Windows refuses os.replace over a file another handle holds open (a GET
# /chat/history streaming the buffer). Same bounded retry as the shared
# file_ops.retry_transient, and never leave the .jsonl.tmp behind.


def _seed_session(persona_dir: Path, sid: str) -> Path:
    from brain.ingest.buffer import _session_path, ingest_turn

    ingest_turn(persona_dir, {"session_id": sid, "speaker": "user", "text": "hi"})
    return _session_path(persona_dir, sid)


def test_rewrite_session_atomic_retries_transient_permission_error(tmp_path, monkeypatch):
    import os as _os

    from brain.ingest import buffer as buf

    sid = "11111111-1111-4111-8111-111111111111"
    path = _seed_session(tmp_path, sid)
    real_replace = _os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(32, "in use")
        return real_replace(src, dst)

    monkeypatch.setattr(buf.os, "replace", flaky)
    monkeypatch.setattr(dev_constants, "FILE_RETRY_SLEEP_S", 0.0)
    buf.rewrite_session_atomic(tmp_path, sid, [{"speaker": "assistant", "text": "rewritten"}])
    assert calls["n"] == 3
    assert path.read_text(encoding="utf-8").strip() == '{"speaker": "assistant", "text": "rewritten"}'
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_rewrite_session_atomic_gives_up_and_removes_tmp(tmp_path, monkeypatch):
    from brain.ingest import buffer as buf

    sid = "22222222-2222-4222-8222-222222222222"
    path = _seed_session(tmp_path, sid)
    before = path.read_text(encoding="utf-8")

    def always(src, dst):
        raise PermissionError(32, "in use")

    monkeypatch.setattr(buf.os, "replace", always)
    monkeypatch.setattr(dev_constants, "FILE_RETRY_SLEEP_S", 0.0)
    with pytest.raises(PermissionError):
        buf.rewrite_session_atomic(tmp_path, sid, [{"speaker": "assistant", "text": "x"}])
    assert path.read_text(encoding="utf-8") == before  # old buffer untouched
    assert not path.with_suffix(".jsonl.tmp").exists()  # no litter


# --- archive_once: the one archive-with-dedup write (#556) ------------------------------------------------------


def _turns(n: int, *, tag: str = "t") -> list[dict]:
    return [
        {"session_id": "s1", "speaker": "user" if i % 2 == 0 else "assistant", "text": f"{tag}-{i}", "ts": f"2026-01-01T00:00:{i:02d}+00:00"}
        for i in range(n)
    ]


def test_archive_once_writes_a_batch_once_per_marker_key(tmp_path):
    from brain.ingest.buffer import archive_once, read_archive

    rows = _turns(3)
    assert archive_once(tmp_path, "s1", rows, new_gen=1) > 0
    # The identical batch again (a crash-retry) is proven archived by the marker: no second write, still success.
    assert archive_once(tmp_path, "s1", rows, new_gen=1) > 0
    assert read_archive(tmp_path, "s1") == rows


def test_archive_once_writes_a_different_batch(tmp_path):
    from brain.ingest.buffer import archive_once, read_archive

    rows = _turns(3)
    archive_once(tmp_path, "s1", rows, new_gen=1)
    archive_once(tmp_path, "s1", rows, new_gen=2)  # same rows, other generation: a different batch
    more = _turns(2, tag="u")
    archive_once(tmp_path, "s1", more, new_gen=2)
    assert read_archive(tmp_path, "s1") == rows + rows + more


def test_archive_once_empty_batch_writes_nothing(tmp_path):
    from brain.ingest.buffer import archive_once, read_archive, read_archive_marker

    assert archive_once(tmp_path, "s1", [], new_gen=1) == 0
    assert read_archive(tmp_path, "s1") == []
    assert read_archive_marker(tmp_path, "s1") is None


def test_archive_once_keeps_two_byte_identical_turns(tmp_path):
    from brain.ingest.buffer import archive_once, read_archive

    row = _turns(1)[0]
    archive_once(tmp_path, "s1", [row, dict(row)], new_gen=1)
    assert read_archive(tmp_path, "s1") == [row, row]


# --- session_append_lock (#556) ---------------------------------------------------------------------------------


def test_session_append_lock_excludes_a_lock_respecting_appender(tmp_path):
    from brain.health.json_io import append_jsonl
    from brain.ingest.buffer import session_append_lock

    sid = "sess_lock"
    path = tmp_path / "active_conversations" / f"{sid}.jsonl"
    with session_append_lock(tmp_path, sid):
        assert append_jsonl(path, {"text": "while held"}, lock="try") is None
    assert append_jsonl(path, {"text": "after"}, lock="try") > 0
    assert [r["text"] for r in read_session(tmp_path, sid)] == ["after"]


@pytest.mark.parametrize("raise_on", ["call", "enter"])
def test_session_append_lock_runs_the_body_once_unlocked_when_the_lock_is_refused(tmp_path, monkeypatch, caplog, raise_on):
    from contextlib import contextmanager

    from brain.ingest import buffer as buf

    if raise_on == "call":

        def refuse(path, **kw):
            raise OSError("no locks here")

    else:

        @contextmanager
        def refuse(path, **kw):
            raise OSError("no locks here")
            yield  # pragma: no cover

    monkeypatch.setattr(buf, "file_lock", refuse)
    runs: list[int] = []
    with caplog.at_level("WARNING", logger=buf.logger.name):
        with buf.session_append_lock(tmp_path, "sess_lock"):
            runs.append(1)
    assert runs == [1]
    assert any("without it" in r.getMessage() for r in caplog.records)


def test_session_append_lock_propagates_a_body_error_once(tmp_path):
    from brain.ingest.buffer import session_append_lock

    runs: list[int] = []
    with pytest.raises(OSError, match="body failed"):
        with session_append_lock(tmp_path, "sess_lock"):
            runs.append(1)
            raise OSError("body failed")
    assert runs == [1]


@pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="the interpreter's integer digit limit is off")
def test_read_session_raises_a_named_error_on_an_over_long_integer_line(tmp_path: Path) -> None:
    """A line holding an integer past the interpreter's digit limit stops the read with ``UnloadableJsonlLineError`` (a known
    defect, kept as it was before #643 and filed as #646; it is a ``ValueError``, so the bridge's ``read_session`` handler
    still skips the session). A fix for #646 that makes the read skip the line must change this test on purpose."""
    from brain.health import jsonl_reader

    folder = tmp_path / "active_conversations"
    folder.mkdir()
    huge = '{"speaker": "user", "n": ' + "9" * (sys.get_int_max_str_digits() + 1) + "}"
    good1 = json.dumps({"speaker": "user", "text": "one"})
    good2 = json.dumps({"speaker": "assistant", "text": "two"})
    session = folder / "sess_big.jsonl"
    session.write_text("\n".join([good1, huge, good2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        read_session(tmp_path, "sess_big")
    assert type(exc.value).__name__ == "UnloadableJsonlLineError"
    assert type(exc.value) is jsonl_reader.UnloadableJsonlLineError
    assert (exc.value.code, exc.value.line, exc.value.path) == ("jsonl_unloadable_line", 2, session)
    assert type(exc.value.__cause__) is ValueError


# ---------------------------------------------------------------------------
# last_user_turn_ts (Engine 6 G3): the one place the time of the user's last turn is computed
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)


def _hours(n: float) -> datetime:
    return _T0 + timedelta(hours=n)


def _row(speaker: str, ts: object, **extra: object) -> dict:
    stamp = ts.isoformat() if isinstance(ts, datetime) else ts
    return {"session_id": "s", "speaker": speaker, "text": "t", "ts": stamp, **extra}


def _put(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _active(root: Path, sid: str, rows: list[dict]) -> Path:
    return _put(root / "active_conversations" / f"{sid}.jsonl", rows)


def _archived(root: Path, name: str, rows: list[dict], *, mtime: datetime | None = None) -> Path:
    path = _put(root / "archived_conversations" / name, rows)
    if mtime is not None:
        stamp_ns = int(mtime.timestamp()) * 10**9 + mtime.microsecond * 1000
        os.utime(path, ns=(stamp_ns, stamp_ns))
    return path


@pytest.fixture
def fresh_throttle():
    """The fail-soft warnings are throttled per key for the life of the process; a test that reads the log starts clean."""
    from brain.utils import log_throttle

    log_throttle._last.clear()
    yield
    log_throttle._last.clear()


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The file names ``iter_jsonl`` is asked to read inside the buffer module, in order."""
    seen: list[str] = []
    real = buffer_mod.iter_jsonl

    def spy(path, *args, **kwargs):
        seen.append(Path(path).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(buffer_mod, "iter_jsonl", spy)
    return seen


def test_last_user_turn_ts_skips_trailing_assistant_summary_and_image_rows(tmp_path: Path) -> None:
    _active(
        tmp_path,
        "sess_a",
        [
            _row("user", _hours(1)),
            _row("user", _hours(2)),
            _row("assistant", _hours(3)),
            _row("summary", _hours(4)),
            _row("assistant", _hours(5), image_shas=["abc"]),
        ],
    )
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(2)


def test_last_user_turn_ts_finds_a_user_row_behind_many_later_rows(tmp_path: Path) -> None:
    rows = [_row("user", _hours(1))] + [_row("assistant", _hours(2 + i / 100)) for i in range(300)]
    _active(tmp_path, "sess_long", rows)
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(1)


def test_last_user_turn_ts_takes_the_newest_across_active_files(tmp_path: Path) -> None:
    _active(tmp_path, "sess_old", [_row("user", _hours(1))])
    _active(tmp_path, "sess_new", [_row("user", _hours(7)), _row("assistant", _hours(8))])
    _active(tmp_path, "sess_mid", [_row("user", _hours(3))])
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(7)


def test_last_user_turn_ts_skips_a_corrupt_line(tmp_path: Path) -> None:
    path = tmp_path / "active_conversations" / "sess_c.jsonl"
    path.parent.mkdir()
    path.write_text(
        json.dumps(_row("user", _hours(2))) + "\nnot json at all\n" + json.dumps(_row("user", _hours(4))) + "\n",
        encoding="utf-8",
    )
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(4)


def test_last_user_turn_ts_reads_a_naive_ts_as_utc(tmp_path: Path) -> None:
    _active(tmp_path, "sess_n", [_row("user", "2026-03-10T15:00:00")])
    result = buffer_mod.last_user_turn_ts(tmp_path)
    assert result == datetime(2026, 3, 10, 15, 0, 0, tzinfo=UTC)
    assert result.tzinfo is not None


def test_last_user_turn_ts_converts_an_offset_ts_to_utc(tmp_path: Path) -> None:
    _active(tmp_path, "sess_o", [_row("user", "2026-03-10T15:00:00+02:00"), _row("user", "2026-03-10T10:00:00Z")])
    assert buffer_mod.last_user_turn_ts(tmp_path) == datetime(2026, 3, 10, 13, 0, 0, tzinfo=UTC)


def test_last_user_turn_ts_skips_a_row_whose_ts_does_not_parse_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fresh_throttle
) -> None:
    rows = [
        _row("user", _hours(2)),
        _row("user", "not a time"),
        _row("user", 12345),
        {"session_id": "s", "speaker": "user", "text": "no ts"},
    ]
    _active(tmp_path, "sess_b", rows)
    with caplog.at_level(logging.WARNING, logger="brain.ingest.buffer"):
        assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(2)
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_last_user_turn_ts_ignores_a_row_that_is_not_the_user_speaker(tmp_path: Path) -> None:
    _active(tmp_path, "sess_s", [_row("assistant", _hours(5)), _row("User", _hours(6)), _row("kestrel", _hours(7))])
    assert buffer_mod.last_user_turn_ts(tmp_path) is None


def test_last_user_turn_ts_returns_none_with_no_folders_and_creates_nothing(tmp_path: Path) -> None:
    assert buffer_mod.last_user_turn_ts(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_last_user_turn_ts_returns_none_with_no_user_turn_anywhere_and_creates_no_archive(tmp_path: Path) -> None:
    _active(tmp_path, "sess_x", [_row("assistant", _hours(1))])
    assert buffer_mod.last_user_turn_ts(tmp_path) is None
    assert not (tmp_path / "archived_conversations").exists()


def test_last_user_turn_ts_reads_the_archive_when_no_active_file_holds_a_user_turn(tmp_path: Path) -> None:
    _active(tmp_path, "sess_now", [_row("summary", _hours(30))])
    _archived(tmp_path, "sess_legacy.jsonl", [_row("user", _hours(5)), _row("assistant", _hours(6))])
    _archived(tmp_path, "sess_seg.000.jsonl", [_row("user", _hours(2))])
    _archived(tmp_path, "sess_seg.001.jsonl", [_row("user", _hours(9)), _row("assistant", _hours(10))])
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(9)


def test_last_user_turn_ts_takes_the_latest_ts_across_sessions_not_the_newest_file(tmp_path: Path) -> None:
    # Session ids are not in time order, and the file touched last is not the session whose user spoke last.
    _archived(tmp_path, "zzz_first.jsonl", [_row("user", _hours(9))], mtime=_hours(9.5))
    _archived(tmp_path, "aaa_second.jsonl", [_row("user", _hours(2)), _row("assistant", _hours(20))], mtime=_hours(20))
    _archived(tmp_path, "mmm_third.jsonl", [_row("user", _hours(4))], mtime=_hours(4.5))
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(9)


def test_last_user_turn_ts_leaves_the_archive_unread_when_an_active_file_holds_a_user_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _active(tmp_path, "sess_live", [_row("user", _hours(3))])
    _archived(tmp_path, "sess_old.jsonl", [_row("user", _hours(99))])

    def boom(folder):
        raise AssertionError(f"the archive folder was listed: {folder}")

    monkeypatch.setattr(buffer_mod, "list_dir", boom)
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(3)


def test_last_user_turn_ts_never_opens_a_session_whose_last_file_is_not_newer_than_the_best_found(
    tmp_path: Path, reads: list[str]
) -> None:
    _archived(tmp_path, "sess_a.000.jsonl", [_row("user", _hours(9)), _row("assistant", _hours(10))], mtime=_hours(10))
    _archived(tmp_path, "sess_b.jsonl", [_row("user", _hours(7.5))], mtime=_hours(8))
    _archived(tmp_path, "sess_c.jsonl", [_row("user", _hours(9.25))], mtime=_hours(9.5))
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(9.25)
    # a (mtime 10) first, c (9.5) is newer than a's best 9 so it is read, b (8) is not newer than 9.25: never opened.
    assert reads == ["sess_a.000.jsonl", "sess_c.jsonl"]


def test_last_user_turn_ts_visits_sessions_newest_file_first_by_modification_time(tmp_path: Path, reads: list[str]) -> None:
    _archived(tmp_path, "a_old.jsonl", [_row("user", _hours(1))], mtime=_hours(1))
    _archived(tmp_path, "z_new.jsonl", [_row("user", _hours(6))], mtime=_hours(6))
    _archived(tmp_path, "m_mid.jsonl", [_row("user", _hours(3))], mtime=_hours(3))
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(6)
    assert reads == ["z_new.jsonl"]


def test_last_user_turn_ts_reads_a_session_from_its_last_file_backwards_and_stops_at_the_first_user_turn(
    tmp_path: Path, reads: list[str]
) -> None:
    _archived(tmp_path, "sess_s.jsonl", [_row("user", _hours(1))])  # the legacy file, the oldest
    _archived(tmp_path, "sess_s.000.jsonl", [_row("user", _hours(2))])
    _archived(tmp_path, "sess_s.001.jsonl", [_row("user", _hours(3))])
    _archived(tmp_path, "sess_s.002.jsonl", [_row("assistant", _hours(4)), _row("summary", _hours(4.5))])
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(3)
    assert reads == ["sess_s.002.jsonl", "sess_s.001.jsonl"]


def test_last_user_turn_ts_reads_segments_in_numeric_order_and_the_legacy_file_first(tmp_path: Path, reads: list[str]) -> None:
    _archived(tmp_path, "sess_t.jsonl", [_row("user", _hours(5))])  # legacy: before every numbered segment
    _archived(tmp_path, "sess_t.002.jsonl", [_row("assistant", _hours(6))])
    _archived(tmp_path, "sess_t.010.jsonl", [_row("assistant", _hours(7))])
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(5)
    assert reads == ["sess_t.010.jsonl", "sess_t.002.jsonl", "sess_t.jsonl"]


def test_last_user_turn_ts_ignores_archive_entries_that_are_not_session_files(tmp_path: Path) -> None:
    _archived(tmp_path, "sess_ok.jsonl", [_row("user", _hours(2))])
    folder = tmp_path / "archived_conversations"
    (folder / "sess_ok.archived").write_text("marker", encoding="utf-8")
    (folder / "notes.txt").write_text("hello", encoding="utf-8")
    _put(folder / "bad name!.jsonl", [_row("user", _hours(50))])
    _put(folder / "sess_ok.tmp.jsonl", [_row("user", _hours(60))])  # not a numbered segment of sess_ok
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(2)


def test_last_user_turn_ts_skips_a_session_whose_last_file_vanished_since_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archived(tmp_path, "gone.jsonl", [_row("user", _hours(9))])
    _archived(tmp_path, "here.jsonl", [_row("user", _hours(2))])
    real = buffer_mod.stat_path
    monkeypatch.setattr(buffer_mod, "stat_path", lambda p: None if Path(p).name == "gone.jsonl" else real(p))
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(2)


def test_last_user_turn_ts_skips_a_session_whose_stat_fails_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fresh_throttle
) -> None:
    _archived(tmp_path, "locked.jsonl", [_row("user", _hours(9))])
    _archived(tmp_path, "open.jsonl", [_row("user", _hours(2))])
    real = buffer_mod.stat_path

    def flaky(path):
        if Path(path).name == "locked.jsonl":
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(buffer_mod, "stat_path", flaky)
    with caplog.at_level(logging.WARNING, logger="brain.ingest.buffer"):
        assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(2)
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_last_user_turn_ts_returns_none_with_a_warning_when_the_archive_folder_cannot_be_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fresh_throttle
) -> None:
    _active(tmp_path, "sess_q", [_row("assistant", _hours(1))])
    _archived(tmp_path, "sess_old.jsonl", [_row("user", _hours(2))])

    def denied(folder):
        raise PermissionError("denied")

    monkeypatch.setattr(buffer_mod, "list_dir", denied)
    with caplog.at_level(logging.WARNING, logger="brain.ingest.buffer"):
        assert buffer_mod.last_user_turn_ts(tmp_path) is None
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_last_user_turn_ts_goes_on_to_the_archive_when_the_active_folder_cannot_be_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fresh_throttle
) -> None:
    _archived(tmp_path, "sess_old.jsonl", [_row("user", _hours(2))])

    def denied(persona_dir):
        raise PermissionError("denied")

    monkeypatch.setattr(buffer_mod, "list_active_sessions", denied)
    with caplog.at_level(logging.WARNING, logger="brain.ingest.buffer"):
        assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(2)
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.skipif(sys.get_int_max_str_digits() == 0, reason="the interpreter's integer digit limit is off")
def test_last_user_turn_ts_stops_a_file_at_an_unloadable_line_keeps_what_it_read_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fresh_throttle
) -> None:
    huge = '{"speaker": "user", "n": ' + "9" * (sys.get_int_max_str_digits() + 1) + "}"
    path = tmp_path / "active_conversations" / "sess_big.jsonl"
    path.parent.mkdir()
    path.write_text(
        json.dumps(_row("user", _hours(2))) + "\n" + huge + "\n" + json.dumps(_row("user", _hours(9))) + "\n",
        encoding="utf-8",
    )
    _active(tmp_path, "sess_fine", [_row("user", _hours(1))])
    with caplog.at_level(logging.WARNING, logger="brain.ingest.buffer"):
        assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(2)
    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_last_user_turn_ts_reads_a_listed_file_that_cannot_be_opened_as_no_turn_and_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brain.health import json_io

    _active(tmp_path, "locked", [_row("user", _hours(9))])
    _active(tmp_path, "open", [_row("user", _hours(2))])
    real_open = open

    def refuse(path, *args, **kwargs):
        if Path(path).name == "locked.jsonl":
            raise PermissionError("denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(json_io, "open", refuse, raising=False)
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(2)


def test_last_user_turn_ts_ignores_an_active_file_whose_name_is_not_a_session_id(tmp_path: Path) -> None:
    _active(tmp_path, "a.b", [_row("user", _hours(9))])  # a.b.jsonl: the stem has a dot, so it is not a session buffer
    _active(tmp_path, "ok", [_row("user", _hours(2))])
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(2)


def test_last_user_turn_ts_takes_a_sessions_mtime_from_its_last_numbered_segment_not_its_legacy_file(
    tmp_path: Path, reads: list[str]
) -> None:
    # Session x: an old legacy file and a newer segment. Its last file is the segment (mtime 9), so x is visited first.
    # z sits between the two mtimes of x: if x were ordered by its legacy file (mtime 20) it would be visited first and z
    # never read; ordered by its last segment (mtime 9) z (mtime 12) comes first and x is not newer than z's best (10).
    _archived(tmp_path, "x.jsonl", [_row("user", _hours(1))], mtime=_hours(20))  # the legacy file was touched last
    _archived(tmp_path, "x.000.jsonl", [_row("user", _hours(9))], mtime=_hours(9))
    _archived(tmp_path, "z.jsonl", [_row("user", _hours(10))], mtime=_hours(12))
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(10)
    assert reads == ["z.jsonl"]


def test_last_user_turn_ts_stops_when_the_best_found_equals_the_next_files_modification_time(
    tmp_path: Path, reads: list[str]
) -> None:
    # "Not older than": a file last written at hour 9 cannot hold a turn after hour 9, so it cannot beat a turn at hour 9.
    _archived(tmp_path, "a.jsonl", [_row("user", _hours(9))], mtime=_hours(10))
    _archived(tmp_path, "b.jsonl", [_row("user", _hours(5))], mtime=_hours(9))
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(9)
    assert reads == ["a.jsonl"]


def test_last_user_turn_ts_does_not_raise_on_a_segment_name_with_a_non_ascii_digit(tmp_path: Path) -> None:
    _archived(tmp_path, "s.jsonl", [_row("user", _hours(3))])
    _put(tmp_path / "archived_conversations" / "s.².jsonl", [_row("user", _hours(50))])  # "s.².jsonl"
    assert buffer_mod.last_user_turn_ts(tmp_path) == _hours(3)


def test_last_user_turn_ts_reads_a_folder_where_a_session_file_should_be_as_no_turn(tmp_path: Path) -> None:
    (tmp_path / "active_conversations" / "s.jsonl").mkdir(parents=True)
    (tmp_path / "archived_conversations" / "t.jsonl").mkdir(parents=True)
    assert buffer_mod.last_user_turn_ts(tmp_path) is None
