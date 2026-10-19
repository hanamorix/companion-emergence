"""#476 (audit consolidation C4): one rotation module, and it takes the appenders' lock.

Every writer of these logs appends under `file_lock(path)`. The rolling-size and yearly rotators held nothing, so a row appended while a
rotation ran was lost (the truncate, or the rewrite, removed it), the yearly pass renamed a new archive over an existing kept-forever year
archive, and the rolling log line reported the size AFTER the truncation (always 0). The rotators now run under the lock, re-check the
size under it, merge into an existing year archive, and log the real size. The size-capped single-backup rotator of
`mcp_server.audit` and the age trim of `felt_time.chat_log` use the same module.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import brain.health.log_rotation as lr
from brain.health.json_io import append_jsonl


def _row(i, **extra):
    return {"i": i, **extra}


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _appender_started_inside(monkeypatch, owner, attr: str, log: Path, late_row: dict):
    """Wrap ``owner.<attr>`` so that, once it has run, a concurrent appender starts and gets a head start."""
    real = getattr(owner, attr)
    threads: list[threading.Thread] = []

    def wrapper(*args, **kwargs):
        result = real(*args, **kwargs)
        if not threads:
            t = threading.Thread(target=append_jsonl, args=(log, late_row))
            threads.append(t)
            t.start()
            time.sleep(0.3)  # the appender runs now if nothing stops it
        return result

    monkeypatch.setattr(owner, attr, wrapper)
    return threads


# --- rolling size -----------------------------------------------------------------------------------------------


def test_a_row_appended_during_a_rolling_rotation_is_not_lost(tmp_path, monkeypatch):
    log = tmp_path / "chat_usage.jsonl"
    _write(log, [_row(i, pad="x" * 1000) for i in range(20)])
    # the rotator's copy step is the seam the append lands in
    threads = _appender_started_inside(monkeypatch, lr.shutil, "copyfileobj", log, _row("late"))
    lr.rotate_rolling_size(log, max_bytes=10_000, archive_keep=3)
    for t in threads:
        t.join(timeout=5)
    everywhere = _read_gz(tmp_path / "chat_usage.jsonl.1.gz") + (_rows(log) if log.exists() else [])
    assert any(r["i"] == "late" for r in everywhere)


def test_rolling_rotation_rechecks_the_size_under_the_lock(tmp_path, monkeypatch):
    log = tmp_path / "chat_usage.jsonl"
    _write(log, [_row(i, pad="x" * 1000) for i in range(20)])
    real_lock = lr.file_lock

    @contextmanager
    def peer_rotated_first(path, **kwargs):
        with real_lock(path, **kwargs) as acquired:
            path.write_text(json.dumps(_row("fresh")) + "\n", encoding="utf-8")  # what a peer rotator left behind
            yield acquired

    monkeypatch.setattr(lr, "file_lock", peer_rotated_first)
    assert lr.rotate_rolling_size(log, max_bytes=10_000, archive_keep=3) is None
    assert _rows(log) == [_row("fresh")]
    assert not (tmp_path / "chat_usage.jsonl.1.gz").exists()


def test_the_rolling_log_line_reports_the_size_that_was_rotated(tmp_path, caplog):
    log = tmp_path / "chat_usage.jsonl"
    _write(log, [_row(i, pad="x" * 1000) for i in range(20)])
    size = log.stat().st_size
    with caplog.at_level(logging.INFO, logger=lr.logger.name):
        lr.rotate_rolling_size(log, max_bytes=10_000, archive_keep=3)
    assert f"({size} bytes)" in caplog.text
    assert "(0 bytes)" not in caplog.text


# --- yearly -----------------------------------------------------------------------------------------------------


NOW = datetime(2026, 6, 15, tzinfo=UTC)


def test_a_row_appended_during_a_yearly_rotation_is_not_lost(tmp_path, monkeypatch):
    log = tmp_path / "soul_audit.jsonl"
    _write(log, [{"ts": "2025-03-01T00:00:00+00:00", "i": 1}, {"ts": "2026-02-01T00:00:00+00:00", "i": 2}])
    real_iter = lr.iter_jsonl_raw
    started: list[threading.Thread] = []

    def iter_then_append(path, **kwargs):
        rows = list(real_iter(path, **kwargs))
        if not started:
            t = threading.Thread(
                target=append_jsonl, args=(log, {"ts": "2026-06-14T00:00:00+00:00", "i": "late"})
            )
            started.append(t)
            t.start()
            time.sleep(0.3)
        return iter(rows)

    monkeypatch.setattr(lr, "iter_jsonl_raw", iter_then_append)
    lr.rotate_age_archive_yearly(log, now=NOW, timestamp_field="ts")
    for t in started:
        t.join(timeout=5)
    assert [r["i"] for r in _rows(log)] == [2, "late"]
    assert [r["i"] for r in _read_gz(tmp_path / "soul_audit.2025.jsonl.gz")] == [1]


def test_a_year_archive_that_already_exists_is_merged_not_replaced(tmp_path):
    log = tmp_path / "soul_audit.jsonl"
    archive = tmp_path / "soul_audit.2025.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as gz:
        gz.write(json.dumps({"ts": "2025-01-01T00:00:00+00:00", "i": "kept-forever"}) + "\n")
    _write(log, [{"ts": "2025-12-01T00:00:00+00:00", "i": "restored"}, {"ts": "2026-02-01T00:00:00+00:00", "i": "now"}])
    new = lr.rotate_age_archive_yearly(log, now=NOW, timestamp_field="ts")
    assert new == [archive]
    assert [r["i"] for r in _read_gz(archive)] == ["kept-forever", "restored"]
    assert [r["i"] for r in _rows(log)] == ["now"]
    assert not list(tmp_path.glob("*.tmp"))


def test_a_failed_merge_leaves_the_existing_archive_and_the_log_alone(tmp_path, monkeypatch):
    log = tmp_path / "soul_audit.jsonl"
    archive = tmp_path / "soul_audit.2025.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as gz:
        gz.write(json.dumps({"ts": "2025-01-01T00:00:00+00:00", "i": "kept-forever"}) + "\n")
    before = archive.read_bytes()
    _write(log, [{"ts": "2025-12-01T00:00:00+00:00", "i": "restored"}, {"ts": "2026-02-01T00:00:00+00:00", "i": "now"}])
    log_before = log.read_bytes()

    real_replace = os.replace

    def boom_on_the_archive(src, dst):
        if str(dst).endswith(".2025.jsonl.gz"):
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", boom_on_the_archive)
    with pytest.raises(OSError):
        lr.rotate_age_archive_yearly(log, now=NOW, timestamp_field="ts")
    assert archive.read_bytes() == before
    assert log.read_bytes() == log_before


# --- single backup (the audit log's policy) ---------------------------------------------------------------------


def test_single_backup_rotation_moves_an_oversize_log_to_dot_one(tmp_path):
    log = tmp_path / "tool_invocations.log.jsonl"
    log.write_text("old\n" * 100, encoding="utf-8")
    lr.rotate_single_backup(log, max_bytes=100)
    assert (tmp_path / "tool_invocations.log.jsonl.1").read_text(encoding="utf-8") == "old\n" * 100
    assert not log.exists()


def test_single_backup_rotation_leaves_a_small_log(tmp_path):
    log = tmp_path / "tool_invocations.log.jsonl"
    log.write_text("small\n", encoding="utf-8")
    lr.rotate_single_backup(log, max_bytes=100)
    assert log.read_text(encoding="utf-8") == "small\n"


def test_audit_log_uses_the_shared_rotation_module(tmp_path, monkeypatch):
    from brain.mcp_server import audit

    seen = []
    monkeypatch.setattr(lr, "rotate_single_backup", lambda path, max_bytes: seen.append((path, max_bytes)))
    audit._rotate_if_needed(tmp_path / "tool_invocations.log.jsonl")
    assert seen == [(tmp_path / "tool_invocations.log.jsonl", audit._MAX_LOG_BYTES)]


# --- age trim (the chat tick log's policy) ----------------------------------------------------------------------


def test_age_trim_drops_old_and_undated_rows_and_keeps_the_rest_verbatim(tmp_path):
    log = tmp_path / "chat_turns.log.jsonl"
    cutoff = datetime(2026, 6, 1, tzinfo=UTC)
    _write(
        log,
        [
            {"ts": "2026-05-01T00:00:00+00:00", "turns": 1},
            {"turns": 5},
            {"ts": "2026-06-10T00:00:00+00:00", "turns": 2},
            {"ts": "2026-06-11T00:00:00", "turns": 3},
        ],
    )
    dropped = lr.trim_jsonl_by_age(log, cutoff, timestamp_field="ts")
    assert dropped == 2
    assert [r["turns"] for r in _rows(log)] == [2, 3]


def test_age_trim_keeps_a_row_appended_since_the_callers_read(tmp_path, monkeypatch):
    log = tmp_path / "chat_turns.log.jsonl"
    _write(log, [{"ts": "2026-05-01T00:00:00+00:00", "turns": 1}, {"ts": "2026-06-10T00:00:00+00:00", "turns": 2}])
    real_lock = lr.file_lock

    @contextmanager
    def appended_before_the_lock_is_held(path, **kwargs):
        append_jsonl(path, {"ts": "2026-06-12T00:00:00+00:00", "turns": 9})
        with real_lock(path, **kwargs) as acquired:
            yield acquired

    monkeypatch.setattr(lr, "file_lock", appended_before_the_lock_is_held)
    lr.trim_jsonl_by_age(log, datetime(2026, 6, 1, tzinfo=UTC), timestamp_field="ts")
    assert [r["turns"] for r in _rows(log)] == [2, 9]


def test_chat_log_trim_goes_through_the_shared_module(tmp_path, monkeypatch):
    import brain.felt_time.chat_log as chat_log

    now = datetime.now(UTC)
    for i in range(15):
        chat_log.append_chat_tick(tmp_path, ts=now - timedelta(days=40, minutes=i), turns=1)
    for i in range(5):
        chat_log.append_chat_tick(tmp_path, ts=now - timedelta(hours=i + 1), turns=2)
    calls = []
    real = chat_log.trim_jsonl_by_age
    monkeypatch.setattr(chat_log, "trim_jsonl_by_age", lambda *a, **k: calls.append(1) or real(*a, **k))
    chat_log.load_recent_samples(tmp_path)
    assert calls


# --- a caller that already holds the lock ------------------------------------------------------------------------


def test_a_caller_holding_the_lock_can_rotate_without_deadlocking(tmp_path):
    from brain.utils.file_lock import file_lock

    log = tmp_path / "parse_failures.jsonl"
    _write(log, [_row(i, pad="x" * 1000) for i in range(20)])
    done = []

    def run():
        with file_lock(log):  # flock is not reentrant: rotating again under it must not take it again
            done.append(lr.rotate_rolling_size(log, max_bytes=10_000, archive_keep=3, lock_held=True))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "rotation under a held lock deadlocked"
    assert done and done[0] == tmp_path / "parse_failures.jsonl.1.gz"


def test_parse_failure_recording_rotates_without_deadlocking(tmp_path, monkeypatch):
    import brain.health.parse_failures as pf

    monkeypatch.setattr(pf, "MAX_BYTES", 500)
    t = threading.Thread(
        target=lambda: [pf.record_parse_failure(tmp_path, site="x", raw="r" * 400, error="e") for _ in range(6)],
        daemon=True,
    )
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "recording a parse failure deadlocked in the rotation"
    assert (tmp_path / "parse_failures.jsonl.1.gz").exists()
