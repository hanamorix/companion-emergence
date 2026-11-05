"""Tests for brain.mcp_server.audit."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from brain.health import log_rotation
from brain.mcp_server import audit
from brain.mcp_server.audit import log_invocation
from tests.thread_support import finish_threads, release_at_teardown


def test_log_invocation_writes_jsonl_line(tmp_path: Path) -> None:
    log_invocation(
        tmp_path,
        name="search_memories",
        arguments={"query": "morning"},
        result_summary="3 hits",
    )
    log_path = tmp_path / "tool_invocations.log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["name"] == "search_memories"
    # Audit 2026-05-07 P3-3: 'query' joined the sensitive-keys allowlist
    # so search terms get redacted in 'redacted' mode — they can be as
    # identifying as content.
    assert rec["arguments"] == {"query": "[REDACTED]"}
    assert rec["result_summary"] == "3 hits"
    assert rec["error"] is None
    # Timestamp ends with Z (UTC)
    assert rec["timestamp"].endswith("Z")


def test_log_invocation_redacts_sensitive_argument_fields(tmp_path: Path) -> None:
    log_invocation(
        tmp_path,
        name="add_memory",
        arguments={"content": "private body text", "metadata": {"text": "journal secret"}},
        result_summary='{"memories": [{"content": "private body text"}]}',
    )
    rec = json.loads((tmp_path / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["arguments"]["content"] == "[REDACTED]"
    assert rec["arguments"]["metadata"]["text"] == "[REDACTED]"
    assert "private body text" not in rec["result_summary"]


def test_log_invocation_redacts_urls(tmp_path: Path) -> None:
    """#273: a visited URL is as identifying as a search query."""
    log_invocation(
        tmp_path,
        name="some_tool",
        arguments={"url": "https://example.com/private/page"},
        result_summary='{"url": "https://example.com/private/page"}',
    )
    rec = json.loads((tmp_path / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["arguments"]["url"] == "[REDACTED]"
    assert "example.com" not in rec["result_summary"]


def test_log_invocation_metadata_mode_omits_arguments_and_summary(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "persona_config.json").write_text(
        json.dumps({"mcp_audit_log_level": "metadata"}), encoding="utf-8"
    )
    log_invocation(
        tmp_path,
        name="add_memory",
        arguments={"content": "private body text"},
        result_summary='{"content": "private body text"}',
    )
    rec = json.loads((tmp_path / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["audit_level"] == "metadata"
    assert rec["arguments"] == "[OMITTED]"
    assert rec["result_summary"] == ""


def test_log_invocation_off_mode_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "persona_config.json").write_text(
        json.dumps({"mcp_audit_log_level": "off"}), encoding="utf-8"
    )
    log_invocation(
        tmp_path,
        name="add_memory",
        arguments={"content": "private body text"},
        result_summary="secret",
    )
    assert not (tmp_path / "tool_invocations.log.jsonl").exists()


def test_log_invocation_persona_config_beats_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NELL_MCP_AUDIT_LOG_LEVEL", "full")
    (tmp_path / "persona_config.json").write_text(
        json.dumps({"mcp_audit_log_level": "metadata"}), encoding="utf-8"
    )

    log_invocation(
        tmp_path,
        name="add_memory",
        arguments={"content": "private body text"},
        result_summary='{"content": "private body text"}',
    )

    rec = json.loads((tmp_path / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["audit_level"] == "metadata"
    assert "private body text" not in json.dumps(rec)


def test_log_invocation_truncates_long_summary(tmp_path: Path) -> None:
    long = "x" * 500
    log_invocation(tmp_path, name="x", arguments={}, result_summary=long)
    rec = json.loads((tmp_path / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    # 140 chars + ellipsis (1 char "…")
    assert rec["result_summary"].endswith("…")
    assert len(rec["result_summary"]) == 141


def test_log_invocation_records_error(tmp_path: Path) -> None:
    log_invocation(
        tmp_path,
        name="add_memory",
        arguments={"text": "x"},
        result_summary="error: boom",
        error="boom",
    )
    rec = json.loads((tmp_path / "tool_invocations.log.jsonl").read_text(encoding="utf-8"))
    assert rec["error"] == "boom"


def test_log_invocation_appends_two_lines(tmp_path: Path) -> None:
    log_invocation(tmp_path, name="a", arguments={}, result_summary="r1")
    log_invocation(tmp_path, name="b", arguments={}, result_summary="r2")
    lines = (tmp_path / "tool_invocations.log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["name"] == "a"
    assert json.loads(lines[1])["name"] == "b"


def test_log_invocation_swallows_oserror(tmp_path: Path, monkeypatch) -> None:
    """Audit is observability — if the disk write fails, we log + swallow."""
    # Make the persona_dir read-only so the open() raises OSError
    persona = tmp_path / "ro"
    persona.mkdir()
    persona.chmod(0o555)
    try:
        # Should not raise
        log_invocation(persona, name="x", arguments={}, result_summary="x")
    finally:
        persona.chmod(0o755)


def test_stale_rotator_does_not_clobber_the_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """#357: a writer that saw an oversize log must not rotate a fresh one.

    B decides to rotate, stalls just before its rename, and meanwhile A
    rotates the same log and a new row lands in a fresh log. Unguarded, B
    then renames that one-row log over ``.1`` and the whole old log is gone.
    Guarded, B holds a lock and re-checks, so A waits (or B rotates first)
    and the old rows survive in either order.
    """
    monkeypatch.setattr(audit, "_MAX_LOG_BYTES", 100)
    log_path = tmp_path / "tool_invocations.log.jsonl"
    backup = log_path.with_name(log_path.name + ".1")
    old_rows = "old-row-" + "x" * 200 + "\n"
    log_path.write_text(old_rows, encoding="utf-8")

    at_gate, release = threading.Event(), threading.Event()
    stalled = threading.local()
    real_replace = Path.replace

    def replace(self: Path, target: Any) -> Any:
        if getattr(stalled, "on", False) and not release.is_set():
            at_gate.set()
            release.wait()
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)

    def stale_rotator() -> None:
        stalled.on = True
        audit._rotate_if_needed(log_path)

    b = threading.Thread(target=stale_rotator)
    threads = [b]
    release_at_teardown(request, release, threads)
    b.start()
    assert at_gate.wait(timeout=5), "B never reached its rename"
    a = threading.Thread(target=audit._rotate_if_needed, args=(log_path,))
    threads.append(a)
    a.start()
    a.join(timeout=0.5)  # unguarded: A finishes here; guarded: A is waiting on B's lock
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("new-row\n")
    release.set()
    finish_threads(b)
    finish_threads(a)

    surviving = "".join(p.read_text(encoding="utf-8") for p in (backup, log_path) if p.exists())
    assert old_rows in surviving


def test_rotation_rechecks_size_after_taking_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#357: a writer that waited on the lock must look again before renaming.

    A rotates and a new row lands in a fresh log while B queues for the lock.
    Once B has the lock the log is small again, so the re-check must skip the
    rename; without it B renames the fresh log over ``.1`` and the old rows
    are gone. (The test above only covers the order where B holds the lock
    first.)
    """
    monkeypatch.setattr(audit, "_MAX_LOG_BYTES", 100)
    log_path = tmp_path / "tool_invocations.log.jsonl"
    backup = log_path.with_name(log_path.name + ".1")
    old_rows = "old-row-" + "x" * 200 + "\n"
    log_path.write_text(old_rows, encoding="utf-8")
    real_file_lock = log_rotation.file_lock

    @contextmanager
    def lock_taken_after_peer_rotated(path: Path, **kwargs: Any) -> Iterator[bool]:
        with real_file_lock(path, **kwargs) as acquired:
            path.replace(backup)  # what A did before releasing the lock
            path.write_text("new-row\n", encoding="utf-8")
            yield acquired

    monkeypatch.setattr(log_rotation, "file_lock", lock_taken_after_peer_rotated)

    audit._rotate_if_needed(log_path)

    assert backup.read_text(encoding="utf-8") == old_rows
    assert log_path.read_text(encoding="utf-8") == "new-row\n"


def test_rotation_still_happens_when_the_lock_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A lock that cannot be taken must not turn rotation off for good.

    On a filesystem where the lock call raises (some network mounts), skipping
    rotation lets the log grow without bound. The worst case should be the
    pre-#357 race, not no rotation at all.
    """
    monkeypatch.setattr(audit, "_MAX_LOG_BYTES", 100)
    log_path = tmp_path / "tool_invocations.log.jsonl"
    backup = log_path.with_name(log_path.name + ".1")
    old_rows = "old-row-" + "x" * 200 + "\n"
    log_path.write_text(old_rows, encoding="utf-8")

    @contextmanager
    def unavailable(path: Path, **kwargs: Any) -> Iterator[bool]:
        raise OSError(37, "No locks available")
        yield True  # pragma: no cover - makes this a generator

    monkeypatch.setattr(log_rotation, "file_lock", unavailable)

    with caplog.at_level(logging.WARNING, logger=log_rotation.logger.name):
        audit._rotate_if_needed(log_path)

    assert backup.read_text(encoding="utf-8") == old_rows
    assert not log_path.exists()
    assert "without a lock" in caplog.text


def test_failed_rename_under_the_lock_is_still_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falling back on a lock error must not also retry a rename that failed."""
    monkeypatch.setattr(audit, "_MAX_LOG_BYTES", 100)
    log_path = tmp_path / "tool_invocations.log.jsonl"
    log_path.write_text("x" * 200, encoding="utf-8")
    renames: list[Path] = []

    def boom(self: Path, target: Any) -> Any:
        renames.append(self)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", boom)

    audit._rotate_if_needed(log_path)  # must not raise

    assert len(renames) == 1
