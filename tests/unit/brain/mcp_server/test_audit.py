"""Tests for brain.mcp_server.audit."""

from __future__ import annotations

import json
from pathlib import Path

from brain.mcp_server.audit import log_invocation


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
    assert rec["arguments"] == {"query": "morning"}
    assert rec["result_summary"] == "3 hits"
    assert rec["error"] is None
    # Timestamp ends with Z (UTC)
    assert rec["timestamp"].endswith("Z")


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
