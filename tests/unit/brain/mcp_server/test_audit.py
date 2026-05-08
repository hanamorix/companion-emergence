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
