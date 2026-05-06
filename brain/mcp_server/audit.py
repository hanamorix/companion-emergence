"""Audit log for MCP-server tool invocations.

Each call to a brain-tool from inside the MCP server appends one JSON line
to <persona_dir>/tool_invocations.log.jsonl. Failures here are observability,
not correctness — they are logged to stderr and swallowed so a broken disk
or full filesystem cannot break tool dispatch.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RESULT_SUMMARY_MAX_CHARS = 140
_LOG_FILENAME = "tool_invocations.log.jsonl"
_MAX_LOG_BYTES = 1_000_000
_REDACTED = "[REDACTED]"
_OMITTED = "[OMITTED]"
_SENSITIVE_KEYS = {"content", "message", "messages", "prompt", "raw", "response", "result", "text"}


def _audit_mode(persona_dir: Path) -> str:
    """Return audit privacy mode from the persona config."""
    try:
        from brain.persona_config import PersonaConfig

        return PersonaConfig.load(persona_dir / "persona_config.json").mcp_audit_log_level
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load persona MCP audit config: %s", exc)
    return "redacted"


def _rotate_if_needed(log_path: Path) -> None:
    """Keep the local audit log bounded with a single .1 backup."""
    try:
        if not log_path.exists() or log_path.stat().st_size <= _MAX_LOG_BYTES:
            return
        backup = log_path.with_name(f"{log_path.name}.1")
        if backup.exists():
            backup.unlink()
        log_path.replace(backup)
    except OSError as exc:
        logger.warning("audit log rotation failed: %s", exc)


def _redact_value(value: Any, *, key: str = "") -> Any:
    """Redact user/private text fields before writing the audit log."""
    if key.lower() in _SENSITIVE_KEYS:
        return _REDACTED
    if isinstance(value, dict):
        return {str(k): _redact_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _redact_summary(summary: str) -> str:
    """Best-effort JSON summary redaction; plain summaries keep existing behavior."""
    try:
        parsed = json.loads(summary)
    except (json.JSONDecodeError, TypeError):
        return summary
    return json.dumps(_redact_value(parsed), ensure_ascii=False, default=str)


def log_invocation(
    persona_dir: Path,
    *,
    name: str,
    arguments: dict[str, Any],
    result_summary: str,
    error: str | None = None,
) -> None:
    """Append one invocation record to <persona_dir>/tool_invocations.log.jsonl.

    Never raises. OSError on the write is logged at WARNING and swallowed.

    Parameters
    ----------
    persona_dir:
        The active persona's directory; the log file is written here.
    name:
        Tool name (e.g. "search_memories").
    arguments:
        Args the LLM passed in. Will be JSON-serialised; non-JSON values
        fall through to ``default=str``.
    result_summary:
        Compact preview of the result. Truncated to 140 chars + "…" if longer.
    error:
        ``None`` on success; ``str(exc)`` on dispatch failure.
    """
    mode = _audit_mode(persona_dir)
    if mode == "off":
        return

    if mode == "full":
        safe_arguments: dict[str, Any] | str = arguments
        safe_summary = result_summary
    elif mode == "metadata":
        safe_arguments = _OMITTED
        safe_summary = ""
    else:
        safe_arguments = _redact_value(arguments)
        safe_summary = _redact_summary(result_summary)

    if len(safe_summary) <= _RESULT_SUMMARY_MAX_CHARS:
        truncated = safe_summary
    else:
        truncated = safe_summary[:_RESULT_SUMMARY_MAX_CHARS] + "…"

    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "name": name,
        "audit_level": mode,
        "arguments": safe_arguments,
        "result_summary": truncated,
        "error": error,
    }
    request_id = os.environ.get("NELL_MCP_AUDIT_REQUEST_ID")
    if request_id:
        record["request_id"] = request_id

    log_path = persona_dir / _LOG_FILENAME
    try:
        _rotate_if_needed(log_path)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("audit log write failed: %s", exc)
