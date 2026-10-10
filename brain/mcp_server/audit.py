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

from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

_RESULT_SUMMARY_MAX_CHARS = 140
_LOG_FILENAME = "tool_invocations.log.jsonl"
_MAX_LOG_BYTES = 1_000_000
_REDACTED = "[REDACTED]"
_OMITTED = "[OMITTED]"
_SENSITIVE_KEYS = {
    # Original (P3 baseline) — content-shaped fields
    "content",
    "message",
    "messages",
    "prompt",
    "raw",
    "response",
    "result",
    "text",
    # Audit 2026-05-07 P3-3: search terms + metadata fields can be
    # just as identifying as content. A "redacted" log shouldn't
    # leak the search query that found Hana, the title of a private
    # work she wrote, or the name of a relationship she's grieving.
    "query",
    "title",
    "summary",
    "emotion",
    "tag",
    "tags",
    "name",
    "id",
    "memory_id",
    "session_id",
    "work_id",
    # #273: a visited URL is as identifying as a search query.
    "url",
}


def _audit_mode(persona_dir: Path) -> str:
    """Return audit privacy mode from the persona config."""
    try:
        from brain.persona_config import PersonaConfig

        return PersonaConfig.load(persona_dir / "persona_config.json").mcp_audit_log_level
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load persona MCP audit config: %s", exc)
    return "redacted"


def _needs_rotation(log_path: Path) -> bool:
    return log_path.exists() and log_path.stat().st_size > _MAX_LOG_BYTES


def _rotate_if_needed(log_path: Path) -> None:
    """Keep the local audit log bounded with a single .1 backup.

    Every appender (MCP children, streaming and blocking flushes) runs this,
    so the check-then-rename is serialised on a sidecar lock and re-checked
    under it (#357): a writer that saw an oversize log must not rename the
    fresh one a concurrent rotator left behind over ``.1``. The size check
    first runs unlocked so the common no-rotation append takes no lock.
    """
    try:
        if not _needs_rotation(log_path):
            return
        with file_lock(log_path):
            if _needs_rotation(log_path):
                # replace() overwrites an existing .1 atomically on every OS.
                log_path.replace(log_path.with_name(f"{log_path.name}.1"))
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
    outcome: str | None = None,
    monologue_text: str | None = None,
    stored_image_path: str | None = None,
    origin: str | None = None,
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
    outcome:
        ``"ok"``, ``"refused"``, or ``"error"`` — mirrors tool_loop.py's
        in-process outcome semantics (#96/#102), so a write_guard denial
        (returned as ``{"error": ...}`` rather than raised) is distinguishable
        from a committed write. ``None`` for callers that don't classify it.
    stored_image_path:
        For a viewable-image read, the content-addressed rel_path
        (``images/<sha>.<ext>``). Recorded as a first-class field so it can be
        surfaced back to the engine and persisted into the durable buffer. It is
        a content HASH (deliberately non-identifying) and metadata, so — like
        ``monologue_text`` — it is written unconditionally, outside the
        privacy-mode branch, and NOT redacted. It is emitted in ``full`` /
        ``redacted`` / ``metadata`` modes; only ``off`` (which skips all logging)
        drops it. NEVER contains base64 / image bytes.
    origin:
        ``None`` for a brain-tools call made through this MCP server. The bridge
        passes ``"cli_builtin"`` for a Claude CLI built-in (WebSearch, WebFetch,
        ToolSearch, ...) it saw in the stream (#273). Metadata, written in every
        mode but ``off``; ``_read_audit_lines_since`` skips such rows so they
        stay audit-only.
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
    elif origin is not None:
        # A CLI built-in's schema is an open set the key denylist was never
        # sized for (TaskCreate.description, SendMessage.to, ...): keep the
        # keys so the trace shows what kind of call it was, hide every value.
        safe_arguments = (
            {str(k): _REDACTED for k in arguments} if isinstance(arguments, dict) else _REDACTED
        )
        safe_summary = _redact_summary(result_summary)
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
        "outcome": outcome,
    }
    if monologue_text:
        record["monologue_text"] = monologue_text
    # A content-addressed image path is a hash (metadata), never content — write
    # it like monologue_text, outside the redaction branch, so memory formation
    # can bind to the image by its content hash. Never base64.
    if stored_image_path:
        record["stored_image_path"] = stored_image_path
    if origin is not None:
        record["origin"] = origin
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
