"""Audit log for MCP-server tool invocations.

Each call to a brain-tool from inside the MCP server appends one JSON line
to <persona_dir>/tool_invocations.log.jsonl. Failures here are observability,
not correctness — they are logged to stderr and swallowed so a broken disk
or full filesystem cannot break tool dispatch.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RESULT_SUMMARY_MAX_CHARS = 140
_LOG_FILENAME = "tool_invocations.log.jsonl"


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
    if len(result_summary) <= _RESULT_SUMMARY_MAX_CHARS:
        truncated = result_summary
    else:
        truncated = result_summary[:_RESULT_SUMMARY_MAX_CHARS] + "…"

    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "name": name,
        "arguments": arguments,
        "result_summary": truncated,
        "error": error,
    }

    log_path = persona_dir / _LOG_FILENAME
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("audit log write failed: %s", exc)
