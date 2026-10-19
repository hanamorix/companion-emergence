# brain/files/audit.py
"""write_audit.jsonl — append-only record of every write step."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.health.json_io import append_jsonl

logger = logging.getLogger(__name__)


def audit(persona_dir: Path, *, event: str, id: str, op: str, path: str,
          content_sha: str = "", outcome: str = "", error: str | None = None) -> bool:
    """Append one row. Fail-soft; returns whether the row was actually written (#346)."""
    return (
        append_jsonl(
            persona_dir / "write_audit.jsonl",
            {
                "ts": datetime.now(UTC).isoformat(), "event": event, "id": id, "op": op,
                "path": path, "content_sha": content_sha, "outcome": outcome, "error": error,
            },
            on_error="log",
            nonfinite="null",
        )
        is not None
    )
