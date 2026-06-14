# brain/files/audit.py
"""write_audit.jsonl — append-only record of every write step."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def audit(persona_dir: Path, *, event: str, id: str, op: str, path: str,
          content_sha: str = "", outcome: str = "", error: str | None = None) -> None:
    try:
        persona_dir.mkdir(parents=True, exist_ok=True)
        with (persona_dir / "write_audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(UTC).isoformat(), "event": event, "id": id, "op": op,
                "path": path, "content_sha": content_sha, "outcome": outcome, "error": error,
            }) + "\n")
    except OSError:
        logger.warning("write_audit append failed", exc_info=True)
