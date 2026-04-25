"""Append-only JSONL log reader with per-line corruption skip.

Generalises the pattern shipped in the Phase 2a hardening PR for the growth log.
Used by every *.log.jsonl reader in the brain — heartbeats, dreams, reflex,
research, growth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_jsonl_skipping_corrupt(path: Path) -> list[dict]:
    """Return parsed lines from `path`, skipping malformed lines with a WARNING.

    Per-line resilience: a single corrupt line never invalidates the lines
    around it. Each skipped line emits a warning that includes the line
    number, the file path, the parse exception, and a 200-char preview of
    the bad content — enough for a human to find and quarantine the line.
    """
    if not path.exists():
        return []
    out: list[dict] = []
    for line_index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "skipping malformed jsonl line %d in %s: %s | content: %r",
                line_index,
                path,
                exc,
                raw[:201],
            )
            continue
        if isinstance(data, dict):
            out.append(data)
    return out
