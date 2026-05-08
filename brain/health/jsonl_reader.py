"""Append-only JSONL log reader with per-line corruption skip.

Generalises the pattern shipped in the Phase 2a hardening PR for the
growth log. Used by every ``*.log.jsonl`` reader in the brain —
heartbeats, dreams, reflex, research, growth.

Reads line-by-line off disk rather than loading the full file into a
single string + splitting. On a 500 MB log, the streaming path peaks
at roughly one line of memory; the previous
``path.read_text().splitlines()`` shape peaked at ~2× file size (the
raw text plus the list of split lines).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def iter_jsonl_skipping_corrupt(path: Path) -> Iterator[dict]:
    """Yield parsed dict lines from ``path``, skipping malformed lines.

    Streaming variant — reads one line at a time off disk so memory
    stays bounded regardless of file size. Use this for tail readers,
    large-log scans, and anywhere the caller doesn't actually need
    the full list materialised. The list-returning
    :func:`read_jsonl_skipping_corrupt` is implemented in terms of
    this generator.

    Per-line resilience: a single corrupt line never invalidates the
    lines around it. Each skipped line emits a warning that includes
    the line number, the file path, the parse exception, and a 200-
    char preview of the bad content — enough for a human to find and
    quarantine the line.

    Non-dict JSON (lists, scalars, null) is skipped because the JSONL
    contract every caller assumes is "one dict per line." Audit
    2026-05-07 P3-4 added a warning for that case so a hand-edit or
    schema-drifted line can't disappear from readers without leaving
    a trail.
    """
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line_index, raw in enumerate(fh, start=1):
            stripped = raw.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "skipping malformed jsonl line %d in %s: %s | content: %r",
                    line_index,
                    path,
                    exc,
                    stripped[:201],
                )
                continue
            if isinstance(data, dict):
                yield data
            else:
                logger.warning(
                    "skipping non-dict jsonl line %d in %s "
                    "(value type=%s) | content: %r",
                    line_index,
                    path,
                    type(data).__name__,
                    stripped[:201],
                )


def read_jsonl_skipping_corrupt(path: Path) -> list[dict]:
    """Return parsed lines from ``path`` as a list, skipping malformed ones.

    Thin list-materialising wrapper around
    :func:`iter_jsonl_skipping_corrupt` so existing callers that want
    all lines at once keep their shape. The streaming path inside
    still avoids the previous memory spike.
    """
    return list(iter_jsonl_skipping_corrupt(path))
