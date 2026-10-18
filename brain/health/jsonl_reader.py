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


class JsonlLineError(ValueError):
    """A bad line met by ``_parse_lines(..., strict=True)``: carries the 1-based line number.

    Defined here, not in ``brain.health.json_io``, so this module never imports it.
    """

    def __init__(self, line: int, reason: str, path: Path) -> None:
        super().__init__(f"{path}: line {line}: {reason}")
        self.line = line
        self.reason = reason
        self.path = path


def _parse_lines(fh, path: Path, *, strict: bool = False) -> Iterator[dict]:
    """Parse an open text stream of JSONL, skipping bad lines with a warning.

    The stream must be opened with ``errors="surrogateescape"``: a line whose
    bytes are not valid UTF-8 (a crash mid-append can cut a multi-byte
    character, and the next append lands on the same line) then decodes with
    lone surrogates instead of raising, so it costs only that line, like a JSON
    error (#433). Strict decoding raised out of the loop and failed every read.
    Text mode keeps the line splitting (\n, \r\n, \r) and the streaming read.

    ``strict=True`` (keyword-only, added for brain.health.json_io) raises
    :class:`JsonlLineError` on the first bad line instead of skipping it, for
    callers that must fail closed. The default is unchanged.
    """
    for line_index, raw in enumerate(fh, start=1):
        try:
            raw.encode("utf-8")
        except UnicodeEncodeError as exc:
            if strict:
                raise JsonlLineError(line_index, f"invalid UTF-8 ({exc.reason})", path) from exc
            logger.warning(
                "skipping malformed jsonl line %d in %s: invalid UTF-8 (%s) | content: %r",
                line_index,
                path,
                exc.reason,
                raw.encode("utf-8", "surrogateescape").decode("utf-8", "replace")[:201],
            )
            continue
        stripped = raw.rstrip("\r\n")
        if not stripped.strip():
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if strict:
                raise JsonlLineError(line_index, f"malformed JSON ({exc})", path) from exc
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
            if strict:
                raise JsonlLineError(line_index, f"not a JSON object ({type(data).__name__})", path)
            logger.warning(
                "skipping non-dict jsonl line %d in %s (value type=%s) | content: %r",
                line_index,
                path,
                type(data).__name__,
                stripped[:201],
            )


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
    with open(path, encoding="utf-8", errors="surrogateescape") as fh:
        yield from _parse_lines(fh, path)
