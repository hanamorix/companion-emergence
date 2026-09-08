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

import gzip
import json
import logging
import os
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
                    "skipping non-dict jsonl line %d in %s (value type=%s) | content: %r",
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


def read_last_n_jsonl_lines(path: Path, n: int, *, chunk_size: int = 8192) -> list[str]:
    """Read the last n raw lines of a JSONL file via a backward seek.

    Bounded I/O: cost scales with n and average line length, not with total
    file size (#225 — the ignore-streak bounded-window redesign). Reads
    backward in byte chunks from EOF, accumulating raw BYTES (never decoding
    per-chunk, so a multi-byte UTF-8 character split across a chunk boundary
    is never corrupted — only the file's own byte sequence, concatenated
    back into original order, is ever decoded, and only once, at the end).
    Stops when either (a) the accumulated buffer contains more than n
    newlines, or (b) BOF is reached — these are tracked as DISTINCT
    conditions, not conflated: only case (a) means the read started
    mid-line (so the leading fragment is a genuine partial line and must be
    dropped); case (b) means byte 0 of the file was reached, so the first
    accumulated line is always complete and must be KEPT. Returns raw JSONL
    text lines (not yet parsed), in original file order, at most n of them
    (or fewer, if the file itself has fewer than n lines). Caller parses
    each with the same corrupt-line-skip discipline as
    ``read_jsonl_skipping_corrupt``.

    Returns ``[]`` immediately for a nonexistent path or ``n <= 0``,
    matching every other reader in this module's missing-file convention.
    """
    if n <= 0:
        return []
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        remaining = f.tell()
        block = b""
        reached_bof = False
        while block.count(b"\n") <= n:
            if remaining <= 0:
                reached_bof = True
                break
            read_size = min(chunk_size, remaining)
            remaining -= read_size
            f.seek(remaining)
            block = f.read(read_size) + block
    # Decode ONCE, on the fully-assembled byte buffer — never per chunk.
    # errors="replace" (matching brain/bridge/daemon.py:cmd_tail_log's own
    # convention) so a mangled leading fragment (dropped below when not at
    # BOF) can't raise and abort an otherwise-good read.
    text = block.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # trailing "" from a final trailing newline
    if not reached_bof and lines:
        lines.pop(0)  # genuine partial leading fragment — discard
    return lines[-n:] if len(lines) > n else lines


def iter_jsonl_streaming(path: Path) -> Iterator[dict]:
    """Stream JSONL entries from ``path``, transparently handling ``.gz``.

    Same per-line resilience as :func:`iter_jsonl_skipping_corrupt`: a
    malformed or non-dict line emits a warning and is skipped, not
    aborted on. Whether the file is gzipped is detected by the
    ``.gz`` suffix.

    Returns immediately if the path doesn't exist (no error). Use this
    when a caller needs to fan out across active + rotated archives.
    """
    if not path.exists():
        return
    is_gz = path.suffix == ".gz"
    open_fn = gzip.open if is_gz else open
    with open_fn(path, "rt", encoding="utf-8") as fh:
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
                    "skipping non-dict jsonl line %d in %s (value type=%s) | content: %r",
                    line_index,
                    path,
                    type(data).__name__,
                    stripped[:201],
                )
