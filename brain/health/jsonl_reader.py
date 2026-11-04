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
from typing import NamedTuple

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


# JSONDecodeError is a ValueError, and so is the error for an integer of more than 4300 digits.
_JSON_PARSE_ERRORS = (ValueError, RecursionError)


class LineVerdict(NamedTuple):
    """What :func:`classify_line` made of one line.

    ``kind`` is None for a usable object (``data`` is it) and for a blank line (``data`` is None too); otherwise ``utf8``,
    ``json`` or ``shape``, with ``detail`` the codec reason, the parser's message or the JSON type name, and ``exc`` the
    exception behind ``utf8`` and ``json`` (None for ``shape``).
    """

    raw: str
    text: str
    data: dict | None
    kind: str | None
    detail: str
    exc: BaseException | None

    @property
    def reason(self) -> str | None:
        """The one-line reason a bad line is bad (None for a usable or a blank line); it never carries the line's content."""
        if self.kind == "utf8":
            return f"invalid UTF-8 ({self.detail})"
        if self.kind == "json":
            return f"malformed JSON ({self.detail})"
        if self.kind == "shape":
            return f"not a JSON object ({self.detail})"
        return None


def classify_line(raw: str) -> LineVerdict:
    """The one place a JSONL line is judged: usable object, blank, or bad (and why).

    ``raw`` is the line as read, decoded with ``errors="surrogateescape"`` (a byte that is not valid UTF-8 is then a lone
    surrogate and costs only this line). The checks run in this order: invalid UTF-8, blank, JSON that does not parse
    (``ValueError``, which includes the integer digit limit, or ``RecursionError`` for a line nested too deep), a value that is
    not an object. Every JSONL reader (``_parse_lines`` here, ``json_io``'s raw reader and its line-position reader) calls
    this, so they agree on what a bad line is; each keeps its own log text and strict-mode error, and what it does with a
    line the JSON parser cannot load: the raw reader skips it and the line-position reader reports it as a bad line, while
    ``_parse_lines`` raises the parser's own exception (see there).
    """
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        return LineVerdict(raw, raw.rstrip("\r\n"), None, "utf8", exc.reason, exc)
    text = raw.rstrip("\r\n")
    if not text.strip():
        return LineVerdict(raw, text, None, None, "", None)
    try:
        data = json.loads(text)
    except _JSON_PARSE_ERRORS as exc:
        return LineVerdict(raw, text, None, "json", str(exc), exc)
    if isinstance(data, dict):
        return LineVerdict(raw, text, data, None, "", None)
    return LineVerdict(raw, text, None, "shape", type(data).__name__, None)


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
    callers that must fail closed. The default is unchanged. Each line is judged by
    :func:`classify_line`.

    One kind of line stops the read instead of being skipped, in strict mode too: a line the JSON parser
    cannot load at all (an integer past the interpreter's digit limit, nesting past the recursion limit).
    The parser's own ``ValueError`` or ``RecursionError`` comes out, as it did before the parsers shared
    :func:`classify_line` (#643).
    """
    for line_index, raw in enumerate(fh, start=1):
        verdict = classify_line(raw)
        if verdict.kind is None:
            if verdict.data is not None:
                yield verdict.data
            continue
        if verdict.kind == "json" and not isinstance(verdict.exc, json.JSONDecodeError):
            # A known defect, kept as these readers behaved before they shared classify_line (#643), not a
            # contract: one such line stops the whole read. Skipping it needs an audit of the callers that
            # rewrite, delete or take a safety set from what they read, tracked in its own issue.
            raise verdict.exc
        if strict:
            raise JsonlLineError(line_index, verdict.reason or "", path) from verdict.exc
        if verdict.kind == "utf8":
            logger.warning(
                "skipping malformed jsonl line %d in %s: invalid UTF-8 (%s) | content: %r",
                line_index,
                path,
                verdict.detail,
                raw.encode("utf-8", "surrogateescape").decode("utf-8", "replace")[:201],
            )
        elif verdict.kind == "json":
            logger.warning(
                "skipping malformed jsonl line %d in %s: %s | content: %r",
                line_index,
                path,
                verdict.detail,
                verdict.text[:201],
            )
        else:
            logger.warning(
                "skipping non-dict jsonl line %d in %s (value type=%s) | content: %r",
                line_index,
                path,
                verdict.detail,
                verdict.text[:201],
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
    lines around it, except a line the JSON parser cannot load (see
    :func:`_parse_lines`), which stops the read. Each skipped line emits a warning that includes
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
