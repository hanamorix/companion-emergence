"""The one TOML reader and writer of the product (#529).

Every TOML file the product reads or writes goes through this module, and ``tests/unit/test_no_direct_toml_io.py`` fails
when any other module imports a TOML library or opens a ``.toml`` path. Why one module: the settings file
(``tunables.toml``) is edited by hand and by the app, so it needs a reader that says WHICH LINE is wrong, a writer that
never half-writes it, and an editor that changes one value without touching a comment the user wrote. The same reader
serves ``persona.toml`` and ``prompt_strings.toml``, so a BOM or a transient Windows sharing violation is handled once.

Surface:

* ``read_toml(path)``: strict. Raises :class:`TomlFileError` (``kind`` is ``missing``, ``io`` or ``damaged``; a damaged file
  carries ``line``, ``column`` and ``reason``).
* ``read_toml_result(path)``: never raises; ``outcome`` is ``ok``, ``missing``, ``empty``, ``damaged`` or ``io``.
* ``read_toml_cached(path)``: strict, parsed again only when the file's ``(st_mtime_ns, st_size, st_ino)`` changes.
* ``parse_toml(text)``, ``dumps_value(value)``, ``flatten_toml(data)``, ``write_toml_text(path, data)`` (atomic; writes
  through a symbolic link), ``remove_orphan_temp_files(path)`` (the leftovers of a crashed write).
* :class:`TomlDocument`: the text of a file as lines. It edits a SINGLE-LINE ``key = value`` entry in place (only the value
  text changes; a trailing ``# comment``, every other line, the BOM and the file's own line endings stay byte for byte),
  inserts a new entry before the first ``[table]`` header, and reads or replaces the comment lines directly above an entry.

Reader: the stdlib ``tomllib`` only. Writer: stdlib only; the flat ``key = value`` plus comments shape needs no TOML
library, and a third-party one would be an extra install on every platform. ``tomllib`` is imported here and nowhere else.

Not handled on purpose: an entry whose value spans several lines, and an inline table, are READ normally but never edited
(the editor refuses with a message naming the key). That keeps the editor small and cannot corrupt a value it does not
understand.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import secrets
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from brain.health.attempt_heal import _replace_with_retry
from brain.health.json_io import _read_bytes

_BOM = b"\xef\xbb\xbf"
_REASON_CAP = 120
_POSITION_TAIL = re.compile(r"\s*\(at line (\d+), column (\d+)\)\s*$")
_END_OF_DOCUMENT = "(at end of document)"


class TomlFileError(OSError, ValueError):
    """A TOML file or value could not be used.

    ``kind`` is ``missing`` (no such file), ``io`` (the file could not be read), ``damaged`` (it is not valid TOML or not
    UTF-8) or ``unsupported`` (a value that cannot be written, or an entry the editor will not change). For ``damaged``
    ``line`` and ``column`` (1-based, None when the parser gave none) and ``reason`` (the parser's short text) say where.
    An OSError and a ValueError, so a handler written for either still catches it.
    """

    def __init__(
        self,
        path: Path | str | None,
        kind: str,
        reason: str = "",
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        where = f"{path}" if path is not None else "TOML"
        pos = ""
        if line is not None:
            pos = f"line {line}" + (f", column {column}" if column is not None else "")
        parts = [where] + ([pos] if pos else []) + ([reason] if reason else [kind])
        super().__init__(": ".join(parts))
        self.path = Path(path) if path is not None else None
        self.kind = kind
        self.reason = reason
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return str(self.args[0])

    def __reduce__(self):
        return (type(self), (self.path, self.kind, self.reason, self.line, self.column))


@dataclass(frozen=True)
class TomlRead:
    """What ``read_toml_result`` found. ``data`` is ``{}`` unless ``outcome`` is ``ok``."""

    outcome: Literal["ok", "missing", "empty", "damaged", "io"]
    data: dict[str, Any]
    raw: bytes = b""
    kind: str | None = None
    reason: str = ""
    line: int | None = None
    column: int | None = None
    path: Path | None = None


# --------------------------------------------------------------------------------------------------
# Errors with a position
# --------------------------------------------------------------------------------------------------


def _clean_reason(reason: str) -> str:
    cleaned = "".join(ch for ch in reason if ch.isprintable())
    return cleaned[:_REASON_CAP]


def _split_position(message: str) -> tuple[int | None, int | None, str]:
    """``(line, column, reason)`` from a parser message ending in ``(at line N, column M)`` (Python 3.12 and 3.13)."""
    m = _POSITION_TAIL.search(message)
    if not m:
        return None, None, message
    return int(m.group(1)), int(m.group(2)), message[: m.start()].rstrip()


def _position_of(exc: BaseException, text: str) -> tuple[int | None, int | None, str]:
    """The parser's ``(line, column, reason)``: its attributes when it has them (Python 3.14), else the message tail."""
    line, col, msg = getattr(exc, "lineno", None), getattr(exc, "colno", None), getattr(exc, "msg", None)
    if isinstance(line, int) and isinstance(col, int) and isinstance(msg, str):
        return line, col, msg
    line, col, reason = _split_position(str(exc))
    if line is None and reason.endswith(_END_OF_DOCUMENT):
        reason = reason[: -len(_END_OF_DOCUMENT)].rstrip()
        rows = text.split("\n")
        return len(rows), len(rows[-1]) + 1, reason
    return line, col, reason


def _parse_text(text: str) -> dict[str, Any]:
    return tomllib.loads(text)


def parse_toml(text: str, *, path: Path | str | None = None) -> dict[str, Any]:
    """Parse TOML text. Raises :class:`TomlFileError` (``damaged``, with line and column when known)."""
    try:
        return _parse_text(text)
    except (ValueError, RecursionError) as exc:
        line, col, reason = _position_of(exc, text)
        raise TomlFileError(path, "damaged", _clean_reason(reason or "not valid TOML"), line, col) from exc


def _decode(raw: bytes, path: Path | None) -> str:
    body = raw[len(_BOM) :] if raw.startswith(_BOM) else raw
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = body.count(b"\n", 0, exc.start) + 1
        col = exc.start - (body.rfind(b"\n", 0, exc.start) + 1) + 1
        raise TomlFileError(path, "damaged", "not valid UTF-8", line, col) from exc


def _result_from_raw(raw: bytes, path: Path) -> TomlRead:
    body = raw[len(_BOM) :] if raw.startswith(_BOM) else raw
    if body == b"":
        return TomlRead("empty", {}, raw, path=path)
    try:
        text = _decode(raw, path)
        data = parse_toml(text, path=path)
    except TomlFileError as exc:
        return TomlRead("damaged", {}, raw, "damaged", exc.reason, exc.line, exc.column, path)
    return TomlRead("ok", data, raw, path=path)


def read_toml_result(path: Path | str) -> TomlRead:
    """Read and parse ``path`` without ever raising. A Windows sharing violation is retried like ``json_io``'s reads."""
    path = Path(path)
    try:
        raw, _sig = _read_bytes(path)
    except FileNotFoundError:
        return TomlRead("missing", {}, kind="missing", reason="no such file", path=path)
    except OSError as exc:
        return TomlRead("io", {}, kind="io", reason=_clean_reason(str(exc)), path=path)
    return _result_from_raw(raw, path)


def read_toml(path: Path | str) -> dict[str, Any]:
    """Read and parse ``path``. Raises :class:`TomlFileError` (``missing``, ``io`` or ``damaged``). A zero-byte file is ``{}``."""
    res = read_toml_result(path)
    if res.outcome in ("ok", "empty"):
        return res.data
    raise TomlFileError(res.path, res.kind or "io", res.reason, res.line, res.column)


_cache_lock = threading.Lock()
_cache: dict[Path, tuple[tuple[int, int, int], dict[str, Any]]] = {}


def read_toml_cached(path: Path | str) -> dict[str, Any]:
    """``read_toml`` that parses again only when the file's ``(st_mtime_ns, st_size, st_ino)`` changes. Thread-safe.

    The dict is shared between callers: treat it as read only. A missing or damaged file raises every time (nothing is
    cached for a failure).
    """
    path = Path(path)
    with _cache_lock:
        try:
            st = path.stat()
        except FileNotFoundError as exc:
            raise TomlFileError(path, "missing", "no such file") from exc
        except OSError as exc:
            raise TomlFileError(path, "io", _clean_reason(str(exc))) from exc
        sig = (st.st_mtime_ns, st.st_size, st.st_ino)
        hit = _cache.get(path)
        if hit is not None and hit[0] == sig:
            return hit[1]
        data = read_toml(path)
        _cache[path] = (sig, data)
        return data


def _reset_cache_for_tests() -> None:
    with _cache_lock:
        _cache.clear()


# --------------------------------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------------------------------

_ESCAPES = {"\\": "\\\\", '"': '\\"', "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def _dumps_string(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:  # a lone surrogate cannot be written to a UTF-8 file
        raise TomlFileError(None, "unsupported", "a string with a lone surrogate cannot be written") from exc
    out = []
    for ch in value:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20 or ch == "\x7f":
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def dumps_value(value: Any) -> str:
    """One TOML value on one line.

    ``bool`` (tested before ``int``), ``int`` of any size, ``float`` (including ``-0.0``, ``inf``, ``nan``), ``str`` (basic
    string, every control character and DEL escaped) and ``list``/``tuple`` of those. Anything else, and a string with a
    lone surrogate, raises :class:`TomlFileError` (``unsupported``).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return _dumps_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(dumps_value(v) for v in value) + "]"
    raise TomlFileError(None, "unsupported", f"a {type(value).__name__} cannot be written as a TOML value")


def flatten_toml(data: dict[str, Any]) -> dict[str, Any]:
    """Nested tables and dotted keys as one dict of dotted keys; a list stays whole.

    A key written two ways (the flat quoted ``"a.b"`` and the nested ``[a]`` ``b``) resolves to the SHALLOWEST spelling,
    so the flat quoted entry wins. Deterministic, and the same rule the editor uses to call the key ambiguous.
    """
    best: dict[str, tuple[int, Any]] = {}

    def walk(node: dict[str, Any], prefix: tuple[str, ...]) -> None:
        for k, v in node.items():
            path = (*prefix, k)
            if isinstance(v, dict):
                walk(v, path)
                continue
            key = ".".join(path)
            cur = best.get(key)
            if cur is None or len(path) < cur[0]:
                best[key] = (len(path), v)

    walk(data, ())
    return {k: v for k, (_d, v) in best.items()}


def same_value(a: Any, b: Any) -> bool:
    """Equality with types (``1``, ``1.0`` and ``True`` differ), NaN equal to NaN, a list equal to a tuple."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


# --------------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------------


def _write_target(path: Path) -> Path:
    """Where a write to ``path`` really lands: a symbolic link to an existing file (a dotfile manager's) is written THROUGH, so
    the link survives; anything else, including a dangling link, is the path itself."""
    try:
        if path.is_symlink() and path.exists():
            return Path(os.path.realpath(path))
    except OSError:
        pass
    return path


_TEMP_SUFFIX = re.compile(r"\.\d+\.[0-9a-f]{8}\.tmp$")


def remove_orphan_temp_files(path: Path | str) -> int:
    """Delete the temp files a crashed :func:`write_toml_text` left beside ``path``; return how many went.

    A write crashing between its temp write and the replace leaves ``.<name>.<pid>.<8 hex>.tmp``. Only that exact shape for
    THIS file is removed. The caller holds the file's lock (every writer of the settings file does), so no live write
    owns a temp file at that moment. Best effort: a file that cannot be removed is skipped.
    """
    target = _write_target(Path(path))
    prefix = f".{target.name}"
    removed = 0
    try:
        names = [c for c in target.parent.iterdir() if c.name.startswith(prefix) and _TEMP_SUFFIX.fullmatch(c.name[len(prefix) :])]
    except OSError:
        return 0
    for orphan in names:
        try:
            orphan.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def write_toml_text(path: Path | str, data: bytes, *, fsync: bool = True) -> None:
    """Atomically replace ``path`` with ``data`` (bytes: no newline translation, a BOM is kept as given).

    A temp file in the same directory is written, flushed and (by default) fsynced BEFORE the replace, and the replace goes
    through ``json_io``'s own ``_replace_with_retry`` (a transient Windows sharing violation is retried). The temp file is
    removed when anything fails, so a reader sees the old file or the new one whole, never a torn one. A symbolic link to an
    existing file is written through (the link stays a link; the temp file is made beside the target).
    """
    path = _write_target(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


# --------------------------------------------------------------------------------------------------
# Lexing one line (just enough to tell where a statement starts and where a value ends)
# --------------------------------------------------------------------------------------------------


def _skip_string(s: str, i: int) -> int:
    """Index after the single-line string that starts at ``s[i]`` (its closing quote), or ``len(s)``."""
    q = s[i]
    i += 1
    while i < len(s):
        if q == '"' and s[i] == "\\":
            i += 2
            continue
        if s[i] == q:
            return i + 1
        i += 1
    return len(s)


def _find_multiline_close(s: str, start: int, delim: str) -> int:
    """Index after the closing ``delim`` of a multi-line string (and up to two extra quotes), or -1 when it is not on this line."""
    i = start
    while i < len(s):
        if delim[0] == '"' and s[i] == "\\":
            i += 2
            continue
        if s.startswith(delim, i):
            i += 3
            extra = 0
            while i < len(s) and s[i] == delim[0] and extra < 2:
                i += 1
                extra += 1
            return i
        i += 1
    return -1


def _lex_line(s: str, depth: int, ml: str | None) -> tuple[int, str | None, int | None]:
    """Walk one line: ``(bracket depth after, open multi-line string delimiter after, index of a comment's ``#`` or None)``."""
    i = 0
    if ml is not None:
        j = _find_multiline_close(s, 0, ml)
        if j < 0:
            return depth, ml, None
        i, ml = j, None
    while i < len(s):
        c = s[i]
        if c == "#":
            return depth, None, i
        if c in "\"'":
            if s.startswith(c * 3, i):
                j = _find_multiline_close(s, i + 3, c * 3)
                if j < 0:
                    return depth, c * 3, None
                i = j
                continue
            i = _skip_string(s, i)
            continue
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        i += 1
    return depth, None, None


class _UnscannableError(Exception):
    """The line is valid TOML but not in a shape the single-line editor understands."""


_BARE_KEY_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")


def _split_key(code: str) -> int:
    """Index of the ``=`` that ends the (possibly dotted, possibly quoted) key at the start of ``code``."""
    i, n = 0, len(code)
    while True:
        while i < n and code[i] in " \t":
            i += 1
        if i >= n:
            raise _UnscannableError("no key")
        if code[i] in "\"'":
            i = _skip_string(code, i)
        else:
            j = i
            while j < n and code[j] in _BARE_KEY_CHARS:
                j += 1
            if j == i:
                raise _UnscannableError("bad key")
            i = j
        while i < n and code[i] in " \t":
            i += 1
        if i < n and code[i] == ".":
            i += 1
            continue
        if i < n and code[i] == "=":
            return i
        raise _UnscannableError("no equals sign")


def _value_end(code: str, start: int) -> int:
    """Index after the single-line value that starts at ``code[start]``; raises for a multi-line string or an inline table."""
    if start >= len(code):
        raise _UnscannableError("no value")
    c = code[start]
    if c in "\"'":
        if code.startswith(c * 3, start):
            raise _UnscannableError("multi-line string")
        return _skip_string(code, start)
    if c == "{":
        raise _UnscannableError("inline table")
    if c == "[":
        depth, i = 0, start
        while i < len(code):
            ch = code[i]
            if ch in "\"'":
                if code.startswith(ch * 3, i):
                    raise _UnscannableError("multi-line string in an array")
                i = _skip_string(code, i)
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        raise _UnscannableError("unclosed array")
    i = start
    while i < len(code) and code[i] not in " \t#":
        i += 1
    return i


def _chain(parsed: dict[str, Any]) -> tuple[str, ...]:
    """The key path of a one-statement parse: follow single-key nested dicts down to the leaf (or to an empty table)."""
    path: list[str] = []
    node: Any = parsed
    while isinstance(node, dict) and len(node) == 1:
        (k, node), = node.items()
        path.append(k)
    if isinstance(node, dict) and node:
        raise _UnscannableError("more than one key on a line")
    return tuple(path)


@dataclass
class _Line:
    content: str
    term: str  # "\n", "\r\n" or "" (the last line of a file with no final newline)


@dataclass(frozen=True)
class _Entry:
    index: int
    value_start: int
    value_end: int
    simple: bool


@dataclass
class _Scan:
    entries: dict[str, list[_Entry]]
    first_header: int | None


class TomlDocument:
    """A TOML file as lines, edited without disturbing anything the editor was not asked to change.

    Loaded from bytes (a leading BOM and the file's own line ending are remembered), edited, and turned back into bytes
    with :meth:`to_bytes`. See the module docstring for what is and is not editable.
    """

    def __init__(self, lines: list[_Line], bom: bool, eol: str, path: Path | None = None) -> None:
        self._lines = lines
        self._bom = bom
        self._eol = eol
        self._path = path
        self._scan_cache: _Scan | None = None
        self._values_cache: dict[str, Any] | None = None

    # -- construction -------------------------------------------------------------------------------------
    @classmethod
    def from_bytes(cls, raw: bytes, path: Path | None = None) -> TomlDocument:
        """Parse ``raw`` (raises :class:`TomlFileError`, ``damaged``, with the failing line) and keep it as lines."""
        text = _decode(raw, path)
        parse_toml(text, path=path)
        pieces = text.split("\n")
        eol = "\r\n" if len(pieces) > 1 and pieces[0].endswith("\r") else "\n"
        lines: list[_Line] = []
        for k, piece in enumerate(pieces):
            if k == len(pieces) - 1:
                if piece:
                    lines.append(_Line(piece, ""))
            elif piece.endswith("\r"):
                lines.append(_Line(piece[:-1], "\r\n"))
            else:
                lines.append(_Line(piece, "\n"))
        return cls(lines, raw.startswith(_BOM), eol, path)

    # -- output -------------------------------------------------------------------------------------------
    def text(self) -> str:
        return "".join(ln.content + ln.term for ln in self._lines)

    def to_bytes(self) -> bytes:
        return (_BOM if self._bom else b"") + self.text().encode("utf-8")

    def _changed(self) -> None:
        self._scan_cache = None
        self._values_cache = None

    # -- reading ------------------------------------------------------------------------------------------
    def values(self) -> dict[str, Any]:
        """Every value, as dotted keys (the shallowest spelling wins when a key is written two ways)."""
        if self._values_cache is None:
            self._values_cache = flatten_toml(parse_toml(self.text(), path=self._path))
        return self._values_cache

    def verify(self, expected: dict[str, Any]) -> bool:
        """Does the text parse, and do its values equal ``expected`` exactly (typed, NaN-aware)?"""
        try:
            got = flatten_toml(parse_toml(self.text(), path=self._path))
        except TomlFileError:
            return False
        return got.keys() == expected.keys() and all(same_value(got[k], expected[k]) for k in got)

    def _scan(self) -> _Scan:
        if self._scan_cache is not None:
            return self._scan_cache
        entries: dict[str, list[_Entry]] = {}
        first_header: int | None = None
        prefix: tuple[str, ...] | None = ()
        depth, ml = 0, None
        for idx, ln in enumerate(self._lines):
            at_start = depth == 0 and ml is None
            new_depth, new_ml, cpos = _lex_line(ln.content, depth, ml)
            if at_start:
                code = ln.content if cpos is None else ln.content[:cpos]
                stripped = code.strip()
                if stripped.startswith("["):
                    if first_header is None:
                        first_header = idx
                    try:
                        prefix = None if stripped.startswith("[[") else _chain(_parse_text(stripped))
                    except (ValueError, RecursionError, _UnscannableError):
                        prefix = None
                elif stripped:
                    single = new_depth == 0 and new_ml is None
                    try:
                        eq = _split_key(code)
                        chain = _chain(_parse_text(code[:eq] + " = 0"))
                    except (ValueError, RecursionError, _UnscannableError):
                        chain = None
                    if chain is not None and prefix is not None:
                        key = ".".join((*prefix, *chain[:-1], chain[-1]))
                        start = eq + 1
                        while start < len(code) and code[start] in " \t":
                            start += 1
                        entry = _Entry(idx, start, start, False)
                        if single:
                            try:
                                end = _value_end(code, start)
                                if code[end:].strip() == "":
                                    _parse_text("x = " + code[start:end])
                                    entry = _Entry(idx, start, end, True)
                            except (ValueError, RecursionError, _UnscannableError):
                                pass
                        entries.setdefault(key, []).append(entry)
            depth, ml = new_depth, new_ml
        self._scan_cache = _Scan(entries, first_header)
        return self._scan_cache

    def state(self, key: str) -> Literal["absent", "single", "ambiguous", "unlocatable"]:
        """``absent`` (no value for the key), ``single`` (one editable single-line entry), ``ambiguous`` (written two ways)
        or ``unlocatable`` (it has a value but no single-line entry: several lines, or inside an inline table)."""
        if key not in self.values():
            return "absent"
        found = self._scan().entries.get(key, [])
        if len(found) > 1:
            return "ambiguous"
        if len(found) == 1 and found[0].simple:
            return "single"
        return "unlocatable"

    def _single(self, key: str) -> _Entry:
        state = self.state(key)
        if state == "single":
            return self._scan().entries[key][0]
        reason = {
            "absent": f"{key} is not in the file",
            "ambiguous": f"{key} is written in two spellings; edit the file by hand",
            "unlocatable": f"{key} spans several lines or is an inline table; edit the file by hand",
        }[state]
        raise TomlFileError(self._path, "unsupported", reason)

    # -- editing ------------------------------------------------------------------------------------------
    def set_value(self, key: str, value: Any) -> bool:
        """Replace the value text of the single-line entry for ``key`` in place. True when the line changed."""
        entry = self._single(key)
        text = dumps_value(value)
        ln = self._lines[entry.index]
        new = ln.content[: entry.value_start] + text + ln.content[entry.value_end :]
        if new == ln.content:
            return False
        ln.content = new
        self._changed()
        return True

    def insert(self, key: str, value: Any, comment_lines: list[str] | tuple[str, ...] = ()) -> None:
        """Add ``key = value`` (a quoted key) after ``comment_lines``, before the first ``[table]`` header or at the end."""
        if self.state(key) != "absent":
            raise TomlFileError(self._path, "unsupported", f"{key} is already in the file")
        for c in comment_lines:
            if not c.lstrip().startswith("#"):
                raise TomlFileError(self._path, "unsupported", "a comment line must start with #")
        block = [*comment_lines, f"{dumps_value(key)} = {dumps_value(value)}"]
        eol = self._eol
        new_lines = [_Line(c, eol) for c in block]
        header = self._scan().first_header
        if header is None:
            if self._lines:
                if self._lines[-1].term == "":
                    self._lines[-1].term = eol
                if self._lines[-1].content.strip() != "":
                    self._lines.append(_Line("", eol))
            self._lines.extend(new_lines)
        else:
            at = header
            while at > 0 and self._lines[at - 1].content.lstrip().startswith("#"):
                at -= 1
            insert = [*new_lines, _Line("", eol)]
            if at > 0 and self._lines[at - 1].content.strip() != "":
                insert.insert(0, _Line("", eol))
            self._lines[at:at] = insert
        self._changed()

    def _comment_run(self, key: str) -> tuple[int, int]:
        entry = self._single(key)
        start = entry.index
        while start > 0 and self._lines[start - 1].content.lstrip().startswith("#"):
            start -= 1
        return start, entry.index

    def comment_above(self, key: str) -> list[str]:
        """The contiguous ``#`` lines directly above the key's entry (a blank line breaks the run)."""
        if self.state(key) != "single":
            return []
        start, end = self._comment_run(key)
        return [ln.content for ln in self._lines[start:end]]

    def set_comment_above(self, key: str, lines: list[str]) -> bool:
        """Replace the contiguous comment run above the key's entry with ``lines``. True when the text changed."""
        start, end = self._comment_run(key)
        old = self._lines[start:end]
        if [ln.content for ln in old] == list(lines):
            return False
        for c in lines:
            if not c.lstrip().startswith("#"):
                raise TomlFileError(self._path, "unsupported", "a comment line must start with #")
        new = [_Line(c, old[i].term if i < len(old) else self._eol) for i, c in enumerate(lines)]
        self._lines[start:end] = new
        self._changed()
        return True
