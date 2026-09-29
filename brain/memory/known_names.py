"""The known-names list: a running list of name words, and the matcher over it.

Name-recall fix, increment N1 (spec section 5, S19, S26, S37, S47, S49, S66,
S70). The kindled's recall protects a word she has been told is a name (or has
learned is one), whatever the stopword and length rules would otherwise do to
it. The list lives in its own small SQLite file in the persona directory
(``known_names.db``): not memories.db, not keyed to memories, and it travels
with the persona directory.

One row per name: ``name_lower`` (the lower-cased form, words split on the
recall tokenizer's word boundary and joined by single spaces), ``display`` (as
extracted), ``source`` (``gate`` / ``reappraiser`` / ``tool``) and
``first_seen`` (UTC ISO time).

Concurrency:

* Writers (:func:`admit_names`) run under the existing OS file lock
  (``brain.utils.file_lock``) and one short SQLite transaction. The file uses
  the rollback journal (``journal_mode=DELETE``), never WAL, so a committed
  write changes the file's modification time AND the SQLite header's file
  change counter (S49, S66). A writer renames the file aside (never deletes
  it) only when SQLite reports a non-busy ``DatabaseError`` and
  ``PRAGMA quick_check`` confirms the file is not a healthy database; a busy
  or locked file is never renamed.
* The reader (:func:`load_known_names`) is on the recall hot path: it reads the
  rows once per process and re-reads only when the file's signature changes.
  The signature is ``(st_mtime_ns, st_size, st_ino, change_counter)`` and is
  taken BEFORE the row read, so a write committed after it can only cause one
  harmless extra re-read, never a stale cache. Some filesystems have coarse
  modification times, and a write can leave the size and inode unchanged (a row
  added inside an existing page), which is why the header change counter (bytes
  24-27 of the SQLite header, big-endian) is part of the signature. A locked
  file (a writer mid-commit, or a hot journal the read-only reader cannot roll
  back) returns the previous list and stores no new signature, so it is retried
  on the next call; a locked file is never mistaken for a corrupt one. A
  corrupt or missing file is an empty list.

Matching (:func:`match_known_names`) tests the message's RAW words, before the
recall selector's stopword and length rules. A multi-word name matches only as
a window of consecutive raw words, and :func:`names_fts_query` sends matched
names to the keyword search as FTS phrases.

Admission (:func:`admit_names`, S70): the one write entry point for every
writer. An entry whose whole lower-cased form is a recall stopword is rejected
(``will``, ``the``); every other extracted entry is admitted as extracted.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain import dev_constants
from brain.memory.recall_stopwords import RECALL_STOPWORDS
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

__all__ = [
    "SOURCES",
    "EMPTY",
    "KnownNames",
    "admit_names",
    "known_names_path",
    "load_known_names",
    "match_known_names",
    "names_fts_query",
    "normalize_name",
]

#: The allowed ``source`` values (S26).
SOURCES: tuple[str, ...] = ("gate", "reappraiser", "tool")

# The recall tokenizer's word boundary (`brain.chat.prompt._extract_recall_tokens`
# matches runs of ASCII letters and digits). Non-Latin and space-less scripts
# are out of scope (#317).
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# SQLite header: the 4-byte big-endian file change counter is at offset 24, so
# the first 28 bytes hold it. Valid because the file is journal_mode=DELETE.
_HEADER_LEN = 28
_COUNTER_OFFSET = 24

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS known_names ("
    " name_lower TEXT PRIMARY KEY,"
    " display TEXT NOT NULL,"
    " source TEXT NOT NULL CHECK (source IN (" + ", ".join(f"'{s}'" for s in SOURCES) + ")),"
    " first_seen TEXT NOT NULL)"
)

_SQLITE_BUSY = 5
_SQLITE_LOCKED = 6


@dataclass(frozen=True)
class KnownNames:
    """An immutable snapshot of the list, shaped for the matcher."""

    names: frozenset[str]
    #: The distinct word counts among the names, ascending (a matcher only
    #: builds windows of these lengths).
    word_counts: tuple[int, ...]

    @classmethod
    def from_names(cls, names: Iterable[str]) -> KnownNames:
        clean = frozenset(n for n in (normalize_name(x) for x in names) if n)
        counts = tuple(sorted({len(n.split(" ")) for n in clean}))
        return cls(names=clean, word_counts=counts)

    def __bool__(self) -> bool:
        return bool(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and normalize_name(name) in self.names


EMPTY = KnownNames(names=frozenset(), word_counts=())


def known_names_path(persona_dir: Path | str) -> Path:
    """The known-names file for a persona directory."""
    return Path(persona_dir) / dev_constants.KNOWN_NAMES_DB_FILENAME


def normalize_name(name: str) -> str:
    """Lower-cased form: words on the recall tokenizer's boundary, single-spaced.

    ``"New York"`` -> ``"new york"``; ``"O'Brien"`` -> ``"o brien"`` (the message
    ``O'Brien`` tokenizes to the same two words). No stopword or length filter
    (S47): admission is :func:`admit_names`'s job, not this function's.
    """
    return " ".join(w.lower() for w in _WORD_RE.findall(name))


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def match_known_names(text: str, known: KnownNames) -> list[str]:
    """The listed names that occur in ``text``, lower-cased, in order of occurrence.

    Tested on the RAW words (``[A-Za-z0-9]+`` runs, lower-cased) before any
    stopword or length rule (S47), so a listed 2-letter name or a listed name
    that is also a stopword still matches. A multi-word name matches only as a
    window of consecutive raw words: ``new york`` matches "in New York" and does
    not match "new dress ... york". A name is reported once however often it
    occurs.
    """
    if not text or not known.names:
        return []
    words = [w.lower() for w in _WORD_RE.findall(text)]
    found: dict[str, None] = {}
    total = len(words)
    for i in range(total):
        for n in known.word_counts:
            if i + n > total:
                break
            candidate = " ".join(words[i : i + n])
            if candidate in known.names:
                found.setdefault(candidate)
    return list(found)


def names_fts_query(names: Sequence[str]) -> str:
    """An FTS5 MATCH expression: the names as quoted phrases, OR-joined.

    ``["pretzel", "new york"]`` -> ``'"pretzel" OR "new york"'``. Each name is
    letters, digits and single spaces only (it came from :func:`normalize_name`),
    so a phrase needs no escaping and a multi-word name matches only as
    consecutive tokens. ``""`` when there are no names.
    """
    return " OR ".join(f'"{n}"' for n in names if n)


# ---------------------------------------------------------------------------
# Reader (hot path)
# ---------------------------------------------------------------------------

_Signature = tuple[int, int, int, "int | None"]


class _SignatureUnavailableError(Exception):
    """The file's stat or header could not be read for a reason other than absence."""


class _TransientError(Exception):
    """The row read hit a locked file (or a hot journal): keep the previous list.

    ``busy`` is True for an ordinary writer-mid-commit lock (expected, debug
    level) and False for anything else the read-only reader cannot get past (a
    hot journal, an unopenable path): those are logged once at warning level so
    name protection going quiet is visible.
    """

    def __init__(self, message: str, *, busy: bool = False) -> None:
        super().__init__(message)
        self.busy = busy


_cache_lock = threading.Lock()
# path string -> (signature the rows were read under, the rows). The signature
# is None for a missing file.
_cache: dict[str, tuple[_Signature | None, KnownNames]] = {}
# (path, message) pairs already logged at warning level (one line per distinct
# problem, not one per recall turn). Guarded by _cache_lock.
_warned: set[tuple[str, str]] = set()


def _reset_cache() -> None:
    """Drop every cached list (test isolation)."""
    with _cache_lock:
        _cache.clear()
        _warned.clear()


def _signature(path: Path) -> _Signature | None:
    """``(st_mtime_ns, st_size, st_ino, change_counter)``, or None for a missing file.

    ``change_counter`` is bytes 24-27 of the SQLite header (big-endian), or None
    when the file is shorter than the header (a plain read of the first 28
    bytes; it never opens a SQLite connection). Raises
    :class:`_SignatureUnavailableError` for any other OS error, so the caller keeps
    its previous list instead of caching a guess.
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _SignatureUnavailableError(str(exc)) from exc
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEADER_LEN)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _SignatureUnavailableError(str(exc)) from exc
    counter: int | None = None
    if len(head) >= _HEADER_LEN:
        counter = int.from_bytes(head[_COUNTER_OFFSET:_HEADER_LEN], "big")
    return (st.st_mtime_ns, st.st_size, st.st_ino, counter)


def _is_locked(exc: sqlite3.OperationalError) -> bool:
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        return (code & 0xFF) in (_SQLITE_BUSY, _SQLITE_LOCKED)
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def _read_rows(path: Path) -> KnownNames:
    """Read the list through a read-only connection with no busy wait.

    Never creates the file. A locked file, a hot journal a read-only connection
    cannot roll back, or a file that vanished mid-read raises
    :class:`_TransientError` (previous list kept). A file with no table yet is an
    empty list; a file SQLite reports as corrupt (a non-Operational
    ``DatabaseError``) is an empty list.
    """
    uri = path.absolute().as_uri() + "?mode=ro"
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=0)
        rows = conn.execute("SELECT name_lower FROM known_names").fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return EMPTY
        raise _TransientError(str(exc), busy=_is_locked(exc)) from exc
    except sqlite3.DatabaseError as exc:
        logger.warning(
            "known names: %s is not a readable database (%s); reading as empty", path, exc
        )
        return EMPTY
    finally:
        if conn is not None:
            conn.close()
    return KnownNames.from_names(r[0] for r in rows if isinstance(r[0], str))


def _warn_once(key: str, message: str, fmt: str, *args: object) -> None:
    """Log at warning level the first time a (path, message) pair is seen.

    Caller holds ``_cache_lock``. Later repeats go to debug, so a permanently
    unreadable file is visible once without a line per recall turn.
    """
    if (key, message) in _warned:
        logger.debug("known names: " + fmt, *args)
        return
    _warned.add((key, message))
    logger.warning("known names: " + fmt, *args)


def load_known_names(persona_dir: Path | str) -> KnownNames:
    """The persona's known names, read once and re-read when the file changes.

    Order matters (CONC-2): the signature is taken BEFORE the row read and the
    rows are stored under it, so a write committed between the two, or after
    the read, leaves the stored signature stale and is seen on the next call.
    A locked file returns the previous list (empty if none) and stores nothing.
    """
    path = known_names_path(persona_dir)
    key = str(path)
    with _cache_lock:
        cached = _cache.get(key)
        previous = cached[1] if cached is not None else EMPTY
        try:
            sig = _signature(path)
        except _SignatureUnavailableError as exc:
            _warn_once(
                key, str(exc), "signature unavailable for %s (%s); previous list kept", path, exc
            )
            return previous
        if cached is not None and cached[0] == sig:
            return cached[1]
        if sig is None:
            names = EMPTY
        else:
            try:
                names = _read_rows(path)
            except _TransientError as exc:
                if exc.busy:
                    logger.debug("known names: %s busy now (%s); previous list kept", path, exc)
                else:
                    _warn_once(
                        key, str(exc), "%s cannot be read now (%s); previous list kept", path, exc
                    )
                return previous
        _cache[key] = (sig, names)
        return names


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _admit(name: object) -> tuple[str, str] | None:
    """``(name_lower, display)`` if the entry is admissible, else None (S70).

    Rejected: a non-string, an entry with no word in it (nothing to match), and
    an entry whose whole lower-cased form is a recall stopword. Everything else
    is admitted as extracted.
    """
    if not isinstance(name, str):
        return None
    # A lone UTF-16 surrogate (a half emoji from a model's JSON escape) cannot be
    # bound to SQLite; replace it so the name itself is still admitted.
    display = name.strip().encode("utf-8", "replace").decode("utf-8")
    lower = normalize_name(display)
    if not lower or lower in RECALL_STOPWORDS:
        return None
    return lower, display


def admit_names(
    persona_dir: Path | str,
    names: Iterable[str],
    source: str,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Add names to the persona's list; the one write entry point for every writer.

    Applies the admission filter (S70), then writes the admitted names under the
    OS file lock in one transaction with ``INSERT OR IGNORE`` (an existing name
    keeps its original display, source and first-seen time). The file is created
    on first write.

    Returns the lower-cased forms of the admitted names that are now in the list
    (including ones already there); ``[]`` when nothing was admitted or the write
    could not be made. A failed write is logged and never raises: the caller's
    own work (a judge verdict, an importance update) must not be lost to it.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown known-names source {source!r}; expected one of {SOURCES}")
    admitted: dict[str, str] = {}
    for raw in names:
        item = _admit(raw)
        if item is not None:
            admitted.setdefault(item[0], item[1])
    if not admitted:
        return []
    stamp = (now or datetime.now(UTC)).isoformat()
    path = known_names_path(persona_dir)
    rows = [(lower, display, source, stamp) for lower, display in admitted.items()]
    try:
        with file_lock(path):
            written = _write_rows(path, rows)
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.warning("known names: could not write %s (%s); write skipped", path, exc)
        return []
    return list(admitted) if written else []


def _insert_rows(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    # No timeout argument: SQLite's own default busy wait (S88). memories.db's 30 s
    # constant is sized for its clustering writes and does not apply to this file.
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()
        if mode is None or str(mode[0]).lower() != "delete":
            logger.warning(
                "known names: %s is in journal mode %r, not delete; the change-counter signature may not hold",
                path,
                mode[0] if mode else None,
            )
        conn.execute(_SCHEMA)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO known_names (name_lower, display, source, first_seen)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
    finally:
        conn.close()


def _write_rows(path: Path, rows: list[tuple[str, str, str, str]]) -> bool:
    """Insert under the caller's file lock; True if the rows are in the file.

    A busy or locked file (``OperationalError``) skips the write and never
    renames anything. Only a non-Operational ``DatabaseError`` that
    ``PRAGMA quick_check`` confirms sends the file aside, after which the write
    is retried once against a fresh file.
    """
    for attempt in range(2):
        try:
            _insert_rows(path, rows)
            return True
        except sqlite3.OperationalError as exc:
            logger.warning("known names: %s busy or unavailable (%s); write skipped", path, exc)
            return False
        except sqlite3.DatabaseError as exc:
            if attempt == 0 and _confirmed_corrupt(path):
                if not _rename_aside(path):
                    return False
                continue
            logger.warning("known names: %s rejected the write (%s); write skipped", path, exc)
            return False
    return False


def _confirmed_corrupt(path: Path) -> bool:
    """True only when SQLite says the file is not a healthy database.

    A locked or busy file is not corrupt (``OperationalError`` -> False).
    """
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(path))
        row = conn.execute("PRAGMA quick_check").fetchone()
    except sqlite3.OperationalError:
        return False
    except sqlite3.DatabaseError:
        return True
    finally:
        if conn is not None:
            conn.close()
    return row is None or str(row[0]).lower() != "ok"


def _rename_aside(path: Path) -> bool:
    """Move a confirmed-corrupt file (and its journal) to ``<name>.corrupt-<UTC ts>``.

    Never deletes. The journal goes first, and with the file, so a stale hot
    journal cannot be replayed into the fresh database that replaces it. An
    OS error (Windows open-file semantics, permissions) is logged and returns
    False: the write is skipped and nothing crashes.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    n = 0
    while target.exists():
        n += 1
        target = path.with_name(f"{path.name}.corrupt-{stamp}-{n}")
    journal = path.with_name(path.name + "-journal")
    try:
        if journal.exists():
            os.replace(journal, target.with_name(target.name + "-journal"))
        os.replace(path, target)
    except OSError as exc:
        logger.warning(
            "known names: could not rename corrupt %s aside (%s); write skipped", path, exc
        )
        return False
    logger.warning(
        "known names: %s was not a healthy database; renamed to %s and recreated", path, target.name
    )
    return True
