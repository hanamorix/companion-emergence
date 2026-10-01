"""The known-names list: a running list of name words, and the matcher over it.

Name-recall fix, increments N1 and "names" (spec section 5 and 7: S19, S26, S37,
S47, S49, S66, S70, S104-S106, S121, S125, S129, S135-S138, S144, S151-S153). The kindled's recall
protects a word she has been told is a name (or has learned is one), whatever the
stopword and length rules would otherwise do to it. The list lives in its own small
SQLite file in the persona directory (``known_names.db``): not memories.db, not
keyed to memories, and it travels with the persona directory.

Tables of the file:

* ``known_names`` (unchanged since N1): one row per name, ``name_lower`` (the
  lower-cased form, words split on the recall tokenizer's word boundary and joined
  by single spaces), ``display`` (as extracted, or as the kindled spelled it),
  ``source`` (``gate`` / ``reappraiser`` / ``tool``) and ``first_seen`` (UTC ISO).
* ``stopword_names``: the S104 flag. A row means the listed name is a stopword-name
  ("Will"), whichever source added it (S144). It is a record only; matching decides
  stopword-ness with the shared ``name_rules.is_stopword_name`` (the graveyard, which
  is handed bare names, uses the same predicate), so no reader reads this table.
* ``rejected_names``: the S135 rejects list. Every removed name goes on it, whoever
  removed it (the kindled with ``remove_name``, or the re-appraiser). Mechanical
  adders are refused a rejected name outright; the kindled's ``add_name`` re-adds it
  only through the S137 confirm step.
* ``pending_readd``: at most one outstanding S137 question ("{name} was previously
  removed ..."), on disk so it outlives the MCP process of one CLI run.

Concurrency:

* Writers (:func:`admit_names`, :func:`remove_names`, :func:`confirm_readd`, and
  :func:`note_tool_call` when it lapses a question) run under the existing OS file
  lock (``brain.utils.file_lock``, not re-entrant: none of them calls another) and
  one short ``BEGIN IMMEDIATE`` transaction. The file uses the rollback journal
  (``journal_mode=DELETE``), never WAL, so a committed write changes the file's
  modification time AND the SQLite header's file change counter (S49, S66). A
  writer renames the file aside (never deletes it) only when SQLite reports a
  non-busy ``DatabaseError`` and ``PRAGMA quick_check`` confirms the file is not a
  healthy database; a busy or locked file is never renamed.
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
recall selector's stopword and length rules, in one pass over the message (S106):
single-word names by a set lookup per word, multi-word names by a word-level
Aho-Corasick automaton (a trie over name words with failure links). A multi-word
name matches only as a window of consecutive raw words, and
:func:`names_fts_query` sends matched names to the keyword search as FTS phrases.

Admission (:func:`admit_names`, :func:`confirm_readd`; S144, S152, S153, S135): a name is
admitted as given from every source (the kindled's add_name, the gate judge, the
re-appraiser): periods, several words and common words all pass, whatever characters it
holds; a one-word stopword-name is flagged (S104). The only refusals are a non-string, a
name longer than ``MAX_NAME_LENGTH`` characters (S153), a name with no word in it (it has
no matchable form), and, for a mechanical source, a name on the rejects list (S135).
"""

from __future__ import annotations

import functools
import logging
import os
import sqlite3
import threading
import unicodedata
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from brain import dev_constants
from brain.memory.name_rules import (
    WORD_RE,
    followed_by_apostrophe_t,
    is_stopword_name,
    stopword_name_occurrence_ok,
)
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

__all__ = [
    "SOURCES",
    "EMPTY",
    "KnownNames",
    "admit_names",
    "confirm_readd",
    "known_names_path",
    "listed_display_forms",
    "load_known_names",
    "match_known_names",
    "name_hit_ok",
    "names_fts_query",
    "normalize_name",
    "note_tool_call",
    "remove_names",
]

#: The allowed ``source`` values (S26).
SOURCES: tuple[str, ...] = ("gate", "reappraiser", "tool")
#: The kindled's own source. Every other source is a mechanical adder (S135: the
#: owner's "purely mechanical additions"): today the gate judge and the
#: re-appraiser. The rejects-list rule is written ``source != TOOL_SOURCE``; this
#: ASSUMES every non-tool source is mechanical. A future NON-mechanical adder (a
#: user-facing add, say) is not covered by that rule and needs its own ruling.
TOOL_SOURCE = "tool"
#: Who can remove a name (S105): the kindled's tool, or the re-appraiser.
REMOVERS: tuple[str, ...] = ("reappraiser", "tool")

# The recall tokenizer's word boundary (`brain.chat.prompt._extract_recall_tokens`
# matches runs of ASCII letters and digits). Non-Latin and space-less scripts
# are out of scope (#317).
_WORD_RE = WORD_RE

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
_STOPWORD_NAMES_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS stopword_names ("
    " name_lower TEXT PRIMARY KEY,"
    " flagged_at TEXT NOT NULL)"
)
_REJECTED_NAMES_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS rejected_names ("
    " name_lower TEXT PRIMARY KEY,"
    " display TEXT NOT NULL,"
    " rejected_by TEXT NOT NULL CHECK (rejected_by IN ("
    + ", ".join(f"'{s}'" for s in REMOVERS)
    + ")),"
    " rejected_at TEXT NOT NULL)"
)
_PENDING_READD_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS pending_readd ("
    " slot INTEGER PRIMARY KEY CHECK (slot = 1),"
    " name_lower TEXT NOT NULL,"
    " asked_at TEXT NOT NULL)"
)

_SQLITE_BUSY = 5
_SQLITE_LOCKED = 6

#: The longest name admitted, in characters, from every source (S153). The one limit that
#: stands in for the old character rule; the extraction prompts' fences are the other.
MAX_NAME_LENGTH = 100


class _WordAutomaton:
    """Aho-Corasick over WORDS: a trie of the multi-word names with failure links.

    ``step`` advances one message word in amortized constant time; ``out[s]`` is the
    name ending exactly at state ``s`` (or None) and ``link[s]`` the nearest proper
    suffix state that ends a name (-1: none), so every name ending at a word is
    reached by following ``link`` from the state (the dictionary-suffix chain).
    """

    __slots__ = ("goto", "fail", "out", "word_count", "link")

    def __init__(self, phrases: Iterable[tuple[str, ...]]) -> None:
        goto: list[dict[str, int]] = [{}]
        out: list[str | None] = [None]
        word_count: list[int] = [0]
        for words in phrases:
            state = 0
            for word in words:
                nxt = goto[state].get(word)
                if nxt is None:
                    nxt = len(goto)
                    goto.append({})
                    out.append(None)
                    word_count.append(0)
                    goto[state][word] = nxt
                state = nxt
            out[state] = " ".join(words)
            word_count[state] = len(words)
        fail = [0] * len(goto)
        link = [-1] * len(goto)
        queue = deque(goto[0].values())
        while queue:
            state = queue.popleft()
            for word, child in goto[state].items():
                f = fail[state]
                while f and word not in goto[f]:
                    f = fail[f]
                target = goto[f].get(word, 0)
                fail[child] = target
                link[child] = target if out[target] is not None else link[target]
                queue.append(child)
        self.goto = goto
        self.fail = fail
        self.out = out
        self.word_count = word_count
        self.link = link

    def step(self, state: int, word: str) -> int:
        goto = self.goto
        fail = self.fail
        while state and word not in goto[state]:
            state = fail[state]
        return goto[state].get(word, 0)


@dataclass(frozen=True)
class KnownNames:
    """An immutable snapshot of the list, shaped for the matcher.

    The matcher's indexes (the single-word set, the stopword-name set and the
    multi-word automaton) are built on first use and cached on the instance, so a
    snapshot built once per process (``load_known_names``) builds them once.
    """

    names: frozenset[str]
    #: The distinct word counts among the names, ascending.
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

    @functools.cached_property
    def _singles(self) -> frozenset[str]:
        return frozenset(n for n in self.names if " " not in n)

    @functools.cached_property
    def _stopword_singles(self) -> frozenset[str]:
        # Every recall stopword is one word, so a stopword-name is single-word.
        return frozenset(n for n in self._singles if is_stopword_name(n))

    @functools.cached_property
    def _automaton(self) -> _WordAutomaton | None:
        multi = sorted(tuple(n.split(" ")) for n in self.names if " " in n)
        return _WordAutomaton(multi) if multi else None


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


def _single_occurrence_ok(text: str, span: tuple[int, int], stopword_name: bool) -> bool:
    """The occurrence rules for a single-word name (the shared ``name_rules``):
    S125, never the stem of an n't contraction ("Don" in "don't"); S104/S121, a
    stopword-name only written capitalized and not at a sentence start."""
    if followed_by_apostrophe_t(text, span[1]):
        return False
    return not (stopword_name and not stopword_name_occurrence_ok(text, [span]))


def match_known_names(text: str, known: KnownNames, *, apply_rules: bool = True) -> list[str]:
    """The listed names that occur in ``text``, lower-cased, in order of occurrence.

    Tested on the RAW words (``[A-Za-z0-9]+`` runs, lower-cased) before any
    stopword or length rule (S47), so a listed 2-letter name or a listed
    stopword-name still matches. A multi-word name matches only as a window of
    consecutive raw words: ``new york`` matches "in New York" and does not match
    "new dress ... york". A name is reported once however often it occurs, at its
    first valid occurrence; the order is by that occurrence's first word, then by
    word count. One pass over the message (S106): the cost grows with the message,
    not with the number of listed names.

    Occurrence rules (single-word names only; every recall stopword is one word):
    S125, "Don" never matches "don't"; S104/S121, a stopword-name ("will") matches
    only where written capitalized and not at a sentence start (message start, a
    newline, or after . ! ?). ``apply_rules=False`` skips both and reports every
    CANDIDATE occurrence (used by :func:`name_hit_ok`).
    """
    if not text or not known.names:
        return []
    spans = list(_WORD_RE.finditer(text))
    if not spans:
        return []
    singles = known._singles  # noqa: SLF001 - the snapshot's own index
    stopword_singles = known._stopword_singles  # noqa: SLF001
    automaton = known._automaton  # noqa: SLF001
    found: dict[str, tuple[int, int]] = {}
    state = 0
    for i, match in enumerate(spans):
        word = match.group().lower()
        if word in singles and word not in found:
            if not apply_rules or _single_occurrence_ok(
                text, match.span(), word in stopword_singles
            ):
                found[word] = (i, 1)
        if automaton is not None:
            state = automaton.step(state, word)
            node = state if automaton.out[state] is not None else automaton.link[state]
            while node > 0:
                name = automaton.out[node]
                if name is None or name in found:
                    # A found name's whole suffix chain was reported when it was
                    # found (multi-word names carry no position rule).
                    break
                count = automaton.word_count[node]
                found[name] = (i - count + 1, count)
                node = automaton.link[node]
    return [name for name, _ in sorted(found.items(), key=lambda kv: kv[1])]


def names_fts_query(names: Sequence[str]) -> str:
    """An FTS5 MATCH expression: the names as quoted phrases, OR-joined.

    ``["pretzel", "new york"]`` -> ``'"pretzel" OR "new york"'``. Each name is
    letters, digits and single spaces only (it came from :func:`normalize_name`),
    so a phrase needs no escaping and a multi-word name matches only as
    consecutive tokens. ``""`` when there are no names.
    """
    return " OR ".join(f'"{n}"' for n in names if n)


def _fold_text(text: str) -> str:
    """The text as FTS5's default tokenizer sees it, approximately (unicode61 with
    diacritic removal): NFKD, combining marks dropped, runs of Unicode letters and
    digits joined by single spaces, lower-cased. Latin text only (#317)."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.category(ch).startswith("M"))
    words: list[str] = []
    current: list[str] = []
    for ch in stripped:
        if ch.isalnum():
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return " ".join(words).lower()


def name_hit_ok(text: str, names: Sequence[str], known: KnownNames | None = None) -> bool:
    """Whether a keyword name-query hit stays (S121, S129), judged per queried name.

    Each name is VALID (an occurrence on the shared word split that passes the
    S104/S125 rules), INVALID (occurrences on the split, none valid), FOLD-ONLY (no
    occurrence on the split, one on the FTS-style fold: "Renée" for ``renee``) or
    ABSENT. The row is kept when any name is VALID or FOLD-ONLY, or when every name
    is ABSENT (unjudgeable: a substring match of the recency fallback); it is
    dropped only when no name is VALID or FOLD-ONLY and at least one is INVALID.
    ``known`` is the snapshot of ``names`` (built once per query by the caller).
    """
    if known is None:
        known = KnownNames.from_names(names)
    if not known.names:
        return True
    valid = match_known_names(text, known)
    if valid:
        return True
    candidates = set(match_known_names(text, known, apply_rules=False))
    absent = [n for n in known.names if n not in candidates]
    # FOLD-ONLY needs a non-ASCII character: for ASCII text the fold and the
    # shared word split are the same runs, so the fold could not change the verdict
    # (and it is the costly step on a long scan of rejected rows).
    if (
        absent
        and not text.isascii()
        and match_known_names(_fold_text(text), KnownNames.from_names(absent), apply_rules=False)
    ):
        return True
    return not candidates


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




def listed_display_forms(persona_dir: Path | str | None, text: str) -> list[str]:
    """The display forms of the listed names that occur in ``text`` (entry-side
    rules applied), in order of occurrence: the re-appraiser's KNOWN NAMES (S105,
    S107). Read-only; ``[]`` on any failure (logged). Off the recall hot path."""
    if persona_dir is None or not text:
        return []
    try:
        lowers = match_known_names(text, load_known_names(persona_dir))
        if not lowers:
            return []
        path = known_names_path(persona_dir)
        conn = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
        try:
            display: dict[str, str] = {}
            for lower in lowers:
                row = conn.execute(
                    "SELECT display FROM known_names WHERE name_lower = ?", (lower,)
                ).fetchone()
                if row is not None and isinstance(row[0], str):
                    display[lower] = row[0]
        finally:
            conn.close()
        return [display[lower] for lower in lowers if lower in display]
    except Exception as exc:  # noqa: BLE001 - the re-appraiser still runs without the list
        logger.warning("known names: display lookup failed (%s); no KNOWN NAMES this call", exc)
        return []


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _admit(name: object, source: str) -> tuple[str, str] | None:
    """``(name_lower, display)`` if the entry is admissible from ``source``, else None.

    Refused (logged at debug, so a refusal leaves a trace): a non-string; a name longer
    than :data:`MAX_NAME_LENGTH` characters once stripped (S153); an entry with no word in
    it (its normalized form is empty, so it can neither be stored nor matched). Everything
    else is admitted as given from every source (S144, S152): no character rule and no
    stopword rule. The rejects list is applied by the writers, not here.
    """
    if not isinstance(name, str):
        logger.debug("known names: refused a non-string entry from %s", source)
        return None
    raw = name.strip()
    if len(raw) > MAX_NAME_LENGTH:
        logger.debug("known names: refused a %d-character entry from %s (limit, S153)", len(raw), source)
        return None
    # A lone UTF-16 surrogate (a half emoji from a model's JSON escape) cannot be
    # bound to SQLite; the replace step turns it into "?" so the name is still admitted.
    display = raw.encode("utf-8", "replace").decode("utf-8")
    lower = normalize_name(display)
    if not lower:
        logger.debug("known names: refused %r from %s (no word)", raw, source)
        return None
    return lower, display


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """``known_names`` and the three side tables, idempotently (every write, under
    the lock, before the transaction). ``known_names`` itself never changes shape."""
    for statement in (
        _SCHEMA,
        _STOPWORD_NAMES_SCHEMA,
        _REJECTED_NAMES_SCHEMA,
        _PENDING_READD_SCHEMA,
    ):
        conn.execute(statement)


def _transact(path: Path, op):  # noqa: ANN001, ANN202 - op(conn) -> any
    """Open the file, keep it in the rollback journal, make the tables, and run
    ``op(conn)`` inside one ``BEGIN IMMEDIATE`` transaction (rolled back on error).
    Caller holds the file lock."""
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
        _ensure_tables(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = op(conn)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
    finally:
        conn.close()
    return result


def _is_rejected(conn: sqlite3.Connection, lower: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM rejected_names WHERE name_lower = ?", (lower,)).fetchone()
        is not None
    )


def _write_tool_row(conn: sqlite3.Connection, row: tuple[str, str, str, str]) -> None:
    """The kindled's add of a name that is not rejected: insert, or make an
    existing row hers (display as she spelled it, source ``tool``; first-seen kept,
    I2), and flag it when it is a stopword-name (S104)."""
    lower, display, _source, stamp = row
    conn.execute(
        "INSERT INTO known_names (name_lower, display, source, first_seen) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(name_lower) DO UPDATE SET display = excluded.display, source = ?",
        (lower, display, TOOL_SOURCE, stamp, TOOL_SOURCE),
    )
    _flag_stopword_name(conn, lower, stamp)


def _flag_stopword_name(conn: sqlite3.Connection, lower: str, stamp: str) -> None:
    """Record a one-word stopword-name ("will") in ``stopword_names``, whichever source
    added it (S144). A record only: matching decides stopword-ness with the shared
    ``name_rules.is_stopword_name``."""
    if is_stopword_name(lower):
        conn.execute(
            "INSERT OR IGNORE INTO stopword_names (name_lower, flagged_at) VALUES (?, ?)",
            (lower, stamp),
        )


def _insert_in_txn(conn: sqlite3.Connection, rows: list[tuple[str, str, str, str]]) -> set[str]:
    """Write the admitted rows; return the lower forms refused because they are on
    the rejects list. Mechanical sources: ``INSERT OR IGNORE`` (an existing name
    keeps its first row); the tool: :func:`_write_tool_row`. Neither ever adds a
    rejected name (only :func:`confirm_readd` can, S137)."""
    refused: set[str] = set()
    for row in rows:
        lower, _display, source, _stamp = row
        if _is_rejected(conn, lower):
            refused.add(lower)
            logger.debug("known names: refused %r from %s (on the rejects list, S135)", lower, source)
            continue
        if source == TOOL_SOURCE:
            _write_tool_row(conn, row)
        else:
            conn.execute(
                "INSERT OR IGNORE INTO known_names (name_lower, display, source, first_seen)"
                " VALUES (?, ?, ?, ?)",
                row,
            )
            _flag_stopword_name(conn, lower, row[3])
    return refused


def _insert_rows(path: Path, rows: list[tuple[str, str, str, str]]) -> set[str] | None:
    """Insert under the caller's file lock, in one transaction. Returns the lower
    forms refused as rejected (None when none was)."""
    refused = _transact(path, lambda conn: _insert_in_txn(conn, rows))
    return refused or None


def _with_recovery(path: Path, attempt):  # noqa: ANN001, ANN202
    """Run ``attempt()`` (a write under the caller's lock); ``(True, value)`` on
    success, ``(False, None)`` when the write was skipped.

    A busy or locked file (``OperationalError``) skips the write and never renames
    anything. Only a non-Operational ``DatabaseError`` that ``PRAGMA quick_check``
    confirms sends the file aside, after which the write is retried once against a
    fresh file.
    """
    for try_no in range(2):
        try:
            return True, attempt()
        except sqlite3.OperationalError as exc:
            logger.warning("known names: %s busy or unavailable (%s); write skipped", path, exc)
            return False, None
        except sqlite3.DatabaseError as exc:
            if try_no == 0 and _confirmed_corrupt(path):
                if not _rename_aside(path):
                    return False, None
                continue
            logger.warning("known names: %s rejected the write (%s); write skipped", path, exc)
            return False, None
    return False, None


def _write_rows(path: Path, rows: list[tuple[str, str, str, str]]) -> set[str] | None:
    """Insert under the caller's file lock with the corrupt-file recovery; the
    lower forms refused as rejected, or None when the write was skipped."""
    ok, refused = _with_recovery(path, lambda: _insert_rows(path, rows))
    if not ok:
        return None
    return set(refused or ())


def admit_names(
    persona_dir: Path | str,
    names: Iterable[str],
    source: str,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Add names to the persona's list; the write entry point for every mechanical
    writer (and for tests; the add_name tool uses :func:`confirm_readd`).

    Applies the admission rules (:func:`_admit`), then writes the admitted names
    under the OS file lock in one transaction. A mechanical source's name that is
    already listed keeps its original row (``INSERT OR IGNORE``); a name on the
    rejects list is refused outright, from every source (S135; this function never
    asks the S137 question). The file is created on first write.

    Returns the lower-cased forms of the admitted names that are now in the list
    (including ones already there); ``[]`` when nothing was admitted or the write
    could not be made. A failed write is logged and never raises: the caller's own
    work (a judge verdict, an importance update) must not be lost to it.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown known-names source {source!r}; expected one of {SOURCES}")
    admitted: dict[str, str] = {}
    for raw in names:
        item = _admit(raw, source)
        if item is not None:
            admitted.setdefault(item[0], item[1])
    if not admitted:
        return []
    stamp = (now or datetime.now(UTC)).isoformat()
    path = known_names_path(persona_dir)
    rows = [(lower, display, source, stamp) for lower, display in admitted.items()]
    try:
        with file_lock(path):
            refused = _write_rows(path, rows)
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.warning("known names: could not write %s (%s); write skipped", path, exc)
        return []
    if refused is None:
        return []
    return [lower for lower in admitted if lower not in refused]


def remove_names(
    persona_dir: Path | str,
    names: Iterable[object],
    remover: str,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Remove names from the list and put each on the rejects list (S105, S135).

    ``remover`` is ``"tool"`` (the kindled: any name) or ``"reappraiser"`` (never a
    name whose source is ``tool``: one the kindled added, S105). Names are compared
    after :func:`normalize_name` (the model echoes the display form "Tuesday").
    The source check and the delete run in one write transaction under the file
    lock. Returns the lower forms actually removed; never creates the file (a
    missing file: ``[]``); never raises (logged, ``[]``).
    """
    if remover not in REMOVERS:
        raise ValueError(f"unknown known-names remover {remover!r}; expected one of {REMOVERS}")
    lowers = list(dict.fromkeys(normalize_name(n) for n in names if isinstance(n, str)))
    lowers = [lower for lower in lowers if lower]
    if not lowers:
        return []
    stamp = (now or datetime.now(UTC)).isoformat()
    path = known_names_path(persona_dir)

    def _op(conn: sqlite3.Connection) -> list[str]:
        removed: list[str] = []
        for lower in lowers:
            row = conn.execute(
                "SELECT display, source FROM known_names WHERE name_lower = ?", (lower,)
            ).fetchone()
            if row is None:
                continue
            if remover != TOOL_SOURCE and row[1] == TOOL_SOURCE:
                continue  # never a name the kindled added (S105)
            conn.execute("DELETE FROM known_names WHERE name_lower = ?", (lower,))
            conn.execute("DELETE FROM stopword_names WHERE name_lower = ?", (lower,))
            conn.execute(
                "INSERT OR REPLACE INTO rejected_names (name_lower, display, rejected_by,"
                " rejected_at) VALUES (?, ?, ?, ?)",
                (lower, row[0], remover, stamp),
            )
            removed.append(lower)
        return removed

    try:
        with file_lock(path):
            if not path.exists():
                return []
            ok, removed = _with_recovery(path, lambda: _transact(path, _op))
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.warning("known names: could not remove from %s (%s); nothing removed", path, exc)
        return []
    return list(removed) if ok and removed else []


ReaddOutcome = Literal["added", "asked", "refused"]


def confirm_readd(
    persona_dir: Path | str, name: object, *, now: datetime | None = None
) -> ReaddOutcome:
    """The add_name tool's one writer (S104, S135, S137). Takes the file lock once
    and runs one transaction:

    * the admission rules refuse the name -> ``"refused"`` (nothing written);
    * the name is on the rejects list and the pending question is for this same
      name (her previous counted tool call asked it) -> take it off the rejects
      list, clear the question, add it (source ``tool``) -> ``"added"``;
    * on the rejects list otherwise -> record the question for this name
      (replacing any earlier one), add nothing -> ``"asked"``;
    * not on the rejects list -> clear any question, add it -> ``"added"``.

    Any failure (a busy file, a corrupt file, a failed insert) rolls back, is
    logged, and returns ``"refused"``: nothing is half-written, so a reject and a
    pending question survive a failed add.
    """
    item = _admit(name, TOOL_SOURCE)
    if item is None:
        return "refused"
    lower, display = item
    stamp = (now or datetime.now(UTC)).isoformat()
    path = known_names_path(persona_dir)

    def _op(conn: sqlite3.Connection) -> ReaddOutcome:
        pending = conn.execute("SELECT name_lower FROM pending_readd WHERE slot = 1").fetchone()
        if _is_rejected(conn, lower):
            if pending is not None and pending[0] == lower:
                conn.execute("DELETE FROM rejected_names WHERE name_lower = ?", (lower,))
                conn.execute("DELETE FROM pending_readd")
                _write_tool_row(conn, (lower, display, TOOL_SOURCE, stamp))
                return "added"
            conn.execute(
                "INSERT OR REPLACE INTO pending_readd (slot, name_lower, asked_at) VALUES (1, ?, ?)",
                (lower, stamp),
            )
            return "asked"
        conn.execute("DELETE FROM pending_readd")
        _write_tool_row(conn, (lower, display, TOOL_SOURCE, stamp))
        return "added"

    try:
        with file_lock(path):
            ok, outcome = _with_recovery(path, lambda: _transact(path, _op))
    except Exception as exc:  # noqa: BLE001 - a tool call never raises on a names write
        logger.warning("known names: add_name write failed for %s (%s)", path, exc)
        return "refused"
    return outcome if ok and outcome else "refused"


def _read_pending(path: Path) -> str | None:
    """The pending S137 question's name, read-only (never creates the file); None
    when there is none (no table, no row)."""
    conn = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    try:
        try:
            row = conn.execute("SELECT name_lower FROM pending_readd WHERE slot = 1").fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return None
            raise
    finally:
        conn.close()
    return row[0] if row is not None and isinstance(row[0], str) else None


def note_tool_call(
    persona_dir: Path | str | None,
    name: str,
    arguments: Mapping[str, object] | None,
    *,
    add_tool_name: str,
) -> None:
    """Called for every tool call that reaches brain tool dispatch (S137, S138): a
    call that is not ``add_name`` with the pending question's name lapses the
    question (the name stays rejected).

    S138 (Planning's reading of S137's derived lapse): "her next tool call" is the
    next call reaching brain tool dispatch, including the MCP ``search_memories``
    forward; the Claude CLI's built-in web tools and calls the MCP library rejects
    before our handler are not counted. The caller passes ``add_tool_name`` (this
    module must not import the tool schemas: a circular import).

    Cheap by design: no file -> return; otherwise a read-only read, and the file
    lock and a write only while a question is pending. Never raises (logged).
    """
    try:
        if persona_dir is None:
            return
        path = known_names_path(persona_dir)
        if not path.exists():
            return
        pending = _read_pending(path)
        if pending is None:
            return
        given = arguments.get("name") if isinstance(arguments, Mapping) else None
        if name == add_tool_name and isinstance(given, str) and normalize_name(given) == pending:
            return  # the confirming call: add_name consumes the question
        with file_lock(path):
            if not path.exists():
                return
            _transact(
                path,
                lambda conn: conn.execute(
                    "DELETE FROM pending_readd WHERE name_lower = ?", (pending,)
                ),
            )
        logger.debug("known names: the add_name question for %r lapsed (%s called)", pending, name)
    except Exception as exc:  # noqa: BLE001 - a tool call is never lost to the hook
        logger.warning("known names: tool-call hook failed (%s); question not lapsed", exc)


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
    False: the write is skipped and nothing crashes. The rejects list and any
    pending question go aside with the file (a known limit).
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
