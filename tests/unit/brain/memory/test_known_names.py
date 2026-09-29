"""Tests for brain.memory.known_names (name-recall fix, increment N1).

Criteria: C5 (matcher half), C6 (missing/corrupt file, admission, change
signature), CONC-2/2b (reader cache), CONC-4 (two writer processes), CONC-5
(reader and writer against a locked file), INV-I13a (rename-aside failure,
hot journal, short header, writers use ``file_lock``), and the stopword-set
move (P-30).

Every test is deterministic: the CONC-2 windows are opened by seams on
``_read_rows``, the CONC-2b same-mtime case restores the modification time with
``os.utime``, the CONC-5 lock is a real SQLite exclusive lock held by the test
itself (the busy timeout is shrunk so nothing waits), and the CONC-4 workers are
released by a barrier file. All data is synthetic, in ``tmp_path``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.memory import known_names as kn
from brain.memory.recall_stopwords import RECALL_STOPWORDS

T0 = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)

# The test's own connections use this, so a seam on ``sqlite3.connect`` (which is the
# same module object the code under test uses) never sees them.
_CONNECT = sqlite3.connect


def _db(persona: Path) -> Path:
    return kn.known_names_path(persona)


def _rows(persona: Path) -> list[tuple[str, str, str, str]]:
    conn = sqlite3.connect(str(_db(persona)))
    try:
        return conn.execute(
            "SELECT name_lower, display, source, first_seen FROM known_names ORDER BY name_lower"
        ).fetchall()
    finally:
        conn.close()


def _names(persona: Path) -> set[str]:
    return set(kn.load_known_names(persona).names)


def _counter(path: Path) -> int:
    """The SQLite header file change counter, parsed independently of the module."""
    head = path.read_bytes()[:28]
    return int.from_bytes(head[24:28], "big")


def _aside(persona: Path) -> list[Path]:
    return sorted(p for p in persona.iterdir() if ".corrupt-" in p.name)


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    d = tmp_path / "persona"
    d.mkdir()
    return d


@pytest.fixture
def quick_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the writers' busy wait so a locked file fails at once.

    The writers pass no timeout (SQLite's default, S88), so the seam fills one in
    on every writer-side connect (the reader's ``uri=True`` connect is left alone)
    and FAILS a writer that brings its own, so a writer that goes back to a fixed
    constant fails these tests at once instead of merely making them slow.
    """
    real = sqlite3.connect

    def quick(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        if not kwargs.get("uri"):
            assert "timeout" not in kwargs and not args, "writers must use SQLite's default timeout"
            kwargs["timeout"] = 0.05
        return real(database, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(kn.sqlite3, "connect", quick)


# ---------------------------------------------------------------------------
# C5 (matcher half): raw words, before the stopword and length rules
# ---------------------------------------------------------------------------


def test_lowercase_typed_name_in_the_list_matches() -> None:
    known = kn.KnownNames.from_names(["Pretzel", "wren"])
    assert kn.match_known_names("did pretzel eat the thing", known) == ["pretzel"]
    assert kn.match_known_names("PRETZEL and Wren", known) == ["pretzel", "wren"]


def test_a_word_not_in_the_list_is_not_matched() -> None:
    known = kn.KnownNames.from_names(["pretzel"])
    assert kn.match_known_names("the kettle is boiling", known) == []


def test_listed_two_letter_name_survives_the_length_rule() -> None:
    known = kn.KnownNames.from_names(["al", "NY"])
    assert kn.match_known_names("ask al about ny", known) == ["al", "ny"]


def test_listed_stopword_survives_the_stopword_rule() -> None:
    # The stopword entry is written straight into the list here: admission (S70)
    # keeps such entries out through every writer (see the admission tests), so
    # this asserts the matcher, not the admission.
    assert "will" in RECALL_STOPWORDS
    known = kn.KnownNames.from_names(["will"])
    assert kn.match_known_names("Will you come", known) == ["will"]
    assert kn.match_known_names("she will come", known) == ["will"]


def test_multiword_name_matches_only_the_consecutive_raw_words() -> None:
    known = kn.KnownNames.from_names(["new york"])
    assert kn.match_known_names("we flew to New York today", known) == ["new york"]
    assert kn.match_known_names("a new dress with a york tag", known) == []
    assert kn.match_known_names("new, york", known) == ["new york"]
    assert kn.match_known_names("york new", known) == []
    assert kn.match_known_names("new", known) == []


def test_overlapping_windows_and_mixed_word_counts_are_all_found() -> None:
    known = kn.KnownNames.from_names(["new york", "york city", "york", "a b c"])
    assert kn.match_known_names("new york city", known) == ["new york", "york", "york city"]
    assert kn.match_known_names("x a b c y", known) == ["a b c"]
    assert kn.match_known_names("a b", known) == []


def test_a_name_is_reported_once_however_often_it_occurs() -> None:
    known = kn.KnownNames.from_names(["pretzel"])
    assert kn.match_known_names("pretzel pretzel Pretzel", known) == ["pretzel"]


def test_matches_come_in_order_of_first_occurrence() -> None:
    known = kn.KnownNames.from_names(["wren", "pretzel", "scrufflet"])
    assert kn.match_known_names("scrufflet met wren and pretzel", known) == [
        "scrufflet",
        "wren",
        "pretzel",
    ]


def test_apostrophes_and_digits_follow_the_recall_tokenizer() -> None:
    # The selector splits on runs of ASCII letters and digits, so "O'Brien" is two
    # words and the stored form is "o brien"; a message spells it the same way.
    assert kn.normalize_name("O'Brien") == "o brien"
    known = kn.KnownNames.from_names(["O'Brien", "R2D2"])
    assert kn.match_known_names("met O'Brien and r2d2", known) == ["o brien", "r2d2"]


def test_empty_list_or_empty_text_matches_nothing() -> None:
    assert kn.match_known_names("anything", kn.EMPTY) == []
    assert kn.match_known_names("", kn.KnownNames.from_names(["x"])) == []
    assert not kn.EMPTY
    assert len(kn.KnownNames.from_names(["a", "A", "b"])) == 2


def test_word_boundary_agrees_with_the_recall_selector() -> None:
    from brain.chat.prompt import _extract_recall_tokens

    text = "Roy's FBI trip: R2D2, café? pretzel-dog 42 times"
    words = {w.lower() for w in kn._WORD_RE.findall(text)}
    assert set(_extract_recall_tokens(text)) <= words
    assert {"roy", "s", "fbi", "r2d2", "pretzel", "dog", "42", "times"} <= words


def test_phrase_output_is_or_joined_quoted_phrases() -> None:
    assert kn.names_fts_query(["pretzel", "new york"]) == '"pretzel" OR "new york"'
    assert kn.names_fts_query([]) == ""
    assert kn.names_fts_query([""]) == ""


def test_phrase_output_matches_a_multiword_name_only_as_consecutive_tokens() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
    rows = ["we flew to New York today", "a new dress with a york tag", "pretzel ate it"]
    conn.executemany("INSERT INTO t(content) VALUES (?)", [(r,) for r in rows])
    known = kn.KnownNames.from_names(["new york", "pretzel"])
    found = kn.match_known_names("new york and pretzel", known)
    query = kn.names_fts_query(found)
    hits = {r[0] for r in conn.execute("SELECT content FROM t WHERE t MATCH ?", (query,))}
    assert hits == {rows[0], rows[2]}
    conn.close()


# ---------------------------------------------------------------------------
# C6: admission (S70), the file, creation on first write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["will", "the", "The", " WILL ", "will.", "Will!", "hello"])
def test_stopword_entries_are_rejected_and_nothing_is_written(persona: Path, entry: str) -> None:
    assert kn.admit_names(persona, [entry], "gate") == []
    assert not _db(persona).exists()


def test_every_recall_stopword_is_rejected_by_every_writer(persona: Path) -> None:
    for source in kn.SOURCES:
        assert kn.admit_names(persona, sorted(RECALL_STOPWORDS), source) == []
    assert not _db(persona).exists()


def test_other_entries_are_admitted_as_extracted(persona: Path) -> None:
    out = kn.admit_names(
        persona, ["Grace", "Will Smith", "NY", "the hague", "of the"], "tool", now=T0
    )
    assert out == ["grace", "will smith", "ny", "the hague", "of the"]
    assert _rows(persona) == [
        ("grace", "Grace", "tool", T0.isoformat()),
        ("ny", "NY", "tool", T0.isoformat()),
        ("of the", "of the", "tool", T0.isoformat()),
        ("the hague", "the hague", "tool", T0.isoformat()),
        ("will smith", "Will Smith", "tool", T0.isoformat()),
    ]


def test_a_multiword_entry_is_rejected_only_when_the_whole_entry_is_a_stopword(
    persona: Path,
) -> None:
    assert "the" in RECALL_STOPWORDS and "hague" not in RECALL_STOPWORDS
    assert kn.admit_names(persona, ["the hague"], "gate") == ["the hague"]
    assert "the hague" not in RECALL_STOPWORDS


def test_entries_with_no_word_or_no_text_are_rejected(persona: Path) -> None:
    junk: list[object] = ["", "   ", "--", "!!!", None, 7, b"pretzel"]
    assert kn.admit_names(persona, junk, "gate") == []  # type: ignore[arg-type]
    assert not _db(persona).exists()


def test_an_unknown_source_is_refused(persona: Path) -> None:
    with pytest.raises(ValueError, match="source"):
        kn.admit_names(persona, ["pretzel"], "guess")
    assert not _db(persona).exists()


def test_all_three_sources_write_and_the_table_refuses_any_other(persona: Path) -> None:
    for source in kn.SOURCES:
        kn.admit_names(persona, [f"name{source}"], source, now=T0)
    assert {r[2] for r in _rows(persona)} == {"gate", "reappraiser", "tool"}
    conn = sqlite3.connect(str(_db(persona)))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO known_names VALUES ('x','x','guess','t')")
    conn.close()


def test_a_missing_file_is_an_empty_list_and_is_created_on_first_write(persona: Path) -> None:
    assert not _db(persona).exists()
    assert kn.load_known_names(persona) is kn.EMPTY
    assert not _db(persona).exists()  # the reader never creates it
    kn.admit_names(persona, ["pretzel"], "tool")
    assert _db(persona).exists()
    assert _names(persona) == {"pretzel"}


def test_a_missing_persona_directory_is_created_by_the_first_write(tmp_path: Path) -> None:
    persona = tmp_path / "not" / "yet"
    assert kn.load_known_names(persona) is kn.EMPTY
    assert kn.admit_names(persona, ["pretzel"], "gate") == ["pretzel"]
    assert _names(persona) == {"pretzel"}


def test_an_existing_name_keeps_its_first_seen_time_display_and_source(persona: Path) -> None:
    kn.admit_names(persona, ["Pretzel"], "gate", now=T0)
    later = T0 + timedelta(days=1)
    out = kn.admit_names(persona, ["PRETZEL", "wren"], "tool", now=later)
    assert out == ["pretzel", "wren"]  # a name already in the list still counts as in the list
    assert _rows(persona) == [
        ("pretzel", "Pretzel", "gate", T0.isoformat()),
        ("wren", "wren", "tool", later.isoformat()),
    ]


def test_the_file_uses_the_rollback_journal_not_wal(persona: Path) -> None:
    kn.admit_names(persona, ["pretzel"], "gate")
    conn = sqlite3.connect(str(_db(persona)))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    conn.close()
    head = _db(persona).read_bytes()[:28]
    assert head[:16] == b"SQLite format 3\x00"
    assert head[18] == 1 and head[19] == 1  # legacy (rollback-journal) file format, not WAL (2)


def test_a_wal_file_is_converted_back_to_the_rollback_journal(persona: Path) -> None:
    conn = sqlite3.connect(str(_db(persona)))
    assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    conn.execute(kn._SCHEMA)
    conn.commit()
    conn.close()
    kn.admit_names(persona, ["pretzel"], "gate")
    conn = sqlite3.connect(str(_db(persona)))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    conn.close()
    assert _names(persona) == {"pretzel"}


def test_the_list_is_read_once_and_reread_only_when_the_file_changes(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kn.admit_names(persona, ["pretzel"], "gate")
    calls: list[Path] = []
    real = kn._read_rows

    def counting(path: Path) -> kn.KnownNames:
        calls.append(path)
        return real(path)

    monkeypatch.setattr(kn, "_read_rows", counting)
    first = kn.load_known_names(persona)
    for _ in range(5):
        assert kn.load_known_names(persona) is first
    assert len(calls) == 1
    kn.admit_names(persona, ["wren"], "gate")
    assert kn.load_known_names(persona).names == frozenset({"pretzel", "wren"})
    assert len(calls) == 2
    assert kn.load_known_names(persona).names == frozenset({"pretzel", "wren"})
    assert len(calls) == 2


def test_a_missing_file_is_cached_and_a_later_first_write_is_seen(persona: Path) -> None:
    assert kn.load_known_names(persona) is kn.EMPTY
    assert kn.load_known_names(persona) is kn.EMPTY
    kn.admit_names(persona, ["pretzel"], "gate")
    assert _names(persona) == {"pretzel"}


def test_lists_of_two_personas_do_not_mix(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    kn.admit_names(a, ["pretzel"], "gate")
    kn.admit_names(b, ["wren"], "gate")
    assert _names(a) == {"pretzel"}
    assert _names(b) == {"wren"}


# ---------------------------------------------------------------------------
# C6: a corrupt file reads as empty, the first write renames it aside
# ---------------------------------------------------------------------------

_GARBAGE = b"this is not a sqlite database, only text\n" * 200


def test_a_corrupt_file_reads_as_empty_and_is_left_alone_by_the_reader(persona: Path) -> None:
    _db(persona).write_bytes(_GARBAGE)
    assert kn.load_known_names(persona) is not None
    assert _names(persona) == set()
    assert _db(persona).read_bytes() == _GARBAGE
    assert _aside(persona) == []


def test_the_first_write_renames_a_corrupt_file_aside_and_recreates_it(persona: Path) -> None:
    _db(persona).write_bytes(_GARBAGE)
    assert _names(persona) == set()
    assert kn.admit_names(persona, ["pretzel"], "tool") == ["pretzel"]
    aside = _aside(persona)
    assert len(aside) == 1 and aside[0].name.startswith(_db(persona).name + ".corrupt-")
    assert aside[0].read_bytes() == _GARBAGE  # never deleted, never altered
    assert _names(persona) == {"pretzel"}  # the writer's rename changed the signature


def test_repeated_corruption_keeps_every_renamed_file(persona: Path) -> None:
    _db(persona).write_bytes(_GARBAGE)
    kn.admit_names(persona, ["pretzel"], "tool")
    _db(persona).write_bytes(_GARBAGE + b"second")
    kn.admit_names(persona, ["wren"], "tool")
    assert len(_aside(persona)) == 2
    assert _names(persona) == {"wren"}


def test_a_zero_byte_file_is_an_empty_database_not_a_corrupt_one(persona: Path) -> None:
    _db(persona).write_bytes(b"")
    assert _names(persona) == set()
    assert kn.admit_names(persona, ["pretzel"], "tool") == ["pretzel"]
    assert _aside(persona) == []
    assert _names(persona) == {"pretzel"}


def test_a_healthy_file_with_the_wrong_schema_is_never_renamed(persona: Path) -> None:
    conn = sqlite3.connect(str(_db(persona)))
    conn.execute("CREATE TABLE something_else (x)")
    conn.commit()
    conn.close()
    assert _names(persona) == set()
    kn.admit_names(persona, ["pretzel"], "tool")
    assert _aside(persona) == []
    assert _names(persona) == {"pretzel"}


def test_a_rename_aside_that_cannot_happen_skips_the_write_and_keeps_the_file(
    persona: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # INV-I13a: Windows open-file semantics (or permissions) raise PermissionError.
    _db(persona).write_bytes(_GARBAGE)

    def refuse(src: object, dst: object) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(kn.os, "replace", refuse)
    with caplog.at_level(logging.WARNING, logger=kn.logger.name):
        assert kn.admit_names(persona, ["pretzel"], "tool") == []
    assert _db(persona).read_bytes() == _GARBAGE
    assert _aside(persona) == []
    assert any("could not rename" in r.message for r in caplog.records)
    assert _names(persona) == set()  # still reads as empty, no crash


# ---------------------------------------------------------------------------
# C6 / CONC-2 / CONC-2b: the change signature
# ---------------------------------------------------------------------------


def test_signature_is_stat_plus_the_header_change_counter(persona: Path) -> None:
    kn.admit_names(persona, ["pretzel"], "gate")
    st = os.stat(_db(persona))
    sig = kn._signature(_db(persona))
    assert sig == (st.st_mtime_ns, st.st_size, st.st_ino, _counter(_db(persona)))
    before = sig[3]
    kn.admit_names(persona, ["wren"], "gate")
    after = kn._signature(_db(persona))
    assert after is not None and after[3] == _counter(_db(persona))
    assert after[3] != before


def test_a_file_shorter_than_the_header_has_no_counter_and_does_not_crash(persona: Path) -> None:
    _db(persona).write_bytes(b"SQLite format 3\x00")  # 16 bytes
    sig = kn._signature(_db(persona))
    assert sig is not None and sig[3] is None
    assert _names(persona) == set()
    _db(persona).write_bytes(b"x" * 27)
    assert kn._signature(_db(persona))[3] is None  # type: ignore[index]
    _db(persona).write_bytes(b"x" * 28)
    assert kn._signature(_db(persona))[3] == int.from_bytes(b"xxxx", "big")  # type: ignore[index]


def test_a_missing_file_has_no_signature(persona: Path) -> None:
    assert kn._signature(_db(persona)) is None


def _same_stat_write(persona: Path, first: str, second: str) -> tuple[kn._Signature, kn._Signature]:
    """Write ``second`` after ``first`` and restore the modification time.

    Returns the signatures before and after. Size and inode are unchanged (the
    row lands in an existing page and SQLite writes the file in place), so only
    the header change counter tells the two states apart.
    """
    kn.admit_names(persona, [first], "gate")
    st = os.stat(_db(persona))
    sig1 = kn._signature(_db(persona))
    assert sig1 is not None
    assert _names(persona) == {first}
    kn.admit_names(persona, [second], "gate")
    os.utime(_db(persona), ns=(st.st_atime_ns, st.st_mtime_ns))
    sig2 = kn._signature(_db(persona))
    assert sig2 is not None
    return sig1, sig2


def test_two_writes_with_the_same_mtime_size_and_inode_still_change_the_signature(
    persona: Path,
) -> None:
    sig1, sig2 = _same_stat_write(persona, "alpha", "bravo")
    if sig1[:3] != sig2[:3]:
        pytest.skip("this filesystem changed the size or inode; the same-stat case cannot be built")
    assert sig1[3] != sig2[3]
    assert sig1 != sig2
    assert _names(persona) == {"alpha", "bravo"}  # CONC-2b: seen on the next lookup


def test_a_three_field_signature_would_miss_the_same_mtime_write(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Able to fail (CONC-2b): the same fixture with the counter dropped stays stale."""
    real = kn._signature

    def three_fields(path: Path) -> kn._Signature | None:
        sig = real(path)
        return None if sig is None else (sig[0], sig[1], sig[2], None)

    monkeypatch.setattr(kn, "_signature", three_fields)
    kn.admit_names(persona, ["alpha"], "gate")
    st = os.stat(_db(persona))
    assert _names(persona) == {"alpha"}
    kn.admit_names(persona, ["bravo"], "gate")
    os.utime(_db(persona), ns=(st.st_atime_ns, st.st_mtime_ns))
    if os.stat(_db(persona)).st_size != st.st_size:
        pytest.skip("this filesystem changed the size; the same-stat case cannot be built")
    assert _names(persona) == {"alpha"}  # the bug the counter exists to prevent


def test_a_write_between_the_signature_and_the_row_read_is_seen_next_lookup(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONC-2: a write committed after the signature was taken and before the rows."""
    kn.admit_names(persona, ["alpha"], "gate")
    real = kn._read_rows
    fired: list[bool] = []

    def write_first(path: Path) -> kn.KnownNames:
        if not fired:
            fired.append(True)
            kn.admit_names(persona, ["late"], "gate")
        return real(path)

    monkeypatch.setattr(kn, "_read_rows", write_first)
    kn._reset_cache()
    assert kn.load_known_names(persona).names == frozenset({"alpha", "late"})
    assert kn.load_known_names(persona).names == frozenset({"alpha", "late"})


def test_a_write_between_the_row_read_and_the_cache_store_is_seen_next_lookup(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONC-2: a write committed after the rows were read and before they were stored."""
    kn.admit_names(persona, ["alpha"], "gate")
    real = kn._read_rows
    fired: list[bool] = []

    def write_after(path: Path) -> kn.KnownNames:
        rows = real(path)
        if not fired:
            fired.append(True)
            kn.admit_names(persona, ["late"], "gate")
        return rows

    monkeypatch.setattr(kn, "_read_rows", write_after)
    kn._reset_cache()
    assert kn.load_known_names(persona).names == frozenset({"alpha"})  # the read predates the write
    assert kn.load_known_names(persona).names == frozenset({"alpha", "late"})  # and it is not lost


# ---------------------------------------------------------------------------
# CONC-5 / INV-I13a: a locked file, a hot journal
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _exclusive_lock(persona: Path):
    """A writer mid-commit: an open exclusive transaction on the names file."""
    conn = _CONNECT(str(_db(persona)), isolation_level=None, timeout=0)
    conn.execute("BEGIN EXCLUSIVE")
    try:
        yield conn
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        conn.close()


def test_a_reader_during_a_commit_returns_the_previous_list_and_stores_nothing(
    persona: Path,
) -> None:
    kn.admit_names(persona, ["alpha"], "gate")
    before = kn.load_known_names(persona)
    key = str(_db(persona))
    cached_before = kn._cache[key]
    with _exclusive_lock(persona) as conn:
        conn.execute("INSERT INTO known_names VALUES ('bravo','bravo','gate','t')")
        assert kn.load_known_names(persona) is before
        assert kn._cache[key] is cached_before  # no new signature stored
        conn.execute("COMMIT")
    assert kn.load_known_names(persona).names == frozenset({"alpha", "bravo"})


def test_the_reader_opens_read_only_and_never_waits_on_a_lock(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kn.admit_names(persona, ["alpha"], "gate")
    kn._reset_cache()
    seen: list[tuple[str, dict]] = []
    real = sqlite3.connect

    def spy(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        seen.append((database, kwargs))
        return real(database, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(kn.sqlite3, "connect", spy)
    assert _names(persona) == {"alpha"}
    assert len(seen) == 1
    database, kwargs = seen[0]
    assert database.startswith("file:") and database.endswith("?mode=ro")
    assert kwargs["uri"] is True and kwargs["timeout"] == 0


def test_a_reader_with_no_previous_list_does_not_cache_an_error_as_empty(persona: Path) -> None:
    kn.admit_names(persona, ["alpha"], "gate")
    kn._reset_cache()
    with _exclusive_lock(persona):
        assert kn.load_known_names(persona) is kn.EMPTY
        assert str(_db(persona)) not in kn._cache
    assert kn.load_known_names(persona).names == frozenset({"alpha"})  # retried, not stuck empty


def test_a_writer_that_meets_a_locked_healthy_file_never_renames_it(
    persona: Path, quick_busy: None, caplog: pytest.LogCaptureFixture
) -> None:
    kn.admit_names(persona, ["alpha"], "gate")
    original = _db(persona).read_bytes()
    with _exclusive_lock(persona), caplog.at_level(logging.WARNING, logger=kn.logger.name):
        assert kn.admit_names(persona, ["bravo"], "gate") == []
    assert _aside(persona) == []
    assert _db(persona).read_bytes() == original
    assert any("write skipped" in r.message for r in caplog.records)
    assert _names(persona) == {"alpha"}
    assert kn.admit_names(persona, ["bravo"], "gate") == ["bravo"]  # the next write succeeds
    assert _names(persona) == {"alpha", "bravo"}


def test_writers_use_sqlites_default_busy_timeout_not_the_memories_db_constant(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S88: no timeout argument on any writer-side connect (SQLite's default, 5 s),
    # whatever memories.db's constant is set to.
    monkeypatch.setattr(dev_constants, "MEMORIES_DB_BUSY_TIMEOUT_S", 123.0)
    seen: list[dict] = []
    real = sqlite3.connect

    def spy(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        seen.append({"args": args, **kwargs})
        return real(database, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(kn.sqlite3, "connect", spy)
    _db(persona).write_bytes(_GARBAGE)  # exercises _insert_rows and _confirmed_corrupt
    assert kn.admit_names(persona, ["pretzel"], "tool") == ["pretzel"]
    assert len(seen) >= 3  # failed insert, quick_check, insert into the fresh file
    for call in seen:
        assert "timeout" not in call and call["args"] == ()
    assert real(str(_db(persona))).execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_a_locked_file_is_not_confirmed_corrupt(persona: Path, quick_busy: None) -> None:
    kn.admit_names(persona, ["alpha"], "gate")
    with _exclusive_lock(persona):
        assert kn._confirmed_corrupt(_db(persona)) is False
    assert kn._confirmed_corrupt(_db(persona)) is False
    _db(persona).write_bytes(_GARBAGE)
    assert kn._confirmed_corrupt(_db(persona)) is True


def _make_hot_journal(src: Path, dst_dir: Path) -> None:
    """Copy ``src`` and its journal mid-transaction, so ``dst_dir`` holds a hot journal.

    The writer's cache is tiny so its pages spill: the database file is already
    modified and a valid rollback journal exists when both files are copied.
    """
    conn = sqlite3.connect(str(src), isolation_level=None)
    conn.execute("PRAGMA cache_size=1")
    conn.execute("BEGIN")
    for i in range(3000):
        conn.execute(
            "INSERT INTO known_names VALUES (?, 'd', 'tool', 't')", (f"spill{i:05d}" + "x" * 200,)
        )
    journal = Path(str(src) + "-journal")
    assert journal.exists() and journal.stat().st_size > 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst_dir / src.name)
    shutil.copy(journal, dst_dir / journal.name)
    conn.execute("ROLLBACK")
    conn.close()


def test_a_read_against_a_hot_journal_returns_the_previous_list_and_does_not_crash(
    persona: Path, tmp_path: Path
) -> None:
    kn.admit_names(persona, ["alpha"], "gate")
    hot = tmp_path / "hot"
    _make_hot_journal(_db(persona), hot)
    # The copy is a persona directory whose writer crashed mid-transaction.
    assert kn.load_known_names(hot) is kn.EMPTY  # no previous list: empty, nothing cached
    assert str(_db(hot)) not in kn._cache
    # A previous list is kept.
    kn._cache[str(_db(hot))] = (None, kn.KnownNames.from_names(["earlier"]))
    assert kn.load_known_names(hot).names == frozenset({"earlier"})


def test_the_next_writer_recovers_a_hot_journal(persona: Path, tmp_path: Path) -> None:
    kn.admit_names(persona, ["alpha"], "gate")
    hot = tmp_path / "hot"
    _make_hot_journal(_db(persona), hot)
    assert kn.admit_names(hot, ["bravo"], "gate") == ["bravo"]
    assert _aside(hot) == []
    assert _names(hot) == {"alpha", "bravo"}  # the crashed transaction was rolled back


# ---------------------------------------------------------------------------
# INV-I13a: every writer holds the OS file lock
# ---------------------------------------------------------------------------


def test_the_write_happens_inside_the_os_file_lock(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    @contextlib.contextmanager
    def recording_lock(path: Path):
        events.append(f"lock {path.name}")
        try:
            yield True
        finally:
            events.append("unlock")

    real_insert = kn._insert_rows

    def recording_insert(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
        events.append("insert")
        real_insert(path, rows)

    monkeypatch.setattr(kn, "file_lock", recording_lock)
    monkeypatch.setattr(kn, "_insert_rows", recording_insert)
    assert kn.admit_names(persona, ["pretzel"], "tool") == ["pretzel"]
    assert events == [f"lock {_db(persona).name}", "insert", "unlock"]
    events.clear()
    kn.admit_names(persona, ["the"], "tool")  # rejected: no lock, no write
    assert events == []


def test_an_os_error_from_the_lock_is_logged_and_never_raised(
    persona: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    @contextlib.contextmanager
    def broken_lock(path: Path):
        raise PermissionError("no lock for you")
        yield  # pragma: no cover

    monkeypatch.setattr(kn, "file_lock", broken_lock)
    with caplog.at_level(logging.WARNING, logger=kn.logger.name):
        assert kn.admit_names(persona, ["pretzel"], "tool") == []
    assert any("could not write" in r.message for r in caplog.records)


def test_a_lone_surrogate_in_an_extracted_name_does_not_lose_the_name_or_raise(
    persona: Path,
) -> None:
    # A model's JSON escape can decode to half an emoji; SQLite cannot bind it.
    out = kn.admit_names(persona, ["Zoe\ud83d", "Wren"], "gate", now=T0)
    assert out == ["zoe", "wren"]
    assert _names(persona) == {"zoe", "wren"}
    # SQLite cannot bind the half emoji; UTF-8 "replace" turns it into "?".
    assert {r[0]: r[1] for r in _rows(persona)} == {"zoe": "Zoe?", "wren": "Wren"}


@pytest.mark.parametrize("exc", [sqlite3.InterfaceError("bad bind"), ValueError("bad value")])
def test_a_write_that_fails_for_any_database_or_value_reason_is_logged_not_raised(
    persona: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exc: Exception,
) -> None:
    def boom(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
        raise exc

    monkeypatch.setattr(kn, "_insert_rows", boom)
    with caplog.at_level(logging.WARNING, logger=kn.logger.name):
        assert kn.admit_names(persona, ["pretzel"], "gate") == []
    assert any("could not write" in r.message for r in caplog.records)


def test_an_unreadable_signature_keeps_the_previous_list_and_warns_once(
    persona: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    kn.admit_names(persona, ["alpha"], "gate")
    before = kn.load_known_names(persona)
    cached = dict(kn._cache)
    real_stat = os.stat

    def refuse(path: object, *a: object, **k: object) -> os.stat_result:
        if str(path) == str(_db(persona)):
            raise PermissionError("denied")
        return real_stat(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(kn.os, "stat", refuse)
    with caplog.at_level(logging.DEBUG, logger=kn.logger.name):
        for _ in range(3):
            assert kn.load_known_names(persona) is before  # not treated as "missing"
    assert kn._cache == cached
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "signature unavailable" in warnings[0].message


def test_a_file_the_reader_cannot_get_past_warns_once_but_plain_busy_does_not(
    persona: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    kn.admit_names(persona, ["alpha"], "gate")
    with caplog.at_level(logging.DEBUG, logger=kn.logger.name):
        with _exclusive_lock(persona):  # an ordinary writer mid-commit
            kn.load_known_names(persona)
            kn.load_known_names(persona)
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
        hot = tmp_path / "hot"  # a crashed writer's hot journal: not transient in practice
        _make_hot_journal(_db(persona), hot)
        for _ in range(3):
            kn.load_known_names(hot)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "cannot be read now" in warnings[0].message


def test_renaming_a_corrupt_file_aside_takes_its_journal_sidecar_with_it(persona: Path) -> None:
    # Called directly: on a real write SQLite itself disposes of an unusable
    # journal before the rename is reached, so the sidecar branch is exercised here.
    journal = Path(str(_db(persona)) + "-journal")
    _db(persona).write_bytes(_GARBAGE)
    journal.write_bytes(b"stale journal bytes")
    assert kn._rename_aside(_db(persona)) is True
    assert not _db(persona).exists() and not journal.exists()
    aside = _aside(persona)
    assert len(aside) == 2
    (moved,) = [p for p in aside if p.name.endswith("-journal")]
    assert moved.read_bytes() == b"stale journal bytes"
    (db_aside,) = [p for p in aside if not p.name.endswith("-journal")]
    assert db_aside.read_bytes() == _GARBAGE
    assert moved.name == db_aside.name + "-journal"


@pytest.mark.parametrize(
    ("entry", "lower"),
    [
        ("José", "jos"),
        ("Zoë", "zo"),
        ("Müller", "m ller"),
        ("Åsa Núñez", "sa n ez"),
        ("Jose\u0301", "jose"),  # decomposed (NFD): stored with the combining mark, not composed
        ("Zoe\u0308", "zoe"),
    ],
)
def test_the_display_form_keeps_accents_exactly_as_extracted(
    persona: Path, entry: str, lower: str
) -> None:
    # S87: matching stays on the shared ASCII tokenizer (#317 covers accented
    # Latin), but the stored display form loses nothing, so it survives #317.
    assert kn.admit_names(persona, [entry], "gate", now=T0) == [lower]
    assert _rows(persona) == [(lower, entry, "gate", T0.isoformat())]
    assert kn.normalize_name(entry) == lower


def test_the_display_form_keeps_case_spacing_and_accents_but_trims_the_ends(
    persona: Path,
) -> None:
    kn.admit_names(persona, ["  Zoë  Núñez\t"], "tool", now=T0)
    ((lower, display, _, _),) = _rows(persona)
    assert display == "Zoë  Núñez"
    assert lower == "zo n ez"


# ---------------------------------------------------------------------------
# CONC-4: writers in two processes
# ---------------------------------------------------------------------------

_WORKER = """
import sys, time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from brain.memory import known_names as kn

persona, go, tag, rnd = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
print("ready", flush=True)
while not go.exists():
    time.sleep(0.001)
stamp = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=(1 if tag == "b" else 0))
for k in range(5):
    kn.admit_names(persona, [f"{tag}name{rnd}x{k}", "sharedname"], "gate", now=stamp)
"""


def test_two_processes_writing_at_once_lose_nothing_and_keep_first_seen(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    env = {**os.environ, "PYTHONPATH": str(root)}
    seeded = datetime(2026, 9, 28, 8, 0, 0, tzinfo=UTC).isoformat()
    for rnd in range(20):
        persona = tmp_path / f"run{rnd}"
        persona.mkdir()
        # The shared name is already in the list, from an earlier day and another
        # source: neither writer may replace its row.
        kn.admit_names(persona, ["SharedName"], "reappraiser", now=datetime.fromisoformat(seeded))
        go = tmp_path / f"go{rnd}"
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _WORKER, str(persona), str(go), tag, str(rnd)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            for tag in ("a", "b")
        ]
        for p in procs:
            assert p.stdout is not None and p.stdout.readline().strip() == "ready"
        go.write_text("go")
        outs = [p.communicate(timeout=120) for p in procs]
        assert [p.returncode for p in procs] == [0, 0], outs
        rows = _rows(persona)
        expected = {f"{t}name{rnd}x{k}" for t in ("a", "b") for k in range(5)} | {"sharedname"}
        assert {r[0] for r in rows} == expected
        assert len(rows) == len(expected)
        assert [r for r in rows if r[0] == "sharedname"] == [
            ("sharedname", "SharedName", "reappraiser", seeded)
        ]


# ---------------------------------------------------------------------------
# P-30: the stopword set moved below the chat layer
# ---------------------------------------------------------------------------


def test_the_selector_and_the_admission_filter_share_one_stopword_set() -> None:
    from brain.chat import prompt

    assert prompt._RECALL_STOPWORDS is RECALL_STOPWORDS
    assert isinstance(RECALL_STOPWORDS, frozenset)
    assert {"will", "the", "no", "hello", "ok"} <= RECALL_STOPWORDS
    for content_word in ("issue", "first", "quick", "memory", "trigger", "signal", "logger"):
        assert content_word not in RECALL_STOPWORDS


def test_known_names_does_not_import_the_chat_layer() -> None:
    root = Path(__file__).resolve().parents[4]
    code = (
        "import sys\n"
        "import brain.memory.known_names\n"
        "bad = sorted(m for m in sys.modules if m.startswith('brain.chat'))\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root)},
        timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_file_name_is_a_dev_constant() -> None:
    assert dev_constants.KNOWN_NAMES_DB_FILENAME == "known_names.db"
    assert kn.known_names_path("/x/persona") == Path("/x/persona/known_names.db")
