"""Name-recall fix S166: a name with no ASCII letter or digit (wholly non-Latin, or only
punctuation or symbols) is admitted and stored with its display form, keyed by its raw
lower-cased string so such names do not collide on an empty key. It cannot match a message
until the recall tokenizer changes (#317): the matcher normalizes keys through it, so the
key drops out of the snapshot. Criteria B1a-B1h of
`name-recall-fix-records/small-followup-2/1.5-criteria.md`.

Synthetic, invented names only; real SQLite files in tmp dirs; no model, no network.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.engines import consolidation as cons
from brain.memory import known_names as kn
from brain.memory.name_rules import WORD_RE
from brain.tools.dispatch import dispatch
from brain.tools.impls.remove_known_name import remove_known_name
from brain.tools.schemas import ADD_NAME_TOOL_NAME

SOURCES = ["gate", "reappraiser", "tool"]


def _rows(persona: Path) -> list[tuple[str, str, str]]:
    conn = sqlite3.connect(str(kn.known_names_path(persona)))
    try:
        return conn.execute(
            "SELECT name_lower, display, source FROM known_names ORDER BY name_lower"
        ).fetchall()
    finally:
        conn.close()


def _q(persona: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(str(kn.known_names_path(persona)))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _admit(persona: Path, names: list[str], source: str) -> list[str]:
    if source == "tool":
        return [kn.name_key(n) for n in names if kn.confirm_readd(persona, n) == "added"]
    return kn.admit_names(persona, names, source)


# --------------------------------------------------------------------------
# B1a: admitted from every source, keyed by the raw lower-cased string
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source", SOURCES)
def test_b1a_a_name_with_no_ascii_letter_or_digit_is_admitted_and_keyed_by_its_raw_lower_case(
    tmp_path: Path, source: str
) -> None:
    names = ["田中", "Ольга", "...", "🐶"]
    assert sorted(_admit(tmp_path, names, source)) == sorted(["田中", "ольга", "...", "🐶"])
    assert _rows(tmp_path) == sorted(
        [("...", "...", source), ("ольга", "Ольга", source), ("田中", "田中", source), ("🐶", "🐶", source)]
    )


@pytest.mark.parametrize("source", SOURCES)
def test_b1a_two_different_such_names_are_two_rows_and_the_same_name_twice_is_one(
    tmp_path: Path, source: str
) -> None:
    _admit(tmp_path, ["田中", "山田", "田中", "  田中  "], source)
    assert [r[0] for r in _rows(tmp_path)] == ["山田", "田中"]


def test_b1a_the_display_form_keeps_its_case_and_is_stripped(tmp_path: Path) -> None:
    kn.admit_names(tmp_path, ["  ΩΜΕΓΑ  "], "gate")
    assert _rows(tmp_path) == [("ωμεγα", "ΩΜΕΓΑ", "gate")]


@pytest.mark.parametrize("source", SOURCES)
def test_b1a2_a_character_whose_lower_case_holds_an_ascii_letter_is_keyed_by_the_string_unchanged(
    tmp_path: Path, source: str
) -> None:
    # U+0130 lower-cases to "i" + a combining dot; U+212A KELVIN SIGN to "k"
    _admit(tmp_path, ["İ", "K"], source)
    assert [r[0] for r in _rows(tmp_path)] == ["İ", "K"]
    snapshot = kn.load_known_names(tmp_path)
    assert snapshot.names == frozenset(), "neither \"i\" nor \"k\" is in the snapshot"
    assert kn.match_known_names("I like k and i", snapshot) == []


def test_b1a2_the_kelvin_sign_does_not_collide_with_a_listed_k(tmp_path: Path) -> None:
    kelvin = "\u212a"
    kn.admit_names(tmp_path, ["K"], "gate")
    kn.admit_names(tmp_path, [kelvin], "gate")
    assert {r[0] for r in _rows(tmp_path)} == {kelvin, "k"}
    assert kn.load_known_names(tmp_path).names == frozenset({"k"})


# --------------------------------------------------------------------------
# B1b: nothing to store, and the length limit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source", SOURCES)
def test_b1b_an_empty_or_whitespace_only_entry_is_still_refused(tmp_path: Path, source: str) -> None:
    assert _admit(tmp_path, ["", "   ", "\n\t"], source) == []
    assert not kn.known_names_path(tmp_path).exists()


@pytest.mark.parametrize("source", SOURCES)
def test_b1b_a_101_character_non_latin_name_is_refused_and_100_is_admitted(tmp_path: Path, source: str) -> None:
    assert _admit(tmp_path, ["田" * 101], source) == []
    assert not kn.known_names_path(tmp_path).exists()
    assert _admit(tmp_path, ["   " + "田" * 100 + "   "], source) == ["田" * 100]


# --------------------------------------------------------------------------
# B1c: the matcher is untouched: such a key drops out of the snapshot
# --------------------------------------------------------------------------


def test_b1c_the_snapshot_holds_only_the_latin_names_and_matching_never_breaks(tmp_path: Path) -> None:
    kn.admit_names(tmp_path, ["田中", "Wren", "...", "Ольга", "New York"], "gate")
    snapshot = kn.load_known_names(tmp_path)
    assert snapshot.names == frozenset({"wren", "new york"})
    text = "田中 met Wren in New York ... Ольга"
    assert kn.match_known_names(text, snapshot) == ["wren", "new york"]
    assert kn.match_known_names("田中 only", snapshot) == []
    assert kn.names_fts_query(kn.match_known_names(text, snapshot)) == '"wren" OR "new york"'


def test_b1c_from_names_drops_a_raw_key() -> None:
    assert kn.KnownNames.from_names(["田中", "wren", "..."]).names == frozenset({"wren"})


# --------------------------------------------------------------------------
# B1d / B1e: she can remove and re-add it; the re-appraiser cannot remove hers
# --------------------------------------------------------------------------


def test_b1d_remove_and_readd_round_trip_through_the_tools(tmp_path: Path) -> None:
    ctx = {"persona_dir": tmp_path, "store": None, "hebbian": None}
    assert dispatch(ADD_NAME_TOOL_NAME, {"name": "田中"}, **ctx) == {"added": True, "name": "田中"}
    assert remove_known_name("田中", persona_dir=tmp_path) == {"removed": True, "name": "田中"}
    assert _rows(tmp_path) == []
    assert _q(tmp_path, "SELECT name_lower, rejected_by FROM rejected_names") == [("田中", "tool")]
    assert kn.admit_names(tmp_path, ["田中"], "gate") == [], "the rejects list outranks a mechanical add (S135)"
    reply = dispatch(ADD_NAME_TOOL_NAME, {"name": "田中"}, **ctx)
    assert reply["added"] is False and "previously removed" in reply["message"]
    assert _q(tmp_path, "SELECT name_lower FROM pending_readd") == [("田中",)]
    assert dispatch(ADD_NAME_TOOL_NAME, {"name": "田中"}, **ctx) == {"added": True, "name": "田中"}
    assert _rows(tmp_path) == [("田中", "田中", "tool")]
    assert _q(tmp_path, "SELECT name_lower FROM rejected_names") == []


def test_b1d_the_pending_question_lapses_on_a_different_next_call_and_holds_on_the_same_name(
    tmp_path: Path,
) -> None:
    kn.admit_names(tmp_path, ["田中"], "gate")
    remove_known_name("田中", persona_dir=tmp_path)
    assert kn.confirm_readd(tmp_path, "田中") == "asked"
    kn.note_tool_call(tmp_path, ADD_NAME_TOOL_NAME, {"name": " 田中 "}, add_tool_name=ADD_NAME_TOOL_NAME)
    assert _q(tmp_path, "SELECT name_lower FROM pending_readd") == [("田中",)], "the confirming call keeps it"
    kn.note_tool_call(tmp_path, ADD_NAME_TOOL_NAME, {"name": "山田"}, add_tool_name=ADD_NAME_TOOL_NAME)
    assert _q(tmp_path, "SELECT name_lower FROM pending_readd") == [], "another name lapses it"


def test_b1e_the_reappraiser_cannot_remove_a_name_she_added_and_a_gate_one_can_be_removed(tmp_path: Path) -> None:
    kn.confirm_readd(tmp_path, "田中")
    kn.admit_names(tmp_path, ["山田"], "gate")
    assert kn.remove_names(tmp_path, ["田中", "山田"], "reappraiser") == ["山田"]
    assert [r[0] for r in _rows(tmp_path)] == ["田中"]


def test_b1e_the_reappraisers_not_names_step_does_not_break_on_such_names(tmp_path: Path) -> None:
    kn.confirm_readd(tmp_path, "田中")
    kn.admit_names(tmp_path, ["Wren"], "gate")
    memory = SimpleNamespace(content="田中 and Wren")
    cons._remove_not_names(tmp_path, memory, ["Wren"], ["田中", "Wren"])  # type: ignore[arg-type]
    assert [r[0] for r in _rows(tmp_path)] == ["wren", "田中"]


# --------------------------------------------------------------------------
# B1f: a name with an ASCII word keeps its normalized key
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "key"),
    [("Wren 田中", "wren"), ("St. Louis", "st louis"), ("O'Brien", "o brien"), ("Wren", "wren"), ("I am", "i am")],
)
def test_b1f_a_name_with_an_ascii_word_keeps_its_normalized_key(given: str, key: str) -> None:
    assert kn.name_key(given) == key == kn.normalize_name(given)


def test_b1f_the_key_of_a_raw_name_never_holds_an_ascii_letter_or_digit() -> None:
    rng = random.Random(166)
    pool = list("田中Ольга🐶.,!?-_ \t\n\u00a0\u3000") + ["İ", "K", "é", "ß", "İ", "ǅ", "Σ", "ﬃ", "①", "٣"]
    for _ in range(3000):
        name = "".join(rng.choice(pool) for _ in range(rng.randint(1, 8)))
        key = kn.name_key(name)
        assert key == kn.name_key(key), "idempotent (remove_names re-keys what the matcher returns)"
        assert key == kn.normalize_name(name) or kn.normalize_name(key) == ""
        if not kn.normalize_name(name) and key:
            assert not WORD_RE.search(key), "a raw key holds no ASCII letter or digit"


# --------------------------------------------------------------------------
# S181 (S1a-S1g): runs of whitespace inside a name collapse to one space in its key
# --------------------------------------------------------------------------

SPELLINGS = ["田中 太郎", "田中  太郎", "田中\t太郎", "田中\n\n太郎", "田中\u00a0太郎", "田中\u3000太郎", "  田中   太郎  "]


@pytest.mark.parametrize("source", SOURCES)
def test_s1a_every_spelling_of_the_spacing_is_one_name_from_every_source(tmp_path: Path, source: str) -> None:
    for spelling in SPELLINGS:
        _admit(tmp_path, [spelling], source)
    rows = _rows(tmp_path)
    assert [r[0] for r in rows] == ["田中 太郎"], "one row, keyed by the single-spaced form"
    assert rows[0][1].replace("\u00a0", " ").replace("\u3000", " ").split() == ["田中", "太郎"]
    assert all(kn.name_key(x) == "田中 太郎" for x in SPELLINGS)


def test_s1a_the_display_form_is_not_collapsed(tmp_path: Path) -> None:
    kn.admit_names(tmp_path, ["  田中   太郎 "], "gate")
    assert _rows(tmp_path) == [("田中 太郎", "田中   太郎", "gate")]


@pytest.mark.parametrize("remove_with", SPELLINGS)
def test_s1b_removing_any_spelling_removes_the_name_and_lists_it_once(tmp_path: Path, remove_with: str) -> None:
    kn.admit_names(tmp_path, ["田中  太郎"], "gate")
    assert kn.remove_names(tmp_path, [remove_with], "tool") == ["田中 太郎"], "it returns the key it removed"
    assert _rows(tmp_path) == []
    assert _q(tmp_path, "SELECT name_lower FROM rejected_names") == [("田中 太郎",)]
    assert kn.admit_names(tmp_path, ["田中\t太郎"], "gate") == [], "the rejects list holds every spelling (S135)"


def test_s1b_the_readd_question_and_the_pending_compare_use_the_same_key(tmp_path: Path) -> None:
    kn.admit_names(tmp_path, ["田中 太郎"], "gate")
    remove_known_name("田中  太郎", persona_dir=tmp_path)
    assert kn.confirm_readd(tmp_path, "田中\u3000太郎") == "asked"
    kn.note_tool_call(tmp_path, ADD_NAME_TOOL_NAME, {"name": "田中   太郎"}, add_tool_name=ADD_NAME_TOOL_NAME)
    assert _q(tmp_path, "SELECT name_lower FROM pending_readd") == [("田中 太郎",)], "a spelling variant holds the question"
    kn.note_tool_call(tmp_path, ADD_NAME_TOOL_NAME, {"name": "山田 太郎"}, add_tool_name=ADD_NAME_TOOL_NAME)
    assert _q(tmp_path, "SELECT name_lower FROM pending_readd") == [], "another name lapses it"
    assert kn.confirm_readd(tmp_path, "田中 太郎") == "asked"
    assert kn.confirm_readd(tmp_path, "田中  太郎") == "added", "the confirming call may use another spelling"


@pytest.mark.parametrize("source", SOURCES)
def test_s1c_the_case_folding_exceptions_collapse_too(tmp_path: Path, source: str) -> None:
    _admit(tmp_path, ["\u0130  \u0130", "\u0130 \u0130", "\u212a\t\u212a"], source)
    keys = [r[0] for r in _rows(tmp_path)]
    assert sorted(keys) == sorted(["\u0130 \u0130", "\u212a \u212a"])
    assert all(not WORD_RE.search(k) for k in keys), "no ASCII letter or digit: out of the snapshot"
    assert kn.load_known_names(tmp_path).names == frozenset()


def test_s1c_the_key_property_holds_with_whitespace_runs() -> None:
    rng = random.Random(181)
    pool = list("田中Ольга🐶.,!?-_ \t\n\u00a0\u3000") + ["İ", "K", "é", "ß", "ﬃ", "①", "٣", "a", "7"]
    for _ in range(3000):
        name = "".join(rng.choice(pool) for _ in range(rng.randint(1, 10)))
        key = kn.name_key(name)
        assert key == kn.name_key(key)
        assert "  " not in key and not any(c.isspace() and c != " " for c in key), "no whitespace run, only single spaces"
        assert key == key.strip()
        if not kn.normalize_name(name) and key:
            assert not WORD_RE.search(key) and kn.normalize_name(key) == ""
        assert kn.name_key(name) == kn.name_key(" ".join(name.split())), "collapsing the name first changes nothing"


@pytest.mark.parametrize("source", SOURCES)
def test_s1d_names_with_an_ascii_word_are_unchanged(tmp_path: Path, source: str) -> None:
    _admit(tmp_path, ["Mary  Ann", "Mary Ann", "mary\tann", "MARY\u00a0ANN"], source)
    assert [r[0] for r in _rows(tmp_path)] == ["mary ann"]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("given", ["...", "!!!", "--", "🐶", "- - -", "* * *"])
def test_s1e_symbol_and_punctuation_only_names_stay_admitted(tmp_path: Path, source: str, given: str) -> None:
    _admit(tmp_path, [given], source)
    assert [r[0] for r in _rows(tmp_path)] == [" ".join(given.split()).lower()]


def test_s1e_spaced_punctuation_collapses_like_any_other_name(tmp_path: Path) -> None:
    kn.admit_names(tmp_path, ["- -", "-  -", "-\t-"], "gate")
    assert [r[0] for r in _rows(tmp_path)] == ["- -"]


def test_s1g_every_key_written_comes_from_name_key() -> None:
    """No migration (spec item 2): the only code that builds a stored key is `name_key`, through
    `_admit`, `remove_names` and `note_tool_call`; a row written by the new rule is single-spaced."""
    import inspect

    for fn in (kn._admit, kn.remove_names, kn.note_tool_call):
        assert "name_key(" in inspect.getsource(fn), fn.__name__
    assert "normalize_name(" not in inspect.getsource(kn._admit)
