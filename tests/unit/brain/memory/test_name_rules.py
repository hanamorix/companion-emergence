"""The one shared definition of the name and contraction rules
(`brain.memory.name_rules`; name-recall fix S104, S121, S125, S126)."""

from __future__ import annotations

import pytest

from brain.memory import name_rules
from brain.memory.known_names import KnownNames, match_known_names
from brain.memory.name_rules import (
    contraction_stems,
    followed_by_apostrophe_t,
    is_sentence_start,
    is_stopword_name,
    stopword_name_occurrence_ok,
)
from brain.memory.relevance import names_in


@pytest.mark.parametrize(
    ("text", "stems"),
    [
        ("I don't know", {"don"}),
        ("it isn't fine, they didn't, we won't, he doesn't, ain't", {"isn", "didn", "won", "doesn", "ain"}),
        ("couldn't wouldn't shouldn't", {"couldn", "wouldn", "shouldn"}),
        ("Bob'll go, they're here, we've left, who'd know, I'm sure", {"bob", "they", "we", "who", "i"}),
        ("DON'T PANIC, WE'VE GONE, DIDN'T", {"don", "we", "didn"}),
        ("don’t and we’ve and they’re", {"don", "we", "they"}),
        # KEPT: the possessive stem, the word after an apostrophe, a stem before t not ending in n
        ("Roy's dog and Don's boat", set()),
        ("at five o'clock", set()),
        ("the do's and don'ts", set()),
        ("Al't rock'n'roll", set()),
    ],
)
def test_contraction_stems(text, stems):
    assert contraction_stems(text) == frozenset(stems)


def test_followed_by_apostrophe_t():
    assert followed_by_apostrophe_t("don't", 3)
    assert followed_by_apostrophe_t("DON'T go", 3)
    assert followed_by_apostrophe_t("don’t", 3)
    assert not followed_by_apostrophe_t("don'ts", 3)  # letters after the t
    assert not followed_by_apostrophe_t("Don's", 3)
    assert not followed_by_apostrophe_t("don", 3)
    assert not followed_by_apostrophe_t("don'", 3)


def test_is_stopword_name():
    assert is_stopword_name("Will")
    assert is_stopword_name("will")
    assert not is_stopword_name("Grace")
    assert not is_stopword_name("new york")


@pytest.mark.parametrize(
    ("text", "pos", "expected"),
    [
        ("Will you", 0, True),
        ("saw Will today", 4, False),
        ("he left. Will you", 9, True),
        ("really! Will you", 8, True),
        ("really? Will you", 8, True),
        ("first line\nWill you", 11, True),
        ("first line \n  Will you", 14, True),
        ("we said Will", 8, False),
        ("Bob: Will you", 5, False),  # S104-literal: a speaker label is not a sentence end
        ('"Will you?"', 1, False),
        ("so... Will you", 6, True),  # the ellipsis IS a period run
    ],
)
def test_is_sentence_start(text, pos, expected):
    assert is_sentence_start(text, pos) is expected


def test_stopword_name_occurrence_ok():
    ok = lambda t, s, e: stopword_name_occurrence_ok(t, [(s, e)])  # noqa: E731
    assert ok("saw Will today", 4, 8)
    assert not ok("Will you", 0, 4)
    assert not ok("I will go", 2, 6)  # lowercase
    assert not stopword_name_occurrence_ok("saw Will", [])


# S125, message side: `match_known_names` and what inherits it.
def test_a_single_word_name_is_not_matched_by_an_nt_stem():
    known = KnownNames.from_names(["Don", "new york", "don juan"])
    assert match_known_names("I don't know", known) == []
    assert match_known_names("Don came", known) == ["don"]
    assert match_known_names("DON'T PANIC", known) == []
    assert match_known_names("don’t", known) == []
    assert match_known_names("Don came. I don't know.", known) == ["don"]
    assert match_known_names("don'ts and dos", known) == ["don"]  # not an n't occurrence
    assert match_known_names("Don's boat", known) == ["don"]  # the possessive is the name
    assert match_known_names("in New York", known) == ["new york"]


def test_a_multi_word_name_is_unaffected_by_the_nt_skip():
    known = KnownNames.from_names(["don juan"])
    assert match_known_names("Don Juan sailed", known) == ["don juan"]
    assert match_known_names("don juan'", known) == ["don juan"]
    # the skip is for a SINGLE-word name only: a multi-word name ending an "x't" stays matched
    assert match_known_names("new york't", KnownNames.from_names(["new york"])) == ["new york"]


def test_names_in_inherits_the_nt_skip(tmp_path):
    from brain.memory import known_names as kn

    kn.admit_names(tmp_path, ["Don"], "tool")
    assert names_in(tmp_path, "I don't know") == []
    assert names_in(tmp_path, "Don came by") == ["don"]


def test_one_shared_definition_the_message_side_follows_name_rules(monkeypatch):
    known = KnownNames.from_names(["don"])
    assert match_known_names("don't", known) == []
    monkeypatch.setattr("brain.memory.known_names.followed_by_apostrophe_t", lambda *_a: False)
    assert match_known_names("don't", known) == ["don"]
    assert name_rules.followed_by_apostrophe_t("don't", 3)
