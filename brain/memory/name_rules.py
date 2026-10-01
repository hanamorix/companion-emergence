"""The one shared definition of the name and contraction rules (name-recall fix).

Lives below the chat layer so the known-names matcher (``brain.memory.known_names``),
the forgotten-memory search (``brain.forgetting.graveyard``) and the names
increment's message-side rules all import the SAME functions and cannot drift
(ledger S121, S125, S126). Pure functions on text, no I/O.

- S125: a single-word listed name never matches an occurrence immediately
  followed by an apostrophe contraction ending in "t" (n't): "Don" is not
  matched by "don't".
- S126: "contraction fragments" are the stem before n't, 'll, 're, 've, 'd or
  'm. The stem before 's is KEPT (it is also the possessive: "Don's boat").
- S104/S121: a stopword-name (a listed name whose whole form is a recall
  stopword, e.g. "Will") matches only where written capitalized and not at
  the start of a sentence (text start, a newline before it, or after . ! ?).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from brain.memory.recall_stopwords import RECALL_STOPWORDS

__all__ = [
    "APOSTROPHES",
    "contraction_stems",
    "WORD_RE",
    "followed_by_apostrophe_t",
    "is_sentence_start",
    "is_stopword_name",
    "stopword_name_occurrence_ok",
]

#: The one word split (runs of ASCII letters and digits): the keyword search's,
#: the known-names matcher's and the forgotten-memory search's (S102).
WORD_RE = re.compile(r"[A-Za-z0-9]+")
#: The apostrophes the contraction rules recognise (straight and curly).
APOSTROPHES: tuple[str, ...] = ("'", "’")
_APOSTROPHES = APOSTROPHES
# The contraction endings whose stem is a fragment, not a word (S126). "t" is
# the n't ending (the stem then ends in "n"); the stem before 's is kept.
_FRAGMENT_SUFFIXES = ("t", "ll", "re", "ve", "d", "m")


def followed_by_apostrophe_t(text: str, end: int) -> bool:
    """True when the word ending at ``end`` is immediately followed by an
    apostrophe (straight or curly), a ``t`` in either case, and then a
    non-letter or the end of the text (an n't contraction, S125)."""
    if end + 1 >= len(text):
        return False
    if text[end] not in _APOSTROPHES or text[end + 1] not in ("t", "T"):
        return False
    return not (end + 2 < len(text) and text[end + 2].isalpha())


def _followed_by_fragment_suffix(text: str, end: int) -> bool:
    """True when the word ending at ``end`` is followed by an apostrophe and one
    of the fragment endings (n't, 'll, 're, 've, 'd, 'm; any case) and then a
    non-letter or the end of the text."""
    if end >= len(text) or text[end] not in _APOSTROPHES:
        return False
    rest = text[end + 1 :]
    for suffix in _FRAGMENT_SUFFIXES:
        if rest[: len(suffix)].lower() == suffix and not (
            len(rest) > len(suffix) and rest[len(suffix)].isalpha()
        ):
            return True
    return False


def contraction_stems(text: str) -> frozenset[str]:
    """The lower-cased stems before n't, 'll, 're, 've, 'd or 'm in ``text`` (S126).

    "don't" gives ``don``, "we've" gives ``we``. The stem before 's ("Roy's")
    and the word after an apostrophe ("o'clock") are not stems. Straight and
    curly apostrophes, any letter case. A stem before n't must end in "n".
    """
    out: set[str] = set()
    for m in WORD_RE.finditer(text):
        end = m.end()
        if not _followed_by_fragment_suffix(text, end):
            continue
        word = m.group().lower()
        rest = text[end + 1 : end + 2].lower()
        if rest == "t" and not word.endswith("n"):
            continue
        out.add(word)
    return frozenset(out)


def is_stopword_name(name: str) -> bool:
    """True when the name's whole normalized form (lower-cased words, single
    spaced) is a recall stopword: exactly the names S70 refuses to admit and
    S104 lets the kindled add with the tool, flagged."""
    return " ".join(w.lower() for w in WORD_RE.findall(name)) in RECALL_STOPWORDS


def is_sentence_start(text: str, pos: int) -> bool:
    """True when ``pos`` starts a sentence (S104): the start of the text, a
    newline in the whitespace run before it, or the previous non-space
    character is one of . ! ?"""
    i = pos - 1
    while i >= 0 and text[i].isspace():
        if text[i] == "\n":
            return True
        i -= 1
    if i < 0:
        return True
    return text[i] in ".!?"


def stopword_name_occurrence_ok(text: str, spans: Sequence[tuple[int, int]]) -> bool:
    """A stopword-name occurrence (the ``(start, end)`` spans of its words in
    ``text``) counts only if every word starts with an uppercase letter and the
    first word is not at a sentence start (S104, S121)."""
    if not spans:
        return False
    if any(not text[start : start + 1].isupper() for start, _ in spans):
        return False
    return not is_sentence_start(text, spans[0][0])
