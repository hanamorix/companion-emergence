"""Tests for the validate_grounded hallucination-control gate."""
from __future__ import annotations

from brain.attunement.schemas import Evidence, PatternCandidate
from brain.attunement.store import BufferTurn, validate_grounded


def _candidate(quote: str, turn_id: str, category: str = "tone") -> PatternCandidate:
    return PatternCandidate(
        category=category,
        canonical_key="key-x",
        description="some pattern",
        evidence=[Evidence(quote=quote, turn_id=turn_id)],
    )


def _turn(turn_id: str, content: str) -> BufferTurn:
    return BufferTurn(id=turn_id, content=content)


def _relational(pairs: list[tuple[str, str]]) -> PatternCandidate:
    return PatternCandidate(
        category="relational",
        canonical_key="rk",
        description="returns to X when Y",
        evidence=[Evidence(quote=q, turn_id=t) for q, t in pairs],
    )


def test_candidate_with_verified_quote_is_accepted() -> None:
    buffer = [_turn("t1", "The dog rolled over today and I cried a little.")]
    candidate = _candidate("The dog rolled over today", "t1")
    assert validate_grounded(candidate, buffer) is True


def test_candidate_with_missing_quote_is_rejected() -> None:
    buffer = [_turn("t1", "Hello world")]
    candidate = _candidate("This text does not exist anywhere", "t1")
    assert validate_grounded(candidate, buffer) is False


def test_candidate_with_quote_from_wrong_turn_is_rejected() -> None:
    buffer = [
        _turn("t1", "First turn content"),
        _turn("t2", "Second turn has the dog story"),
    ]
    candidate = _candidate("the dog story", "t1")
    assert validate_grounded(candidate, buffer) is False


def test_candidate_with_fabricated_quote_is_rejected() -> None:
    buffer = [_turn("t1", "The cat rolled over today")]
    candidate = _candidate("The dog rolled over today", "t1")  # cat→dog
    assert validate_grounded(candidate, buffer) is False


def test_normalisation_handles_whitespace() -> None:
    buffer = [_turn("t1", "She  said    hi")]
    candidate = _candidate("She said hi", "t1")
    assert validate_grounded(candidate, buffer) is True


def test_normalisation_handles_case() -> None:
    buffer = [_turn("t1", "She SAID hi")]
    candidate = _candidate("she said hi", "t1")
    assert validate_grounded(candidate, buffer) is True


def test_normalisation_handles_precomposed_vs_decomposed_unicode() -> None:
    # Buffer holds NFC (precomposed é, U+00E9); candidate holds NFD
    # (e + combining acute U+0301). Visually identical but byte-distinct.
    # NFC normalisation in _normalise() makes them compare equal.
    buffer = [_turn("t1", "café au lait")]
    candidate = _candidate("café au lait", "t1")
    assert validate_grounded(candidate, buffer) is True


def test_normalisation_handles_german_eszett() -> None:
    # casefold() maps ß → ss; .lower() would leave ß.
    buffer = [_turn("t1", "Straße ist breit")]
    candidate = _candidate("strasse ist breit", "t1")
    assert validate_grounded(candidate, buffer) is True


def test_validate_grounded_rejects_when_turn_id_not_in_buffer() -> None:
    buffer = [_turn("t1", "Hello")]
    candidate = _candidate("Hello", "t99")
    assert validate_grounded(candidate, buffer) is False


def test_validate_grounded_handles_empty_buffer() -> None:
    candidate = _candidate("anything", "t1")
    assert validate_grounded(candidate, []) is False


def test_relational_with_two_grounded_quotes_accepted() -> None:
    buffer = [_turn("t1", "work has been brutal"), _turn("t2", "anyway, my brother called")]
    cand = _relational([("work has been brutal", "t1"), ("my brother called", "t2")])
    assert validate_grounded(cand, buffer) is True


def test_relational_with_fabricated_second_turn_rejected() -> None:
    buffer = [_turn("t1", "work has been brutal"), _turn("t2", "anyway, my brother called")]
    cand = _relational([("work has been brutal", "t1"), ("invented quote", "t2")])
    assert validate_grounded(cand, buffer) is False
