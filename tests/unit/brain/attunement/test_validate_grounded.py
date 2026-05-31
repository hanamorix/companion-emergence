"""Tests for the validate_grounded hallucination-control gate."""
from __future__ import annotations

from brain.attunement.schemas import PatternCandidate
from brain.attunement.store import BufferTurn, validate_grounded


def _candidate(quote: str, turn_id: str) -> PatternCandidate:
    return PatternCandidate(
        category="tone",
        canonical_key="key-x",
        description="some pattern",
        evidence_quote=quote,
        evidence_turn_id=turn_id,
    )


def _turn(turn_id: str, content: str) -> BufferTurn:
    return BufferTurn(id=turn_id, content=content)


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


def test_validate_grounded_rejects_when_turn_id_not_in_buffer() -> None:
    buffer = [_turn("t1", "Hello")]
    candidate = _candidate("Hello", "t99")
    assert validate_grounded(candidate, buffer) is False


def test_validate_grounded_handles_empty_buffer() -> None:
    candidate = _candidate("anything", "t1")
    assert validate_grounded(candidate, []) is False
