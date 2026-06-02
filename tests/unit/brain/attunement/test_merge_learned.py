"""Tests for store.merge_into_learned (canonical-key match + maturity update)."""
from __future__ import annotations

from pathlib import Path

from brain.attunement.schemas import Evidence, PatternCandidate
from brain.attunement.store import (
    BufferTurn,
    merge_into_learned,
    read_learned_patterns,
)


def _candidate(category: str, key: str, quote: str, turn_id: str) -> PatternCandidate:
    return PatternCandidate(
        category=category,
        canonical_key=key,
        description=f"desc for {key}",
        evidence=[Evidence(quote=quote, turn_id=turn_id)],
    )


def _turn(turn_id: str, content: str) -> BufferTurn:
    return BufferTurn(id=turn_id, content=content)


def test_new_candidate_creates_immature_pattern(tmp_path: Path) -> None:
    buffer = [_turn("t1", "hello world")]
    cand = _candidate("tone", "key-1", "hello", "t1")
    merge_into_learned(tmp_path, [cand], buffer, now_iso="2026-05-31T12:00:00Z")
    patterns = read_learned_patterns(tmp_path)
    assert len(patterns) == 1
    assert patterns[0].evidence_count == 1
    assert patterns[0].maturity == "immature"
    assert patterns[0].first_seen_at == "2026-05-31T12:00:00Z"


def test_matching_canonical_key_increments_evidence_count(tmp_path: Path) -> None:
    buffer1 = [_turn("t1", "first hello")]
    buffer2 = [_turn("t2", "second hello world")]
    merge_into_learned(
        tmp_path,
        [_candidate("tone", "key-1", "first hello", "t1")],
        buffer1,
        now_iso="2026-05-31T10:00:00Z",
    )
    merge_into_learned(
        tmp_path,
        [_candidate("tone", "key-1", "second hello world", "t2")],
        buffer2,
        now_iso="2026-05-31T12:00:00Z",
    )
    patterns = read_learned_patterns(tmp_path)
    assert len(patterns) == 1
    assert patterns[0].evidence_count == 2
    assert patterns[0].first_seen_at == "2026-05-31T10:00:00Z"
    assert patterns[0].last_confirmed_at == "2026-05-31T12:00:00Z"


def test_different_canonical_keys_create_separate_patterns(tmp_path: Path) -> None:
    buffer = [_turn("t1", "hello")]
    merge_into_learned(
        tmp_path,
        [
            _candidate("tone", "key-1", "hello", "t1"),
            _candidate("tone", "key-2", "hello", "t1"),
        ],
        buffer,
        now_iso="2026-05-31T12:00:00Z",
    )
    patterns = read_learned_patterns(tmp_path)
    assert len(patterns) == 2


def test_ungrounded_candidates_are_rejected_not_merged(tmp_path: Path) -> None:
    buffer = [_turn("t1", "hello")]
    # quote not in buffer
    bad = _candidate("tone", "key-1", "this text isn't here", "t1")
    good = _candidate("tone", "key-2", "hello", "t1")
    merge_into_learned(tmp_path, [bad, good], buffer, now_iso="2026-05-31T12:00:00Z")
    patterns = read_learned_patterns(tmp_path)
    assert len(patterns) == 1
    assert patterns[0].canonical_key == "key-2"


def test_rejected_candidate_logged_to_rejections_jsonl(tmp_path: Path) -> None:
    buffer = [_turn("t1", "hello")]
    bad = _candidate("tone", "key-1", "fake quote", "t1")
    merge_into_learned(tmp_path, [bad], buffer, now_iso="2026-05-31T12:00:00Z")
    rejections_path = tmp_path / "attunement_rejections.jsonl"
    assert rejections_path.exists()
    assert "key-1" in rejections_path.read_text()


def test_evidence_examples_accumulate_capped_at_five(tmp_path: Path) -> None:
    buffer = [_turn(f"t{i}", f"hello {i}") for i in range(7)]
    for i in range(7):
        merge_into_learned(
            tmp_path,
            [_candidate("tone", "key-1", f"hello {i}", f"t{i}")],
            buffer,
            now_iso=f"2026-05-31T12:00:{i:02d}Z",
        )
    patterns = read_learned_patterns(tmp_path)
    assert len(patterns) == 1
    assert len(patterns[0].examples) == 5  # truncated to cap


def test_new_pattern_examples_contains_evidence_quote(tmp_path: Path) -> None:
    """examples list is populated from evidence.quote, not a removed flat field."""
    buffer = [_turn("t1", "hello world")]
    cand = _candidate("tone", "key-1", "hello", "t1")
    merge_into_learned(tmp_path, [cand], buffer, now_iso="2026-05-31T12:00:00Z")
    patterns = read_learned_patterns(tmp_path)
    assert patterns[0].examples == ["hello"]


def test_last_entry_per_id_wins_on_read(tmp_path: Path) -> None:
    """Append-only with last-wins semantics — read filters older entries with same id."""
    buffer = [_turn("t1", "hello")]
    merge_into_learned(
        tmp_path,
        [_candidate("tone", "key-1", "hello", "t1")],
        buffer,
        now_iso="2026-05-31T10:00:00Z",
    )
    merge_into_learned(
        tmp_path,
        [_candidate("tone", "key-1", "hello", "t1")],
        buffer,
        now_iso="2026-05-31T12:00:00Z",
    )
    # File has 2 lines but read returns 1 pattern with merged state
    raw_lines = (tmp_path / "attunement" / "learned_patterns.jsonl").read_text().strip().split("\n")
    assert len(raw_lines) == 2
    patterns = read_learned_patterns(tmp_path)
    assert len(patterns) == 1
    assert patterns[0].last_confirmed_at == "2026-05-31T12:00:00Z"
