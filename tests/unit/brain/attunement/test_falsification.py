"""Tests for negative-evidence falsification of learned patterns."""
from __future__ import annotations

from pathlib import Path

from brain.attunement.schemas import PatternCandidate
from brain.attunement.store import (
    BufferTurn,
    apply_contradiction,
    merge_into_learned,
    read_learned_patterns,
)


def _seed_known_pattern(tmp_path: Path) -> None:
    buffer = [BufferTurn(id=f"t{i}", content=f"hello {i}") for i in range(10)]
    for i in range(10):
        merge_into_learned(
            tmp_path,
            [PatternCandidate(
                category="tone",
                canonical_key="key-1",
                description="warm when greeting",
                evidence_quote=f"hello {i}",
                evidence_turn_id=f"t{i}",
            )],
            buffer,
            now_iso=f"2026-05-01T10:{i:02d}:00Z",
        )


def test_contradiction_decrements_evidence_count(tmp_path: Path) -> None:
    _seed_known_pattern(tmp_path)
    pid = read_learned_patterns(tmp_path)[0].id
    apply_contradiction(tmp_path, pid, now_iso="2026-05-31T12:00:00Z")
    patterns = read_learned_patterns(tmp_path)
    assert patterns[0].evidence_count == 9


def test_pattern_falsified_below_floor_returns_to_forming(tmp_path: Path) -> None:
    _seed_known_pattern(tmp_path)
    pid = read_learned_patterns(tmp_path)[0].id
    for i in range(3):
        apply_contradiction(tmp_path, pid, now_iso=f"2026-05-31T12:{i:02d}:00Z")
    patterns = read_learned_patterns(tmp_path)
    assert patterns[0].evidence_count == 7
    assert patterns[0].maturity == "forming"


def test_sustained_contradiction_marks_falsified(tmp_path: Path) -> None:
    _seed_known_pattern(tmp_path)
    pid = read_learned_patterns(tmp_path)[0].id
    for i in range(8):
        apply_contradiction(tmp_path, pid, now_iso=f"2026-05-31T12:{i:02d}:00Z")
    patterns = read_learned_patterns(tmp_path)
    assert patterns[0].evidence_count == 2
    assert patterns[0].maturity == "falsified"
    assert patterns[0].falsified_at == "2026-05-31T12:07:00Z"


def test_falsified_pattern_can_recover_with_confirmation(tmp_path: Path) -> None:
    """A falsified pattern is not deleted — new evidence can revive it."""
    _seed_known_pattern(tmp_path)
    pid = read_learned_patterns(tmp_path)[0].id
    for i in range(8):
        apply_contradiction(tmp_path, pid, now_iso=f"2026-05-31T12:{i:02d}:00Z")
    # Now feed a fresh confirmation
    buffer = [BufferTurn(id="t-new", content="hello again")]
    merge_into_learned(
        tmp_path,
        [PatternCandidate(
            category="tone",
            canonical_key="key-1",
            description="warm when greeting",
            evidence_quote="hello again",
            evidence_turn_id="t-new",
        )],
        buffer,
        now_iso="2026-06-01T10:00:00Z",
    )
    patterns = read_learned_patterns(tmp_path)
    assert patterns[0].evidence_count == 3
    assert patterns[0].maturity == "forming"
    assert patterns[0].falsified_at is None  # cleared by confirmation


def test_contradiction_does_not_go_below_zero(tmp_path: Path) -> None:
    buffer = [BufferTurn(id="t1", content="hello")]
    merge_into_learned(
        tmp_path,
        [PatternCandidate(
            category="tone",
            canonical_key="key-1",
            description="x",
            evidence_quote="hello",
            evidence_turn_id="t1",
        )],
        buffer,
        now_iso="2026-05-31T10:00:00Z",
    )
    pid = read_learned_patterns(tmp_path)[0].id
    for i in range(5):
        apply_contradiction(tmp_path, pid, now_iso=f"2026-05-31T12:{i:02d}:00Z")
    patterns = read_learned_patterns(tmp_path)
    assert patterns[0].evidence_count == 0
    assert patterns[0].maturity == "falsified"


def test_apply_contradiction_to_unknown_pattern_is_noop(tmp_path: Path) -> None:
    apply_contradiction(tmp_path, "no-such-id", now_iso="2026-05-31T12:00:00Z")
    assert read_learned_patterns(tmp_path) == []
