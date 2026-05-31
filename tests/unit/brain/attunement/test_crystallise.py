"""Tests for crystallise.check_crystallisations — first-cross feed-event detection."""
from __future__ import annotations

from pathlib import Path

from brain.attunement.crystallise import check_crystallisations
from brain.attunement.schemas import SCHEMA_VERSION, LearnedPattern
from brain.attunement.store import _append_pattern


def _pattern(
    id_: str,
    evidence_count: int,
    maturity: str,
    crystallised_at: str | None = None,
) -> LearnedPattern:
    return LearnedPattern(
        id=id_,
        category="tone",
        canonical_key=f"key-{id_}",
        description="x",
        evidence_count=evidence_count,
        maturity=maturity,
        first_seen_at="2026-05-01T00:00:00Z",
        last_confirmed_at="2026-05-31T12:00:00Z",
        last_addressed_at=None,
        crystallised_at=crystallised_at,
        falsified_at=None,
        examples=[],
        schema_version=SCHEMA_VERSION,
    )


def test_first_cross_into_known_emits_event(tmp_path: Path) -> None:
    _append_pattern(tmp_path, _pattern("p1", evidence_count=10, maturity="known"))
    events = check_crystallisations(tmp_path, now_iso="2026-05-31T12:00:00Z")
    assert len(events) == 1
    assert events[0].pattern_id == "p1"


def test_already_crystallised_pattern_does_not_re_emit(tmp_path: Path) -> None:
    _append_pattern(
        tmp_path,
        _pattern("p1", evidence_count=15, maturity="known", crystallised_at="2026-05-15T00:00:00Z"),
    )
    events = check_crystallisations(tmp_path, now_iso="2026-05-31T12:00:00Z")
    assert events == []


def test_forming_pattern_does_not_emit(tmp_path: Path) -> None:
    _append_pattern(tmp_path, _pattern("p1", evidence_count=5, maturity="forming"))
    events = check_crystallisations(tmp_path, now_iso="2026-05-31T12:00:00Z")
    assert events == []


def test_crystallisation_writes_back_crystallised_at(tmp_path: Path) -> None:
    from brain.attunement.store import read_learned_patterns
    _append_pattern(tmp_path, _pattern("p1", evidence_count=10, maturity="known"))
    check_crystallisations(tmp_path, now_iso="2026-05-31T12:00:00Z")
    patterns = read_learned_patterns(tmp_path)
    assert patterns[0].crystallised_at == "2026-05-31T12:00:00Z"


def test_check_is_idempotent_after_initial_crystallisation(tmp_path: Path) -> None:
    _append_pattern(tmp_path, _pattern("p1", evidence_count=10, maturity="known"))
    check_crystallisations(tmp_path, now_iso="2026-05-31T12:00:00Z")
    events = check_crystallisations(tmp_path, now_iso="2026-05-31T13:00:00Z")
    assert events == []  # already crystallised


def test_multiple_patterns_crystallising_in_same_check_each_emit(tmp_path: Path) -> None:
    _append_pattern(tmp_path, _pattern("p1", evidence_count=10, maturity="known"))
    _append_pattern(tmp_path, _pattern("p2", evidence_count=12, maturity="known"))
    events = check_crystallisations(tmp_path, now_iso="2026-05-31T12:00:00Z")
    assert {e.pattern_id for e in events} == {"p1", "p2"}
