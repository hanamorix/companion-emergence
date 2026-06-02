from __future__ import annotations

from pathlib import Path

from brain.attunement.schemas import Evidence, PatternCandidate
from brain.attunement.store import (
    BufferTurn,
    mark_addressed,
    merge_into_learned,
    read_learned_patterns,
)


def _seed(persona_dir: Path, key: str, turn_id: str = "t1") -> str:
    cand = PatternCandidate(
        category="tone",
        canonical_key=key,
        description="warm and teasing",
        evidence=[Evidence(quote="hello love", turn_id=turn_id)],
    )
    buf = [BufferTurn(id=turn_id, content="hello love")]
    merge_into_learned(persona_dir, [cand], buf, now_iso="2026-06-02T10:00:00+00:00")
    from brain.attunement.schemas import pattern_id
    return pattern_id("tone", key)


def test_mark_addressed_stamps_last_addressed_at(tmp_path: Path):
    pid = _seed(tmp_path, "warm-teasing")
    mark_addressed(tmp_path, [pid], now_iso="2026-06-02T12:00:00+00:00")
    p = {x.id: x for x in read_learned_patterns(tmp_path)}[pid]
    assert p.last_addressed_at == "2026-06-02T12:00:00+00:00"
    # other fields preserved
    assert p.canonical_key == "warm-teasing"
    assert p.evidence_count == 1


def test_mark_addressed_ignores_unknown_ids(tmp_path: Path):
    pid = _seed(tmp_path, "warm-teasing")
    # unknown id must not raise and must not affect the known one
    mark_addressed(tmp_path, ["does-not-exist", pid], now_iso="2026-06-02T12:00:00+00:00")
    p = {x.id: x for x in read_learned_patterns(tmp_path)}[pid]
    assert p.last_addressed_at == "2026-06-02T12:00:00+00:00"


def test_mark_addressed_empty_list_is_noop(tmp_path: Path):
    pid = _seed(tmp_path, "warm-teasing")
    mark_addressed(tmp_path, [], now_iso="2026-06-02T12:00:00+00:00")
    p = {x.id: x for x in read_learned_patterns(tmp_path)}[pid]
    assert p.last_addressed_at is None
