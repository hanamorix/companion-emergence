"""Side-effect application tests for the pass-2 extractor.

Uses a real temp NELLBRAIN_HOME per the project no-mocked-state-files rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.chat.extractor import (
    CrystallisationCandidate,
    ExtractorOutput,
    MemoryWrite,
    ReflexAuditEntry,
    apply_side_effects,
)


@pytest.fixture
def persona_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh persona dir with a writable memories.db and empty audit logs."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona = tmp_path / "personas" / "nell"
    persona.mkdir(parents=True)
    return persona


def test_memory_writes_land_in_memorystore(persona_dir: Path):
    """Salience 0.0-1.0 must be translated to importance 0.0-10.0 per the
    contract documented on MemoryWrite."""
    from brain.memory.store import MemoryStore

    out = ExtractorOutput(
        memory_writes=[MemoryWrite(episode="searched for Loopy, nothing surfaced", salience=0.4)]
    )
    apply_side_effects(out, persona_dir=persona_dir)

    store = MemoryStore(persona_dir / "memories.db")
    try:
        recent = store.list_active(limit=10)
        match = next((m for m in recent if "Loopy" in m.content), None)
        assert match is not None, (
            f"no monologue-sourced memory found in {[m.content for m in recent]}"
        )
        # Importance should be salience * 10 = 4.0
        assert abs(match.importance - 4.0) < 0.001, (
            f"expected importance=4.0 (salience 0.4 * 10), got {match.importance}"
        )
        # memory_type should be "monologue"
        assert match.memory_type == "monologue", (
            f"expected memory_type='monologue', got {match.memory_type!r}"
        )
    finally:
        store.close()


def test_reflex_audit_writes_to_reflex_audit_jsonl(persona_dir: Path):
    out = ExtractorOutput(
        reflex_audit=[
            ReflexAuditEntry(tool="search_memories", reason="user referenced Loopy as known")
        ]
    )
    apply_side_effects(out, persona_dir=persona_dir)

    log = persona_dir / "reflex_audit.jsonl"
    assert log.exists()
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["tool"] == "search_memories"


def test_empty_output_writes_nothing(persona_dir: Path):
    """No memory file, no digest log, no audit log when output is empty."""
    apply_side_effects(ExtractorOutput(), persona_dir=persona_dir)

    assert not (persona_dir / "reflex_audit.jsonl").exists()


def test_emotion_delta_applies_memory_with_emotion(persona_dir: Path):
    """Emotion delta writes a memory carrying the emotion channels to MemoryStore.

    The emotion system derives state from memory aggregation — the monologue
    write acts as a tiny influence event. We verify a memory was committed
    with the emotion vector and that no exception was raised.
    """
    from brain.memory.store import MemoryStore

    out = ExtractorOutput(emotion_delta={"curiosity": 0.15})
    apply_side_effects(out, persona_dir=persona_dir)

    store = MemoryStore(persona_dir / "memories.db")
    try:
        recent = store.list_active(limit=10)
        emotion_memories = [m for m in recent if m.emotions.get("curiosity", 0.0) > 0]
        assert emotion_memories, (
            f"expected at least one memory with curiosity emotion, got: "
            f"{[m.to_dict() for m in recent]}"
        )
    finally:
        store.close()


def test_crystallisation_queues_soul_candidate(persona_dir: Path):
    """Crystallisation candidates land in soul_candidates.jsonl."""
    out = ExtractorOutput(
        crystallisation=[
            CrystallisationCandidate(
                theme="she notices grief as connection",
                evidence="monologue referenced missing Loopy and felt tender",
            )
        ]
    )
    apply_side_effects(out, persona_dir=persona_dir)

    candidates_path = persona_dir / "soul_candidates.jsonl"
    assert candidates_path.exists(), "soul_candidates.jsonl should be written"
    entry = json.loads(candidates_path.read_text().splitlines()[0])
    assert "grief as connection" in entry.get("text", ""), (
        f"expected theme text in candidate, got: {entry}"
    )
    # session_id="monologue" is used as the source identifier
    assert entry.get("session_id") == "monologue", (
        f"expected session_id='monologue', got: {entry.get('session_id')!r}"
    )


def test_emotion_delta_excludes_zero_channels_from_both_content_and_vector(persona_dir: Path):
    """Zero channels appear in neither the content string nor the stored vector."""
    from brain.memory.store import MemoryStore

    out = ExtractorOutput(emotion_delta={"curious": 0.1, "fear": 0.0})
    apply_side_effects(out, persona_dir=persona_dir)

    store = MemoryStore(persona_dir / "memories.db")
    try:
        recent = list(store.list_active(limit=10))
        emotion_mems = [m for m in recent if "monologue emotion influence" in m.content]
        assert emotion_mems, "no emotion-influence memory written"
        m = emotion_mems[0]
        assert "fear" not in m.content, (
            f"zero-value channel 'fear' should not appear in content: {m.content!r}"
        )
        assert "curious" in m.content, (
            f"non-zero channel 'curious' should appear in content: {m.content!r}"
        )
        assert "fear" not in m.emotions, (
            f"zero-value channel 'fear' should not be in emotions dict: {m.emotions}"
        )
    finally:
        store.close()


def test_partial_failure_is_isolated(persona_dir: Path):
    """If one apply step throws, others still run + the failure is logged."""
    # Make memories.db a directory so SQLite can't write to it.
    (persona_dir / "memories.db").mkdir()

    out = ExtractorOutput(
        memory_writes=[MemoryWrite(episode="will fail", salience=0.5)],
        reflex_audit=[ReflexAuditEntry(tool="search_memories", reason="will succeed")],
    )
    apply_side_effects(out, persona_dir=persona_dir)

    # Reflex audit should still have landed.
    assert (persona_dir / "reflex_audit.jsonl").exists()
    # Error log should have an entry for the failed memory write step.
    error_log = persona_dir / "extractor_errors.jsonl"
    assert error_log.exists()
    entry = json.loads(error_log.read_text().splitlines()[0])
    assert entry["step"] == "memory_writes"
