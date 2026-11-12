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


def _promote_pending(persona_dir: Path) -> None:
    """Promote the persona's pending-candidate queue into memories.db.

    memory-consolidation migration: the extractor's monologue
    / monologue_soul_candidate writes are GATED types — apply_side_effects enqueues
    them as candidates rather than writing memories.db rows. Tests that assert the
    end-state DB memory run the consolidation gate with a promote-all classifier
    (candidate id preserved) so their store.list_active()/store.get() reads hold.
    """
    from brain.engines.consolidation import Decision, run_consolidation
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        run_consolidation(
            store,
            persona_dir=persona_dir,
            classifier=lambda _c, _ctx: Decision("new"),
        )
    finally:
        store.close()

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
    _promote_pending(persona_dir)

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
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["tool"] == "search_memories"


def test_empty_output_writes_nothing(persona_dir: Path):
    """No memory file, no digest log, no audit log when output is empty."""
    apply_side_effects(ExtractorOutput(), persona_dir=persona_dir)

    assert not (persona_dir / "reflex_audit.jsonl").exists()


def test_crystallisation_queues_soul_candidate(persona_dir: Path):
    """Crystallisation candidates land in soul_candidates.jsonl."""
    out = ExtractorOutput(
        crystallisation=[
            CrystallisationCandidate(
                theme="she notices grief as connection",
                evidence="monologue referenced missing Loopy and felt tender",
                importance=8,
            )
        ]
    )
    apply_side_effects(out, persona_dir=persona_dir)

    candidates_path = persona_dir / "soul_candidates.jsonl"
    assert candidates_path.exists(), "soul_candidates.jsonl should be written"
    entry = json.loads(candidates_path.read_text(encoding="utf-8").splitlines()[0])
    assert "grief as connection" in entry.get("text", ""), (
        f"expected theme text in candidate, got: {entry}"
    )
    # session_id="monologue" is used as the source identifier
    assert entry.get("session_id") == "monologue", (
        f"expected session_id='monologue', got: {entry.get('session_id')!r}"
    )


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
    entry = json.loads(error_log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["step"] == "memory_writes"


def test_emotion_delta_writes_no_row_and_queues_nothing(persona_dir: Path):
    """The pass-2 emotion delta no longer becomes a standalone `monologue_emotion` row: the old aggregate that
    pooled those rows is gone, so such a row would only be a recallable memory with no reader of its emotions."""
    from brain.memory.pending import PendingQueue
    from brain.memory.store import MemoryStore

    apply_side_effects(ExtractorOutput(emotion_delta={"curiosity": 0.15}), persona_dir=persona_dir)

    store = MemoryStore(persona_dir / "memories.db")
    try:
        assert [m for m in store.list_active(limit=50) if m.memory_type == "monologue_emotion"] == []
    finally:
        store.close()
    assert PendingQueue(persona_dir).read_recent("monologue_emotion", limit=5) == []
    assert not (persona_dir / "extractor_errors.jsonl").exists()


# ---------------------------------------------------------------------------
# F1 — Queue the moment, not the label (soul-candidate context fix)
# ---------------------------------------------------------------------------


def test_crystallisation_queues_theme_and_evidence_combined(persona_dir: Path):
    """Queued record text must be 'theme — evidence'; placeholder memory must agree.

    Before F1 the queued text was only c.theme (the short label), burying
    c.evidence in a placeholder memory the reviewer never fetched.  After F1
    both surfaces carry the combined string so the soul-review LLM actually
    sees the source moment.
    """
    from brain.memory.store import MemoryStore

    out = ExtractorOutput(
        crystallisation=[
            CrystallisationCandidate(
                theme="Ordinary trust",
                evidence="she let the goodnight be ordinary instead of performing it",
                importance=8,
            )
        ]
    )
    apply_side_effects(out, persona_dir=persona_dir)

    _promote_pending(persona_dir)

    expected_text = "Ordinary trust — she let the goodnight be ordinary instead of performing it"

    # soul_candidates.jsonl must carry the combined string.
    candidates_path = persona_dir / "soul_candidates.jsonl"
    assert candidates_path.exists(), "soul_candidates.jsonl should be written"
    entry = json.loads(candidates_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry.get("text") == expected_text, (
        f"queued text should be 'theme — evidence'; got: {entry.get('text')!r}"
    )

    # The placeholder memory (memory_id in the candidate) must also carry the same string.
    memory_id = entry.get("memory_id")
    assert memory_id, f"candidate entry missing memory_id: {entry}"

    store = MemoryStore(persona_dir / "memories.db")
    try:
        mem = store.get(memory_id)
        assert mem is not None, f"placeholder memory {memory_id!r} not found in MemoryStore"
        assert mem.content == expected_text, (
            f"placeholder memory content should equal combined text; "
            f"got: {mem.content!r}"
        )
    finally:
        store.close()


def test_crystallisation_skips_empty_evidence(persona_dir: Path):
    """A candidate whose evidence is whitespace-only must be skipped entirely.

    Pydantic's min_length=1 rejects truly empty strings and the field_validator
    rejects whitespace-only strings too.  We use model_construct to bypass
    validation so we can test the runtime guard in _apply_crystallisation
    directly (the guard exists as defence-in-depth for any future path that
    bypasses pydantic).  No record must be appended to soul_candidates.jsonl
    and no placeholder memory must be written.
    """
    from brain.memory.store import MemoryStore

    c = CrystallisationCandidate.model_construct(theme="Hollow fragment", evidence="   ")
    # Build an ExtractorOutput bypassing field validation so the whitespace-only
    # candidate reaches _apply_crystallisation.
    out = ExtractorOutput.model_construct(
        memory_writes=[],
        emotion_delta={},
        crystallisation=[c],
        reflex_audit=[],
    )

    apply_side_effects(out, persona_dir=persona_dir)

    candidates_path = persona_dir / "soul_candidates.jsonl"
    assert not candidates_path.exists(), (
        "soul_candidates.jsonl must NOT be written for a whitespace-only evidence candidate"
    )

    store = MemoryStore(persona_dir / "memories.db")
    try:
        recent = store.list_active(limit=10)
        soul_mems = [m for m in recent if m.memory_type == "monologue_soul_candidate"]
        assert not soul_mems, (
            f"no monologue_soul_candidate memory should be written; found: "
            f"{[m.to_dict() for m in soul_mems]}"
        )
    finally:
        store.close()


def test_extractor_prompt_asks_for_grounded_evidence():
    """_SYSTEM_PROMPT must instruct the LLM to provide 1-2 grounded sentences for evidence.

    The old prompt used '<short>' for both theme and evidence, producing
    abstract labels that could never ground a soul-review decision.
    """
    from brain.chat.extractor import _SYSTEM_PROMPT

    assert "1-2 grounded sentences" in _SYSTEM_PROMPT, (
        f"_SYSTEM_PROMPT should ask for '1-2 grounded sentences' in the crystallisation "
        f"evidence field; current prompt:\n{_SYSTEM_PROMPT}"
    )


# ---------------------------------------------------------------------------
# B5 — extractor-scored importance gates queueing
# ---------------------------------------------------------------------------


def test_crystallisation_importance_below_threshold_not_queued(persona_dir: Path):
    """importance=6 < DEFAULT_SOUL_THRESHOLD(8) → candidate NOT queued."""
    out = ExtractorOutput(
        crystallisation=[
            CrystallisationCandidate(
                theme="weak theme",
                evidence="not very formative, just a passing thought",
                importance=6,
            )
        ]
    )
    apply_side_effects(out, persona_dir=persona_dir)

    candidates_path = persona_dir / "soul_candidates.jsonl"
    assert not candidates_path.exists(), (
        "soul_candidates.jsonl must NOT be written for importance < DEFAULT_SOUL_THRESHOLD"
    )


def test_crystallisation_importance_at_threshold_is_queued(persona_dir: Path):
    """importance=8 == DEFAULT_SOUL_THRESHOLD → candidate IS queued."""
    out = ExtractorOutput(
        crystallisation=[
            CrystallisationCandidate(
                theme="at threshold theme",
                evidence="this moment matters and should be queued",
                importance=8,
            )
        ]
    )
    apply_side_effects(out, persona_dir=persona_dir)

    candidates_path = persona_dir / "soul_candidates.jsonl"
    assert candidates_path.exists(), "soul_candidates.jsonl must exist for importance == 8"
    entry = json.loads(candidates_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry.get("status") == "auto_pending"
    assert "at threshold theme" in entry.get("text", "")


# ---------------------------------------------------------------------------
# B4 — dedup by theme
# ---------------------------------------------------------------------------


def test_crystallisation_within_batch_dedup_same_theme(persona_dir: Path):
    """Two candidates with identical theme in one apply call → only ONE queued."""
    out = ExtractorOutput(
        crystallisation=[
            CrystallisationCandidate(
                theme="Recurring solitude",
                evidence="first moment — she sat alone again",
                importance=9,
            ),
            CrystallisationCandidate(
                theme="Recurring solitude",
                evidence="second moment — still alone, different night",
                importance=9,
            ),
        ]
    )
    apply_side_effects(out, persona_dir=persona_dir)

    candidates_path = persona_dir / "soul_candidates.jsonl"
    assert candidates_path.exists()
    lines = candidates_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, (
        f"Expected 1 queued candidate (dedup), got {len(lines)}: {lines}"
    )


def test_crystallisation_rejected_theme_does_not_block_requeue(persona_dir: Path):
    """A candidate whose theme matches a *rejected* record must still be queued.

    Only auto_pending status blocks re-queueing; terminal statuses (rejected,
    accepted, expired) are cleared — the theme may be worth revisiting.
    """
    # Pre-populate soul_candidates.jsonl with a rejected record for the same theme.
    rejected_record = {
        "memory_id": "mem-old-001",
        "text": "Recurring solitude — an older moment, now rejected",
        "label": "observation",
        "importance": 8,
        "session_id": "monologue",
        "queued_at": "2026-06-01T10:00:00+00:00",
        "status": "rejected",
    }
    candidates_path = persona_dir / "soul_candidates.jsonl"
    candidates_path.write_text(json.dumps(rejected_record) + "\n", encoding="utf-8")

    out = ExtractorOutput(
        crystallisation=[
            CrystallisationCandidate(
                theme="Recurring solitude",
                evidence="fresh moment — a new instance of solitude worth reviewing",
                importance=9,
            )
        ]
    )
    apply_side_effects(out, persona_dir=persona_dir)

    lines = candidates_path.read_text(encoding="utf-8").strip().splitlines()
    # Original rejected + 1 new auto_pending = 2 lines total
    assert len(lines) == 2, (
        f"Expected 2 lines (rejected + new auto_pending), got {len(lines)}: {lines}"
    )
    new_entry = json.loads(lines[1])
    assert new_entry.get("status") == "auto_pending"
    assert "fresh moment" in new_entry.get("text", "")


# ---------------------------------------------------------------------------
# Emotion rework B1 PR 2: signed changes once the engine is live
# ---------------------------------------------------------------------------


def test_signed_change_values_keeps_sign_caps_and_skips_zero():
    from brain.chat.extractor import signed_change_values

    assert signed_change_values({"joy": -0.2, "awe": 0.5, "grief": 0.0, "fear": -1.0}) == pytest.approx(
        {"joy": -2.0, "awe": 3.0, "fear": -3.0}
    )


def _go_live(persona_dir: Path, **seed: float):
    """A live persona with no state file applies nothing, so seed a memory first, then write the switch."""
    from datetime import UTC, datetime

    from brain.emotion import engine
    from brain.memory.store import Memory, MemoryStore
    from tests.unit.brain.emotion._engine_support import make_live

    store = MemoryStore(persona_dir / "memories.db")
    try:
        store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=seed or {"joy": 5.0}))
    finally:
        store.close()
    make_live(persona_dir, datetime.now(UTC))
    return engine.engine_for(persona_dir)


def _settled(eng):
    from datetime import UTC, datetime

    eng.note_active(datetime.now(UTC))
    return eng.read()


def _rows_of(persona_dir: Path, memory_type: str) -> list:
    from brain.memory.pending import PendingQueue
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        stored = store.list_by_type(memory_type, active_only=False)
    finally:
        store.close()
    return stored + PendingQueue(persona_dir).read_recent(memory_type, limit=50)


def test_live_pass2_lowers_the_feeling_and_writes_no_row(persona_dir: Path):
    eng = _go_live(persona_dir, joy=5.0)
    before = _settled(eng)["joy"].short
    apply_side_effects(ExtractorOutput(emotion_delta={"joy": -0.2}), persona_dir=persona_dir)
    assert _settled(eng)["joy"].short < before
    assert _rows_of(persona_dir, "monologue_emotion") == []


@pytest.mark.no_base_failure(
    reason="pins that the live fold keeps the vocabulary filter (B1 PR 2); main drops an unregistered name too, on the old path"
)
def test_live_pass2_unregistered_name_folds_nothing(persona_dir: Path):
    from brain.emotion import engine_files

    eng = _go_live(persona_dir, joy=5.0)
    before = _settled(eng)
    apply_side_effects(ExtractorOutput(emotion_delta={"zorblesque": 0.5}), persona_dir=persona_dir)
    after = _settled(eng)
    assert "zorblesque" not in after and after == before
    events, _ = engine_files.read_queue(persona_dir, from_line=0)
    assert [e for e in events if e.source == "monologue_emotion"] == []
    assert _rows_of(persona_dir, "monologue_emotion") == []
