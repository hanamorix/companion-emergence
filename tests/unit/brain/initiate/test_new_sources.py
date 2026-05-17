"""Tests for the v0.0.10 new emitter gates and emitters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.initiate.new_sources import GateThresholds, load_gate_thresholds


@dataclass(frozen=True)
class _FakeReflexFiring:
    """Test double for the fields gate_reflex_firing inspects."""

    pattern_id: str
    confidence: float
    flinch_intensity: float
    linked_memory_ids: list[str]
    triggered_by_companion_outbound: bool
    ts: datetime


def test_load_gate_thresholds_returns_defaults_when_no_file(tmp_path: Path):
    persona = tmp_path / "fresh"
    persona.mkdir()
    t = load_gate_thresholds(persona)
    assert t.reflex_confidence_min == 0.70
    assert t.reflex_flinch_intensity_min == 0.60
    assert t.research_maturity_min == 0.75
    assert t.research_topic_overlap_min == 0.30
    assert t.research_freshness_minutes == 30


def test_load_gate_thresholds_overrides_from_persona_file(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "gate_thresholds.json").write_text('{"reflex_confidence_min": 0.5}')
    t = load_gate_thresholds(persona)
    assert t.reflex_confidence_min == 0.5
    # Unset fields use defaults.
    assert t.reflex_flinch_intensity_min == 0.60


def test_load_gate_thresholds_non_dict_json_returns_defaults(tmp_path: Path):
    """A gate_thresholds.json file that contains valid JSON but isn't a dict
    falls back to defaults rather than crashing."""
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "gate_thresholds.json").write_text("[]")
    t = load_gate_thresholds(persona)
    assert t.reflex_confidence_min == 0.70  # default


def test_write_gate_rejection_appends_jsonl(tmp_path: Path):
    from datetime import UTC, datetime

    from brain.initiate.new_sources import write_gate_rejection

    persona = tmp_path / "p"
    write_gate_rejection(
        persona,
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
        source="reflex_firing",
        source_id="r1",
        gate_name="confidence_min",
        threshold_value=0.70,
        observed_value=0.5,
    )
    write_gate_rejection(
        persona,
        ts=datetime(2026, 5, 12, 10, 1, 0, tzinfo=UTC),
        source="research_completion",
        source_id="t9",
        gate_name="topic_overlap_min",
        threshold_value=0.30,
        observed_value=0.10,
    )
    path = persona / "gate_rejections.jsonl"
    rows = [json.loads(line) for line in path.read_text().strip().split("\n")]
    assert len(rows) == 2
    assert rows[0]["source"] == "reflex_firing"
    assert rows[0]["gate_name"] == "confidence_min"
    assert rows[0]["observed_value"] == 0.5
    assert rows[1]["source"] == "research_completion"


def test_check_shared_meta_gates_blocks_in_rest_state(tmp_path: Path):
    from datetime import UTC, datetime

    from brain.initiate.new_sources import check_shared_meta_gates

    persona = tmp_path / "p"
    persona.mkdir()
    allowed, reason = check_shared_meta_gates(
        persona,
        source="reflex_firing",
        now=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
        is_rest_state=True,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "rest_state"


def test_check_shared_meta_gates_blocks_on_per_source_anti_flood(tmp_path: Path):
    from datetime import UTC, datetime, timedelta

    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.new_sources import check_shared_meta_gates
    from brain.initiate.schemas import SemanticContext

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    # Pre-seed a recent reflex_firing candidate.
    emit_initiate_candidate(
        persona,
        kind="message",
        source="reflex_firing",
        source_id="r_prev",
        semantic_context=SemanticContext(),
        now=now - timedelta(minutes=10),
    )
    allowed, reason = check_shared_meta_gates(
        persona,
        source="reflex_firing",
        now=now,
        is_rest_state=False,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "per_source_anti_flood"


def test_check_shared_meta_gates_blocks_on_queue_depth(tmp_path: Path):
    from datetime import UTC, datetime, timedelta

    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.new_sources import check_shared_meta_gates
    from brain.initiate.schemas import SemanticContext

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    # Pre-seed 6 candidates of mixed sources, all OUTSIDE the anti-flood window
    # so per-source anti-flood doesn't fire first.
    for i in range(6):
        emit_initiate_candidate(
            persona,
            kind="message",
            source="dream",
            source_id=f"d{i}",
            semantic_context=SemanticContext(),
            now=now - timedelta(hours=1, minutes=i),
        )
    allowed, reason = check_shared_meta_gates(
        persona,
        source="reflex_firing",
        now=now,
        is_rest_state=False,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "queue_depth_max"


def test_check_shared_meta_gates_passes(tmp_path: Path):
    from datetime import UTC, datetime

    from brain.initiate.new_sources import check_shared_meta_gates

    persona = tmp_path / "p"
    persona.mkdir()
    allowed, reason = check_shared_meta_gates(
        persona,
        source="reflex_firing",
        now=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
        is_rest_state=False,
        thresholds=GateThresholds(),
    )
    assert allowed is True
    assert reason is None


def test_gate_reflex_firing_passes_when_above_thresholds(tmp_path: Path):
    from brain.initiate.new_sources import gate_reflex_firing

    persona = tmp_path / "p"
    persona.mkdir()
    firing = _FakeReflexFiring(
        pattern_id="p1",
        confidence=0.80,
        flinch_intensity=0.70,
        linked_memory_ids=["m1"],
        triggered_by_companion_outbound=False,
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    allowed, reason = gate_reflex_firing(
        persona,
        firing=firing,
        thresholds=GateThresholds(),
    )
    assert allowed is True
    assert reason is None


def test_gate_reflex_firing_blocks_on_low_confidence(tmp_path: Path):
    from brain.initiate.new_sources import gate_reflex_firing

    persona = tmp_path / "p"
    persona.mkdir()
    firing = _FakeReflexFiring(
        pattern_id="p2",
        confidence=0.50,  # below 0.70
        flinch_intensity=0.80,
        linked_memory_ids=[],
        triggered_by_companion_outbound=False,
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    allowed, reason = gate_reflex_firing(
        persona,
        firing=firing,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "confidence_min"


def test_gate_reflex_firing_blocks_on_low_flinch(tmp_path: Path):
    from brain.initiate.new_sources import gate_reflex_firing

    persona = tmp_path / "p"
    persona.mkdir()
    firing = _FakeReflexFiring(
        pattern_id="p3",
        confidence=0.90,
        flinch_intensity=0.30,
        linked_memory_ids=[],
        triggered_by_companion_outbound=False,
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    allowed, reason = gate_reflex_firing(
        persona,
        firing=firing,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "flinch_intensity_min"


def test_gate_reflex_firing_blocks_on_anti_feedback(tmp_path: Path):
    from brain.initiate.new_sources import gate_reflex_firing

    persona = tmp_path / "p"
    persona.mkdir()
    firing = _FakeReflexFiring(
        pattern_id="p4",
        confidence=0.80,
        flinch_intensity=0.70,
        linked_memory_ids=[],
        triggered_by_companion_outbound=True,  # anti-feedback guard
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    allowed, reason = gate_reflex_firing(
        persona,
        firing=firing,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "anti_feedback"


def test_gate_reflex_firing_blocks_on_pattern_anti_flood(tmp_path: Path):
    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.new_sources import gate_reflex_firing
    from brain.initiate.schemas import SemanticContext

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    # Pre-seed a candidate for the same pattern within the 4h window.
    emit_initiate_candidate(
        persona,
        kind="message",
        source="reflex_firing",
        source_id="r_old",
        semantic_context=SemanticContext(source_meta={"pattern_id": "shared_pattern"}),
        now=now - timedelta(hours=2),
    )
    firing = _FakeReflexFiring(
        pattern_id="shared_pattern",
        confidence=0.85,
        flinch_intensity=0.65,
        linked_memory_ids=[],
        triggered_by_companion_outbound=False,
        ts=now,
    )
    allowed, reason = gate_reflex_firing(
        persona,
        firing=firing,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "pattern_anti_flood"


def test_emit_reflex_firing_candidate_writes_queue_row(tmp_path: Path):
    from brain.initiate.emit import read_candidates
    from brain.initiate.new_sources import emit_reflex_firing_candidate

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    firing = _FakeReflexFiring(
        pattern_id="p_emit",
        confidence=0.85,
        flinch_intensity=0.70,
        linked_memory_ids=["m_a", "m_b"],
        triggered_by_companion_outbound=False,
        ts=now,
    )
    emit_reflex_firing_candidate(persona, firing=firing, firing_log_id="rfx_001", now=now)
    out = read_candidates(persona)
    assert len(out) == 1
    c = out[0]
    assert c.source == "reflex_firing"
    assert c.source_id == "rfx_001"
    assert c.semantic_context.linked_memory_ids == ["m_a", "m_b"]
    assert c.semantic_context.source_meta == {
        "pattern_id": "p_emit",
        "confidence": 0.85,
        "flinch_intensity": 0.70,
    }


@dataclass(frozen=True)
class _FakeResearchThread:
    thread_id: str
    topic: str
    maturity_score: float
    summary_excerpt: str
    linked_memory_ids: list[str]
    completed_at: datetime
    previously_linked_to_audit: bool


def test_gate_research_completion_passes(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion

    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t1",
        topic="quiet rivers",
        maturity_score=0.80,
        summary_excerpt="...",
        linked_memory_ids=[],
        completed_at=now - timedelta(minutes=15),
        previously_linked_to_audit=False,
    )
    allowed, reason = gate_research_completion(
        persona,
        thread=thread,
        now=now,
        topic_overlap_score=0.40,
        thresholds=GateThresholds(),
    )
    assert allowed is True
    assert reason is None


def test_gate_research_completion_blocks_on_low_maturity(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion

    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t2",
        topic="x",
        maturity_score=0.50,
        summary_excerpt="...",
        linked_memory_ids=[],
        completed_at=now,
        previously_linked_to_audit=False,
    )
    allowed, reason = gate_research_completion(
        persona,
        thread=thread,
        now=now,
        topic_overlap_score=0.40,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "maturity_min"


def test_gate_research_completion_blocks_on_previously_linked(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion

    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t3",
        topic="x",
        maturity_score=0.90,
        summary_excerpt="...",
        linked_memory_ids=[],
        completed_at=now,
        previously_linked_to_audit=True,
    )
    allowed, reason = gate_research_completion(
        persona,
        thread=thread,
        now=now,
        topic_overlap_score=0.40,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "previously_linked"


def test_gate_research_completion_blocks_on_low_topic_overlap(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion

    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t4",
        topic="x",
        maturity_score=0.90,
        summary_excerpt="...",
        linked_memory_ids=[],
        completed_at=now,
        previously_linked_to_audit=False,
    )
    allowed, reason = gate_research_completion(
        persona,
        thread=thread,
        now=now,
        topic_overlap_score=0.10,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "topic_overlap_min"


def test_gate_research_completion_blocks_on_stale(tmp_path: Path):
    from brain.initiate.new_sources import gate_research_completion

    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t5",
        topic="x",
        maturity_score=0.90,
        summary_excerpt="...",
        linked_memory_ids=[],
        completed_at=now - timedelta(hours=2),  # outside 30-min freshness
        previously_linked_to_audit=False,
    )
    allowed, reason = gate_research_completion(
        persona,
        thread=thread,
        now=now,
        topic_overlap_score=0.40,
        thresholds=GateThresholds(),
    )
    assert allowed is False
    assert reason == "freshness_window"


def test_emit_research_completion_candidate_writes_queue_row(tmp_path: Path):
    from brain.initiate.emit import read_candidates
    from brain.initiate.new_sources import emit_research_completion_candidate

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    thread = _FakeResearchThread(
        thread_id="t_emit",
        topic="midnight gardens",
        maturity_score=0.85,
        summary_excerpt="A study of...",
        linked_memory_ids=["m_x"],
        completed_at=now,
        previously_linked_to_audit=False,
    )
    emit_research_completion_candidate(
        persona,
        thread=thread,
        topic_overlap_score=0.45,
        now=now,
    )
    out = read_candidates(persona)
    assert len(out) == 1
    c = out[0]
    assert c.source == "research_completion"
    assert c.source_id == "t_emit"
    assert c.semantic_context.source_meta == {
        "thread_topic": "midnight gardens",
        "maturity_score": 0.85,
        "summary_excerpt": "A study of...",
        "topic_overlap_score": 0.45,
    }


def test_load_gate_thresholds_includes_resonance_defaults(tmp_path):
    persona = tmp_path / "fresh"
    persona.mkdir()
    t = load_gate_thresholds(persona)
    assert t.recall_resonance_z_threshold == 2.5
    assert t.recall_resonance_staleness_min_days == 7
    assert t.recall_resonance_top_n == 50
    assert t.recall_resonance_ema_alpha == 0.08
    assert t.recall_resonance_bootstrap_min_count == 10
    assert t.recall_resonance_anti_flood_hours == 24.0


def test_load_gate_thresholds_resonance_override(tmp_path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "gate_thresholds.json").write_text('{"recall_resonance_z_threshold": 3.0}')
    t = load_gate_thresholds(persona)
    assert t.recall_resonance_z_threshold == 3.0
    assert t.recall_resonance_top_n == 50  # default preserved
