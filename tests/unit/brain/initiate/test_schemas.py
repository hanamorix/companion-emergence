"""Tests for brain.initiate.schemas — candidate + audit dataclasses."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from brain.initiate.schemas import (
    AuditRow,
    EmotionalSnapshot,
    InitiateCandidate,
    SemanticContext,
    StateTransition,
)


def test_emotional_snapshot_round_trips_to_dict():
    snap = EmotionalSnapshot(
        vector={"joy": 4, "longing": 7},
        rolling_baseline_mean=5.1,
        rolling_baseline_stdev=1.3,
        current_resonance=7.4,
        delta_sigma=1.77,
    )
    d = snap.to_dict()
    assert d["vector"] == {"joy": 4, "longing": 7}
    assert d["delta_sigma"] == 1.77
    assert EmotionalSnapshot.from_dict(d) == snap


def test_initiate_candidate_round_trips_to_jsonl():
    ts = "2026-05-11T14:32:04.123456+00:00"
    cand = InitiateCandidate(
        candidate_id="ic_2026-05-11T14-32-04_a3f1",
        ts=ts,
        kind="message",
        source="dream",
        source_id="dream_abc123",
        emotional_snapshot=EmotionalSnapshot(
            vector={"longing": 7},
            rolling_baseline_mean=5.0,
            rolling_baseline_stdev=1.0,
            current_resonance=7.4,
            delta_sigma=2.4,
        ),
        semantic_context=SemanticContext(
            linked_memory_ids=["m_xyz"],
            topic_tags=["dream", "workshop"],
        ),
        claimed_at=None,
    )
    line = cand.to_jsonl()
    reconstructed = InitiateCandidate.from_jsonl(line)
    assert reconstructed == cand


def test_audit_row_state_transitions_append():
    row = AuditRow(
        audit_id="ia_xyz",
        candidate_id="ic_abc",
        ts="2026-05-11T14:47:09+00:00",
        kind="message",
        subject="the dream",
        tone_rendered="the dream from this morning landed somewhere",
        decision="send_quiet",
        decision_reasoning="resonance is real but hour is late",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row.record_transition("delivered", "2026-05-11T14:47:09.5+00:00")
    row.record_transition("read", "2026-05-11T18:34:21+00:00")
    assert row.delivery is not None
    assert row.delivery["current_state"] == "read"
    assert len(row.delivery["state_transitions"]) == 2
    assert row.delivery["state_transitions"][0]["to"] == "delivered"


def test_initiate_candidate_id_generation():
    """candidate_id must be sortable and unique."""
    from brain.initiate.schemas import make_candidate_id

    a = make_candidate_id(datetime(2026, 5, 11, 14, 32, 4, tzinfo=timezone.utc))
    b = make_candidate_id(datetime(2026, 5, 11, 14, 32, 5, tzinfo=timezone.utc))
    assert a < b  # sortable
    assert a != b  # unique
    assert a.startswith("ic_")
