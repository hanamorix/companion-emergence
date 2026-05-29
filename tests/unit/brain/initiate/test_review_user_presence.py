"""Test that run_initiate_review_tick threads user_presence to _process_one_candidate."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.initiate.user_pattern import UserPresence


def test_run_initiate_review_tick_passes_user_presence_to_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """user_presence passed to run_initiate_review_tick must reach _process_one_candidate."""
    from brain.initiate.review import run_initiate_review_tick

    captured_presence: list = []

    def _fake_process(persona_dir, candidate, *, provider, voice_template, now,
                      user_name, companion_name, user_presence=None):
        captured_presence.append(user_presence)

    monkeypatch.setattr("brain.initiate.review.run_calibration_closer_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.initiate.review.run_resonance_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.initiate.ambient.build_outbound_recall_block", lambda *a, **k: None)
    monkeypatch.setattr("brain.initiate.review._process_one_candidate", _fake_process)
    monkeypatch.setattr("brain.initiate.review.append_d_call_row", lambda *a, **k: None)
    monkeypatch.setattr("brain.initiate.review.detect_drift", lambda *a, **k: None)

    from brain.initiate.schemas import InitiateCandidate, SemanticContext
    fake_candidate = InitiateCandidate(
        candidate_id="c1",
        ts=datetime.now(UTC).isoformat(),
        kind="message",
        source="dream",
        source_id="src1",
        semantic_context=SemanticContext(),
    )
    monkeypatch.setattr("brain.initiate.review.read_candidates", lambda *a, **k: [fake_candidate])

    from brain.initiate.d_call_schema import DCallRow
    from brain.initiate.reflection import DDecision, DReflectionResult
    fake_decision = DDecision(
        candidate_index=0, decision="send_notify", reason="ok", confidence="high"
    )
    fake_result = DReflectionResult(decisions=[fake_decision], tick_note=None)
    fake_dcall = DCallRow(
        d_call_id="dc1",
        ts=datetime.now(UTC).isoformat(),
        tick_id="t1",
        model_tier_used="haiku",
        candidates_in=1,
        promoted_out=1,
        filtered_out=0,
        latency_ms=10,
        tokens_input=0,
        tokens_output=0,
    )
    monkeypatch.setattr("brain.initiate.review.reflection_run", lambda *a, **k: (fake_result, fake_dcall))

    presence = UserPresence(
        silence_days=4.0, ignore_streak=0, likely_active=True, response_lag_p50=None
    )
    run_initiate_review_tick(
        tmp_path,
        provider=MagicMock(),
        voice_template="",
        user_presence=presence,
    )

    assert len(captured_presence) >= 1, "_process_one_candidate must have been called"
    assert captured_presence[0] is presence, "user_presence must be forwarded"
