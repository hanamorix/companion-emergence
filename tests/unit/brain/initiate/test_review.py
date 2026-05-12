"""Tests for brain.initiate.review — orchestrator tick."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.review import run_initiate_review_tick
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext


def _snap() -> EmotionalSnapshot:
    return EmotionalSnapshot(
        vector={"longing": 7},
        rolling_baseline_mean=5.0,
        rolling_baseline_stdev=1.0,
        current_resonance=7.4,
        delta_sigma=2.4,
    )


def _ctx() -> SemanticContext:
    return SemanticContext(linked_memory_ids=["m_xyz"], topic_tags=["dream"])


def _fake_provider(decision: str = "send_quiet") -> MagicMock:
    """Provider that returns canned outputs for subject/tone/decision."""
    provider = MagicMock()
    responses = [
        "the dream from this morning",  # subject
        "the dream from this morning landed somewhere",  # tone
        f'{{"decision": "{decision}", "reasoning": "x"}}',  # decision
    ]
    provider.complete = MagicMock(side_effect=responses)
    return provider


def _promote_all_reflection_run(candidates, *, deps):
    """Stub for reflection_run that promotes every candidate unconditionally.

    Used by existing tests that pre-date D-reflection wiring so they
    continue to exercise composition behaviour without hitting real LLMs.
    """
    from brain.initiate.d_call_schema import DCallRow, make_d_call_id
    from brain.initiate.reflection import DDecision, DReflectionResult

    decisions = [
        DDecision(i, "promote", "test stub", "high")
        for i in range(len(candidates))
    ]
    result = DReflectionResult(decisions=decisions, tick_note=None)
    dcall = DCallRow(
        d_call_id=make_d_call_id(deps.now),
        ts=deps.now.isoformat(),
        tick_id=deps.tick_id,
        model_tier_used="haiku",
        candidates_in=len(candidates),
        promoted_out=len(candidates),
        filtered_out=0,
        latency_ms=0,
        tokens_input=0,
        tokens_output=0,
    )
    return result, dcall


def test_review_tick_processes_queued_candidate_writes_audit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all_reflection_run)
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    run_initiate_review_tick(
        tmp_path,
        provider=_fake_provider("send_quiet"),
        voice_template="be warm",
        cap_per_tick=3,
    )
    audit_path = tmp_path / "initiate_audit.jsonl"
    assert audit_path.exists()
    lines = audit_path.read_text().splitlines()
    assert len(lines) == 1
    assert '"decision": "send_quiet"' in lines[0]


def test_review_tick_removes_processed_candidate_from_queue(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all_reflection_run)
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    run_initiate_review_tick(
        tmp_path, provider=_fake_provider(), voice_template="x", cap_per_tick=3,
    )
    assert read_candidates(tmp_path) == []


def test_review_tick_respects_cap_per_tick(tmp_path: Path, monkeypatch) -> None:
    """Only `cap_per_tick` candidates are processed in one call."""
    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all_reflection_run)
    for i in range(5):
        emit_initiate_candidate(
            tmp_path,
            kind="message",
            source="dream",
            source_id=f"dream_{i}",
            emotional_snapshot=_snap(),
            semantic_context=_ctx(),
        )
    provider = MagicMock()
    canned = ["subject", "tone", '{"decision": "send_quiet", "reasoning": "x"}'] * 3
    provider.complete = MagicMock(side_effect=canned)
    run_initiate_review_tick(
        tmp_path, provider=provider, voice_template="x", cap_per_tick=3,
    )
    # 5 emitted, 3 processed, 2 remaining
    assert len(read_candidates(tmp_path)) == 2
    audit_lines = (tmp_path / "initiate_audit.jsonl").read_text().splitlines()
    assert len(audit_lines) == 3


def test_review_tick_gate_blocks_send_records_hold(tmp_path: Path, monkeypatch) -> None:
    """When decision = send_notify but gate denies (blackout), audit shows hold."""
    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all_reflection_run)
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    provider = _fake_provider("send_notify")
    blackout_time = datetime(2026, 5, 11, 1, 30, tzinfo=UTC)
    with patch("brain.initiate.review.datetime") as mock_dt:
        mock_dt.now = MagicMock(return_value=blackout_time)
        mock_dt.fromisoformat = datetime.fromisoformat
        run_initiate_review_tick(
            tmp_path,
            provider=provider,
            voice_template="x",
            cap_per_tick=3,
            now=blackout_time,
        )
    audit_line = (tmp_path / "initiate_audit.jsonl").read_text().strip()
    assert '"decision": "hold"' in audit_line
    assert "blackout" in audit_line


def test_review_tick_handles_compose_exception_as_error_decision(
    tmp_path: Path, monkeypatch
) -> None:
    """A composition failure produces decision=error, candidate not requeued."""
    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all_reflection_run)
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    provider = MagicMock(complete=MagicMock(side_effect=RuntimeError("boom")))
    # generate() is called by D's LLMCall wrapper; complete() by composition.
    # We want generate to succeed (so D promotes) but complete to fail (so
    # composition errors). Provide generate as a pass-through that returns
    # canned JSON so D doesn't escalate, then composition errors via complete.
    provider.generate = MagicMock(return_value='{"decisions":[]}')
    run_initiate_review_tick(
        tmp_path, provider=provider, voice_template="x", cap_per_tick=3,
    )
    audit_line = (tmp_path / "initiate_audit.jsonl").read_text().strip()
    assert '"decision": "error"' in audit_line
    # Candidate is dropped from the queue (the fresh emission next event
    # will rejoin if still relevant).
    assert read_candidates(tmp_path) == []


def test_review_tick_no_op_when_queue_empty(tmp_path: Path) -> None:
    """Empty queue → no audit writes, no errors, no LLM calls."""
    provider = MagicMock(complete=MagicMock())
    run_initiate_review_tick(
        tmp_path, provider=provider, voice_template="x", cap_per_tick=3,
    )
    assert not (tmp_path / "initiate_audit.jsonl").exists()
    provider.complete.assert_not_called()


def test_review_tick_publishes_initiate_delivered_on_send(
    tmp_path: Path, monkeypatch
) -> None:
    """When the decision is to send, the review tick MUST publish an
    ``initiate_delivered`` event so the frontend banner pipeline wakes up.

    The frontend ChatPanel subscribes to this event on the bridge /events
    stream; without the publish, the brain's outreach is invisible to the
    user even though the audit row says "delivered".
    """
    from brain.bridge import events

    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all_reflection_run)
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )

    captured: list[dict] = []
    events.set_publisher(captured.append)
    try:
        run_initiate_review_tick(
            tmp_path,
            provider=_fake_provider("send_notify"),
            voice_template="be warm",
            cap_per_tick=3,
        )
    finally:
        events.set_publisher(None)

    delivered = [e for e in captured if e.get("type") == "initiate_delivered"]
    assert len(delivered) == 1, f"expected 1 initiate_delivered event, got: {captured}"
    event = delivered[0]
    assert event["urgency"] == "notify"
    assert event["state"] == "delivered"
    assert event["audit_id"].startswith("ia_")
    assert isinstance(event["body"], str) and event["body"]
    assert isinstance(event["timestamp"], str) and event["timestamp"]


def test_review_tick_does_not_publish_on_hold(tmp_path: Path, monkeypatch) -> None:
    """Hold decisions must NOT publish initiate_delivered — the user
    should only see banners for outreach the brain actually committed to."""
    from brain.bridge import events

    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all_reflection_run)
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )

    captured: list[dict] = []
    events.set_publisher(captured.append)
    try:
        run_initiate_review_tick(
            tmp_path,
            provider=_fake_provider("hold"),
            voice_template="be warm",
            cap_per_tick=3,
        )
    finally:
        events.set_publisher(None)

    delivered = [e for e in captured if e.get("type") == "initiate_delivered"]
    assert delivered == []


def test_run_initiate_review_tick_skips_d_when_queue_empty(
    tmp_path: Path, monkeypatch
) -> None:
    """No candidates → no D call, no composition."""
    called = {"d": 0}

    def fake_run(candidates, *, deps):
        called["d"] += 1
        raise AssertionError("D should not be called on empty queue")

    monkeypatch.setattr("brain.initiate.review.reflection_run", fake_run)
    persona = tmp_path / "persona"
    persona.mkdir()
    provider = MagicMock()
    run_initiate_review_tick(persona, provider=provider, voice_template="x", cap_per_tick=3)
    assert called["d"] == 0


def test_run_initiate_review_tick_demotes_filtered_to_draft(
    tmp_path: Path, monkeypatch
) -> None:
    """D-filter decision → candidate goes to draft_space.md, NOT composition."""
    from brain.initiate.d_call_schema import DCallRow, make_d_call_id
    from brain.initiate.reflection import DDecision, DReflectionResult

    persona = tmp_path / "p"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    emit_initiate_candidate(
        persona,
        kind="message",
        source="dream",
        source_id="d_x",
        semantic_context=SemanticContext(),
        now=now,
    )

    def fake_reflection_run(candidates, *, deps):
        result = DReflectionResult(
            decisions=[DDecision(0, "filter", "private weather", "high")],
            tick_note=None,
        )
        dcall = DCallRow(
            d_call_id=make_d_call_id(now),
            ts=now.isoformat(),
            tick_id=deps.tick_id,
            model_tier_used="haiku",
            candidates_in=1,
            promoted_out=0,
            filtered_out=1,
            latency_ms=10,
            tokens_input=10,
            tokens_output=10,
        )
        return result, dcall

    compose_called: list = []

    def fake_process_one(persona_dir, candidate, *, provider, voice_template, now):
        compose_called.append(candidate.candidate_id)

    monkeypatch.setattr("brain.initiate.review.reflection_run", fake_reflection_run)
    monkeypatch.setattr("brain.initiate.review._process_one_candidate", fake_process_one)

    provider = MagicMock()
    run_initiate_review_tick(persona, provider=provider, voice_template="x", cap_per_tick=3, now=now)

    assert read_candidates(persona) == []
    assert (persona / "draft_space.md").exists()
    assert compose_called == []


def test_three_consecutive_failures_promote_all_fallback(tmp_path, monkeypatch):
    """After 3 consecutive timeout/provider_error failures across the same
    candidate cohort, fall through to 'promote all' so candidates aren't stranded."""
    from datetime import UTC, datetime, timedelta

    from brain.initiate.d_call_schema import DCallRow
    from brain.initiate.emit import emit_initiate_candidate, read_candidates
    from brain.initiate.reflection import DReflectionResult
    from brain.initiate.review import run_initiate_review_tick
    from brain.initiate.schemas import SemanticContext

    persona = tmp_path / "p"
    base_time = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)

    emit_initiate_candidate(
        persona, kind="message", source="dream", source_id="d_pf",
        semantic_context=SemanticContext(), now=base_time,
    )

    invocation_count = {"n": 0}

    def fake_reflection_run(candidates, *, deps):
        invocation_count["n"] += 1
        return (
            DReflectionResult(decisions=[], tick_note=None),
            DCallRow(
                d_call_id=f"dc_{invocation_count['n']}",
                ts=deps.now.isoformat(),
                tick_id=f"t_{invocation_count['n']}",
                model_tier_used="haiku",
                candidates_in=1,
                promoted_out=0,
                filtered_out=0,
                latency_ms=30000,
                tokens_input=0,
                tokens_output=0,
                failure_type="timeout",
            ),
        )

    compose_calls: list[str] = []

    def fake_compose(persona_dir, candidate, *, provider, voice_template, now):
        compose_calls.append(candidate.candidate_id)

    monkeypatch.setattr("brain.initiate.review.reflection_run", fake_reflection_run)
    monkeypatch.setattr("brain.initiate.review._process_one_candidate", fake_compose)

    # Tick 1 and 2: only 2 consecutive failures — candidates remain in queue.
    tick1_time = base_time + timedelta(minutes=1)
    tick2_time = base_time + timedelta(minutes=2)
    tick3_time = base_time + timedelta(minutes=3)

    run_initiate_review_tick(persona, provider=MagicMock(), voice_template="x", cap_per_tick=3, now=tick1_time)
    run_initiate_review_tick(persona, provider=MagicMock(), voice_template="x", cap_per_tick=3, now=tick2_time)
    assert len(read_candidates(persona)) == 1
    assert compose_calls == []

    # Tick 3: third consecutive failure — fall through to promote-all.
    run_initiate_review_tick(persona, provider=MagicMock(), voice_template="x", cap_per_tick=3, now=tick3_time)
    assert len(compose_calls) == 1
    assert "ic_" in compose_calls[0]  # candidate_id is always ic_<timestamp>_<rand>
