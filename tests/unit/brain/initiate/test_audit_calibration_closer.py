"""Tests for run_calibration_closer_tick in brain.initiate.audit."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.initiate.adaptive import read_recent_calibration_rows
from brain.initiate.audit import append_audit_row, run_calibration_closer_tick
from brain.initiate.schemas import AuditRow


def _make_promoted_audit_row(
    *,
    audit_id: str,
    candidate_id: str,
    ts: str,
    state: str,
) -> AuditRow:
    """Build an audit row representing a candidate D promoted that has
    reached the given delivery state."""
    row = AuditRow(
        audit_id=audit_id,
        candidate_id=candidate_id,
        ts=ts,
        kind="message",
        subject="test subject",
        tone_rendered="test tone",
        decision="promoted_by_d",
        decision_reasoning="resonant",
        gate_check={"allowed": True, "reason": None},
    )
    row.delivery = {
        "delivered_at": ts,
        "state_transitions": [
            {"to": "delivered", "at": ts},
            {"to": "read", "at": ts},
            {"to": state, "at": ts},
        ],
        "current_state": state,
    }
    return row


def test_closer_writes_row_for_promoted_replied(tmp_path):
    persona = tmp_path / "p"
    now = datetime(2026, 5, 13, 10, 0, 0, tzinfo=UTC)
    append_audit_row(persona, _make_promoted_audit_row(
        audit_id="ia_1", candidate_id="ic_1",
        ts=now.isoformat(), state="replied_explicit",
    ))
    run_calibration_closer_tick(persona, now=now + timedelta(minutes=10))
    rows = list(read_recent_calibration_rows(persona, limit=10))
    assert len(rows) == 1
    assert rows[0].candidate_id == "ic_1"
    assert rows[0].promoted_to_state == "replied_explicit"
    assert rows[0].decision == "promote"


def test_closer_skips_promoted_still_pending(tmp_path):
    persona = tmp_path / "p"
    now = datetime(2026, 5, 13, 10, 0, 0, tzinfo=UTC)
    row = _make_promoted_audit_row(
        audit_id="ia_p", candidate_id="ic_p",
        ts=now.isoformat(), state="replied_explicit",
    )
    # Replace current_state with non-terminal:
    row.delivery["current_state"] = "delivered"
    row.delivery["state_transitions"] = [{"to": "delivered", "at": now.isoformat()}]
    append_audit_row(persona, row)

    run_calibration_closer_tick(persona, now=now + timedelta(minutes=10))
    rows = list(read_recent_calibration_rows(persona, limit=10))
    assert rows == []


def test_closer_dedupes_promoted_rows(tmp_path):
    persona = tmp_path / "p"
    now = datetime(2026, 5, 13, 10, 0, 0, tzinfo=UTC)
    append_audit_row(persona, _make_promoted_audit_row(
        audit_id="ia_x", candidate_id="ic_x",
        ts=now.isoformat(), state="replied_explicit",
    ))
    # Run twice — should only write one calibration row.
    run_calibration_closer_tick(persona, now=now + timedelta(minutes=10))
    run_calibration_closer_tick(persona, now=now + timedelta(minutes=20))
    rows = list(read_recent_calibration_rows(persona, limit=10))
    assert len(rows) == 1


def _make_filtered_audit_row(
    *,
    audit_id: str,
    candidate_id: str,
    ts: str,
    decision_value: str = "filtered_pre_compose",
    source: str = "dream",
    source_id: str | None = None,
) -> AuditRow:
    gate_check = {"allowed": True, "reason": None, "source": source}
    if source_id is not None:
        gate_check["source_id"] = source_id
    return AuditRow(
        audit_id=audit_id,
        candidate_id=candidate_id,
        ts=ts,
        kind="message",
        subject="",
        tone_rendered="",
        decision=decision_value,
        decision_reasoning="weather passing",
        gate_check=gate_check,
    )


def test_closer_writes_filtered_row_after_48h_no_recurrence(tmp_path):
    persona = tmp_path / "p"
    decision_ts = datetime(2026, 5, 11, 10, 0, 0, tzinfo=UTC)  # 2 days ago
    now = datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC)
    append_audit_row(persona, _make_filtered_audit_row(
        audit_id="ia_f1", candidate_id="ic_f1",
        ts=decision_ts.isoformat(),
    ))
    run_calibration_closer_tick(persona, now=now)
    rows = list(read_recent_calibration_rows(persona, limit=10))
    assert len(rows) == 1
    assert rows[0].candidate_id == "ic_f1"
    assert rows[0].decision == "filter"
    assert rows[0].filtered_recurred is False


def test_closer_writes_filtered_row_with_recurrence(tmp_path):
    """If the same source_id was re-emitted as a candidate within 48h
    of the filter, filtered_recurred=True."""
    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.schemas import SemanticContext

    persona = tmp_path / "p"
    decision_ts = datetime(2026, 5, 11, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC)
    append_audit_row(persona, _make_filtered_audit_row(
        audit_id="ia_f2", candidate_id="ic_f2",
        ts=decision_ts.isoformat(),
        source="dream", source_id="d_recur",
    ))
    # Re-emit a candidate with the same (source, source_id) 12 hours after.
    emit_initiate_candidate(
        persona, kind="message", source="dream", source_id="d_recur",
        semantic_context=SemanticContext(),
        now=decision_ts + timedelta(hours=12),
    )
    run_calibration_closer_tick(persona, now=now)
    rows = list(read_recent_calibration_rows(persona, limit=10))
    assert len(rows) == 1
    assert rows[0].filtered_recurred is True


def test_closer_skips_filtered_row_before_48h(tmp_path):
    persona = tmp_path / "p"
    decision_ts = datetime(2026, 5, 13, 10, 0, 0, tzinfo=UTC)
    now = decision_ts + timedelta(hours=24)  # only 24h elapsed
    append_audit_row(persona, _make_filtered_audit_row(
        audit_id="ia_f3", candidate_id="ic_f3",
        ts=decision_ts.isoformat(),
    ))
    run_calibration_closer_tick(persona, now=now)
    rows = list(read_recent_calibration_rows(persona, limit=10))
    assert rows == []


def test_closer_idempotent_across_multiple_runs(tmp_path):
    persona = tmp_path / "p"
    decision_ts = datetime(2026, 5, 11, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC)
    append_audit_row(persona, _make_filtered_audit_row(
        audit_id="ia_f4", candidate_id="ic_f4",
        ts=decision_ts.isoformat(),
    ))
    run_calibration_closer_tick(persona, now=now)
    run_calibration_closer_tick(persona, now=now + timedelta(minutes=10))
    rows = list(read_recent_calibration_rows(persona, limit=10))
    assert len(rows) == 1
