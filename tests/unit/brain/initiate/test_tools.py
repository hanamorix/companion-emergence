"""Tests for brain.initiate.tools — on-demand verify tools."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.initiate.audit import append_audit_row
from brain.initiate.schemas import AuditRow
from brain.initiate.tools import (
    recall_initiate_audit,
    recall_soul_audit,
    recall_voice_evolution,
)


def _seed_audit(persona_dir: Path, audit_id: str, ts: str | None = None) -> None:
    from datetime import UTC, datetime
    if ts is None:
        ts = datetime.now(UTC).isoformat()
    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts=ts,
        kind="message",
        subject="the dream",
        tone_rendered="rendered",
        decision="send_quiet",
        decision_reasoning="x",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row.record_transition("delivered", ts)
    append_audit_row(persona_dir, row)


def test_recall_initiate_audit_24h(tmp_path: Path) -> None:
    _seed_audit(tmp_path, "ia_1")
    out = recall_initiate_audit(tmp_path, window="24h")
    assert "ia_1" in out or "the dream" in out
    assert isinstance(out, str)


def test_recall_initiate_audit_filter_by_state(tmp_path: Path) -> None:
    """Filter parameter constrains to rows with that current state."""
    row1 = AuditRow(
        audit_id="ia_1",
        candidate_id="ic_1",
        ts=datetime.now(UTC).isoformat(),
        kind="message",
        subject="A",
        tone_rendered="",
        decision="send_quiet",
        decision_reasoning="",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row1.record_transition("delivered", row1.ts)
    row1.record_transition("read", row1.ts)
    append_audit_row(tmp_path, row1)
    row2 = AuditRow(
        audit_id="ia_2",
        candidate_id="ic_2",
        ts=datetime.now(UTC).isoformat(),
        kind="message",
        subject="B",
        tone_rendered="",
        decision="send_quiet",
        decision_reasoning="",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row2.record_transition("delivered", row2.ts)
    append_audit_row(tmp_path, row2)
    out = recall_initiate_audit(tmp_path, window="24h", filter_state="read")
    assert "A" in out
    assert "B" not in out


def test_recall_initiate_audit_empty(tmp_path: Path) -> None:
    out = recall_initiate_audit(tmp_path, window="24h")
    assert isinstance(out, str)
    assert "no recent" in out.lower() or out.strip() == "" or "empty" in out.lower()


def test_recall_voice_evolution_returns_chronological(tmp_path: Path) -> None:
    from brain.soul.store import SoulStore, VoiceEvolution

    store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        store.save_voice_evolution(
            VoiceEvolution(
                id="ve_1",
                accepted_at="2026-01-01T00:00:00+00:00",
                diff="",
                old_text="A",
                new_text="B",
                rationale="x",
                evidence=[],
                audit_id="ia_1",
                user_modified=False,
            )
        )
        store.save_voice_evolution(
            VoiceEvolution(
                id="ve_2",
                accepted_at="2026-05-01T00:00:00+00:00",
                diff="",
                old_text="C",
                new_text="D",
                rationale="y",
                evidence=[],
                audit_id="ia_2",
                user_modified=False,
            )
        )
    finally:
        store.close()
    out = recall_voice_evolution(tmp_path)
    assert out.index("A") < out.index("C")  # chronological


def test_recall_soul_audit_empty(tmp_path: Path) -> None:
    out = recall_soul_audit(tmp_path, window="30d")
    assert isinstance(out, str)
    assert "no recent" in out.lower() or out.strip() == ""
