"""End-to-end: acknowledged_unclear surfaces in ambient block for next chat turn."""

from __future__ import annotations

import pytest
pytest.importorskip("brain.initiate")

from datetime import UTC, datetime
from pathlib import Path

from brain.initiate.ambient import build_outbound_recall_block
from brain.initiate.audit import append_audit_row, update_audit_state
from brain.initiate.schemas import AuditRow


def test_acknowledged_unclear_shows_in_ambient_block(tmp_path: Path) -> None:
    """An acknowledged_unclear entry surfaces in build_outbound_recall_block."""
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    # Seed a delivered + read send.
    row = AuditRow(
        audit_id="ia_001",
        candidate_id="ic_001",
        ts="2026-05-11T14:00:00+00:00",
        kind="message",
        subject="the dream from this morning",
        tone_rendered="the dream from this morning landed",
        decision="send_quiet",
        decision_reasoning="x",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row.record_transition("delivered", row.ts)
    append_audit_row(persona_dir, row)

    # Renderer reports read.
    update_audit_state(
        persona_dir,
        audit_id="ia_001",
        new_state="read",
        at="2026-05-11T18:00:00+00:00",
    )
    # Chat engine later marks acknowledged_unclear.
    update_audit_state(
        persona_dir,
        audit_id="ia_001",
        new_state="acknowledged_unclear",
        at="2026-05-11T19:30:00+00:00",
    )

    now = datetime(2026, 5, 11, 20, 0, tzinfo=UTC)
    block = build_outbound_recall_block(persona_dir, now=now)
    assert block is not None
    assert "Pending uncertainty" in block
    assert "acknowledged_unclear" in block
    assert "the dream from this morning" in block
