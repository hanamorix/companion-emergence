"""End-to-end: voice-edit proposal accepted writes to voice + SoulStore + audit.

The gravity of self-modification is encoded as a three-place atomic write:
the voice template file, the SoulStore `voice_evolution` table, and the
audit row's state transition. This integration test exercises all three
through the live `/initiate/voice-edit/accept` endpoint.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.initiate.audit import append_audit_row, iter_initiate_audit_full
from brain.initiate.schemas import AuditRow
from brain.soul.store import SoulStore


def _make_client(persona_dir: Path) -> TestClient:
    """Build the FastAPI app pinned to the given persona dir."""
    app = build_app(persona_dir=persona_dir, client_origin="tests")
    return TestClient(app)


def _seed_voice_edit_audit(
    persona_dir: Path,
    *,
    audit_id: str,
    old_text: str,
    new_text: str,
) -> None:
    """Append a voice_edit_proposal audit row in the delivered state."""
    diff = f"- {old_text}\n+ {new_text}\n"
    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts="2026-05-11T14:47:09+00:00",
        kind="voice_edit_proposal",
        subject="a small edit to my voice",
        tone_rendered=(
            f"Proposing to change my voice: {old_text!r} -> {new_text!r}."
        ),
        decision="send_quiet",
        decision_reasoning="the pattern showed up three times",
        gate_check={"allowed": True, "reason": None},
        delivery={
            "delivered_at": "2026-05-11T14:47:09+00:00",
            "state_transitions": [
                {"to": "delivered", "at": "2026-05-11T14:47:09+00:00"},
            ],
            "current_state": "delivered",
        },
        diff=diff,
    )
    append_audit_row(persona_dir, row)


def test_voice_edit_accept_writes_voice_audit_and_soul(tmp_path: Path) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    (persona_dir / "nell-voice.md").write_text("line A\nold line\nline C\n")
    _seed_voice_edit_audit(
        persona_dir,
        audit_id="ia_ve_001",
        old_text="old line",
        new_text="new line",
    )

    with _make_client(persona_dir) as client:
        r = client.post(
            "/initiate/voice-edit/accept",
            json={"audit_id": "ia_ve_001", "with_edits": None},
        )
    assert r.status_code == 200, r.text

    # Place 1: voice template updated.
    voice_body = (persona_dir / "nell-voice.md").read_text()
    assert "new line" in voice_body
    assert "old line" not in voice_body

    # Place 2: SoulStore voice_evolution record exists.
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        evolutions = store.list_voice_evolution()
    finally:
        store.close()
    assert len(evolutions) == 1
    assert evolutions[0].audit_id == "ia_ve_001"

    # Place 3: audit row state mutated to replied_explicit.
    matched = next(
        row for row in iter_initiate_audit_full(persona_dir)
        if row.audit_id == "ia_ve_001"
    )
    assert matched.delivery is not None
    assert matched.delivery["current_state"] == "replied_explicit"
