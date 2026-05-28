"""Tests that server.py helper functions thread user_name from PersonaConfig."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_apply_replied_explicit_transition_passes_user_name(tmp_path: Path) -> None:
    """_apply_replied_explicit_transition must pass user_name loaded from PersonaConfig."""
    from brain.bridge.server import _apply_replied_explicit_transition

    (tmp_path / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "user_name": "Henryk"})
    )

    captured: list[dict] = []

    fake_row = MagicMock()
    fake_row.audit_id = "ia_001"
    fake_row.subject = "a dream"
    fake_row.tone_rendered = "rendered message"

    def fake_update_memory(store, *, audit_id, subject, message, new_state, ts, user_name="my user"):
        captured.append({"user_name": user_name})

    fake_store = MagicMock()
    fake_store.__enter__ = lambda s: s
    fake_store.__exit__ = MagicMock(return_value=False)

    with (
        patch("brain.initiate.audit.update_audit_state", lambda *a, **kw: None),
        patch("brain.initiate.audit.iter_initiate_audit_full", lambda pd: [fake_row]),
        patch("brain.initiate.memory.update_initiate_memory_for_state", fake_update_memory),
        patch("brain.bridge.server.MemoryStore", return_value=fake_store),
    ):
        _apply_replied_explicit_transition(tmp_path, "ia_001")

    assert captured, "update_initiate_memory_for_state was not called"
    assert captured[0]["user_name"] == "Henryk"
