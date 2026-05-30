"""Persona state response no longer carries thinking_budget_tokens (removed in v0.0.26)."""
from __future__ import annotations

from pathlib import Path


def test_persona_state_connection_block_has_no_thinking_field(tmp_path: Path) -> None:
    """The connection block in /persona/state must NOT expose thinking_budget_tokens.

    The field was removed from PersonaConfig in v0.0.26 (T1) and the bridge
    endpoint surface must follow — callers cannot rely on this key existing.
    """
    from brain.bridge.persona_state import build_persona_state

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (persona_dir / "voice.md").write_text("placeholder")

    state = build_persona_state(persona_dir=persona_dir)
    assert "thinking_budget_tokens" not in state.get("connection", {})
