"""Persona config files with the v0.0.25 `thinking_budget_tokens` field still load cleanly."""
from __future__ import annotations

import json
from pathlib import Path


def test_legacy_thinking_budget_field_ignored(tmp_path: Path):
    """Pydantic ignores unknown fields by default; existing persona files in the wild
    with thinking_budget_tokens set must continue loading without error."""
    from brain.persona_config import PersonaConfig

    cfg_path = tmp_path / "persona_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "provider": "claude-cli",
                "searcher": "ddgs",
                "mcp_audit_log_level": "redacted",
                "user_name": "Hana",
                "model": "sonnet",
                "thinking_budget_tokens": 8000,  # legacy field
            }
        )
    )
    cfg = PersonaConfig.load(cfg_path)
    assert cfg.user_name == "Hana"
    assert cfg.model == "sonnet"
    assert not hasattr(cfg, "thinking_budget_tokens")
