"""last_opened_at field on PersonaConfig powers the PersonaPicker sort
in NellFace v0.0.18+ (recency-ordered when multiple personas exist)."""
import json
from datetime import UTC, datetime
from pathlib import Path

from brain.persona_config import PersonaConfig


def test_load_default_is_none(tmp_path: Path) -> None:
    path = tmp_path / "persona_config.json"
    path.write_text(
        json.dumps(
            {
                "user_name": "zero",
                "provider": "claude-cli",
                "model": "sonnet",
            }
        )
    )
    cfg = PersonaConfig.load(path)
    assert cfg.last_opened_at is None


def test_touch_writes_iso_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "persona_config.json"
    cfg = PersonaConfig(provider="claude-cli", model="sonnet")
    cfg.save(path)

    cfg = PersonaConfig.load(path)
    cfg.touch_last_opened()
    cfg.save(path)

    reloaded = PersonaConfig.load(path)
    assert reloaded.last_opened_at is not None
    # Parse roundtrip — should be ISO8601, ends with Z
    parsed = datetime.fromisoformat(reloaded.last_opened_at.replace("Z", "+00:00"))
    # Recent (within last 10 seconds)
    delta = (datetime.now(UTC) - parsed).total_seconds()
    assert 0 <= delta < 10
    assert reloaded.last_opened_at.endswith("Z")


def test_existing_persona_without_field_loads_cleanly(tmp_path: Path) -> None:
    """Backward-compat: old persona_config.json without last_opened_at loads as None."""
    path = tmp_path / "persona_config.json"
    # Old-style file: no last_opened_at key at all
    path.write_text(json.dumps({"provider": "claude-cli", "model": "sonnet"}))
    cfg = PersonaConfig.load(path)
    assert cfg.last_opened_at is None
