"""get_provider reads model from PersonaConfig; explicit override wins."""

import json
from pathlib import Path

from brain.bridge.provider import get_provider, ClaudeCliProvider


def test_factory_reads_model_from_config(tmp_path: Path):
    (tmp_path / "persona_config.json").write_text(
        json.dumps({"provider": "claude-cli", "searcher": "noop", "model": "opus"})
    )
    provider = get_provider("claude-cli", persona_dir=tmp_path)
    assert isinstance(provider, ClaudeCliProvider)
    assert provider._model == "opus"


def test_factory_override_beats_config(tmp_path: Path):
    (tmp_path / "persona_config.json").write_text(
        json.dumps({"provider": "claude-cli", "searcher": "noop", "model": "opus"})
    )
    provider = get_provider("claude-cli", persona_dir=tmp_path, model_override="sonnet")
    assert isinstance(provider, ClaudeCliProvider)
    assert provider._model == "sonnet"


def test_factory_default_when_no_config(tmp_path: Path):
    # No persona_config.json — fall back to default.
    provider = get_provider("claude-cli", persona_dir=tmp_path)
    assert isinstance(provider, ClaudeCliProvider)
    assert provider._model == "sonnet"


def test_factory_no_persona_dir_uses_default():
    # Legacy call shape (name only) — must still work and default to sonnet.
    provider = get_provider("claude-cli")
    assert isinstance(provider, ClaudeCliProvider)
    assert provider._model == "sonnet"
