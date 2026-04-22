"""Tests for brain.config — three-source config merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import config


def _write_persona(persona_dir: Path, toml_body: str) -> None:
    """Helper: write persona.toml with the given body to a persona dir."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    (persona_dir / "persona.toml").write_text(toml_body)


def test_persona_toml_provides_baseline(
    tmp_path: Path, clean_env: None
) -> None:
    """Values from persona.toml become the baseline config."""
    persona_dir = tmp_path / "nell"
    _write_persona(
        persona_dir,
        """
[bridge]
bind = "127.0.0.1:9000"

[model]
provider = "ollama"
tag = "my-model"
""",
    )

    result = config.load_config(persona_dir)
    assert result.bridge_bind == "127.0.0.1:9000"
    assert result.provider == "ollama"
    assert result.model == "my-model"
    assert result.source_trace["BRIDGE_BIND"] == "persona.toml"
    assert result.source_trace["PROVIDER"] == "persona.toml"
    assert result.source_trace["MODEL"] == "persona.toml"


def test_env_var_overrides_persona_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variables override persona.toml values."""
    persona_dir = tmp_path / "nell"
    _write_persona(
        persona_dir,
        """
[bridge]
bind = "127.0.0.1:9000"
""",
    )

    monkeypatch.setenv("BRIDGE_BIND", "127.0.0.1:8000")

    result = config.load_config(persona_dir)
    assert result.bridge_bind == "127.0.0.1:8000"
    assert result.source_trace["BRIDGE_BIND"] == "env"


def test_env_file_overrides_persona_toml(
    tmp_path: Path, clean_env: None
) -> None:
    """A .env file overrides persona.toml when no env var is set."""
    persona_dir = tmp_path / "nell"
    _write_persona(
        persona_dir,
        '[model]\nprovider = "from-toml"\n',
    )

    env_file = tmp_path / ".env"
    env_file.write_text("PROVIDER=from-env-file\n")

    result = config.load_config(persona_dir, env_file=env_file)
    assert result.provider == "from-env-file"
    assert result.source_trace["PROVIDER"] == ".env"


def test_env_var_beats_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var takes precedence over .env file."""
    persona_dir = tmp_path / "nell"
    _write_persona(persona_dir, '[model]\nprovider = "from-toml"\n')

    env_file = tmp_path / ".env"
    env_file.write_text("PROVIDER=from-env-file\n")

    monkeypatch.setenv("PROVIDER", "from-env-var")

    result = config.load_config(persona_dir, env_file=env_file)
    assert result.provider == "from-env-var"
    assert result.source_trace["PROVIDER"] == "env"


def test_sensible_defaults_when_nothing_configured(
    tmp_path: Path, clean_env: None
) -> None:
    """When no config present, defaults apply AND are recorded in source_trace."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()

    result = config.load_config(persona_dir)
    assert result.bridge_bind == "127.0.0.1:8765"
    assert result.provider == "ollama"
    assert result.model == ""
    # Defaults must be traceable so startup logging can distinguish
    # "default" from "value missing / not loaded".
    assert result.source_trace["BRIDGE_BIND"] == "default"
    assert result.source_trace["PROVIDER"] == "default"
    assert result.source_trace["MODEL"] == "default"


def test_ipc_jid_reads_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NELL_IPC_JID env var populates Config.ipc_jid."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    monkeypatch.setenv("NELL_IPC_JID", "15551234567@s.whatsapp.net")

    result = config.load_config(persona_dir)
    assert result.ipc_jid == "15551234567@s.whatsapp.net"
    assert result.source_trace["NELL_IPC_JID"] == "env"


def test_ipc_jid_defaults_empty(
    tmp_path: Path, clean_env: None
) -> None:
    """Unset NELL_IPC_JID leaves ipc_jid empty, not absent, with default source."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()

    result = config.load_config(persona_dir)
    assert result.ipc_jid == ""
    assert result.source_trace["NELL_IPC_JID"] == "default"


def test_env_file_strips_inline_comments(
    tmp_path: Path, clean_env: None
) -> None:
    """`KEY=value # note` parses as value, not value-with-comment."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()

    env_file = tmp_path / ".env"
    env_file.write_text(
        "PROVIDER=ollama  # local running on this box\n"
        'MODEL="nell-stage13" # the good one\n'
    )

    result = config.load_config(persona_dir, env_file=env_file)
    assert result.provider == "ollama"
    assert result.model == "nell-stage13"


def test_env_file_ignores_comments_and_blank_lines(
    tmp_path: Path, clean_env: None
) -> None:
    """.env parser skips # comments and blank lines."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# this is a comment\n"
        "\n"
        "PROVIDER=openai\n"
        "# another comment\n"
        'MODEL="claude-sonnet-4"\n'
    )

    result = config.load_config(persona_dir, env_file=env_file)
    assert result.provider == "openai"
    assert result.model == "claude-sonnet-4"


def test_persona_name_derived_from_dir(
    tmp_path: Path, clean_env: None
) -> None:
    """persona_name on the Config matches the persona_dir basename."""
    persona_dir = tmp_path / "sage"
    persona_dir.mkdir()

    result = config.load_config(persona_dir)
    assert result.persona_name == "sage"
