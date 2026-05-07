"""Tests for brain.setup — pure persona-setup helpers used by `nell init`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.setup import (
    VOICE_TEMPLATES,
    install_voice_template,
    validate_persona_name,
    write_persona_config,
)

# ---- validate_persona_name ----


def test_validate_persona_name_accepts_simple_names() -> None:
    for name in ["nell", "siren", "nell_2", "Nell-test", "x", "a" * 40]:
        validate_persona_name(name)  # no raise


def test_validate_persona_name_rejects_path_traversal_and_garbage() -> None:
    for evil in ["../escape", "a/b", "..", "", "n e l l", "x" * 41,
                 "weird:char", "name.with.dots"]:
        with pytest.raises(ValueError, match="invalid persona name"):
            validate_persona_name(evil)


def test_validate_persona_name_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="invalid persona name"):
        validate_persona_name(None)  # type: ignore[arg-type]


# ---- write_persona_config ----


def test_write_persona_config_creates_file_with_user_name(tmp_path: Path) -> None:
    persona_dir = tmp_path / "siren"
    path = write_persona_config(persona_dir, user_name="Hana")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["user_name"] == "Hana"
    # Defaults preserved for the other fields
    assert data["provider"] == "claude-cli"


def test_write_persona_config_preserves_existing_fields(tmp_path: Path) -> None:
    """Re-running init shouldn't clobber an existing provider override."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    config_path = persona_dir / "persona_config.json"
    config_path.write_text(json.dumps({
        "provider": "ollama",
        "searcher": "ddgs",
        "mcp_audit_log_level": "full",
    }))

    write_persona_config(persona_dir, user_name="Hana")
    data = json.loads(config_path.read_text())
    assert data["user_name"] == "Hana"
    assert data["provider"] == "ollama"  # preserved
    assert data["mcp_audit_log_level"] == "full"  # preserved


def test_write_persona_config_strips_empty_user_name_to_none(tmp_path: Path) -> None:
    persona_dir = tmp_path / "nell"
    write_persona_config(persona_dir, user_name="   ")
    data = json.loads((persona_dir / "persona_config.json").read_text())
    assert data["user_name"] is None


def test_write_persona_config_creates_dir_if_missing(tmp_path: Path) -> None:
    persona_dir = tmp_path / "deep" / "nested" / "nell"
    write_persona_config(persona_dir, user_name="Hana")
    assert (persona_dir / "persona_config.json").exists()


# ---- install_voice_template ----


def test_install_voice_template_default_writes_no_file(tmp_path: Path) -> None:
    """default + skip → no voice.md (DEFAULT_VOICE_TEMPLATE applies on chat)."""
    persona_dir = tmp_path / "siren"
    for template in ("default", "skip"):
        result = install_voice_template(persona_dir, template)
        assert result is None
        assert not (persona_dir / "voice.md").exists()


def test_install_voice_template_nell_example_copies_packaged_file(
    tmp_path: Path,
) -> None:
    """nell-example writes the packaged brain/voice_templates/nell-voice.md.

    The template ships inside the brain wheel so it's available whether
    the framework is installed from source, from a wheel, or from inside
    the Phase 7 bundled NellFace.app — no `repo_root` lookup needed.
    """
    persona_dir = tmp_path / "persona"
    result = install_voice_template(persona_dir, "nell-example")
    assert result == persona_dir / "voice.md"
    content = result.read_text(encoding="utf-8")
    # The shipped Nell voice draft is the canonical one — opens with
    # the section-1 header.
    assert "## 1. Who you are" in content
    assert len(content) > 1000  # not an empty file


def test_install_voice_template_unknown_raises(tmp_path: Path) -> None:
    persona_dir = tmp_path / "nell"
    with pytest.raises(ValueError, match="unknown voice template"):
        install_voice_template(persona_dir, "no-such-template")


def test_voice_templates_keys_are_documented() -> None:
    """Every key has a non-empty human-readable description so the wizard
    can surface the choices to the user."""
    assert set(VOICE_TEMPLATES.keys()) == {"default", "nell-example", "skip"}
    for desc in VOICE_TEMPLATES.values():
        assert isinstance(desc, str) and len(desc) > 20
