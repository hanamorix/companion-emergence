"""Tests for brain.chat.voice — voice.md loader + attempt_heal_text."""

from __future__ import annotations

from pathlib import Path

from brain.chat.voice import DEFAULT_VOICE_TEMPLATE, load_voice
from brain.health.attempt_heal import attempt_heal_text

# ── DEFAULT_VOICE_TEMPLATE shape ──────────────────────────────────────────────


def test_default_voice_template_has_four_sections() -> None:
    """The template must contain all 4 required section headers."""
    template = DEFAULT_VOICE_TEMPLATE
    assert "## 1. Who you are" in template
    assert "## 2. What's in your head" in template
    assert "## 3. How emotion shapes your voice" in template
    assert "## 4. Your boundaries with the user" in template


# ── load_voice — healthy paths ────────────────────────────────────────────────


def test_load_voice_missing_file_returns_default_template(tmp_path: Path) -> None:
    """Missing voice.md → creates file + returns default template, no anomaly."""
    content, anomaly = load_voice(tmp_path)
    assert anomaly is None
    assert "## 1. Who you are" in content
    # persona_name substituted
    assert tmp_path.name in content
    # File written for next load
    assert (tmp_path / "voice.md").exists()


def test_load_voice_well_formed_returns_content(tmp_path: Path) -> None:
    """Well-formed voice.md → returns content as-is, no anomaly."""
    expected = "# TestPersona\n\nHello world. This is my voice."
    (tmp_path / "voice.md").write_text(expected, encoding="utf-8")
    content, anomaly = load_voice(tmp_path)
    assert content == expected
    assert anomaly is None


def test_load_voice_corrupt_empty_quarantines_and_uses_bak(tmp_path: Path) -> None:
    """Empty voice.md → quarantine, restore from .bak1, return anomaly."""
    voice_path = tmp_path / "voice.md"
    bak_path = tmp_path / "voice.md.bak1"
    good_content = "# Persona\n\nGood backup content."
    voice_path.write_text("", encoding="utf-8")  # empty = corrupt
    bak_path.write_text(good_content, encoding="utf-8")

    content, anomaly = load_voice(tmp_path)
    assert content == good_content
    assert anomaly is not None
    assert anomaly.action == "restored_from_bak1"
    # Quarantine file created
    quarantines = list(tmp_path.glob("voice.md.corrupt-*"))
    assert len(quarantines) == 1


def test_load_voice_all_baks_missing_resets_to_default(tmp_path: Path) -> None:
    """Empty voice.md + no baks → reset to default template + return anomaly."""
    voice_path = tmp_path / "voice.md"
    voice_path.write_text("", encoding="utf-8")

    content, anomaly = load_voice(tmp_path)
    assert anomaly is not None
    assert anomaly.action == "reset_to_default"
    assert "## 1. Who you are" in content


# ── attempt_heal_text — unit tests ────────────────────────────────────────────


def test_attempt_heal_text_missing_writes_default(tmp_path: Path) -> None:
    """Missing file → writes default, returns (default, None)."""
    path = tmp_path / "identity.txt"
    content, anomaly = attempt_heal_text(path, default_factory=lambda: "DEFAULT")
    assert content == "DEFAULT"
    assert anomaly is None
    assert path.read_text(encoding="utf-8") == "DEFAULT"


def test_attempt_heal_text_well_formed_returns_content(tmp_path: Path) -> None:
    """Non-empty file → returns content, no anomaly."""
    path = tmp_path / "identity.txt"
    path.write_text("Hello there.", encoding="utf-8")
    content, anomaly = attempt_heal_text(path, default_factory=lambda: "DEFAULT")
    assert content == "Hello there."
    assert anomaly is None


def test_attempt_heal_text_empty_restores_from_bak(tmp_path: Path) -> None:
    """Empty primary + good bak1 → restore bak1, return anomaly."""
    path = tmp_path / "identity.txt"
    bak = tmp_path / "identity.txt.bak1"
    path.write_text("", encoding="utf-8")
    bak.write_text("good backup", encoding="utf-8")

    content, anomaly = attempt_heal_text(path, default_factory=lambda: "DEFAULT")
    assert content == "good backup"
    assert anomaly is not None
    assert "bak1" in anomaly.action


def test_attempt_heal_text_all_baks_empty_resets(tmp_path: Path) -> None:
    """Empty primary + empty baks → reset to default, return anomaly."""
    path = tmp_path / "identity.txt"
    path.write_text("", encoding="utf-8")
    # All baks also empty
    for i in (1, 2, 3):
        (tmp_path / f"identity.txt.bak{i}").write_text("", encoding="utf-8")

    content, anomaly = attempt_heal_text(path, default_factory=lambda: "RESET")
    assert content == "RESET"
    assert anomaly is not None
    assert anomaly.action == "reset_to_default"
