"""Tests for the `nell init` CLI handler — interactive wizard + flag-driven."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain import cli


def _run_init(monkeypatch, tmp_path: Path, argv: list[str], inputs: list[str] | None = None) -> int:
    """Run cli.main(['init', ...]) with NELLBRAIN_HOME redirected and any
    interactive input(...) calls served from `inputs` in order."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    if inputs is not None:
        it = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(it))
    return cli.main(["init", *argv])


def test_init_fresh_with_all_flags_writes_config(monkeypatch, tmp_path: Path) -> None:
    """Non-interactive: all flags supplied → no prompts, persona ready."""
    rc = _run_init(
        monkeypatch,
        tmp_path,
        ["--persona", "siren", "--user-name", "Hana", "--fresh", "--voice-template", "default"],
    )
    assert rc == 0
    persona_dir = tmp_path / "personas" / "siren"
    assert persona_dir.exists()
    cfg = json.loads((persona_dir / "persona_config.json").read_text())
    assert cfg["user_name"] == "Hana"
    # default template → no voice.md written
    assert not (persona_dir / "voice.md").exists()


def test_init_interactive_prompts_for_missing(monkeypatch, tmp_path: Path) -> None:
    """No flags → prompts for everything. Defaults accepted via empty input."""
    inputs = [
        "siren",      # persona name
        "Hana",       # user name
        "n",          # migrate? → no
        "default",    # voice template
    ]
    rc = _run_init(monkeypatch, tmp_path, [], inputs=inputs)
    assert rc == 0
    cfg = json.loads((tmp_path / "personas" / "siren" / "persona_config.json").read_text())
    assert cfg["user_name"] == "Hana"


def test_init_invalid_persona_name_returns_1(monkeypatch, tmp_path: Path, capsys) -> None:
    rc = _run_init(
        monkeypatch,
        tmp_path,
        ["--persona", "../escape", "--user-name", "x", "--fresh", "--voice-template", "default"],
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "invalid persona name" in err


def test_init_existing_persona_refuses_without_force(monkeypatch, tmp_path: Path, capsys) -> None:
    """Don't clobber an existing persona's config without --force."""
    persona_dir = tmp_path / "personas" / "siren"
    persona_dir.mkdir(parents=True)
    (persona_dir / "marker.txt").write_text("dont touch")

    rc = _run_init(
        monkeypatch,
        tmp_path,
        ["--persona", "siren", "--user-name", "Hana", "--fresh", "--voice-template", "default"],
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "already exists" in err
    assert (persona_dir / "marker.txt").exists()  # not clobbered


def test_init_existing_persona_with_force_writes_config(monkeypatch, tmp_path: Path) -> None:
    """--force on an existing persona updates persona_config.json without
    touching memories/soul/etc — destructive only to config + voice.md."""
    persona_dir = tmp_path / "personas" / "siren"
    persona_dir.mkdir(parents=True)
    (persona_dir / "memories.db").write_text("fake_db")  # would-be data

    rc = _run_init(
        monkeypatch,
        tmp_path,
        ["--persona", "siren", "--user-name", "Hana", "--fresh", "--voice-template", "default", "--force"],
    )
    assert rc == 0
    assert (persona_dir / "memories.db").read_text() == "fake_db"  # preserved
    cfg = json.loads((persona_dir / "persona_config.json").read_text())
    assert cfg["user_name"] == "Hana"


def test_init_voice_template_nell_example_copies_voice_md(monkeypatch, tmp_path: Path) -> None:
    """nell-example template copies the canonical Nell voice as a starter."""
    rc = _run_init(
        monkeypatch,
        tmp_path,
        ["--persona", "siren", "--user-name", "Hana", "--fresh", "--voice-template", "nell-example"],
    )
    assert rc == 0
    voice = tmp_path / "personas" / "siren" / "voice.md"
    assert voice.exists()
    # Nell's canonical voice opens with the lede line
    assert "You are Nell" in voice.read_text()


def test_init_unknown_voice_template_argparse_rejects(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """argparse rejects unknown choices at parse time and raises SystemExit.
    This is standard argparse behaviour — the user sees the allowed list."""
    with pytest.raises(SystemExit) as excinfo:
        _run_init(
            monkeypatch,
            tmp_path,
            ["--persona", "siren", "--user-name", "x", "--fresh", "--voice-template", "wat"],
        )
    assert excinfo.value.code == 2  # argparse parse-error code
    err = capsys.readouterr().err
    assert "wat" in err


def test_init_invalid_voice_template_via_prompt_returns_1(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """If the prompt path receives an invalid voice template (user typo),
    we return 1 with a clear error rather than crashing in install_voice_template."""
    inputs = [
        "siren",
        "Hana",
        "n",
        "wat",  # bad template via prompt
    ]
    rc = _run_init(monkeypatch, tmp_path, [], inputs=inputs)
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown voice template" in err


def test_init_summary_includes_next_step_command(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _run_init(
        monkeypatch,
        tmp_path,
        ["--persona", "siren", "--user-name", "Hana", "--fresh", "--voice-template", "default"],
    )
    out = capsys.readouterr().out
    assert "uv run nell chat --persona siren" in out


def test_init_summary_is_encodable_on_windows_cp1252(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    rc = _run_init(
        monkeypatch,
        tmp_path,
        ["--persona", "siren", "--user-name", "Hana", "--fresh", "--voice-template", "default"],
    )
    assert rc == 0
    out = capsys.readouterr().out

    # Windows Git Bash / Actions may expose a cp1252 stdout. The summary
    # must not contain glyphs such as "✓" that crash print() there.
    out.encode("cp1252")
    assert "OK persona 'siren' ready" in out
