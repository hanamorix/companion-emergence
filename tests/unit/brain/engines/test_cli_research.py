"""Tests for `nell research` + `nell interest *` CLI handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.cli import main


def _setup_persona(personas_root: Path, name: str = "testpersona") -> Path:
    persona_dir = personas_root / name
    persona_dir.mkdir(parents=True)
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    MemoryStore(db_path=persona_dir / "memories.db").close()
    HebbianMatrix(db_path=persona_dir / "hebbian.db").close()
    return persona_dir


def test_cli_research_dry_run_no_interests(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    (persona_dir / "interests.json").write_text('{"version": 1, "interests": []}', encoding="utf-8")
    rc = main(
        [
            "research",
            "--persona",
            "testpersona",
            "--provider",
            "fake",
            "--searcher",
            "noop",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "no" in out  # "no interests" or "no eligible" — either acceptable


def test_cli_research_missing_persona_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        main(
            [
                "research",
                "--persona",
                "no_such",
                "--provider",
                "fake",
                "--searcher",
                "noop",
                "--dry-run",
            ]
        )


def test_cli_interest_list_empty(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    (persona_dir / "interests.json").write_text('{"version": 1, "interests": []}', encoding="utf-8")
    rc = main(["interest", "list", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "testpersona" in out  # persona name appears in output


def test_cli_interest_add_subcommand_removed(monkeypatch, tmp_path: Path):
    """`interest add` is no longer a user surface — brain develops interests."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas")
    with pytest.raises(SystemExit):
        main(
            [
                "interest",
                "add",
                "deep sea creatures",
                "--persona",
                "testpersona",
            ]
        )


def test_cli_interest_bump_subcommand_removed(monkeypatch, tmp_path: Path):
    """`interest bump` is no longer a user surface — brain owns pull_scores."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas")
    with pytest.raises(SystemExit):
        main(
            [
                "interest",
                "bump",
                "anything",
                "--persona",
                "testpersona",
            ]
        )


def test_cli_research_interest_flag_removed(monkeypatch, tmp_path: Path):
    """`nell research --interest` is gone — brain picks its own topic."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas")
    with pytest.raises(SystemExit):
        main(
            [
                "research",
                "--persona",
                "testpersona",
                "--provider",
                "fake",
                "--searcher",
                "noop",
                "--interest",
                "anything",
            ]
        )


# ---- PR-B: provider/searcher resolution from persona_config.json ----


def test_cli_research_reads_provider_from_persona_config(
    monkeypatch, tmp_path: Path, capsys
):
    """When --provider is omitted, persona_config.json drives the choice."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    (persona_dir / "interests.json").write_text(
        '{"version": 1, "interests": []}', encoding="utf-8"
    )
    (persona_dir / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}\n', encoding="utf-8"
    )
    # No --provider / --searcher passed — must come from the file.
    rc = main(["research", "--persona", "testpersona", "--dry-run"])
    assert rc == 0


def test_cli_research_provider_flag_overrides_persona_config(
    monkeypatch, tmp_path: Path
):
    """CLI --provider overrides persona_config.json (developer override)."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    (persona_dir / "interests.json").write_text(
        '{"version": 1, "interests": []}', encoding="utf-8"
    )
    # Persona file says use ollama (which would NotImplementedError).
    # CLI override forces fake — proves CLI wins.
    (persona_dir / "persona_config.json").write_text(
        '{"provider": "ollama", "searcher": "noop"}\n', encoding="utf-8"
    )
    rc = main(
        [
            "research",
            "--persona",
            "testpersona",
            "--provider",
            "fake",
            "--searcher",
            "noop",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_cli_research_no_config_no_flag_uses_framework_default(
    monkeypatch, tmp_path: Path
):
    """No persona_config.json + no flag → claude-cli default would fire.

    We can't actually invoke claude-cli in tests, so we assert the resolve
    helper picks 'claude-cli' when nothing else is provided. Direct unit
    test — CLI integration is covered by the override + read-from-file
    cases above.
    """
    from brain.persona_config import DEFAULT_PROVIDER, DEFAULT_SEARCHER, PersonaConfig

    cfg = PersonaConfig.load(tmp_path / "missing.json")
    assert cfg.provider == DEFAULT_PROVIDER == "claude-cli"
    assert cfg.searcher == DEFAULT_SEARCHER == "ddgs"
