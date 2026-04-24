"""Tests for `nell research` + `nell interest *` CLI handlers."""

from __future__ import annotations

import json
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


def test_cli_interest_add_then_list(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    (persona_dir / "interests.json").write_text('{"version": 1, "interests": []}', encoding="utf-8")
    rc = main(
        [
            "interest",
            "add",
            "deep sea creatures",
            "--keywords",
            "octopus,bioluminescence,ocean",
            "--scope",
            "either",
            "--persona",
            "testpersona",
        ]
    )
    assert rc == 0

    rc = main(["interest", "list", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "deep sea creatures" in out

    data = json.loads((persona_dir / "interests.json").read_text(encoding="utf-8"))
    assert len(data["interests"]) == 1
    assert data["interests"][0]["topic"] == "deep sea creatures"
    assert data["interests"][0]["related_keywords"] == [
        "octopus",
        "bioluminescence",
        "ocean",
    ]
    assert data["interests"][0]["scope"] == "either"
    assert data["interests"][0]["pull_score"] == 5.0  # below default threshold


def test_cli_interest_bump_increments_pull_score(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")

    # Seed one interest
    main(
        [
            "interest",
            "add",
            "seed topic",
            "--keywords",
            "s",
            "--scope",
            "either",
            "--persona",
            "testpersona",
        ]
    )

    rc = main(
        [
            "interest",
            "bump",
            "seed topic",
            "--amount",
            "2.0",
            "--persona",
            "testpersona",
        ]
    )
    assert rc == 0

    data = json.loads((persona_dir / "interests.json").read_text(encoding="utf-8"))
    assert data["interests"][0]["pull_score"] == 7.0  # 5.0 + 2.0
