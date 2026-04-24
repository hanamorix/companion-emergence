"""Tests for `nell reflex` CLI handler."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.cli import main


def _setup_persona(personas_root: Path, persona_name: str = "testpersona") -> Path:
    """Create a persona dir with an empty memories DB + hebbian DB."""
    persona_dir = personas_root / persona_name
    persona_dir.mkdir(parents=True)
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(db_path=persona_dir / "memories.db")
    store.close()
    hm = HebbianMatrix(db_path=persona_dir / "hebbian.db")
    hm.close()
    return persona_dir


def test_cli_reflex_dry_run_no_arcs(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas", "testpersona")
    (tmp_path / "personas" / "testpersona" / "reflex_arcs.json").write_text(
        '{"version": 1, "arcs": []}', encoding="utf-8"
    )

    rc = main(["reflex", "--persona", "testpersona", "--provider", "fake", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no arc" in out.lower() or "no_arcs_defined" in out.lower()


def test_cli_reflex_missing_persona_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        main(["reflex", "--persona", "no_such", "--provider", "fake", "--dry-run"])
