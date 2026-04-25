# tests/unit/brain/growth/test_cli_growth.py
"""Tests for `nell growth log` CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.cli import main
from brain.growth.log import GrowthLogEvent, append_growth_event


def _setup_persona(personas_root: Path, name: str = "testpersona") -> Path:
    persona_dir = personas_root / name
    persona_dir.mkdir(parents=True)
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    MemoryStore(db_path=persona_dir / "memories.db").close()
    HebbianMatrix(db_path=persona_dir / "hebbian.db").close()
    return persona_dir


def _seed_growth_event(
    persona_dir: Path, name: str, when: datetime, relational_context: str | None = None
) -> None:
    append_growth_event(
        persona_dir / "emotion_growth.log.jsonl",
        GrowthLogEvent(
            timestamp=when,
            type="emotion_added",
            name=name,
            description=f"description of {name}",
            decay_half_life_days=7.0,
            reason="seeded",
            evidence_memory_ids=("mem_a",),
            score=0.7,
            relational_context=relational_context,
        ),
    )


def test_cli_growth_log_empty(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas")
    rc = main(["growth", "log", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    # Should mention zero events / empty log gracefully
    assert "0 events" in out or "empty" in out.lower() or "no events" in out.lower()


def test_cli_growth_log_displays_events(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    _seed_growth_event(
        persona_dir,
        "lingering",
        datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        relational_context="during Hana's tender messages",
    )
    rc = main(["growth", "log", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lingering" in out
    assert "during Hana's tender messages" in out
    assert "2026-04-25" in out


def test_cli_growth_log_limit_flag(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    for i in range(5):
        _seed_growth_event(
            persona_dir,
            f"e{i}",
            datetime(2026, 4, 20 + i, tzinfo=UTC),
        )
    rc = main(["growth", "log", "--persona", "testpersona", "--limit", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "e3" in out
    assert "e4" in out
    assert "e0" not in out
    assert "e1" not in out


def test_cli_growth_log_missing_persona_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="persona"):
        main(["growth", "log", "--persona", "ghost"])


def test_cli_growth_no_action_commands(monkeypatch, tmp_path: Path):
    """Phase 2a only ships `log` — no `add`, `approve`, `reject`, `force`."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    _setup_persona(tmp_path / "personas")
    for forbidden in ("add", "approve", "reject", "force"):
        with pytest.raises(SystemExit):
            main(["growth", forbidden, "--persona", "testpersona"])
