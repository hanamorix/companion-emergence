"""Tests for the `nell dream` CLI subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def nell_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a minimal nell persona dir with one recent memory + empty hebbian."""
    root = tmp_path / "persona_root"
    root.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(root))

    from brain.paths import get_persona_dir

    persona = get_persona_dir("nell")
    persona.mkdir(parents=True)

    store = MemoryStore(db_path=persona / "memories.db")
    seed = Memory.create_new(
        content="first meeting test seed",
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0},
    )
    seed.importance = 8.0
    store.create(seed)
    store.close()

    from brain.memory.hebbian import HebbianMatrix

    h = HebbianMatrix(db_path=persona / "hebbian.db")
    h.close()
    return persona


def test_nell_dream_dry_run_with_fake_provider(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """nell dream --dry-run --provider fake runs without writes + prints summary."""
    from brain.cli import main

    rc = main(["dream", "--dry-run", "--provider", "fake"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "first meeting test seed" in out or "Dry run" in out


def test_nell_dream_real_cycle_with_fake_provider(
    nell_persona: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """nell dream --provider fake writes a new dream memory."""
    from brain.cli import main

    rc = main(["dream", "--provider", "fake"])
    assert rc == 0

    store = MemoryStore(db_path=nell_persona / "memories.db")
    dreams = store.list_by_type("dream")
    store.close()
    assert len(dreams) == 1


def test_nell_dream_ollama_surfaces_not_implemented(nell_persona: Path) -> None:
    """--provider ollama fails cleanly with NotImplementedError."""
    from brain.cli import main

    with pytest.raises(NotImplementedError):
        main(["dream", "--provider", "ollama"])


def test_nell_dream_unknown_provider_fails(nell_persona: Path) -> None:
    """Unknown provider name raises ValueError."""
    from brain.cli import main

    with pytest.raises(ValueError, match="Unknown provider"):
        main(["dream", "--provider", "nonsense"])


def test_nell_dream_unknown_persona_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persona dir missing → FileNotFoundError mentioning 'persona'."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "empty"))
    from brain.cli import main

    with pytest.raises(FileNotFoundError, match="persona"):
        main(["dream", "--persona", "ghost", "--provider", "fake"])
