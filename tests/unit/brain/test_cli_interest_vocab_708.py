"""#708: `nell interest list` only reads, so it must not repair the emotion vocabulary.

It loaded the vocabulary with the default heal and no store, so a corrupt
`emotion_vocabulary.json` with no usable backup was quarantined and reset to
`emotions: []` with no rebuild from memories: the persona's names were lost from a
command that should only list interests. Through the real CLI entry point with a
temp KINDLED_HOME; no mocked state files.
"""
from __future__ import annotations

from pathlib import Path

from brain.cli import main


def _persona(home: Path, name: str = "testpersona") -> Path:
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = home / "personas" / name
    persona_dir.mkdir(parents=True)
    MemoryStore(db_path=persona_dir / "memories.db").close()
    HebbianMatrix(db_path=persona_dir / "hebbian.db").close()
    (persona_dir / "interests.json").write_text('{"version": 1, "interests": []}', encoding="utf-8")
    return persona_dir


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def test_interest_list_leaves_a_corrupt_vocabulary_file_untouched(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _persona(tmp_path)
    (persona_dir / "emotion_vocabulary.json").write_text('{"version": 1, "emotions": [{"name": "fernweh"', encoding="utf-8")
    before = _snapshot(persona_dir)

    assert main(["interest", "list", "--persona", "testpersona"]) == 0
    assert "testpersona" in capsys.readouterr().out
    assert _snapshot(persona_dir) == before
