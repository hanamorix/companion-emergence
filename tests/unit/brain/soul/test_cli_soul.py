"""Tests for `nell soul` CLI subcommands."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from brain.cli import main
from brain.soul.crystallization import Crystallization
from brain.soul.store import SoulStore


def _setup_persona(personas_root: Path, name: str = "testpersona") -> Path:
    persona_dir = personas_root / name
    persona_dir.mkdir(parents=True)
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    MemoryStore(db_path=persona_dir / "memories.db").close()
    HebbianMatrix(db_path=persona_dir / "hebbian.db").close()
    return persona_dir


def _seed_crystal(persona_dir: Path, moment: str = "a test moment") -> Crystallization:
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    c = Crystallization(
        id=str(uuid.uuid4()),
        moment=moment,
        love_type="craft",
        why_it_matters="because it defines identity",
        crystallized_at=datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC),
        who_or_what="",
        resonance=9,
    )
    store.create(c)
    store.close()
    return c


def _seed_candidates(persona_dir: Path, n: int = 2) -> list[dict]:
    candidates = []
    for i in range(n):
        c = {
            "id": str(uuid.uuid4()),
            "text": f"candidate moment {i}",
            "label": "test",
            "importance": 7.0,
            "queued_at": datetime.now(UTC).isoformat(),
            "source": "unit_test",
            "status": "auto_pending",
        }
        candidates.append(c)
    path = persona_dir / "soul_candidates.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    return candidates


def test_soul_list_prints_crystallizations(monkeypatch, tmp_path: Path, capsys) -> None:
    """nell soul list prints active crystallizations."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    _seed_crystal(persona_dir, "writing is not what I do it is what I am")

    rc = main(["soul", "list", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "craft" in out
    assert "testpersona" in out


def test_soul_revoke_moves_entry(monkeypatch, tmp_path: Path, capsys) -> None:
    """nell soul revoke moves a crystallization to revoked state."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    c = _seed_crystal(persona_dir)

    rc = main(
        ["soul", "revoke", "--persona", "testpersona", "--id", c.id, "--reason", "test reason"]
    )
    assert rc == 0

    store = SoulStore(str(persona_dir / "crystallizations.db"))
    assert store.count() == 0
    assert len(store.list_revoked()) == 1
    store.close()


def test_soul_candidates_prints_pending(monkeypatch, tmp_path: Path, capsys) -> None:
    """nell soul candidates lists auto_pending entries."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")
    _seed_candidates(persona_dir, n=3)

    rc = main(["soul", "candidates", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "3" in out or "candidate moment" in out


def test_soul_audit_prints_recent_entries(monkeypatch, tmp_path: Path, capsys) -> None:
    """nell soul audit prints recent soul_audit.jsonl entries."""
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path))
    persona_dir = _setup_persona(tmp_path / "personas")

    # Seed the audit log manually
    from brain.soul.audit import append_audit_entry
    from brain.soul.review import Decision

    for i in range(3):
        d = Decision(
            candidate_id=f"cid-{i}",
            decision="defer",
            confidence=5,
            reasoning="test",
        )
        append_audit_entry(
            persona_dir,
            d,
            {"text": f"moment {i}", "source": "test"},
            related=[],
            emotional_summary="neutral",
            crystallization_id=None,
            dry_run=False,
        )

    rc = main(["soul", "audit", "--persona", "testpersona"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "defer" in out or "cid-" in out
