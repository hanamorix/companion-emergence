"""Tests for `nell initiate` CLI subcommands."""

from __future__ import annotations

import argparse
import pytest
pytest.importorskip("brain.initiate")

from pathlib import Path

from brain.cli import (
    _initiate_audit_handler,
    _initiate_candidates_handler,
    _initiate_voice_evolution_handler,
)
from brain.initiate.audit import append_audit_row
from brain.initiate.schemas import AuditRow


def _args(persona_dir: Path, **kw) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "persona": persona_dir.name,
        "limit": 20,
        "full": False,
    }
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _seed_one_audit(persona_dir: Path) -> None:
    row = AuditRow(
        audit_id="ia_001", candidate_id="ic_001",
        ts="2026-05-11T14:00:00+00:00", kind="message",
        subject="the dream", tone_rendered="x",
        decision="send_quiet", decision_reasoning="x",
        gate_check={"allowed": True, "reason": None}, delivery=None,
    )
    row.record_transition("delivered", row.ts)
    append_audit_row(persona_dir, row)


def test_initiate_audit_default_tails_active(tmp_path: Path, capsys, monkeypatch) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    _seed_one_audit(persona_dir)
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_audit_handler(_args(persona_dir, full=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "ia_001" in out or "the dream" in out


def test_initiate_audit_full_walks_archives(tmp_path: Path, capsys, monkeypatch) -> None:
    import gzip
    import json
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    _seed_one_audit(persona_dir)
    # Add an archive file.
    archive = persona_dir / "initiate_audit.2024.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as gz:
        gz.write(json.dumps({
            "audit_id": "ia_old", "candidate_id": "ic_old",
            "ts": "2024-06-15T00:00:00+00:00", "kind": "message",
            "subject": "old subject", "tone_rendered": "",
            "decision": "send_quiet", "decision_reasoning": "",
            "gate_check": {"allowed": True, "reason": None},
            "delivery": None,
        }) + "\n")
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_audit_handler(_args(persona_dir, full=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "old subject" in out
    assert out.index("old subject") < out.index("the dream")


def test_initiate_candidates_shows_queue(tmp_path: Path, capsys, monkeypatch) -> None:
    from brain.initiate.emit import emit_initiate_candidate
    from brain.initiate.schemas import EmotionalSnapshot, SemanticContext
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    emit_initiate_candidate(
        persona_dir,
        kind="message", source="dream", source_id="dream_abc",
        emotional_snapshot=EmotionalSnapshot(
            vector={}, rolling_baseline_mean=0, rolling_baseline_stdev=0,
            current_resonance=0, delta_sigma=0,
        ),
        semantic_context=SemanticContext(),
    )
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_candidates_handler(_args(persona_dir))
    assert rc == 0
    out = capsys.readouterr().out
    assert "dream_abc" in out


def test_initiate_voice_evolution_lists_records(tmp_path: Path, capsys, monkeypatch) -> None:
    from brain.soul.store import SoulStore, VoiceEvolution
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        store.save_voice_evolution(VoiceEvolution(
            id="ve_1", accepted_at="2026-05-11T00:00:00+00:00",
            diff="", old_text="A", new_text="B", rationale="x",
            evidence=[], audit_id="ia_1", user_modified=False,
        ))
    finally:
        store.close()
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _initiate_voice_evolution_handler(_args(persona_dir))
    assert rc == 0
    out = capsys.readouterr().out
    assert "A" in out and "B" in out
