"""Tests for `nell soul audit` and `nell soul audit --full`.

Spec: docs/superpowers/specs/2026-05-11-jsonl-log-retention-design.md (Phase 5)
"""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from brain.cli import _soul_audit_handler


def _ts(year: int) -> str:
    return datetime(year, 6, 15, tzinfo=UTC).isoformat()


def _make_persona(
    tmp_path: Path,
    *,
    active: list[dict] | None = None,
    archives: dict[int, list[dict]] | None = None,
) -> Path:
    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    if active:
        with (persona_dir / "soul_audit.jsonl").open("w", encoding="utf-8") as f:
            for e in active:
                f.write(json.dumps(e) + "\n")
    if archives:
        for year, entries in archives.items():
            with gzip.open(
                persona_dir / f"soul_audit.{year}.jsonl.gz", "wt", encoding="utf-8"
            ) as gz:
                for e in entries:
                    gz.write(json.dumps(e) + "\n")
    return persona_dir


def _args(persona_dir: Path, *, limit: int = 20, full: bool = False) -> argparse.Namespace:
    return argparse.Namespace(persona=persona_dir.name, limit=limit, full=full)


def _entry(year: int, marker: str) -> dict:
    return {
        "ts": _ts(year),
        "candidate_id": f"cid-{marker}",
        "decision": "accept",
        "confidence": 9,
        "love_type": "test",
        "reasoning": f"reason-{marker}",
        "dry_run": False,
    }


def test_soul_audit_default_uses_active_file_only(tmp_path: Path, capsys, monkeypatch) -> None:
    """Without --full, only the active soul_audit.jsonl tail is shown."""
    persona_dir = _make_persona(
        tmp_path,
        active=[_entry(2026, "active1"), _entry(2026, "active2")],
        archives={2024: [_entry(2024, "old1")]},
    )
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _soul_audit_handler(_args(persona_dir, full=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "active1" in out
    assert "active2" in out
    assert "old1" not in out


def test_soul_audit_full_walks_all_archives_chronologically(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """With --full, archived 2024/2025 entries appear before 2026 active ones."""
    persona_dir = _make_persona(
        tmp_path,
        active=[_entry(2026, "active1")],
        archives={
            2024: [_entry(2024, "y24_a"), _entry(2024, "y24_b")],
            2025: [_entry(2025, "y25_a")],
        },
    )
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _soul_audit_handler(_args(persona_dir, full=True))
    assert rc == 0
    out = capsys.readouterr().out
    # All entries present.
    for marker in ("y24_a", "y24_b", "y25_a", "active1"):
        assert marker in out, f"missing {marker} in output"
    # Order: 2024 → 2025 → 2026.
    assert out.index("y24_a") < out.index("y25_a") < out.index("active1")


def test_soul_audit_full_no_archives_falls_back_to_active(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """--full with no archives still works — just yields active entries."""
    persona_dir = _make_persona(
        tmp_path,
        active=[_entry(2026, "only_active")],
    )
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _soul_audit_handler(_args(persona_dir, full=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "only_active" in out


def test_soul_audit_full_empty_persona_returns_zero(tmp_path: Path, capsys, monkeypatch) -> None:
    """--full with no active file and no archives → clean empty output, rc=0."""
    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    monkeypatch.setattr("brain.cli.get_persona_dir", lambda _name: persona_dir)
    rc = _soul_audit_handler(_args(persona_dir, full=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "(empty)" in out
