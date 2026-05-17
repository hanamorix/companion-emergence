"""Tests for soul-audit fan-out reader spanning archives.

Spec: docs/superpowers/specs/2026-05-11-jsonl-log-retention-design.md
Plan: docs/superpowers/plans/2026-05-11-jsonl-log-retention.md (Phase 3)
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.soul.audit import iter_audit_full


def _ts(year: int) -> str:
    return datetime(year, 6, 15, tzinfo=UTC).isoformat()


def _write_active(persona_dir: Path, entries: list[dict]) -> None:
    p = persona_dir / "soul_audit.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _write_archive(persona_dir: Path, year: int, entries: list[dict]) -> None:
    p = persona_dir / f"soul_audit.{year}.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as gz:
        for e in entries:
            gz.write(json.dumps(e) + "\n")


def test_iter_audit_full_no_logs_returns_empty(tmp_path: Path) -> None:
    """Fresh persona with no audit history → empty iterator."""
    assert list(iter_audit_full(tmp_path)) == []


def test_iter_audit_full_active_only(tmp_path: Path) -> None:
    """No archives → yields just active-file lines."""
    _write_active(
        tmp_path,
        [
            {"ts": _ts(2026), "seq": 0},
            {"ts": _ts(2026), "seq": 1},
        ],
    )
    out = list(iter_audit_full(tmp_path))
    assert [e["seq"] for e in out] == [0, 1]


def test_iter_audit_full_chronological_across_archives(tmp_path: Path) -> None:
    """Active + 2024 + 2025 archives → 2024 entries first, then 2025, then active."""
    _write_archive(
        tmp_path,
        2025,
        [
            {"ts": _ts(2025), "seq": "y25_a"},
            {"ts": _ts(2025), "seq": "y25_b"},
        ],
    )
    _write_archive(
        tmp_path,
        2024,
        [
            {"ts": _ts(2024), "seq": "y24_a"},
        ],
    )
    _write_active(
        tmp_path,
        [
            {"ts": _ts(2026), "seq": "active_a"},
            {"ts": _ts(2026), "seq": "active_b"},
        ],
    )
    out = list(iter_audit_full(tmp_path))
    seqs = [e["seq"] for e in out]
    assert seqs == ["y24_a", "y25_a", "y25_b", "active_a", "active_b"]


def test_iter_audit_full_skips_corrupt_in_gz_archive(tmp_path: Path, caplog) -> None:
    """A malformed line inside a .gz archive is skipped, not aborted on."""
    caplog.set_level(logging.WARNING)
    # Hand-craft an archive with one bad line in the middle.
    archive = tmp_path / "soul_audit.2024.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as gz:
        gz.write(json.dumps({"ts": _ts(2024), "seq": "a"}) + "\n")
        gz.write("not valid json\n")
        gz.write(json.dumps({"ts": _ts(2024), "seq": "b"}) + "\n")
    _write_active(tmp_path, [{"ts": _ts(2026), "seq": "active"}])

    out = list(iter_audit_full(tmp_path))
    seqs = [e["seq"] for e in out]
    assert seqs == ["a", "b", "active"]
    # The skip emitted a warning.
    assert any(
        "malformed" in r.message.lower() or "skipping" in r.message.lower() for r in caplog.records
    )


def test_iter_audit_full_archives_only_no_active(tmp_path: Path) -> None:
    """Archives exist but active file doesn't → still yields archive entries."""
    _write_archive(tmp_path, 2024, [{"ts": _ts(2024), "seq": "x"}])
    out = list(iter_audit_full(tmp_path))
    assert [e["seq"] for e in out] == ["x"]


def test_iter_audit_full_is_streaming_generator(tmp_path: Path) -> None:
    """iter_audit_full returns a generator that can be partially consumed."""
    _write_archive(tmp_path, 2024, [{"ts": _ts(2024), "seq": i} for i in range(100)])
    it = iter_audit_full(tmp_path)
    # Take just the first entry; this proves the rest isn't materialised.
    first = next(iter(it))
    assert first["seq"] == 0
