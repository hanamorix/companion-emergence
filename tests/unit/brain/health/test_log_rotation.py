"""Tests for brain.health.log_rotation — rolling-size + yearly-archive rotation.

Spec: docs/superpowers/specs/2026-05-11-jsonl-log-retention-design.md
Plan: docs/superpowers/plans/2026-05-11-jsonl-log-retention.md (Phase 2)
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from brain.health.log_rotation import (
    rotate_age_archive_yearly,
    rotate_rolling_size,
)

# ---------------------------------------------------------------------------
# rotate_rolling_size
# ---------------------------------------------------------------------------


def _write_lines(path: Path, n: int, prefix: str = "x") -> None:
    """Write n lines of roughly 1 KB each so size hits cap quickly."""
    big = "x" * 1000
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"i": i, "p": prefix, "pad": big}) + "\n")


def test_rotate_rolling_size_no_op_below_cap(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    _write_lines(log, 1)  # ~1 KB
    result = rotate_rolling_size(log, max_bytes=10_000, archive_keep=3)
    assert result is None
    assert log.exists()
    assert not (tmp_path / "log.jsonl.1.gz").exists()


def test_rotate_rolling_size_missing_log_returns_none(tmp_path: Path) -> None:
    """No log file → no-op, no error."""
    log = tmp_path / "never_written.jsonl"
    result = rotate_rolling_size(log, max_bytes=1000, archive_keep=3)
    assert result is None


def test_rotate_rolling_size_at_cap_creates_archive(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    _write_lines(log, 20)  # ~20 KB
    result = rotate_rolling_size(log, max_bytes=10_000, archive_keep=3)
    archive = tmp_path / "log.jsonl.1.gz"
    assert result == archive
    assert archive.exists()
    # Active log was rotated out — either gone or freshly recreated empty.
    if log.exists():
        assert log.stat().st_size == 0


def test_rotate_rolling_size_archive_content_round_trips(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    _write_lines(log, 20, prefix="payload")
    original = log.read_bytes()
    rotate_rolling_size(log, max_bytes=10_000, archive_keep=3)
    archive = tmp_path / "log.jsonl.1.gz"
    with gzip.open(archive, "rb") as gz:
        assert gz.read() == original


def test_rotate_rolling_size_shifts_existing_archives(tmp_path: Path) -> None:
    """Pre-existing .1.gz + .2.gz shift up; new content lands in .1.gz."""
    log = tmp_path / "log.jsonl"
    # Seed existing archives so we can verify the shift.
    (tmp_path / "log.jsonl.1.gz").write_bytes(gzip.compress(b"old1\n"))
    (tmp_path / "log.jsonl.2.gz").write_bytes(gzip.compress(b"old2\n"))

    _write_lines(log, 20)
    rotate_rolling_size(log, max_bytes=10_000, archive_keep=3)

    # New archive at .1.gz; old .1.gz → .2.gz; old .2.gz → .3.gz.
    assert (tmp_path / "log.jsonl.1.gz").exists()
    assert gzip.decompress((tmp_path / "log.jsonl.2.gz").read_bytes()) == b"old1\n"
    assert gzip.decompress((tmp_path / "log.jsonl.3.gz").read_bytes()) == b"old2\n"


def test_rotate_rolling_size_evicts_oldest_when_keep_reached(tmp_path: Path) -> None:
    """archive_keep=3, run four rotations → only .1/.2/.3.gz remain."""
    log = tmp_path / "log.jsonl"
    for i in range(4):
        _write_lines(log, 20, prefix=f"gen{i}")
        rotate_rolling_size(log, max_bytes=10_000, archive_keep=3)

    assert (tmp_path / "log.jsonl.1.gz").exists()
    assert (tmp_path / "log.jsonl.2.gz").exists()
    assert (tmp_path / "log.jsonl.3.gz").exists()
    assert not (tmp_path / "log.jsonl.4.gz").exists()

    # Newest archive (.1.gz) holds gen3, oldest (.3.gz) holds gen1, gen0 is evicted.
    newest = gzip.decompress((tmp_path / "log.jsonl.1.gz").read_bytes())
    oldest = gzip.decompress((tmp_path / "log.jsonl.3.gz").read_bytes())
    assert b"gen3" in newest
    assert b"gen1" in oldest


# ---------------------------------------------------------------------------
# rotate_age_archive_yearly
# ---------------------------------------------------------------------------


def _ts(year: int, month: int = 1, day: int = 1) -> str:
    return datetime(year, month, day, tzinfo=timezone.utc).isoformat()


def _write_yearly_lines(path: Path, entries_per_year: dict[int, int]) -> None:
    """Write JSONL entries with `at` timestamps spread across given years."""
    with path.open("w", encoding="utf-8") as f:
        for year, count in entries_per_year.items():
            for i in range(count):
                f.write(json.dumps({"at": _ts(year, 1, 1), "seq": i}) + "\n")


def test_rotate_age_archive_no_op_when_current_year_only(tmp_path: Path) -> None:
    log = tmp_path / "soul_audit.jsonl"
    _write_yearly_lines(log, {2026: 5})
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    result = rotate_age_archive_yearly(log, now=now)
    assert result == []
    # Active file unchanged.
    assert sum(1 for _ in log.open()) == 5


def test_rotate_age_archive_missing_log_returns_empty(tmp_path: Path) -> None:
    result = rotate_age_archive_yearly(tmp_path / "missing.jsonl")
    assert result == []


def test_rotate_age_archive_splits_by_year(tmp_path: Path) -> None:
    log = tmp_path / "soul_audit.jsonl"
    _write_yearly_lines(log, {2024: 3, 2025: 4, 2026: 2})
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    archives = rotate_age_archive_yearly(log, now=now)

    # Two archives for the two cold years.
    archive_paths = {a.name for a in archives}
    assert archive_paths == {"soul_audit.2024.jsonl.gz", "soul_audit.2025.jsonl.gz"}

    # 2024 archive has its 3 lines.
    with gzip.open(tmp_path / "soul_audit.2024.jsonl.gz", "rt") as gz:
        rows = [json.loads(line) for line in gz if line.strip()]
    assert len(rows) == 3
    assert all("2024" in r["at"] for r in rows)

    # Active file has only 2026 entries.
    with log.open() as f:
        active_rows = [json.loads(line) for line in f if line.strip()]
    assert len(active_rows) == 2
    assert all("2026" in r["at"] for r in active_rows)


def test_rotate_age_archive_preserves_within_year_order(tmp_path: Path) -> None:
    log = tmp_path / "soul_audit.jsonl"
    _write_yearly_lines(log, {2024: 5})
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    rotate_age_archive_yearly(log, now=now)

    with gzip.open(tmp_path / "soul_audit.2024.jsonl.gz", "rt") as gz:
        rows = [json.loads(line) for line in gz if line.strip()]
    assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4]


def test_rotate_age_archive_idempotent_same_year(tmp_path: Path) -> None:
    """Calling twice within the same year shouldn't re-archive 2026 entries."""
    log = tmp_path / "soul_audit.jsonl"
    _write_yearly_lines(log, {2025: 3, 2026: 2})
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)

    first = rotate_age_archive_yearly(log, now=now)
    second = rotate_age_archive_yearly(log, now=now)

    assert len(first) == 1
    assert second == []
    # Active still has the two 2026 lines.
    with log.open() as f:
        active = [json.loads(line) for line in f if line.strip()]
    assert len(active) == 2


def test_rotate_age_archive_skips_corrupt_lines(tmp_path: Path) -> None:
    """A malformed line in the middle doesn't abort the split."""
    log = tmp_path / "soul_audit.jsonl"
    log.write_text(
        json.dumps({"at": _ts(2024), "seq": 0}) + "\n"
        + "{this is not json\n"
        + json.dumps({"at": _ts(2024), "seq": 1}) + "\n"
        + json.dumps({"at": _ts(2026), "seq": 2}) + "\n",
        encoding="utf-8",
    )
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    archives = rotate_age_archive_yearly(log, now=now)
    assert len(archives) == 1

    with gzip.open(tmp_path / "soul_audit.2024.jsonl.gz", "rt") as gz:
        rows = [json.loads(line) for line in gz if line.strip()]
    assert [r["seq"] for r in rows] == [0, 1]


# ---------------------------------------------------------------------------
# Atomicity — partial-failure recovery
# ---------------------------------------------------------------------------


def test_rotate_rolling_size_active_intact_on_gzip_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If gzip write fails mid-rotation, the active log must still be readable.

    Simulates a disk-full / permissions failure during the archive write.
    Rotation should leave the active log in a usable state — even if the
    rotation itself fails — because the alternative is losing audit data.
    """
    log = tmp_path / "log.jsonl"
    _write_lines(log, 20)
    original = log.read_bytes()

    def boom(*a, **kw):
        raise OSError("simulated disk full")

    monkeypatch.setattr("brain.health.log_rotation.gzip.open", boom)
    with pytest.raises(OSError):
        rotate_rolling_size(log, max_bytes=10_000, archive_keep=3)

    # Active log still exists with its original content.
    assert log.exists()
    assert log.read_bytes() == original
