"""#176: age-based reaping of stale lock archives + corruption quarantines."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.sidecar_sweep import sweep_stale_sidecars

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def _touch(path: Path, *, age_days: float) -> Path:
    path.write_bytes(b"")
    ts = (_NOW - timedelta(days=age_days)).timestamp()
    os.utime(path, (ts, ts))
    return path


def test_reaps_old_stale_locks_and_corrupt_quarantines(tmp_path: Path) -> None:
    old_stale = _touch(tmp_path / "bridge.json.lock.stale-20260801T000000Z", age_days=30)
    old_corrupt = _touch(tmp_path / "emotion_vocabulary.json.corrupt-20260801T000000Z", age_days=30)
    old_bak_corrupt = _touch(tmp_path / "soul.json.bak1.corrupt-20260801T000000Z", age_days=30)

    removed = sweep_stale_sidecars(tmp_path, now=_NOW)

    assert set(removed) == {old_stale, old_corrupt, old_bak_corrupt}
    assert not old_stale.exists() and not old_corrupt.exists() and not old_bak_corrupt.exists()


def test_keeps_young_sidecars_and_live_locks(tmp_path: Path) -> None:
    young_stale = _touch(tmp_path / "bridge.json.lock.stale-20260903T000000Z", age_days=2)
    young_corrupt = _touch(tmp_path / "state.json.corrupt-20260903T000000Z", age_days=2)
    live_lock = _touch(tmp_path / "pending_candidates.jsonl.lock", age_days=400)
    data = _touch(tmp_path / "memories.db", age_days=400)

    removed = sweep_stale_sidecars(tmp_path, now=_NOW)

    assert removed == []
    for p in (young_stale, young_corrupt, live_lock, data):
        assert p.exists()


def test_missing_dir_is_a_noop(tmp_path: Path) -> None:
    assert sweep_stale_sidecars(tmp_path / "nope", now=_NOW) == []
