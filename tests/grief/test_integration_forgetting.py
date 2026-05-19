"""test_integration_forgetting.py — drop-time grief writes inside forgetting.run_pass."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from brain.forgetting import run_pass
from brain.memory.store import Memory, MemoryStore


class _NullBus:
    def publish(self, _evt: Any) -> None:
        pass


def _seed_memory(
    store: MemoryStore,
    *,
    id_: str,
    joy: float,
    last_accessed_days_ago: int,
) -> None:
    m = Memory.create_new(
        content="we drove out to the rooftop and watched the morning come up over the trees",
        memory_type="episodic",
        domain="memory",
        emotions={"joy": joy},
    )
    object.__setattr__(m, "id", id_)
    # Back-date created_at so the recent-buffer exemption (RECENT_LIVED_HOURS=24h
    # wall-clock) does not protect this memory from the forgetting pass.
    object.__setattr__(
        m, "created_at", datetime.now(UTC) - timedelta(days=last_accessed_days_ago)
    )
    store.create(m)
    cutoff = (
        datetime.now(UTC).replace(microsecond=0) - timedelta(days=last_accessed_days_ago)
    ).isoformat()
    store._conn.execute(
        "UPDATE memories SET last_accessed_at = ?, recall_count = 0 WHERE id = ?",
        (cutoff, id_),
    )
    store._conn.commit()


def test_forgetting_pass_writes_grief_breadcrumb_on_high_emotion_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A high-emotion memory that ages out via the forgetting pass should
    produce a grief breadcrumb attached to its now-deleted id.

    The intensity formula (emotion × salience × 7.0) yields < 0.70 at LOSE
    time because salience < LOST_THRESHOLD (0.10).  We monkeypatch
    brain.grief.policy.THRESHOLD to 0.0 so the test verifies the wiring —
    handle_drop is called and writes a breadcrumb — independently of the
    threshold gate, which is covered by unit tests in test_breadcrumb.py.
    """
    import brain.grief.policy as grief_policy

    monkeypatch.setattr(grief_policy, "THRESHOLD", 0.0)

    from brain.felt_time.state import FeltTimeState
    from brain.felt_time.state import persist as persist_felt_time

    persist_felt_time(
        FeltTimeState(lived_age_hours=720.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )
    store = MemoryStore(tmp_path / "memories.db")
    # joy=0.0 → emotion salience ≈ 0; created_at + last_accessed 60 days ago →
    # freshness ≈ 0; no soul link, no hebbian, no recalls → salience ≈ 0 <
    # LOST_THRESHOLD (0.10) so the LOSE transition fires after LOST_PASS_COUNT passes.
    _seed_memory(store, id_="mem-target", joy=0.0, last_accessed_days_ago=60)
    store.close()

    bus = _NullBus()
    for _ in range(5):
        run_pass(tmp_path, event_bus=bus)

    store = MemoryStore(tmp_path / "memories.db")
    rows = store._conn.execute("SELECT id FROM memories WHERE id = 'mem-target'").fetchall()
    assert rows == [], "target memory should have been hard-deleted by the forgetting pass"

    grief_rows = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE memory_type = 'grief_event'"
    ).fetchall()
    assert len(grief_rows) >= 1, "drop-time grief breadcrumb should have been written"
    meta = json.loads(grief_rows[0]["metadata_json"])
    assert meta["grief_referent_id"] == "mem-target"
    assert meta["grief_subtype"] == "drop"
    store.close()
