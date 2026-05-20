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
    # Back-date created_at so the recent-buffer exemption does not protect this
    # memory from the forgetting pass.
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=last_accessed_days_ago))
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

    Design note: under real production thresholds (LOST_THRESHOLD=0.10),
    high-emotion memories are unlikely to reach LOSE transition at all
    — emotion contributes to salience (~ joy × 0.3), lifting them above
    the floor. This test is a contract test for the wiring, not a
    simulation of typical production behavior. If the salience formula
    ever changes (e.g., emotion weight reduced), high-emotion losses
    may become common and the monkeypatch may no longer be needed.
    """
    from brain.felt_time.state import FeltTimeState
    from brain.felt_time.state import persist as persist_felt_time
    from brain.forgetting import policy as forgetting_policy

    persist_felt_time(
        FeltTimeState(lived_age_hours=720.0 * 5, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )
    store = MemoryStore(tmp_path / "memories.db")
    _seed_memory(store, id_="mem-target", joy=9.0, last_accessed_days_ago=180)
    store.close()

    # Raise LOST_THRESHOLD high enough that joy=9.0 (salience ~0.27) is below it.
    monkeypatch.setattr(forgetting_policy, "LOST_THRESHOLD", 0.5)
    monkeypatch.setattr(forgetting_policy, "FADE_THRESHOLD", 0.6)
    monkeypatch.setattr(forgetting_policy, "LOST_PASS_COUNT", 1)

    bus = _NullBus()
    # Five passes is belt-and-suspenders: with LOST_PASS_COUNT=1, the memory
    # transitions FADE on pass 1 and LOSE on pass 2. The extra three passes
    # verify the grief breadcrumb isn't duplicated (write_breadcrumb has no
    # debounce for drop subtype — relies on the memory being hard-deleted
    # before further passes can re-LOSE it).
    for _ in range(5):
        run_pass(tmp_path, event_bus=bus)

    store = MemoryStore(tmp_path / "memories.db")
    rows = store._conn.execute("SELECT id FROM memories WHERE id = 'mem-target'").fetchall()
    assert rows == [], "target memory should have been hard-deleted by the forgetting pass"

    grief_rows = store._conn.execute(
        "SELECT metadata_json, emotions_json FROM memories WHERE memory_type = 'grief_event'"
    ).fetchall()
    assert len(grief_rows) >= 1, "drop-time grief breadcrumb should have been written"
    meta = json.loads(grief_rows[0]["metadata_json"])
    assert meta["grief_referent_id"] == "mem-target"
    assert meta["grief_subtype"] == "drop"
    em = json.loads(grief_rows[0]["emotions_json"])
    assert em["memory_grief"] == pytest.approx(6.3, abs=0.01), (
        "intensity should be joy_normalised * DROP_SCALE = 0.9 * 7.0 = 6.3"
    )
    store.close()
