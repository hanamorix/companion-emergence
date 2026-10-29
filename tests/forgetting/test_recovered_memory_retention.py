# tests/forgetting/test_recovered_memory_retention.py
"""A memory recovered from a stored research note carries the date of its fire but arrived now (#591).

Its recent buffer counts from ``metadata["recovered_at"]``, so the recovery does not undo itself: without it a memory
dated more than 30 days back would be faded by the next forgetting pass.
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import run_pass
from brain.forgetting.policy import is_exempt
from brain.memory.store import Memory, MemoryStore


def _research_memory(memory_id: str, *, age_days: float, recovered_days_ago: float | None) -> Memory:
    now = datetime.now(UTC)
    metadata = {}
    if recovered_days_ago is not None:
        metadata["recovered_at"] = (now - timedelta(days=recovered_days_ago)).isoformat()
    return Memory(
        id=memory_id,
        content=f"research memory {memory_id}",
        memory_type="research",
        domain="us",
        created_at=now - timedelta(days=age_days),
        importance=5.0,
        metadata=metadata,
    )


def _exempt(memory: Memory) -> bool:
    return is_exempt(memory, soul_crystallised_ids=set(), under_review_ids=set(), now_lived_age_hours=100.0)


def test_recovered_memory_is_exempt_for_30_days_after_recovery_and_the_normal_rules_resume_after():
    assert _exempt(_research_memory("a", age_days=45, recovered_days_ago=0))
    assert _exempt(_research_memory("b", age_days=45, recovered_days_ago=29))
    # Not exempt: no mark, or a mark older than the buffer.
    assert not _exempt(_research_memory("c", age_days=45, recovered_days_ago=None))
    assert not _exempt(_research_memory("d", age_days=80, recovered_days_ago=31))


def test_an_unparseable_or_naive_recovered_at_is_ignored():
    assert _exempt(_research_memory("ok", age_days=45, recovered_days_ago=0))  # the mark works when it is well formed
    for raw in ("not a date", "2026-10-28T12:00:00", 5):
        memory = _research_memory("e", age_days=45, recovered_days_ago=None)
        memory.metadata["recovered_at"] = raw
        assert not _exempt(memory)


def test_run_pass_keeps_a_recovered_memory_active_and_fades_the_same_memory_without_the_mark(tmp_path):
    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"), tmp_path
    )
    store = MemoryStore(tmp_path / "memories.db")
    store.create(_research_memory("recovered", age_days=45, recovered_days_ago=0))
    store.create(_research_memory("plain", age_days=45, recovered_days_ago=None))
    store.close()

    run_pass(tmp_path, event_bus=MagicMock())

    store = MemoryStore(tmp_path / "memories.db")
    states = {
        row["id"]: row["state"] for row in store._conn.execute("SELECT id, state FROM memories").fetchall()
    }
    store.close()
    assert states["recovered"] == "active"
    assert states["plain"] == "fading"
