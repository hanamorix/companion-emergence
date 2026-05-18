# tests/forgetting/test_orchestrator.py
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import run_pass
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def persona_with_low_salience_memory(tmp_path):
    # Set up FeltTimeState with non-cold lived_age so recent-buffer
    # exemption doesn't apply.
    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )
    store = MemoryStore(tmp_path / "memories.db")
    # An OLD memory with no emotions, no soul link, no recalls → very low salience
    m = Memory.create_new(
        content="boring old memory body",
        memory_type="episodic",
        domain="chat",
        emotions={},
    )
    # Make it old enough to escape the recent-buffer exemption.
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=10))
    store.create(m)
    store.close()
    return tmp_path, m.id


def test_run_pass_fades_low_salience_memory(persona_with_low_salience_memory):
    persona_dir, mem_id = persona_with_low_salience_memory
    event_bus = MagicMock()
    summary = run_pass(persona_dir, event_bus=event_bus)
    assert summary["faded"] >= 1

    store = MemoryStore(persona_dir / "memories.db")
    row = store._conn.execute("SELECT state FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert row["state"] == "fading"
    store.close()


def test_run_pass_exempts_recent_buffer(tmp_path):
    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"), tmp_path
    )
    store = MemoryStore(tmp_path / "memories.db")
    # Just-created memory — within RECENT_LIVED_HOURS exemption window
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.close()

    event_bus = MagicMock()
    summary = run_pass(tmp_path, event_bus=event_bus)
    assert summary["faded"] == 0
    assert summary["exempt"] >= 1


def test_run_pass_lost_after_two_consecutive_low_passes(persona_with_low_salience_memory):
    persona_dir, mem_id = persona_with_low_salience_memory
    event_bus = MagicMock()
    # First pass: fades the memory (active → fading), consecutive_low_passes=1
    summary1 = run_pass(persona_dir, event_bus=event_bus)
    assert summary1["faded"] >= 1
    # Second pass: counter increments to 2, but LOSE threshold needs salience<0.10
    # — with no emotions, no recalls, no soul, no freshness, salience is essentially 0
    summary2 = run_pass(persona_dir, event_bus=event_bus)
    summary3 = run_pass(persona_dir, event_bus=event_bus)
    # By the third pass the memory should be lost OR fading->lost transition fired
    total_lost = summary1["lost"] + summary2["lost"] + summary3["lost"]
    assert total_lost >= 1

    store = MemoryStore(persona_dir / "memories.db")
    assert store.get(mem_id) is None  # row deleted
    store.close()

    # Graveyard entry exists
    from brain.forgetting import graveyard

    entries = graveyard.read_all(persona_dir)
    assert any(e["memory_id"] == mem_id for e in entries)


def test_run_pass_publishes_aggregate_event(persona_with_low_salience_memory):
    persona_dir, _mem_id = persona_with_low_salience_memory
    event_bus = MagicMock()
    summary = run_pass(persona_dir, event_bus=event_bus)
    assert "faded" in summary
    assert "unfaded" in summary
    assert "lost" in summary
    assert "exempt" in summary
    assert "total" in summary
    assert "duration_ms" in summary
    # event_bus should have been called once with this event
    event_bus.publish.assert_called_once()


def test_run_pass_cold_start_no_memories(tmp_path):
    event_bus = MagicMock()
    summary = run_pass(tmp_path, event_bus=event_bus)
    assert summary["total"] == 0
    assert summary["faded"] == 0
    assert summary["lost"] == 0


def test_run_pass_recovers_from_corrupt_forgetting_state(persona_with_low_salience_memory):
    persona_dir, _mem_id = persona_with_low_salience_memory
    # Pre-corrupt the state file.
    (persona_dir / "forgetting_state.json").write_text("{not valid json")
    event_bus = MagicMock()
    summary = run_pass(persona_dir, event_bus=event_bus)
    # Should NOT crash; consecutive_low_passes counters all reset to 0
    assert "faded" in summary
