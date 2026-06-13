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
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=40))
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


def test_run_pass_exempts_recent_factual_memory_within_grace(tmp_path):
    """Part A through-pass: the exact user report — a memory from a few days
    ago, no emotion/recall/soul, must stay ACTIVE under the 30-day grace.

    RED before Part A: 5 wall-days (120h) exceeds the old 24h grace, so this
    memory fades. GREEN after: 120h < 720h → exempt, stays active.
    """
    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"), tmp_path
    )
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(
        content="we talked about the dog's name a few days ago",
        memory_type="episodic",
        domain="chat",
        emotions={},
    )
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=5))
    store.create(m)
    store.close()

    summary = run_pass(tmp_path, event_bus=MagicMock())
    assert summary["faded"] == 0
    assert summary["exempt"] >= 1

    store = MemoryStore(tmp_path / "memories.db")
    row = store._conn.execute(
        "SELECT state FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["state"] == "active"
    store.close()


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


def test_run_pass_intensity_drivers_protects_borderline_memory(tmp_path):
    """intensity_drivers.narrative_weight threads to policy and protects memories from fading."""
    from datetime import timedelta
    from unittest.mock import patch

    from brain.felt_time.lived_age import IntensityDrivers

    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(content="arc memory", memory_type="episodic", domain="chat", emotions={})
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=40))
    store.create(m)
    store.close()

    # Pin salience to just below baseline FADE_THRESHOLD but above the arc-adjusted threshold.
    # effective_fade at narrative_weight=1.0 → FADE_THRESHOLD / 2 = 0.125
    borderline_salience = 0.20  # 0.125 < 0.20 < 0.25

    event_bus = MagicMock()
    with patch("brain.forgetting.salience.score", return_value=borderline_salience):
        # Without arc pressure: salience 0.20 < FADE_THRESHOLD 0.25 → should fade
        summary_no_arc = run_pass(tmp_path, event_bus=event_bus)
        assert summary_no_arc["faded"] >= 1, "baseline: borderline memory should fade without arc"

    # Reload memory to active state
    store = MemoryStore(tmp_path / "memories.db")
    store._conn.execute("UPDATE memories SET state = 'active' WHERE id = ?", (m.id,))
    store._conn.commit()
    store.close()

    event_bus2 = MagicMock()
    drivers = IntensityDrivers(narrative_weight=1.0)
    with patch("brain.forgetting.salience.score", return_value=borderline_salience):
        # With full arc pressure: effective_fade = 0.125, 0.20 >= 0.125 → should NOT fade
        summary_arc = run_pass(tmp_path, event_bus=event_bus2, intensity_drivers=drivers)
        assert summary_arc["faded"] == 0, "arc pressure: borderline memory should be protected"


def test_run_pass_recovers_from_corrupt_forgetting_state(persona_with_low_salience_memory):
    persona_dir, _mem_id = persona_with_low_salience_memory
    # Pre-corrupt the state file.
    (persona_dir / "forgetting_state.json").write_text("{not valid json")
    event_bus = MagicMock()
    summary = run_pass(persona_dir, event_bus=event_bus)
    # Should NOT crash; consecutive_low_passes counters all reset to 0
    assert "faded" in summary


def test_run_pass_exempts_memory_under_soul_review_through_real_candidates_file(tmp_path):
    """CANARY — pins the full composition run_pass → _load_soul_linked_ids →
    brain.soul.candidates.list_under_review_memory_ids against a REAL
    soul_candidates.jsonl.

    History: forgetting's lazy import of brain.soul.candidates referenced a
    module that did not exist from v0.0.14 until 2026-06-12 (the fail-soft
    except swallowed the ImportError), so the under-review exemption was
    silently dead for ~18 versions. This test fails if that composition ever
    breaks again (module rename, signature drift, status-set drift).
    """
    from brain.ingest.soul_queue import queue_soul_candidate
    from brain.ingest.types import ExtractedItem

    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )
    store = MemoryStore(tmp_path / "memories.db")
    # Identical low-salience shape to the fades test — old, no emotions, no recalls.
    m = Memory.create_new(
        content="quiet moment awaiting soul review",
        memory_type="monologue_soul_candidate",
        domain="monologue",
        emotions={},
    )
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=40))
    store.create(m)
    store.close()

    # Real candidates file via the real queue path: auto_pending = under review.
    queue_soul_candidate(
        tmp_path,
        memory_id=m.id,
        item=ExtractedItem(text="quiet moment awaiting soul review", label="observation", importance=8),
        session_id="monologue",
    )

    summary = run_pass(tmp_path, event_bus=MagicMock())

    store = MemoryStore(tmp_path / "memories.db")
    row = store._conn.execute("SELECT state FROM memories WHERE id = ?", (m.id,)).fetchone()
    store.close()
    assert row["state"] == "active", (
        "memory under soul review must be exempt from fading — the "
        "run_pass→soul.candidates composition is broken"
    )
    assert summary["exempt"] >= 1
