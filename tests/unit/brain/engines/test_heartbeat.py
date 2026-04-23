"""Tests for brain.engines.heartbeat — event-driven orchestrator tick."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.engines.heartbeat import (
    HeartbeatConfig,
    HeartbeatEngine,
    HeartbeatResult,
    HeartbeatState,
)


def test_heartbeat_config_defaults() -> None:
    """HeartbeatConfig has sensible defaults per spec."""
    c = HeartbeatConfig()
    assert c.dream_every_hours == 24.0
    assert c.decay_rate_per_tick == 0.01
    assert c.gc_threshold == 0.01
    assert c.emit_memory == "conditional"


def test_heartbeat_config_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    """Missing config file → defaults-populated config."""
    c = HeartbeatConfig.load(tmp_path / "does_not_exist.json")
    assert c.dream_every_hours == 24.0
    assert c.emit_memory == "conditional"


def test_heartbeat_config_save_and_reload_round_trips(tmp_path: Path) -> None:
    """save() then load() preserves all fields."""
    original = HeartbeatConfig(
        dream_every_hours=6.0,
        decay_rate_per_tick=0.05,
        gc_threshold=0.02,
        emit_memory="always",
    )
    path = tmp_path / "cfg.json"
    original.save(path)
    restored = HeartbeatConfig.load(path)
    assert restored == original


def test_heartbeat_config_load_tolerates_unknown_fields(tmp_path: Path) -> None:
    """Forward-compat: unknown fields in config JSON are ignored."""
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "dream_every_hours": 12.0,
                "unknown_future_field": "abc",
            }
        )
    )
    c = HeartbeatConfig.load(path)
    assert c.dream_every_hours == 12.0


def test_heartbeat_config_invalid_emit_memory_falls_back_to_default(tmp_path: Path) -> None:
    """Invalid emit_memory value falls back to 'conditional' (safe default)."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"emit_memory": "nonsense"}))
    c = HeartbeatConfig.load(path)
    assert c.emit_memory == "conditional"


def test_heartbeat_state_load_missing_file_returns_none(tmp_path: Path) -> None:
    """HeartbeatState.load() returns None for first-ever tick detection."""
    assert HeartbeatState.load(tmp_path / "state.json") is None


def test_heartbeat_state_fresh_creates_baseline() -> None:
    """HeartbeatState.fresh() returns a state with all timestamps=now, tick_count=0."""
    s = HeartbeatState.fresh(trigger="init")
    assert s.tick_count == 0
    assert s.last_trigger == "init"
    now = datetime.now(UTC)
    assert abs((now - s.last_tick_at).total_seconds()) < 5
    assert abs((now - s.last_dream_at).total_seconds()) < 5


def test_heartbeat_state_save_and_load_round_trips(tmp_path: Path) -> None:
    """State JSON round-trips: ISO8601 Z-suffix timestamps parse cleanly."""
    when = datetime(2026, 4, 23, 10, 0, 0, tzinfo=UTC)
    original = HeartbeatState(
        last_tick_at=when,
        last_dream_at=when - timedelta(hours=6),
        last_research_at=when,
        tick_count=5,
        last_trigger="open",
    )
    path = tmp_path / "state.json"
    original.save(path)

    loaded = HeartbeatState.load(path)
    assert loaded is not None
    assert loaded.tick_count == 5
    assert loaded.last_trigger == "open"
    assert loaded.last_tick_at == when


def test_heartbeat_state_save_is_atomic(tmp_path: Path) -> None:
    """save() writes to <path>.new then renames — no partial writes on crash."""
    path = tmp_path / "state.json"
    s = HeartbeatState.fresh(trigger="init")
    s.save(path)
    assert path.exists()
    # .new temp must be cleaned up
    temp_path = path.with_suffix(path.suffix + ".new")
    assert not temp_path.exists()


def test_heartbeat_state_save_overwrites_existing(tmp_path: Path) -> None:
    """Saving over an existing state file succeeds (atomic rename overwrites)."""
    path = tmp_path / "state.json"
    HeartbeatState.fresh(trigger="init").save(path)
    updated = HeartbeatState.fresh(trigger="close")
    updated.save(path)
    loaded = HeartbeatState.load(path)
    assert loaded is not None
    assert loaded.last_trigger == "close"


def test_heartbeat_result_fields() -> None:
    """HeartbeatResult is a frozen dataclass with the expected fields."""
    r = HeartbeatResult(
        trigger="open",
        elapsed_seconds=3600.0,
        memories_decayed=5,
        edges_pruned=2,
        dream_id=None,
        dream_gated_reason="not_due",
        research_deferred=True,
        heartbeat_memory_id=None,
        initialized=False,
    )
    assert r.trigger == "open"
    assert r.initialized is False
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclass raises FrozenInstanceError, not a specific public exception
        r.trigger = "close"  # type: ignore[misc]


def test_heartbeat_engine_construction(tmp_path: Path) -> None:
    """HeartbeatEngine constructs from store + hebbian + provider + paths."""
    from brain.bridge.provider import FakeProvider
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    engine = HeartbeatEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        state_path=tmp_path / "hb_state.json",
        config_path=tmp_path / "hb_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
    )
    assert engine.store is store
    assert engine.hebbian is hebbian
    store.close()
    hebbian.close()


def test_heartbeat_config_load_invalid_json_returns_defaults(tmp_path: Path) -> None:
    """Malformed JSON in config → defaults, no crash."""
    path = tmp_path / "cfg.json"
    path.write_text("not json at all {[}")
    c = HeartbeatConfig.load(path)
    assert c.dream_every_hours == 24.0


def test_heartbeat_config_load_wrong_type_value_returns_defaults(tmp_path: Path) -> None:
    """Hand-edited config with wrong-type values (dream_every_hours={} etc.)
    must NOT crash the CLI with TypeError. Graceful degrade to defaults.
    """
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"dream_every_hours": {}}))
    c = HeartbeatConfig.load(path)
    assert c.dream_every_hours == 24.0


def test_heartbeat_state_load_corrupt_returns_none(tmp_path: Path) -> None:
    """A corrupt state file returns None (→ engine treats as first-ever tick
    and reinitialises) rather than crashing with a traceback.
    """
    path = tmp_path / "state.json"
    path.write_text("not json")
    assert HeartbeatState.load(path) is None


def test_heartbeat_state_load_missing_keys_returns_none(tmp_path: Path) -> None:
    """State file missing required keys returns None — same graceful recovery."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"tick_count": 5}))
    assert HeartbeatState.load(path) is None


def test_heartbeat_state_save_rejects_naive_datetime(tmp_path: Path) -> None:
    """Naive datetime in HeartbeatState.save raises — contract is tz-aware only."""
    path = tmp_path / "state.json"
    naive_dt = datetime(2026, 4, 23, 10, 0, 0)
    s = HeartbeatState(
        last_tick_at=naive_dt,
        last_dream_at=naive_dt,
        last_research_at=naive_dt,
        tick_count=0,
        last_trigger="init",
    )
    with pytest.raises(ValueError, match="tz-aware"):
        s.save(path)
