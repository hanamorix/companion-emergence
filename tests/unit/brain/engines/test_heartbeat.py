"""Tests for brain.engines.heartbeat — event-driven orchestrator tick."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines.heartbeat import (
    HeartbeatConfig,
    HeartbeatEngine,
    HeartbeatResult,
    HeartbeatState,
)
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find pyproject.toml above {here}")


DEFAULT_REFLEX_ARCS_PATH = _find_repo_root() / "brain" / "engines" / "default_reflex_arcs.json"
DEFAULT_INTERESTS_PATH = _find_repo_root() / "brain" / "engines" / "default_interests.json"


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


# ---- Phase 2a T5: growth_enabled / growth_every_hours / last_growth_at ----


def test_heartbeat_config_has_growth_defaults() -> None:
    """HeartbeatConfig has growth_enabled=True + growth_every_hours=168.0 (weekly)."""
    c = HeartbeatConfig()
    assert c.growth_enabled is True
    assert c.growth_every_hours == 168.0


def test_heartbeat_config_round_trip_preserves_growth(tmp_path: Path) -> None:
    """save() then load() preserves growth fields."""
    original = HeartbeatConfig(growth_enabled=False, growth_every_hours=24.0)
    path = tmp_path / "cfg.json"
    original.save(path)
    restored = HeartbeatConfig.load(path)
    assert restored.growth_enabled is False
    assert restored.growth_every_hours == 24.0


def test_heartbeat_config_back_compat_missing_growth_fields(tmp_path: Path) -> None:
    """Old config files without growth_* fields load with defaults."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"dream_every_hours": 12.0}), encoding="utf-8")
    c = HeartbeatConfig.load(path)
    assert c.growth_enabled is True
    assert c.growth_every_hours == 168.0


def test_heartbeat_state_includes_last_growth_at() -> None:
    """HeartbeatState.fresh() initialises last_growth_at = now."""
    s = HeartbeatState.fresh(trigger="open")
    assert s.last_growth_at is not None
    assert s.last_growth_at.tzinfo is not None  # tz-aware


def test_heartbeat_state_round_trips_last_growth_at(tmp_path: Path) -> None:
    s = HeartbeatState.fresh(trigger="open")
    path = tmp_path / "state.json"
    s.save(path)
    restored = HeartbeatState.load(path)
    assert restored is not None
    assert restored.last_growth_at == s.last_growth_at


def test_heartbeat_state_back_compat_missing_last_growth_at(tmp_path: Path) -> None:
    """Old state files without last_growth_at load with last_growth_at=now (delays
    first growth tick by growth_every_hours, which is the safe back-compat default)."""
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "last_tick_at": "2026-04-20T10:00:00Z",
                "last_dream_at": "2026-04-20T10:00:00Z",
                "last_research_at": "2026-04-20T10:00:00Z",
                "tick_count": 5,
                "last_trigger": "open",
            }
        ),
        encoding="utf-8",
    )
    s = HeartbeatState.load(path)
    assert s is not None
    assert s.last_growth_at is not None  # backfilled to now-ish on load
    assert s.last_growth_at.tzinfo is not None


def test_heartbeat_result_default_growth_emotions_added_is_zero() -> None:
    """HeartbeatResult.growth_emotions_added defaults to 0."""
    r = HeartbeatResult(
        trigger="manual",
        elapsed_seconds=0.0,
        memories_decayed=0,
        edges_pruned=0,
        dream_id=None,
        dream_gated_reason=None,
        research_deferred=False,
        heartbeat_memory_id=None,
        initialized=False,
    )
    assert r.growth_emotions_added == 0


# ---- PR-C: user_preferences.json merges over heartbeat_config.json ----


def test_user_preferences_overrides_dream_every_hours(tmp_path: Path) -> None:
    """When user_preferences.json sets dream_every_hours, it wins over heartbeat_config.json."""
    cfg_path = tmp_path / "heartbeat_config.json"
    cfg_path.write_text(json.dumps({"dream_every_hours": 24.0}))
    (tmp_path / "user_preferences.json").write_text(json.dumps({"dream_every_hours": 6.0}))
    c = HeartbeatConfig.load(cfg_path)
    assert c.dream_every_hours == 6.0


def test_user_preferences_missing_falls_back_to_heartbeat_config(tmp_path: Path) -> None:
    """No user_preferences.json → heartbeat_config.json's value stands (back-compat)."""
    cfg_path = tmp_path / "heartbeat_config.json"
    cfg_path.write_text(json.dumps({"dream_every_hours": 8.0}))
    c = HeartbeatConfig.load(cfg_path)
    assert c.dream_every_hours == 8.0


def test_user_preferences_present_but_omits_field(tmp_path: Path) -> None:
    """user_preferences.json without dream_every_hours doesn't shadow heartbeat_config.json.

    Critical for back-compat: a future user_preferences.json with new fields
    must not silently reset dream_every_hours to the default.
    """
    cfg_path = tmp_path / "heartbeat_config.json"
    cfg_path.write_text(json.dumps({"dream_every_hours": 8.0}))
    (tmp_path / "user_preferences.json").write_text(json.dumps({"some_future_field": "x"}))
    c = HeartbeatConfig.load(cfg_path)
    assert c.dream_every_hours == 8.0


def test_user_preferences_only_no_heartbeat_config(tmp_path: Path) -> None:
    """user_preferences.json drives dream_every_hours when heartbeat_config.json is absent."""
    (tmp_path / "user_preferences.json").write_text(json.dumps({"dream_every_hours": 12.0}))
    c = HeartbeatConfig.load(tmp_path / "heartbeat_config.json")  # path doesn't exist
    assert c.dream_every_hours == 12.0


def test_user_preferences_corrupt_does_not_break_load(tmp_path: Path) -> None:
    """Corrupt user_preferences.json doesn't crash HeartbeatConfig.load()."""
    cfg_path = tmp_path / "heartbeat_config.json"
    cfg_path.write_text(json.dumps({"dream_every_hours": 8.0}))
    (tmp_path / "user_preferences.json").write_text("not json")
    c = HeartbeatConfig.load(cfg_path)
    assert c.dream_every_hours == 8.0  # falls back to heartbeat_config.json


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
        last_growth_at=when,
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
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
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
        last_growth_at=naive_dt,
        tick_count=0,
        last_trigger="init",
    )
    with pytest.raises(ValueError, match="tz-aware"):
        s.save(path)


@pytest.fixture
def live_engine(tmp_path: Path) -> HeartbeatEngine:
    """Engine with in-memory store/hebbian and tmp log/state paths."""
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    return HeartbeatEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        state_path=tmp_path / "hb_state.json",
        config_path=tmp_path / "hb_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
    )


def _seed_conversation(store: MemoryStore, content: str, importance: float = 5.0) -> Memory:
    m = Memory.create_new(
        content=content,
        memory_type="conversation",
        domain="us",
        emotions={"love": 9.0, "tenderness": 5.0},
    )
    m.importance = importance
    store.create(m)
    return m


def test_first_tick_initializes_and_defers_work(live_engine: HeartbeatEngine) -> None:
    """First-ever tick returns initialized=True, writes state, does no work."""
    _seed_conversation(live_engine.store, "seed")

    result = live_engine.run_tick(trigger="open")
    assert result.initialized is True
    assert result.memories_decayed == 0
    assert result.edges_pruned == 0
    assert result.dream_id is None
    assert result.dream_gated_reason == "first_tick"
    assert live_engine.state_path.exists()

    second = live_engine.run_tick(trigger="close")
    assert second.initialized is False


def test_second_tick_applies_decay(live_engine: HeartbeatEngine) -> None:
    """Second tick after simulated elapsed time decays emotions."""
    m = _seed_conversation(live_engine.store, "seed")
    live_engine.run_tick(trigger="open")  # init

    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_tick_at = state.last_tick_at - timedelta(hours=48)
    state.last_dream_at = state.last_dream_at - timedelta(hours=48)
    state.save(live_engine.state_path)

    result = live_engine.run_tick(trigger="close")
    assert result.initialized is False
    assert result.elapsed_seconds >= 48 * 3600 - 10

    reloaded = live_engine.store.get(m.id)
    assert reloaded is not None
    # love has decay_half_life_days=None (identity-level) so it never decays.
    # tenderness has a 7-day half-life; after 48 h it should drop from 5.0.
    assert reloaded.emotions.get("tenderness", 0.0) < 5.0


def test_dream_gate_respects_config(live_engine: HeartbeatEngine) -> None:
    """dream_every_hours config gates dream firing."""
    _seed_conversation(live_engine.store, "seed", importance=9.0)
    cfg = HeartbeatConfig(dream_every_hours=0.001)
    cfg.save(live_engine.config_path)

    live_engine.run_tick(trigger="open")  # init, no dream
    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_dream_at = state.last_dream_at - timedelta(hours=1)
    state.save(live_engine.state_path)

    result = live_engine.run_tick(trigger="close")
    assert result.dream_id is not None
    assert result.dream_gated_reason is None


def test_dream_gated_when_not_due(live_engine: HeartbeatEngine) -> None:
    """Dream does not fire when last_dream_at + dream_every_hours > now."""
    _seed_conversation(live_engine.store, "seed", importance=9.0)
    live_engine.run_tick(trigger="open")  # init, no dream

    result = live_engine.run_tick(trigger="close")
    assert result.dream_id is None
    assert result.dream_gated_reason == "not_due"


def test_protected_memories_skipped_by_decay(live_engine: HeartbeatEngine) -> None:
    """protected=True memories don't get their emotions decayed."""
    m = _seed_conversation(live_engine.store, "protected")
    live_engine.store.update(m.id, protected=True)
    live_engine.run_tick(trigger="open")

    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_tick_at = state.last_tick_at - timedelta(hours=72)
    state.save(live_engine.state_path)

    live_engine.run_tick(trigger="close")

    reloaded = live_engine.store.get(m.id)
    assert reloaded is not None
    assert reloaded.emotions.get("love", 0.0) == 9.0


def test_hebbian_gc_prunes_weak_edges(live_engine: HeartbeatEngine) -> None:
    """decay_all + gc removes weak edges."""
    live_engine.hebbian.strengthen("a", "b", delta=0.005)
    live_engine.hebbian.strengthen("c", "d", delta=0.5)
    live_engine.run_tick(trigger="open")
    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_tick_at = state.last_tick_at - timedelta(hours=48)
    state.save(live_engine.state_path)

    result = live_engine.run_tick(trigger="close")
    assert result.edges_pruned >= 1
    assert live_engine.hebbian.weight("a", "b") == 0.0
    assert live_engine.hebbian.weight("c", "d") > 0.0


def test_research_always_deferred(live_engine: HeartbeatEngine) -> None:
    """research_deferred=True on every non-init tick (engine not built yet)."""
    _seed_conversation(live_engine.store, "seed")
    live_engine.run_tick(trigger="open")  # init
    result = live_engine.run_tick(trigger="close")
    assert result.research_deferred is True


def test_dry_run_does_not_mutate_store_or_state(live_engine: HeartbeatEngine) -> None:
    """--dry-run: no state file written, no memory decay, no dream."""
    m = _seed_conversation(live_engine.store, "seed", importance=9.0)
    live_engine.run_tick(trigger="manual", dry_run=True)

    assert not live_engine.state_path.exists()
    reloaded = live_engine.store.get(m.id)
    assert reloaded is not None
    assert reloaded.emotions.get("love", 0.0) == 9.0


def test_heartbeats_log_has_init_entry(live_engine: HeartbeatEngine) -> None:
    """First tick writes a JSONL entry marked initialized=true."""
    live_engine.run_tick(trigger="open")
    text = live_engine.heartbeat_log_path.read_text().strip()
    line = json.loads(text.splitlines()[-1])
    assert line["initialized"] is True
    assert line["trigger"] == "open"


def test_heartbeats_log_has_tick_entry(live_engine: HeartbeatEngine) -> None:
    """Second tick writes a JSONL entry with elapsed/memories_decayed/etc."""
    _seed_conversation(live_engine.store, "seed")
    live_engine.run_tick(trigger="open")
    live_engine.run_tick(trigger="close")

    lines = live_engine.heartbeat_log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    tick = json.loads(lines[-1])
    assert tick["trigger"] == "close"
    assert tick.get("initialized") is False
    assert "elapsed_seconds" in tick
    assert "tick_count" in tick


def test_heartbeat_memory_emitted_when_always_mode(live_engine: HeartbeatEngine) -> None:
    """emit_memory='always' → every non-init tick writes a heartbeat memory."""
    _seed_conversation(live_engine.store, "seed")
    HeartbeatConfig(dream_every_hours=999, emit_memory="always").save(live_engine.config_path)
    live_engine.run_tick(trigger="open")  # init, no memory
    live_engine.run_tick(trigger="close")

    hb_memories = live_engine.store.list_by_type("heartbeat")
    assert len(hb_memories) == 1
    assert hb_memories[0].content.startswith("HEARTBEAT:")


def test_heartbeat_memory_skipped_when_never_mode(live_engine: HeartbeatEngine) -> None:
    """emit_memory='never' → no heartbeat memory regardless of tick outcome."""
    _seed_conversation(live_engine.store, "seed", importance=9.0)
    HeartbeatConfig(dream_every_hours=0.001, emit_memory="never").save(live_engine.config_path)
    live_engine.run_tick(trigger="open")
    live_engine.run_tick(trigger="close")

    hb_memories = live_engine.store.list_by_type("heartbeat")
    assert len(hb_memories) == 0


def test_dry_run_dream_eligible_returns_would_fire_reason(
    live_engine: HeartbeatEngine,
) -> None:
    """dry_run=True when dream gate is open reports 'would_fire_but_dry_run'."""
    _seed_conversation(live_engine.store, "seed", importance=9.0)
    HeartbeatConfig(dream_every_hours=0.001).save(live_engine.config_path)
    live_engine.run_tick(trigger="open")  # init
    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_dream_at = state.last_dream_at - timedelta(hours=1)
    state.save(live_engine.state_path)

    result = live_engine.run_tick(trigger="close", dry_run=True)
    assert result.dream_gated_reason == "would_fire_but_dry_run"
    assert result.dream_id is None


def test_heartbeat_memory_metadata_links_to_dream_id(
    live_engine: HeartbeatEngine,
) -> None:
    """The heartbeat memory's metadata.dream_id links back to the dream that
    prompted it — the chain that consumers will traverse to find which
    associative event triggered the meta-reflection.
    """
    _seed_conversation(live_engine.store, "seed", importance=9.0)
    HeartbeatConfig(dream_every_hours=0.001, emit_memory="always").save(live_engine.config_path)
    live_engine.run_tick(trigger="open")  # init
    state = HeartbeatState.load(live_engine.state_path)
    assert state is not None
    state.last_dream_at = state.last_dream_at - timedelta(hours=1)
    state.save(live_engine.state_path)

    result = live_engine.run_tick(trigger="close")
    assert result.dream_id is not None
    assert result.heartbeat_memory_id is not None

    hb_mem = live_engine.store.get(result.heartbeat_memory_id)
    assert hb_mem is not None
    assert hb_mem.metadata.get("dream_id") == result.dream_id


def test_dream_gate_no_seed_reason_does_not_poison(live_engine: HeartbeatEngine) -> None:
    """If the dream gate is open but NoSeedAvailable fires (no conversation
    memories), dream_gated_reason='no_seed_available' and last_dream_at is
    NOT advanced — so the next tick tries again instead of being silently
    suppressed for 24h.
    """
    # NO conversation memory seeded; only a non-conversation memory that
    # can't be selected as a dream seed.
    other = Memory.create_new(
        content="dream: an old dream",
        memory_type="dream",
        domain="us",
        emotions={"love": 8.0},
    )
    live_engine.store.create(other)

    HeartbeatConfig(dream_every_hours=0.001).save(live_engine.config_path)
    live_engine.run_tick(trigger="open")  # init
    state_before = HeartbeatState.load(live_engine.state_path)
    assert state_before is not None
    state_before.last_dream_at = state_before.last_dream_at - timedelta(hours=1)
    state_before.save(live_engine.state_path)
    dream_at_before = state_before.last_dream_at

    result = live_engine.run_tick(trigger="close")
    assert result.dream_id is None
    assert result.dream_gated_reason == "no_seed_available"

    # last_dream_at NOT advanced by failed dream attempt
    state_after = HeartbeatState.load(live_engine.state_path)
    assert state_after is not None
    # last_tick_at advanced (every tick does), but last_dream_at stayed put
    assert state_after.last_dream_at == dream_at_before


def test_heartbeat_runs_reflex_when_enabled(tmp_path: Path) -> None:
    """Heartbeat fires reflex arc when enabled and trigger met."""
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    # Write a single easy-to-trigger arc
    arcs_path = tmp_path / "reflex_arcs.json"
    arc = {
        "name": "test_arc",
        "description": "d",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "a",
        "output_memory_type": "reflex_journal",
        "prompt_template": "Hi {persona_name}.",
    }
    arcs_path.write_text(json.dumps({"version": 1, "arcs": [arc]}), encoding="utf-8")

    # Enable reflex in heartbeat config
    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(reflex_enabled=True).save(config_path)

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="s",
                memory_type="conversation",
                domain="us",
                emotions={"love": 8.0},
            )
        )
        # Pre-seed state so run_tick skips the first-tick-defer path and
        # actually runs reflex + decay this iteration.
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store,
            hebbian=hm,
            provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=arcs_path,
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.reflex_fired == ("test_arc",)
    finally:
        store.close()
        hm.close()


def test_heartbeat_skips_reflex_when_disabled(tmp_path: Path) -> None:
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    arcs_path = tmp_path / "reflex_arcs.json"
    arc = {
        "name": "test_arc",
        "description": "d",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "a",
        "output_memory_type": "reflex_journal",
        "prompt_template": "Hi.",
    }
    arcs_path.write_text(json.dumps({"version": 1, "arcs": [arc]}), encoding="utf-8")

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(reflex_enabled=False).save(config_path)

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="s",
                memory_type="conversation",
                domain="us",
                emotions={"love": 8.0},
            )
        )
        # Pre-seed state so run_tick skips the first-tick-defer path and
        # actually runs reflex + decay this iteration.
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store,
            hebbian=hm,
            provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=arcs_path,
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.reflex_fired == ()
    finally:
        store.close()
        hm.close()


def test_heartbeat_isolates_reflex_llm_failure(tmp_path: Path, caplog) -> None:
    """Reflex LLM failure is isolated from the tick — decay/state still run."""
    import logging

    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    arcs_path = tmp_path / "reflex_arcs.json"
    arc = {
        "name": "test_arc",
        "description": "d",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "a",
        "output_memory_type": "reflex_journal",
        "prompt_template": "Hi.",
    }
    arcs_path.write_text(json.dumps({"version": 1, "arcs": [arc]}), encoding="utf-8")

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(reflex_enabled=True).save(config_path)

    class FailingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            raise RuntimeError("simulated LLM failure")

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="s",
                memory_type="conversation",
                domain="us",
                emotions={"love": 8.0},
            )
        )
        # Pre-seed state so run_tick skips the first-tick-defer path and
        # actually runs reflex + decay this iteration.
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store,
            hebbian=hm,
            provider=FailingProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=arcs_path,
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        with caplog.at_level(logging.WARNING, logger="brain.engines.heartbeat"):
            result = engine.run_tick(trigger="manual", dry_run=False)
        # Tick completed successfully despite reflex failure
        assert result.reflex_fired == ()
        # Warning was logged
        assert any("reflex tick raised" in r.message for r in caplog.records)
        # Decay still ran
        assert result.memories_decayed >= 0
    finally:
        store.close()
        hm.close()


def test_heartbeat_runs_research_when_enabled(tmp_path: Path) -> None:
    """Heartbeat fires research when interest is eligible + trigger signal present."""
    import json

    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.search.base import NoopWebSearcher

    interests_path = tmp_path / "interests.json"
    interests_path.write_text(
        json.dumps(
            {
                "version": 1,
                "interests": [
                    {
                        "id": "i1",
                        "topic": "marine bio",
                        "pull_score": 8.0,
                        "scope": "either",
                        "related_keywords": ["marine", "bio"],
                        "notes": "",
                        "first_seen": "2026-01-01T00:00:00Z",
                        "last_fed": "2026-01-01T00:00:00Z",
                        "last_researched_at": None,
                        "feed_count": 1,
                        "source_types": ["manual"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(
        reflex_enabled=False,
        research_enabled=True,
        research_days_since_human_min=1.5,
    ).save(config_path)

    from datetime import UTC, datetime, timedelta

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        old_mem = Memory.create_new(
            content="Hana and I talked long ago",
            memory_type="conversation",
            domain="us",
            emotions={},
        )
        store.create(old_mem)
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(days=3)).isoformat(), old_mem.id),
        )
        store._conn.commit()  # type: ignore[attr-defined]

        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store,
            hebbian=hm,
            provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=tmp_path / "reflex_arcs.json",
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            searcher=NoopWebSearcher(),
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.research_fired == "marine bio"
    finally:
        store.close()
        hm.close()


def test_heartbeat_reflex_wins_tie_over_research(tmp_path: Path) -> None:
    """When both reflex and research eligible, reflex fires, research skipped."""
    import json

    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.search.base import NoopWebSearcher

    arcs_path = tmp_path / "reflex_arcs.json"
    arcs_path.write_text(
        json.dumps(
            {
                "version": 1,
                "arcs": [
                    {
                        "name": "test_arc",
                        "description": "d",
                        "trigger": {"love": 5},
                        "days_since_human_min": 0,
                        "cooldown_hours": 1.0,
                        "action": "a",
                        "output_memory_type": "reflex_journal",
                        "prompt_template": "Hi.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    interests_path = tmp_path / "interests.json"
    interests_path.write_text(
        json.dumps(
            {
                "version": 1,
                "interests": [
                    {
                        "id": "i1",
                        "topic": "marine bio",
                        "pull_score": 8.0,
                        "scope": "either",
                        "related_keywords": ["marine"],
                        "notes": "",
                        "first_seen": "2026-01-01T00:00:00Z",
                        "last_fed": "2026-01-01T00:00:00Z",
                        "last_researched_at": None,
                        "feed_count": 1,
                        "source_types": ["manual"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(
        reflex_enabled=True,
        research_enabled=True,
        research_emotion_threshold=5.0,
    ).save(config_path)

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="s",
                memory_type="conversation",
                domain="us",
                emotions={"love": 8.0},
            )
        )
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store,
            hebbian=hm,
            provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=arcs_path,
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            searcher=NoopWebSearcher(),
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.reflex_fired == ("test_arc",)
        assert result.research_fired is None
        assert result.research_gated_reason == "reflex_won_tie"
    finally:
        store.close()
        hm.close()


def test_heartbeat_interest_bump_hook(tmp_path: Path) -> None:
    """Conversation memory with matching keyword bumps interest pull_score."""
    import json

    from brain.bridge.provider import FakeProvider
    from brain.engines._interests import InterestSet
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.search.base import NoopWebSearcher

    interests_path = tmp_path / "interests.json"
    interests_path.write_text(
        json.dumps(
            {
                "version": 1,
                "interests": [
                    {
                        "id": "i1",
                        "topic": "lispector",
                        "pull_score": 5.0,
                        "scope": "either",
                        "related_keywords": ["lispector", "clarice"],
                        "notes": "",
                        "first_seen": "2026-01-01T00:00:00Z",
                        "last_fed": "2026-01-01T00:00:00Z",
                        "last_researched_at": None,
                        "feed_count": 3,
                        "source_types": ["manual"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(
        reflex_enabled=False,
        research_enabled=False,
        interest_bump_per_match=0.5,
    ).save(config_path)

    from datetime import UTC, datetime, timedelta

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        # Backdate state.last_tick_at to 1h ago so the memory seeded "now"
        # qualifies as "conversation since last tick" for the ingestion hook.
        prior_state = HeartbeatState.fresh("manual")
        prior_state.last_tick_at = datetime.now(UTC) - timedelta(hours=1)
        prior_state.save(tmp_path / "heartbeat_state.json")

        store.create(
            Memory.create_new(
                content="Hana sent me a passage about clarice lispector today",
                memory_type="fact",
                domain="brain",
                emotions={},
                tags=["auto_ingest", "conversation", "fact"],
                metadata={"source_summary": "conversation:sess-bump"},
            )
        )

        engine = HeartbeatEngine(
            store=store,
            hebbian=hm,
            provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=tmp_path / "reflex_arcs.json",
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            searcher=NoopWebSearcher(),
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.interests_bumped == 1
        reloaded = InterestSet.load(interests_path, default_path=DEFAULT_INTERESTS_PATH)
        assert reloaded.interests[0].pull_score == 5.5
        assert reloaded.interests[0].feed_count == 4
    finally:
        store.close()
        hm.close()


def test_heartbeat_isolates_research_failure(tmp_path: Path, caplog) -> None:
    """Research LLM failure is isolated — tick completes."""
    import json
    import logging

    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.search.base import NoopWebSearcher

    interests_path = tmp_path / "interests.json"
    interests_path.write_text(
        json.dumps(
            {
                "version": 1,
                "interests": [
                    {
                        "id": "i1",
                        "topic": "t",
                        "pull_score": 8.0,
                        "scope": "either",
                        "related_keywords": ["t"],
                        "notes": "",
                        "first_seen": "2026-01-01T00:00:00Z",
                        "last_fed": "2026-01-01T00:00:00Z",
                        "last_researched_at": None,
                        "feed_count": 1,
                        "source_types": ["manual"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(
        reflex_enabled=False, research_enabled=True, research_emotion_threshold=5.0
    ).save(config_path)

    class FailingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            raise RuntimeError("simulated LLM failure")

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="s",
                memory_type="conversation",
                domain="us",
                emotions={"love": 8.0},
            )
        )
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store,
            hebbian=hm,
            provider=FailingProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=tmp_path / "reflex_arcs.json",
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            searcher=NoopWebSearcher(),
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        with caplog.at_level(logging.WARNING, logger="brain.engines.heartbeat"):
            result = engine.run_tick(trigger="manual", dry_run=False)
        assert result.research_fired is None
        assert result.research_gated_reason == "research_raised"
        assert any("research tick raised" in r.message for r in caplog.records)
    finally:
        store.close()
        hm.close()


def test_heartbeat_memory_uses_persona_name(tmp_path: Path) -> None:
    """_emit_heartbeat_memory must render persona name into system prompt,
    not hardcode 'Nell'.
    """
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    captured = {}

    class CapturingProvider(FakeProvider):
        def generate(self, prompt, *, system=None):
            captured["system"] = system
            return "HEARTBEAT: tended"

    config_path = tmp_path / "heartbeat_config.json"
    HeartbeatConfig(
        reflex_enabled=False,
        research_enabled=False,
        emit_memory="always",
    ).save(config_path)

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="s",
                memory_type="conversation",
                domain="us",
                emotions={"love": 5.0},
            )
        )
        prior = HeartbeatState.fresh("manual")
        prior.last_tick_at = datetime.now(UTC) - timedelta(hours=1)
        prior.save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store,
            hebbian=hm,
            provider=CapturingProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=config_path,
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            persona_name="Iris",
            persona_system_prompt="You are Iris.",
        )
        engine.run_tick(trigger="manual", dry_run=False)

        assert captured.get("system") is not None
        assert "Iris" in captured["system"]
        assert "Nell" not in captured["system"]
    finally:
        store.close()
        hm.close()


def test_heartbeat_engine_empty_persona_raises() -> None:
    """HeartbeatEngine must reject empty persona_name / persona_system_prompt
    to force callers to be explicit rather than silently get defaults.
    """
    from pathlib import Path as _Path

    import pytest

    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        with pytest.raises(ValueError, match="persona_name"):
            HeartbeatEngine(
                store=store,
                hebbian=hm,
                provider=FakeProvider(),
                state_path=_Path("/tmp/a"),
                config_path=_Path("/tmp/b"),
                dream_log_path=_Path("/tmp/c"),
                heartbeat_log_path=_Path("/tmp/d"),
                # persona_name omitted → should raise
            )
    finally:
        store.close()
        hm.close()


def test_heartbeat_config_save_is_atomic(tmp_path: Path) -> None:
    """HeartbeatConfig.save must use .new + os.replace so a crash mid-write
    leaves either the old valid file or the new valid file — never a
    partial write. Corruption would silently revert to defaults on reload,
    losing user-tuned values.
    """
    path = tmp_path / "cfg.json"
    HeartbeatConfig(dream_every_hours=12.0, reflex_enabled=False).save(path)
    assert path.exists()
    # .new temp must not linger after save
    assert not path.with_suffix(path.suffix + ".new").exists()
    # Reloads cleanly with tuned values preserved
    loaded = HeartbeatConfig.load(path)
    assert loaded.dream_every_hours == 12.0
    assert loaded.reflex_enabled is False


# ---- Phase 2a T6: _try_run_growth + heartbeat tick wiring ----


def test_heartbeat_run_tick_calls_growth_after_research(tmp_path: Path) -> None:
    """Growth tick fires when due; heartbeat reports the count."""
    from unittest.mock import patch

    from brain.engines.heartbeat import HeartbeatEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    # Seed a vocabulary so the scheduler has somewhere to write.
    (persona_dir / "emotion_vocabulary.json").write_text(
        json.dumps({"version": 1, "emotions": []}), encoding="utf-8"
    )

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="test",
            persona_system_prompt="You are test.",
        )

        # First tick — initializes state, defers all work.
        engine.run_tick(trigger="open")

        # Force last_growth_at older than 168h so growth is due.
        from brain.engines.heartbeat import HeartbeatState

        state = HeartbeatState.load(persona_dir / "heartbeat_state.json")
        assert state is not None
        old = datetime.now(UTC) - timedelta(hours=200)
        state.last_growth_at = old
        state.last_tick_at = old
        state.save(persona_dir / "heartbeat_state.json")

        # Inject a fake crystallizer that returns one proposal.
        from brain.growth.proposal import EmotionProposal

        proposal = EmotionProposal(
            name="lingering",
            description="x",
            decay_half_life_days=None,
            evidence_memory_ids=(),
            score=0.5,
            relational_context=None,
        )
        with patch(
            "brain.growth.scheduler.crystallize_vocabulary",
            return_value=[proposal],
        ):
            result = engine.run_tick(trigger="manual")

        assert result.growth_emotions_added == 1

        # Audit log entry has growth sub-object
        log_lines = (persona_dir / "heartbeats.log.jsonl").read_text(encoding="utf-8").splitlines()
        last_entry = json.loads(log_lines[-1])
        assert "growth" in last_entry
        assert last_entry["growth"]["enabled"] is True
        assert last_entry["growth"]["ran"] is True
        assert last_entry["growth"]["emotions_added"] == 1
    finally:
        store.close()
        hebbian.close()


def test_heartbeat_growth_disabled_skips_growth_tick(tmp_path: Path) -> None:
    """growth_enabled=False short-circuits before scheduler runs."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    cfg_path = persona_dir / "heartbeat_config.json"
    cfg_path.write_text(json.dumps({"growth_enabled": False}), encoding="utf-8")

    from brain.engines.heartbeat import HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=cfg_path,
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="test",
            persona_system_prompt="You are test.",
        )
        engine.run_tick(trigger="open")  # init
        # Push state back so growth would otherwise be due.
        s = HeartbeatState.load(persona_dir / "heartbeat_state.json")
        assert s is not None
        s.last_growth_at = datetime.now(UTC) - timedelta(hours=200)
        s.last_tick_at = datetime.now(UTC) - timedelta(hours=200)
        s.save(persona_dir / "heartbeat_state.json")

        result = engine.run_tick(trigger="manual")
        assert result.growth_emotions_added == 0

        # Hardening: audit log must record growth.enabled=False so a forensic
        # reader can distinguish "disabled" from "not due" from "no persona dir".
        audit_lines = (
            (persona_dir / "heartbeats.log.jsonl").read_text(encoding="utf-8").splitlines()
        )
        last_audit = json.loads(audit_lines[-1])
        assert last_audit["growth"]["enabled"] is False
        assert last_audit["growth"]["ran"] is False
        assert last_audit["growth"]["emotions_added"] == 0
    finally:
        store.close()
        hebbian.close()


def test_heartbeat_growth_not_due_returns_zero(tmp_path: Path) -> None:
    """Within growth_every_hours window, growth tick returns zero."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    from brain.engines.heartbeat import HeartbeatEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="test",
            persona_system_prompt="You are test.",
        )
        engine.run_tick(trigger="open")  # init
        result = engine.run_tick(trigger="manual")  # not 168h later
        assert result.growth_emotions_added == 0
    finally:
        store.close()
        hebbian.close()


def test_heartbeat_growth_fault_isolated(tmp_path: Path) -> None:
    """If the growth tick raises, heartbeat continues — count is 0."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    from unittest.mock import patch

    from brain.engines.heartbeat import HeartbeatEngine, HeartbeatState
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="test",
            persona_system_prompt="You are test.",
        )
        engine.run_tick(trigger="open")
        s = HeartbeatState.load(persona_dir / "heartbeat_state.json")
        assert s is not None
        s.last_growth_at = datetime.now(UTC) - timedelta(hours=200)
        s.last_tick_at = datetime.now(UTC) - timedelta(hours=200)
        s.save(persona_dir / "heartbeat_state.json")

        with patch(
            "brain.growth.scheduler.run_growth_tick",
            side_effect=RuntimeError("simulated crash"),
        ):
            result = engine.run_tick(trigger="manual")
        assert result.growth_emotions_added == 0
        # Heartbeat tick still completed
        assert not result.initialized
    finally:
        store.close()
        hebbian.close()


# ---- Task 9: attempt_heal wiring for HeartbeatConfig + HeartbeatState ----


def test_heartbeat_config_load_corrupt_file_quarantines_and_resets(tmp_path: Path) -> None:
    """Corrupt heartbeat_config.json → defaults returned + quarantine file present."""
    path = tmp_path / "heartbeat_config.json"
    path.write_text("{corrupt json{{", encoding="utf-8")

    cfg, anomaly = HeartbeatConfig._load_internal_with_anomaly(path)

    assert cfg.dream_every_hours == 24.0
    assert cfg.emit_memory == "conditional"
    assert anomaly is not None
    assert anomaly.kind == "json_parse_error"
    corrupt_files = list(tmp_path.glob("heartbeat_config.json.corrupt-*"))
    assert len(corrupt_files) == 1


def test_heartbeat_config_load_corrupt_file_restores_from_bak(tmp_path: Path) -> None:
    """Valid .bak1 + corrupt live heartbeat_config.json → .bak1 content returned."""
    path = tmp_path / "heartbeat_config.json"
    bak1 = tmp_path / "heartbeat_config.json.bak1"
    bak1.write_text(
        json.dumps({"dream_every_hours": 6.0, "emit_memory": "always"}), encoding="utf-8"
    )
    path.write_text("{corrupt", encoding="utf-8")

    cfg, anomaly = HeartbeatConfig._load_internal_with_anomaly(path)

    assert cfg.dream_every_hours == 6.0
    assert cfg.emit_memory == "always"
    assert anomaly is not None
    assert "bak1" in anomaly.action


def test_heartbeat_state_load_corrupt_file_quarantines_and_resets(tmp_path: Path) -> None:
    """Corrupt heartbeat_state.json → None returned + quarantine file present."""
    path = tmp_path / "heartbeat_state.json"
    path.write_text("{corrupt json{{", encoding="utf-8")

    state, anomaly = HeartbeatState.load_with_anomaly(path)

    assert state is None
    assert anomaly is not None
    assert anomaly.kind == "json_parse_error"
    corrupt_files = list(tmp_path.glob("heartbeat_state.json.corrupt-*"))
    assert len(corrupt_files) == 1


def test_heartbeat_state_load_corrupt_file_restores_from_bak(tmp_path: Path) -> None:
    """Valid .bak1 + corrupt live heartbeat_state.json → .bak1 content returned."""
    path = tmp_path / "heartbeat_state.json"
    bak1 = tmp_path / "heartbeat_state.json.bak1"
    valid_state = {
        "last_tick_at": "2026-04-25T10:00:00Z",
        "last_dream_at": "2026-04-25T10:00:00Z",
        "last_research_at": "2026-04-25T10:00:00Z",
        "last_growth_at": "2026-04-25T10:00:00Z",
        "tick_count": 3,
        "last_trigger": "open",
    }
    bak1.write_text(json.dumps(valid_state), encoding="utf-8")
    path.write_text("{corrupt", encoding="utf-8")

    state, anomaly = HeartbeatState.load_with_anomaly(path)

    assert state is not None
    assert state.tick_count == 3
    assert state.last_trigger == "open"
    assert anomaly is not None
    assert "bak1" in anomaly.action


# ---- Task 12: anomaly aggregation in heartbeat tick ----


def test_heartbeat_audit_log_anomalies_field_always_present(live_engine: HeartbeatEngine) -> None:
    """Every audit entry has 'anomalies' key (empty list on clean tick)."""
    _seed_conversation(live_engine.store, "seed")
    live_engine.run_tick(trigger="open")  # init tick
    live_engine.run_tick(trigger="close")  # real tick

    lines = live_engine.heartbeat_log_path.read_text().strip().splitlines()
    for line in lines:
        entry = json.loads(line)
        assert "anomalies" in entry, f"Missing 'anomalies' key in audit entry: {entry}"
        assert isinstance(entry["anomalies"], list)
    # Clean tick: anomalies should be empty
    tick_entry = json.loads(lines[-1])
    assert tick_entry["anomalies"] == []
    assert "pending_alarms_count" in tick_entry
    assert tick_entry["pending_alarms_count"] == 0


def test_heartbeat_corrupt_state_file_self_heals_and_records(tmp_path: Path) -> None:
    """Corrupting heartbeat_state.json triggers heal + anomaly entry in audit log + result."""
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=tmp_path / "heartbeat_config.json",
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        # Corrupt the state file — will trigger load_with_anomaly to return an anomaly
        state_path = tmp_path / "heartbeat_state.json"
        # Write a valid state first so the file exists
        HeartbeatState.fresh("manual").save(state_path)
        # Now corrupt it
        state_path.write_text("{corrupt json{{{", encoding="utf-8")

        result = engine.run_tick(trigger="manual")
        # Corrupt state → engine treats as first-tick (initialized=True)
        # and should record the anomaly
        assert result.initialized is True
        # The anomaly should be in the result
        assert len(result.anomalies) >= 1
        anomaly_kinds = [a.kind for a in result.anomalies]
        assert "json_parse_error" in anomaly_kinds

        # Audit log entry also has the anomaly
        log_lines = engine.heartbeat_log_path.read_text().strip().splitlines()
        entry = json.loads(log_lines[-1])
        assert "anomalies" in entry
        assert len(entry["anomalies"]) >= 1
        assert entry["anomalies"][0]["kind"] == "json_parse_error"
    finally:
        store.close()
        hebbian.close()


def test_heartbeat_multi_anomaly_tick_triggers_walk(tmp_path: Path) -> None:
    """>=2 anomalies trigger cross-file walk; merged into audit entry."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        # Corrupt BOTH config and state to generate 2 anomalies
        (persona_dir / "heartbeat_config.json").write_text("{bad json{", encoding="utf-8")
        (persona_dir / "heartbeat_state.json").write_text("{bad state{", encoding="utf-8")

        from unittest.mock import patch

        walk_calls = []

        def fake_walk(d: Path) -> list:
            walk_calls.append(d)
            return []

        with patch("brain.engines.heartbeat.walk_persona", side_effect=fake_walk):
            result = engine.run_tick(trigger="manual")

        # walk_persona must have been called because we had >=2 anomalies
        assert len(walk_calls) >= 1
        # Result should have >=2 anomalies
        assert len(result.anomalies) >= 2
    finally:
        store.close()
        hebbian.close()


def test_heartbeat_alarm_increments_pending_alarms_count(tmp_path: Path) -> None:
    """An identity-file reset_to_default → pending_alarms_count >= 1."""
    from datetime import UTC, datetime
    from unittest.mock import patch

    from brain.health.anomaly import AlarmEntry

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )

        # Patch compute_pending_alarms to return one alarm (simulates an
        # identity-file reset_to_default becoming a pending alarm entry).
        alarm = AlarmEntry(
            file="emotion_vocabulary.json",
            kind="json_parse_error",
            first_seen_at=datetime.now(UTC),
            occurrences_in_window=1,
        )
        with patch("brain.engines.heartbeat.compute_pending_alarms", return_value=[alarm]):
            result = engine.run_tick(trigger="manual")

        assert result.pending_alarms_count >= 1
        # Audit log also has pending_alarms_count >= 1
        log_lines = engine.heartbeat_log_path.read_text().strip().splitlines()
        entry = json.loads(log_lines[-1])
        assert entry["pending_alarms_count"] >= 1
    finally:
        store.close()
        hebbian.close()


# ---- F3: growth-tick-internal anomalies surface in heartbeat audit log ----


def test_heartbeat_growth_anomaly_surfaces_in_audit_log(tmp_path: Path) -> None:
    """When growth tick reads a corrupt vocabulary file, the anomaly appears
    in the heartbeat audit log + HeartbeatResult.anomalies — not just logged
    as a warning inside growth.

    Followup F3 from the brain-health module.
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    # Seed an empty interests file so growth has a persona dir to work with.
    (persona_dir / "interests.json").write_text(
        json.dumps({"version": 1, "interests": []}), encoding="utf-8"
    )

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="test",
            persona_system_prompt="You are test.",
        )

        # First tick initializes state — work deferred (first-tick semantics).
        engine.run_tick(trigger="open")

        # Force last_growth_at older than growth_every_hours so growth fires.
        from brain.engines.heartbeat import HeartbeatState

        s = HeartbeatState.load(persona_dir / "heartbeat_state.json")
        assert s is not None
        s.last_growth_at = datetime.now(UTC) - timedelta(hours=200)
        s.last_tick_at = datetime.now(UTC) - timedelta(hours=200)
        s.save(persona_dir / "heartbeat_state.json")

        # Corrupt vocab file (no .bak → reset_to_default fires inside growth).
        (persona_dir / "emotion_vocabulary.json").write_text("{not json", encoding="utf-8")

        # Second tick runs growth. _read_current_vocabulary_names detects the
        # corruption, heals it, and the anomaly is appended to tick_anomalies
        # via the F3 wiring.
        result = engine.run_tick(trigger="manual")

        # The anomaly from inside growth surfaces in the result + audit log.
        vocab_anomalies = [a for a in result.anomalies if a.file == "emotion_vocabulary.json"]
        assert len(vocab_anomalies) >= 1
        assert vocab_anomalies[0].action == "reset_to_default"

        # Audit log also has it.
        log_lines = (persona_dir / "heartbeats.log.jsonl").read_text().strip().splitlines()
        last_entry = json.loads(log_lines[-1])
        files_in_audit = {a["file"] for a in last_entry["anomalies"]}
        assert "emotion_vocabulary.json" in files_in_audit
    finally:
        store.close()
        hebbian.close()


# ---------------------------------------------------------------------------
# SP-2 daemon_state integration tests
# ---------------------------------------------------------------------------


def _make_engine_with_persona_dir(tmp_path: Path) -> tuple[HeartbeatEngine, Path]:
    """Build a HeartbeatEngine wired to tmp_path as persona_dir.

    Persona_dir is passed via interests_path.parent so the engine knows where
    to write daemon_state.json.
    """
    interests_path = tmp_path / "interests.json"
    interests_path.write_text(json.dumps({"version": 1, "interests": []}), encoding="utf-8")
    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    engine = HeartbeatEngine(
        store=store,
        hebbian=hm,
        provider=FakeProvider(),
        state_path=tmp_path / "heartbeat_state.json",
        config_path=tmp_path / "heartbeat_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
        interests_path=interests_path,
        research_log_path=tmp_path / "research_log.json",
        default_interests_path=DEFAULT_INTERESTS_PATH,
        persona_name="Nell",
        persona_system_prompt="You are Nell.",
    )
    return engine, tmp_path


def test_daemon_state_heartbeat_entry_written_after_tick(tmp_path: Path) -> None:
    """After any non-init tick, daemon_state.json has a last_heartbeat entry."""
    from brain.engines.daemon_state import load_daemon_state

    engine, persona_dir = _make_engine_with_persona_dir(tmp_path)
    try:
        store = engine.store
        store.create(
            Memory.create_new(
                content="seed memory",
                memory_type="conversation",
                domain="us",
                emotions={"love": 8.0},
            )
        )
        engine.run_tick(trigger="open")  # init tick
        engine.run_tick(trigger="close")  # real tick

        state, anomaly = load_daemon_state(persona_dir)
        assert anomaly is None
        assert state.last_heartbeat is not None
        assert state.last_heartbeat.theme == "heartbeat tick"
    finally:
        engine.store.close()
        engine.hebbian.close()


def test_daemon_state_dream_entry_written_when_dream_fires(tmp_path: Path) -> None:
    """When a dream fires, daemon_state.json has last_dream populated."""
    from brain.engines.daemon_state import load_daemon_state

    engine, persona_dir = _make_engine_with_persona_dir(tmp_path)
    try:
        store = engine.store
        store.create(
            Memory.create_new(
                content="a vivid conversation memory about the world",
                memory_type="conversation",
                domain="us",
                emotions={"love": 9.0, "emergence": 8.0},
                importance=9.0,
            )
        )
        HeartbeatConfig(dream_every_hours=0.001, emit_memory="never").save(engine.config_path)
        engine.run_tick(trigger="open")  # init
        state_obj = HeartbeatState.load(engine.state_path)
        assert state_obj is not None
        state_obj.last_dream_at = state_obj.last_dream_at - timedelta(hours=1)
        state_obj.save(engine.state_path)

        result = engine.run_tick(trigger="close")
        if result.dream_id is None:
            pytest.skip("dream didn't fire (no seed) — skip daemon_state dream check")

        daemon_state, anomaly = load_daemon_state(persona_dir)
        assert anomaly is None
        assert daemon_state.last_dream is not None
        assert daemon_state.last_dream.dominant_emotion != ""
    finally:
        engine.store.close()
        engine.hebbian.close()


def test_daemon_state_reflex_entry_written_when_reflex_fires(tmp_path: Path) -> None:
    """When a reflex arc fires, daemon_state.json has last_reflex with trigger set."""
    from brain.engines.daemon_state import load_daemon_state

    arcs_path = tmp_path / "reflex_arcs.json"
    arc = {
        "name": "love_arc",
        "description": "d",
        "trigger": {"love": 5},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "a",
        "output_memory_type": "reflex_journal",
        "prompt_template": "Hi {persona_name}.",
    }
    arcs_path.write_text(json.dumps({"version": 1, "arcs": [arc]}), encoding="utf-8")

    interests_path = tmp_path / "interests.json"
    interests_path.write_text(json.dumps({"version": 1, "interests": []}), encoding="utf-8")
    HeartbeatConfig(reflex_enabled=True).save(tmp_path / "heartbeat_config.json")

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="seed",
                memory_type="conversation",
                domain="us",
                emotions={"love": 8.0},
            )
        )
        # Pre-seed state to skip init-defer path.
        HeartbeatState.fresh("manual").save(tmp_path / "heartbeat_state.json")

        engine = HeartbeatEngine(
            store=store,
            hebbian=hm,
            provider=FakeProvider(),
            state_path=tmp_path / "heartbeat_state.json",
            config_path=tmp_path / "heartbeat_config.json",
            dream_log_path=tmp_path / "dreams.log.jsonl",
            heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
            reflex_arcs_path=arcs_path,
            reflex_log_path=tmp_path / "reflex_log.json",
            reflex_default_arcs_path=DEFAULT_REFLEX_ARCS_PATH,
            interests_path=interests_path,
            research_log_path=tmp_path / "research_log.json",
            default_interests_path=DEFAULT_INTERESTS_PATH,
            persona_name="Nell",
            persona_system_prompt="You are Nell.",
        )
        result = engine.run_tick(trigger="manual")
        assert result.reflex_fired == ("love_arc",)

        daemon_state, anomaly = load_daemon_state(tmp_path)
        assert anomaly is None
        assert daemon_state.last_reflex is not None
        assert daemon_state.last_reflex.trigger == "love_arc"
    finally:
        store.close()
        hm.close()


def test_daemon_state_dry_run_does_not_write_file(tmp_path: Path) -> None:
    """dry_run=True prevents daemon_state.json from being written."""
    engine, persona_dir = _make_engine_with_persona_dir(tmp_path)
    try:
        store = engine.store
        store.create(
            Memory.create_new(
                content="seed",
                memory_type="conversation",
                domain="us",
                emotions={"love": 7.0},
            )
        )
        engine.run_tick(trigger="manual", dry_run=True)
        assert not (persona_dir / "daemon_state.json").exists()
    finally:
        engine.store.close()
        engine.hebbian.close()


def test_daemon_state_writer_error_is_fault_isolated(tmp_path: Path, caplog) -> None:
    """A daemon_state write failure must not abort the heartbeat tick."""
    import logging

    import brain.engines.heartbeat as hb_module

    engine, persona_dir = _make_engine_with_persona_dir(tmp_path)
    try:
        store = engine.store
        store.create(
            Memory.create_new(
                content="seed",
                memory_type="conversation",
                domain="us",
                emotions={"love": 7.0},
            )
        )
        engine.run_tick(trigger="open")  # init

        # Patch update_daemon_state in the heartbeat module's namespace (where
        # it was imported via `from brain.engines.daemon_state import …`) so the
        # fault-isolation wrappers actually see the raised exception.
        with patch.object(hb_module, "update_daemon_state", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING, logger="brain.engines.heartbeat"):
                result = engine.run_tick(trigger="close")

        # Tick must have completed successfully
        assert result.initialized is False
        # Warning was emitted (at least one daemon_state write attempted)
        assert any("daemon_state write" in r.message for r in caplog.records)
    finally:
        engine.store.close()
        engine.hebbian.close()
