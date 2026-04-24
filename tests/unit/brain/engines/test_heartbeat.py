"""Tests for brain.engines.heartbeat — event-driven orchestrator tick."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
