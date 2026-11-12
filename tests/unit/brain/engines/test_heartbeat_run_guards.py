"""The heartbeat's run guards: the cross-process lock (C30) and the in-flight-reply check (C9(b)).

These three tests were ported from the removed per-memory decay batching tests (Engine 6 G4). The behaviour they
check is unchanged by the removal of the decay pass, so each passes on the code before G4 as well; what they now
observe is the tick's state and log, not a count of decayed memories.

Synthetic data only: nothing here touches a real persona's data directory.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest

from brain.bridge import cli_throttle
from brain.bridge.events import EventBus
from brain.bridge.provider import FakeProvider
from brain.bridge.supervisor import _heartbeat_and_felt_time
from brain.engines.heartbeat import HeartbeatEngine, HeartbeatState
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.utils.file_lock import file_lock

_PORTED = pytest.mark.no_base_failure(
    reason="ported from the removed test_heartbeat_decay_batching.py: the guard it checks is unchanged by G4, so the same check passed on the code before G4"
)


@pytest.fixture(autouse=True)
def _reset_cli_throttle():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


@pytest.fixture(autouse=True)
def _safe_tier_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep dream/reflex/research/growth from constructing a real CLI provider against a bare tmp_path persona."""
    monkeypatch.setattr("brain.engines.heartbeat.build_tier_provider", lambda *a, **k: FakeProvider())


def _seed_memory(store: MemoryStore) -> Memory:
    m = Memory.create_new(
        content="seed", memory_type="conversation", domain="us", emotions={"love": 9.0, "tenderness": 5.0}
    )
    store.create(m)
    return m


def _engine(tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix) -> HeartbeatEngine:
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


def _log_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@_PORTED
def test_a_pass_runs_normally_once_the_other_holder_releases_the_lock(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        _seed_memory(store)
        engine.run_tick(trigger="open")
        state = HeartbeatState.load(engine.state_path)
        assert state is not None
        state.last_tick_at = state.last_tick_at - timedelta(hours=48)
        state.save(engine.state_path)

        with file_lock(engine.state_path, blocking=False) as acquired:
            assert acquired
            # A second pass attempted while the first "holds" the lock (this thread stands in for the holder) skips.
            skipped = engine.run_tick(trigger="close")
        assert skipped.skipped_reason == "heartbeat_locked"
        assert HeartbeatState.load(engine.state_path).tick_count == 0, "a skipped pass must not count as a tick"

        # Lock released: a real pass now runs normally and is the tick that counts.
        ran = engine.run_tick(trigger="close")
        assert ran.skipped_reason is None
        assert ran.initialized is False
        assert HeartbeatState.load(engine.state_path).tick_count == 1
    finally:
        store.close()
        hebbian.close()


@_PORTED
def test_the_state_is_read_only_after_the_lock_is_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`HeartbeatState` must be loaded from disk only AFTER `run_tick` holds the lock, never before: reading it first let
    a caller that won the lock only after another pass had finished work on a stale snapshot and overwrite that pass's
    state on save. Proven by a separate non-blocking lock attempt on the same sidecar failing at the moment
    `HeartbeatState.load_with_anomaly` is called."""
    from brain.engines import heartbeat as heartbeat_module

    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    engine = _engine(tmp_path, store, hebbian)
    try:
        engine.run_tick(trigger="open")  # create heartbeat_state.json

        real_load = heartbeat_module.HeartbeatState.load_with_anomaly
        observations: list[bool] = []  # every call, not just the last

        def _spy_load(path):  # noqa: ANN001
            with file_lock(path, blocking=False) as could_also_acquire:
                observations.append(not could_also_acquire)
            return real_load(path)

        monkeypatch.setattr(heartbeat_module.HeartbeatState, "load_with_anomaly", staticmethod(_spy_load))

        engine.run_tick(trigger="close")

        assert observations, "HeartbeatState.load_with_anomaly must be called at least once"
        assert all(observations), "the lock must already be held at EVERY moment the state is read from disk"
    finally:
        store.close()
        hebbian.close()


@_PORTED
def test_no_pass_starts_while_a_reply_is_in_flight(tmp_path: Path) -> None:
    """The supervisor's call site builds its own engine against the canonical persona-dir paths, so this seeds those
    paths directly instead of using `_engine`."""
    persona_dir = tmp_path
    store = MemoryStore(persona_dir / "memories.db")
    _seed_memory(store)
    store.close()

    state_path = persona_dir / "heartbeat_state.json"
    fresh = HeartbeatState.fresh(trigger="open")
    fresh.last_tick_at = fresh.last_tick_at - timedelta(hours=48)
    fresh.save(state_path)
    log_path = persona_dir / "heartbeats.log.jsonl"

    cli_throttle.note_user_message()  # a reply is "in flight"
    event_bus = EventBus()
    result = _heartbeat_and_felt_time(persona_dir, FakeProvider(), event_bus, last_heartbeat_at=time.monotonic())
    assert result is None, "must return early, doing nothing, while a reply is in flight"
    unchanged = HeartbeatState.load(state_path)
    assert unchanged is not None
    assert unchanged.last_tick_at == fresh.last_tick_at
    assert unchanged.tick_count == 0
    assert _log_rows(log_path) == []

    cli_throttle.note_reply_end()
    result2 = _heartbeat_and_felt_time(persona_dir, FakeProvider(), event_bus, last_heartbeat_at=time.monotonic())
    assert result2 is not None
    advanced = HeartbeatState.load(state_path)
    assert advanced is not None
    assert advanced.tick_count == 1 and advanced.last_tick_at > fresh.last_tick_at, "once no reply is in flight, the pass proceeds"
    assert [row.get("initialized") for row in _log_rows(log_path)] == [False]
