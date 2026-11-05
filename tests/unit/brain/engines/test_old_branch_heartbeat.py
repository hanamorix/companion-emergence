"""Pins for the heartbeat's own emotion arithmetic, before the readers move to the engine.

The heartbeat summary and the spike resonance sum each emotion over every active memory (the other readers take
a maximum), clamp the result to 10, and fall back to calm 3 with nothing to sum. Deleted with the old branch when
the reader switch-over lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines.daemon_state import load_daemon_state
from brain.engines.heartbeat import HeartbeatEngine
from brain.health.json_io import write_json
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore

_SUMMARY_REASON = (
    "pins the old heartbeat summary branch (per-name sum, clamp to 10, calm 3 default); "
    "passes on the current code by design and is deleted with the branch at phase C"
)
_RESONANCE_REASON = (
    "pins the old heartbeat resonance branch (summed peak, clamped to 10); "
    "passes on the current code by design and is deleted with the branch at phase C"
)


def _engine(tmp_path: Path) -> HeartbeatEngine:
    interests_path = tmp_path / "interests.json"
    write_json(interests_path, {"version": 1, "interests": []})
    return HeartbeatEngine(
        store=MemoryStore(tmp_path / "memories.db"),
        hebbian=HebbianMatrix(":memory:"),
        provider=FakeProvider(),
        state_path=tmp_path / "heartbeat_state.json",
        config_path=tmp_path / "heartbeat_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
        interests_path=interests_path,
        research_log_path=tmp_path / "research_log.json",
        persona_name="Test",
        persona_system_prompt="You are Test.",
    )


def _close(engine: HeartbeatEngine) -> None:
    engine.store.close()
    engine.hebbian.close()


def _add(engine: HeartbeatEngine, emotions: dict[str, float]) -> None:
    engine.store.create(
        Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=emotions)
    )


def _summary(engine: HeartbeatEngine, tmp_path: Path) -> tuple[str, int]:
    engine._write_daemon_state_for_heartbeat(
        tmp_path, elapsed_seconds=900.0, memories_decayed=0, edges_pruned=0, dream_id=None
    )
    state, anomaly = load_daemon_state(tmp_path)
    assert anomaly is None
    entry = state.last_heartbeat
    assert entry is not None
    return entry.dominant_emotion, entry.intensity


@pytest.mark.no_base_failure(reason=_SUMMARY_REASON)
def test_status_summary_sums_values_across_memories(tmp_path: Path):
    engine = _engine(tmp_path)
    try:
        _add(engine, {"love": 4.0})
        _add(engine, {"love": 3.0, "joy": 1.0})

        assert _summary(engine, tmp_path) == ("love", 7)
    finally:
        _close(engine)


@pytest.mark.no_base_failure(reason=_SUMMARY_REASON)
def test_status_summary_clamps_the_sum_to_ten(tmp_path: Path):
    engine = _engine(tmp_path)
    try:
        _add(engine, {"love": 8.0})
        _add(engine, {"love": 7.0})

        assert _summary(engine, tmp_path) == ("love", 10)
    finally:
        _close(engine)


@pytest.mark.no_base_failure(reason=_SUMMARY_REASON)
def test_status_summary_falls_back_to_calm_three(tmp_path: Path):
    engine = _engine(tmp_path)
    try:
        _add(engine, {})

        assert _summary(engine, tmp_path) == ("calm", 3)
    finally:
        _close(engine)


@pytest.mark.no_base_failure(reason=_RESONANCE_REASON)
def test_resonance_is_the_summed_peak_clamped_to_ten(tmp_path: Path):
    engine = _engine(tmp_path)
    try:
        _add(engine, {"love": 8.0})
        _add(engine, {"love": 7.0})

        assert engine._compute_current_resonance(None) == (10.0, {"love": 15.0})
    finally:
        _close(engine)


@pytest.mark.no_base_failure(reason=_RESONANCE_REASON)
def test_resonance_of_an_empty_store_is_zero(tmp_path: Path):
    engine = _engine(tmp_path)
    try:
        assert engine._compute_current_resonance(None) == (0.0, {})
    finally:
        _close(engine)
