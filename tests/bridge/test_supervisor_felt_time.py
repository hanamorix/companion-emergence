"""Integration test: supervisor wires FeltTime.tick() into the heartbeat cadence.

Verifies that after two heartbeat cadence cycles, FeltTime.tick was called
at least twice — i.e. the felt-time integration is present and fault-isolated.

Entry point under test: brain.bridge.supervisor.run_folded.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_heartbeat_and_felt_time_passes_real_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_heartbeat_and_felt_time must pass real chat_n and reflex_n to _run_felt_time_tick."""
    import brain.bridge.supervisor as sup_mod
    from brain.engines.heartbeat import HeartbeatResult
    from brain.ingest.buffer import ingest_turn

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')

    # Seed two recent turns in an active session buffer
    import time as _time
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    session_id = "sess_testabcd"
    for i in range(2):
        ingest_turn(
            persona_dir,
            {
                "session_id": session_id,
                "speaker": "user",
                "text": f"turn {i}",
                "ts": (now - timedelta(seconds=i + 1)).isoformat(),
            },
        )

    # Heartbeat result carries two reflex firings
    fake_result = HeartbeatResult(
        trigger="background",
        elapsed_seconds=0.1,
        memories_decayed=0,
        edges_pruned=0,
        dream_id=None,
        dream_gated_reason=None,
        research_deferred=False,
        heartbeat_memory_id=None,
        initialized=True,
        reflex_fired=("arc_x", "arc_y"),
    )
    monkeypatch.setattr(
        sup_mod, "_run_heartbeat_tick", lambda *a, **k: fake_result
    )

    # Capture what _run_felt_time_tick receives
    felt_time_calls: list[dict] = []
    real_felt_time_tick = sup_mod._run_felt_time_tick

    def _spy_felt_time_tick(persona_dir, *, wall_clock_s_since_last, heartbeats_since_last,
                            chat_turns_since_last, reflex_firings_since_last):
        felt_time_calls.append({
            "chat_turns": chat_turns_since_last,
            "reflex_firings": reflex_firings_since_last,
        })
        return real_felt_time_tick(
            persona_dir,
            wall_clock_s_since_last=wall_clock_s_since_last,
            heartbeats_since_last=heartbeats_since_last,
            chat_turns_since_last=chat_turns_since_last,
            reflex_firings_since_last=reflex_firings_since_last,
        )

    monkeypatch.setattr(sup_mod, "_run_felt_time_tick", _spy_felt_time_tick)

    provider = MagicMock()
    event_bus = MagicMock()
    last_heartbeat_at = _time.monotonic() - 60.0  # 60s ago

    sup_mod._heartbeat_and_felt_time(persona_dir, provider, event_bus, last_heartbeat_at)

    assert len(felt_time_calls) == 1, "Expected exactly one _run_felt_time_tick call"
    call = felt_time_calls[0]
    assert call["chat_turns"] == 2, (
        f"Expected 2 chat turns (the seeded buffer), got {call['chat_turns']}"
    )
    assert call["reflex_firings"] == 2, (
        f"Expected 2 reflex firings (from fake HeartbeatResult), got {call['reflex_firings']}"
    )


def test_heartbeat_tick_returns_result_with_reflex_fired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_run_heartbeat_tick must return the HeartbeatResult from engine.run_tick."""
    from brain.bridge.supervisor import _run_heartbeat_tick
    from brain.engines.heartbeat import HeartbeatEngine, HeartbeatResult

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')

    fake_result = HeartbeatResult(
        trigger="background",
        elapsed_seconds=0.1,
        memories_decayed=0,
        edges_pruned=0,
        dream_id=None,
        dream_gated_reason=None,
        research_deferred=False,
        heartbeat_memory_id=None,
        initialized=True,
        reflex_fired=("arc_a", "arc_b"),
    )

    monkeypatch.setattr(
        HeartbeatEngine,
        "run_tick",
        lambda self, **kwargs: fake_result,
    )

    provider = MagicMock()
    event_bus = MagicMock()

    result = _run_heartbeat_tick(persona_dir, provider, event_bus)

    assert result is not None, "_run_heartbeat_tick must return the HeartbeatResult"
    assert len(result.reflex_fired) == 2, (
        f"Expected 2 reflex_fired entries, got {len(result.reflex_fired)}"
    )


def test_supervisor_invokes_felt_time_tick_on_heartbeat_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_folded fires FeltTime.tick at least twice across two heartbeat cycles."""
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')

    # Count tick() calls via a fake FeltTime.
    tick_calls: list[tuple] = []

    class _FakeFeltTime:
        def __init__(self, *, persona_dir: Path) -> None:
            tick_calls.append(("init", persona_dir))

        def tick(self, ctx: object) -> None:
            tick_calls.append(("tick", ctx))

        def get_state(self) -> None:
            return None

    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", _FakeFeltTime)

    stop_event = threading.Event()
    provider = MagicMock()
    event_bus = MagicMock()

    # After two tick() calls stop the loop.
    def _stop_after_two_ticks(*_args: object, **_kwargs: object) -> None:
        tick_count = sum(1 for c in tick_calls if c[0] == "tick")
        if tick_count >= 2:
            stop_event.set()

    monkeypatch.setattr(
        "brain.bridge.supervisor._run_heartbeat_tick",
        lambda *a, **k: _stop_after_two_ticks(),
    )

    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=provider,
        event_bus=event_bus,
        tick_interval_s=0.05,
        heartbeat_interval_s=0.05,
        soul_review_interval_s=None,
        finalize_interval_s=None,
        log_rotation_interval_s=None,
        initiate_review_interval_s=None,
        voice_reflection_interval_s=None,
    )

    tick_count = sum(1 for c in tick_calls if c[0] == "tick")
    assert tick_count >= 2, f"Expected ≥2 felt-time tick() calls, got {tick_count}"


def test_supervisor_felt_time_tick_fault_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FeltTime.tick() raise must not crash the supervisor loop.

    The heartbeat tick still increments last_heartbeat_at and the loop
    continues running — fault-isolation is non-negotiable per spec.
    """
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')

    heartbeat_calls: list[int] = []

    # FeltTime that always raises on tick().
    class _ExplodingFeltTime:
        def __init__(self, *, persona_dir: Path) -> None:
            pass

        def tick(self, ctx: object) -> None:
            raise RuntimeError("felt-time tick exploded")

        def get_state(self) -> None:
            return None

    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", _ExplodingFeltTime)

    stop_event = threading.Event()
    provider = MagicMock()
    event_bus = MagicMock()

    # Stop after two heartbeat ticks (loop survived the exploding tick).
    def _counting_heartbeat(*_args: object, **_kwargs: object) -> None:
        heartbeat_calls.append(1)
        if len(heartbeat_calls) >= 2:
            stop_event.set()

    monkeypatch.setattr(
        "brain.bridge.supervisor._run_heartbeat_tick",
        _counting_heartbeat,
    )

    # Should not raise — fault isolation prevents propagation.
    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=provider,
        event_bus=event_bus,
        tick_interval_s=0.05,
        heartbeat_interval_s=0.05,
        soul_review_interval_s=None,
        finalize_interval_s=None,
        log_rotation_interval_s=None,
        initiate_review_interval_s=None,
        voice_reflection_interval_s=None,
    )

    assert len(heartbeat_calls) >= 2, (
        f"Loop should have run ≥2 heartbeat ticks despite exploding felt-time; "
        f"got {len(heartbeat_calls)}"
    )


def _open_arc(arc_id: str, *, lived_age_at_open: float, emotion: float):
    from brain.narrative_memory.arc import Arc

    return Arc(
        id=arc_id,
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dreams.log.jsonl:1",
        seed_memory_ids=("m1",),
        title="an open thread",
        opened_at_iso="2026-05-20T10:00:00+00:00",
        lived_age_at_open=lived_age_at_open,
        last_extended_at_iso="2026-05-20T10:00:00+00:00",
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(),
        max_member_emotion_normalised=emotion,
    )


def test_derive_intensity_drivers_sets_narrative_weight(tmp_path: Path) -> None:
    from brain.bridge.supervisor import _derive_intensity_drivers
    from brain.felt_time.state import FeltTimeState, persist
    from brain.narrative_memory.state import ArcsState, save_state

    # lived age 200h; arc opened at 0h => 200 open lived-hours (> horizon),
    # emotion 0.8 => narrative_weight 0.8.
    persist(FeltTimeState(lived_age_hours=200.0), tmp_path)
    save_state(tmp_path, ArcsState(open={"a1": _open_arc("a1", lived_age_at_open=0.0, emotion=0.8)}))

    drivers = _derive_intensity_drivers(tmp_path, chat_turns_in_tick=0, wall_clock_s_in_tick=3600.0)
    assert drivers.narrative_weight == pytest.approx(0.8, abs=1e-9)


def test_derive_intensity_drivers_no_open_arcs_zero_weight(tmp_path: Path) -> None:
    from brain.bridge.supervisor import _derive_intensity_drivers
    from brain.felt_time.state import FeltTimeState, persist
    from brain.narrative_memory.state import ArcsState, save_state

    persist(FeltTimeState(lived_age_hours=200.0), tmp_path)
    save_state(tmp_path, ArcsState(open={}))

    drivers = _derive_intensity_drivers(tmp_path, chat_turns_in_tick=0, wall_clock_s_in_tick=3600.0)
    assert drivers.narrative_weight == 0.0


def test_run_felt_time_tick_appends_chat_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_felt_time_tick must write a chat_turns.log.jsonl row after each tick."""
    from brain.bridge.supervisor import _run_felt_time_tick
    from brain.felt_time.chat_log import CHAT_TURNS_LOG_FILENAME

    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())

    _run_felt_time_tick(
        tmp_path,
        wall_clock_s_since_last=900.0,
        heartbeats_since_last=1,
        chat_turns_since_last=4,
        reflex_firings_since_last=0,
    )

    log_path = tmp_path / CHAT_TURNS_LOG_FILENAME
    assert log_path.exists(), "chat_turns.log.jsonl should be written after tick"
    import json as _json
    row = _json.loads(log_path.read_text().strip())
    assert row["turns"] == 4


def test_derive_intensity_drivers_chat_activity_uses_rolling_baseline(tmp_path: Path) -> None:
    """When chat_turns.log.jsonl has enough data, rolling mean is used as baseline.

    Uses values that produce different results under fixed vs rolling baseline:
      wall_clock_s = 3600s → fixed baseline = max(0.1, 6.0*1.0) = 6.0
      5 ticks at 10 turns → rolling mean = 10.0
      5 turns in tick → fixed gives min(1.0, 5/6)≈0.833; rolling gives min(1.0, 5/10)=0.5
    """
    from datetime import UTC, datetime, timedelta

    from brain.bridge.supervisor import _derive_intensity_drivers
    from brain.felt_time.chat_log import append_chat_tick

    now = datetime.now(UTC)
    for i in range(5):
        append_chat_tick(tmp_path, ts=now - timedelta(hours=i + 1), turns=10)

    # Rolling mean = 10 → 5 turns gives chat_activity = 0.5
    drivers = _derive_intensity_drivers(tmp_path, chat_turns_in_tick=5, wall_clock_s_in_tick=3600.0)
    assert drivers.chat_activity == pytest.approx(0.5)


def test_derive_intensity_drivers_chat_activity_fallback_without_log(tmp_path: Path) -> None:
    """Without chat_turns.log.jsonl the fixed 6-turns/h baseline is used."""
    from brain.bridge.supervisor import _derive_intensity_drivers

    # 900s tick, fixed baseline = 6 * (900/3600) = 1.5 turns/tick
    # 3 turns → chat_activity = min(1.0, 3/1.5) = 1.0
    drivers = _derive_intensity_drivers(tmp_path, chat_turns_in_tick=3, wall_clock_s_in_tick=900.0)
    assert drivers.chat_activity == pytest.approx(1.0)


def _make_emotion_memory(store, *, channel: str, value: float, days_ago: float):
    from datetime import UTC, datetime, timedelta

    from brain.memory.store import Memory

    m = Memory.create_new(
        content="x", memory_type="episodic", domain="chat", emotions={channel: value}
    )
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=days_ago))
    store.create(m)


def test_derive_intensity_drivers_emotional_intensity_high_deviation(tmp_path: Path) -> None:
    """A memory with sorrow far above the 15-memory baseline saturates the driver."""
    from brain.bridge.supervisor import _derive_intensity_drivers
    from brain.felt_time.state import FeltTimeState, persist
    from brain.memory.store import MemoryStore

    persist(FeltTimeState(lived_age_hours=200.0), tmp_path)
    store = MemoryStore(tmp_path / "memories.db")
    # 15 memories at sorrow=1.0 establish baseline mean ≈ 1.0
    for i in range(15):
        _make_emotion_memory(store, channel="sorrow", value=1.0, days_ago=20 + i)
    # 1 recent memory at sorrow=9.0 drives aggregate max high
    _make_emotion_memory(store, channel="sorrow", value=9.0, days_ago=0.1)
    store.close()

    drivers = _derive_intensity_drivers(tmp_path, chat_turns_in_tick=0, wall_clock_s_in_tick=900.0)
    # deviation >> 3σ → clipped to 1.0
    assert drivers.emotional_intensity == pytest.approx(1.0)


def test_derive_intensity_drivers_emotional_intensity_zero_when_no_memories(tmp_path: Path) -> None:
    """No memories.db → emotional_intensity falls back to 0.0."""
    from brain.bridge.supervisor import _derive_intensity_drivers

    drivers = _derive_intensity_drivers(tmp_path, chat_turns_in_tick=0, wall_clock_s_in_tick=900.0)
    assert drivers.emotional_intensity == 0.0


def test_derive_intensity_drivers_emotional_intensity_zero_below_sample_threshold(
    tmp_path: Path,
) -> None:
    """Fewer than 10 samples per channel → no baseline → emotional_intensity stays 0.0."""
    from brain.bridge.supervisor import _derive_intensity_drivers
    from brain.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    for i in range(5):  # only 5 — below the 10-sample minimum
        _make_emotion_memory(store, channel="sorrow", value=9.0, days_ago=20 + i)
    store.close()

    drivers = _derive_intensity_drivers(tmp_path, chat_turns_in_tick=0, wall_clock_s_in_tick=900.0)
    assert drivers.emotional_intensity == 0.0


def test_derive_intensity_drivers_emotional_intensity_zero_when_at_baseline(tmp_path: Path) -> None:
    """When all memories share the same emotion value, sigma=0 → intensity stays 0."""
    from brain.bridge.supervisor import _derive_intensity_drivers
    from brain.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    # All memories at exactly sorrow=3.0 — mean=3.0, sigma=0, deviation undefined
    for i in range(15):
        _make_emotion_memory(store, channel="sorrow", value=3.0, days_ago=20 + i)
    store.close()

    drivers = _derive_intensity_drivers(tmp_path, chat_turns_in_tick=0, wall_clock_s_in_tick=900.0)
    assert drivers.emotional_intensity == 0.0
