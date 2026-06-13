"""Tests for the self-model reflection tick on the live supervisor path.

Task 7 (Organ DoD): the producer (self-model reflection) fires through the
supervisor's own persisted-cadence block — NOT a monotonic timer. The tick
composes the whole organ end-to-end: declared (aggregate_state) vs derived
(compute_derived) → gap → state persistence → cadence advance, fail-isolated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.provider import FakeProvider
from brain.bridge.supervisor import _run_self_model_tick


class _CapturingBus:
    """Duck-typed event bus that records every published dict."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, event: dict) -> None:
        self.events.append(event)


def _persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}')
    return p


def _seed_divergent_memories(persona_dir: Path) -> None:
    """An OLD high-intensity joy + several RECENT grief memories.

    max-pool (declared) surfaces joy; recency-mean (derived) leans grief →
    a non-zero gap that declared and derived genuinely disagree on.
    """
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        old_joy = Memory.create_new(
            content="an old bright day",
            memory_type="episodic",
            domain="self",
            emotions={"joy": 9.0},
            importance=8.0,
        )
        object.__setattr__(old_joy, "created_at", datetime(2026, 1, 1, tzinfo=UTC))
        store.create(old_joy)
        for i in range(4):
            m = Memory.create_new(
                content=f"a recent ache #{i}",
                memory_type="episodic",
                domain="self",
                emotions={"grief": 4.0},
                importance=7.0,
            )
            store.create(m)
    finally:
        store.close()


def test_self_model_tick_runs_end_to_end_and_persists_state(tmp_path: Path) -> None:
    """When cadence.is_due is true, the tick composes the whole organ:
    persists self_model_state.json and advances + persists the cadence.
    """
    from brain.self_model import cadence as sm_cadence
    from brain.self_model import state as sm_state

    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir)
    bus = _CapturingBus()

    # Fresh cadence (next_reflection_at=None) → due now.
    assert sm_cadence.is_due(sm_cadence.load(persona_dir), now=datetime.now(UTC))

    _run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=bus)

    # State file persisted, with an active current_gap that derived/declared
    # genuinely diverge on.
    state_file = persona_dir / "self_model_state.json"
    assert state_file.exists(), "tick must persist self_model_state.json"
    state, recovered = sm_state.load_or_recover(persona_dir)
    assert not recovered
    assert state.current_gap is not None, "a divergent seed must yield a gap"
    assert state.current_gap.magnitude > 0.0

    # Cadence advanced (no longer immediately due).
    advanced = sm_cadence.load(persona_dir)
    assert advanced.next_reflection_at is not None, "cadence must advance after a tick"
    assert not sm_cadence.is_due(advanced, now=datetime.now(UTC))


def test_self_model_tick_fail_isolated_on_derived_error(tmp_path: Path) -> None:
    """If compute_derived raises, the tick logs and does NOT propagate —
    the supervisor survives (Organ DoD fail-isolation). The cadence still
    advances (with a backoff) so the tick doesn't busy-loop on the error.
    """
    from unittest.mock import patch

    from brain.self_model import cadence as sm_cadence

    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir)
    bus = _CapturingBus()

    with patch(
        "brain.bridge.supervisor.compute_derived",
        side_effect=RuntimeError("derived blew up"),
    ):
        # Must NOT raise — fail-isolation.
        _run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=bus)

    # Cadence advanced despite the crash (failure backoff) → no busy-loop.
    advanced = sm_cadence.load(persona_dir)
    assert advanced.next_reflection_at is not None
    assert advanced.consecutive_failures >= 1


def test_self_model_tick_increments_gaps_surfaced_when_gap_active(tmp_path: Path) -> None:
    """An active gap bumps the dead-loop observability counter (R-E4)."""
    from brain.self_model.resolve import load_audit

    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir)
    bus = _CapturingBus()

    _run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=bus)

    audit = load_audit(persona_dir)
    assert audit["gaps_surfaced"] >= 1


def test_self_model_tick_not_due_returns_early(tmp_path: Path) -> None:
    """When the persisted cadence is NOT due, the tick is a no-op:
    it does not reflect or write self_model_state.json."""
    from datetime import timedelta

    from brain.self_model import cadence as sm_cadence

    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir)
    bus = _CapturingBus()

    # Force the cadence far into the future → not due.
    future = sm_cadence.SelfModelCadenceState(
        next_reflection_at=datetime.now(UTC) + timedelta(hours=12),
        consecutive_failures=0,
    )
    sm_cadence.save(persona_dir, future)

    _run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=bus)

    # No reflection happened — no state file written.
    assert not (persona_dir / "self_model_state.json").exists()


def test_self_model_tick_fail_isolated_on_articulate_error(tmp_path: Path) -> None:
    """If articulate raises, the tick logs and does NOT propagate."""
    from unittest.mock import patch

    from brain.self_model import cadence as sm_cadence

    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir)
    bus = _CapturingBus()

    with patch(
        "brain.bridge.supervisor.sm_articulate",
        side_effect=RuntimeError("articulate blew up"),
    ):
        _run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=bus)

    advanced = sm_cadence.load(persona_dir)
    assert advanced.next_reflection_at is not None
    assert advanced.consecutive_failures >= 1
