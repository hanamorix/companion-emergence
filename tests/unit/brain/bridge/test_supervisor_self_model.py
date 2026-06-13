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


# ── Organ-DoD live-path resolution test (Fix C1) ──────────────────────────────


def _capture_feed_events():
    """Set a capturing feed publisher; return (events_list, teardown_fn)."""
    from brain.bridge import events as ev

    captured: list[dict] = []
    ev.set_publisher(captured.append)
    return captured, lambda: ev.set_publisher(None)


def _clear_emotion_memories(persona_dir: Path) -> None:
    """Deactivate every emotion-bearing memory so declared/derived re-match.

    After this the self-model gap collapses to magnitude 0 (natural
    reconvergence — no reconcile tool call).
    """
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        store._conn.execute("UPDATE memories SET active = 0")  # noqa: SLF001
        store._conn.commit()
    finally:
        store.close()


def test_sustained_gap_resolved_through_live_tick_emits_soul_candidate(
    tmp_path: Path,
) -> None:
    """Organ DoD (C1): a gap that genuinely sustains across REAL reflection ticks
    and then resolves emits a soul candidate + a feed event — on BOTH paths.

    Drives the REAL ``_self_model_reflect`` body, persisting state between calls
    so sustained_ticks accumulate through the real path (not hand-constructed).
    """
    from datetime import timedelta

    from brain.bridge.supervisor import _self_model_reflect
    from brain.ingest.soul_queue import list_soul_candidates
    from brain.self_model import state as sm_state
    from brain.self_model.reconcile import reconcile_self_read
    from brain.self_model.resolve import _SUSTAINED_TICKS

    provider = FakeProvider()
    base = datetime(2026, 6, 1, tzinfo=UTC)

    def _tick(persona_dir: Path, now: datetime) -> None:
        _self_model_reflect(
            persona_dir, provider=provider, event_bus=_CapturingBus(), now=now
        )

    # ── PATH A — reconcile ───────────────────────────────────────────────────
    pa = _persona_dir(tmp_path)
    _seed_divergent_memories(pa)

    # Sustain the gap over enough real ticks to cross _SUSTAINED_TICKS.
    for i in range(_SUSTAINED_TICKS + 1):
        _tick(pa, base + timedelta(hours=6 * i))

    sustained = sm_state.load_or_recover(pa)[0].current_gap
    assert sustained is not None and sustained.status == "open"
    assert sustained.sustained_ticks >= _SUSTAINED_TICKS, (
        "the gap must genuinely sustain through the real ticks"
    )

    # She reconciles — the tool flips the persisted current_gap to acknowledged.
    reconcile_self_read(persona_dir=pa, action="dismiss", channel="joy")

    events_a, teardown_a = _capture_feed_events()
    try:
        # One more REAL tick: prior (acknowledged + sustained) must resolve.
        _tick(pa, base + timedelta(hours=6 * (_SUSTAINED_TICKS + 1)))
    finally:
        teardown_a()

    cands_a = list_soul_candidates(pa)
    assert len(cands_a) == 1, "reconcile path must queue a soul candidate via the live tick"
    resolved_evt_a = [e for e in events_a if e.get("type") == "self_model_gap_resolved"]
    assert len(resolved_evt_a) == 1
    assert resolved_evt_a[0].get("resolution_path") == "reconcile"

    # ── PATH B — natural reconvergence ───────────────────────────────────────
    pb_root = tmp_path / "b"
    pb_root.mkdir()
    pb = _persona_dir(pb_root)
    _seed_divergent_memories(pb)

    for i in range(_SUSTAINED_TICKS + 1):
        _tick(pb, base + timedelta(hours=6 * i))

    sustained_b = sm_state.load_or_recover(pb)[0].current_gap
    assert sustained_b is not None and sustained_b.sustained_ticks >= _SUSTAINED_TICKS

    # Memories re-match (declared == derived → magnitude collapses to 0). No tool.
    _clear_emotion_memories(pb)

    events_b, teardown_b = _capture_feed_events()
    try:
        _tick(pb, base + timedelta(hours=6 * (_SUSTAINED_TICKS + 1)))
    finally:
        teardown_b()

    cands_b = list_soul_candidates(pb)
    assert len(cands_b) == 1, "natural path must queue a soul candidate via the live tick"
    resolved_evt_b = [e for e in events_b if e.get("type") == "self_model_gap_resolved"]
    assert len(resolved_evt_b) == 1
    assert resolved_evt_b[0].get("resolution_path") == "natural"


# ── R-B2 live cooldown test (Fix C3) ──────────────────────────────────────────


def test_reconcile_cooldown_suppresses_channel_on_next_live_tick(tmp_path: Path) -> None:
    """C3 (R-B2): after a reconcile sets a cooldown on a channel, the NEXT live
    reflection tick drops that channel from the surfaced gap and carries the
    (non-expired) cooldown forward so it survives the recompute.
    """
    from datetime import timedelta

    from brain.bridge.supervisor import _self_model_reflect
    from brain.self_model import state as sm_state
    from brain.self_model.reconcile import is_channel_in_cooldown, reconcile_self_read

    provider = FakeProvider()
    base = datetime(2026, 6, 1, tzinfo=UTC)

    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir)

    # Tick once → a gap surfaces on joy (and grief).
    _self_model_reflect(
        persona_dir, provider=provider, event_bus=_CapturingBus(), now=base
    )
    gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert gap is not None and "joy" in gap.per_channel

    # She reconciles joy → cooldown set on joy.
    reconcile_self_read(persona_dir=persona_dir, action="accept", channel="joy", delta=0.1)
    gap_after = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert is_channel_in_cooldown(gap_after, "joy", now=base) is True

    # Next live tick, still inside the cooldown window: joy must be suppressed
    # from the surfaced gap, and the cooldown carried forward.
    _self_model_reflect(
        persona_dir, provider=provider, event_bus=_CapturingBus(), now=base + timedelta(hours=6)
    )
    new_gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert new_gap is not None
    assert "joy" not in new_gap.per_channel, "joy is in cooldown → must not re-surface"
    assert is_channel_in_cooldown(new_gap, "joy", now=base + timedelta(hours=6)) is True, (
        "non-expired cooldown must survive the recompute / status flip"
    )
