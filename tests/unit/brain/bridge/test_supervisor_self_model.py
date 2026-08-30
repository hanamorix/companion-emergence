"""Tests for the self-model reflection tick on the live supervisor path.

Task 7 (Organ DoD): the producer (self-model reflection) fires through the
supervisor's own persisted-cadence block — NOT a monotonic timer. The tick
composes the whole organ end-to-end: short (compute_derived) vs long
(compute_baseline) → gap → state persistence → cadence advance, fail-isolated.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
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


def _seed_divergent_memories(persona_dir: Path, now: datetime | None = None) -> None:
    """A genuine, sustained short-vs-long (recent-vs-baseline) cross on two channels.

    Placed RELATIVE to ``now`` (the tick's reference instant) so "recent" is recent
    for that tick:
      - joy:   OLD low events (2.0) + RECENT high events (9.0) → the SHORT read runs
               above the LONG baseline → a GOLDEN cross (delta > +0.5).
      - grief: OLD high events (9.0) + RECENT low events (2.0) → the SHORT read runs
               below baseline → a DEATH cross (delta < -0.5).
    Each channel is MULTI-event (a single event would give short == long under the
    normalized mean, hence no gap). The deltas are ~±1.8, robustly above the 0.5
    noise floor, so {joy, grief} is a stable per_channel key set across ticks — as
    ``now`` advances over a fixed memory set every event ages by the same amount, a
    uniform time-shift the normalized mean is invariant to, so the cross persists and
    sustained_ticks accumulates. Two divergent channels so a reconcile on joy leaves
    the gap non-empty (grief remains).
    """
    from brain.memory.store import Memory, MemoryStore

    now = now or datetime.now(UTC)
    store = MemoryStore(persona_dir / "memories.db")
    try:
        for name, old_val, new_val in (("joy", 2.0, 9.0), ("grief", 9.0, 2.0)):
            for days in (40, 45, 50):  # OLD events
                m = Memory.create_new(
                    content=f"an old {name} memory",
                    memory_type="episodic", domain="self",
                    emotions={name: old_val}, importance=7.0,
                )
                object.__setattr__(m, "created_at", now - timedelta(days=days))
                store.create(m)
            for hours in (1, 3, 6):  # RECENT events
                m = Memory.create_new(
                    content=f"a recent {name} memory",
                    memory_type="episodic", domain="self",
                    emotions={name: new_val}, importance=7.0,
                )
                object.__setattr__(m, "created_at", now - timedelta(hours=hours))
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

    # State file persisted, with an active current_gap that short/long
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


# ── Throttle-denial fix (self-model-note-null-fix, C1/C4/C8/C9) ───────────────
#
# The note went silently null on every tick since PR #151 because the shared
# throttle's 300s idle bar was essentially never met on a chatty persona.
# Revision 4's fix: a pre-flight cli_throttle.slot_available() peek in
# _run_self_model_tick, checked BEFORE _self_model_reflect is called at all,
# so a denied attempt costs nothing (C9) - and articulate()'s own real
# acquire_background() stays as the authoritative fallback for the rare race
# where the peek granted but a different accessor (or resumed chat) won the
# slot first (C8 route 2).


def test_preflight_peek_denial_is_zero_side_effect(tmp_path: Path) -> None:
    """C9: when the pre-flight throttle peek denies (the common case during a
    sustained-chat spell), NOTHING except the cadence advance and the C1
    error-sink log happens - no gap computation, no self_model_state.json
    write, no daily-budget consumption, no gaps_surfaced change, no feed
    event. Seeds a gap that WOULD be computed/consumed if a granted attempt
    ran, so this test can actually detect a regression rather than merely
    observing an absence because nothing would have happened anyway.

    Oracle (H6): this assertion set fails against the pre-Revision-4 design
    (throttle check inside articulate(), downstream of gap computation and
    budget consumption) - that design calls _self_model_reflect on every
    denied attempt, so all of these assertions would fail against it.
    """
    from brain.bridge import cli_throttle
    from brain.self_model import cadence as sm_cadence
    from brain.self_model.resolve import load_audit

    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir)
    bus = _CapturingBus()

    cli_throttle.mark_interactive_active()  # chat "just happened" — denies the peek

    _run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=bus)

    assert not (persona_dir / "self_model_state.json").exists(), (
        "a denied attempt must not compute or persist a gap"
    )
    assert not (persona_dir / "self_model" / "daily_articulate_budget.json").exists(), (
        "a denied attempt must not consume the daily articulate budget"
    )
    audit = load_audit(persona_dir)
    assert audit.get("gaps_surfaced", 0) == 0, (
        "a denied attempt must not bump the gaps_surfaced counter"
    )
    assert bus.events == [], "a denied attempt must not publish a self_model_tick event"

    error_path = persona_dir / "self_model_articulate_errors.jsonl"
    assert error_path.exists()
    rows = [json.loads(line) for line in error_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["kind"] == "self_model_articulate_deferred"

    advanced = sm_cadence.load(persona_dir)
    assert advanced.next_reflection_at is not None
    delta = (advanced.next_reflection_at - datetime.now(UTC)).total_seconds()
    assert 0 < delta <= 65, "cadence must advance to the short ~60s deferred retry, not ~6h"
    assert advanced.consecutive_failures == 0, "a deferral must not trigger the failure backoff"


def test_rare_race_throttle_deferred_still_persists_state_but_defers_cadence(
    tmp_path: Path,
) -> None:
    """C8 route 2: if articulate() raises ThrottleDeferred (the rare
    peek-then-lost race — pre-flight peek granted, but the real acquire then
    lost the slot), the cadence still advances to the short "deferred" retry
    (not the 6h normal interval, not the escalating failure backoff) — but
    UNLIKE route 1 (C9's zero-side-effects guarantee), self_model_state.json
    WAS written this tick, because gap computation already ran before the
    race was discovered. This is the accepted, maker-precedented tradeoff for
    the rare race, distinct from route 1's pure no-op."""
    from unittest.mock import patch

    from brain.bridge import cli_throttle
    from brain.self_model import cadence as sm_cadence

    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir)
    bus = _CapturingBus()

    with patch(
        "brain.bridge.supervisor.sm_articulate",
        side_effect=cli_throttle.ThrottleDeferred("lost the race"),
    ):
        _run_self_model_tick(persona_dir, provider=FakeProvider(), event_bus=bus)

    assert (persona_dir / "self_model_state.json").exists(), (
        "the rare-race path already ran gap computation before the race was "
        "discovered — state IS written, unlike route 1's pre-flight denial"
    )
    advanced = sm_cadence.load(persona_dir)
    assert advanced.next_reflection_at is not None
    delta = (advanced.next_reflection_at - datetime.now(UTC)).total_seconds()
    assert 0 < delta <= 65, "cadence must use the short deferred retry, not 6h or a backoff"
    assert advanced.consecutive_failures == 0, "a deferral must not trigger the failure backoff"


def test_carry_forward_same_channels_open_status_note_present(tmp_path: Path) -> None:
    """C4 case (i): same channels + prior status == "open" + prior note
    present → the note IS carried forward onto the new gap."""
    from unittest.mock import patch

    from brain.bridge.supervisor import _self_model_reflect
    from brain.self_model import state as sm_state
    from brain.self_model.gap import Gap

    provider = FakeProvider()
    base = datetime(2026, 6, 1, tzinfo=UTC)
    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir, now=base)

    prior_gap = Gap(
        per_channel={"joy": 1.8, "grief": -1.8},
        magnitude=3.6,
        unnamed_pressure=0.0,
        note="a note that should carry forward",
        status="open",
        first_seen_ts=base.isoformat(),
        last_seen_ts=base.isoformat(),
        sustained_ticks=1,
    )
    sm_state.save(
        persona_dir,
        sm_state.SelfModelState(current_gap=prior_gap, gap_history=[]),
    )

    with patch("brain.bridge.supervisor.sm_articulate", return_value=None):
        _self_model_reflect(
            persona_dir,
            provider=provider,
            event_bus=_CapturingBus(),
            now=base + timedelta(hours=6),
        )

    new_gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert new_gap is not None
    assert set(new_gap.per_channel) == {"joy", "grief"}, (
        "test setup check: the new tick's channel set must match the prior's"
    )
    assert new_gap.note == "a note that should carry forward"


def test_carry_forward_blocked_on_different_channels(tmp_path: Path) -> None:
    """C4 case (ii), MANDATORY: the adversarial cross-channel case that was
    the original gate-3 bounce-1 MAJOR finding, demonstrated live on the
    real persona's own gap_history (see decisions.md). A prior gap with a
    note, still open, but on a DIFFERENT channel set than the new tick's
    gap, must NOT have its stale, off-channel note carried forward."""
    from unittest.mock import patch

    from brain.bridge.supervisor import _self_model_reflect
    from brain.self_model import state as sm_state
    from brain.self_model.gap import Gap

    provider = FakeProvider()
    base = datetime(2026, 6, 1, tzinfo=UTC)
    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir, now=base)  # new tick will be {joy, grief}

    prior_gap = Gap(
        per_channel={"focus": 1.2},  # disjoint from {joy, grief}
        magnitude=1.2,
        unnamed_pressure=0.0,
        note="a stale note about an unrelated channel",
        status="open",
        first_seen_ts=base.isoformat(),
        last_seen_ts=base.isoformat(),
        sustained_ticks=1,
    )
    sm_state.save(
        persona_dir,
        sm_state.SelfModelState(current_gap=prior_gap, gap_history=[]),
    )

    with patch("brain.bridge.supervisor.sm_articulate", return_value=None):
        _self_model_reflect(
            persona_dir,
            provider=provider,
            event_bus=_CapturingBus(),
            now=base + timedelta(hours=6),
        )

    new_gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert new_gap is not None
    assert set(new_gap.per_channel) == {"joy", "grief"}
    assert new_gap.note is None, (
        "a stale, off-channel note must never be attached — different "
        "channel sets must block carry-forward"
    )


def test_carry_forward_blocked_when_no_prior_note(tmp_path: Path) -> None:
    """C4 case (iii): no prior gap at all (fresh persona) → new_gap.note
    stays None (nothing to carry forward)."""
    from unittest.mock import patch

    from brain.bridge.supervisor import _self_model_reflect
    from brain.self_model import state as sm_state

    provider = FakeProvider()
    base = datetime(2026, 6, 1, tzinfo=UTC)
    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir, now=base)
    # No prior state file written — this is a fresh persona's first tick.

    with patch("brain.bridge.supervisor.sm_articulate", return_value=None):
        _self_model_reflect(
            persona_dir, provider=provider, event_bus=_CapturingBus(), now=base
        )

    new_gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert new_gap is not None
    assert new_gap.note is None


def test_carry_forward_requires_prior_status_open_not_just_same_channels(
    tmp_path: Path,
) -> None:
    """C4 case (iv): a prior gap on the SAME channel set with a note, but
    whose status is NOT "open" (e.g. already reconciled/dismissed), must NOT
    have its note carried forward — same_channels requires
    prior_gap.status == "open" too, not just a channel-set match. Regression
    test for the gate-3 pass-5 MINOR fidelity finding (the criterion's prior
    wording named only the channel-set half of the predicate it reuses)."""
    from unittest.mock import patch

    from brain.bridge.supervisor import _self_model_reflect
    from brain.self_model import state as sm_state
    from brain.self_model.gap import Gap

    provider = FakeProvider()
    base = datetime(2026, 6, 1, tzinfo=UTC)
    persona_dir = _persona_dir(tmp_path)
    _seed_divergent_memories(persona_dir, now=base)

    # Hand-construct a prior state: an "acknowledged" (NOT open) gap on the
    # exact channel set the seeded memories will produce ({joy, grief}), with
    # a note present.
    prior_gap = Gap(
        per_channel={"joy": 1.8, "grief": -1.8},
        magnitude=3.6,
        unnamed_pressure=0.0,
        note="an old note that must not resurface",
        status="acknowledged",
        first_seen_ts=base.isoformat(),
        last_seen_ts=base.isoformat(),
        sustained_ticks=1,
    )
    sm_state.save(
        persona_dir,
        sm_state.SelfModelState(current_gap=prior_gap, gap_history=[]),
    )

    with patch("brain.bridge.supervisor.sm_articulate", return_value=None):
        _self_model_reflect(
            persona_dir,
            provider=provider,
            event_bus=_CapturingBus(),
            now=base + timedelta(hours=6),
        )

    new_gap = sm_state.load_or_recover(persona_dir)[0].current_gap
    assert new_gap is not None
    assert set(new_gap.per_channel) == {"joy", "grief"}, (
        "test setup check: the new tick's channel set must match the prior's"
    )
    assert new_gap.note is None, (
        "prior_gap.status != 'open' must block carry-forward even though the "
        "channel sets match"
    )


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
    _seed_divergent_memories(pa, now=base)

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
    _seed_divergent_memories(pb, now=base)

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
    _seed_divergent_memories(persona_dir, now=base)

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
