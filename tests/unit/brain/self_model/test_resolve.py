"""Task 6 — resolution tracking: sustained gap → soul candidate + feed event.

Two resolution paths (R-E4 dead-loop guard):
  PATH A: gap sustained >= _SUSTAINED_TICKS, then reconciled (status acknowledged/dismissed)
  PATH B: gap sustained >= _SUSTAINED_TICKS, then naturally reconverged (magnitude < threshold,
           NO tool call)

Tests added one at a time per TDD discipline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from brain.self_model.gap import Gap
from brain.self_model.resolve import (
    _SUSTAINED_TICKS,
    check_and_emit_resolution,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_gap(*, status: str = "open", sustained_ticks: int = 0, magnitude: float = 2.0) -> Gap:
    return Gap(
        per_channel={"grief": magnitude},
        magnitude=magnitude,
        unnamed_pressure=0.0,
        status=status,
        sustained_ticks=sustained_ticks,
    )


def _capture_events() -> tuple[list[dict], Any]:
    """Set a capturing publisher; return (events_list, teardown_fn)."""
    from brain.bridge import events as ev

    captured: list[dict] = []

    def _pub(event: dict) -> None:
        captured.append(event)

    ev.set_publisher(_pub)
    return captured, lambda: ev.set_publisher(None)


def _read_soul_candidates(persona_dir: Path) -> list[dict]:
    from brain.ingest.soul_queue import list_soul_candidates

    return list_soul_candidates(persona_dir)


# ── Test 1: sustained + reconcile → soul candidate + feed event ───────────────


def test_sustained_reconcile_queues_soul_candidate_and_feed_event(tmp_path: Path) -> None:
    """PATH A: gap sustained >= _SUSTAINED_TICKS, then marked acknowledged → emits both."""
    prior = _make_gap(status="open", sustained_ticks=_SUSTAINED_TICKS - 1, magnitude=2.0)
    # After reconcile the gap status flips to acknowledged
    resolved = _make_gap(status="acknowledged", sustained_ticks=_SUSTAINED_TICKS, magnitude=2.0)

    events, teardown = _capture_events()
    try:
        check_and_emit_resolution(prior, resolved, persona_dir=tmp_path, session_id="test_session")
    finally:
        teardown()

    # Soul candidate must be queued
    candidates = _read_soul_candidates(tmp_path)
    assert len(candidates) == 1, "expected exactly one soul candidate"
    c = candidates[0]
    assert c["status"] == "auto_pending"
    assert c["text"].strip()

    # Feed event must have been published
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "self_model_gap_resolved"
    assert evt.get("resolution_path") == "reconcile"


# ── Test 2: sustained + natural reconvergence → soul candidate + feed event ───


def test_sustained_natural_reconvergence_queues_soul_candidate_and_feed_event(
    tmp_path: Path,
) -> None:
    """PATH B (R-E4): resolution without tool call — magnitude drops to zero.

    The gap was open + sustained, and the NEW gap's magnitude is zero
    (declared re-matched derived). No reconcile tool was called.
    """
    prior = _make_gap(status="open", sustained_ticks=_SUSTAINED_TICKS, magnitude=2.0)
    # New gap: still open (no tool call), but magnitude collapsed to zero
    resolved = _make_gap(status="open", sustained_ticks=_SUSTAINED_TICKS, magnitude=0.0)

    events, teardown = _capture_events()
    try:
        check_and_emit_resolution(prior, resolved, persona_dir=tmp_path, session_id="test_session")
    finally:
        teardown()

    candidates = _read_soul_candidates(tmp_path)
    assert len(candidates) == 1, "PATH B must also queue a soul candidate (R-E4 not tool-hostage)"

    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "self_model_gap_resolved"
    assert evt.get("resolution_path") == "natural"


# ── Test 3: transient (< _SUSTAINED_TICKS) → NO soul candidate ────────────────


def test_transient_gap_does_not_crystallise(tmp_path: Path) -> None:
    """Only sustained gaps crystallise — a gap resolved before _SUSTAINED_TICKS is ignored."""
    prior = _make_gap(status="open", sustained_ticks=0, magnitude=2.0)
    resolved = _make_gap(status="acknowledged", sustained_ticks=_SUSTAINED_TICKS - 1, magnitude=2.0)

    events, teardown = _capture_events()
    try:
        check_and_emit_resolution(prior, resolved, persona_dir=tmp_path, session_id="test_session")
    finally:
        teardown()

    candidates = _read_soul_candidates(tmp_path)
    assert candidates == [], "transient gap must NOT queue a soul candidate"
    assert events == [], "transient gap must NOT publish a feed event"


# ── Test 4: audit counter increments ──────────────────────────────────────────


def test_audit_counter_increments_on_surface_and_reconcile(tmp_path: Path) -> None:
    """Audit counter records gaps_surfaced and reconciles_called (dead-loop guard)."""
    from brain.self_model.resolve import increment_gaps_surfaced, increment_reconciles, load_audit

    audit = load_audit(tmp_path)
    assert audit["gaps_surfaced"] == 0
    assert audit["reconciles_called"] == 0

    increment_gaps_surfaced(tmp_path)
    audit = load_audit(tmp_path)
    assert audit["gaps_surfaced"] == 1
    assert audit["reconciles_called"] == 0

    increment_gaps_surfaced(tmp_path)
    increment_reconciles(tmp_path)
    audit = load_audit(tmp_path)
    assert audit["gaps_surfaced"] == 2
    assert audit["reconciles_called"] == 1
