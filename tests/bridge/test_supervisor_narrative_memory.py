"""Integration tests: supervisor wires narrative_memory_run_pass into soul-review cadence.

Verifies two behaviours:

1. The narrative-memory pass fires on the soul-review cadence, AFTER the
   forgetting pass within the same cadence — order matters because forgetting
   drops memories first, then arc-update reads the surviving pool.
2. An exception raised by the narrative-memory wrapper is fault-isolated —
   the soul-review loop keeps running undisturbed.

Mirrors ``tests/bridge/test_supervisor_forgetting.py`` precedent (drives the
full ``run_folded`` loop with short cadences rather than a one-shot helper).

Entry point under test: ``brain.bridge.supervisor.run_folded``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge.supervisor import run_folded


def test_supervisor_runs_arc_update_after_forgetting_on_soul_review_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One soul-review tick should call forgetting then arc-update, in that order."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    call_order: list[str] = []
    stop_event = threading.Event()

    def _fake_forgetting(persona_dir, *, event_bus, **_):
        call_order.append("forgetting")
        return {"faded": 0, "lost": 0, "total": 0, "exempt": 0, "unfaded": 0, "duration_ms": 0}

    def _fake_arc_update(*args, **kwargs):
        call_order.append("arc_update")
        if call_order.count("arc_update") >= 1:
            stop_event.set()

    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", _fake_forgetting)
    monkeypatch.setattr(
        "brain.bridge.supervisor._run_narrative_memory_pass", _fake_arc_update
    )
    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())

    provider = MagicMock()
    event_bus = MagicMock()

    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=provider,
        event_bus=event_bus,
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=0.05,
        finalize_interval_s=None,
    )

    # Forgetting must precede arc_update inside the same cadence tick.
    assert "forgetting" in call_order
    assert "arc_update" in call_order
    forgetting_idx = call_order.index("forgetting")
    arc_idx = call_order.index("arc_update")
    assert forgetting_idx < arc_idx, (
        f"forgetting must run before arc_update; got order: {call_order}"
    )


def test_supervisor_arc_update_failure_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the narrative-memory wrapper raises, the supervisor loop keeps running."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    def _exploding_arc_update(*args, **kwargs):
        raise RuntimeError("synthetic arc-update failure")

    monkeypatch.setattr(
        "brain.bridge.supervisor._run_narrative_memory_pass", _exploding_arc_update
    )
    monkeypatch.setattr(
        "brain.bridge.supervisor.forgetting_run_pass",
        lambda *a, **k: {
            "faded": 0, "lost": 0, "total": 0, "exempt": 0, "unfaded": 0, "duration_ms": 0,
        },
    )
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())

    soul_review_calls: list[int] = [0]
    stop_event = threading.Event()

    def _soul_review_counter(*a, **k):
        soul_review_calls[0] += 1
        if soul_review_calls[0] >= 2:
            stop_event.set()

    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", _soul_review_counter)

    provider = MagicMock()
    event_bus = MagicMock()

    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=provider,
        event_bus=event_bus,
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=0.05,
        finalize_interval_s=None,
    )

    # Soul-review kept running even though arc-update raised each tick.
    assert soul_review_calls[0] >= 2
