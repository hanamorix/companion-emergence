"""Integration tests: supervisor wires forgetting_run_pass into soul-review cadence.

Verifies two behaviours:
1. forgetting_run_pass is invoked at least twice across two soul-review cadence cycles.
2. An exception raised by forgetting_run_pass is fault-isolated — the soul-review
   loop keeps running undisturbed.

Entry point under test: brain.bridge.supervisor.run_folded.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge.supervisor import run_folded


def test_supervisor_invokes_forgetting_pass_on_soul_review_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_folded calls forgetting_run_pass at least twice across two soul-review cycles."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    forgetting_calls: list = []
    stop_event = threading.Event()

    def _fake_forgetting_pass(persona_dir, *, event_bus, **_):
        forgetting_calls.append(persona_dir)
        if len(forgetting_calls) >= 2:
            stop_event.set()
        return {"faded": 0, "lost": 0, "total": 0, "exempt": 0, "unfaded": 0, "duration_ms": 0}

    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", _fake_forgetting_pass)
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

    assert len(forgetting_calls) >= 2


def test_supervisor_forgetting_pass_fault_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception in forgetting_run_pass must not break the supervisor loop."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    def _exploding_pass(persona_dir, *, event_bus, **_):
        raise RuntimeError("boom")

    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", _exploding_pass)
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

    # Soul-review kept running even though forgetting raised each time.
    assert soul_review_calls[0] >= 2


def test_supervisor_passes_intensity_drivers_to_forgetting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_folded caches IntensityDrivers from the felt-time tick and forwards them
    to forgetting_run_pass so arc pressure can modulate the fade threshold."""
    from brain.felt_time.lived_age import IntensityDrivers

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    fake_drivers = IntensityDrivers(narrative_weight=0.7)
    captured_drivers: list = []
    stop_event = threading.Event()

    def _fake_felt_time_tick(*a, **k):
        return fake_drivers  # supervisor caches this return value

    def _fake_forgetting_pass(persona_dir, *, event_bus, intensity_drivers=None, **_):
        captured_drivers.append(intensity_drivers)
        stop_event.set()
        return {"faded": 0, "lost": 0, "total": 0, "exempt": 0, "unfaded": 0, "duration_ms": 0}

    monkeypatch.setattr("brain.bridge.supervisor._run_felt_time_tick", _fake_felt_time_tick)
    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", _fake_forgetting_pass)
    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: None)

    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.05,
        # Both intervals set to 0 so they fire on the first loop iteration.
        # Heartbeat block runs first in the loop body, then soul-review — so
        # drivers computed during heartbeat are available when forgetting fires.
        heartbeat_interval_s=0.0,
        soul_review_interval_s=0.0,
        finalize_interval_s=None,
        initiate_review_interval_s=None,
        voice_reflection_interval_s=None,
        log_rotation_interval_s=None,
    )

    assert len(captured_drivers) >= 1, "forgetting pass should have been called"
    assert captured_drivers[0] is not None, "intensity_drivers should be forwarded, not None"
    assert captured_drivers[0].narrative_weight == pytest.approx(0.7)
