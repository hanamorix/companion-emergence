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
