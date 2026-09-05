"""Through-path: the supervisor reaps stale sidecars on the maintenance
cadence (#176), alongside forgetting + narrative + pending-write sweep.

Entry point under test: brain.bridge.supervisor.run_folded.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge.supervisor import run_folded


def test_supervisor_sweeps_stale_sidecars_on_maintenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    sweep_calls: list = []
    stop_event = threading.Event()

    def _fake_sweep(persona_dir, *, now):
        sweep_calls.append(persona_dir)
        stop_event.set()
        return []

    monkeypatch.setattr("brain.health.sidecar_sweep.sweep_stale_sidecars", _fake_sweep)
    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", lambda *a, **k: {})
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", lambda *a, **k: (0, 0))
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())
    # #154: voice-reflection (background-generative tier) now builds its own
    # real Sonnet-tier provider, and calls the LLM unconditionally before its
    # own evidence gate — a real-subprocess hazard on this bare tmp_path
    # persona (no persona_config.json); not about this test, neutralise it.
    monkeypatch.setattr(
        "brain.bridge.supervisor._run_voice_reflection_tick", lambda *a, **k: None
    )
    threading.Timer(3.0, stop_event.set).start()

    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=0.05,
        finalize_interval_s=None,
    )

    assert len(sweep_calls) >= 1
