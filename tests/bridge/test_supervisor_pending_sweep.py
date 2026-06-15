"""Integration test: supervisor sweeps expired pending file-writes on the
maintenance cadence (alongside forgetting + narrative).

Entry point under test: brain.bridge.supervisor.run_folded.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge.supervisor import run_folded


def test_supervisor_sweeps_expired_pending_writes_on_maintenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_folded calls pending.sweep_expired at least once on the maintenance tick."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    sweep_calls: list = []
    stop_event = threading.Event()

    def _fake_sweep(persona_dir, *, now):
        sweep_calls.append(persona_dir)
        stop_event.set()
        return 0

    monkeypatch.setattr("brain.files.pending.sweep_expired", _fake_sweep)
    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", lambda *a, **k: {})
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", lambda *a, **k: (0, 0))
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())

    # Watchdog: if the sweep never fires, stop the loop after a short window so
    # the test fails on the assertion below rather than hanging forever.
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
