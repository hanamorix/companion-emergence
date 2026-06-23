"""voice/maintenance/finalize cadences are persisted wall-clock (defer #21).

Mirrors test_supervisor_soul_cadence.py — a due-in-the-past persisted state fires
on a fresh process even with a long interval the monotonic timer would never have
reached. A 2s watchdog stops the loop so a non-firing (RED) state can't hang.
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge.supervisor import run_folded


def _neutralise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_felt_time_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())


def test_finalize_fires_from_persisted_due_time_on_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "finalize_cadence.json").write_text(
        json.dumps({"next_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat()})
    )
    _neutralise(monkeypatch)
    calls = [0]
    stop = threading.Event()

    def _counter(*a, **k):
        calls[0] += 1
        stop.set()

    monkeypatch.setattr("brain.bridge.supervisor._run_finalize_tick", _counter)

    watchdog = threading.Timer(2.0, stop.set)
    watchdog.start()
    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=None,
        finalize_interval_s=3600.0,  # 1h: monotonic alone would NOT fire on a fresh proc
    )
    watchdog.cancel()

    assert calls[0] >= 1, "persisted past-due finalize must fire on a fresh process"
    saved = json.loads((persona_dir / "finalize_cadence.json").read_text())
    assert datetime.fromisoformat(saved["next_at"]) > datetime.now(UTC) + timedelta(minutes=50)
