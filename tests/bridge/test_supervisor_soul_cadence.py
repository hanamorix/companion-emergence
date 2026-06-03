"""Soul-review cadence is persisted + self-pacing (regression for the backlog
a user hit after a multi-day Claude session-limit outage).

The old cadence timed off process-relative time.monotonic(), which reset on
every supervisor restart and didn't advance during sleep — so a 6h interval
rarely elapsed on a desktop app and candidates piled up. The cadence now
persists its next-due time (wall clock), so a due-in-the-past state fires on a
fresh process regardless of the interval.
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge.supervisor import run_folded


def _neutralise_other_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", lambda *a, **k: {})
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor._run_felt_time_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())


def test_soul_review_fires_from_persisted_due_time_on_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted next_review_at in the past fires soul review immediately on a
    fresh supervisor — even with a 6h interval, which the monotonic-only timer
    would have made wait 6h (the under-firing defect)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "soul_review_state.json").write_text(
        json.dumps(
            {
                "next_review_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                "consecutive_failures": 0,
            }
        )
    )
    _neutralise_other_ticks(monkeypatch)

    calls = [0]
    stop = threading.Event()

    def _counter(*a, **k):
        calls[0] += 1
        stop.set()
        return 0, 0  # (model_failures, eligible_pending)

    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", _counter)

    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=6 * 3600.0,  # 6h: monotonic alone would NOT fire
        finalize_interval_s=None,
    )

    assert calls[0] >= 1, "persisted past-due next_review_at must fire on a fresh process"
    # Clean drain → cadence advanced ~6h ahead and was persisted (save works).
    state = json.loads((persona_dir / "soul_review_state.json").read_text())
    saved_next = datetime.fromisoformat(state["next_review_at"])
    assert saved_next > datetime.now(UTC) + timedelta(hours=5)
