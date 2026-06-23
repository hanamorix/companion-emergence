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


def test_maintenance_fires_from_persisted_due_time_on_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "maintenance_cadence.json").write_text(
        json.dumps({"next_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat()})
    )
    # far-future soul state so soul review doesn't interfere
    (persona_dir / "soul_review_state.json").write_text(
        json.dumps(
            {
                "next_review_at": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                "consecutive_failures": 0,
            }
        )
    )
    _neutralise(monkeypatch)
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: None)
    calls = [0]
    stop = threading.Event()

    def _counter(*a, **k):
        calls[0] += 1
        stop.set()
        return {}

    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", _counter)

    watchdog = threading.Timer(2.0, stop.set)
    watchdog.start()
    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=6 * 3600.0,
        finalize_interval_s=None,
    )
    watchdog.cancel()

    assert calls[0] >= 1, "persisted past-due maintenance must fire on a fresh process"
    saved = json.loads((persona_dir / "maintenance_cadence.json").read_text())
    assert datetime.fromisoformat(saved["next_at"]) > datetime.now(UTC) + timedelta(hours=5)


def test_voice_reflection_fires_from_persisted_due_time_on_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "voice_reflection_cadence.json").write_text(
        json.dumps({"next_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat()})
    )
    _neutralise(monkeypatch)
    calls = [0]
    stop = threading.Event()

    def _counter(*a, **k):
        calls[0] += 1
        stop.set()

    monkeypatch.setattr("brain.bridge.supervisor._run_voice_reflection_tick", _counter)

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
        finalize_interval_s=None,
        voice_reflection_interval_s=86400.0,  # 24h: monotonic alone would never fire
    )
    watchdog.cancel()

    assert calls[0] >= 1, "persisted past-due voice reflection must fire on a fresh process"
    saved = json.loads((persona_dir / "voice_reflection_cadence.json").read_text())
    assert datetime.fromisoformat(saved["next_at"]) > datetime.now(UTC) + timedelta(hours=23)


def test_cadence_advances_even_when_tick_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The always-advance-on-exception contract (spec §Ordering contract): a tick
    # that RAISES must still advance+save the cadence — one re-fire per interval,
    # NOT a tight retry storm. Guards against a future refactor moving advance
    # into the try block (the plan's "major if violated"). Red-team finding M1.
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "finalize_cadence.json").write_text(
        json.dumps({"next_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat()})
    )
    _neutralise(monkeypatch)
    stop = threading.Event()
    raised = [0]

    def _raiser(*a, **k):
        raised[0] += 1
        stop.set()
        raise RuntimeError("boom")

    monkeypatch.setattr("brain.bridge.supervisor._run_finalize_tick", _raiser)

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
        finalize_interval_s=3600.0,
    )
    watchdog.cancel()

    assert raised[0] >= 1, "the raising tick must have fired"
    # Despite the tick raising, the cadence advanced to the future (not still past-due):
    saved = json.loads((persona_dir / "finalize_cadence.json").read_text())
    assert datetime.fromisoformat(saved["next_at"]) > datetime.now(UTC) + timedelta(minutes=50), (
        "advance+save must run in finally even when the tick raises"
    )


def test_maintenance_advances_even_when_forgetting_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintenance uses end-of-block advance (NO finally) — it relies on every
    # inner tick being individually try/except-wrapped. Pin that: forgetting
    # raises, yet the cadence still advances. Guards the fragile no-finally
    # structure against a future un-wrapped statement above the advance.
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "maintenance_cadence.json").write_text(
        json.dumps({"next_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat()})
    )
    _neutralise(monkeypatch)
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: None)
    stop = threading.Event()
    raised = [0]

    def _raiser(*a, **k):
        raised[0] += 1
        stop.set()
        raise RuntimeError("boom")

    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", _raiser)

    watchdog = threading.Timer(2.0, stop.set)
    watchdog.start()
    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=6 * 3600.0,
        finalize_interval_s=None,
    )
    watchdog.cancel()

    assert raised[0] >= 1
    saved = json.loads((persona_dir / "maintenance_cadence.json").read_text())
    assert datetime.fromisoformat(saved["next_at"]) > datetime.now(UTC) + timedelta(hours=5), (
        "maintenance must advance even when forgetting raises (end-of-block, no finally)"
    )


def test_disabled_cadence_writes_no_state_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # interval_s=None disables a cadence — it must never load or write its state
    # file. Run with all three off and a fast no-op heartbeat to drive a tick.
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _neutralise(monkeypatch)
    stop = threading.Event()
    monkeypatch.setattr(
        "brain.bridge.supervisor._run_heartbeat_tick",
        lambda *a, **k: stop.set(),
    )

    watchdog = threading.Timer(2.0, stop.set)
    watchdog.start()
    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=MagicMock(),
        event_bus=MagicMock(),
        tick_interval_s=0.05,
        heartbeat_interval_s=0.01,
        soul_review_interval_s=None,
        finalize_interval_s=None,
        voice_reflection_interval_s=None,
    )
    watchdog.cancel()

    for name in ("finalize_cadence.json", "maintenance_cadence.json", "voice_reflection_cadence.json"):
        assert not (persona_dir / name).exists(), f"disabled cadence must not write {name}"
