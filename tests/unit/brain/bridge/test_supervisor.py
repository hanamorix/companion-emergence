"""Tests for brain.bridge.supervisor.run_folded — the bridge supervisor
thread that runs session-cleanup AND autonomous heartbeat cadences."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain.bridge.events import EventBus
from brain.bridge.provider import FakeProvider
from brain.bridge.supervisor import _run_heartbeat_tick, run_folded


def _persona_dir(tmp_path: Path) -> Path:
    """Minimal persona dir for the supervisor to operate against."""
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}'
    )
    return p


def test_run_folded_exits_when_stop_event_is_set(tmp_path: Path) -> None:
    """The supervisor loop honors stop_event and returns cleanly."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    stop.set()  # already set — loop should exit immediately
    # Should return without hanging
    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=FakeProvider(),
        event_bus=bus,
        tick_interval_s=0.1,
        heartbeat_interval_s=None,
    )


def test_run_folded_skips_heartbeat_when_disabled(tmp_path: Path) -> None:
    """heartbeat_interval_s=None disables the autonomous cadence."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired: list[int] = []

    def fake_heartbeat(*args, **kwargs):
        fired.append(1)

    def runner():
        with patch(
            "brain.bridge.supervisor._run_heartbeat_tick", side_effect=fake_heartbeat
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    # Let it run a few cycles then stop
    import time as _t

    _t.sleep(0.3)
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert len(fired) == 0, "heartbeat fired even though disabled"


def test_run_folded_fires_heartbeat_after_interval(tmp_path: Path) -> None:
    """When heartbeat_interval_s elapses, _run_heartbeat_tick is called.

    Drives the loop in a worker thread so we can stop it from the test
    once we've observed the heartbeat fire. tick_interval_s is small
    so the loop's stop_event.wait() exits promptly when we set stop.
    """
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def fake_heartbeat(*args, **kwargs):
        fired.set()

    def runner():
        with patch(
            "brain.bridge.supervisor._run_heartbeat_tick", side_effect=fake_heartbeat
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=0.0,  # fire on first iteration
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    assert fired.wait(timeout=5.0), "heartbeat never fired despite zero interval"
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive(), "supervisor loop did not exit after stop_event"


def test_heartbeat_failure_does_not_break_supervisor_loop(tmp_path: Path) -> None:
    """A heartbeat exception is fault-isolated; supervisor keeps ticking.

    We let the patched heartbeat raise on every call, observe at least
    two attempts (proving the loop survived the first one), then stop.
    """
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    attempts: list[int] = []
    second_attempt = threading.Event()

    def boom(*args, **kwargs):
        attempts.append(1)
        if len(attempts) >= 2:
            second_attempt.set()
        raise RuntimeError("simulated heartbeat failure")

    def runner():
        with patch("brain.bridge.supervisor._run_heartbeat_tick", side_effect=boom):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=0.0,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    assert second_attempt.wait(timeout=5.0), "loop did not survive first heartbeat exception"
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive()


def test_run_heartbeat_tick_publishes_result_event(tmp_path: Path) -> None:
    """_run_heartbeat_tick fires HeartbeatEngine.run_tick and publishes."""
    persona_dir = _persona_dir(tmp_path)
    bus_events: list = []
    bus = MagicMock()
    bus.publish.side_effect = lambda payload: bus_events.append(payload)

    # Fake HeartbeatEngine returns a result we can predict
    fake_result = MagicMock(
        memories_decayed=0,
        edges_pruned=0,
        dream_id=None,
        reflex_fired=(),
        research_fired=None,
        growth_emotions_added=0,
        reflex_error=None,
        growth_error=None,
    )
    fake_engine_class = MagicMock(return_value=MagicMock(run_tick=MagicMock(return_value=fake_result)))
    with (
        patch("brain.engines.heartbeat.HeartbeatEngine", fake_engine_class),
        patch("brain.emotion.persona_loader.load_persona_vocabulary"),
    ):
        _run_heartbeat_tick(persona_dir, FakeProvider(), bus)

    assert any(
        e.get("type") == "heartbeat_tick" for e in bus_events
    ), "heartbeat_tick event was not published"
