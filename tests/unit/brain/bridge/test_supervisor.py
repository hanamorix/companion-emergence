"""Tests for brain.bridge.supervisor.run_folded — the bridge supervisor
thread that runs session-cleanup AND autonomous heartbeat cadences."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain.bridge.events import EventBus
from brain.bridge.provider import FakeProvider
from brain.bridge.supervisor import (
    _run_heartbeat_tick,
    _run_initiate_review_tick,  # noqa: F401 — imported to assert symbol exists
    _run_log_rotation_tick,
    run_folded,
)


def _persona_dir(tmp_path: Path) -> Path:
    """Minimal persona dir for the supervisor to operate against."""
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}'
    )
    return p


class _CapturingBus:
    """In-process bus stand-in that records every published event.

    The real EventBus.publish() requires a bound asyncio loop and is a
    no-op otherwise; tests don't run a loop so we substitute a duck-typed
    bus that records every dict the supervisor publishes.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, event: dict) -> None:
        self.events.append(event)


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
    bus = _CapturingBus()

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
        e.get("type") == "heartbeat_tick" for e in bus.events
    ), "heartbeat_tick event was not published"


# ---------------------------------------------------------------------------
# Snapshot sweep + finalize cadence tests (Phase B sticky sessions)
# ---------------------------------------------------------------------------


def test_supervisor_snapshot_sweep_keeps_session_alive(tmp_path: Path) -> None:
    """After a snapshot sweep, the session must remain in _SESSIONS and its
    buffer file on disk."""
    from brain.chat.session import create_session, get_session, reset_registry
    from brain.ingest.buffer import ingest_turn

    reset_registry()
    persona_dir = _persona_dir(tmp_path)
    sess = create_session(persona_dir.name)
    sid = sess.session_id
    old_ts = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()
    ingest_turn(persona_dir, {"session_id": sid, "speaker": "user",
                              "text": "earlier", "ts": old_ts})

    bus = _CapturingBus()

    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": FakeProvider(),
            "event_bus": bus,
            "tick_interval_s": 0.1,
            "silence_minutes": 5.0,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_interval_s": None,
        },
    )
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=2.0)

    buf = persona_dir / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists(), "snapshot sweep must NOT delete the buffer"
    assert get_session(sid) is not None, "snapshot sweep must NOT evict from _SESSIONS"
    types = [e.get("type") for e in bus.events]
    assert "session_snapshot" in types
    assert "session_closed" not in types
    reset_registry()


def test_supervisor_finalize_cadence_drops_old_sessions(tmp_path: Path) -> None:
    """The hourly finalize cadence at 24h silence runs, deletes the buffer
    + cursor, evicts from _SESSIONS, and publishes session_finalized."""
    from brain.chat.session import create_session, get_session, reset_registry
    from brain.ingest.buffer import ingest_turn

    reset_registry()
    persona_dir = _persona_dir(tmp_path)
    sess = create_session(persona_dir.name)
    sid = sess.session_id
    old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    ingest_turn(persona_dir, {"session_id": sid, "speaker": "user",
                              "text": "ancient", "ts": old_ts})

    bus = _CapturingBus()

    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": FakeProvider(),
            "event_bus": bus,
            "tick_interval_s": 0.1,
            "silence_minutes": 5.0,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_after_hours": 24.0,
            "finalize_interval_s": 0.05,  # fire near-immediately
        },
    )
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=2.0)

    buf = persona_dir / "active_conversations" / f"{sid}.jsonl"
    assert not buf.exists(), "finalize must delete the buffer"
    assert get_session(sid) is None, "finalize must remove from _SESSIONS"
    types = [e.get("type") for e in bus.events]
    assert "session_finalized" in types
    reset_registry()


# ---------------------------------------------------------------------------
# Log rotation tick (Phase 4 of JSONL retention plan)
# ---------------------------------------------------------------------------


def _write_oversize_log(persona_dir: Path, name: str, size_bytes: int) -> Path:
    """Write a JSONL log file padded past size_bytes for rotation tests."""
    p = persona_dir / name
    big = "x" * 1000
    import json as _json
    with p.open("w", encoding="utf-8") as f:
        line = _json.dumps({"at": "2026-01-01T00:00:00+00:00", "pad": big}) + "\n"
        # Write enough lines to exceed size_bytes.
        per_line = len(line.encode())
        n = (size_bytes // per_line) + 2
        for _ in range(n):
            f.write(line)
    return p


def test_run_log_rotation_tick_rotates_oversize_heartbeat(tmp_path: Path) -> None:
    """An oversize heartbeats.log.jsonl gets rotated to .1.gz."""
    persona_dir = _persona_dir(tmp_path)
    _write_oversize_log(persona_dir, "heartbeats.log.jsonl", size_bytes=200_000)
    bus = _CapturingBus()

    # Force the cap small so the seeded log trips it.
    _run_log_rotation_tick(
        persona_dir,
        bus,
        rolling_size_bytes=10_000,
    )

    assert (persona_dir / "heartbeats.log.jsonl.1.gz").exists()
    types = [e.get("type") for e in bus.events]
    assert "log_rotation" in types


def test_run_log_rotation_tick_archives_old_year_in_soul_audit(
    tmp_path: Path,
) -> None:
    """soul_audit.jsonl with 2024 entries → archived to .2024.jsonl.gz."""
    persona_dir = _persona_dir(tmp_path)
    audit = persona_dir / "soul_audit.jsonl"
    import json as _json
    with audit.open("w", encoding="utf-8") as f:
        f.write(_json.dumps({"ts": "2024-06-15T00:00:00+00:00", "seq": 0}) + "\n")
        f.write(_json.dumps({"ts": "2026-06-15T00:00:00+00:00", "seq": 1}) + "\n")

    bus = _CapturingBus()
    _run_log_rotation_tick(
        persona_dir,
        bus,
        now=datetime(2026, 5, 11, tzinfo=UTC),
    )

    assert (persona_dir / "soul_audit.2024.jsonl.gz").exists()
    # Active retains only 2026 entries.
    remaining = [
        _json.loads(line) for line in audit.read_text().splitlines() if line.strip()
    ]
    assert len(remaining) == 1
    assert remaining[0]["seq"] == 1


def test_run_log_rotation_tick_archives_old_year_in_initiate_audit(
    tmp_path: Path,
) -> None:
    """initiate_audit.jsonl with 2024 entries → archived to .2024.jsonl.gz."""
    persona_dir = _persona_dir(tmp_path)
    audit = persona_dir / "initiate_audit.jsonl"
    import json as _json
    with audit.open("w", encoding="utf-8") as f:
        f.write(_json.dumps({"ts": "2024-06-15T00:00:00+00:00", "seq": 0}) + "\n")
        f.write(_json.dumps({"ts": "2026-06-15T00:00:00+00:00", "seq": 1}) + "\n")
    bus = _CapturingBus()
    _run_log_rotation_tick(
        persona_dir, bus, now=datetime(2026, 5, 11, tzinfo=UTC)
    )
    assert (persona_dir / "initiate_audit.2024.jsonl.gz").exists()


def test_run_log_rotation_tick_no_op_when_within_caps(tmp_path: Path) -> None:
    """Small logs → tick runs cleanly, creates nothing."""
    persona_dir = _persona_dir(tmp_path)
    (persona_dir / "heartbeats.log.jsonl").write_text("{}\n", encoding="utf-8")
    bus = _CapturingBus()
    _run_log_rotation_tick(persona_dir, bus)
    # No archive should exist.
    assert list(persona_dir.glob("*.gz")) == []


def test_run_log_rotation_tick_fault_isolated_per_log(tmp_path: Path) -> None:
    """A failure rotating one log doesn't block other logs in the same tick."""
    persona_dir = _persona_dir(tmp_path)
    # Seed an oversize heartbeats AND an oversize dreams. Patch the rolling
    # rotator to raise on heartbeats only; dreams must still rotate.
    _write_oversize_log(persona_dir, "heartbeats.log.jsonl", 200_000)
    _write_oversize_log(persona_dir, "dreams.log.jsonl", 200_000)
    bus = _CapturingBus()

    real_rotate = __import__(
        "brain.health.log_rotation", fromlist=["rotate_rolling_size"]
    ).rotate_rolling_size

    def selective(log_path, **kw):
        if "heartbeats" in log_path.name:
            raise OSError("simulated disk error")
        return real_rotate(log_path, **kw)

    with patch(
        "brain.bridge.supervisor.rotate_rolling_size", side_effect=selective
    ):
        _run_log_rotation_tick(persona_dir, bus, rolling_size_bytes=10_000)

    # heartbeats archive missing (failed), dreams archive present.
    assert not (persona_dir / "heartbeats.log.jsonl.1.gz").exists()
    assert (persona_dir / "dreams.log.jsonl.1.gz").exists()


def test_run_log_rotation_tick_emits_event_per_rotation(tmp_path: Path) -> None:
    """Each rotation publishes a structured log_rotation event."""
    persona_dir = _persona_dir(tmp_path)
    _write_oversize_log(persona_dir, "heartbeats.log.jsonl", 200_000)
    bus = _CapturingBus()
    _run_log_rotation_tick(persona_dir, bus, rolling_size_bytes=10_000)

    rotations = [e for e in bus.events if e.get("type") == "log_rotation"]
    assert len(rotations) >= 1
    e = rotations[0]
    assert e["log"] == "heartbeats.log.jsonl"
    assert e["action"] == "rotated"


def test_run_folded_fires_log_rotation_after_interval(tmp_path: Path) -> None:
    """run_folded wires log rotation into the cadence loop."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def fake_rotation(*args, **kwargs):
        fired.set()

    def runner():
        with patch(
            "brain.bridge.supervisor._run_log_rotation_tick",
            side_effect=fake_rotation,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=0.0,  # fire on first iteration
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    assert fired.wait(timeout=5.0), "log rotation never fired despite zero interval"
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive()


def test_run_folded_skips_log_rotation_when_disabled(tmp_path: Path) -> None:
    """log_rotation_interval_s=None disables the cadence."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired: list[int] = []

    def fake_rotation(*args, **kwargs):
        fired.append(1)

    def runner():
        with patch(
            "brain.bridge.supervisor._run_log_rotation_tick",
            side_effect=fake_rotation,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert fired == [], "log rotation fired even though disabled"


def test_run_folded_fires_initiate_review_after_interval(tmp_path: Path) -> None:
    """run_folded wires initiate review into the cadence loop."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def fake_initiate(*args, **kwargs):
        fired.set()

    def runner():
        with patch(
            "brain.bridge.supervisor._run_initiate_review_tick",
            side_effect=fake_initiate,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=0.0,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    assert fired.wait(timeout=5.0), "initiate review never fired"
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive()


def test_run_folded_skips_initiate_review_when_disabled(tmp_path: Path) -> None:
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired: list[int] = []

    def fake_initiate(*args, **kwargs):
        fired.append(1)

    def runner():
        with patch(
            "brain.bridge.supervisor._run_initiate_review_tick",
            side_effect=fake_initiate,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=None,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert fired == []
