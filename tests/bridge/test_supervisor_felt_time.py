"""Integration test: supervisor wires FeltTime.tick() into the heartbeat cadence.

Verifies that after two heartbeat cadence cycles, FeltTime.tick was called
at least twice — i.e. the felt-time integration is present and fault-isolated.

Entry point under test: brain.bridge.supervisor.run_folded.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_supervisor_invokes_felt_time_tick_on_heartbeat_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_folded fires FeltTime.tick at least twice across two heartbeat cycles."""
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')

    # Count tick() calls via a fake FeltTime.
    tick_calls: list[tuple] = []

    class _FakeFeltTime:
        def __init__(self, *, persona_dir: Path) -> None:
            tick_calls.append(("init", persona_dir))

        def tick(self, ctx: object) -> None:
            tick_calls.append(("tick", ctx))

        def get_state(self) -> None:
            return None

    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", _FakeFeltTime)

    stop_event = threading.Event()
    provider = MagicMock()
    event_bus = MagicMock()

    # After two tick() calls stop the loop.
    def _stop_after_two_ticks(*_args: object, **_kwargs: object) -> None:
        tick_count = sum(1 for c in tick_calls if c[0] == "tick")
        if tick_count >= 2:
            stop_event.set()

    monkeypatch.setattr(
        "brain.bridge.supervisor._run_heartbeat_tick",
        lambda *a, **k: _stop_after_two_ticks(),
    )

    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=provider,
        event_bus=event_bus,
        tick_interval_s=0.05,
        heartbeat_interval_s=0.05,
        soul_review_interval_s=None,
        finalize_interval_s=None,
        log_rotation_interval_s=None,
        initiate_review_interval_s=None,
        voice_reflection_interval_s=None,
    )

    tick_count = sum(1 for c in tick_calls if c[0] == "tick")
    assert tick_count >= 2, f"Expected ≥2 felt-time tick() calls, got {tick_count}"


def test_supervisor_felt_time_tick_fault_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FeltTime.tick() raise must not crash the supervisor loop.

    The heartbeat tick still increments last_heartbeat_at and the loop
    continues running — fault-isolation is non-negotiable per spec.
    """
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')

    heartbeat_calls: list[int] = []

    # FeltTime that always raises on tick().
    class _ExplodingFeltTime:
        def __init__(self, *, persona_dir: Path) -> None:
            pass

        def tick(self, ctx: object) -> None:
            raise RuntimeError("felt-time tick exploded")

        def get_state(self) -> None:
            return None

    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", _ExplodingFeltTime)

    stop_event = threading.Event()
    provider = MagicMock()
    event_bus = MagicMock()

    # Stop after two heartbeat ticks (loop survived the exploding tick).
    def _counting_heartbeat(*_args: object, **_kwargs: object) -> None:
        heartbeat_calls.append(1)
        if len(heartbeat_calls) >= 2:
            stop_event.set()

    monkeypatch.setattr(
        "brain.bridge.supervisor._run_heartbeat_tick",
        _counting_heartbeat,
    )

    # Should not raise — fault isolation prevents propagation.
    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=provider,
        event_bus=event_bus,
        tick_interval_s=0.05,
        heartbeat_interval_s=0.05,
        soul_review_interval_s=None,
        finalize_interval_s=None,
        log_rotation_interval_s=None,
        initiate_review_interval_s=None,
        voice_reflection_interval_s=None,
    )

    assert len(heartbeat_calls) >= 2, (
        f"Loop should have run ≥2 heartbeat ticks despite exploding felt-time; "
        f"got {len(heartbeat_calls)}"
    )
