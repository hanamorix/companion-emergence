"""The supervisor's default tick is the ``embedding_backfill.tick_interval_seconds`` setting (#531).

``run_folded`` and ``build_app`` each used to carry their own ``60.0`` for the tick. The setting is the same quantity
(its description is the supervisor's tick), so a change to it missed the loop. ``tick_interval_s=None`` now means "read
the setting"; an explicit value, including ``0.0``, still wins.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import brain.bridge.provider as provider_mod
import brain.bridge.supervisor as supervisor_mod
from brain import tunables
from brain.bridge.events import EventBus
from brain.memory import embedding_backfill
from tests.thread_support import finish_threads

_KEY = "embedding_backfill.tick_interval_seconds"


class _RecordingEvent(threading.Event):
    """A stop event whose ``wait`` records the timeout it was given and ends the loop at once (no real waiting)."""

    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        self.set()
        return True


def _waits_of_one_loop(persona_dir: Path, monkeypatch, **tick) -> list[float | None]:
    monkeypatch.setattr(supervisor_mod.central_cadence, "run_central_pass", lambda *_a, **_kw: [])
    monkeypatch.setattr(supervisor_mod, "build_tier_provider", lambda *_a, **_kw: provider_mod.FakeProvider())
    stop_event = _RecordingEvent()
    kwargs = {
        "stop_event": stop_event,
        "persona_dir": persona_dir,
        "provider": provider_mod.FakeProvider(),
        "event_bus": EventBus(),
        "heartbeat_interval_s": None,
        "soul_review_interval_s": None,
        "finalize_interval_s": None,
        "log_rotation_interval_s": None,
        "initiate_review_interval_s": None,
        "voice_reflection_interval_s": None,
        "self_model_interval_s": None,
        "compaction_interval_s": None,
        "calibration_interval_s": None,
        "interest_sweep_interval_s": None,
        "judge_selftune_interval_s": None,
        "clustering_interval_s": None,
        "vocab_repair_interval_s": None,
        "pending_reconcile_interval_s": None,
        "maker_enabled": False,
        "notes_enabled": False,
        "kindled_link_enabled": False,
        **tick,
    }
    thread = threading.Thread(target=lambda: supervisor_mod.run_folded(**kwargs), daemon=True)
    thread.start()
    finish_threads(thread)
    return stop_event.waits


def test_the_default_tick_follows_the_setting_and_an_explicit_tick_wins(tmp_path, monkeypatch) -> None:
    persona_dir = tmp_path / "persona"
    (persona_dir / "active_conversations").mkdir(parents=True)
    tunables.set_values({_KEY: 7.5})

    waits = _waits_of_one_loop(persona_dir, monkeypatch)
    assert 7.5 in waits and 60.0 not in waits, waits

    explicit = _waits_of_one_loop(persona_dir, monkeypatch, tick_interval_s=0.0)
    assert 0.0 in explicit and 7.5 not in explicit, explicit


# A hand-edited value that is not a finite number above 0 reads as the registered default (#531): 0, a negative number and
# `nan` would make the wait return at once, and `inf` makes Event.wait raise OverflowError and ends the supervisor.
_BAD_TICKS = (0.0, -1.0, float("nan"), float("inf"))


def test_a_bad_tick_setting_runs_the_loop_at_the_registered_default(tmp_path, monkeypatch) -> None:
    persona_dir = tmp_path / "persona"
    (persona_dir / "active_conversations").mkdir(parents=True)
    default = embedding_backfill._TICK_INTERVAL_SECONDS_DEFAULT  # noqa: SLF001
    tunables.set_values({_KEY: 7.5})
    waits = _waits_of_one_loop(persona_dir, monkeypatch)
    assert 7.5 in waits and default not in waits, waits  # the loop follows the setting, so the next phases mean something

    for bad in _BAD_TICKS:
        tunables.set_values({_KEY: bad})
        waits = _waits_of_one_loop(persona_dir, monkeypatch)
        assert default in waits, (bad, waits)

    tunables.set_values({_KEY: 0.0})
    explicit = _waits_of_one_loop(persona_dir, monkeypatch, tick_interval_s=0.0)
    assert 0.0 in explicit and default not in explicit, explicit  # an explicit value still wins


def test_a_bad_tick_setting_reads_as_the_default_and_is_logged_once(monkeypatch, caplog) -> None:
    default = embedding_backfill._TICK_INTERVAL_SECONDS_DEFAULT  # noqa: SLF001
    monkeypatch.setattr(tunables, "_warned_keys", set())
    with caplog.at_level(logging.WARNING, logger="brain.tunables"):
        for bad in _BAD_TICKS:
            tunables.set_values({_KEY: bad})
            caplog.clear()
            assert embedding_backfill.tick_interval_seconds() == default
            assert embedding_backfill.tick_interval_seconds() == default
            texts = [r.getMessage() for r in caplog.records]
            assert len(texts) == 1 and _KEY in texts[0] and repr(bad) in texts[0], (bad, texts)
        tunables.set_values({_KEY: 7.5})
        caplog.clear()
        assert embedding_backfill.tick_interval_seconds() == 7.5
    assert caplog.records == []


# A tick above threading.TIMEOUT_MAX is a value the platform does not allow as a wait timeout: Event.wait raises OverflowError
# for it and ends the supervisor, so it reads as the registered default too (#531). The limit differs per platform, so the
# tests build every value from the constant.
def _over_the_wait_limit() -> tuple[float, ...]:
    return (
        math.nextafter(threading.TIMEOUT_MAX, math.inf),
        threading.TIMEOUT_MAX * 2,
        threading.TIMEOUT_MAX * 1e3,
    )


def test_a_tick_above_the_wait_limit_runs_the_loop_at_the_registered_default(tmp_path, monkeypatch) -> None:
    persona_dir = tmp_path / "persona"
    (persona_dir / "active_conversations").mkdir(parents=True)
    default = embedding_backfill._TICK_INTERVAL_SECONDS_DEFAULT  # noqa: SLF001
    tunables.set_values({_KEY: 7.5})
    waits = _waits_of_one_loop(persona_dir, monkeypatch)
    assert 7.5 in waits and default not in waits, waits  # the loop follows the setting, so the next phases mean something

    for too_long in _over_the_wait_limit():
        tunables.set_values({_KEY: too_long})
        waits = _waits_of_one_loop(persona_dir, monkeypatch)
        assert default in waits and too_long not in waits, (too_long, waits)

    for allowed in (threading.TIMEOUT_MAX, math.nextafter(threading.TIMEOUT_MAX, 0)):
        tunables.set_values({_KEY: allowed})
        waits = _waits_of_one_loop(persona_dir, monkeypatch)
        assert allowed in waits and default not in waits, (allowed, waits)  # the limit itself is allowed

    # The platform really accepts the limit as a wait timeout (a timer ends the wait at once; no OverflowError).
    event = threading.Event()
    timer = threading.Timer(0.05, event.set)
    timer.start()
    try:
        assert event.wait(timeout=threading.TIMEOUT_MAX) is True
    finally:
        timer.cancel()
        timer.join()


def test_a_tick_above_the_wait_limit_reads_as_the_default_and_is_logged_once(monkeypatch, caplog) -> None:
    default = embedding_backfill._TICK_INTERVAL_SECONDS_DEFAULT  # noqa: SLF001
    monkeypatch.setattr(tunables, "_warned_keys", set())
    with caplog.at_level(logging.WARNING, logger="brain.tunables"):
        for too_long in _over_the_wait_limit():
            tunables.set_values({_KEY: too_long})
            caplog.clear()
            assert embedding_backfill.tick_interval_seconds() == default
            assert embedding_backfill.tick_interval_seconds() == default
            texts = [r.getMessage() for r in caplog.records]
            assert len(texts) == 1 and _KEY in texts[0] and repr(too_long) in texts[0], (too_long, texts)
            assert repr(threading.TIMEOUT_MAX) in texts[0], texts  # the text names the limit
        tunables.set_values({_KEY: threading.TIMEOUT_MAX})
        caplog.clear()
        assert embedding_backfill.tick_interval_seconds() == threading.TIMEOUT_MAX
    assert caplog.records == []
