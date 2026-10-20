"""The supervisor survives a crashed loop once, and says so (C25, C26; Q6, D8, D9).

A loop death is logged once with its traceback and the loop runs again with the same state (the startup one-shots are
not re-run); a second death before one completed iteration is "failed". A startup death re-runs the whole startup
once. A requested stop is never a death. Status is shown on /health and /persona/state.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

import brain.bridge.provider as provider_mod
import brain.bridge.supervisor as sup
from brain.bridge import cli_throttle
from brain.bridge.events import EventBus


@pytest.fixture(autouse=True)
def _clean():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


def _kwargs(pd: Path) -> dict:
    return {
        "persona_dir": pd, "provider": provider_mod.FakeProvider(), "event_bus": EventBus(), "tick_interval_s": 0.01,
        "heartbeat_interval_s": None, "soul_review_interval_s": 3600.0, "finalize_interval_s": None,
        "log_rotation_interval_s": None, "initiate_review_interval_s": None, "voice_reflection_interval_s": None,
        "self_model_interval_s": None, "compaction_interval_s": None, "calibration_interval_s": None,
        "interest_sweep_interval_s": None, "judge_selftune_interval_s": None, "clustering_interval_s": None,
        "vocab_repair_interval_s": None, "pending_reconcile_interval_s": None, "maker_enabled": False,
        "notes_enabled": False, "kindled_link_enabled": False,
    }


@pytest.fixture
def pd(tmp_path):
    p = tmp_path / "nell"
    (p / "active_conversations").mkdir(parents=True)
    return p


def _run(pd: Path, monkeypatch, *, saves, stop_after_iterations: int = 3, startup_raises=0):
    """Run the supervised supervisor in a thread. `saves` decides what each soul-cadence save does (a loop block's
    unguarded step): "ok" or "die". Returns (status, startup one-shot count, error logs)."""
    status = sup.SupervisorStatus()
    stop = threading.Event()
    calls = {"save": 0, "startup": 0, "startup_raised": 0}
    monkeypatch.setattr(sup, "_run_soul_review_tick", lambda *a, **k: (0, 1))  # backlog: due again at once
    monkeypatch.setattr(sup.soul_cadence, "compute_next_state",
                        lambda **k: sup.soul_cadence.ReviewCadenceState(next_review_at=None, consecutive_failures=0))

    def _save(persona_dir, state):
        i = calls["save"]
        calls["save"] += 1
        what = saves[i] if i < len(saves) else "ok"
        if what == "die":
            raise RuntimeError(f"loop died at save {i}")

    monkeypatch.setattr(sup.soul_cadence, "save_cadence_state", _save)
    real_reconcile = sup._run_pending_reconcile_tick

    def _startup_spy(persona_dir, *, stale_after=None):
        if stale_after is not None:  # the startup one-shot (the loop's call passes none)
            calls["startup"] += 1
            if calls["startup_raised"] < startup_raises:
                calls["startup_raised"] += 1
                raise RuntimeError("startup died")
        return real_reconcile(persona_dir, stale_after=stale_after) if stale_after is None else None

    monkeypatch.setattr(sup, "_run_pending_reconcile_tick", _startup_spy)
    status.on_iteration = lambda: status.iterations >= stop_after_iterations and stop.set()
    t = threading.Thread(target=lambda: sup.run_supervised(stop, status=status, **_kwargs(pd)), daemon=True)
    t.start()
    t.join(20)
    stop.set()
    t.join(10)
    assert not t.is_alive()
    return status, calls


def test_a_loop_death_restarts_the_loop_once_and_says_so(pd, monkeypatch, caplog):
    with caplog.at_level(logging.ERROR, logger="brain.bridge.supervisor"):
        status, calls = _run(pd, monkeypatch, saves=["die"])
    deaths = [r for r in caplog.records if "supervisor loop died" in r.getMessage()]
    assert len(deaths) == 1 and deaths[0].exc_info is not None
    assert status.state == "stopped" and status.restarts == 1
    assert status.last_error and "loop died at save 0" in status.last_error
    assert calls["startup"] == 1  # the startup one-shots were not re-run


def test_a_second_death_before_an_iteration_is_failed(pd, monkeypatch):
    status, _ = _run(pd, monkeypatch, saves=["die", "die", "die"], stop_after_iterations=99)
    assert status.state == "failed" and status.restarts == 1


def test_a_death_after_a_completed_iteration_restarts_again(pd, monkeypatch):
    status, _ = _run(pd, monkeypatch, saves=["die", "ok", "die"], stop_after_iterations=3)
    assert status.restarts == 2 and status.state == "stopped"


def test_a_startup_death_reruns_the_startup_once(pd, monkeypatch):
    status, calls = _run(pd, monkeypatch, saves=[], startup_raises=1)
    assert calls["startup"] == 2 and status.state == "stopped" and status.restarts == 1


def test_a_second_startup_death_is_failed(pd, monkeypatch):
    status, calls = _run(pd, monkeypatch, saves=[], startup_raises=2)
    assert calls["startup"] == 2 and status.state == "failed"


def test_a_requested_stop_is_never_a_death(pd, monkeypatch):
    status = sup.SupervisorStatus()
    stop = threading.Event()

    def _save(persona_dir, state):
        stop.set()
        raise RuntimeError("raised while stopping")

    monkeypatch.setattr(sup, "_run_soul_review_tick", lambda *a, **k: (0, 0))
    monkeypatch.setattr(sup.soul_cadence, "save_cadence_state", _save)
    sup.run_supervised(stop, status=status, **_kwargs(pd))
    assert status.state == "stopped" and status.restarts == 0


def test_invalid_utf8_cadence_files_and_a_failing_save_do_not_end_the_loop(pd, monkeypatch):
    """C26."""
    from brain.paths import cadence_state_path

    for name in ("soul_review_state.json", "log_rotation_cadence.json", "voice_reflection_cadence.json",
                 "vocab_repair_cadence.json", "pending_reconcile_cadence.json"):
        path = cadence_state_path(pd, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xfe\xfa not utf-8")
    status = sup.SupervisorStatus()
    stop = threading.Event()
    monkeypatch.setattr(sup, "_run_soul_review_tick", lambda *a, **k: (0, 1))

    def _save(persona_dir, state):
        raise OSError("disk full")

    monkeypatch.setattr(sup.soul_cadence, "save_cadence_state", _save)
    status.on_iteration = lambda: status.iterations >= 3 and stop.set()
    kw = _kwargs(pd)
    kw.update(log_rotation_interval_s=3600.0, voice_reflection_interval_s=3600.0, vocab_repair_interval_s=3600.0,
              pending_reconcile_interval_s=3600.0)
    monkeypatch.setattr(sup, "_run_log_rotation_tick", lambda *a, **k: None)
    monkeypatch.setattr(sup, "_run_voice_reflection_tick", lambda *a, **k: None)
    monkeypatch.setattr(sup, "_run_vocab_repair_tick", lambda *a, **k: None)
    t = threading.Thread(target=lambda: sup.run_supervised(stop, status=status, **kw), daemon=True)
    t.start()
    t.join(20)
    stop.set()
    t.join(10)
    assert status.iterations >= 3 and status.restarts == 0 and status.state == "stopped"


def test_health_and_persona_state_show_the_supervisor_status(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from brain.bridge.server import build_app

    pd = tmp_path / "nell"
    (pd / "active_conversations").mkdir(parents=True)
    (pd / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}', encoding="utf-8")
    started = threading.Event()

    def fake_supervised(stop_event, *, status, **kw):
        status.state = "restarted"
        status.restarts = 1
        status.last_error = "RuntimeError: boom"
        started.set()
        stop_event.wait()
        status.state = "stopped"

    monkeypatch.setattr(sup, "run_supervised", fake_supervised)
    with TestClient(build_app(persona_dir=pd, client_origin="tests", background_threads=True)) as c:
        assert started.wait(10)
        h = c.get("/health").json()["supervisor"]
        assert h["state"] == "restarted" and h["restarts"] == 1 and "boom" in h["last_error"]
        ps = c.get("/persona/state").json()["supervisor"]
        assert ps["state"] == "restarted"
