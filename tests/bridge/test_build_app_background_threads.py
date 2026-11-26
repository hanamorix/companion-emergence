"""build_app must not start its background threads inside tests unless asked (C1/C2).

Diagnosis: hunts/bridge-order-pollution-flakes/diagnosis.md — the supervisor thread that
build_app's lifespan starts raced every bridge endpoint test's seeded session.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain.bridge import server

_BG = {"sp7-supervisor", "compaction-backlog-migration"}


def _thread_names() -> set[str]:
    return {t.name for t in threading.enumerate()}


def test_default_in_tests_starts_no_background_threads(persona_dir: Path, caplog) -> None:
    import logging

    app = server.build_app(persona_dir=persona_dir, client_origin="tests")
    with caplog.at_level(logging.WARNING, logger="brain.bridge.server"):
        with TestClient(app) as c:
            assert not (_thread_names() & _BG), _thread_names() & _BG
            assert app.state.bridge.supervisor_thread is None
            assert app.state.bridge.migration_thread is None
            assert c.get("/health").json()["supervisor_thread"] == "not-started"
    # The one operator-visible artifact of threads-off is the WARNING line (stage-6 F-1).
    assert any("background threads OFF" in r.getMessage() for r in caplog.records)


def test_flag_off_starts_both_threads(persona_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production value of the flag (False) + kwarg None → both threads run, as runner.py sees it."""
    monkeypatch.setattr(server, "_background_threads_inhibited", False)
    app = server.build_app(persona_dir=persona_dir, client_origin="tests")
    with TestClient(app):
        assert "sp7-supervisor" in _thread_names()
        assert app.state.bridge.supervisor_thread.is_alive()
        assert app.state.bridge.migration_thread is not None  # started (daemon; may already be done)
    assert not app.state.bridge.supervisor_thread.is_alive()  # joined on lifespan exit


def test_kwarg_true_overrides_inhibit(persona_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_background_threads_inhibited", True)
    app = server.build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)
    with TestClient(app):
        assert "sp7-supervisor" in _thread_names()
        assert app.state.bridge.migration_thread is not None


def test_a_stray_test_hooks_env_var_does_not_stub_a_real_bridge(persona_dir: Path, monkeypatch) -> None:
    """Pre-merge review O-5: the #456 stop tests' hooks (a stub supervisor, stub tasks, no other background threads)
    are installed only in-process by the test launcher, never by an environment variable, so a stray
    KINDLED_TEST_HOOKS=1 cannot start a bridge with no supervisor."""
    from brain.bridge import supervisor as supervisor_mod

    monkeypatch.setenv("KINDLED_TEST_HOOKS", "1")
    monkeypatch.setenv("KINDLED_TEST_STUB_TASKS", "0")
    ran: list[str] = []

    def fake_run_supervised(stop_event, **_kw):
        ran.append("supervisor")
        stop_event.wait()

    monkeypatch.setattr(supervisor_mod, "run_supervised", fake_run_supervised)
    app = server.build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)
    with TestClient(app):
        assert app.state.bridge.migration_thread is not None
        deadline = time.monotonic() + 10
        while not ran and time.monotonic() < deadline:
            time.sleep(0.02)
    assert ran == ["supervisor"]


def test_the_lifespan_never_calls_the_test_registry_reset(persona_dir: Path, monkeypatch) -> None:
    """Pre-merge review O-5 and its fix-check: ``shutdown_progress.reset_for_tests()`` is test code and the production
    lifespan never calls it, not even after an earlier app in the process sealed the registry. Tests get a fresh
    registry from the root conftest (per test) or call the reset themselves between two apps."""
    from brain.bridge import shutdown_progress

    fresh = shutdown_progress.reset_for_tests()
    fresh.seal()  # as an earlier app's stop leaves it
    calls: list[int] = []
    monkeypatch.setattr(shutdown_progress, "reset_for_tests", lambda: calls.append(1))
    app = server.build_app(persona_dir=persona_dir, client_origin="tests")
    try:
        with TestClient(app):
            pass
    except Exception:  # noqa: BLE001 - a sealed registry may refuse the app's own work; only the reset is asserted
        pass
    assert calls == []
