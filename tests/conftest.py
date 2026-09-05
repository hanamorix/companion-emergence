"""Shared pytest fixtures and configuration for companion-emergence tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from brain.bridge import cli_throttle
from brain.chat import pass2_queue


@pytest.fixture(autouse=True)
def _reset_cli_throttle() -> Iterator[None]:
    """Reset cli_throttle global state before each test.

    background_slot() reads process-global monotonic timestamps. Without
    this reset, a test that calls mark_interactive_active() contaminates
    subsequent tests in the same process — causing background-engine calls
    to be gated when the test expects them to fire.
    """
    cli_throttle.reset()
    yield
    cli_throttle.reset()


@pytest.fixture(autouse=True)
def _inhibit_bridge_background_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep build_app's lifespan from starting the supervisor + migration threads.

    Every bridge endpoint test enters build_app's lifespan; the real supervisor
    thread it started ran a startup catch-up compaction that rolled over the
    months-old session the test had just seeded and deleted its buffer, racing
    the test's own request (hunts/bridge-order-pollution-flakes/diagnosis.md —
    #155, #161, and the c8 / WinError-32 CI reds). Mirrors _reset_pass2_queue:
    tests that genuinely exercise those threads pass background_threads=True.
    """
    from brain.bridge import server

    # raising=False: a bisect onto a pre-flag commit must not error every test at setup.
    monkeypatch.setattr(server, "_background_threads_inhibited", True, raising=False)


@pytest.fixture(autouse=True)
def _reset_pass2_queue() -> Iterator[None]:
    """Reset pass2_queue global state before and after each test.

    Mirrors _reset_cli_throttle.  Without this, tests that call enqueue()
    leave items in the queue (or a running worker thread) that contaminate
    subsequent tests in the same process.

    Also inhibits the daemon worker so enqueue() never spawns a background
    thread during tests — tests drive drain_pending() synchronously, and a
    live worker would race those drains.
    """
    pass2_queue._worker_inhibited = True
    pass2_queue.reset()
    yield
    pass2_queue.reset()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Walk upward from this file to find the repo root (pyproject.toml).

    Replaces brittle `Path(__file__).parents[N]` patterns in tests that
    need absolute paths to checked-in resources.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find pyproject.toml above {here}")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove companion-emergence-relevant env vars for isolation.

    Used by tests for brain.paths and brain.config (Tasks 2+).
    Each key here corresponds to an env var the framework reads at runtime.
    """
    for key in [
        "NELLBRAIN_HOME",
        "NELL_IPC_JID",
        "BRIDGE_BIND",
        "PROVIDER",
        "MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield
