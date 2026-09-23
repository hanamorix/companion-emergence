"""Harness unit tests: one live-companion-service verdict per session (#264).

``sandbox()`` runs a live-service pre-check on every entry (``live_check="raise"`` by default),
which is right for a live run but wrong in shape for this directory: with the launchd / systemd /
Task Scheduler companion service running (the normal state after following the install docs), every
test that enters the sandbox failed on its own with the same ``LiveServiceDetected`` message, and the
offline suite read as a 42-test regression.

This conftest asks the same scanner (``_live_bridges`` on the real engine home) ONCE per session.
When it finds a live bridge, each test's own ``sandbox()`` call is what decides the skip: the
pre-check runs unchanged, and only a ``LiveServiceDetected`` raised from scanning the REAL engine
home (the one the session scan looked at) becomes a skip with a short reason. A call with
``live_check="off"`` or ``"warn"`` never raises, so it never skips; a test that points the engine
resolver at a ``tmp_path`` home and seeds its own bridge (``test_live_service_precheck.py``) gets
its ``LiveServiceDetected`` re-raised, so its ``pytest.raises`` still sees it. The full message
prints once in the terminal summary. The guard inside ``sandbox()`` is untouched (it still raises
for direct callers and live runs); only the reporting shape changes.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from brain.paths import get_home
from tests.harness.sandbox import LiveServiceDetected, _live_bridges

# ``tests.harness.__init__`` re-exports the ``sandbox`` FUNCTION, which shadows the submodule
# attribute on the package — fetch the real module object for the monkeypatch below.
_sandbox_mod = importlib.import_module("tests.harness.sandbox")

_SKIP_REASON = "live companion service detected (#264; see the terminal summary)"

_LIVE_KEY = pytest.StashKey[list[tuple[int, str]]]()


@pytest.fixture(scope="session")
def _live_companion_services(request: pytest.FixtureRequest) -> list[tuple[int, str]]:
    """The live-bridge scan, computed once for the whole session (empty = none live)."""
    live = _live_bridges()
    if live:
        request.config.stash[_LIVE_KEY] = live
    return live


@pytest.fixture(scope="session")
def _real_engine_home() -> Path | None:
    """The engine home the session scan looked at, captured before any test patches the resolver."""
    try:
        return Path(get_home()).resolve()
    except Exception:  # noqa: BLE001 — a broken resolver must not break the suite
        return None


@pytest.fixture(autouse=True)
def _skip_when_the_live_check_hits_the_real_home(
    monkeypatch: pytest.MonkeyPatch,
    _live_companion_services: list[tuple[int, str]],
    _real_engine_home: Path | None,
) -> None:
    if not _live_companion_services:
        return

    original = _sandbox_mod._run_live_check

    def scans_real_home(engine_home: Path | None) -> bool:
        if engine_home is None or _real_engine_home is None:
            return True  # ``None`` ⇒ the scanner resolves the real home itself
        return Path(engine_home).resolve() == _real_engine_home

    def guarded(policy, snapshot_fn, *, engine_home: Path | None = None, **kwargs):
        try:
            return original(policy, snapshot_fn, engine_home=engine_home, **kwargs)
        except LiveServiceDetected:
            if scans_real_home(engine_home):
                pytest.skip(_SKIP_REASON)
            raise

    monkeypatch.setattr(_sandbox_mod, "_run_live_check", guarded)


def pytest_terminal_summary(terminalreporter, config: pytest.Config) -> None:
    live = config.stash.get(_LIVE_KEY, None)
    if not live:
        return
    listed = ", ".join(f"pid {pid} (persona {name})" for pid, name in live)
    terminalreporter.section("live companion service (#264)", sep="-", yellow=True)
    terminalreporter.write_line(
        "harness sandbox tests whose live-service pre-check hit the real engine home were "
        f"skipped: a live companion service was detected at session start: {listed}. Quit your "
        "companion bridge (and any launchd/systemd/task-scheduler service) before running the "
        "harness tests, then retry."
    )
