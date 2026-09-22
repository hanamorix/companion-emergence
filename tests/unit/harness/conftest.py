"""Harness unit tests: one live-companion-service verdict per session (#264).

``sandbox()`` runs a live-service pre-check on every entry (``live_check="raise"`` by default),
which is right for a live run but wrong in shape for this directory: with the launchd / systemd /
Task Scheduler companion service running (the normal state after following the install docs), every
test that enters the sandbox failed on its own with the same ``LiveServiceDetected`` message, and the
offline suite read as a 42-test regression.

This conftest asks the same scanner (``_live_bridges`` on the real engine home) ONCE per session,
skips every test in a module that uses ``sandbox`` with a short reason, and prints the full message
once in the terminal summary. The guard inside ``sandbox()`` is untouched (it still raises for direct
callers and live runs); only the reporting shape changes. Modules here that never enter the sandbox
keep running.
"""

from __future__ import annotations

import pytest

from tests.harness.sandbox import _live_bridges

_SKIP_REASON = "live companion service detected (#264; see the terminal summary)"


@pytest.fixture(scope="session")
def _live_companion_services(request: pytest.FixtureRequest) -> list[tuple[int, str]]:
    """The live-bridge scan, computed once for the whole session (empty = none live)."""
    live = _live_bridges()
    if live:
        request.config.stash[_LIVE_KEY] = live
    return live


_LIVE_KEY = pytest.StashKey[list[tuple[int, str]]]()


@pytest.fixture(autouse=True)
def _skip_sandbox_tests_when_a_companion_is_live(
    request: pytest.FixtureRequest, _live_companion_services: list[tuple[int, str]]
) -> None:
    if not _live_companion_services:
        return
    if getattr(request.module, "sandbox", None) is None:
        return
    pytest.skip(_SKIP_REASON)


def pytest_terminal_summary(terminalreporter, config: pytest.Config) -> None:
    live = config.stash.get(_LIVE_KEY, None)
    if not live:
        return
    listed = ", ".join(f"pid {pid} (persona {name})" for pid, name in live)
    terminalreporter.section("live companion service (#264)", sep="-", yellow=True)
    terminalreporter.write_line(
        "harness sandbox tests were skipped: a live companion service was detected at "
        f"session start: {listed}. Quit your companion bridge (and any launchd/systemd/"
        "task-scheduler service) before running the harness tests, then retry."
    )
