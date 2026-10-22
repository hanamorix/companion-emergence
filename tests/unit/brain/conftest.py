"""TEMPORARY CI diagnostic plugin (reverted right after one manual run). Not part of the PR."""
from __future__ import annotations

import faulthandler
import tempfile

import pytest

_DUMPS: list[tuple[str, str]] = []
_SLOW: list[tuple[float, str]] = []
_PHASES: dict[str, dict[str, float]] = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    f = tempfile.TemporaryFile(mode="w+")
    faulthandler.dump_traceback_later(1.5, repeat=False, file=f)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        f.flush()
        f.seek(0)
        text = f.read()
        f.close()
        if text:
            _DUMPS.append((item.nodeid, text))


def pytest_runtest_logreport(report):
    _PHASES.setdefault(report.nodeid, {})[report.when] = report.duration
    if report.duration > 1.5:
        _SLOW.append((report.duration, f"{report.nodeid} [{report.when}]"))


def pytest_terminal_summary(terminalreporter):
    w = terminalreporter.write_line
    w("=== DIAG: slow phases (>1.5s) ===")
    for d, n in sorted(_SLOW, reverse=True)[:40]:
        w(f"{d:7.2f} {n}")
    w(f"=== DIAG: {len(_DUMPS)} stack dumps ===")
    for nodeid, text in _DUMPS[:12]:
        w(f"--- {nodeid}")
        w(text[:2500])
