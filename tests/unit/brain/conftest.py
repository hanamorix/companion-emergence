"""TEMPORARY CI diagnostic plugin (reverted right after one manual run). Not part of the PR."""
from __future__ import annotations

import sys
import threading
import time
import traceback

import pytest

_DUMPS: list[tuple[str, str]] = []
_SLOW: list[tuple[float, str]] = []


def _sample(nodeid: str, started: float, done: threading.Event) -> None:
    if done.wait(1.5):
        return
    samples = []
    me = threading.get_ident()
    for _ in range(3):
        parts = [f"@{time.monotonic() - started:.1f}s"]
        for tid, frame in sys._current_frames().items():
            if tid == me:
                continue
            stack = traceback.extract_stack(frame)[-6:]
            parts.append(
                f"  thread {tid}: " + " <- ".join(f"{s.filename.replace(chr(92), '/').split('companion-emergence/')[-1][-45:]}:{s.lineno}:{s.name}" for s in reversed(stack))
            )
        samples.append("\n".join(parts))
        if done.wait(1.0):
            break
    _DUMPS.append((nodeid, "\n".join(samples)))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    done = threading.Event()
    th = threading.Thread(target=_sample, args=(item.nodeid, time.monotonic(), done), daemon=True)
    th.start()
    try:
        yield
    finally:
        done.set()
        th.join(timeout=2)


def pytest_runtest_logreport(report):
    if report.duration > 1.5:
        _SLOW.append((report.duration, f"{report.nodeid} [{report.when}]"))


def pytest_terminal_summary(terminalreporter):
    w = terminalreporter.write_line
    w("=== DIAG: slow phases (>1.5s) ===")
    for d, n in sorted(_SLOW, reverse=True)[:60]:
        w(f"{d:7.2f} {n}")
    w(f"=== DIAG: {len(_DUMPS)} stall samples ===")
    for nodeid, text in _DUMPS[:25]:
        w(f"--- {nodeid}")
        w(text[:3000])
