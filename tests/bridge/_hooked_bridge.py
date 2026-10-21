"""Test-only bridge launcher for the real-process stop tests (#456; spec section 8's test-only hooks).

The tests set what they want in the environment; this module turns it into the in-process seam
``brain.bridge.server._lifespan_test_hooks`` before the bridge starts. A production bridge never reads these variables
(pre-merge review O-5). With the hooks installed:

- the supervisor is a stub that, after the stop, waits for ``KINDLED_TEST_SUPERVISOR_HOLD`` (a file path) to exist;
- ``KINDLED_TEST_STUB_TASKS`` counted stub tasks each end when ``<KINDLED_TEST_STUB_DIR>/release-<i>`` exists;
- ``KINDLED_TEST_CALLS`` ("module:function,...") runs each function with the persona dir as a counted task (a call
  in flight during the stop);
- no other background thread starts.

Usage: ``python -m tests.bridge._hooked_bridge <runner args>`` (the runner, as ``python -m brain.bridge.runner``), or
``import tests.bridge._hooked_bridge as h; h.install()`` before ``daemon.cmd_run`` in a ``python -c`` snippet.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _wait_for_file(path: str | None) -> None:
    while path and not Path(path).exists():
        time.sleep(0.05)


def _run_call(spec: str, persona_dir: Path) -> None:
    module, _, name = spec.partition(":")
    try:
        getattr(importlib.import_module(module), name)(persona_dir)
    except Exception:  # noqa: BLE001 - a test hook never takes the bridge down
        logger.exception("test hook call %s raised", spec)


def install() -> None:
    from brain.bridge import server, shutdown_progress

    hold = os.environ.get("KINDLED_TEST_SUPERVISOR_HOLD") or None
    tasks = int(os.environ.get("KINDLED_TEST_STUB_TASKS") or 0)
    stub_dir = os.environ.get("KINDLED_TEST_STUB_DIR") or None
    calls = [c for c in (os.environ.get("KINDLED_TEST_CALLS") or "").split(",") if c]

    def supervisor(stop_event) -> None:
        stop_event.wait()
        _wait_for_file(hold)

    def start_tasks(persona_dir: Path) -> None:
        for i in range(tasks):
            release = str(Path(stub_dir) / f"release-{i}") if stub_dir else None
            shutdown_progress.start_tracked_thread(
                f"stub task {i}", lambda r=release: _wait_for_file(r), name=f"stub-task-{i}"
            )
        for spec in calls:
            shutdown_progress.start_tracked_thread(
                spec.rpartition(":")[2], lambda sp=spec: _run_call(sp, persona_dir), name=f"call-{spec}"
            )

    server._lifespan_test_hooks = {"supervisor": supervisor, "start_tasks": start_tasks}


def main(argv: list[str]) -> int:
    install()
    from brain.bridge import runner

    return runner.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
