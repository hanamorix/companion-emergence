"""Standalone worker process for C19 multi-PROCESS bridge-lock tests.

NOT a pytest test module (no `test_*` name — pytest's `python_files` glob
skips it) and not meant to be imported; it is invoked as a real subprocess
via ``sys.executable <this file> --persona-dir ... --marker ... --stop-file
...`` by ``tests/bridge/test_runner_mutual_exclusion.py`` and
``tests/bridge/test_cmd_start_lock_handoff.py``.

Exercises the REAL ``brain.bridge.runner.main()`` entrypoint end to end —
including its real ``is_running`` check and its real
``brain.bridge.daemon.acquire_lock``/``release_lock`` OS-level lock — so
these tests prove S57's actual exclusive point (a losing child exits before
writing bridge.json or binding a port) across real OS processes, not a
same-process thread proxy. The only thing stubbed is
``run_bridge_foreground`` itself: no real persona, model load, uvicorn
server, or network I/O happens. The stub is reached ONLY by whichever
process actually wins the OS lock (that is what "won" means here), so:

* it writes ``--marker`` (this process's own pid) the instant it is called,
  standing in for "wrote bridge.json / bound a port" without doing either
  — proving the LOSING sibling process never reaches this point at all;
* it then blocks (polling for ``--stop-file`` to appear) so the test can
  observe "exactly one process is holding the lock" from outside before
  releasing it.

``--startup-delay`` sleeps before calling ``runner.main`` at all, so a test
can force one sibling to arrive at ``acquire_lock`` well after another.

``--write-state-port PORT`` additionally makes the winner write a REAL
``brain.bridge.state_file.BridgeState`` (this process's own pid, that port)
the instant it wins — used by the ``cmd_start`` handoff test so `cmd_start`'s
own genuine readiness poll (which reads state_file + probes ``/health``) can
observe a real, matching pid/port for the winning side without a real
uvicorn server ever running.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona-dir", required=True)
    p.add_argument("--marker", required=True)
    p.add_argument("--stop-file", required=True)
    p.add_argument("--startup-delay", type=float, default=0.0)
    p.add_argument("--write-state-port", type=int, default=None)
    args = p.parse_args()

    if args.startup_delay > 0:
        time.sleep(args.startup_delay)

    from brain.bridge import runner

    def fake_run_bridge_foreground(persona_dir, *, client_origin="cli", idle_shutdown_seconds=None):
        # Reached only by the sibling that actually won runner.main's
        # acquire_lock (see module docstring). Standing in for "wrote
        # bridge.json / bound a port" without doing either.
        Path(args.marker).write_text(str(os.getpid()), encoding="utf-8")
        if args.write_state_port is not None:
            from brain.bridge import state_file

            state_file.write(
                Path(persona_dir),
                state_file.BridgeState(
                    persona=Path(persona_dir).name,
                    pid=os.getpid(),
                    port=args.write_state_port,
                    started_at=datetime.now(UTC).isoformat(),
                    stopped_at=None,
                    shutdown_clean=False,
                    client_origin=client_origin,
                ),
            )
        stop_path = Path(args.stop_file)
        deadline = time.time() + 30.0
        while not stop_path.exists() and time.time() < deadline:
            time.sleep(0.05)
        return 0

    runner.run_bridge_foreground = fake_run_bridge_foreground
    return runner.main(["--persona-dir", args.persona_dir])


if __name__ == "__main__":
    sys.exit(main())
