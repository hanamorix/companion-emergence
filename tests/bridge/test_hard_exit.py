"""The hard exit (C14 a, b, e, f, g, i; D17).

A subprocess stands in for the bridge: it owns a bridge.json record, has a child with a grandchild, and calls
`shutdown.hard_exit`. The hard exit sets the abandoning flag before its first kill, kills every descendant (and, on a
second enumeration, one that started after the first), never waits on a child, leaves bridge.json not clean with reason
"hard_exit", and exits 0 even when the record write raises.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest import mock

import psutil
import pytest

from brain.bridge import cli_throttle, shutdown_progress, state_file

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process tree; Windows leg in CI")

REPO = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import json, os, subprocess, sys, time
    from pathlib import Path
    import psutil
    from brain import process_identity as pi
    from brain.bridge import shutdown, shutdown_progress, state_file

    pd, log, raise_update = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"

    def note(line):
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\\n")

    state_file.write(pd, state_file.BridgeState(
        persona="nell", pid=os.getpid(), port=1, started_at="t", stopped_at=None, shutdown_clean=False,
        client_origin="tests", pid_token=pi.own_token(), identity="token", phase="stopping", stop_reason="user_stop"))
    child = subprocess.Popen([sys.executable, "-c",
        "import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)', "
        "'brain.mcp_server']); print(g.pid, flush=True); time.sleep(600)", "claude"], stdout=subprocess.PIPE, text=True)
    grand = int(child.stdout.readline())
    note(f"pids {child.pid} {grand}")

    real_kill = psutil.Process.kill
    def spy_kill(self):
        note(f"kill {self.pid} abandoning={shutdown_progress.abandoning.is_set()}")
        return real_kill(self)
    psutil.Process.kill = spy_kill
    for name in ("wait",):
        setattr(psutil.Process, name, lambda self, *a, **k: note("WAIT psutil") )
    psutil.wait_procs = lambda *a, **k: note("WAIT wait_procs")
    subprocess.Popen.wait = lambda self, *a, **k: note("WAIT Popen")

    def late():
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "late-claude"])
        note(f"late {p.pid}")
    shutdown._between_enumerations = late
    if raise_update:
        def boom(*a, **k):
            raise RuntimeError("disk on fire")
        state_file.update = boom
    note("calling")
    shutdown.hard_exit(pd, "hard_exit")
    note("RETURNED")
    """
)


def _run(tmp_path: Path, raise_update: bool = False):
    pd = tmp_path / "nell"
    pd.mkdir()
    log = tmp_path / "log"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    err = tmp_path / "stderr"
    with open(err, "w", encoding="utf-8") as ef:  # a file, not a pipe: a surviving child must not hold the test
        proc = subprocess.run([sys.executable, "-c", _SCRIPT, str(pd), str(log), "1" if raise_update else "0"],
                              env=env, timeout=60, stdout=subprocess.DEVNULL, stderr=ef, text=True)
    proc.stderr = err.read_text(encoding="utf-8")
    lines = log.read_text(encoding="utf-8").splitlines()
    return proc, pd, lines


@pytest.fixture
def reap():
    """Kill any test process left alive, AFTER the test's own checks (a survivor is a failure, not a leak)."""
    seen: list[list[str]] = []
    yield seen
    for lines in seen:
        for line in lines:
            if line.startswith(("pids", "late")):
                for pid in line.split()[1:]:
                    try:
                        psutil.Process(int(pid)).kill()
                    except (psutil.NoSuchProcess, ValueError):
                        pass


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def test_hard_exit_abandons_first_kills_the_tree_never_waits_and_exits_0(tmp_path, reap):
    proc, pd, lines = _run(tmp_path)
    reap.append(lines)
    assert proc.returncode == 0, proc.stderr
    assert "RETURNED" not in lines
    _, child, grand = next(line for line in lines if line.startswith("pids")).split()
    late = next(line for line in lines if line.startswith("late")).split()[1]
    kills = [line for line in lines if line.startswith("kill")]
    assert kills and all(line.endswith("abandoning=True") for line in kills)  # (a)
    killed = {line.split()[1] for line in kills}
    assert {child, grand, late} <= killed  # (b), (i)
    for pid in (child, grand, late):
        assert _gone(int(pid)), pid
    assert not [line for line in lines if line.startswith("WAIT")]  # (e)
    st = state_file.read(pd)
    assert st.shutdown_clean is False and st.phase == "hard_exit" and st.stop_reason == "hard_exit"  # (c)
    rec = shutdown_progress.read_record(pd, shutdown_progress.PROGRESS_FILENAME)
    assert rec is not None and rec.get("phase") == "hard_exit"


def test_hard_exit_still_exits_0_when_the_record_write_raises(tmp_path, reap):
    """C14 (g)."""
    proc, pd, lines = _run(tmp_path, raise_update=True)
    reap.append(lines)
    assert proc.returncode == 0, proc.stderr
    _, child, grand = next(line for line in lines if line.startswith("pids")).split()
    assert _gone(int(child)) and _gone(int(grand))


@pytest.mark.parametrize("site", ["generate", "chat"])
def test_while_abandoning_no_claude_is_spawned(site):
    """C14 (f): a call attempted after the abandoning flag spawns nothing (it blocks until the process exits)."""
    from brain.bridge.provider import ChatMessage, ClaudeCliProvider

    cli_throttle.reset()
    shutdown_progress.abandoning.set()
    try:
        with mock.patch.object(subprocess, "run") as run, mock.patch.object(subprocess, "Popen") as popen:
            def call():
                if site == "generate":
                    ClaudeCliProvider().generate("hi")
                else:
                    ClaudeCliProvider().chat([ChatMessage(role="user", content="hi")])

            t = threading.Thread(target=call, daemon=True)
            t.start()
            t.join(0.5)
            assert t.is_alive()  # blocked at the gate, not returned and not raised
            assert not run.called and not popen.called
    finally:
        shutdown_progress.abandoning.clear()


def test_abandoning_is_cleared_for_a_new_bridge_app():
    shutdown_progress.abandoning.set()
    shutdown_progress.reset_for_tests()
    assert not shutdown_progress.abandoning.is_set()

