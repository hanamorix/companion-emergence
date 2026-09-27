"""Bug 2a (v0.0.36): cmd_start must NOT hold the per-persona lockfile across the
detached spawn — the lock belongs to the bridge process (the runner) for its
lifetime. If cmd_start holds it, the child runner can't acquire it and dies.
The runner owns the lock now (test_runner_mutual_exclusion); cmd_start keeps
only the cheap is_running pre-check."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import brain.bridge.daemon as daemon
from brain.bridge import state_file


def test_cmd_start_leaves_lock_free_for_child_runner(tmp_path, monkeypatch):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)
    monkeypatch.setattr(state_file, "is_running", lambda pd: False)

    seen = {}

    def _fake_spawn(pd, idle, origin, log):
        # The bridge process (child) is about to start and will acquire the lock.
        # cmd_start must NOT be holding it, or the child can never bind. The
        # lock FILE itself now persists across acquire/release (S56 — never
        # unlinked), so "free" means "not OS-locked", checked by taking it
        # ourselves and releasing again, not by the file's mere existence.
        probe_fd = daemon.acquire_lock(pd)
        seen["lock_free_at_spawn"] = probe_fd is not None
        if probe_fd is not None:
            daemon.release_lock(pd, probe_fd)
        return 4242

    monkeypatch.setattr(daemon, "spawn_detached", _fake_spawn)
    monkeypatch.setattr(daemon, "run_recovery_if_needed", lambda pd: None)

    # Make the readiness probe succeed immediately so cmd_start returns fast.
    ready = SimpleNamespace(pid=4242, port=51999, auth_token="t", shutdown_clean=True)
    monkeypatch.setattr(state_file, "read", lambda pd: ready)

    class _Resp:
        status_code = 200

    monkeypatch.setattr(daemon.httpx, "get", lambda *a, **k: _Resp())

    args = SimpleNamespace(persona="p", idle_shutdown=0, client_origin="task-scheduler")
    rc = daemon.cmd_start(args)
    assert rc == 0
    assert seen["lock_free_at_spawn"] is True


_WORKER_SCRIPT = str(Path(__file__).parent / "_runner_lock_worker.py")


def test_c19a_two_cmd_starts_forced_into_handoff_window(tmp_path, monkeypatch):
    """C19(a): two `cmd_start` runs forced into the handoff window — the
    second launched after the first parent released its own pre-flight lock
    and before its (deliberately delayed) child acquires the real one.
    Exactly one bridge ends up running and holding the lock, with one
    bridge.json-equivalent state write and one bound "port"; the loser's
    real child process exits 2 (the runner.main-level exclusion S57 names)
    having never reached that point; the lock file is never unlinked and its
    inode is unchanged across the whole race.

    Real OS processes throughout (sys.executable + subprocess, no fork-only
    API — see `_runner_lock_worker.py`), so this runs unmodified on the
    Windows/macOS CI runners too. One intentional real-time cost: the LOSING
    side's own `cmd_start` call runs its genuine ~50s /health readiness poll
    to its natural timeout (nothing here mocks `time.time()`, since the
    winning side's poll loop shares that same module-level clock and must
    not be corrupted) — bounded by a generous thread-join timeout below, so
    nothing is left running past this test.

    Timing constants below are deliberately generous (real Windows CI, 2026-
    09-26: this test's original 0.3s/2.0s margins were tight enough that
    role-b's `cmd_start` sometimes hadn't resolved within 15s — Windows
    process creation plus importing brain.bridge.runner's own dependency
    chain in a fresh interpreter is meaningfully slower there than on
    Linux/macOS). The delay differential (role-a's child sleeps far longer
    than the inter-thread-start gap) is what actually proves the race, not
    the absolute values, so widening both is free: it costs real wall time
    only on the slow platform that needs it, and this test already pays a
    genuine ~50s floor regardless (role-a's own `cmd_start` readiness
    timeout), so a few more seconds of slack elsewhere is negligible by
    comparison.
    """
    handoff_gap_s = 1.5  # gap between starting role-a's and role-b's cmd_start
    role_a_child_delay_s = 8.0  # role-a's child's startup-delay before it even tries the lock
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)
    monkeypatch.setattr(state_file, "is_running", lambda pd: False)
    monkeypatch.setattr(daemon, "run_recovery_if_needed", lambda pd: None)

    # No real uvicorn server ever runs; only the HTTP hop of cmd_start's
    # readiness probe is faked. The winner's real child still has to write a
    # REAL, matching state_file (via --write-state-port below) for that
    # probe to even be reached with the right pid/port.
    class _HealthyResp:
        status_code = 200

    monkeypatch.setattr(daemon.httpx, "get", lambda *a, **kw: _HealthyResp())

    stop_file = tmp_path / "stop"
    launched: dict[str, subprocess.Popen] = {}
    launch_lock = threading.Lock()

    def fake_spawn_detached(pd, idle, origin, log_path):
        # `origin` == this call's client_origin, abused purely as an
        # in-test signal of which role (see _run below) is spawning —
        # cmd_start itself only reads client_origin for the (irrelevant
        # here) launchd log-truncation branch.
        delayed = origin == "role-a"
        marker = tmp_path / f"marker-{origin}.txt"
        proc = subprocess.Popen(
            [
                sys.executable,
                _WORKER_SCRIPT,
                "--persona-dir",
                str(pd),
                "--marker",
                str(marker),
                "--stop-file",
                str(stop_file),
                "--startup-delay",
                str(role_a_child_delay_s) if delayed else "0.0",
                "--write-state-port",
                "51900" if origin == "role-a" else "51901",
            ]
        )
        with launch_lock:
            launched[origin] = proc
        return proc.pid

    monkeypatch.setattr(daemon, "spawn_detached", fake_spawn_detached)

    results: dict[str, int] = {}

    def _run(role: str) -> None:
        args = SimpleNamespace(persona="persona", idle_shutdown=0, client_origin=role)
        results[role] = daemon.cmd_start(args)

    lock_path = persona_dir / daemon.LOCKFILE
    t_a = threading.Thread(target=_run, args=("role-a",))
    t_b = threading.Thread(target=_run, args=("role-b",))

    try:
        # Force the handoff window: role-a's cmd_start runs first (its own
        # pre-flight acquire+release completes near-instantly, well before
        # its DELAYED child ever tries the real lock); role-b's cmd_start
        # starts an instant later, sees the OS lock free (role-a already
        # released its own pre-flight hold before spawning), and its
        # non-delayed child reaches the real lock first and wins.
        t_a.start()
        time.sleep(handoff_gap_s)
        t_b.start()

        # role-b's cmd_start should resolve fast (its child wins the race
        # and writes a matching state_file almost immediately) — generous
        # bound for slow Windows process/import startup, not because this
        # side is expected to actually take that long.
        t_b.join(timeout=30.0)
        assert not t_b.is_alive(), "role-b's cmd_start should have returned quickly"
        assert results["role-b"] == 0

        winner_marker = tmp_path / "marker-role-b.txt"
        loser_marker = tmp_path / "marker-role-a.txt"
        assert winner_marker.exists()
        assert not loser_marker.exists(), "the losing child must never reach run_bridge_foreground"

        inode_while_held = lock_path.stat().st_ino

        # Wait for role-a's DELAYED child to finish its own attempt (its
        # role_a_child_delay_s startup delay, then a real acquire_lock
        # against the winner STILL holding the lock at this point —
        # stop_file is not written until after this) before releasing the
        # winner. If the winner were released first, role-a's late child
        # would find the lock free and win trivially, proving nothing about
        # the actual race.
        assert launched["role-a"].wait(timeout=role_a_child_delay_s + 20.0) == 2
        assert not loser_marker.exists()

        # Now release the winner so its cmd_start's spawned child (still
        # parked in the stub) can exit cleanly.
        stop_file.write_text("go", encoding="utf-8")
        assert launched["role-b"].wait(timeout=30.0) == 0

        # role-a's cmd_start is still spinning its genuine ~50s readiness
        # poll waiting for a state_file match that will never come (its own
        # child already lost and exited above) — bounded here rather than
        # mocked (see docstring). Generous margin above the fixed ~50s floor
        # for slow-Windows overhead on top of it.
        t_a.join(timeout=90.0)
        assert not t_a.is_alive(), "role-a's cmd_start must give up within its own ~50s deadline"
        assert results["role-a"] == 1  # readiness timeout -> orphan-kill path

        assert lock_path.exists()  # S56: never unlinked
        assert lock_path.stat().st_ino == inode_while_held
    finally:
        stop_file.write_text("go", encoding="utf-8")
        for proc in launched.values():
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10.0)
        for t in (t_a, t_b):
            t.join(timeout=5.0)
