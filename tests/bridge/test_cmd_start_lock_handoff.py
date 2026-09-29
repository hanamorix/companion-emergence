"""Bug 2a (v0.0.36): cmd_start must NOT hold the per-persona lockfile across the
detached spawn — the lock belongs to the bridge process (the runner) for its
lifetime. If cmd_start holds it, the child runner can't acquire it and dies.
The runner owns the lock now (test_runner_mutual_exclusion); cmd_start keeps
only the cheap is_running pre-check."""
from __future__ import annotations

import subprocess
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


def test_c19a_two_cmd_starts_forced_into_handoff_window(tmp_path, monkeypatch, capsys):
    """C19(a): two `cmd_start` runs forced into the handoff window — the
    second launched after the first parent released its own pre-flight lock
    and before its (deliberately held-back) child acquires the real one.
    Exactly one bridge ends up running and holding the lock, with one
    bridge.json-equivalent state write and one bound "port"; the loser's
    real child process exits 2 (the runner.main-level exclusion S57 names)
    having never reached that point; the lock file is never unlinked and its
    inode is unchanged across the whole race.

    Real OS processes throughout, spawned by the REAL `spawn_detached`
    (interpreter resolution, detach flags, Popen bookkeeping); only the
    child's argv is swapped for `_runner_lock_worker.py` (no fork-only API),
    so this runs unmodified on the Windows/macOS CI runners too.

    The LOSING side's `cmd_start` must report promptly with the S57
    wording and return 2, not wait out its ~50s readiness window: it
    notices its own child already exited (Popen.poll()). Before that fix it
    returned 1 via the 50s timeout/orphan-kill path, which the rc==2 and
    wording asserts below catch.

    Sequencing (2026-09-27 rewrite — see history below): role-a's child is
    held at a `--wait-file` gate BEFORE it imports anything or attempts the
    lock, so it is released ONLY once role-b's child has already proven it
    holds the lock (its marker file has appeared). This makes "role-a's
    child attempts strictly after role-b's child already won" a DETERMINISTIC
    fact this test controls, not a wall-clock guess about whose subprocess
    starts up faster — see the history note for why that guess failed.

    History: an earlier version used a fixed `--startup-delay` on role-a's
    child; it was replaced by the `--wait-file` gate on the theory that the
    windows-latest failure (`role-b's cmd_start should have returned
    quickly`) was a startup-timing race. It was not: the gated version failed
    identically, AFTER role-b's child had provably won (its marker existed).
    Root cause (a production bug, not test timing): on Windows a venv's
    `python.exe` is a redirector that runs the real interpreter as a second
    process, so `Popen(...).pid` (what `spawn_detached` returns and
    `cmd_start` waits for) was the redirector's pid while the winning child
    wrote its own `os.getpid()` into state_file. `cmd_start`'s `s.pid == pid`
    readiness match therefore never succeeded on a Windows venv, and a
    healthy bridge was reported failed after 50s. Fixed in
    `daemon.bridge_python()` (spawn the base interpreter with
    `__PYVENV_LAUNCHER__`, as multiprocessing does); the children below are
    spawned by the real `spawn_detached`, so on windows-latest this test
    fails without the fix.
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)
    # state_file.is_running is REAL here (no persona has a bridge.json until
    # role-b's child writes one), so the loser's report picks the applicable
    # S57 wording from real state.
    monkeypatch.setattr(daemon, "run_recovery_if_needed", lambda pd: None)
    monkeypatch.setattr("brain.paths.get_log_dir", lambda: tmp_path / "logs")

    # No real uvicorn server ever runs; only the HTTP hop of cmd_start's
    # readiness probe is faked. The winner's real child still has to write a
    # REAL, matching state_file (via --write-state-port below) for that
    # probe to even be reached with the right pid/port.
    class _HealthyResp:
        status_code = 200

    monkeypatch.setattr(daemon.httpx, "get", lambda *a, **kw: _HealthyResp())

    stop_file = tmp_path / "stop"
    wait_gate_a = tmp_path / "wait-gate-role-a"  # role-a's child blocks until this exists
    winner_marker = tmp_path / "marker-role-b.txt"
    loser_marker = tmp_path / "marker-role-a.txt"
    launched: dict[str, subprocess.Popen] = {}
    launch_lock = threading.Lock()

    def worker_argv(pd, idle, origin):
        # `origin` == this call's client_origin, abused purely as an
        # in-test signal of which role (see _run below) is spawning —
        # cmd_start itself only reads client_origin for the (irrelevant
        # here) launchd log-truncation branch.
        argv = [
            _WORKER_SCRIPT,
            "--persona-dir",
            str(pd),
            "--marker",
            str(tmp_path / f"marker-{origin}.txt"),
            "--stop-file",
            str(stop_file),
            "--write-state-port",
            "51900" if origin == "role-a" else "51901",
        ]
        if origin == "role-a":
            argv += ["--wait-file", str(wait_gate_a)]
        return argv

    monkeypatch.setattr(daemon, "_runner_argv", worker_argv)
    real_spawn_detached = daemon.spawn_detached

    def spying_spawn_detached(pd, idle, origin, log_path):
        # The real spawn; only records its Popen (the same object cmd_start
        # polls) so the test can check exit codes and clean up.
        pid = real_spawn_detached(pd, idle, origin, log_path)
        with launch_lock:
            launched[origin] = daemon._spawned_children[pid]
        return pid

    monkeypatch.setattr(daemon, "spawn_detached", spying_spawn_detached)

    results: dict[str, int] = {}

    def _run(role: str) -> None:
        args = SimpleNamespace(persona="persona", idle_shutdown=0, client_origin=role)
        results[role] = daemon.cmd_start(args)

    def _poll_until(predicate, *, timeout: float, message: str) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError(message)

    lock_path = persona_dir / daemon.LOCKFILE
    t_a = threading.Thread(target=_run, args=("role-a",))
    t_b = threading.Thread(target=_run, args=("role-b",))

    try:
        # Force the handoff window: role-a's cmd_start runs first — its own
        # pre-flight acquire+release completes near-instantly (CPU-only, no
        # subprocess involved), releasing the OS lock BEFORE it spawns its
        # (gated) child. Wait for that spawn to be recorded (a fast, in-
        # process signal — not a subprocess-timing guess) before starting
        # role-b, so role-b's own pre-flight genuinely runs after role-a's
        # has already released the lock, matching S57's handoff scenario.
        t_a.start()
        _poll_until(
            lambda: "role-a" in launched,
            timeout=10.0,
            message="role-a's cmd_start never reached spawn_detached",
        )
        t_b.start()

        # role-b's child is never gated, so with role-a's child still
        # blocked at wait_gate_a, role-b's child is the only one competing
        # for the lock — it wins as soon as it gets there. Generous bound
        # for slow Windows subprocess/import startup; this is the one place
        # that genuinely waits on it, not a race against role-a.
        _poll_until(
            lambda: winner_marker.exists(),
            timeout=60.0,
            message="role-b's child never reached the stub (never won the lock)",
        )
        assert not loser_marker.exists(), "role-a's child must still be blocked at its wait-gate"

        # role-b's cmd_start should resolve very shortly after its child's
        # marker appears (the child writes the matching state_file right
        # after the marker, in the same stub call) — generous bound for
        # slow Windows scheduling, not because this side is expected to
        # actually take that long.
        t_b.join(timeout=30.0)
        assert not t_b.is_alive(), "role-b's cmd_start should have returned quickly"
        assert results["role-b"] == 0

        inode_while_held = lock_path.stat().st_ino

        # NOW release role-a's child. The lock is guaranteed held by role-b's
        # child at this point (just proven above), so role-a's child is
        # GUARANTEED to lose when it attempts — no timing assumption left.
        wait_gate_a.write_text("go", encoding="utf-8")
        assert launched["role-a"].wait(timeout=60.0) == 2
        assert not loser_marker.exists(), "the losing child must never reach run_bridge_foreground"

        # role-a's cmd_start must notice its child exited and report now,
        # while the winner is STILL running (stop_file not yet written), not
        # at its 50s deadline. Generous bound for slow Windows scheduling;
        # the rc == 2 below is the part only the early-exit path can produce
        # (the deadline path returns 1).
        t_a.join(timeout=30.0)
        assert not t_a.is_alive(), "role-a's cmd_start must return promptly once its child lost"
        assert results["role-a"] == 2
        winner_pid = launched["role-b"].pid
        assert (
            f"bridge already running on port 51901 (pid {winner_pid})" in capsys.readouterr().err
        ), "loser must report with the existing S57 'already running' wording"

        # Now release the winner so its cmd_start's spawned child (still
        # parked in the stub) can exit cleanly.
        stop_file.write_text("go", encoding="utf-8")
        assert launched["role-b"].wait(timeout=30.0) == 0
        # cmd_start dropped both handles once done waiting (no leak).
        assert not {p.pid for p in launched.values()} & set(daemon._spawned_children)

        assert lock_path.exists()  # S56: never unlinked
        assert lock_path.stat().st_ino == inode_while_held
    finally:
        stop_file.write_text("go", encoding="utf-8")
        wait_gate_a.write_text("go", encoding="utf-8")
        for proc in launched.values():
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10.0)
        for t in (t_a, t_b):
            t.join(timeout=5.0)
