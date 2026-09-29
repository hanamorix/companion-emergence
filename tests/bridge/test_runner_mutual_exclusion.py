"""Bug 2a (v0.0.36): the detached bridge runner (`python -m brain.bridge.runner`,
spawned by cmd_start on app open) had NO mutual-exclusion — only cmd_run (the
task path) held the lockfile. So two concurrent `supervisor start` calls each
spawned a live bridge (two supervisors, parallel soul review). The runner now
holds the same is_running + lockfile guard cmd_run does."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import brain.bridge.runner as runner
from brain.bridge import daemon, state_file


def test_runner_refuses_when_a_bridge_is_already_running(tmp_path, monkeypatch):
    monkeypatch.setattr(state_file, "is_running", lambda pd: True)
    called = []
    monkeypatch.setattr(runner, "run_bridge_foreground",
                        lambda *a, **k: called.append(1) or 0)
    rc = runner.main(["--persona-dir", str(tmp_path)])
    assert rc == 2
    assert called == []  # did NOT spawn a second bridge


def test_runner_acquires_lock_runs_then_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(state_file, "is_running", lambda pd: False)
    seen = {}

    def _fake_run(pd, **k):
        seen["lock_held_during_run"] = (pd / daemon.LOCKFILE).exists()
        return 0

    monkeypatch.setattr(runner, "run_bridge_foreground", _fake_run)
    rc = runner.main(["--persona-dir", str(tmp_path)])
    assert rc == 0
    assert seen["lock_held_during_run"] is True   # lock held while binding
    # S56: release_lock never unlinks — the file persists, but the OS lock on
    # it must be free again (a fresh acquire succeeds).
    assert (tmp_path / daemon.LOCKFILE).exists()
    fd = daemon.acquire_lock(tmp_path)
    assert fd is not None
    daemon.release_lock(tmp_path, fd)


_WORKER_SCRIPT = str(Path(__file__).parent / "_runner_lock_worker.py")


def test_c19b_two_runner_children_started_at_once_exactly_one_wins(tmp_path):
    """C19(b): two `runner.main` children started directly at once (real OS
    processes, real `brain.bridge.daemon.acquire_lock`/msvcrt-or-flock lock —
    see `_runner_lock_worker.py`'s docstring for exactly what's stubbed and
    why it still exercises the real exclusion). Exactly one ends up holding
    the lock; the other exits 2 having never reached `run_bridge_foreground`
    (never "wrote bridge.json or bound a port"); the lock file's inode is
    unchanged across the whole race, and it is never unlinked.

    Uses `sys.executable` + a real subprocess (no fork-only API), so this
    runs unmodified on the Windows/macOS CI runners too.
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    lock_path = persona_dir / daemon.LOCKFILE
    stop_file = tmp_path / "stop"
    marker_a = tmp_path / "marker-a.txt"
    marker_b = tmp_path / "marker-b.txt"

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                _WORKER_SCRIPT,
                "--persona-dir",
                str(persona_dir),
                "--marker",
                str(marker),
                "--stop-file",
                str(stop_file),
            ]
        )
        for marker in (marker_a, marker_b)
    ]

    try:
        # Wait for the race to settle: one sibling exits quickly (the loser,
        # rc=2, no marker); the other blocks in the stub (the winner).
        deadline = time.time() + 20.0
        loser_idx = None
        while time.time() < deadline:
            for i, proc in enumerate(procs):
                if proc.poll() is not None:
                    loser_idx = i
                    break
            if loser_idx is not None:
                break
            time.sleep(0.05)
        assert loser_idx is not None, "one sibling should have exited quickly as the loser"
        winner_idx = 1 - loser_idx
        loser, winner = procs[loser_idx], procs[winner_idx]

        assert loser.returncode == 2
        assert winner.poll() is None  # still holding the lock, blocked in the stub

        existing_markers = [m for m in (marker_a, marker_b) if m.exists()]
        assert len(existing_markers) == 1, "only the winner may have reached run_bridge_foreground"

        inode_while_held = lock_path.stat().st_ino

        # Release the winner.
        stop_file.write_text("go", encoding="utf-8")
        assert winner.wait(timeout=15.0) == 0

        assert lock_path.exists()  # S56: never unlinked
        assert lock_path.stat().st_ino == inode_while_held
    finally:
        stop_file.write_text("go", encoding="utf-8")
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10.0)


def test_runner_usage_error_does_not_exit_with_the_refusal_code(tmp_path):
    """Exit 2 means only the S57 refusal (cmd_start reports it with the
    refusal wording); argparse's own usage-error 2 must not look like one."""
    assert runner.main(["--no-such-flag"]) == 1
    assert runner.main([]) == 1  # --persona-dir missing

