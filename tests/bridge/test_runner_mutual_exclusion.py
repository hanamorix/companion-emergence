"""Bug 2a (v0.0.36): the detached bridge runner (`python -m brain.bridge.runner`,
spawned by cmd_start on app open) had NO mutual-exclusion — only cmd_run (the
task path) held the lockfile. So two concurrent `supervisor start` calls each
spawned a live bridge (two supervisors, parallel soul review). The runner now
holds the same is_running + lockfile guard cmd_run does."""
from __future__ import annotations

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
    assert seen["lock_held_during_run"] is True       # lock held while binding
    assert not (tmp_path / daemon.LOCKFILE).exists()  # released after
