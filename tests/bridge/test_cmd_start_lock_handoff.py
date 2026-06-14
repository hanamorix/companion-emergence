"""Bug 2a (v0.0.36): cmd_start must NOT hold the per-persona lockfile across the
detached spawn — the lock belongs to the bridge process (the runner) for its
lifetime. If cmd_start holds it, the child runner can't acquire it and dies.
The runner owns the lock now (test_runner_mutual_exclusion); cmd_start keeps
only the cheap is_running pre-check."""
from __future__ import annotations

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
        # cmd_start must NOT be holding it, or the child can never bind.
        seen["lock_free_at_spawn"] = not (pd / daemon.LOCKFILE).exists()
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
