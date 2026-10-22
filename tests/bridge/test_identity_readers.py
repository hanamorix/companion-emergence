"""bridge.json pid readers use process identity (C21, C23, C33, C43 c-e, C61, C63), not a bare pid."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import process_identity as pi
from brain.bridge import state_file

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux token reader")


def _write(persona_dir: Path, **kw) -> state_file.BridgeState:
    base = {
        "persona": persona_dir.name,
        "pid": None,
        "port": 51234,
        "started_at": "2026-10-19T10:00:00Z",
        "stopped_at": None,
        "shutdown_clean": False,
        "client_origin": "cli",
    }
    base.update(kw)
    s = state_file.BridgeState(**base)
    state_file.write(persona_dir, s)
    return s


@pytest.fixture
def sleeper():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield p
    p.kill()
    p.wait()


@linux_only
def test_a_reused_pid_with_a_token_reads_as_not_running(persona_dir, sleeper):
    # C23: the record's token is the token of a process that no longer runs; the pid now belongs to an unrelated one.
    _write(persona_dir, pid=sleeper.pid, pid_token="linux:other-boot:1")
    assert state_file.liveness(persona_dir) == state_file.GONE
    assert state_file.is_running(persona_dir) is False
    assert state_file.recovery_needed(persona_dir) is True  # dirty and gone


@linux_only
def test_the_recorded_process_itself_reads_as_running(persona_dir, sleeper):
    _write(persona_dir, pid=sleeper.pid, pid_token=pi.token_of(sleeper.pid))
    assert state_file.liveness(persona_dir) == state_file.ALIVE
    assert state_file.is_running(persona_dir) is True
    assert state_file.recovery_needed(persona_dir) is False


def test_unknown_identity_is_not_running_for_starts_but_live_for_recover(
    persona_dir, monkeypatch, sleeper
):
    # C20 / S-a / C63: an unreadable token does not block a start (the OS lock decides) but blocks `nell recover`.
    _write(persona_dir, pid=sleeper.pid, pid_token="linux:x:1")
    monkeypatch.setattr(pi, "token_of", lambda pid: pi.UNKNOWN)
    assert state_file.liveness(persona_dir) == state_file.UNKNOWN
    assert state_file.is_running(persona_dir) is False
    from brain.recovery import cli as rcli

    assert rcli._bridge_is_live(persona_dir) is True


def test_471_production_shaped_drain_errors_state_needs_recovery(persona_dir):
    # C33: the state the pre-#456 `_write_clean_shutdown` left after a drain-errors record: pid and port cleared,
    # stopped_at set, shutdown_clean False. That writer is gone (plan I4: no shutdown drain any more, Q-B), so the
    # shape is written as the older version left it on disk.
    _write(persona_dir, pid=None, port=None, stopped_at="2026-10-01T00:00:00+00:00", shutdown_clean=False,
           drain_errors=2)
    assert state_file.recovery_needed(persona_dir) is True


def test_471_shape_nell_status_says_recovery_needed(persona_dir, capsys):
    """C33 `nell status` leg (review M5): the #471 record (pid cleared, not clean, drain errors) says recovery is
    needed, not just "not running"."""
    from brain import cli

    _write(persona_dir, pid=None, port=None, stopped_at="2026-10-01T00:00:00+00:00", shutdown_clean=False,
           drain_errors=2)
    cli._print_bridge_status(state_file.read(persona_dir), persona_dir)
    out = capsys.readouterr().out
    assert "bridge: not running" in out
    assert "recovery: needed on next bridge start" in out


def test_clean_stopped_record_nell_status_says_no_recovery(persona_dir, capsys):
    from brain import cli

    _write(persona_dir, pid=None, port=None, shutdown_clean=True)
    cli._print_bridge_status(state_file.read(persona_dir), persona_dir)
    assert "recovery" not in capsys.readouterr().out


def test_471_shape_supervisor_status_names_no_pid_and_uses_plain_punctuation(persona_dir, capsys, monkeypatch):
    """Review out-of-scope note: `nell supervisor status` on a record with no pid does not print "pid None", and the
    line has no em-dash."""
    import brain.paths
    from brain.bridge import daemon

    monkeypatch.setattr(brain.paths, "get_persona_dir", lambda _name: persona_dir)
    _write(persona_dir, pid=None, port=None, stopped_at="2026-10-01T00:00:00+00:00", shutdown_clean=False,
           drain_errors=2)
    assert daemon.cmd_status(SimpleNamespace(persona=persona_dir.name)) == 0
    out = capsys.readouterr().out
    assert "previous process crashed dirty" in out
    assert "pid None" not in out and "\u2014" not in out


def test_471_older_version_file_shape_needs_recovery(persona_dir):
    _write(persona_dir, pid=None, shutdown_clean=False, drain_errors=1)
    assert state_file.recovery_needed(persona_dir) is True


def test_clean_record_needs_no_recovery(persona_dir):
    _write(persona_dir, pid=None, port=None, shutdown_clean=True)
    assert state_file.recovery_needed(persona_dir) is False


# ---- pre-PR records (no pid_token): owner's decision D22, C43 (c)-(e), C61 ----


def test_unproven_pre_pr_record_with_an_unrelated_live_pid_is_stale(persona_dir, sleeper, capsys):
    _write(persona_dir, pid=sleeper.pid)  # old format: no pid_token field written by an old bridge
    assert state_file.liveness(persona_dir) == state_file.GONE
    assert state_file.is_running(persona_dir) is False
    assert state_file.recovery_needed(persona_dir) is True
    # `nell recover` still refuses: pid alive but not proven counts as live for that guard (S-a, C63).
    from brain.recovery import cli as rcli

    assert rcli._bridge_is_live(persona_dir) is True


def test_unproven_pre_pr_record_is_never_signalled_by_stop(persona_dir, sleeper, monkeypatch):
    # C21 / C43 (c): cmd_stop on a stale record sends no signal and reports a stale record.
    from brain.bridge import daemon

    _write(persona_dir, pid=sleeper.pid)
    sent: list = []
    monkeypatch.setattr(daemon, "_kill_if_alive", lambda pid, sig: sent.append((pid, sig)) or True)
    monkeypatch.setattr(
        daemon, "_request_shutdown_via_http", lambda s, **kw: (_ for _ in ()).throw(OSError("no"))
    )
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)
    rc = daemon.cmd_stop(SimpleNamespace(persona=persona_dir.name, timeout=1.0, force=True))
    assert rc == 0
    assert sent == []
    assert sleeper.poll() is None
    after = state_file.read(persona_dir)
    assert after.pid is None and after.port is None


@linux_only
def test_token_record_failing_identity_is_never_signalled(
    persona_dir, sleeper, monkeypatch, capsys
):
    from brain.bridge import daemon

    _write(persona_dir, pid=sleeper.pid, pid_token="linux:other:1")
    sent: list = []
    monkeypatch.setattr(daemon, "_kill_if_alive", lambda pid, sig: sent.append((pid, sig)) or True)
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)
    rc = daemon.cmd_stop(SimpleNamespace(persona=persona_dir.name, timeout=1.0, force=True))
    assert rc == 0 and sent == []
    assert "stale record" in capsys.readouterr().out


def _bridge_like(persona_dir: Path, *, form: str, env: dict | None = None) -> subprocess.Popen:
    """A process whose command line looks like a bridge for this persona (it only sleeps)."""
    code = "import time; time.sleep(60)"
    if form == "runner":
        argv = [
            sys.executable,
            "-c",
            code,
            "-m",
            "brain.bridge.runner",
            "--persona-dir",
            str(persona_dir),
        ]
    else:
        argv = [sys.executable, "-c", code, "supervisor", "run", "--persona", persona_dir.name]
    return subprocess.Popen(argv, env=env)


def test_proven_pre_pr_runner_is_alive(persona_dir):
    p = _bridge_like(persona_dir, form="runner")
    try:
        _write(persona_dir, pid=p.pid)
        assert state_file.prove_bridge_cmdline(p.pid, persona_dir, persona_dir.name) is True
        assert state_file.liveness(persona_dir) == state_file.ALIVE
    finally:
        p.kill()
        p.wait()


def test_pre_pr_runner_of_another_persona_dir_is_unproven(persona_dir, tmp_path):
    other = tmp_path / "other-persona"
    other.mkdir()
    p = _bridge_like(other, form="runner")
    try:
        _write(persona_dir, pid=p.pid)
        assert state_file.liveness(persona_dir) == state_file.GONE
    finally:
        p.kill()
        p.wait()


def _env(**kw) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("KINDLED_HOME", "NELLBRAIN_HOME", "XDG_DATA_HOME")
    }
    env.update(kw)
    return env


def test_service_form_needs_the_same_home(persona_dir, tmp_path, monkeypatch):
    # C43 (e): name match alone is not enough; the target's effective home must equal ours.
    home = tmp_path / "home-a"
    monkeypatch.setenv("KINDLED_HOME", str(home))
    same = _bridge_like(persona_dir, form="service", env=_env(KINDLED_HOME=str(home)))
    other = _bridge_like(
        persona_dir, form="service", env=_env(KINDLED_HOME=str(tmp_path / "home-b"))
    )
    nell = _bridge_like(
        persona_dir, form="service", env=_env(NELLBRAIN_HOME=str(tmp_path / "home-c"))
    )
    try:
        assert state_file.prove_bridge_cmdline(same.pid, persona_dir, persona_dir.name) is True
        assert state_file.prove_bridge_cmdline(other.pid, persona_dir, persona_dir.name) is False
        assert state_file.prove_bridge_cmdline(nell.pid, persona_dir, persona_dir.name) is False
    finally:
        for p in (same, other, nell):
            p.kill()
            p.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="XDG default home")
def test_service_form_unset_home_matches_our_default_and_xdg_differs(
    persona_dir, tmp_path, monkeypatch, real_platform_dirs
):
    monkeypatch.delenv("KINDLED_HOME", raising=False)
    monkeypatch.delenv("NELLBRAIN_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-ours"))
    ours = _bridge_like(
        persona_dir, form="service", env=_env(XDG_DATA_HOME=str(tmp_path / "xdg-ours"))
    )
    theirs = _bridge_like(
        persona_dir, form="service", env=_env(XDG_DATA_HOME=str(tmp_path / "xdg-theirs"))
    )
    try:
        assert state_file.prove_bridge_cmdline(ours.pid, persona_dir, persona_dir.name) is True
        assert state_file.prove_bridge_cmdline(theirs.pid, persona_dir, persona_dir.name) is False
    finally:
        for p in (ours, theirs):
            p.kill()
            p.wait()


def test_unreadable_cmdline_or_environment_is_unproven(persona_dir, monkeypatch):
    import psutil

    p = _bridge_like(persona_dir, form="service")
    try:
        monkeypatch.setattr(
            psutil.Process,
            "environ",
            lambda self: (_ for _ in ()).throw(psutil.AccessDenied(self.pid)),
        )
        assert state_file.prove_bridge_cmdline(p.pid, persona_dir, persona_dir.name) is False
        monkeypatch.setattr(
            psutil.Process,
            "cmdline",
            lambda self: (_ for _ in ()).throw(psutil.AccessDenied(self.pid)),
        )
        assert state_file.prove_bridge_cmdline(p.pid, persona_dir, persona_dir.name) is False
    finally:
        p.kill()
        p.wait()


def test_runner_main_starts_over_an_unproven_pre_pr_record(persona_dir, sleeper, monkeypatch):
    # C61 / S2 for the app spawn: runner.main no longer refuses because an unrelated process holds the old pid.
    from brain.bridge import runner

    _write(persona_dir, pid=sleeper.pid)
    called = {}
    monkeypatch.setattr(
        runner, "run_bridge_foreground", lambda pd, **kw: called.setdefault("ran", True) and 0
    )
    assert runner.main(["--persona-dir", str(persona_dir)]) == 0
    assert called.get("ran") is True


def test_status_json_reports_legacy_and_proven(persona_dir, sleeper, monkeypatch, capsys):
    import json


    _write(persona_dir, pid=sleeper.pid)
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)
    from brain import cli

    monkeypatch.setattr(cli, "get_persona_dir", lambda name: persona_dir)
    assert cli._supervisor_status_handler(SimpleNamespace(persona=persona_dir.name, json=True)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["legacy"] is True and out["proven"] is False and out["alive"] == "gone"


@linux_only
def test_nell_model_set_judges_the_bridge_by_its_identity_not_a_live_pid(persona_dir, sleeper):
    """#472 comment: `nell model set` (and its mid-command check) treats a record as a running bridge only when its pid
    carries the recorded identity token, like every other reader of the recorded pid; a reused pid is not a bridge."""
    from brain import cli

    _write(persona_dir, pid=sleeper.pid, port=51234, pid_token="linux:other-boot:1", identity="token")
    assert cli._bridge_running(persona_dir) is None  # the pid is alive, but it is not the recorded bridge
    _write(persona_dir, pid=sleeper.pid, port=51234, pid_token=pi.token_of(sleeper.pid), identity="token")
    st = cli._bridge_running(persona_dir)
    assert st is not None and st.pid == sleeper.pid


def test_nell_model_set_treats_an_unreadable_identity_as_a_running_bridge(persona_dir, sleeper, monkeypatch):
    """Review MG-3: when the recorded pid's identity cannot be read (UNKNOWN), `nell model set` takes the cautious
    side and goes through the bridge (which fails without writing if nothing answers), never writing
    persona_config.json under a bridge that may be live."""
    from brain import cli

    _write(persona_dir, pid=sleeper.pid, port=51234, pid_token="linux:x:1", identity="token")
    monkeypatch.setattr(pi, "token_of", lambda pid: pi.UNKNOWN)
    st = cli._bridge_running(persona_dir)
    assert st is not None and st.pid == sleeper.pid
