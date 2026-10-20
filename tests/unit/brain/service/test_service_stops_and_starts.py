"""Service ownership, the manager's stop, service starts and installs (#456: C45 e, C58 a, d; D20, D25, S-k, S5-i).

Every manager call goes through a spy; no real systemctl / launchctl / schtasks runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from brain.bridge import kill, shutdown_progress, state_file


def _cp(rc: int = 0, out: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr="")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    pd = tmp_path / "home" / "personas" / "nell"
    pd.mkdir(parents=True)
    return pd


# ---------------------------------------------------------------- ownership (kill step 4)


def test_systemd_ownership_is_the_units_main_pid(monkeypatch):
    from brain.service import systemd

    calls: list[list[str]] = []
    monkeypatch.setattr(systemd, "run_systemctl", lambda args: calls.append(args) or _cp(0, "4242\n"))
    assert kill.owned_by_service("nell", 4242, "systemd", platform="linux") is True
    assert kill.owned_by_service("nell", 4243, "systemd", platform="linux") is False
    assert calls[0] == ["show", "-p", "MainPID", "--value", systemd.service_unit_name("nell")]


@pytest.mark.parametrize(
    ("out", "owned"),
    [("state = running\n\tpid = 4242\n", True), ("pid = 4242\npid = 4242\n", False), ("state = waiting\n", False)],
)
def test_launchd_ownership_needs_exactly_one_matching_pid_line(monkeypatch, out, owned):
    from brain.service import launchd

    calls: list = []
    monkeypatch.setattr(launchd, "run_launchctl", lambda args, timeout=None: calls.append((args, timeout)) or _cp(0, out))
    assert kill.owned_by_service("nell", 4242, "launchd", platform="darwin") is owned
    assert calls[0][0][0] == "print" and calls[0][1] == kill.MANAGER_CAP_S


def test_windows_ownership_is_the_task_scheduler_origin():
    assert kill.owned_by_service("nell", 1, "task-scheduler", platform="win32") is True
    assert kill.owned_by_service("nell", 1, "tauri", platform="win32") is False


def test_ownership_is_asked_only_when_the_service_is_installed(monkeypatch, tmp_path):
    """C45 (e): an uninstalled service is never asked (the helper takes its own kill path)."""
    asked: list = []
    monkeypatch.setattr(kill, "service_installed", lambda persona: False)
    monkeypatch.setattr(kill, "owned_by_service", lambda *a, **k: asked.append(a) or True)
    pd = tmp_path / "nell"
    pd.mkdir()
    kill.kill_bridge(pd, timeout=1)  # no record: "not running" before any question
    assert asked == []


# ---------------------------------------------------------------- the manager's stop (kill step 5)


def test_systemd_manager_stop_is_no_block(monkeypatch):
    from brain.service import systemd

    calls: list = []
    monkeypatch.setattr(systemd, "run_systemctl", lambda args: calls.append(args) or _cp(0))
    kill.manager_stop("nell", platform="linux")
    assert calls == [["stop", "--no-block", systemd.service_unit_name("nell")]]


def test_launchd_manager_stop_is_a_capped_bootout(monkeypatch):
    from brain.service import launchd

    calls: list = []
    monkeypatch.setattr(launchd, "run_launchctl", lambda args, timeout=None: calls.append((args, timeout)) or _cp(124))
    kill.manager_stop("nell", platform="darwin")  # a timeout at the cap is not raised
    assert calls == [(["bootout", launchd.launchctl_target("nell")], kill.MANAGER_CAP_S)]


def test_windows_has_no_manager_stop(monkeypatch):
    from brain.service import windows_service

    calls: list = []
    monkeypatch.setattr(windows_service, "run_schtasks", lambda args: calls.append(args) or _cp(0))
    kill.manager_stop("nell", platform="win32")
    assert calls == []


def test_launchctl_cap_turns_a_timeout_into_a_failed_result(monkeypatch):
    from brain.service import launchd

    def _hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="launchctl", timeout=k.get("timeout"))

    monkeypatch.setattr(launchd.subprocess, "run", _hang)
    r = launchd.run_launchctl(["bootout", "gui/1/x"], timeout=15)
    assert r.returncode == 124


# ---------------------------------------------------------------- service starts (D20, C58 a) and the stop intent (D25)


def _intent(pd: Path) -> None:
    shutdown_progress.write_record(pd, shutdown_progress.INTENT_FILENAME,
                                   {"pid": 1, "pid_token": "t", "state": "committed", "nonce": "n"})


def test_systemd_start_service_clears_the_intent_and_starts_the_unit(home, monkeypatch):
    from brain.service import systemd

    _intent(home)
    calls: list = []
    monkeypatch.setattr(systemd, "run_systemctl", lambda args: calls.append(args) or _cp(0))
    systemd.start_service(persona="nell")
    assert not shutdown_progress.path(home, shutdown_progress.INTENT_FILENAME).exists()
    assert calls == [["start", systemd.service_unit_name("nell")]]


@pytest.mark.parametrize("loaded", [True, False])
def test_launchd_start_service_bootstraps_when_unloaded_then_plain_kickstart(home, monkeypatch, loaded):
    from brain.service import launchd

    _intent(home)
    calls: list = []

    def fake(args, timeout=None):
        calls.append(args)
        if args[0] == "print":
            return _cp(0 if loaded else 113)
        return _cp(0)

    monkeypatch.setattr(launchd, "run_launchctl", fake)
    launchd.start_service(persona="nell")
    assert not shutdown_progress.path(home, shutdown_progress.INTENT_FILENAME).exists()
    verbs = [c[0] for c in calls]
    assert verbs == (["print", "kickstart"] if loaded else ["print", "bootstrap", "kickstart"])
    assert all("-k" not in c for c in calls)


def test_windows_start_service_runs_the_task(home, monkeypatch):
    from brain.service import windows_service

    _intent(home)
    calls: list = []
    monkeypatch.setattr(windows_service, "run_schtasks", lambda args: calls.append(args) or _cp(0))
    windows_service.start_service(persona="nell")
    assert not shutdown_progress.path(home, shutdown_progress.INTENT_FILENAME).exists()
    assert calls == [["/Run", "/TN", windows_service.task_name("nell")]]


# ---------------------------------------------------------------- installs (C58 d, S-k, S5-i)


@pytest.fixture
def running(monkeypatch):
    monkeypatch.setattr(state_file, "liveness", lambda pd: state_file.ALIVE)


@pytest.mark.parametrize("backend_name", ["systemd", "launchd", "windows_service"])
def test_install_refuses_without_a_tty_while_the_bridge_runs_and_changes_nothing(
    home, monkeypatch, running, backend_name, tmp_path
):
    import importlib

    from brain.service import lifecycle

    backend = importlib.import_module(f"brain.service.{backend_name}")
    calls: list = []
    for seam in ("run_systemctl", "run_launchctl", "run_schtasks"):
        if hasattr(backend, seam):
            monkeypatch.setattr(backend, seam, lambda *a, **k: calls.append(a) or _cp(0))
    monkeypatch.setattr(lifecycle, "_interactive", lambda: False)
    nell = tmp_path / "nell-bin"
    nell.write_text("#!/bin/sh\n")
    nell.chmod(0o755)
    with pytest.raises(lifecycle.BridgeRunningError, match="the bridge is running; stop it first"):
        backend.install_service(persona="nell", nell_path=nell)
    assert calls == []


@pytest.mark.parametrize("backend_name", ["systemd", "launchd", "windows_service"])
def test_install_on_a_stopped_bridge_clears_the_intent_first_and_never_kickstart_k(
    home, monkeypatch, backend_name, tmp_path
):
    import importlib

    backend = importlib.import_module(f"brain.service.{backend_name}")
    monkeypatch.setattr(state_file, "liveness", lambda pd: state_file.GONE)
    _intent(home)
    calls: list = []
    seen_intent: list = []

    def fake(*a, **k):
        seen_intent.append(shutdown_progress.path(home, shutdown_progress.INTENT_FILENAME).exists())
        calls.append(list(a[0]))
        return _cp(113 if a[0][:1] == ["print"] else 0)

    for seam in ("run_systemctl", "run_launchctl", "run_schtasks"):
        if hasattr(backend, seam):
            monkeypatch.setattr(backend, seam, fake)
    for attr in ("user_unit_dir", "launch_agents_dir", "task_xml_dir"):
        if hasattr(backend, attr):
            monkeypatch.setattr(backend, attr, lambda: tmp_path / "svc")
    if hasattr(backend, "unstable_nell_path_reason"):
        monkeypatch.setattr(backend, "unstable_nell_path_reason", lambda p: None)
    if hasattr(backend, "task_user_id"):
        monkeypatch.setattr(backend, "task_user_id", lambda: "user")
    nell = tmp_path / "nell-bin"
    nell.write_text("#!/bin/sh\n")
    nell.chmod(0o755)
    backend.install_service(persona="nell", nell_path=nell)
    assert seen_intent and not any(seen_intent)  # deleted before the first manager call
    assert not any("-k" in c for c in calls)


def test_service_installed_reads_the_backend_install_state(tmp_path, monkeypatch):
    """The real `service_installed` (the root conftest stubs it for every other test)."""
    from brain.service import systemd

    real = kill._service_installed_check

    monkeypatch.setattr(systemd, "user_unit_dir", lambda: tmp_path)
    assert real("nell", platform="linux") is False
    (tmp_path / systemd.service_unit_name("nell")).write_text("[Unit]\n")
    assert real("nell", platform="linux") is True


def test_run_systemctl_reports_a_hung_systemctl_as_a_failed_call(monkeypatch):
    """Review out-of-scope note: a hung systemctl (TimeoutExpired) comes back as a failed CompletedProcess, so every
    caller (install, uninstall, status, start) handles it like any other failure instead of raising."""
    import subprocess

    from brain.service import systemd

    def _hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd=a[0], timeout=15)

    monkeypatch.setattr(subprocess, "run", _hang)
    r = systemd.run_systemctl(["start", "x.service"])
    assert r.returncode != 0 and "timed out" in r.stderr
