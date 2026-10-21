"""What the OS service managers are told, given the bridge's own start back-off (C70; D26 j).

systemd and launchd need no change (the wait is inside the process, so no start limit or count can trip); Task
Scheduler's restart count must be raised (its default of 3 would leave no bridge after a persistent failure), and
installed tasks are refreshed in place like the systemd KillMode migration (C47).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from brain.service import launchd, systemd, windows_service

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    nell = tmp_path / "bin" / "nell"
    nell.parent.mkdir()
    nell.write_text("#!/bin/sh\n")
    nell.chmod(0o755)
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    return nell


# ---- (a) systemd: unchanged -----------------------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="systemd unit text uses POSIX paths")
def test_the_systemd_unit_keeps_its_restart_lines_and_has_no_start_limit(env, tmp_path, monkeypatch):
    monkeypatch.setattr(systemd, "_user_home", lambda: tmp_path)
    body = systemd.build_systemd_unit_text(persona="nell", nell_path=env)
    lines = body.splitlines()
    assert "Restart=on-failure" in lines and "RestartSec=2" in lines
    assert not any(ln.startswith(("StartLimit", "RestartSteps", "RestartMaxDelaySec")) for ln in lines)


# ---- (b) launchd: unchanged -----------------------------------------------------------------------------------------


def test_the_launchd_plist_keeps_its_keep_alive_and_adds_no_throttle(env):
    plist = launchd.build_launchd_plist(persona="nell", nell_path=env)
    assert plist["KeepAlive"] == {"Crashed": True, "SuccessfulExit": False}
    assert "ThrottleInterval" not in plist


# ---- (c) Task Scheduler: the numbers ---------------------------------------------------------------------------------


def _restart(body: str) -> tuple[str, str]:
    root = ET.fromstring(body.encode("utf-16"))
    node = root.find("t:Settings/t:RestartOnFailure", NS)
    assert node is not None
    return node.findtext("t:Interval", namespaces=NS), node.findtext("t:Count", namespaces=NS)


def test_the_built_task_restarts_every_minute_up_to_the_scheduler_maximum(env):
    body = windows_service.build_task_xml(persona="nell", nell_path=env)
    # PT1M: the scheduler's minimum interval; 255: its maximum count (MS-TSCH: Count 1 to 255, no "unlimited").
    assert _restart(body) == ("PT1M", "255")


def test_every_other_task_setting_is_as_before(env):
    body = windows_service.build_task_xml(persona="nell", nell_path=env)
    old = body.replace("<Interval>PT1M</Interval>", "<Interval>PT2M</Interval>").replace(
        "<Count>255</Count>", "<Count>3</Count>"
    )
    assert _restart(old) == ("PT2M", "3")
    assert body.replace("PT1M", "PT2M").replace("255", "3") == old


# ---- (d) the in-place refresh of an installed task -------------------------------------------------------------------


class Schtasks:
    def __init__(self, registered: str | None, create_rc: int = 0, query_rc: int = 0) -> None:
        self.registered = registered
        self.create_rc = create_rc
        self.query_rc = query_rc
        self.calls: list[list[str]] = []
        self.created_xml: str | None = None

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        if args[0] == "/Query":
            return subprocess.CompletedProcess(args, self.query_rc, stdout=self.registered or "", stderr="")
        if args[0] == "/Create":
            self.created_xml = Path(args[2]).read_bytes().decode("utf-16")
            self.registered = self.created_xml
            return subprocess.CompletedProcess(args, self.create_rc, stdout="", stderr="denied" if self.create_rc else "")
        raise AssertionError(args)


def _old_task(env) -> str:
    body = windows_service.build_task_xml(persona="nell", nell_path=env)
    return body.replace("<Interval>PT1M</Interval>", "<Interval>PT2M</Interval>").replace(
        "<Count>255</Count>", "<Count>3</Count>"
    )


def test_an_installed_task_of_ours_is_re_registered_once_with_only_the_two_values_changed(env, monkeypatch, caplog):
    old = _old_task(env)
    sch = Schtasks(old)
    monkeypatch.setattr(windows_service, "run_schtasks", sch)
    with caplog.at_level(logging.INFO, logger="brain.service.windows_service"):
        assert windows_service.refresh_installed_task("nell") is True
    creates = [c for c in sch.calls if c[0] == "/Create"]
    assert len(creates) == 1
    assert creates[0][1] == "/XML" and creates[0][3:] == ["/TN", "CompanionEmergence-nell", "/F"]
    assert sch.created_xml == old.replace("PT2M", "PT1M").replace("<Count>3</Count>", "<Count>255</Count>")
    assert sum("refreshed" in r.message for r in caplog.records) == 1
    # idempotent: the second run changes nothing and logs nothing
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="brain.service.windows_service"):
        assert windows_service.refresh_installed_task("nell") is False
    assert len([c for c in sch.calls if c[0] == "/Create"]) == 1
    assert not [r for r in caplog.records if "refreshed" in r.message]


def test_a_task_whose_command_is_not_ours_is_left_alone(env, monkeypatch):
    other = _old_task(env).replace("supervisor run", "something else")
    sch = Schtasks(other)
    monkeypatch.setattr(windows_service, "run_schtasks", sch)
    assert windows_service.refresh_installed_task("nell") is False
    assert not [c for c in sch.calls if c[0] == "/Create"]


def test_no_installed_task_is_a_noop(env, monkeypatch):
    sch = Schtasks(None, query_rc=1)
    monkeypatch.setattr(windows_service, "run_schtasks", sch)
    assert windows_service.refresh_installed_task("nell") is False
    assert [c[0] for c in sch.calls] == ["/Query"]


def test_a_failing_create_is_a_warning_not_an_error(env, monkeypatch, caplog):
    sch = Schtasks(_old_task(env), create_rc=1)
    monkeypatch.setattr(windows_service, "run_schtasks", sch)
    with caplog.at_level(logging.WARNING, logger="brain.service.windows_service"):
        assert windows_service.refresh_installed_task("nell") is False
    assert any("task" in r.message and r.levelno == logging.WARNING for r in caplog.records)


def test_a_registered_task_with_non_ascii_text_is_never_re_registered(env, monkeypatch, caplog):
    """Review of D26 (MF2): schtasks' output is decoded as UTF-8 with replacement, so a profile path with a non-ASCII
    letter would come back mangled, and re-registering that text would silently break the task's command path. Any
    non-ASCII (U+FFFD included) or NUL in the queried XML skips the refresh and logs one WARNING."""
    old = _old_task(env)
    for bad in (
        old.replace("<Arguments>", "<WorkingDirectory>C:\\Users\\Jos\u00e9</WorkingDirectory><Arguments>", 1),
        old.replace("<Arguments>", "<WorkingDirectory>C:\\Users\\Jos\ufffd</WorkingDirectory><Arguments>", 1),
        old.replace("<Count>3</Count>", "<Count>3\x00</Count>"),
    ):
        sch = Schtasks(bad)
        monkeypatch.setattr(windows_service, "run_schtasks", sch)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="brain.service.windows_service"):
            assert windows_service.refresh_installed_task("nell") is False
        assert not [c for c in sch.calls if c[0] == "/Create"]
        assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1


def test_cmd_run_never_refreshes_the_task_from_inside_a_running_instance(tmp_path, monkeypatch):
    """Review of D26 (MF2): no origin's `supervisor run` refreshes the Task Scheduler task (the instance would be
    replacing its own definition); the refresh belongs to the paths that run outside any instance."""
    from types import SimpleNamespace

    from brain.bridge import daemon

    pd = tmp_path / "home" / "personas" / "nell"
    pd.mkdir(parents=True)
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: pd)
    monkeypatch.setattr(
        "brain.service.windows_service.refresh_installed_task", lambda persona: pytest.fail("refreshed in a run")
    )
    monkeypatch.setattr("brain.service.windows_service.run_schtasks", lambda args: pytest.fail("schtasks in a run"))
    monkeypatch.setattr("brain.service.systemd.refresh_installed_unit", lambda persona: False)
    monkeypatch.setattr("brain.service.launchd.truncate_launchd_logs_if_large", lambda persona: None)
    monkeypatch.setattr("brain.bridge.runner._setup_runtime_logging", lambda p: None)
    ran: list[int] = []
    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", lambda p, **kw: ran.append(1) or 0)
    for origin in ("systemd", "launchd", "cli", "task-scheduler"):
        assert daemon.cmd_run(SimpleNamespace(persona="nell", idle_shutdown=0, client_origin=origin)) == 0
    assert ran == [1, 1, 1, 1]


def test_start_service_refreshes_the_installed_task_before_it_runs_it(env, tmp_path, monkeypatch):
    """Review of D26 (MF2): `start_service` runs outside any instance; the refresh comes before `/Run`, so an instance
    the scheduler might end on an update is started again right after."""
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: tmp_path / "home" / "personas" / name)
    (tmp_path / "home" / "personas" / "nell").mkdir(parents=True)
    sch = Schtasks(_old_task(env))
    real_call = sch.__call__

    def with_run(args):
        if args[0] == "/Run":
            sch.calls.append(list(args))
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return real_call(args)

    monkeypatch.setattr(windows_service, "run_schtasks", with_run)
    windows_service.start_service(persona="nell")
    assert [c[0] for c in sch.calls] == ["/Query", "/Create", "/Run"]
    sch.calls.clear()
    windows_service.start_service(persona="nell")  # already current
    assert [c[0] for c in sch.calls] == ["/Query", "/Run"]


def test_a_refresh_that_raises_never_stops_start_service(env, tmp_path, monkeypatch):
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: tmp_path / "home" / "personas" / name)
    (tmp_path / "home" / "personas" / "nell").mkdir(parents=True)
    ran: list[list[str]] = []

    def boom(persona):
        raise RuntimeError("schtasks exploded")

    monkeypatch.setattr(windows_service, "refresh_installed_task", boom)
    monkeypatch.setattr(
        windows_service, "run_schtasks", lambda args: ran.append(list(args)) or subprocess.CompletedProcess(args, 0, "", "")
    )
    windows_service.start_service(persona="nell")
    assert ran == [["/Run", "/TN", "CompanionEmergence-nell"]]


def test_install_writes_fresh_xml_and_does_not_round_trip_the_registered_one(env, tmp_path, monkeypatch):
    """An install regenerates the task from `build_task_xml`; it never queries and patches the registered text."""
    from brain.service import lifecycle

    monkeypatch.setattr(lifecycle, "before_install", lambda persona: None)
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: tmp_path / "home" / "personas" / name)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        windows_service, "run_schtasks", lambda args: calls.append(list(args)) or subprocess.CompletedProcess(args, 0, "", "")
    )
    windows_service.install_service(persona="nell", nell_path=env)
    assert [c[0] for c in calls] == ["/Create", "/Run"]


# ---- explicit starts through the service layer reset the record ---------------------------------------------------------


@pytest.mark.parametrize("backend", ["systemd", "launchd", "windows_service"])
def test_start_service_and_install_reset_the_start_back_off(env, tmp_path, monkeypatch, backend):
    from brain.bridge import start_backoff
    from brain.service import lifecycle

    pd = tmp_path / "home" / "personas" / "nell"
    pd.mkdir(parents=True)
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: pd)
    start_backoff.note_start(pd)
    mod = {"systemd": systemd, "launchd": launchd, "windows_service": windows_service}[backend]
    for name in ("run_systemctl", "run_launchctl", "run_schtasks"):
        if hasattr(mod, name):
            monkeypatch.setattr(
                mod, name, lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr="")
            )
    mod.start_service(persona="nell")
    assert start_backoff.read(pd).unproven_starts == 0
    start_backoff.note_start(pd)
    monkeypatch.setattr(lifecycle, "bridge_running", lambda persona: False)
    lifecycle.before_install("nell")
    assert start_backoff.read(pd).unproven_starts == 0
