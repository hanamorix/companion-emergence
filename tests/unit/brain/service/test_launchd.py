from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from brain.service import launchd

# launchd is macOS-only. The tests construct shell-style executables
# and rely on POSIX file modes + launchctl semantics; running them on
# Linux/Windows surfaces unrelated failures (chmod no-ops on Windows,
# AppTranslocation paths don't exist, etc). Skip the whole module on
# non-darwin — Linux/Windows backends are covered by test_dispatch.
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="launchd backend is macOS-only"
)


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_service_label_validates_persona_name() -> None:
    assert launchd.service_label("nell_01") == "com.companion-emergence.supervisor.nell_01"
    with pytest.raises(launchd.LaunchdConfigError):
        launchd.service_label("../nell")


def test_build_launchd_plist_uses_foreground_supervisor_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    data_home = home / "data"
    persona_dir = data_home / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data_home))
    nell = _make_executable(tmp_path / "nell")

    plist = launchd.build_launchd_plist(
        persona="nell",
        nell_path=nell,
        env_path="/custom/bin:/usr/bin:/bin",
        nellbrain_home=data_home,
        working_directory=home,
    )

    assert plist["Label"] == "com.companion-emergence.supervisor.nell"
    assert plist["ProgramArguments"] == [
        str(nell),
        "supervisor",
        "run",
        "--persona",
        "nell",
        "--client-origin",
        "launchd",
        "--idle-shutdown",
        "0",
    ]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] == {"Crashed": True, "SuccessfulExit": False}
    assert plist["EnvironmentVariables"] == {
        "PATH": "/custom/bin:/usr/bin:/bin",
        "NELLBRAIN_HOME": str(data_home.resolve()),
    }
    assert plist["StandardOutPath"] == str(data_home / "logs" / "supervisor-nell.out.log")
    assert plist["StandardErrorPath"] == str(data_home / "logs" / "supervisor-nell.err.log")


def test_build_launchd_plist_xml_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    nell = _make_executable(tmp_path / "nell")

    xml = launchd.build_launchd_plist_xml(persona="nell", nell_path=nell)
    parsed = plistlib.loads(xml.encode("utf-8"))

    assert parsed["Label"] == "com.companion-emergence.supervisor.nell"
    assert parsed["ProgramArguments"][:3] == [str(nell), "supervisor", "run"]


def test_resolve_nell_path_requires_absolute_executable(tmp_path: Path) -> None:
    with pytest.raises(launchd.LaunchdConfigError, match="absolute"):
        launchd.resolve_nell_path("relative/nell")

    missing = tmp_path / "missing-nell"
    with pytest.raises(launchd.LaunchdConfigError, match="does not exist"):
        launchd.resolve_nell_path(str(missing))

    non_exec = tmp_path / "nell"
    non_exec.write_text("", encoding="utf-8")
    non_exec.chmod(0o644)
    with pytest.raises(launchd.LaunchdConfigError, match="not executable"):
        launchd.resolve_nell_path(str(non_exec))

    executable = _make_executable(tmp_path / "real-nell")
    assert launchd.resolve_nell_path(str(executable)) == executable.resolve()


def test_resolve_nell_path_prefers_absolute_argv0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = _make_executable(tmp_path / "python-runtime" / "bin" / "nell")
    monkeypatch.setattr(sys, "argv", [str(bundled), "service", "install"])
    monkeypatch.setattr(launchd.shutil, "which", lambda name: None)

    assert launchd.resolve_nell_path() == bundled.resolve()


def test_unstable_nell_path_reason_flags_dmg_and_translocation() -> None:
    assert launchd.unstable_nell_path_reason(
        "/Volumes/Companion Emergence/Companion Emergence.app/Contents/Resources/python-runtime/bin/nell"
    )
    assert launchd.unstable_nell_path_reason(
        "/private/var/folders/x/AppTranslocation/ABC/d/Companion Emergence.app/Contents/Resources/python-runtime/bin/nell"
    )
    assert launchd.unstable_nell_path_reason(
        "/Applications/Companion Emergence.app/Contents/Resources/python-runtime/bin/nell"
    ) is None


def test_doctor_checks_reports_launchd_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    data_home = home / "data"
    persona_dir = data_home / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    (home / "Library").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nell = _make_executable(bin_dir / "nell")
    _make_executable(bin_dir / "claude")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data_home))
    monkeypatch.setattr(launchd.platform, "system", lambda: "Darwin")

    checks = launchd.doctor_checks(
        persona="nell",
        nell_path=str(nell),
        env_path=str(bin_dir),
    )

    by_name = {check.name: check for check in checks}
    assert by_name["platform"].ok is True
    assert by_name["persona_name"].ok is True
    assert by_name["persona_dir"].ok is True
    assert by_name["nell_path"].ok is True
    assert by_name["nell_path_stable"].ok is True
    assert by_name["claude_cli"].ok is True


def test_install_service_rejects_unstable_nell_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launchd,
        "unstable_nell_path_reason",
        lambda path: "unstable test path",
    )

    with pytest.raises(launchd.LaunchdConfigError, match="unstable test path"):
        launchd.install_service(persona="nell", nell_path="/Applications/Nell.app/nell")


def test_truncate_launchd_logs_if_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    data_home = home / "data"
    (data_home / "personas" / "nell").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data_home))
    paths = launchd.paths_for_persona("nell")
    paths.stdout_path.parent.mkdir(parents=True)
    paths.stdout_path.write_text("x" * 20, encoding="utf-8")
    paths.stderr_path.write_text("small", encoding="utf-8")

    truncated = launchd.truncate_launchd_logs_if_large("nell", max_bytes=10)

    assert truncated == [paths.stdout_path]
    assert paths.stdout_path.read_text(encoding="utf-8") == ""
    assert paths.stderr_path.read_text(encoding="utf-8") == "small"


def test_doctor_checks_reports_missing_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    data_home = home / "data"
    (data_home / "personas" / "nell").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nell = _make_executable(bin_dir / "nell")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data_home))
    monkeypatch.setattr(launchd.platform, "system", lambda: "Darwin")

    checks = launchd.doctor_checks(
        persona="nell",
        nell_path=str(nell),
        env_path=str(bin_dir),
    )

    by_name = {check.name: check for check in checks}
    assert by_name["claude_cli"].ok is False
    assert "claude not found" in by_name["claude_cli"].detail


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["launchctl"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_install_service_writes_plist_and_bootstraps_launchd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    data_home = home / "data"
    (data_home / "personas" / "nell").mkdir(parents=True)
    (home / "Library").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data_home))
    nell = _make_executable(tmp_path / "nell")
    calls: list[list[str]] = []

    def fake_launchctl(args: list[str]):
        calls.append(args)
        return _completed(0)

    monkeypatch.setattr(launchd, "run_launchctl", fake_launchctl)

    plist_path = launchd.install_service(persona="nell", nell_path=nell)

    assert plist_path.exists()
    target = launchd.launchctl_target("nell")
    assert calls == [
        ["bootout", target],
        ["bootstrap", launchd.launchctl_domain(), str(plist_path)],
        ["kickstart", "-k", target],
    ]


def test_install_service_raises_on_bootstrap_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    data_home = home / "data"
    (data_home / "personas" / "nell").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data_home))
    nell = _make_executable(tmp_path / "nell")

    def fake_launchctl(args: list[str]):
        if args[0] == "bootstrap":
            return _completed(5, stderr="boom")
        return _completed(0)

    monkeypatch.setattr(launchd, "run_launchctl", fake_launchctl)

    with pytest.raises(launchd.LaunchdCommandError, match="bootstrap.*boom"):
        launchd.install_service(persona="nell", nell_path=nell)


def test_uninstall_service_boots_out_and_removes_plist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    data_home = home / "data"
    (data_home / "personas" / "nell").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data_home))
    nell = _make_executable(tmp_path / "nell")
    plist_path = launchd.write_launchd_plist(persona="nell", nell_path=nell)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        launchd,
        "run_launchctl",
        lambda args: calls.append(args) or _completed(0),
    )

    removed_path = launchd.uninstall_service(persona="nell")

    assert removed_path == plist_path
    assert not plist_path.exists()
    assert calls == [["bootout", launchd.launchctl_target("nell")]]


def test_service_status_combines_plist_and_launchctl_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    data_home = home / "data"
    (data_home / "personas" / "nell").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data_home))
    nell = _make_executable(tmp_path / "nell")
    plist_path = launchd.write_launchd_plist(persona="nell", nell_path=nell)
    monkeypatch.setattr(
        launchd,
        "run_launchctl",
        lambda args: _completed(0, stdout="state = running"),
    )

    status = launchd.service_status(persona="nell")

    assert status.installed is True
    assert status.loaded is True
    assert status.plist_path == plist_path
    assert "running" in status.detail
