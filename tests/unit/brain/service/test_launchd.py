from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from brain.service import launchd


def _make_executable(path: Path) -> Path:
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
    assert by_name["claude_cli"].ok is True


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
