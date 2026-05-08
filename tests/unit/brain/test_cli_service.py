from __future__ import annotations

import plistlib
from pathlib import Path

from brain import cli
from brain.service.launchd import DoctorCheck


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_service_print_plist_outputs_launchagent_xml(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    home = tmp_path / "home"
    data_home = home / "data"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data_home))
    nell = _make_executable(tmp_path / "nell")

    rc = cli.main(
        [
            "service",
            "print-plist",
            "--persona",
            "nell",
            "--nell-path",
            str(nell),
            "--env-path",
            "/custom/bin:/usr/bin:/bin",
            "--nellbrain-home",
            str(data_home),
        ]
    )

    assert rc == 0
    parsed = plistlib.loads(capsys.readouterr().out.encode("utf-8"))
    assert parsed["Label"] == "com.companion-emergence.supervisor.nell"
    assert parsed["ProgramArguments"][:3] == [str(nell.resolve()), "supervisor", "run"]
    assert parsed["EnvironmentVariables"]["NELLBRAIN_HOME"] == str(data_home.resolve())


def test_service_print_plist_invalid_nell_path_returns_1(capsys) -> None:
    rc = cli.main(
        [
            "service",
            "print-plist",
            "--persona",
            "nell",
            "--nell-path",
            "relative/nell",
        ]
    )

    assert rc == 1
    assert "--nell-path must be absolute" in capsys.readouterr().err


def test_service_doctor_prints_checks_and_returns_1_on_failure(monkeypatch, capsys) -> None:
    def fake_checks(*, persona, nell_path, env_path):
        assert persona == "nell"
        assert nell_path == "/tmp/nell"
        assert env_path == "/tmp/bin"
        return [
            DoctorCheck("platform", True, "macOS launchd available"),
            DoctorCheck("claude_cli", False, "claude not found in launchd PATH"),
        ]

    monkeypatch.setattr("brain.service.launchd.doctor_checks", fake_checks)

    rc = cli.main(
        [
            "service",
            "doctor",
            "--persona",
            "nell",
            "--nell-path",
            "/tmp/nell",
            "--env-path",
            "/tmp/bin",
        ]
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "ok   platform" in out
    assert "FAIL claude_cli" in out
