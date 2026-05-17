from __future__ import annotations

import plistlib
from pathlib import Path

from brain import cli
from brain.service.launchd import DoctorCheck, ServiceStatus


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class _FakeBackendConfigError(ValueError):
    pass


class _FakeBackendCommandError(RuntimeError):
    pass


class _FakeBackend:
    ConfigError = _FakeBackendConfigError
    CommandError = _FakeBackendCommandError

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.calls: list[tuple[str, dict]] = []

    def resolve_nell_path(self, explicit=None):
        self.calls.append(("resolve_nell_path", {"explicit": explicit}))
        return Path(explicit or self.tmp_path / "nell")

    def doctor_checks(self, *, persona, nell_path, env_path):
        self.calls.append(
            ("doctor_checks", {"persona": persona, "nell_path": nell_path, "env_path": env_path})
        )
        return [DoctorCheck("backend", True, "fake backend ok")]

    def install_service(self, *, persona, nell_path, env_path, nellbrain_home):
        self.calls.append(
            (
                "install_service",
                {
                    "persona": persona,
                    "nell_path": nell_path,
                    "env_path": env_path,
                    "nellbrain_home": nellbrain_home,
                },
            )
        )
        return self.tmp_path / "fake.service"

    def uninstall_service(self, *, persona, keep_plist):
        self.calls.append(("uninstall_service", {"persona": persona, "keep_plist": keep_plist}))
        return self.tmp_path / "fake.service"

    def service_status(self, *, persona):
        self.calls.append(("service_status", {"persona": persona}))
        return ServiceStatus(
            label=f"fake.{persona}",
            plist_path=self.tmp_path / "fake.service",
            installed=True,
            loaded=True,
            detail="fake running",
        )

    def render_service_config(self, *, persona, nell_path, env_path, nellbrain_home):
        self.calls.append(
            (
                "render_service_config",
                {
                    "persona": persona,
                    "nell_path": nell_path,
                    "env_path": env_path,
                    "nellbrain_home": nellbrain_home,
                },
            )
        )
        return "fake dry-run config\n"


def test_service_print_plist_outputs_launchagent_xml(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
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
    assert parsed["EnvironmentVariables"]["KINDLED_HOME"] == str(data_home.resolve())


def test_service_print_plist_invalid_nell_path_returns_1(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
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


def test_service_doctor_uses_current_backend(monkeypatch, tmp_path: Path, capsys) -> None:
    backend = _FakeBackend(tmp_path)
    monkeypatch.setattr("brain.service.current_backend", lambda: backend)

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

    assert rc == 0
    assert (
        "doctor_checks",
        {"persona": "nell", "nell_path": "/tmp/nell", "env_path": "/tmp/bin"},
    ) in backend.calls
    assert "fake backend ok" in capsys.readouterr().out


def test_service_lifecycle_commands_use_current_backend(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    backend = _FakeBackend(tmp_path)
    monkeypatch.setattr("brain.service.current_backend", lambda: backend)

    assert cli.main(["service", "install", "--persona", "nell", "--nell-path", "/tmp/nell"]) == 0
    assert cli.main(["service", "uninstall", "--persona", "nell", "--keep-plist"]) == 0
    assert cli.main(["service", "status", "--persona", "nell"]) == 0

    names = [name for name, _ in backend.calls]
    assert "install_service" in names
    assert "uninstall_service" in names
    assert "service_status" in names
    out = capsys.readouterr().out
    assert "service uninstalled: config kept at" in out
    assert "config: " in out
    assert "fake running" in out


def test_service_print_plist_is_cleanly_unsupported_off_macos(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.platform", "linux")

    rc = cli.main(["service", "print-plist", "--persona", "nell", "--nell-path", "/tmp/nell"])

    assert rc == 1
    assert "macOS" in capsys.readouterr().err


def test_service_install_dry_run_prints_plist_without_bootstrap(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    nell = _make_executable(tmp_path / "nell")

    backend = _FakeBackend(tmp_path)
    monkeypatch.setattr("brain.service.current_backend", lambda: backend)

    rc = cli.main(
        [
            "service",
            "install",
            "--persona",
            "nell",
            "--nell-path",
            str(nell),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert "fake dry-run config" in capsys.readouterr().out


def test_service_install_calls_launchd_install(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    nell = _make_executable(tmp_path / "nell")

    backend = _FakeBackend(tmp_path)
    monkeypatch.setattr("brain.service.current_backend", lambda: backend)

    rc = cli.main(
        [
            "service",
            "install",
            "--persona",
            "nell",
            "--nell-path",
            str(nell),
            "--env-path",
            "/tmp/bin",
        ]
    )

    assert rc == 0
    install_call = (
        "install_service",
        {
            "persona": "nell",
            "nell_path": nell.resolve(),
            "env_path": "/tmp/bin",
            "nellbrain_home": None,
        },
    )
    assert install_call in backend.calls
    assert "service installed" in capsys.readouterr().out


def test_service_uninstall_calls_launchd_uninstall(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("sys.platform", "darwin")

    backend = _FakeBackend(tmp_path)
    monkeypatch.setattr("brain.service.current_backend", lambda: backend)

    rc = cli.main(["service", "uninstall", "--persona", "nell", "--keep-plist"])

    assert rc == 0
    assert ("uninstall_service", {"persona": "nell", "keep_plist": True}) in backend.calls
    assert "config kept" in capsys.readouterr().out


def test_service_status_prints_launchd_state(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("sys.platform", "darwin")

    backend = _FakeBackend(tmp_path)
    monkeypatch.setattr("brain.service.current_backend", lambda: backend)

    rc = cli.main(["service", "status", "--persona", "nell"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "installed: yes" in out
    assert "loaded: yes" in out
    assert "fake running" in out
