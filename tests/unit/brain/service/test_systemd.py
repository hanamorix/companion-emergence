"""Tests for the Linux systemd-user service backend.

Mirrors ``test_launchd.py`` in shape: pure-function unit-file
generation, name + path resolution, doctor-check semantics. The
``systemctl --user`` calls themselves are not exercised here —
they need a real Linux user session and are covered by the
release-time live smoke instead.

These tests run on every platform: the unit-file generator is
pure string work, so macOS / Windows runners can validate the
shape without ever talking to systemd.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from brain.service import systemd

# systemd-user backend is Linux-only at runtime. The unit-file
# generator is pure string work and runs on macOS just fine via the
# monkeypatched HOME, but Windows ``Path.home()`` reads
# ``USERPROFILE`` regardless of $HOME, so the fake-home pattern these
# tests use can't be honored. Skip on Windows — the dispatcher's
# Windows branch returns the windows_service stub anyway.
pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="systemd backend is Linux-only; tests rely on POSIX $HOME semantics",
)


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# service_unit_name + paths_for_persona
# ---------------------------------------------------------------------------


def test_service_unit_name_uses_canonical_prefix() -> None:
    assert systemd.service_unit_name("nell") == "companion-emergence-nell.service"


def test_service_unit_name_validates_persona() -> None:
    # validate_persona_name rejects path traversal etc — the same rule
    # macOS uses, so unit names can never contain shell metacharacters.
    with pytest.raises(ValueError):
        systemd.service_unit_name("../etc")


def test_paths_for_persona_assembles_expected_layout(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    data = home / "data"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data))

    paths = systemd.paths_for_persona("nell")
    assert paths.unit_name == "companion-emergence-nell.service"
    assert paths.unit_path == home / ".config/systemd/user/companion-emergence-nell.service"
    assert paths.stdout_path.name == "supervisor-nell.out.log"
    assert paths.stderr_path.name == "supervisor-nell.err.log"


# ---------------------------------------------------------------------------
# resolve_nell_path
# ---------------------------------------------------------------------------


def test_resolve_nell_path_explicit_absolute(tmp_path: Path) -> None:
    nell = _make_executable(tmp_path / "nell")
    resolved = systemd.resolve_nell_path(str(nell))
    assert resolved == nell.resolve()


def test_resolve_nell_path_rejects_relative(tmp_path: Path) -> None:
    with pytest.raises(systemd.SystemdConfigError, match="absolute"):
        systemd.resolve_nell_path("./nell")


def test_resolve_nell_path_rejects_non_executable(tmp_path: Path) -> None:
    not_exec = tmp_path / "nell"
    not_exec.write_text("#!/bin/sh\nexit 0\n")
    not_exec.chmod(0o644)
    with pytest.raises(systemd.SystemdConfigError, match="not executable"):
        systemd.resolve_nell_path(str(not_exec))


# ---------------------------------------------------------------------------
# build_systemd_unit_text
# ---------------------------------------------------------------------------


def test_build_systemd_unit_text_contains_required_sections(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    nell = _make_executable(tmp_path / "nell")

    body = systemd.build_systemd_unit_text(persona="nell", nell_path=nell)

    # Standard systemd unit-file structure
    assert "[Unit]" in body
    assert "[Service]" in body
    assert "[Install]" in body
    # Service uses the foreground-supervisor entry point
    assert (
        f"ExecStart={nell.resolve()} supervisor run --persona nell "
        "--client-origin systemd --idle-shutdown 0"
    ) in body
    # KeepAlive equivalent
    assert "Restart=on-failure" in body
    # Login-time start
    assert "WantedBy=default.target" in body


def test_build_systemd_unit_text_embeds_nellbrain_home_env_when_given(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell")
    custom_home = tmp_path / "custom-data"
    custom_home.mkdir()

    body = systemd.build_systemd_unit_text(
        persona="nell",
        nell_path=nell,
        nellbrain_home=str(custom_home),
    )
    assert f'Environment="KINDLED_HOME={custom_home.resolve()}"' in body
    assert 'Environment="NELLBRAIN_HOME=' not in body


def test_systemd_unit_emits_kindled_not_nellbrain_env_line(tmp_path, monkeypatch):
    """v0.0.13 rename: newly built systemd units must not embed NELLBRAIN_HOME."""
    custom_home = tmp_path / "custom"
    custom_home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    nell = _make_executable(tmp_path / "nell")
    body = systemd.build_systemd_unit_text(
        persona="nell",
        nell_path=nell,
        nellbrain_home=str(custom_home),
    )
    assert 'Environment="KINDLED_HOME=' in body
    assert 'Environment="NELLBRAIN_HOME=' not in body


def test_build_systemd_unit_text_embeds_path_environment(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell")

    body = systemd.build_systemd_unit_text(
        persona="nell",
        nell_path=nell,
        env_path="/custom/bin:/usr/bin",
    )
    assert 'Environment="PATH=/custom/bin:/usr/bin"' in body


def test_build_systemd_unit_text_default_path_includes_user_local_bin(
    tmp_path: Path, monkeypatch
) -> None:
    """Default PATH must include ~/.local/bin so claude is reachable.

    Mirrors the launchd fix for the same issue on macOS.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell")

    body = systemd.build_systemd_unit_text(persona="nell", nell_path=nell)
    assert ".local/bin" in body


def test_build_systemd_unit_text_rejects_missing_nell_binary(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "nell"
    with pytest.raises(systemd.SystemdConfigError, match="not found"):
        systemd.build_systemd_unit_text(persona="nell", nell_path=missing)


# ---------------------------------------------------------------------------
# write_systemd_unit
# ---------------------------------------------------------------------------


def test_write_systemd_unit_creates_file_at_expected_path(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    nell = _make_executable(tmp_path / "nell")

    written = systemd.write_systemd_unit(persona="nell", nell_path=nell)
    assert written == home / ".config/systemd/user/companion-emergence-nell.service"
    assert written.exists()
    assert "[Service]" in written.read_text()


# ---------------------------------------------------------------------------
# build_launchd_plist_xml shim — dispatcher uses this name uniformly
# ---------------------------------------------------------------------------


def test_build_launchd_plist_xml_alias_returns_unit_text_on_linux(
    tmp_path: Path, monkeypatch
) -> None:
    """The dispatcher calls ``build_launchd_plist_xml`` regardless of OS;
    on Linux it must return systemd unit text, not plist XML."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell")

    out = systemd.build_launchd_plist_xml(persona="nell", nell_path=nell)
    assert "[Unit]" in out
    assert "<?xml" not in out  # not a plist


# ---------------------------------------------------------------------------
# doctor_checks — non-mutating preflight
# ---------------------------------------------------------------------------


def test_doctor_checks_returns_full_check_set(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    nell = _make_executable(tmp_path / "nell")

    checks = systemd.doctor_checks(persona="nell", nell_path=str(nell))
    names = [c.name for c in checks]
    # Same set the operator expects from launchd doctor, adjusted for Linux.
    expected = {
        "platform",
        "persona_name",
        "persona_dir",
        "nell_path",
        "systemd_user",
        "user_unit_dir",
        "log_dir",
        "claude_cli",
        "home",
    }
    assert expected.issubset(set(names))


def test_doctor_checks_persona_name_invalid(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell")

    checks = systemd.doctor_checks(persona="../etc", nell_path=str(nell))
    persona_check = next(c for c in checks if c.name == "persona_name")
    assert not persona_check.ok
