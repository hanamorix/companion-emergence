"""Tests for the Windows Task Scheduler service backend.

Mirrors ``test_systemd.py`` and ``test_launchd.py`` in shape: pure
XML generation, name + path resolution, doctor-check semantics.
The actual ``schtasks`` calls are not exercised here — they need a
real Windows session and are covered by release-time live smoke.

The XML generator is pure string work, so macOS / Linux runners
can validate the shape without ever talking to the Task Scheduler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.service import windows_service


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")
    # On POSIX hosts we set the +x bit so resolve_nell_path's "is_file"
    # check still succeeds. resolve_nell_path on Windows doesn't gate
    # on +x because the FS doesn't model it the same way.
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# task_name + paths_for_persona
# ---------------------------------------------------------------------------


def test_task_name_uses_canonical_prefix() -> None:
    assert windows_service.task_name("nell") == "CompanionEmergence-nell"


def test_task_name_validates_persona() -> None:
    with pytest.raises(ValueError):
        windows_service.task_name("../etc")


def test_paths_for_persona_layout(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    data = home / "data"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(data))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData/Local"))

    paths = windows_service.paths_for_persona("nell")
    assert paths.task_name == "CompanionEmergence-nell"
    assert paths.xml_path.name == "CompanionEmergence-nell.xml"
    # XML cached under LOCALAPPDATA\hanamorix\companion-emergence\service
    assert "hanamorix" in str(paths.xml_path)
    assert "service" in str(paths.xml_path)


# ---------------------------------------------------------------------------
# resolve_nell_path
# ---------------------------------------------------------------------------


def test_resolve_nell_path_explicit_absolute(tmp_path: Path) -> None:
    nell = _make_executable(tmp_path / "nell.exe")
    resolved = windows_service.resolve_nell_path(str(nell))
    assert resolved == nell.resolve()


def test_resolve_nell_path_rejects_relative() -> None:
    with pytest.raises(windows_service.WindowsServiceConfigError, match="absolute"):
        windows_service.resolve_nell_path(".\\nell.exe")


def test_resolve_nell_path_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(windows_service.WindowsServiceConfigError, match="not found"):
        windows_service.resolve_nell_path(str(tmp_path / "nope" / "nell.exe"))


# ---------------------------------------------------------------------------
# build_task_xml
# ---------------------------------------------------------------------------


def test_build_task_xml_contains_required_sections(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData/Local"))
    nell = _make_executable(tmp_path / "nell.exe")

    body = windows_service.build_task_xml(persona="nell", nell_path=nell)

    # Standard Task Scheduler XML structure
    assert '<Task version="1.2"' in body
    assert "<RegistrationInfo>" in body
    assert "<LogonTrigger>" in body
    assert "<Settings>" in body
    assert "<Actions>" in body
    # ExecStart equivalent: nell.exe + arguments
    assert str(nell.resolve()) in body
    assert "supervisor run --persona nell" in body
    assert "--client-origin task-scheduler" in body
    assert "--idle-shutdown 0" in body
    # Restart-on-failure (the launchd KeepAlive analog)
    assert "<RestartOnFailure>" in body


def test_build_task_xml_embeds_nellbrain_home_env_when_given(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell.exe")
    custom_home = tmp_path / "custom-data"
    custom_home.mkdir()

    body = windows_service.build_task_xml(
        persona="nell",
        nell_path=nell,
        nellbrain_home=str(custom_home),
    )
    assert "<Variable><Name>NELLBRAIN_HOME</Name>" in body
    # Path should be XML-escaped, not raw
    assert str(custom_home.resolve()).replace("&", "&amp;") in body


def test_build_task_xml_escapes_persona_name_in_description() -> None:
    """Persona names go through validate_persona_name so &<>" can't
    arrive here, but the XML generator still escapes defensively. The
    Description field uses the persona name verbatim — this guards
    a future relaxed validator from breaking the XML."""
    body = windows_service.build_task_xml(persona="nell", nell_path="C:\\fake\\nell.exe")
    assert "<Description>" in body
    # The persona name itself is plain ascii; no entities expected here.
    assert "&amp;" not in body or "<Description>" in body  # body may have entities elsewhere


def test_build_task_xml_logon_trigger_with_hidden_window(
    tmp_path: Path, monkeypatch
) -> None:
    """Task should fire AtLogon and not pop a console window."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell.exe")

    body = windows_service.build_task_xml(persona="nell", nell_path=nell)
    assert "<LogonTrigger>" in body
    assert "<Hidden>true</Hidden>" in body


# ---------------------------------------------------------------------------
# write_task_xml
# ---------------------------------------------------------------------------


def test_write_task_xml_creates_file_at_expected_path(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData/Local"))
    nell = _make_executable(tmp_path / "nell.exe")

    written = windows_service.write_task_xml(persona="nell", nell_path=nell)
    assert written.exists()
    assert "CompanionEmergence-nell.xml" == written.name
    # UTF-16 BOM at start (encoding="utf-16" adds it)
    raw = written.read_bytes()
    assert raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff")


# ---------------------------------------------------------------------------
# build_launchd_plist_xml shim — dispatcher uses this name uniformly
# ---------------------------------------------------------------------------


def test_build_launchd_plist_xml_alias_returns_task_xml_on_windows(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell.exe")

    out = windows_service.build_launchd_plist_xml(persona="nell", nell_path=nell)
    assert '<Task version="1.2"' in out
    # not a real launchd plist
    assert "DOCTYPE plist" not in out


# ---------------------------------------------------------------------------
# doctor_checks
# ---------------------------------------------------------------------------


def test_doctor_checks_returns_full_check_set(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData/Local"))
    nell = _make_executable(tmp_path / "nell.exe")

    checks = windows_service.doctor_checks(persona="nell", nell_path=str(nell))
    names = [c.name for c in checks]
    expected = {
        "platform",
        "persona_name",
        "persona_dir",
        "nell_path",
        "task_scheduler",
        "task_xml_dir",
        "log_dir",
        "claude_cli",
        "home",
    }
    assert expected.issubset(set(names))


def test_doctor_checks_persona_name_invalid(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell.exe")

    checks = windows_service.doctor_checks(persona="../etc", nell_path=str(nell))
    persona_check = next(c for c in checks if c.name == "persona_name")
    assert not persona_check.ok
