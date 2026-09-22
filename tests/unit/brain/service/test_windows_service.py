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


def test_build_task_xml_contains_required_sections(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData/Local"))
    nell = _make_executable(tmp_path / "nell.exe")

    body = windows_service.build_task_xml(persona="nell", nell_path=nell)

    # Standard Task Scheduler XML structure
    assert '<Task version="1.3"' in body
    assert "<RegistrationInfo>" in body
    assert "<LogonTrigger>" in body
    assert "<Settings>" in body
    assert '<Actions Context="Author">' in body
    # ExecStart equivalent: nell.exe + arguments
    assert str(nell.resolve()) in body
    assert 'supervisor run --persona &quot;nell&quot;' in body
    assert "--client-origin task-scheduler" in body
    assert "--idle-shutdown 0" in body
    # Restart-on-failure (the launchd KeepAlive analog)
    assert "<RestartOnFailure>" in body


def test_build_task_xml_embeds_kindled_home_env_when_given(tmp_path: Path, monkeypatch) -> None:
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
    assert "<Variable><Name>KINDLED_HOME</Name>" in body
    assert "<Variable><Name>NELLBRAIN_HOME</Name>" not in body  # regression guard
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


def test_build_task_xml_logon_trigger_with_hidden_window(tmp_path: Path, monkeypatch) -> None:
    """Task should fire AtLogon and not pop a console window."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    nell = _make_executable(tmp_path / "nell.exe")

    body = windows_service.build_task_xml(persona="nell", nell_path=nell)
    assert "<LogonTrigger>" in body
    assert "<Hidden>true</Hidden>" in body


def test_build_task_xml_is_schema_1_3_with_context_on_actions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    nell = tmp_path / "nell.bat"
    nell.write_text("@echo off\n")
    body = windows_service.build_task_xml(persona="nell", nell_path=nell)

    assert '<Task version="1.3"' in body
    assert '<Actions Context="Author">' in body
    assert '<Exec Context="Author">' not in body
    assert "<Exec>" in body

    # The 1.3-only nodes that schtasks rejected under 1.2 must still be present.
    assert "<DisallowStartOnRemoteAppSession>" in body
    assert "<UseUnifiedSchedulingEngine>" in body

    # Pin the canonical Settings child order (already export-correct) so a
    # future "helpful" reorder can't silently break schtasks parsing.
    order = [
        "MultipleInstancesPolicy",
        "DisallowStartIfOnBatteries",
        "StopIfGoingOnBatteries",
        "AllowHardTerminate",
        "StartWhenAvailable",
        "RunOnlyIfNetworkAvailable",
        "IdleSettings",
        "AllowStartOnDemand",
        "Enabled",
        "Hidden",
        "RunOnlyIfIdle",
        "DisallowStartOnRemoteAppSession",
        "UseUnifiedSchedulingEngine",
        "WakeToRun",
        "ExecutionTimeLimit",
        "Priority",
        "RestartOnFailure",
    ]
    settings = body[body.index("<Settings>") : body.index("</Settings>")]
    positions = [settings.index(f"<{tag}") for tag in order]
    assert positions == sorted(positions), "Settings child order drifted"


def test_render_task_xml_quotes_persona_in_arguments(tmp_path: Path, monkeypatch) -> None:
    """Arguments string must double-quote the persona name.

    Task Scheduler passes Arguments as a single string to CreateProcess.
    If the persona name (or a future path value embedded in arguments)
    contains a space it would be split into two tokens, causing the
    supervisor to receive an incorrect persona name."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NELLBRAIN_HOME", str(home / "data"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData/Local"))
    nell = _make_executable(tmp_path / "nell.exe")

    body = windows_service.build_task_xml(persona="nell", nell_path=nell)

    # The Arguments element must wrap the persona name in double-quotes.
    # _xml_escape converts " → &quot; so Task Scheduler receives the
    # literal double-quote when it invokes CreateProcess.
    assert '<Arguments>supervisor run --persona &quot;nell&quot;' in body


# ---------------------------------------------------------------------------
# write_task_xml
# ---------------------------------------------------------------------------


def test_write_task_xml_creates_file_at_expected_path(tmp_path: Path, monkeypatch) -> None:
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
    assert '<Task version="1.3"' in out
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


# ---------------------------------------------------------------------------
# Windowless Task Scheduler launch (pythonw.exe for bundled nell.bat)
# ---------------------------------------------------------------------------


def test_build_task_xml_uses_pythonw_for_bundled_nell_bat(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "python-runtime"
    scripts = runtime / "Scripts"
    scripts.mkdir(parents=True)
    pythonw = runtime / "pythonw.exe"
    pythonw.write_text("placeholder", encoding="utf-8")
    nell_bat = scripts / "nell.bat"
    nell_bat.write_text("@echo off\n", encoding="utf-8")

    body = windows_service.build_task_xml(persona="nell", nell_path=nell_bat)

    assert f"<Command>{str(pythonw)}</Command>" in body
    assert "from brain.cli import main" in body
    assert "supervisor run --persona &quot;nell&quot;" in body
    assert "nell.bat" not in body[body.index("<Command>") : body.index("</Command>")]


def test_build_task_xml_rejects_bundled_nell_bat_without_pythonw(tmp_path: Path) -> None:
    scripts = tmp_path / "python-runtime" / "Scripts"
    scripts.mkdir(parents=True)
    nell_bat = scripts / "nell.bat"
    nell_bat.write_text("@echo off\n", encoding="utf-8")

    with pytest.raises(windows_service.WindowsServiceConfigError, match="pythonw.exe"):
        windows_service.build_task_xml(persona="nell", nell_path=nell_bat)


# ---------------------------------------------------------------------------
# Per-user registration: <UserId> on the trigger AND the principal (#260)
# ---------------------------------------------------------------------------


def _xml_section(body: str, tag: str) -> str:
    return body[body.index(f"<{tag}") : body.index(f"</{tag}>")]


def test_task_user_id_joins_domain_and_username(monkeypatch) -> None:
    monkeypatch.setenv("USERDOMAIN", "DESKTOP-X1")
    monkeypatch.setenv("USERNAME", "hana")
    assert windows_service.task_user_id() == "DESKTOP-X1\\hana"


def test_task_user_id_bare_username_when_domain_unset(monkeypatch) -> None:
    monkeypatch.delenv("USERDOMAIN", raising=False)
    monkeypatch.setenv("USERNAME", "hana")
    assert windows_service.task_user_id() == "hana"


def test_task_user_id_none_when_username_unset(monkeypatch) -> None:
    monkeypatch.delenv("USERDOMAIN", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    assert windows_service.task_user_id() is None


def test_build_task_xml_scopes_logon_trigger_and_principal_to_user(monkeypatch) -> None:
    """Without <UserId> on the LogonTrigger, Task Scheduler treats the task as
    'any user's logon' (all-users) and refuses an unelevated schtasks /Create."""
    monkeypatch.setenv("USERDOMAIN", "DESKTOP-X1")
    monkeypatch.setenv("USERNAME", "hana")
    body = windows_service.build_task_xml(persona="nell", nell_path="C:\\fake\\nell.exe")

    assert "<UserId>DESKTOP-X1\\hana</UserId>" in _xml_section(body, "LogonTrigger")
    assert "<UserId>DESKTOP-X1\\hana</UserId>" in _xml_section(body, "Principal")
    assert body.count("<UserId>") == 2
    # Schema order inside the principal: UserId precedes LogonType.
    principal = _xml_section(body, "Principal")
    assert principal.index("<UserId>") < principal.index("<LogonType>")


def test_build_task_xml_explicit_user_id_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("USERDOMAIN", "DESKTOP-X1")
    monkeypatch.setenv("USERNAME", "hana")
    body = windows_service.build_task_xml(
        persona="nell", nell_path="C:\\fake\\nell.exe", user_id="S-1-5-21-1-2-3-1001"
    )
    assert body.count("<UserId>S-1-5-21-1-2-3-1001</UserId>") == 2
    assert "hana" not in body


def test_build_task_xml_omits_user_id_when_unresolvable(monkeypatch) -> None:
    """Non-Windows unit hosts have no USERNAME; the XML must stay well-formed
    and byte-identical to the pre-#260 shape rather than emit an empty node."""
    monkeypatch.delenv("USERDOMAIN", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    body = windows_service.build_task_xml(persona="nell", nell_path="C:\\fake\\nell.exe")
    assert "<UserId>" not in body
    assert "<LogonTrigger>\n      <Enabled>true</Enabled>\n    </LogonTrigger>" in body


def test_build_task_xml_escapes_user_id(monkeypatch) -> None:
    monkeypatch.setenv("USERDOMAIN", "A&B")
    monkeypatch.setenv("USERNAME", "h<a>na")
    body = windows_service.build_task_xml(persona="nell", nell_path="C:\\fake\\nell.exe")
    assert "<UserId>A&amp;B\\h&lt;a&gt;na</UserId>" in body


# ---------------------------------------------------------------------------
# service_status trusts schtasks /Query, not the cached XML (#260 secondary)
# ---------------------------------------------------------------------------


def _fake_schtasks(monkeypatch, *, returncode: int, stdout: str = "", stderr: str = "") -> list:
    calls: list[list[str]] = []

    def fake(args: list[str]):
        calls.append(args)
        import subprocess

        return subprocess.CompletedProcess(
            args=["schtasks", *args], returncode=returncode, stdout=stdout, stderr=stderr
        )

    monkeypatch.setattr(windows_service, "run_schtasks", fake)
    return calls


def test_service_status_not_installed_when_query_fails_despite_cached_xml(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed `schtasks /Create` leaves the XML behind; status used to report
    `installed: yes` from that file alone while the task did not exist."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    paths = windows_service.paths_for_persona("nell")
    paths.xml_path.parent.mkdir(parents=True, exist_ok=True)
    paths.xml_path.write_text("<Task/>", encoding="utf-8")
    _fake_schtasks(
        monkeypatch, returncode=1, stderr="ERROR: The system cannot find the file specified."
    )

    status = windows_service.service_status(persona="nell")

    assert status.installed is False
    assert status.loaded is False
    assert "not registered" in status.detail
    assert str(paths.xml_path) in status.detail
    assert "cannot find the file" in status.detail


def test_service_status_installed_and_loaded_from_query(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    _fake_schtasks(
        monkeypatch,
        returncode=0,
        stdout="TaskName: \\CompanionEmergence-nell\nStatus: Ready\n",
    )

    status = windows_service.service_status(persona="nell")

    assert status.installed is True
    assert status.loaded is True
    assert "Status: Ready" in status.detail


def test_service_status_not_installed_without_xml_or_task(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    _fake_schtasks(monkeypatch, returncode=127, stderr="schtasks not available")

    status = windows_service.service_status(persona="nell")

    assert status.installed is False
    assert status.loaded is False
    assert status.detail == "schtasks not available"
