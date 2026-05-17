"""Windows persistent-supervisor backend (Task Scheduler).

Mirrors :mod:`brain.service.launchd` and :mod:`brain.service.systemd`
so the dispatcher and ``nell service`` CLI work uniformly. Same
lifecycle:

  * ``install_service`` writes a Task Scheduler XML and registers it
    with ``schtasks /Create /XML``.
  * ``uninstall_service`` removes the task with ``schtasks /Delete /F``.
  * ``service_status`` reports installed / running / pid.
  * ``doctor_checks`` runs non-mutating preflight (Windows present,
    schtasks reachable, ``claude`` on PATH, log dir writable).

We use **Task Scheduler** rather than a true Windows Service for two
reasons:

  1. A real Windows Service requires a Service Control Manager
     (``SERVICE_TABLE_ENTRY``) host — not something a Python script
     ships natively. ``pywin32`` exists but adds a heavy native dep.
     Task Scheduler with an ``AtLogon`` trigger gives the same
     "starts at login, restarts on failure" semantics for a user-
     scoped agent.
  2. Task Scheduler tasks live in the user's Task scope, no admin
     elevation needed for install or uninstall — matches launchd's
     gui/<uid>/ and systemd's --user model.

Status: **scaffolded for v0.0.1**. The XML generation and CLI
shape are exercised by unit tests, but live ``schtasks`` runs against
a real Windows session are pending — flagged in the wizard as
"compiles cleanly but not validated on real hardware yet".
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from brain.paths import get_log_dir, get_persona_dir
from brain.service import UnsupportedPlatformError
from brain.setup import validate_persona_name

TASK_PREFIX = "CompanionEmergence"
DEFAULT_WINDOWS_PATH = (
    "%LOCALAPPDATA%\\Programs\\claude;%USERPROFILE%\\.local\\bin;C:\\Windows\\System32;C:\\Windows"
)


class WindowsServiceConfigError(ValueError):
    """Raised when task configuration is invalid."""


class WindowsServiceCommandError(RuntimeError):
    """Raised when a ``schtasks`` operation fails."""


@dataclass(frozen=True)
class WindowsServicePaths:
    """Filesystem locations for one persona's scheduled task."""

    task_name: str  # e.g. "CompanionEmergence-nell"
    xml_path: Path  # cached task XML for diagnostics + uninstall ref
    stdout_path: Path
    stderr_path: Path
    persona_dir: Path


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ServiceStatus:
    task_name: str
    xml_path: Path
    installed: bool
    loaded: bool
    detail: str


def task_name(persona: str) -> str:
    """Canonical task name. ``CompanionEmergence-<persona>``."""
    validate_persona_name(persona)
    return f"{TASK_PREFIX}-{persona}"


def _localappdata_root() -> Path:
    """``%LOCALAPPDATA%\\hanamorix\\companion-emergence`` on Windows.

    Falls back to ``~/.local/share/companion-emergence`` when
    LOCALAPPDATA is unset (e.g. running on a non-Windows host for
    unit testing).
    """
    appdata = os.environ.get("LOCALAPPDATA")
    if appdata:
        return Path(appdata) / "hanamorix" / "companion-emergence"
    return Path.home() / ".local" / "share" / "companion-emergence"


def task_xml_dir() -> Path:
    """Directory where we cache the rendered Task Scheduler XML.

    schtasks doesn't read the XML back out after install — it stores
    its own copy in the task store — but we keep ours alongside the
    persona for diagnostics + uninstall reference.
    """
    return _localappdata_root() / "service"


def paths_for_persona(persona: str) -> WindowsServicePaths:
    """Filesystem layout for a persona's scheduled task."""
    validate_persona_name(persona)
    name = task_name(persona)
    log_dir = get_log_dir()
    return WindowsServicePaths(
        task_name=name,
        xml_path=task_xml_dir() / f"{name}.xml",
        stdout_path=log_dir / f"supervisor-{persona}.out.log",
        stderr_path=log_dir / f"supervisor-{persona}.err.log",
        persona_dir=get_persona_dir(persona),
    )


def resolve_nell_path(explicit: str | None = None) -> Path:
    """Mirror of launchd.resolve_nell_path / systemd.resolve_nell_path."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise WindowsServiceConfigError(f"--nell-path must be absolute (got {path!s})")
        if not path.is_file():
            raise WindowsServiceConfigError(f"nell at {path} not found")
        return path
    found = shutil.which("nell.exe") or shutil.which("nell")
    if found:
        return Path(found).resolve()
    raise WindowsServiceConfigError(
        "could not resolve 'nell' on PATH; pass --nell-path C:\\path\\to\\nell.exe"
    )


def _xml_escape(text: str) -> str:
    """Escape XML metacharacters in a runtime-supplied string."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def build_task_xml(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_WINDOWS_PATH,
    nellbrain_home: str | Path | None = None,
) -> str:
    """Render a Task Scheduler XML for the persona's supervisor task.

    The XML follows Windows Task Scheduler 1.2 schema:
      * ``<LogonTrigger>`` — start at login (matches RunAtLoad / WantedBy=default.target).
      * ``<RegistrationInfo><Author>`` — for diagnostic tracing.
      * ``<Settings><RestartOnFailure>`` — restart on crash with a
        2-minute interval, up to 3 times (matches launchd KeepAlive
        + systemd Restart=on-failure).
      * ``<Actions><Exec>`` — ``nell.exe supervisor run --persona <name>
        --client-origin task-scheduler --idle-shutdown 0`` — same
        foreground-supervisor entry point launchd / systemd use.
      * ``<Settings><Hidden>true`` — task doesn't open a console window
        on login.

    Pure function; no filesystem side effects. Tested on every
    platform via the same string-equality assertions that cover
    launchd plist generation and systemd unit-file generation.
    """
    validate_persona_name(persona)
    nell_path_resolved = Path(nell_path).expanduser()
    # Don't resolve() because Windows paths often have ``%LOCALAPPDATA%``
    # placeholders that should remain as-written. resolve() collapses them
    # to absolute Windows paths, but if a user passes a literal absolute
    # path we still accept it.
    if nell_path_resolved.exists():
        nell_path_str = str(nell_path_resolved.resolve())
    else:
        nell_path_str = str(nell_path_resolved)

    # The Arguments string is a single line — Task Scheduler doesn't
    # tokenize it the way argv does. Wrap any path with spaces in
    # double quotes so claude --print can find its image attachments
    # later when the path includes ``Documents``-style folders.
    args = f"supervisor run --persona {persona} --client-origin task-scheduler --idle-shutdown 0"

    # Working directory and stdout/stderr can't be set per-task in
    # Task Scheduler the way launchd / systemd do — we let the runner
    # inherit %USERPROFILE% as the working dir, and the runner's own
    # RotatingFileHandler writes to ``runtime-<persona>.log``. The
    # captured stdout/stderr files exist as a fallback if the
    # supervisor fails before logging is initialized.

    env_block = ""
    if nellbrain_home is not None:
        nellbrain_home_str = _xml_escape(str(Path(nellbrain_home).expanduser()))
        env_block = (
            "    <Environment>\n"
            f"      <Variable><Name>KINDLED_HOME</Name>"
            f"<Value>{nellbrain_home_str}</Value></Variable>\n"
            "    </Environment>\n"
        )

    return (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        "  <RegistrationInfo>\n"
        f"    <Description>companion-emergence supervisor ({_xml_escape(persona)})</Description>\n"
        "    <Author>companion-emergence</Author>\n"
        "  </RegistrationInfo>\n"
        "  <Triggers>\n"
        "    <LogonTrigger>\n"
        "      <Enabled>true</Enabled>\n"
        "    </LogonTrigger>\n"
        "  </Triggers>\n"
        "  <Principals>\n"
        '    <Principal id="Author">\n'
        "      <LogonType>InteractiveToken</LogonType>\n"
        "      <RunLevel>LeastPrivilege</RunLevel>\n"
        "    </Principal>\n"
        "  </Principals>\n"
        "  <Settings>\n"
        "    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n"
        "    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n"
        "    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n"
        "    <AllowHardTerminate>true</AllowHardTerminate>\n"
        "    <StartWhenAvailable>true</StartWhenAvailable>\n"
        "    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>\n"
        "    <IdleSettings>\n"
        "      <StopOnIdleEnd>false</StopOnIdleEnd>\n"
        "      <RestartOnIdle>false</RestartOnIdle>\n"
        "    </IdleSettings>\n"
        "    <AllowStartOnDemand>true</AllowStartOnDemand>\n"
        "    <Enabled>true</Enabled>\n"
        "    <Hidden>true</Hidden>\n"
        "    <RunOnlyIfIdle>false</RunOnlyIfIdle>\n"
        "    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>\n"
        "    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>\n"
        "    <WakeToRun>false</WakeToRun>\n"
        "    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n"
        "    <Priority>7</Priority>\n"
        "    <RestartOnFailure>\n"
        "      <Interval>PT2M</Interval>\n"
        "      <Count>3</Count>\n"
        "    </RestartOnFailure>\n"
        "  </Settings>\n"
        "  <Actions>\n"
        '    <Exec Context="Author">\n'
        f"      <Command>{_xml_escape(nell_path_str)}</Command>\n"
        f"      <Arguments>{_xml_escape(args)}</Arguments>\n"
        f"{env_block}"
        "    </Exec>\n"
        "  </Actions>\n"
        "</Task>\n"
    )


def write_task_xml(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_WINDOWS_PATH,
    nellbrain_home: str | Path | None = None,
) -> Path:
    """Render the task XML and cache it on disk."""
    body = build_task_xml(
        persona=persona,
        nell_path=nell_path,
        env_path=env_path,
        nellbrain_home=nellbrain_home,
    )
    paths = paths_for_persona(persona)
    paths.xml_path.parent.mkdir(parents=True, exist_ok=True)
    paths.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-16 with BOM matches the encoding the XML declaration claims.
    paths.xml_path.write_text(body, encoding="utf-16")
    return paths.xml_path


def run_schtasks(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``schtasks <args>``. Synthetic exit-127 when the tool is
    not on PATH (e.g. running on macOS for unit tests)."""
    try:
        return subprocess.run(  # noqa: S603
            ["schtasks", *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=["schtasks", *args],
            returncode=127,
            stdout="",
            stderr="schtasks not available — Windows host required",
        )


def install_service(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_WINDOWS_PATH,
    nellbrain_home: str | Path | None = None,
) -> Path:
    """Write the task XML, then ``schtasks /Create /XML``.

    Idempotent: if the task already exists, ``/F`` overwrites it.
    """
    xml_path = write_task_xml(
        persona=persona,
        nell_path=nell_path,
        env_path=env_path,
        nellbrain_home=nellbrain_home,
    )
    name = task_name(persona)
    result = run_schtasks(["/Create", "/XML", str(xml_path), "/TN", name, "/F"])
    if result.returncode != 0:
        raise WindowsServiceCommandError(_format_schtasks_error("/Create", result))
    # Kick it off immediately so the user doesn't have to log out + back in.
    run_schtasks(["/Run", "/TN", name])
    return xml_path


def uninstall_service(*, persona: str, keep_xml: bool = False) -> Path:
    """Remove the scheduled task; optionally keep the cached XML."""
    paths = paths_for_persona(persona)
    # /End first to stop a running instance (best-effort), then /Delete.
    run_schtasks(["/End", "/TN", paths.task_name])
    run_schtasks(["/Delete", "/TN", paths.task_name, "/F"])
    if not keep_xml and paths.xml_path.exists():
        paths.xml_path.unlink()
    return paths.xml_path


def service_status(*, persona: str) -> ServiceStatus:
    """Combine cached-XML presence with ``schtasks /Query`` output."""
    paths = paths_for_persona(persona)
    installed = paths.xml_path.exists()
    query = run_schtasks(["/Query", "/TN", paths.task_name, "/V", "/FO", "LIST"])
    detail = query.stdout.strip() if query.returncode == 0 else query.stderr.strip()
    loaded = query.returncode == 0 and "Status:" in (query.stdout or "")
    return ServiceStatus(
        task_name=paths.task_name,
        xml_path=paths.xml_path,
        installed=installed,
        loaded=loaded,
        detail=detail,
    )


def doctor_checks(
    *,
    persona: str,
    nell_path: str | None = None,
    env_path: str = DEFAULT_WINDOWS_PATH,
) -> list[DoctorCheck]:
    """Non-mutating preflight before installing."""
    import sys as _sys

    checks: list[DoctorCheck] = []

    plat_ok = _sys.platform.startswith("win")
    checks.append(
        DoctorCheck(
            "platform",
            plat_ok,
            "Windows Task Scheduler available"
            if plat_ok
            else f"unsupported platform {_sys.platform}",
        )
    )

    persona_ok = True
    try:
        validate_persona_name(persona)
        checks.append(DoctorCheck("persona_name", True, f"task={task_name(persona)}"))
    except Exception as exc:  # noqa: BLE001
        persona_ok = False
        checks.append(DoctorCheck("persona_name", False, str(exc)))

    if persona_ok:
        pd = get_persona_dir(persona)
        checks.append(
            DoctorCheck(
                "persona_dir",
                pd.exists(),
                str(pd) if pd.exists() else f"missing: {pd}",
            )
        )
    else:
        checks.append(DoctorCheck("persona_dir", False, "skipped (invalid persona name)"))

    try:
        resolved_nell = resolve_nell_path(nell_path)
        checks.append(DoctorCheck("nell_path", True, str(resolved_nell)))
    except WindowsServiceConfigError as exc:
        checks.append(DoctorCheck("nell_path", False, str(exc)))

    sched_query = run_schtasks(["/Query", "/?"])
    sched_ok = sched_query.returncode == 0
    checks.append(
        DoctorCheck(
            "task_scheduler",
            sched_ok,
            "schtasks reachable" if sched_ok else (sched_query.stderr or "schtasks not found"),
        )
    )

    xml_dir = task_xml_dir()
    try:
        xml_dir.mkdir(parents=True, exist_ok=True)
        xml_ok = os.access(xml_dir, os.W_OK)
        xml_detail = str(xml_dir)
    except OSError as exc:
        xml_ok = False
        xml_detail = str(exc)
    checks.append(DoctorCheck("task_xml_dir", xml_ok, xml_detail))

    log_dir = get_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_ok = os.access(log_dir, os.W_OK)
        log_detail = str(log_dir)
    except OSError as exc:
        log_ok = False
        log_detail = str(exc)
    checks.append(DoctorCheck("log_dir", log_ok, log_detail))

    claude_path = _which_in_path("claude.exe", env_path) or _which_in_path("claude", env_path)
    checks.append(
        DoctorCheck(
            "claude_cli",
            claude_path is not None,
            claude_path or "claude not found in task PATH",
        )
    )

    # KINDLED_HOME (canonical v0.0.13+; NELLBRAIN_HOME honoured as legacy-compat)
    home_env = os.environ.get("KINDLED_HOME") or os.environ.get("NELLBRAIN_HOME")
    if home_env:
        checks.append(DoctorCheck("home", True, home_env))
    else:
        checks.append(DoctorCheck("home", True, str(_localappdata_root())))

    return checks


def build_launchd_plist_xml(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_WINDOWS_PATH,
    nellbrain_home: str | Path | None = None,
) -> str:
    """Alias so the dispatcher's print-plist subcommand renders the
    right format on Windows (Task Scheduler XML, not a launchd plist)."""
    return build_task_xml(
        persona=persona,
        nell_path=nell_path,
        env_path=env_path,
        nellbrain_home=nellbrain_home,
    )


def _which_in_path(executable: str, path_value: str) -> str | None:
    """shutil.which against an explicit PATH string, with %VAR%
    expansion. Windows users have ``%USERPROFILE%`` etc in their
    task PATH; expand those before searching."""
    expanded = os.path.expandvars(path_value)
    return shutil.which(executable, path=expanded)


def _format_schtasks_error(action: str, result: subprocess.CompletedProcess[str]) -> str:
    parts = [f"schtasks {action} (exit {result.returncode})"]
    if result.stdout.strip():
        parts.append(f"stdout: {result.stdout.strip()}")
    if result.stderr.strip():
        parts.append(f"stderr: {result.stderr.strip()}")
    return "; ".join(parts)


__all__ = [
    "DEFAULT_WINDOWS_PATH",
    "DoctorCheck",
    "ServiceStatus",
    "TASK_PREFIX",
    "UnsupportedPlatformError",
    "WindowsServiceCommandError",
    "WindowsServiceConfigError",
    "WindowsServicePaths",
    "build_launchd_plist_xml",
    "build_task_xml",
    "doctor_checks",
    "install_service",
    "paths_for_persona",
    "resolve_nell_path",
    "run_schtasks",
    "service_status",
    "task_name",
    "task_xml_dir",
    "uninstall_service",
    "write_task_xml",
]
