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

Status: the XML is now Task Scheduler **schema-1.3-correct** (version
bumped to 1.3 so the ``DisallowStartOnRemoteAppSession`` /
``UseUnifiedSchedulingEngine`` nodes are legal, and ``Context`` lives
on ``<Actions>`` where the schema requires it). The XML generation and
CLI shape are exercised by unit tests; a live ``schtasks /Create`` run
against a real Windows session is pending validation by a Windows user.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from brain.paths import get_log_dir, get_persona_dir
from brain.service import UnsupportedPlatformError
from brain.setup import validate_persona_name
from brain.utils.atomic_write import save_with_backup_text

logger = logging.getLogger(__name__)

TASK_PREFIX = "CompanionEmergence"

# RestartOnFailure (D26): the bridge's own runner spaces the restarts of a bridge that keeps failing (30 s up to 10 min,
# ``brain/bridge/start_backoff.py``), so the scheduler only has to keep restarting, as soon and as often as it allows:
# the shortest interval it accepts (1 minute) and the largest count (255); MS-TSCH gives Count as 1 to 255 and has no
# "unlimited". After 255 failed restarts in a row it stops until the next logon or an explicit start.
RESTART_INTERVAL = "PT1M"
RESTART_COUNT = 255
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


_NELL_CMD_SHIM_MARKER = "companion-emergence-nell-shim"


def _is_nell_cmd_shim(path: Path) -> bool:
    """True for the `nell.cmd` shortcut the desktop app writes (marker in its first lines)."""
    if path.suffix.lower() != ".cmd":
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
    except OSError:
        return False
    return any(_NELL_CMD_SHIM_MARKER in line for line in head)


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
        resolved = Path(found).resolve()
        if _is_nell_cmd_shim(resolved):
            # The desktop app's `nell.cmd` shortcut is on PATH. Registering it would make Task
            # Scheduler open a console window at every logon, so use the bundled runtime's own
            # `Scripts\nell.bat` (the shortcut runs the bundled python, so this interpreter is it).
            bundled = Path(sys.executable).parent / "Scripts" / "nell.bat"
            if bundled.is_file():
                return bundled
            raise WindowsServiceConfigError(
                "found the Companion Emergence nell.cmd shortcut on PATH but not the bundled "
                "nell.bat next to this Python; pass --nell-path C:\\path\\to\\nell.bat"
            )
        return resolved
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


@dataclass(frozen=True)
class WindowsTaskAction:
    """Resolved command + arguments for the Task Scheduler ``<Exec>`` block."""

    command: str
    arguments: str


def _supervisor_args(persona: str) -> str:
    """Build the supervisor CLI arguments string for a Task Scheduler action."""
    return f'supervisor run --persona "{persona}" --client-origin task-scheduler --idle-shutdown 0'


def _task_action_for_nell_path(persona: str, nell_path: Path) -> WindowsTaskAction:
    """Resolve the Task Scheduler ``<Exec>`` command + arguments.

    For the bundled-runtime case (``nell.bat`` sitting in a ``Scripts/``
    subdirectory next to a ``python-runtime/`` tree), launching ``nell.bat``
    directly opens a ``cmd.exe`` console window at logon.  The
    ``<Settings><Hidden>true`` flag only hides the task from the Task
    Scheduler UI — it does NOT suppress the console window.

    To avoid the console window we launch ``pythonw.exe`` (the
    windowless Python interpreter that ships alongside ``python.exe`` in
    every python-build-standalone Windows distribution) directly with a
    ``-c`` inline command that replicates exactly what ``nell.bat`` does::

        pythonw.exe -c "import sys; from brain.cli import main; sys.exit(main())" <args>

    ENV PARITY NOTE: ``nell.bat`` sets no environment variables — it
    only resolves ``python.exe`` by relative path and passes ``%*``.
    The ``pythonw.exe -c`` action is therefore parity-equivalent.
    KINDLED_HOME is already passed via the Task XML ``<Environment>``
    block when ``nellbrain_home`` is provided; PYTHONPATH for a bundled
    runtime is satisfied by site-packages being inside the runtime tree.

    For any other nell entry point (``nell.exe``, system PATH ``nell``,
    absolute paths), the command is used as-is — those are console apps
    but they're not launched at logon via Task Scheduler from a bundled
    runtime, so the window-visibility concern does not apply.
    """
    args = _supervisor_args(persona)
    if nell_path.name.lower() == "nell.bat" and nell_path.parent.name.lower() == "scripts":
        runtime_dir = nell_path.parent.parent
        pythonw = runtime_dir / "pythonw.exe"
        if not pythonw.exists():
            raise WindowsServiceConfigError(
                f"windowless Task Scheduler launch requires pythonw.exe next to bundled runtime: {pythonw}"
            )
        code = "import sys; from brain.cli import main; sys.exit(main())"
        return WindowsTaskAction(
            command=str(pythonw),
            arguments=f'-c "{code}" {args}',
        )
    return WindowsTaskAction(command=str(nell_path), arguments=args)


def task_user_id() -> str | None:
    """The account a per-user task is registered for, as ``DOMAIN\\user`` (#260).

    Task Scheduler reads a ``<LogonTrigger>`` without ``<UserId>`` as "at ANY
    user's logon" — an all-users registration that needs elevation, so
    ``schtasks /Create`` from a normal shell fails with "Access is denied".
    Naming the current account on both the trigger and the principal keeps
    the registration per-user (the launchd ``gui/<uid>/`` and systemd
    ``--user`` analog) and lets it register unelevated.

    Resolved from ``USERDOMAIN`` + ``USERNAME``; a bare ``USERNAME`` is
    accepted (Task Scheduler resolves it against the local machine). ``None``
    when ``USERNAME`` is unset — non-Windows hosts running the unit tests —
    so the XML is rendered without the node.
    """
    username = (os.environ.get("USERNAME") or "").strip()
    if not username:
        return None
    domain = (os.environ.get("USERDOMAIN") or "").strip()
    return f"{domain}\\{username}" if domain else username


def build_task_xml(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_WINDOWS_PATH,
    nellbrain_home: str | Path | None = None,
    user_id: str | None = None,
) -> str:
    """Render a Task Scheduler XML for the persona's supervisor task.

    The XML follows Windows Task Scheduler 1.3 schema:
      * ``<LogonTrigger>`` — start at login (matches RunAtLoad / WantedBy=default.target),
        scoped to the current account via ``<UserId>`` on both the trigger and the
        principal (#260); ``user_id`` overrides the environment-derived account,
        and the node is omitted when neither resolves.
      * ``<RegistrationInfo><Author>`` — for diagnostic tracing.
      * ``<Settings><RestartOnFailure>`` — restart on crash every minute,
        up to 255 times (the scheduler's limits, ``RESTART_INTERVAL`` and
        ``RESTART_COUNT``); the bridge itself spaces repeated start failures
        (matches launchd KeepAlive + systemd Restart=on-failure).
      * ``<Actions><Exec>`` — supervisor entry point; for bundled Windows
        installs uses ``pythonw.exe`` (windowless) instead of ``nell.bat``
        (which would open a cmd.exe console at logon).
      * ``<Settings><Hidden>true`` — hides the task from the Task Scheduler
        UI only; it does NOT suppress a console window. Windowlessness
        comes from launching pythonw.exe (see _task_action_for_nell_path),
        not from <Hidden>.

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
        nell_path_resolved = nell_path_resolved.resolve()

    action = _task_action_for_nell_path(persona, nell_path_resolved)

    # Working directory and stdout/stderr can't be set per-task in
    # Task Scheduler the way launchd / systemd do — we let the runner
    # inherit %USERPROFILE% as the working dir, and the runner's own
    # RotatingFileHandler writes to ``runtime-<persona>.log``. The
    # captured stdout/stderr files exist as a fallback if the
    # supervisor fails before logging is initialized.

    resolved_user = user_id if user_id is not None else task_user_id()
    user_node = f"      <UserId>{_xml_escape(resolved_user)}</UserId>\n" if resolved_user else ""

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
        '<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        "  <RegistrationInfo>\n"
        f"    <Description>companion-emergence supervisor ({_xml_escape(persona)})</Description>\n"
        "    <Author>companion-emergence</Author>\n"
        "  </RegistrationInfo>\n"
        "  <Triggers>\n"
        "    <LogonTrigger>\n"
        "      <Enabled>true</Enabled>\n"
        f"{user_node}"
        "    </LogonTrigger>\n"
        "  </Triggers>\n"
        "  <Principals>\n"
        '    <Principal id="Author">\n'
        f"{user_node}"
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
        f"      <Interval>{RESTART_INTERVAL}</Interval>\n"
        f"      <Count>{RESTART_COUNT}</Count>\n"
        "    </RestartOnFailure>\n"
        "  </Settings>\n"
        '  <Actions Context="Author">\n'
        "    <Exec>\n"
        f"      <Command>{_xml_escape(action.command)}</Command>\n"
        f"      <Arguments>{_xml_escape(action.arguments)}</Arguments>\n"
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
    save_with_backup_text(paths.xml_path, body, keep_backups=False, encoding="utf-16")  # atomic (#484)
    return paths.xml_path


def run_schtasks(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``schtasks <args>``. Synthetic exit-127 when the tool is
    not on PATH (e.g. running on macOS for unit tests)."""
    try:
        return subprocess.run(  # noqa: S603
            ["schtasks", *args],
            capture_output=True,
            text=True,
            # Windows' default subprocess codec is not UTF-8; schtasks writes in the console's
            # OEM code page, so a strict decode could raise on a non-ASCII task name (#363).
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=["schtasks", *args],
            returncode=127,
            stdout="",
            stderr="schtasks not available: Windows host required",
        )
    except subprocess.TimeoutExpired:
        # A hung schtasks is a failed call for every caller (the same code timeout(1) and run_systemctl use).
        return subprocess.CompletedProcess(
            args=["schtasks", *args],
            returncode=124,
            stdout="",
            stderr="schtasks timed out",
        )


def install_service(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_WINDOWS_PATH,
    nellbrain_home: str | Path | None = None,
) -> Path:
    """Write the task XML, then ``schtasks /Create /XML``.

    Idempotent: if the task already exists, ``/F`` overwrites it. #456: never behind a running bridge's back, and
    any stop intent and start back-off record are cleared first
    (``lifecycle.before_install``).
    """
    from brain.service import lifecycle

    lifecycle.before_install(persona)
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


_RESTART_BLOCK = re.compile(
    r"(<RestartOnFailure>\s*<Interval>)([^<]*)(</Interval>\s*<Count>)([^<]*)(</Count>)"
)


def refresh_installed_task(persona: str) -> bool:
    """Raise the restart settings of OUR already-installed task for ``persona`` (D26, the D14 pattern).

    Called by ``start_service`` only: a path that runs outside any task instance (an instance replacing its own
    definition is not something we can test or document). Reads the registered XML (``schtasks /Query /XML``) and,
    for a task whose command is the one this module writes and whose ``RestartOnFailure`` is not ``RESTART_INTERVAL`` /
    ``RESTART_COUNT``, changes only those two values and registers the text again with ``/Create /XML /F`` (the cached
    XML is rewritten too).

    Never from text that may have been damaged: ``run_schtasks`` decodes schtasks' console output as UTF-8 with
    replacement, so a non-ASCII character (a profile path with an accented letter, which comes back as U+FFFD or as
    another letter) or a NUL (UTF-16 output) would be written back wrong and break the task's command. Any non-ASCII
    character or NUL in the queried XML skips the refresh with one WARNING: that task keeps its old settings until its
    next ``nell service install``. One INFO line on success; idempotent; a failure is a WARNING and returns False.
    Returns True when the task was changed."""
    name = task_name(persona)
    queried = run_schtasks(["/Query", "/TN", name, "/XML"])
    if queried.returncode != 0:
        return False
    xml = queried.stdout or ""
    ours = re.search(
        rf"<Arguments>[^<]*supervisor run --persona (?:&quot;|\")?{re.escape(persona)}(?:&quot;|\")?[^<]*</Arguments>", xml
    )
    block = _RESTART_BLOCK.search(xml)
    if ours is None or block is None or (block.group(2), block.group(4)) == (RESTART_INTERVAL, str(RESTART_COUNT)):
        return False
    if not xml.isascii() or "\x00" in xml:
        logger.warning(
            "task scheduler task not refreshed: its registered XML cannot be read back without loss "
            "(non-ASCII text); it keeps its old restart settings until `nell service install` is run again"
        )
        return False
    new_xml = (
        xml[: block.start()]
        + f"{block.group(1)}{RESTART_INTERVAL}{block.group(3)}{RESTART_COUNT}{block.group(5)}"
        + xml[block.end() :]
    )
    xml_path = paths_for_persona(persona).xml_path
    try:
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        save_with_backup_text(xml_path, new_xml, keep_backups=False, encoding="utf-16")  # atomic (#484)
    except OSError:
        logger.warning("task scheduler task not refreshed: could not write %s", xml_path, exc_info=True)
        return False
    created = run_schtasks(["/Create", "/XML", str(xml_path), "/TN", name, "/F"])
    if created.returncode != 0:
        logger.warning("task scheduler task not refreshed: %s", _format_schtasks_error("/Create", created))
        return False
    logger.info(
        "task scheduler task refreshed: restart every %s up to %s times (was every %s, %s times)",
        RESTART_INTERVAL, RESTART_COUNT, block.group(2), block.group(4),
    )
    return True


def start_service(*, persona: str) -> subprocess.CompletedProcess[str]:
    """Run the installed task (#456 D20). Any stop intent and start back-off record are cleared first (D25, D26), and
    the task's restart settings are brought up to date (``refresh_installed_task``)."""
    from brain.service import lifecycle

    lifecycle.note_explicit_start(persona)
    try:
        refresh_installed_task(persona)  # outside any instance; before /Run, so an instance it ends is started again
    except Exception:  # noqa: BLE001 - best effort; the task keeps working with its old restart settings
        logger.warning("task scheduler task refresh failed", exc_info=True)
    return run_schtasks(["/Run", "/TN", task_name(persona)])


def uninstall_service(*, persona: str, keep_xml: bool = False) -> Path:
    """Remove the scheduled task; optionally keep the cached XML."""
    paths = paths_for_persona(persona)
    # #456 (C54): no /End here. The CLI stops a running bridge first through its own request and watcher; /End is only
    # the kill path's step after an ignored hard exit (it would cut in-flight work).
    run_schtasks(["/Delete", "/TN", paths.task_name, "/F"])
    if not keep_xml and paths.xml_path.exists():
        paths.xml_path.unlink()
    return paths.xml_path


def service_status(*, persona: str) -> ServiceStatus:
    """Report the task as Task Scheduler sees it (``schtasks /Query``).

    ``installed`` is true only when the query finds the task registered. The
    cached XML under ``<root>\\service\\`` is NOT proof of installation: a
    failed ``schtasks /Create`` leaves it behind, and trusting it reported a
    denied install as ``installed: yes`` (#260). When the XML exists but the
    task is missing, ``detail`` says so on top of the schtasks output.
    """
    paths = paths_for_persona(persona)
    query = run_schtasks(["/Query", "/TN", paths.task_name, "/V", "/FO", "LIST"])
    installed = query.returncode == 0
    detail = query.stdout.strip() if installed else query.stderr.strip()
    if not installed and paths.xml_path.exists():
        detail = (
            f"task {paths.task_name} is not registered with Task Scheduler although "
            f"{paths.xml_path} exists (a failed install leaves the XML behind; "
            "re-run `nell service install`)" + (f"; schtasks: {detail}" if detail else "")
        )
    loaded = installed and "Status:" in (query.stdout or "")
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
    "WindowsTaskAction",
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
