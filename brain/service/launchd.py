"""macOS launchd service helpers for companion-emergence.

This module is intentionally side-effect-light. The first service CLI slice uses
it to render LaunchAgent plists and run doctor checks without bootstrapping or
unloading jobs. Real launchctl install/start/stop operations can build on these
pure helpers.
"""

from __future__ import annotations

import os
import platform
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from brain.paths import get_home, get_log_dir, get_persona_dir
from brain.setup import validate_persona_name

LABEL_PREFIX = "com.companion-emergence.supervisor"


def _default_launchd_path() -> str:
    """Resolve the launchd PATH at plist-build time.

    launchd starts agents with a stripped environment — the user's shell
    PATH is NOT inherited. The classic ``/opt/homebrew/bin`` /
    ``/usr/local/bin`` chain covers Homebrew's Python and the official
    macOS layout, but ``claude`` (Anthropic's CLI) installs to
    ``~/.local/bin/claude`` by default, which isn't on any system path.
    Resolving the user's home means the agent can find ``claude``
    without manual ``--env-path`` overrides on first install.
    """
    user_local = str(Path("~/.local/bin").expanduser())
    return f"{user_local}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


DEFAULT_LAUNCHD_PATH = _default_launchd_path()


class LaunchdConfigError(ValueError):
    """Raised when service plist configuration is invalid."""


class LaunchdCommandError(RuntimeError):
    """Raised when a launchctl operation fails."""


@dataclass(frozen=True)
class LaunchdPaths:
    """Derived filesystem locations for one persona's LaunchAgent."""

    label: str
    plist_path: Path
    stdout_path: Path
    stderr_path: Path
    persona_dir: Path


@dataclass(frozen=True)
class DoctorCheck:
    """One service preflight/doctor check."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ServiceStatus:
    """launchd status summary for one persona service."""

    label: str
    plist_path: Path
    installed: bool
    loaded: bool
    detail: str


def service_label(persona: str) -> str:
    """Return the stable launchd label for a persona."""
    try:
        validate_persona_name(persona)
    except ValueError as exc:
        raise LaunchdConfigError(str(exc)) from exc
    return f"{LABEL_PREFIX}.{persona}"


def launch_agents_dir() -> Path:
    """Per-user LaunchAgents directory."""
    return Path.home() / "Library" / "LaunchAgents"


def paths_for_persona(persona: str) -> LaunchdPaths:
    """Compute plist/log/persona paths for one persona."""
    label = service_label(persona)
    log_dir = get_log_dir()
    return LaunchdPaths(
        label=label,
        plist_path=launch_agents_dir() / f"{label}.plist",
        stdout_path=log_dir / f"supervisor-{persona}.out.log",
        stderr_path=log_dir / f"supervisor-{persona}.err.log",
        persona_dir=get_persona_dir(persona),
    )


def resolve_nell_path(explicit: str | None = None) -> Path:
    """Resolve the executable path launchd should run.

    launchd does not invoke a shell, so ProgramArguments[0] must be an
    absolute executable path. In an installed runtime this is normally the
    ``nell`` console script. Source-tree/dev callers can pass ``--nell-path``.
    """
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_absolute():
            raise LaunchdConfigError(f"--nell-path must be absolute, got {explicit!r}")
        if not candidate.exists():
            raise LaunchdConfigError(f"--nell-path does not exist: {candidate}")
        if not os.access(candidate, os.X_OK):
            raise LaunchdConfigError(f"--nell-path is not executable: {candidate}")
        return candidate.resolve()

    # When the bundled app invokes ``.../Resources/python-runtime/bin/nell
    # service install`` from Finder, that executable is not necessarily on
    # PATH. Prefer the current absolute argv[0] before falling back to PATH so
    # launchd can still be installed from a bundled runtime even if the caller
    # forgot to pass --nell-path.
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.is_absolute() and argv0.exists() and os.access(argv0, os.X_OK):
        return argv0.resolve()

    found = shutil.which("nell")
    if found:
        return Path(found).resolve()
    raise LaunchdConfigError(
        "could not resolve 'nell' on PATH; pass --nell-path /absolute/path/to/nell"
    )


def unstable_nell_path_reason(nell_path: str | Path) -> str | None:
    """Return why a launchd ProgramArguments path is not stable enough.

    LaunchAgents persist across reboots. Pointing them at a DMG mount or a
    translocated app bundle produces a plist that works only until eject or
    until macOS picks a new translocation path.
    """
    path_text = str(Path(nell_path).expanduser())
    if path_text.startswith("/Volumes/"):
        return (
            "nell executable is under /Volumes; move Companion Emergence to "
            "/Applications before installing the launchd service"
        )
    if "/AppTranslocation/" in path_text:
        return (
            "nell executable is in a macOS AppTranslocation path; move "
            "Companion Emergence to /Applications and relaunch before "
            "installing the launchd service"
        )
    return None


def build_launchd_plist(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_LAUNCHD_PATH,
    nellbrain_home: str | Path | None = None,
    working_directory: str | Path | None = None,
) -> dict:
    """Build a launchd plist dictionary for one persona service."""
    paths = paths_for_persona(persona)
    nell = Path(nell_path)
    if not nell.is_absolute():
        raise LaunchdConfigError(f"nell_path must be absolute, got {nell_path!r}")

    env = {"PATH": env_path}
    if nellbrain_home is not None:
        env["KINDLED_HOME"] = str(Path(nellbrain_home).expanduser().resolve())

    return {
        "Label": paths.label,
        "ProgramArguments": [
            str(nell),
            "supervisor",
            "run",
            "--persona",
            persona,
            "--client-origin",
            "launchd",
            "--idle-shutdown",
            "0",
        ],
        "RunAtLoad": True,
        "KeepAlive": {
            "Crashed": True,
            "SuccessfulExit": False,
        },
        "WorkingDirectory": str(Path(working_directory).expanduser().resolve())
        if working_directory is not None
        else str(Path.home()),
        "EnvironmentVariables": env,
        "StandardOutPath": str(paths.stdout_path),
        "StandardErrorPath": str(paths.stderr_path),
    }


def build_launchd_plist_xml(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_LAUNCHD_PATH,
    nellbrain_home: str | Path | None = None,
    working_directory: str | Path | None = None,
) -> str:
    """Return plist XML text for one persona service."""
    plist = build_launchd_plist(
        persona=persona,
        nell_path=nell_path,
        env_path=env_path,
        nellbrain_home=nellbrain_home,
        working_directory=working_directory,
    )
    return plistlib.dumps(plist, sort_keys=False).decode("utf-8")


def write_launchd_plist(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_LAUNCHD_PATH,
    nellbrain_home: str | Path | None = None,
    working_directory: str | Path | None = None,
) -> Path:
    """Atomically write the persona LaunchAgent plist and return its path."""
    paths = paths_for_persona(persona)
    xml = build_launchd_plist_xml(
        persona=persona,
        nell_path=nell_path,
        env_path=env_path,
        nellbrain_home=nellbrain_home,
        working_directory=working_directory,
    )
    paths.plist_path.parent.mkdir(parents=True, exist_ok=True)
    paths.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.plist_path.with_suffix(paths.plist_path.suffix + ".tmp")
    tmp.write_text(xml, encoding="utf-8")
    tmp.chmod(0o644)
    os.replace(tmp, paths.plist_path)
    paths.plist_path.chmod(0o644)
    return paths.plist_path


def truncate_launchd_logs_if_large(
    persona: str,
    *,
    max_bytes: int = 5 * 1024 * 1024,
) -> list[Path]:
    """Truncate launchd-captured stdout/stderr logs if they exceed a budget.

    Python's normal rotating logging does not cover the file descriptors that
    launchd opens for StandardOutPath/StandardErrorPath before the interpreter
    starts. This bounded truncation keeps crash-loop stderr from growing without
    limit while retaining small recent logs for diagnosis.
    """
    paths = paths_for_persona(persona)
    truncated: list[Path] = []
    for log_path in (paths.stdout_path, paths.stderr_path):
        try:
            if log_path.exists() and log_path.stat().st_size > max_bytes:
                log_path.write_text("", encoding="utf-8")
                truncated.append(log_path)
        except OSError:
            # Log management must never prevent service startup/install.
            continue
    return truncated


def launchctl_domain() -> str:
    """Current user's launchd GUI domain."""
    return f"gui/{os.getuid()}"


def launchctl_target(persona: str) -> str:
    """launchctl target string for one persona service."""
    return f"{launchctl_domain()}/{service_label(persona)}"


def run_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run launchctl with captured output. Tests mock this seam."""
    return subprocess.run(
        ["launchctl", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def install_service(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_LAUNCHD_PATH,
    nellbrain_home: str | Path | None = None,
) -> Path:
    """Write, bootstrap, and kickstart the persona LaunchAgent."""
    reason = unstable_nell_path_reason(nell_path)
    if reason is not None:
        raise LaunchdConfigError(reason)
    truncate_launchd_logs_if_large(persona)
    plist_path = write_launchd_plist(
        persona=persona,
        nell_path=nell_path,
        env_path=env_path,
        nellbrain_home=nellbrain_home,
    )
    # Idempotent reinstall: ignore bootout failure when the job is not loaded.
    run_launchctl(["bootout", launchctl_target(persona)])

    result = run_launchctl(["bootstrap", launchctl_domain(), str(plist_path)])
    if result.returncode != 0:
        raise LaunchdCommandError(_format_launchctl_error("bootstrap", result))

    result = run_launchctl(["kickstart", "-k", launchctl_target(persona)])
    if result.returncode != 0:
        raise LaunchdCommandError(_format_launchctl_error("kickstart", result))

    return plist_path


def uninstall_service(*, persona: str, keep_plist: bool = False) -> Path:
    """Boot out the persona LaunchAgent and remove its plist unless requested."""
    paths = paths_for_persona(persona)
    # Uninstall should be idempotent: a not-loaded service is already stopped.
    run_launchctl(["bootout", launchctl_target(persona)])
    if not keep_plist:
        try:
            paths.plist_path.unlink()
        except FileNotFoundError:
            pass
    return paths.plist_path


def service_status(*, persona: str) -> ServiceStatus:
    """Return launchd installed/loaded state for one persona."""
    paths = paths_for_persona(persona)
    result = run_launchctl(["print", launchctl_target(persona)])
    loaded = result.returncode == 0
    detail = result.stdout.strip() if loaded else result.stderr.strip()
    return ServiceStatus(
        label=paths.label,
        plist_path=paths.plist_path,
        installed=paths.plist_path.exists(),
        loaded=loaded,
        detail=detail,
    )


def doctor_checks(
    *,
    persona: str,
    nell_path: str | None = None,
    env_path: str = DEFAULT_LAUNCHD_PATH,
) -> list[DoctorCheck]:
    """Return non-mutating launchd service preflight checks."""
    checks: list[DoctorCheck] = []

    is_macos = platform.system() == "Darwin"
    checks.append(
        DoctorCheck(
            "platform",
            is_macos,
            "macOS launchd available" if is_macos else "launchd service backend requires macOS",
        )
    )

    try:
        paths = paths_for_persona(persona)
    except LaunchdConfigError as exc:
        checks.append(DoctorCheck("persona_name", False, str(exc)))
        return checks

    checks.append(DoctorCheck("persona_name", True, f"label={paths.label}"))
    checks.append(
        DoctorCheck(
            "persona_dir",
            paths.persona_dir.exists(),
            str(paths.persona_dir),
        )
    )

    try:
        resolved_nell = resolve_nell_path(nell_path)
        checks.append(DoctorCheck("nell_path", True, str(resolved_nell)))
        stable_reason = unstable_nell_path_reason(resolved_nell)
        checks.append(
            DoctorCheck(
                "nell_path_stable",
                stable_reason is None,
                stable_reason or "launchd executable path is stable",
            )
        )
    except LaunchdConfigError as exc:
        checks.append(DoctorCheck("nell_path", False, str(exc)))

    launch_dir = launch_agents_dir()
    checks.append(
        DoctorCheck(
            "launch_agents_dir",
            launch_dir.exists() or launch_dir.parent.exists(),
            str(launch_dir),
        )
    )

    log_dir = get_log_dir()
    checks.append(
        DoctorCheck(
            "log_dir",
            log_dir.exists() or log_dir.parent.exists(),
            str(log_dir),
        )
    )

    claude_path = _which_in_path("claude", env_path)
    checks.append(
        DoctorCheck(
            "claude_cli",
            claude_path is not None,
            claude_path or "claude not found in launchd PATH",
        )
    )

    home = get_home()
    checks.append(DoctorCheck("home", home.exists() or home.parent.exists(), str(home)))
    return checks


def _which_in_path(executable: str, path_value: str) -> str | None:
    """shutil.which against an explicit PATH string."""
    return shutil.which(executable, path=path_value)


def _format_launchctl_error(action: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    return f"launchctl {action} failed (exit {result.returncode}): {detail}"
