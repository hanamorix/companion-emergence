"""Linux ``systemd --user`` backend for the per-persona supervisor.

Mirrors :mod:`brain.service.launchd` so the ``nell service`` CLI
dispatcher works uniformly across platforms. Same lifecycle:

  * ``install_service`` writes a user unit file and starts it.
  * ``uninstall_service`` stops + removes the unit.
  * ``service_status`` reports loaded / running / pid.
  * ``doctor_checks`` runs non-mutating preflight (systemd present,
    user-instance reachable, ``claude`` on PATH, log dir writable).

Unit files live at::

    ~/.config/systemd/user/companion-emergence-<persona>.service

``ExecStart`` invokes ``nell supervisor run --persona <name>
--client-origin systemd --idle-shutdown 0`` exactly as the launchd
plist does — same foreground-supervisor entry point.

Status: **scaffolded for v0.0.1**. The unit-file generation and CLI
shape are exercised by unit tests, but live ``systemctl --user``
runs against a real Linux box are pending — flagged in the wizard
as "compiles cleanly but not validated on real hardware yet". When
those run, this module is the place errors will surface.
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


def _user_home() -> Path:
    """Real OS home directory, ignoring ``NELLBRAIN_HOME``.

    ``brain.paths.get_home`` returns the data root (which can be
    ``NELLBRAIN_HOME``-overridden for sandboxed installs); systemd
    user-unit files have to live under the actual user-shell home,
    so we resolve that explicitly here.
    """
    return Path.home()

UNIT_PREFIX = "companion-emergence"
DEFAULT_SYSTEMD_PATH = (
    f"{Path('~/.local/bin').expanduser()}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
)


class SystemdConfigError(ValueError):
    """Raised when service unit configuration is invalid."""


class SystemdCommandError(RuntimeError):
    """Raised when a ``systemctl`` operation fails."""


@dataclass(frozen=True)
class SystemdPaths:
    """Filesystem locations for one persona's systemd user service."""

    unit_name: str  # e.g. "companion-emergence-nell.service"
    unit_path: Path
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
    unit_name: str
    unit_path: Path
    installed: bool
    loaded: bool
    detail: str


def service_unit_name(persona: str) -> str:
    """Return the canonical unit-file name for a persona."""
    validate_persona_name(persona)
    return f"{UNIT_PREFIX}-{persona}.service"


def user_unit_dir() -> Path:
    """Return ``~/.config/systemd/user``."""
    return _user_home() / ".config" / "systemd" / "user"


def paths_for_persona(persona: str) -> SystemdPaths:
    """Filesystem layout for the persona's user service."""
    validate_persona_name(persona)
    unit_name = service_unit_name(persona)
    log_dir = get_log_dir()
    return SystemdPaths(
        unit_name=unit_name,
        unit_path=user_unit_dir() / unit_name,
        stdout_path=log_dir / f"supervisor-{persona}.out.log",
        stderr_path=log_dir / f"supervisor-{persona}.err.log",
        persona_dir=get_persona_dir(persona),
    )


def resolve_nell_path(explicit: str | None = None) -> Path:
    """Mirror of launchd.resolve_nell_path."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise SystemdConfigError(
                f"--nell-path must be absolute (got {path!s})"
            )
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemdConfigError(f"nell at {path} is not executable")
        return path
    found = shutil.which("nell")
    if found:
        return Path(found).resolve()
    raise SystemdConfigError(
        "could not resolve 'nell' on PATH; pass --nell-path /absolute/path/to/nell"
    )


def build_systemd_unit_text(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_SYSTEMD_PATH,
    nellbrain_home: str | Path | None = None,
) -> str:
    """Render the user-service unit file body for a persona.

    Pure function — no filesystem side effects. Tested by feeding
    different (persona, nell_path) inputs and asserting the
    resulting INI-style body. Mirrors ``build_launchd_plist`` on the
    macOS side.
    """
    validate_persona_name(persona)
    nell_path = Path(nell_path).expanduser().resolve()
    if not nell_path.is_file():
        raise SystemdConfigError(f"nell binary not found at {nell_path}")
    paths = paths_for_persona(persona)

    env_lines = [f'Environment="PATH={env_path}"']
    if nellbrain_home is not None:
        resolved_home = str(Path(nellbrain_home).expanduser().resolve())
        env_lines.append(f'Environment="NELLBRAIN_HOME={resolved_home}"')

    env_block = "\n".join(env_lines)

    return (
        "[Unit]\n"
        f"Description=companion-emergence supervisor ({persona})\n"
        "After=default.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={nell_path} supervisor run --persona {persona} "
        "--client-origin systemd --idle-shutdown 0\n"
        "Restart=on-failure\n"
        "RestartSec=2\n"
        f"{env_block}\n"
        f"StandardOutput=append:{paths.stdout_path}\n"
        f"StandardError=append:{paths.stderr_path}\n"
        f"WorkingDirectory={Path(_user_home())}\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def write_systemd_unit(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_SYSTEMD_PATH,
    nellbrain_home: str | Path | None = None,
) -> Path:
    """Render the unit file and write it to disk. Returns the path."""
    body = build_systemd_unit_text(
        persona=persona,
        nell_path=nell_path,
        env_path=env_path,
        nellbrain_home=nellbrain_home,
    )
    paths = paths_for_persona(persona)
    paths.unit_path.parent.mkdir(parents=True, exist_ok=True)
    paths.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    paths.unit_path.write_text(body, encoding="utf-8")
    return paths.unit_path


def run_systemctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``systemctl --user <args>`` with a short timeout.

    Returns a synthetic ``CompletedProcess`` with exit code 127 and a
    helpful stderr when systemctl isn't on PATH (i.e. running on a
    non-Linux host or a barebones container) so callers can treat
    "missing tool" the same as "command failed" rather than catching
    FileNotFoundError everywhere.
    """
    try:
        return subprocess.run(  # noqa: S603
            ["systemctl", "--user", *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=["systemctl", "--user", *args],
            returncode=127,
            stdout="",
            stderr="systemctl not available — Linux user-instance required",
        )


def install_service(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_SYSTEMD_PATH,
    nellbrain_home: str | Path | None = None,
) -> Path:
    """Write the unit file, daemon-reload, then enable + start it."""
    unit_path = write_systemd_unit(
        persona=persona,
        nell_path=nell_path,
        env_path=env_path,
        nellbrain_home=nellbrain_home,
    )

    reload_result = run_systemctl(["daemon-reload"])
    if reload_result.returncode != 0:
        raise SystemdCommandError(
            _format_systemctl_error("daemon-reload", reload_result)
        )

    enable_result = run_systemctl(["enable", "--now", service_unit_name(persona)])
    if enable_result.returncode != 0:
        raise SystemdCommandError(
            _format_systemctl_error("enable --now", enable_result)
        )

    return unit_path


def uninstall_service(*, persona: str, keep_unit: bool = False) -> Path:
    """Disable + stop the unit, remove its file unless requested."""
    paths = paths_for_persona(persona)
    # ``disable --now`` covers stop + disable in one call. Ignore
    # failure here so a half-broken state can still be cleaned up.
    run_systemctl(["disable", "--now", paths.unit_name])

    if not keep_unit and paths.unit_path.exists():
        paths.unit_path.unlink()
        # daemon-reload to make systemd notice the file is gone.
        run_systemctl(["daemon-reload"])
    return paths.unit_path


def service_status(*, persona: str) -> ServiceStatus:
    """Combine unit-file presence with ``systemctl is-active`` output."""
    paths = paths_for_persona(persona)
    installed = paths.unit_path.exists()

    show = run_systemctl(
        ["show", paths.unit_name, "--property=ActiveState,LoadState,MainPID"]
    )
    detail = show.stdout.strip() if show.returncode == 0 else show.stderr.strip()
    loaded = (
        show.returncode == 0
        and "LoadState=loaded" in show.stdout
        and "ActiveState=active" in show.stdout
    )
    return ServiceStatus(
        unit_name=paths.unit_name,
        unit_path=paths.unit_path,
        installed=installed,
        loaded=loaded,
        detail=detail,
    )


def doctor_checks(
    *,
    persona: str,
    nell_path: str | None = None,
    env_path: str = DEFAULT_SYSTEMD_PATH,
) -> list[DoctorCheck]:
    """Non-mutating preflight before installing."""
    checks: list[DoctorCheck] = []

    # Platform
    import sys as _sys

    plat_ok = _sys.platform == "linux"
    checks.append(
        DoctorCheck(
            "platform",
            plat_ok,
            "Linux systemd --user available" if plat_ok else f"unsupported platform {_sys.platform}",
        )
    )

    # Persona name validity. Skip the persona-dir + nell-path checks
    # below if the name is invalid since those would raise the same
    # ValueError from validate_persona_name.
    persona_name_ok = True
    try:
        validate_persona_name(persona)
        checks.append(DoctorCheck("persona_name", True, f"unit={service_unit_name(persona)}"))
    except Exception as exc:  # noqa: BLE001
        persona_name_ok = False
        checks.append(DoctorCheck("persona_name", False, str(exc)))

    if persona_name_ok:
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

    # nell binary resolves
    try:
        resolved_nell = resolve_nell_path(nell_path)
        checks.append(DoctorCheck("nell_path", True, str(resolved_nell)))
    except SystemdConfigError as exc:
        checks.append(DoctorCheck("nell_path", False, str(exc)))

    # systemd user instance reachable
    user_check = run_systemctl(["is-system-running"])
    user_ok = user_check.returncode in (0, 1, 2, 3)  # running / degraded / starting still ok
    checks.append(
        DoctorCheck(
            "systemd_user",
            user_ok,
            user_check.stdout.strip() or user_check.stderr.strip() or "no output",
        )
    )

    # User unit dir writable
    udir = user_unit_dir()
    try:
        udir.mkdir(parents=True, exist_ok=True)
        udir_ok = os.access(udir, os.W_OK)
    except OSError as exc:
        udir_ok = False
        udir_detail = str(exc)
    else:
        udir_detail = str(udir)
    checks.append(DoctorCheck("user_unit_dir", udir_ok, udir_detail))

    # Log dir writable
    log_dir = get_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_ok = os.access(log_dir, os.W_OK)
    except OSError as exc:
        log_ok = False
        log_detail = str(exc)
    else:
        log_detail = str(log_dir)
    checks.append(DoctorCheck("log_dir", log_ok, log_detail))

    # claude on PATH
    claude_path = _which_in_path("claude", env_path)
    checks.append(
        DoctorCheck(
            "claude_cli",
            claude_path is not None,
            claude_path or "claude not found in systemd PATH",
        )
    )

    # NELLBRAIN_HOME (if set)
    home_env = os.environ.get("NELLBRAIN_HOME")
    if home_env:
        checks.append(DoctorCheck("home", True, home_env))
    else:
        checks.append(DoctorCheck("home", True, str(Path(_user_home()))))

    return checks


# ---------------------------------------------------------------------------
# Compatibility shim: launchd's API exports build_launchd_plist_xml. The
# CLI dispatcher introspects that name across backends; alias it here so
# the dispatcher needn't branch on platform when generating "service
# print-plist" output.
# ---------------------------------------------------------------------------


def build_launchd_plist_xml(
    *,
    persona: str,
    nell_path: str | Path,
    env_path: str = DEFAULT_SYSTEMD_PATH,
    nellbrain_home: str | Path | None = None,
) -> str:
    """Alias for :func:`build_systemd_unit_text` so the dispatcher's
    print-plist subcommand renders the right format on Linux."""
    return build_systemd_unit_text(
        persona=persona,
        nell_path=nell_path,
        env_path=env_path,
        nellbrain_home=nellbrain_home,
    )


def _which_in_path(executable: str, path_value: str) -> str | None:
    """shutil.which against an explicit PATH string."""
    return shutil.which(executable, path=path_value)


def _format_systemctl_error(
    action: str, result: subprocess.CompletedProcess[str]
) -> str:
    parts = [f"systemctl --user {action} (exit {result.returncode})"]
    if result.stdout.strip():
        parts.append(f"stdout: {result.stdout.strip()}")
    if result.stderr.strip():
        parts.append(f"stderr: {result.stderr.strip()}")
    return "; ".join(parts)


# Re-export the unsupported error so callers that imported it from this
# module before the real implementation landed still resolve.
__all__ = [
    "DEFAULT_SYSTEMD_PATH",
    "DoctorCheck",
    "ServiceStatus",
    "SystemdCommandError",
    "SystemdConfigError",
    "SystemdPaths",
    "UNIT_PREFIX",
    "UnsupportedPlatformError",
    "build_launchd_plist_xml",
    "build_systemd_unit_text",
    "doctor_checks",
    "install_service",
    "paths_for_persona",
    "resolve_nell_path",
    "run_systemctl",
    "service_status",
    "service_unit_name",
    "uninstall_service",
    "user_unit_dir",
    "write_systemd_unit",
]
