"""Windows backend stub for the per-persona supervisor.

Status: **stub**. The interface mirrors :mod:`brain.service.launchd`
so the dispatcher can route to this module on Windows, but every
entry point raises :class:`brain.service.UnsupportedPlatformError`.
A real Windows implementation needs:

1. **Decision** — a real Windows Service (managed by ``sc.exe`` / SCM)
   needs a service host that bridges ``ServiceMain``; the canonical
   path is ``pywin32`` + ``win32serviceutil``. Lighter alternatives:
   * Task Scheduler with ``StartAtLogon`` triggers — no service-host
     boilerplate, easier to install/uninstall via ``schtasks``, but
     less "service-y" feel.
   * NSSM (Non-Sucking Service Manager) wraps any executable as a
     proper service — bundle as a third-party dep.

   Default lean: Task Scheduler. Aligns with the "user-level service"
   semantics of macOS LaunchAgent and Linux ``systemd --user``.
2. **Service path equivalent**:
   ``%LOCALAPPDATA%\\companion-emergence\\service\\<persona>.xml``
   (Task Scheduler XML) for the Task Scheduler path, or a generated
   ``.bat`` wrapper for the NSSM path.
3. **Lifecycle commands**:
   * Install: ``schtasks /Create /XML <xml> /TN <name>``
   * Uninstall: ``schtasks /Delete /TN <name> /F``
   * Status: ``schtasks /Query /TN <name> /V /FO LIST``
4. **Logging**: redirect to ``%LOCALAPPDATA%\\companion-emergence\\Logs\\
   supervisor-<persona>.log``, rotated by the runner's
   ``RotatingFileHandler`` (already in place).
5. **Doctor checks**: PowerShell version, claude on PATH (``where claude``),
   the appropriate Logs/Service folders writable.

Until that lands the .app falls back to the legacy
"spawned by the desktop app" lifecycle on Windows.
"""

from __future__ import annotations

from brain.service import UnsupportedPlatformError

_NOT_YET = (
    "Windows service backend is not yet implemented. "
    "Use ``nell supervisor start --persona <name>`` (legacy spawn path) "
    "until the Windows backend lands. Tracking: "
    "docs/superpowers/plans/2026-05-08-launchd-supervisor-agent.md"
)


def install_service(*args: object, **kwargs: object) -> None:
    raise UnsupportedPlatformError(_NOT_YET)


def uninstall_service(*args: object, **kwargs: object) -> None:
    raise UnsupportedPlatformError(_NOT_YET)


def service_status(*args: object, **kwargs: object) -> None:
    raise UnsupportedPlatformError(_NOT_YET)


def doctor_checks(*args: object, **kwargs: object) -> None:
    raise UnsupportedPlatformError(_NOT_YET)


def build_launchd_plist_xml(*args: object, **kwargs: object) -> None:
    """Symmetry with :func:`brain.service.launchd.build_launchd_plist_xml`.

    Real impl will return Task Scheduler XML (or NSSM .bat content).
    """
    raise UnsupportedPlatformError(_NOT_YET)
