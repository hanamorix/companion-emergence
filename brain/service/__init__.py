"""OS service integration for companion-emergence — platform dispatch layer.

The supervisor runs as a user-level OS service so the brain stays alive
when the desktop app is closed. macOS uses launchd via
``brain.service.launchd``; Linux ``systemd --user`` and Windows Service /
Task Scheduler analogs are planned but not yet implemented (the stubs
return ``UnsupportedPlatformError`` until real validation has happened
on those targets).

Public surface — everything the ``nell service`` CLI needs:

* :class:`UnsupportedPlatformError` — raised by the stubs on
  unimplemented platforms; callers should catch it and surface a
  user-readable "this OS isn't supported yet" message.
* :func:`current_backend` — picks the right backend module for the
  running OS, or raises ``UnsupportedPlatformError``.

Existing macOS-only call sites import from ``brain.service.launchd``
directly. New callers should prefer ``current_backend()`` so they pick
up Linux/Windows support automatically when those backends ship.
"""

from __future__ import annotations

import sys
from types import ModuleType


class UnsupportedPlatformError(RuntimeError):
    """Raised when the active OS does not yet have a service backend.

    Carries the platform name and a short hint pointing the user at
    the relevant tracking issue / plan doc rather than a stack trace.
    """


def current_backend() -> ModuleType:
    """Return the OS-appropriate service backend module.

    The returned module is one of ``brain.service.launchd`` (macOS),
    ``brain.service.systemd`` (Linux user services — stub), or
    ``brain.service.windows_service`` (Windows — stub). The two stub
    backends raise ``UnsupportedPlatformError`` on every call so the
    abstraction is in place but no half-working install gets attempted.
    """
    if sys.platform == "darwin":
        from brain.service import launchd

        return launchd
    if sys.platform == "linux":
        from brain.service import systemd

        return systemd
    if sys.platform.startswith("win"):
        from brain.service import windows_service

        return windows_service
    raise UnsupportedPlatformError(
        f"No service backend for sys.platform={sys.platform!r}. "
        "See docs/superpowers/plans/2026-05-08-launchd-supervisor-agent.md "
        "for the supported targets."
    )
