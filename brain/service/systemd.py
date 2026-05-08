"""Linux ``systemd --user`` backend stub for the per-persona supervisor.

Status: **stub**. The interface mirrors :mod:`brain.service.launchd` so
the ``nell service`` CLI dispatcher can route to this module on Linux
without surprise, but every entry point raises
:class:`brain.service.UnsupportedPlatformError`. Real implementation
needs a Linux box to validate against (`systemctl --user`, the user
service unit file shape, journalctl integration), and that hasn't
happened yet.

Sketch of what the real implementation will look like, recorded here
so the next implementer doesn't redo the design phase:

* User unit file at ``~/.config/systemd/user/companion-emergence-<persona>.service``.
* ``Service.ExecStart`` = the same ``nell supervisor run --persona
  <name> --client-origin systemd --idle-shutdown 0`` that launchd
  uses.
* ``Service.Restart=on-failure`` matches launchd's KeepAlive / Crashed.
* ``Install.WantedBy=default.target`` for "start at user login".
* ``Service.StandardOutput`` / ``StandardError`` go to journald by
  default; override to a file path if we want parity with the
  ``runtime-<persona>.log`` rotation.
* ``systemctl --user daemon-reload`` after writing.
* ``systemctl --user enable --now companion-emergence-<persona>.service``
  to install + start, ``disable --now`` to uninstall.
* Doctor checks: systemd-running-as-user (``systemctl --user
  is-system-running``), ``XDG_RUNTIME_DIR`` exported, claude on the
  user PATH, log dir writable.
"""

from __future__ import annotations

from brain.service import UnsupportedPlatformError

_NOT_YET = (
    "Linux ``systemd --user`` backend is not yet implemented. "
    "Use ``nell supervisor start --persona <name>`` (legacy spawn path) "
    "until the systemd backend lands. Tracking: "
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

    Linux uses systemd unit files, not plists; this name just keeps
    the dispatcher's call site uniform. Real impl will be
    ``build_systemd_unit_text`` and the dispatcher will pick the
    correct attribute by feature, not by exact name.
    """
    raise UnsupportedPlatformError(_NOT_YET)
