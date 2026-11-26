"""Keep the systemd install integration test from leaving a real unit behind (#454).

`test_systemd_install.py` installs a REAL user unit, `companion-emergence-ci_test.service` (`Restart=on-failure`, `RestartSec=2`), and
a killed run (a timeout, an OOM kill, a frozen machine) skips its `finally`, leaving an enabled unit that systemd restarts every two
seconds. Three layers, all best effort and none of them able to raise into the test:

1. :func:`remove_unit` at the START removes a leftover of this unit from an earlier killed run;
2. :func:`schedule_failsafe` registers a transient systemd TIMER before the install. It lives in the user manager, which outlives the
   test process, so it removes the unit after :data:`FAILSAFE_DELAY_S` even when the process is killed;
3. :func:`unit_guard` wraps the whole test: sweep, failsafe, body, then sweep again and cancel the failsafe, in a ``finally``.

Every function takes the command runner and the unit directory as arguments, so `tests/unit/brain/service/test_ci_systemd_cleanup_454.py`
tests all of it against a fake `systemctl` and a throwaway directory, without touching the real user manager or `~/.config/systemd/user`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

UNIT = "companion-emergence-ci_test"
FAILSAFE = f"{UNIT}-failsafe"
# The test (an install call limited to 30 s plus a few seconds of polling) finishes well inside this; a killed run is cleaned by the
# timer within it. Long enough not to fire under a slow full-suite run, short enough to bound a restarting unit.
FAILSAFE_DELAY_S = 300
_STEP_TIMEOUT_S = 15

Runner = Callable[..., subprocess.CompletedProcess]

ALLOW_REAL_INSTALL_ENV = "KINDLED_ALLOW_REAL_SERVICE_INSTALL"


def real_install_blocked_reason(environ: Mapping[str, str] | None = None) -> str | None:
    """None when this process may install the real test unit, else the reason to skip.

    Allowed on GitHub Actions (``GITHUB_ACTIONS == "true"``) or when the developer opts in with
    ``KINDLED_ALLOW_REAL_SERVICE_INSTALL=1``. A bare ``pytest`` on a developer box that has a reachable user systemd must not install
    a real unit under ``~/.config/systemd/user`` by accident.
    """
    environ = os.environ if environ is None else environ
    if environ.get("GITHUB_ACTIONS") == "true" or environ.get(ALLOW_REAL_INSTALL_ENV) == "1":
        return None
    return (
        "installs a real systemd --user unit under ~/.config/systemd/user: runs only on GitHub Actions "
        f"(GITHUB_ACTIONS=true) or with {ALLOW_REAL_INSTALL_ENV}=1"
    )


def unit_directory() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def _try(run: Runner, argv: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return run(argv, capture_output=True, text=True, timeout=_STEP_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):  # no systemctl, a frozen bus, a refused call: this is cleanup, never a failure
        return None


def remove_unit(*, run: Runner = subprocess.run, unit_dir: Path | None = None) -> None:
    """Stop and disable this test's unit, delete its file, and make the manager forget it. Idempotent, never raises."""
    unit_dir = unit_dir if unit_dir is not None else unit_directory()
    service = f"{UNIT}.service"
    _try(run, ["systemctl", "--user", "disable", "--now", service])
    _try(run, ["systemctl", "--user", "stop", service])
    try:
        (unit_dir / service).unlink(missing_ok=True)
    except OSError:
        pass
    _try(run, ["systemctl", "--user", "daemon-reload"])
    _try(run, ["systemctl", "--user", "reset-failed", service])


def failsafe_command(*, unit_dir: Path | None = None) -> list[str]:
    """The `systemd-run` call that schedules the delayed removal of this test's unit (and only that unit)."""
    unit_dir = unit_dir if unit_dir is not None else unit_directory()
    service = f"{UNIT}.service"
    script = (
        f"systemctl --user disable --now {service}; "
        f"rm -f {shlex.quote(str(unit_dir / service))}; "
        "systemctl --user daemon-reload"
    )
    return [
        "systemd-run",
        "--user",
        f"--on-active={FAILSAFE_DELAY_S}s",
        f"--unit={FAILSAFE}",
        "--collect",
        "/bin/sh",
        "-c",
        script,
    ]


def schedule_failsafe(*, run: Runner = subprocess.run, unit_dir: Path | None = None) -> bool:
    """Register the fail-safe timer. True if the manager accepted it."""
    result = _try(run, failsafe_command(unit_dir=unit_dir))
    return result is not None and result.returncode == 0


def cancel_failsafe(*, run: Runner = subprocess.run) -> None:
    _try(run, ["systemctl", "--user", "stop", f"{FAILSAFE}.timer"])


class FailsafeNotArmedError(RuntimeError):
    """The fail-safe timer could not be registered, so the test must not install a real unit."""


@contextmanager
def unit_guard(*, run: Runner = subprocess.run, unit_dir: Path | None = None) -> Iterator[None]:
    """Around the test: sweep a leftover, arm the fail-safe, run the body, then sweep again and disarm (also if the body raises).

    Raises :class:`FailsafeNotArmedError` (before the body, so nothing is installed) when the timer cannot be registered: without it only the
    in-process cleanup stands between a killed run and a leftover restarting unit.
    """
    remove_unit(run=run, unit_dir=unit_dir)
    if not schedule_failsafe(run=run, unit_dir=unit_dir):
        raise FailsafeNotArmedError("could not register the fail-safe timer (systemd-run unavailable or refused); not installing a real unit")
    try:
        yield
    finally:
        remove_unit(run=run, unit_dir=unit_dir)
        cancel_failsafe(run=run)
