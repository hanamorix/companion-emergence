"""Drive ``nell service install`` against a real systemd --user instance.

Skipped on macOS / Windows. Also skips on Linux hosts where
systemd --user isn't reachable (e.g. bare containers without a
DBus session — most GitHub ubuntu runners do NOT have a live
systemd --user instance, so this test self-skips there and
documents what a real Linux machine should see).

When the test runs it verifies:

  1. ``nell service install --persona ci_test`` exits 0.
  2. The unit file lands at
     ``~/.config/systemd/user/companion-emergence-ci_test.service``.
  3. The unit file's ``ExecStart`` line contains
     ``--client-origin systemd`` (proves the systemd backend was
     invoked, not the launchd one).
  4. systemd loads the unit (``LoadState=loaded``). Deliberately not
     ``is-active``: the unit's bridge has no persona to boot under the
     test's bare ``KINDLED_HOME``, so asserting liveness races its exit
     (#160).

Teardown via ``nell service uninstall --persona ci_test`` runs in a
``try/finally`` so a mid-test assertion failure still cleans up.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# The entire module is Linux-only.
pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="systemd integration test is Linux-only",
)


def _systemd_user_available() -> bool:
    """Return True if a live systemd --user instance is reachable.

    A real user session running systemd --user is required — this is
    absent in most containers (including GitHub's ubuntu runners by
    default, which do not run a user-session DBus). The check uses
    ``systemctl --user is-system-running``; acceptable return codes
    are 0 (running) and 1 (degraded — still functional).

    FileNotFoundError means systemctl isn't installed at all;
    TimeoutExpired means DBus is frozen — both map to "not available".
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # 0 = running, 1 = degraded; both are functional user instances.
        return result.returncode in (0, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def test_install_creates_unit_file_and_systemd_loads_it(tmp_path: Path) -> None:
    """Round-trip: install → assert unit file + systemd loaded it → uninstall."""
    if not _systemd_user_available():
        pytest.skip("systemd --user not available on this host (expected in CI containers)")

    # KINDLED_HOME → data dir. The systemd backend writes unit files to
    # the real ~/.config/systemd/user/, not under KINDLED_HOME, so the
    # persona_dir itself doesn't need to exist for install to succeed.
    env = os.environ.copy()
    env["KINDLED_HOME"] = str(tmp_path)

    unit_path = Path.home() / ".config" / "systemd" / "user" / "companion-emergence-ci_test.service"

    install = subprocess.run(
        [
            "uv",
            "run",
            "nell",
            "service",
            "install",
            "--persona",
            "ci_test",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        assert install.returncode == 0, (
            f"nell service install failed\n"
            f"  stdout: {install.stdout!r}\n"
            f"  stderr: {install.stderr!r}"
        )

        # 1. Unit file must exist at the canonical path.
        assert unit_path.exists(), (
            f"unit file missing at {unit_path}\n  install stdout: {install.stdout!r}"
        )

        unit_content = unit_path.read_text(encoding="utf-8")

        # 2. Persona name appears in the unit file (guards against
        #    install writing the wrong persona's unit).
        assert "ci_test" in unit_content, (
            f"persona name 'ci_test' not found in unit file:\n{unit_content}"
        )

        # 3. ExecStart must bake --client-origin systemd. This is the
        #    canonical proof that brain.service.systemd was invoked
        #    rather than the launchd backend.
        assert "--client-origin" in unit_content and "systemd" in unit_content, (
            f"unit file ExecStart should contain '--client-origin systemd':\n{unit_content}"
        )

        # 4. systemd must have LOADED the unit — i.e. parsed it and accepted it
        #    into the user manager. This is what `service install` actually owns.
        #
        #    Deliberately NOT `is-active`: KINDLED_HOME here is a bare tmp_path with
        #    no persona dir, so the bridge the unit starts exits almost immediately.
        #    Polling is-active right after install races that exit — a fast crash
        #    reports "failed", a slow one is still "activating" — which is the
        #    intermittent failure in #160. Bridge boot-success is a different
        #    subject with different preconditions; it is not this test's claim.
        load_state = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                "-p",
                "LoadState",
                "--value",
                "companion-emergence-ci_test",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert load_state.stdout.strip() == "loaded", (
            "systemd did not load the installed unit; "
            f"LoadState={load_state.stdout.strip()!r} stderr={load_state.stderr!r}"
        )

    finally:
        # Always uninstall so the CI runner is left clean.
        subprocess.run(
            [
                "uv",
                "run",
                "nell",
                "service",
                "uninstall",
                "--persona",
                "ci_test",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
