"""#454: the systemd install integration test must not be able to leave a real, enabled, restarting unit behind.

`tests/integration/test_systemd_install.py` installs a REAL user unit (`companion-emergence-ci_test.service`, `Restart=on-failure`,
`RestartSec=2`) and relied on a `finally` to remove it, which a killed run skips. The fix is in layers, and all of it is tested here
against a fake `systemctl` / `systemd-run` runner and a throwaway unit directory, so NOTHING in this file touches the real
`~/.config/systemd/user` or the real user manager:

* a sweep at the START of the test removes a leftover of its own unit (stop, disable, delete the file, reload);
* a fail-safe transient timer, registered with the user manager BEFORE the install, removes the unit after a delay even if the test
  process is killed (the manager outlives the process);
* the end-of-test cleanup does the same sweep and cancels the timer, and runs in a context manager so an exception cannot skip it.

The integration test itself is not run here (it is marked `integration` and would install a real unit).
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from tests.integration import systemd_cleanup as sc

_INTEGRATION_TEST = Path(__file__).resolve().parents[4] / "tests" / "integration" / "test_systemd_install.py"


class FakeRun:
    """Records every command; `fail` maps a command word to an exception to raise or a return code."""

    def __init__(self, fail=None, returncode=0):
        self.calls: list[list[str]] = []
        self.fail = fail or {}
        self.returncode = returncode

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        for word, outcome in self.fail.items():
            if word in argv:
                if isinstance(outcome, BaseException):
                    raise outcome
                return subprocess.CompletedProcess(argv, outcome, "", "")
        return subprocess.CompletedProcess(argv, self.returncode, "", "")


def _words(run: FakeRun) -> list[str]:
    return [" ".join(c) for c in run.calls]


def test_remove_unit_stops_disables_deletes_the_file_and_reloads(tmp_path):
    unit = tmp_path / f"{sc.UNIT}.service"
    unit.write_text("[Service]\nRestart=on-failure\n")
    run = FakeRun()

    sc.remove_unit(run=run, unit_dir=tmp_path)

    joined = _words(run)
    assert any("disable --now" in c and f"{sc.UNIT}.service" in c for c in joined)
    assert any(c.endswith(f"stop {sc.UNIT}.service") for c in joined)
    assert any("daemon-reload" in c for c in joined)
    assert any("reset-failed" in c for c in joined)
    assert not unit.exists()
    # the file is gone BEFORE the reload, so the manager forgets the unit
    assert joined.index(next(c for c in joined if "daemon-reload" in c)) > joined.index(next(c for c in joined if "disable --now" in c))


def test_remove_unit_is_harmless_when_nothing_is_there(tmp_path):
    run = FakeRun(returncode=5)  # systemctl: "unit not loaded"

    sc.remove_unit(run=run, unit_dir=tmp_path)  # must not raise

    assert run.calls  # it still asked, it does not pre-check


def test_remove_unit_never_raises_when_systemctl_is_missing_or_hangs(tmp_path):
    for exc in (FileNotFoundError("systemctl"), subprocess.TimeoutExpired("systemctl", 15), OSError("bus down")):
        sc.remove_unit(run=FakeRun(fail={"systemctl": exc}), unit_dir=tmp_path)


def test_remove_unit_touches_only_its_own_unit(tmp_path):
    other = tmp_path / "companion-emergence-nell.service"
    other.write_text("keep me")
    run = FakeRun()

    sc.remove_unit(run=run, unit_dir=tmp_path)

    assert other.read_text() == "keep me"
    for call in run.calls:
        for word in call:
            assert "nell" not in word and "companion-emergence-" not in word.replace(sc.UNIT, "")


def test_the_failsafe_is_a_transient_timer_that_removes_only_the_test_unit(tmp_path):
    cmd = sc.failsafe_command(unit_dir=tmp_path)

    assert cmd[:2] == ["systemd-run", "--user"]
    assert any(a.startswith("--on-active=") for a in cmd)
    assert f"--unit={sc.FAILSAFE}" in cmd and "--collect" in cmd
    script = cmd[-1]
    assert f"disable --now {sc.UNIT}.service" in script
    assert str(tmp_path / f"{sc.UNIT}.service") in script
    assert "daemon-reload" in script
    assert "nell" not in script.replace(sc.UNIT, "")


def test_the_failsafe_outlasts_the_test_but_is_bounded():
    # It must not fire while the test is still running (install + polls take well under a minute), and it must be bounded.
    assert sc.FAILSAFE_DELAY_S >= 120 and sc.FAILSAFE_DELAY_S <= 1800


def test_schedule_and_cancel_the_failsafe(tmp_path):
    run = FakeRun()

    assert sc.schedule_failsafe(run=run, unit_dir=tmp_path) is True
    assert run.calls[0][0] == "systemd-run"
    sc.cancel_failsafe(run=run)
    assert run.calls[1] == ["systemctl", "--user", "stop", f"{sc.FAILSAFE}.timer"]


def test_a_failsafe_that_cannot_be_scheduled_is_reported_not_raised(tmp_path):
    assert sc.schedule_failsafe(run=FakeRun(returncode=1), unit_dir=tmp_path) is False
    assert sc.schedule_failsafe(run=FakeRun(fail={"systemd-run": FileNotFoundError("systemd-run")}), unit_dir=tmp_path) is False


def test_the_guard_sweeps_first_schedules_the_failsafe_then_cleans_up_and_cancels(tmp_path):
    run = FakeRun()
    leftover = tmp_path / f"{sc.UNIT}.service"
    leftover.write_text("a unit left by a killed run")

    with sc.unit_guard(run=run, unit_dir=tmp_path):
        first_sweep_done = not leftover.exists()
        scheduled_before_body = any(c[0] == "systemd-run" for c in run.calls)

    assert first_sweep_done and scheduled_before_body  # swept at the START, failsafe registered BEFORE the body
    assert run.calls[-1] == ["systemctl", "--user", "stop", f"{sc.FAILSAFE}.timer"]  # cancelled last
    assert sum(c[0] == "systemctl" and "disable" in c and "--now" in c for c in run.calls) == 2  # start sweep + end cleanup


def test_the_guard_cleans_up_even_when_the_body_raises(tmp_path):
    run = FakeRun()

    with pytest.raises(RuntimeError, match="test failed"):
        with sc.unit_guard(run=run, unit_dir=tmp_path):
            (tmp_path / f"{sc.UNIT}.service").write_text("installed by the test")
            raise RuntimeError("test failed")

    assert not (tmp_path / f"{sc.UNIT}.service").exists()
    assert run.calls[-1][-1] == f"{sc.FAILSAFE}.timer"


def test_the_integration_test_runs_inside_the_guard():
    # The real test is not executed here; its source must wire the guard (an autouse fixture around every test in the module).
    tree = ast.parse(_INTEGRATION_TEST.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "unit_guard" in names
    fixtures = [
        f for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef)
        and any("fixture" in ast.dump(d) and "autouse" in ast.dump(d) for d in f.decorator_list)
    ]
    assert fixtures, "an autouse fixture must hold the guard"


def test_the_guard_refuses_to_run_the_body_when_the_failsafe_cannot_be_armed(tmp_path):
    """Code review 3a, NB6: without the fail-safe timer the only thing between a killed run and a leftover restarting unit is the
    in-process `finally`. The guard must not let the test install a real unit in that case."""
    body_ran = False
    for run in (FakeRun(returncode=1), FakeRun(fail={"systemd-run": FileNotFoundError("systemd-run")})):
        with pytest.raises(sc.FailsafeNotArmedError, match="fail-safe"):
            with sc.unit_guard(run=run, unit_dir=tmp_path):
                body_ran = True
    assert not body_ran


def test_the_integration_fixture_skips_when_the_failsafe_cannot_be_armed():
    tree = ast.parse(_INTEGRATION_TEST.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "FailsafeNotArmedError" in names
    assert "skip" in names


@pytest.mark.parametrize(
    ("environ", "allowed"),
    [
        ({"GITHUB_ACTIONS": "true"}, True),
        ({sc.ALLOW_REAL_INSTALL_ENV: "1"}, True),
        ({"GITHUB_ACTIONS": "true", sc.ALLOW_REAL_INSTALL_ENV: "0"}, True),
        ({}, False),
        ({"GITHUB_ACTIONS": "false"}, False),
        ({"GITHUB_ACTIONS": "1"}, False),
        ({sc.ALLOW_REAL_INSTALL_ENV: "0"}, False),
        ({sc.ALLOW_REAL_INSTALL_ENV: "true"}, False),
        ({sc.ALLOW_REAL_INSTALL_ENV: ""}, False),
    ],
)
def test_a_real_unit_install_needs_github_actions_or_an_explicit_opt_in(environ, allowed):
    """A bare pytest on a developer box with a reachable user systemd must not install the real test unit."""
    reason = sc.real_install_blocked_reason(environ)
    assert (reason is None) is allowed
    if not allowed:
        assert "GITHUB_ACTIONS" in reason and sc.ALLOW_REAL_INSTALL_ENV in reason


def test_the_integration_fixture_checks_the_opt_in_before_anything_touches_systemd():
    """The skip must come before the sweep, the fail-safe and the install: the first statements of the autouse fixture call the
    opt-in check and skip, ahead of `unit_guard` and the `systemctl` availability probe."""
    tree = ast.parse(_INTEGRATION_TEST.read_text(encoding="utf-8"))
    fixture = next(
        f for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef) and f.name == "_systemd_unit_guard"
    )
    body = [ast.dump(stmt) for stmt in fixture.body if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))]
    gate = next(i for i, text in enumerate(body) if "real_install_blocked_reason" in text)
    skip = next(i for i, text in enumerate(body) if "attr='skip'" in text)
    # the skip is conditional on what the opt-in check returned
    assert "id='blocked'" in body[skip] and "If(" in body[skip][:4]
    probe = next(i for i, text in enumerate(body) if "_systemd_user_available" in text)
    guard = next(i for i, text in enumerate(body) if "unit_guard" in text)
    assert gate <= skip < probe < guard
