"""The guard against tests that start a real `claude` in CI (#540, J11): the rule, and the planted tests that show it bites.

The rule is tested as a pure function with the event shapes `subprocess.Popen` raises on POSIX (a list argv) and on
Windows (`executable` None and the argv already one command-line string). The behaviour is tested by running small
planted test files in a SEPARATE pytest process (`runpytest_subprocess`) with the guard forced on or off through the
environment, so the audit hook, the flag and the records are never shared with the test that checks them.
"""

from __future__ import annotations

import os
import stat
import sys
import threading
from pathlib import Path

import pytest

import tests.claude_spawn_guard as guard
from tests.claude_spawn_guard import enabled, violation

REPO = Path(__file__).resolve().parents[2]


def _fake(directory: Path, name: str = "claude") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_a_bare_claude_that_resolves_nowhere_is_an_attempt(tmp_path: Path) -> None:
    message = violation(None, ["claude", "--help"], {"PATH": str(tmp_path)}, [str(tmp_path)])
    assert message is not None
    assert "no such program" in message


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_other_programs_and_explicit_paths_are_never_blocked(tmp_path: Path) -> None:
    roots = [str(tmp_path)]
    assert violation(None, ["git", "status"], {"PATH": ""}, roots) is None
    assert violation(None, ["/usr/bin/claude", "-p"], {"PATH": ""}, roots) is None
    assert violation(None, [str(tmp_path / "fake" / "claude")], {"PATH": ""}, roots) is None
    assert violation(None, ["./claude"], {"PATH": ""}, roots) is None
    assert violation(None, ["claude-code-helper"], {"PATH": ""}, roots) is None


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
@pytest.mark.skipif(sys.platform == "win32", reason="the fake is a /bin/sh script")
def test_a_bare_claude_that_resolves_to_a_fake_under_the_temp_root_is_allowed_and_elsewhere_is_not(tmp_path: Path) -> None:
    inside = tmp_path / "bin"
    _fake(inside)
    root = os.path.normcase(os.path.realpath(str(tmp_path)))
    assert violation("claude", ["claude"], {"PATH": str(inside)}, [root]) is None
    other_root = os.path.normcase(os.path.realpath(str(tmp_path / "elsewhere")))
    message = violation("claude", ["claude"], {"PATH": str(inside)}, [other_root])
    assert message is not None
    assert "resolves to" in message


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need elevation on Windows")
def test_a_temp_root_reached_through_a_symlink_matches_the_real_path(tmp_path: Path) -> None:
    """macOS: gettempdir() is /var/folders/... and pytest's tmp_path is the resolved /private/var/folders/... form."""
    real = tmp_path / "real"
    _fake(real / "bin")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    root_via_link = os.path.normcase(os.path.realpath(str(link)))
    assert violation(None, ["claude"], {"PATH": str(link / "bin")}, [root_via_link]) is None


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_the_windows_event_shape_a_command_line_string_is_parsed(tmp_path: Path) -> None:
    roots = [str(tmp_path)]
    assert violation(None, "claude --version", {"PATH": ""}, roots) is not None
    assert violation(None, '"claude.exe" -p hello', {"PATH": ""}, roots) is not None
    assert violation(None, '"C:\\Program Files\\claude\\claude.exe" -p hello', {"PATH": ""}, roots) is None
    assert violation(None, "git status", {"PATH": ""}, roots) is None


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_path_like_bytes_and_a_differently_cased_path_key_are_understood(tmp_path: Path) -> None:
    roots = [str(tmp_path)]
    assert violation(None, [Path("claude")], {"Path": str(tmp_path)}, roots) is not None
    assert violation(None, [b"claude"], {"PATH": str(tmp_path)}, roots) is not None
    assert violation(None, ["CLAUDE.EXE"], {"PATH": str(tmp_path)}, roots) is not None
    assert violation(None, [], {"PATH": ""}, roots) is None


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_the_guard_is_on_in_ci_or_when_asked_and_off_otherwise() -> None:
    assert enabled({"GITHUB_ACTIONS": "true"})
    assert enabled({"KINDLED_GUARD_CLAUDE": "1"})
    assert not enabled({})
    assert not enabled({"GITHUB_ACTIONS": ""})


_INNER_CONFTEST = "from tests.claude_spawn_guard import no_real_claude_in_ci  # noqa: F401\n"
_INNER_INI = "[pytest]\nmarkers =\n    requires_claude_cli: needs the claude CLI\n"
_SPAWN_PLANTS = '''
import subprocess
import sys
import pytest


def _empty_path(tmp_path):
    (tmp_path / "empty").mkdir()
    return {"PATH": str(tmp_path / "empty")}


def test_unmarked_spawn_by_name(tmp_path):
    with pytest.raises(FileNotFoundError):
        subprocess.run(["claude", "--version"], env=_empty_path(tmp_path), check=False)


@pytest.mark.requires_claude_cli
def test_marked_spawn_by_name(tmp_path):
    with pytest.raises(FileNotFoundError):
        subprocess.run(["claude", "--version"], env=_empty_path(tmp_path), check=False)


def test_swallowed_spawn_by_name(tmp_path):
    try:
        subprocess.run(["claude", "--version"], env=_empty_path(tmp_path), check=False)
    except Exception:
        pass


def test_patched_popen(monkeypatch):
    class _Fake:
        def __init__(self, *a, **k):
            self.returncode = 0

    monkeypatch.setattr(subprocess, "Popen", _Fake)
    assert subprocess.Popen(["claude", "--version"]).returncode == 0


def test_other_program():
    subprocess.run([sys.executable, "-c", "pass"], check=True)
'''
_FAKE_PLANTS = '''
import os
import stat
import subprocess


def _fake(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "claude"
    fake.write_text("#!/bin/sh\\nexit 0\\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return bin_dir, fake


def test_fake_by_explicit_path(tmp_path):
    _, fake = _fake(tmp_path)
    assert subprocess.run([str(fake)], check=False).returncode == 0


def test_fake_first_on_the_path_by_name(tmp_path):
    bin_dir, _ = _fake(tmp_path)
    assert subprocess.run(["claude"], env={"PATH": str(bin_dir)}, check=False).returncode == 0
'''


def _run_inner(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, *, guard: bool, source: str):
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("KINDLED_GUARD_CLAUDE", raising=False)
    if guard:
        monkeypatch.setenv("KINDLED_GUARD_CLAUDE", "1")
    pytester.makeini(_INNER_INI)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(source)
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:randomly", "-p", "no:tdd-guard", "-rfE")


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_with_the_guard_on_an_unmarked_spawn_of_claude_fails_and_names_the_ways_out(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_inner(pytester, monkeypatch, guard=True, source=_SPAWN_PLANTS)
    result.assert_outcomes(passed=4, failed=1, errors=2)
    result.stdout.fnmatch_lines(
        [
            "FAILED *::test_unmarked_spawn_by_name*",
            "ERROR *::test_unmarked_spawn_by_name*",
            "ERROR *::test_swallowed_spawn_by_name*",  # swallowed by the test: the teardown still fails it
        ]
    )
    result.stdout.fnmatch_lines(["*pytest.mark.requires_claude_cli*"])


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_with_the_guard_off_the_same_tests_all_pass(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_inner(pytester, monkeypatch, guard=False, source=_SPAWN_PLANTS)
    result.assert_outcomes(passed=5)


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
@pytest.mark.skipif(sys.platform == "win32", reason="the fakes are /bin/sh scripts")
def test_a_fake_claude_by_explicit_path_or_first_on_the_path_is_allowed(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_inner(pytester, monkeypatch, guard=True, source=_FAKE_PLANTS)
    result.assert_outcomes(passed=2)


_THREAD_PLANTS = """
import subprocess
import threading

_go = threading.Event()
_done = threading.Event()
_errors = []


def _spawn_when_told():
    _go.wait(10)
    try:
        subprocess.run(["claude"], env={"PATH": ""}, check=False)
    except Exception as exc:  # noqa: BLE001
        _errors.append(type(exc).__name__)
    _done.set()


# started at import time, so it already runs when the first test begins
threading.Thread(target=_spawn_when_told, daemon=True).start()


def test_a_thread_that_predates_the_test_is_not_blamed():
    _go.set()
    assert _done.wait(10)
    assert _errors == ["FileNotFoundError"]


def test_a_thread_started_by_the_test_is_blamed():
    errors = []

    def spawn():
        try:
            subprocess.run(["claude"], env={"PATH": ""}, check=False)
        except Exception as exc:  # noqa: BLE001
            errors.append(type(exc).__name__)

    thread = threading.Thread(target=spawn)
    thread.start()
    thread.join(10)
    assert errors == ["RuntimeError"]
"""


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_a_spawn_from_a_thread_that_predates_the_test_is_not_blamed_on_it(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_inner(pytester, monkeypatch, guard=True, source=_THREAD_PLANTS)
    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines(["ERROR *::test_a_thread_started_by_the_test_is_blamed*"])


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
@pytest.mark.skipif(sys.platform == "win32", reason="the fake is a /bin/sh script")
def test_a_fake_under_pytests_base_temp_is_allowed_even_when_the_system_temp_is_elsewhere(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--basetemp` can sit outside tempfile.gettempdir(): the base temp the fixture registers is the second allowed root."""
    system_temp = tmp_path / "system-temp"
    system_temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(system_temp))
    monkeypatch.setenv("TEMP", str(system_temp))
    monkeypatch.setenv("TMP", str(system_temp))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("KINDLED_GUARD_CLAUDE", "1")
    pytester.makeini(_INNER_INI)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(_FAKE_PLANTS)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:randomly", "-p", "no:tdd-guard", f"--basetemp={tmp_path / 'base'}")
    result.assert_outcomes(passed=2)


def _fake_record(name: str):
    return guard._Record(nodeid=name, thread=threading.get_ident(), threads_before=frozenset())


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_the_innermost_active_test_is_blamed_and_an_analysis_bug_never_breaks_a_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer, inner = _fake_record("outer"), _fake_record("inner")
    guard._ACTIVE.extend([outer, inner])
    try:
        monkeypatch.setattr(guard, "violation", lambda *a, **k: "it spawns claude")
        with pytest.raises(RuntimeError, match="inner: it spawns claude"):
            guard._audit("subprocess.Popen", ("claude", ["claude"], None, None))
        assert inner.violations and not outer.violations

        def broken(*_a, **_k):
            raise ValueError("analysis bug")

        monkeypatch.setattr(guard, "violation", broken)
        guard._audit("subprocess.Popen", ("claude", ["claude"], None, None))  # must not raise
        assert len(inner.violations) == 1
    finally:
        guard._ACTIVE.remove(outer)
        guard._ACTIVE.remove(inner)
