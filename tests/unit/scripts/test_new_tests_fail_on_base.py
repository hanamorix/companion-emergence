"""The check that every NEW test of a pull request fails on the code the PR started from (#540, J7).

The pure helpers are tested directly; one end-to-end test builds a small throw-away git repository with a base
commit and a head commit that adds three tests, and runs the real script on it: an always-passing test is reported, a
test of code that is new at the head bites, and an always-passing test carrying ``no_base_failure(reason=...)`` is
exempt and counted; a test of a module that only exists at the head cannot be collected at the base and bites too; a
``git mv`` of a test file creates no new test.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.no_base_failure import marker_problems

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("new_tests_fail_on_base", REPO / "scripts" / "new_tests_fail_on_base.py")
ntf = importlib.util.module_from_spec(_spec)
sys.modules["new_tests_fail_on_base"] = ntf
_spec.loader.exec_module(ntf)


def test_a_name_status_listing_with_renames_and_copies_is_parsed() -> None:
    raw = "A\0tests/a/test_x.py\0M\0tests/test_y.py\0R087\0tests/old/test_z.py\0tests/new/test_z.py\0D\0tests/test_gone.py\0"
    assert ntf.parse_name_status(raw) == [
        ("A", "tests/a/test_x.py", "tests/a/test_x.py"),
        ("M", "tests/test_y.py", "tests/test_y.py"),
        ("R", "tests/old/test_z.py", "tests/new/test_z.py"),
        ("D", "tests/test_gone.py", "tests/test_gone.py"),
    ]


def test_only_added_modified_and_renamed_test_files_are_changed_test_files() -> None:
    entries = [
        ("A", "tests/a/test_x.py", "tests/a/test_x.py"),
        ("M", "tests/conftest.py", "tests/conftest.py"),
        ("M", "brain/x.py", "brain/x.py"),
        ("R", "tests/old/test_z.py", "tests/new/test_z.py"),
        ("D", "tests/test_gone.py", "tests/test_gone.py"),
        ("M", "tests/unit/foo_test.py", "tests/unit/foo_test.py"),
        ("M", "tests/snapshots/dev_constants.json", "tests/snapshots/dev_constants.json"),
    ]
    files, renamed = ntf.changed_test_files(entries)
    assert files == ["tests/a/test_x.py", "tests/new/test_z.py", "tests/unit/foo_test.py"]
    assert renamed == {"tests/new/test_z.py": "tests/old/test_z.py"}


def test_a_pure_move_creates_no_new_test_but_a_new_parameter_does() -> None:
    renamed = {"tests/new/test_z.py": "tests/old/test_z.py"}
    base_ids = ["tests/old/test_z.py::test_a", "tests/old/test_z.py::test_b[1]", "tests/test_y.py::test_c"]
    head_ids = {
        "tests/new/test_z.py::test_a",
        "tests/new/test_z.py::test_b[1]",
        "tests/new/test_z.py::test_b[2]",
        "tests/test_y.py::test_c",
        "tests/test_y.py::test_d",
    }
    new = head_ids - ntf.remap_base_ids(base_ids, renamed)
    assert new == {"tests/new/test_z.py::test_b[2]", "tests/test_y.py::test_d"}


def test_collected_ids_are_read_from_lines_that_start_with_a_tests_path() -> None:
    out = "tests/a/test_x.py::test_1\ntests/a/test_x.py::TestC::test_2[a b-1]\n\nERROR tests/b/test_bad.py\n== warnings ==\ntests/c.py: 1 warning\n"
    assert ntf.parse_collected(out) == ["tests/a/test_x.py::test_1", "tests/a/test_x.py::TestC::test_2[a b-1]"]


def test_judge_separates_bites_from_tests_that_pass_skip_or_never_ran() -> None:
    outcomes = {
        "t::fails": [("setup", "passed"), ("call", "failed"), ("teardown", "passed")],
        "t::setup_error": [("setup", "failed")],
        "t::passes": [("setup", "passed"), ("call", "passed"), ("teardown", "passed")],
        "t::skips": [("setup", "skipped")],
    }
    collectable = {"t::fails", "t::setup_error", "t::passes", "t::skips", "t::lost"}
    verdict = ntf.judge(["t::fails", "t::setup_error", "t::passes", "t::skips", "t::lost", "t::new_code"], collectable, outcomes)
    assert set(verdict.bites) == {"t::fails", "t::setup_error", "t::new_code"}
    assert verdict.bites["t::new_code"].startswith("cannot be collected")
    assert verdict.violations["t::passes"] == "passes against the base code"
    assert verdict.violations["t::skips"].startswith("skipped")
    assert verdict.violations["t::lost"].startswith("collected but did not run")


def test_the_canary_is_the_smallest_unchanged_test_file_that_is_not_tiny() -> None:
    listing = "\n".join(
        [
            "100644 blob aaa     120\ttests/unit/test_tiny.py",
            "100644 blob bbb     900\ttests/unit/test_small.py",
            "100644 blob ccc     500\ttests/unit/test_changed.py",
            "100644 blob ddd    5000\ttests/unit/test_big.py",
            "100644 blob eee     450\ttests/unit/helper.py",
        ]
    )
    assert ntf.smallest_unchanged_test(listing, {"tests/unit/test_changed.py"}) == "tests/unit/test_small.py"
    assert ntf.smallest_unchanged_test("", set()) is None


def _item(nodeid: str, *marks: dict, inherited: int = 0) -> SimpleNamespace:
    """A stand-in collected item: ``marks`` on the function itself, ``inherited`` more from its class or module."""
    own = [SimpleNamespace(name="no_base_failure", kwargs=m) for m in marks]
    outer = [SimpleNamespace(name="no_base_failure", kwargs={"reason": "x"}) for _ in range(inherited)]
    return SimpleNamespace(nodeid=nodeid, own_markers=own, iter_markers=lambda name: [*own, *outer] if name == "no_base_failure" else [])


def test_a_no_base_failure_mark_without_a_reason_or_on_a_module_or_class_is_a_problem() -> None:
    items = [
        _item("t::ok", {"reason": "guards what already exists"}),
        _item("t::none"),
        _item("t::empty", {"reason": "  "}),
        _item("t::missing", {}),
        _item("t::not_a_string", {"reason": 3}),
        _item("t::on_the_module", inherited=1),
        _item("t::both", {"reason": "own"}, inherited=1),
    ]
    problems = marker_problems(items)
    assert [p.split(": @")[0] for p in problems] == ["t::empty", "t::missing", "t::not_a_string", "t::on_the_module", "t::both"]
    assert all("module or class" in p for p in problems[3:])


_MINI_PYPROJECT = textwrap.dedent(
    """
    [tool.pytest.ini_options]
    pythonpath = ["."]
    markers = ["no_base_failure(reason): exempt"]
    """
)
_CANARY = '"""A small unchanged test that passes on the base. It is long enough that the check picks it as the canary: the smallest unchanged unit test file of a few hundred bytes."""\n\nimport brain\n\n\ndef test_old_function_works():\n    assert brain.old() == 1\n\n\ndef test_old_function_is_stable():\n    assert brain.old() + brain.old() == 2\n'
_NEW_TESTS = textwrap.dedent(
    """
    import brain
    import pytest


    def test_always_passes():
        assert True


    def test_new_function_works():
        assert brain.new() == 2


    @pytest.mark.no_base_failure(reason="guards what already exists")
    def test_exempt_always_passes():
        assert True
    """
)


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.email=t@example.test", "-c", "user.name=t", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _run_script(repo: Path, base: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess, Path]:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "new_tests_fail_on_base.py"), "--base", base, "--repo", str(repo), "--workers", "1"],
        cwd=tmp_path,
        env={**os.environ, "TMPDIR": str(scratch), "TEMP": str(scratch), "TMP": str(scratch), "PYTEST_ADDOPTS": "-qq"},  # must not be inherited
        capture_output=True,
        text=True,
        check=False,
    )
    return done, scratch


def _base_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "brain").mkdir(parents=True)
    (repo / "tests" / "unit").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(_MINI_PYPROJECT, encoding="utf-8")
    (repo / "brain" / "__init__.py").write_text("def old():\n    return 1\n", encoding="utf-8")
    (repo / "tests" / "unit" / "test_canary.py").write_text(_CANARY, encoding="utf-8")
    (repo / "tests" / "unit" / "test_moved_before.py").write_text(_CANARY.replace("canary", "moved"), encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


@pytest.mark.skipif(sys.platform != "linux", reason="the check runs on the ubuntu PR job; it uses git worktrees and POSIX paths")
def test_the_script_reports_an_always_passing_new_test_and_exempts_a_marked_one(tmp_path: Path) -> None:
    repo, base = _base_repo(tmp_path)
    (repo / "brain" / "__init__.py").write_text("def old():\n    return 1\n\n\ndef new():\n    return 2\n", encoding="utf-8")
    (repo / "brain" / "newmod.py").write_text("VALUE = 3\n", encoding="utf-8")
    (repo / "tests" / "unit" / "test_new.py").write_text(_NEW_TESTS, encoding="utf-8")
    (repo / "tests" / "unit" / "test_new_module.py").write_text(
        "import brain.newmod\n\n\ndef test_the_new_module_has_its_value():\n    assert brain.newmod.VALUE == 3\n", encoding="utf-8"
    )
    _git(repo, "mv", "tests/unit/test_moved_before.py", "tests/unit/test_moved_after.py")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "head")

    done, scratch = _run_script(repo, base, tmp_path)
    out = done.stdout
    assert done.returncode == 1, out + done.stderr
    assert "tests/unit/test_new.py::test_always_passes: passes against the base code" in out
    assert "test_new_function_works" not in out.split("do NOT fail")[-1]
    assert "exempt: tests/unit/test_new.py::test_exempt_always_passes" in out
    # a pure `git mv` creates no new test; the three tests of test_new.py and the one of the new module are new
    assert "new 4  exempt (no_base_failure) 1  to check 3" in out
    assert "2 bite, 1 do not, 1 exempt" in out
    assert list(scratch.iterdir()) == [], "the scratch directory with both worktrees is removed"
    assert len(_git(repo, "worktree", "list").splitlines()) == 1, "no worktree is left registered"


@pytest.mark.skipif(sys.platform != "linux", reason="the check runs on the ubuntu PR job; it uses git worktrees and POSIX paths")
def test_the_script_refuses_to_pass_when_collection_itself_fails(tmp_path: Path) -> None:
    """A conftest that stops collection must not read as "no tests to check" (the silent-green case)."""
    repo, base = _base_repo(tmp_path)
    (repo / "tests" / "unit" / "test_new.py").write_text(_NEW_TESTS.replace("brain.new()", "brain.old()"), encoding="utf-8")
    (repo / "conftest.py").write_text(
        "import pytest\n\n\ndef pytest_collection_modifyitems(items):\n    raise pytest.UsageError('a refused mark')\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "head")

    done, scratch = _run_script(repo, base, tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "CHECK COULD NOT JUDGE THIS PR" in done.stdout
    assert "nothing to check" not in done.stdout
    assert list(scratch.iterdir()) == []


def test_the_collected_count_is_read_from_pytests_last_line() -> None:
    assert ntf.collected_count("tests/a.py::t\n\n1 test collected in 0.01s\n") == 1
    assert ntf.collected_count("tests/a.py::t\n\n191/4812 tests collected (4621 deselected) in 1.2s\n") == 191
    assert ntf.collected_count("no tests collected in 0.01s\n") is None
