"""GM-2 (#456 merge check): the `real_platform_dirs` fixture really runs its guard. An inner test run (pytester) that
requests the fixture and writes into an exposed directory must fail at teardown; one that only compares paths must not.

Linux only: there #488's XDG variables keep the exposed directories inside the inner run's tmp root, so the inner
test's write cannot reach a real user directory (on macOS / Windows it would)."""

from __future__ import annotations

import sys

import pytest

pytest_plugins = ["pytester"]

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the write must stay in a tmp root")

_CONFTEST = """
from tests.conftest import _isolate_user_data_home, real_platform_dirs  # noqa: F401
"""

_TESTS = """
from pathlib import Path

from brain import paths


def test_writes_into_an_exposed_directory(real_platform_dirs):
    d = Path(paths._dirs.user_data_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "written-by-a-test.txt").write_text("x", encoding="utf-8")


def test_only_compares_paths(real_platform_dirs):
    assert Path(paths._dirs.user_data_dir).name
"""


def test_the_fixture_fails_a_test_that_writes_into_an_exposed_directory(pytester):
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(test_inner=_TESTS)
    result = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly")
    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines(["*ERROR at teardown of test_writes_into_an_exposed_directory*"])
    result.stdout.fnmatch_lines(["*created or changed a real user directory*"])
