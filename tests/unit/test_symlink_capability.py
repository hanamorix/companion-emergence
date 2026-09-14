"""#262: symlink-dependent tests must skip, not fail, where the host denies symlink creation
(non-elevated Windows without Developer Mode raises ``OSError: [WinError 1314]``)."""

from __future__ import annotations

import os

import pytest

from tests.conftest import _symlinks_available


def test_probe_reports_true_where_symlinks_work(tmp_path) -> None:
    # Control: on the hosts that run this suite's CI, symlinks are creatable.
    target = tmp_path / "t"
    target.mkdir()
    try:
        (tmp_path / "l").symlink_to(target)
    except OSError:
        pytest.skip("host denies symlinks; the False branch is covered below")
    assert _symlinks_available.__wrapped__() is True


def test_probe_reports_false_when_symlink_denied(monkeypatch) -> None:
    def _deny(*a, **k):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(os, "symlink", _deny)
    assert _symlinks_available.__wrapped__() is False


def test_requires_symlinks_fixture_skips_when_denied(pytester) -> None:
    pytester.makeconftest(
        "import functools, pytest\n"
        "from tests.conftest import requires_symlinks  # noqa: F401\n"
        "import tests.conftest as c\n"
        "c._symlinks_available = functools.cache(lambda: False)\n"
    )
    pytester.makepyfile("def test_x(requires_symlinks):\n    assert False\n")
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(skipped=1)
