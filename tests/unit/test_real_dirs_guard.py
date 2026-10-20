"""Self-test of the `real_platform_dirs` guard (#456 merge check): the fixture exposes the platform's real user
directories, so it fails the test that requested it if any of them is created or changed during that test."""

from __future__ import annotations

import os
import sys

import pytest

from tests.real_dirs_guard import assert_untouched, fingerprint


def test_an_untouched_tree_passes(tmp_path):
    d = tmp_path / "data"
    (d / "personas" / "x").mkdir(parents=True)
    (d / "personas" / "x" / "memories.db").write_bytes(b"1")
    before = fingerprint([d, tmp_path / "missing"])
    assert_untouched(before, fingerprint([d, tmp_path / "missing"]))


def test_creating_a_missing_directory_fails(tmp_path):
    missing = tmp_path / "log"
    before = fingerprint([missing])
    missing.mkdir()
    with pytest.raises(AssertionError, match="real user directory"):
        assert_untouched(before, fingerprint([missing]))


@pytest.mark.parametrize("change", ["new_file", "modified_file", "removed_file"])
def test_a_change_anywhere_inside_fails(tmp_path, change):
    d = tmp_path / "data"
    nested = d / "personas" / "x"
    nested.mkdir(parents=True)
    f = nested / "persona_config.json"
    f.write_text("{}", encoding="utf-8")
    old = os.stat(f).st_mtime_ns
    before = fingerprint([d])
    if change == "new_file":
        (nested / "tunables.json").write_text("{}", encoding="utf-8")
    elif change == "modified_file":
        f.write_text('{"model": "sonnet"}', encoding="utf-8")
        os.utime(f, ns=(old + 10**9, old + 10**9))
    else:
        f.unlink()
    with pytest.raises(AssertionError, match="real user directory"):
        assert_untouched(before, fingerprint([d]))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux: #488's XDG variables point into the tmp root")
def test_on_linux_the_exposed_directories_stay_in_the_tests_tmp_root(real_platform_dirs, tmp_path_factory):
    from brain import paths

    root = tmp_path_factory.getbasetemp()
    for prop in ("user_data_dir", "user_state_dir"):
        assert str(getattr(paths._dirs, prop)).startswith(str(root)), prop
