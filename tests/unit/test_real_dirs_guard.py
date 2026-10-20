"""Self-test of the `real_platform_dirs` guard (#456 merge check): the fixture exposes the platform's real user
directories, so it fails the test that requested it if one of them is created or gains or loses a top-level entry
during that test. An unrelated live process (a developer's running bridge) that only writes to its existing files
inside those directories does not trip it (GM-1)."""

from __future__ import annotations

import os
import sys

import pytest

from tests.real_dirs_guard import assert_untouched, fingerprint


def _tree(tmp_path):
    d = tmp_path / "data"
    nested = d / "personas" / "x"
    nested.mkdir(parents=True)
    (nested / "memories.db").write_bytes(b"1")
    (d / "runtime-x.log").write_text("start\n", encoding="utf-8")
    return d, nested


def test_an_untouched_tree_passes(tmp_path):
    d, _ = _tree(tmp_path)
    before = fingerprint([d, tmp_path / "missing"])
    assert_untouched(before, fingerprint([d, tmp_path / "missing"]))


def test_creating_a_missing_directory_fails(tmp_path):
    missing = tmp_path / "log"
    before = fingerprint([missing])
    missing.mkdir()
    with pytest.raises(AssertionError, match="real user directory"):
        assert_untouched(before, fingerprint([missing]))


@pytest.mark.parametrize("change", ["new_top_level_file", "new_top_level_dir", "removed_top_level_entry"])
def test_a_new_or_removed_top_level_entry_fails(tmp_path, change):
    d, _ = _tree(tmp_path)
    before = fingerprint([d])
    if change == "new_top_level_file":
        (d / "tunables.json").write_text("{}", encoding="utf-8")
    elif change == "new_top_level_dir":
        (d / "service").mkdir()
    else:
        (d / "runtime-x.log").unlink()
    with pytest.raises(AssertionError, match="real user directory"):
        assert_untouched(before, fingerprint([d]))


def test_a_live_process_writing_to_its_existing_files_does_not_trip_it(tmp_path):
    """GM-1: a developer's running bridge appends to its log and rewrites state files in place, deep inside."""
    d, nested = _tree(tmp_path)
    before = fingerprint([d])
    with open(d / "runtime-x.log", "a", encoding="utf-8") as f:
        f.write("a later line\n")
    (nested / "memories.db").write_bytes(b"22")
    (nested / "bridge.json").write_text("{}", encoding="utf-8")  # a nested file: below the top level
    later = os.stat(d / "runtime-x.log").st_mtime_ns + 10**9
    os.utime(d / "runtime-x.log", ns=(later, later))
    assert_untouched(before, fingerprint([d]))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux: #488's XDG variables point into the tmp root")
def test_on_linux_the_exposed_directories_stay_in_the_tests_tmp_root(real_platform_dirs, tmp_path_factory):
    from brain import paths

    root = tmp_path_factory.getbasetemp()
    for prop in ("user_data_dir", "user_state_dir"):
        assert str(getattr(paths._dirs, prop)).startswith(str(root)), prop
