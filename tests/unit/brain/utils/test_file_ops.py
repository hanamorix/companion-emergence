"""The shared file accessors (#643): one retry for a transient Windows ``PermissionError``, a byte reader, replace, delete,
list and stat. ``time.sleep`` is patched in every test that retries, so none of them waits."""

from __future__ import annotations

import os
import pathlib
import shutil
from pathlib import Path

import pytest

from brain import dev_constants
from brain.utils import file_ops


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(file_ops.time, "sleep", lambda s: calls.append(s))
    return calls


class Flaky:
    """A callable that raises PermissionError on its first ``fail`` calls, then returns ``result``."""

    def __init__(self, fail: int, result: object = "ok") -> None:
        self.fail, self.result, self.calls = fail, result, 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        self.args, self.kwargs = args, kwargs
        if self.calls <= self.fail:
            raise PermissionError(13, "in use")
        return self.result


def _linear(n: int, step: float = 0.05) -> list[float]:
    return [pytest.approx(step * k) for k in range(1, n + 1)]


# --- retry_transient ---------------------------------------------------------------------------------------------


def test_retry_passes_arguments_and_returns_the_result(sleeps: list[float]) -> None:
    fn = Flaky(0, result=42)
    assert file_ops.retry_transient(fn, 1, 2, key="v") == 42
    assert (fn.args, fn.kwargs, fn.calls, sleeps) == ((1, 2), {"key": "v"}, 1, [])


def test_retry_rides_out_failures_then_succeeds_with_a_growing_sleep(sleeps: list[float]) -> None:
    fn = Flaky(3)
    assert file_ops.retry_transient(fn) == "ok"
    assert fn.calls == 4 and sleeps == _linear(3)


def test_retry_default_schedule_is_eight_tries_about_one_and_a_half_seconds(sleeps: list[float]) -> None:
    fn = Flaky(10_000)
    with pytest.raises(PermissionError):
        file_ops.retry_transient(fn)
    assert fn.calls == 8
    assert sleeps == _linear(7)
    assert sum(sleeps) == pytest.approx(1.4)


def test_retry_attempts_and_sleep_can_be_given(sleeps: list[float]) -> None:
    fn = Flaky(10_000)
    with pytest.raises(PermissionError):
        file_ops.retry_transient(fn, attempts=3, sleep_s=0.5)
    assert fn.calls == 3 and sleeps == _linear(2, step=0.5)


def test_retry_defaults_are_read_from_the_dev_constants_at_call_time(
    sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dev_constants, "FILE_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(dev_constants, "FILE_RETRY_SLEEP_S", 0.25)
    fn = Flaky(10_000)
    with pytest.raises(PermissionError):
        file_ops.retry_transient(fn)
    assert fn.calls == 2 and sleeps == [0.25]


def test_retry_does_not_retry_another_error(sleeps: list[float]) -> None:
    calls = []

    def boom() -> None:
        calls.append(1)
        raise FileNotFoundError("gone")

    with pytest.raises(FileNotFoundError):
        file_ops.retry_transient(boom)
    assert calls == [1] and sleeps == []


@pytest.mark.parametrize("attempts", [0, -1])
def test_retry_needs_at_least_one_attempt(attempts: int) -> None:
    with pytest.raises(ValueError):
        file_ops.retry_transient(lambda: None, attempts=attempts)


def test_windows_only_makes_one_attempt_off_windows(sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_ops, "_IS_WINDOWS", False)
    fn = Flaky(10_000)
    with pytest.raises(PermissionError):
        file_ops.retry_transient(fn, windows_only=True)
    assert fn.calls == 1 and sleeps == []


def test_windows_only_runs_the_full_schedule_on_windows(sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_ops, "_IS_WINDOWS", True)
    fn = Flaky(10_000)
    with pytest.raises(PermissionError):
        file_ops.retry_transient(fn, attempts=3, windows_only=True)
    assert fn.calls == 3 and sleeps == _linear(2)


# --- read_bytes, read_bytes_stat, stat_key ---------------------------------------------------------------------------


def test_read_bytes_stat_gives_the_bytes_and_the_stat_key(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello")
    data, key = file_ops.read_bytes_stat(path)
    st = os.stat(path)
    assert data == b"hello"
    assert key == (st.st_ino, st.st_mtime_ns, st.st_size) == file_ops.stat_key(st)
    assert file_ops.read_bytes(path) == b"hello" and file_ops.read_bytes(str(path)) == b"hello"


@pytest.mark.skipif(os.name == "nt", reason="a file that is open cannot be replaced on Windows")
def test_read_bytes_stat_takes_the_key_from_the_open_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"old")
    old_key = file_ops.stat_key(os.stat(path))
    other = tmp_path / "b.bin"
    other.write_bytes(b"a different, longer file")
    real_open = open

    def swapping_open(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        os.replace(other, path)  # the path now names another file; the open handle still has the old one
        return handle

    monkeypatch.setattr(file_ops, "open", swapping_open, raising=False)
    data, key = file_ops.read_bytes_stat(path)
    assert data == b"old" and key == old_key
    assert file_ops.stat_key(os.stat(path)) != old_key


def test_read_bytes_of_a_missing_file_raises_at_once(tmp_path: Path, sleeps: list[float]) -> None:
    with pytest.raises(FileNotFoundError):
        file_ops.read_bytes(tmp_path / "missing")
    assert sleeps == []


def _deny_open(monkeypatch: pytest.MonkeyPatch, counter: list[int]) -> None:
    def denied(file, *args, **kwargs):
        counter.append(1)
        raise PermissionError(13, "busy")

    monkeypatch.setattr(file_ops, "open", denied, raising=False)


def test_a_read_retries_a_sharing_violation_on_windows_three_times(
    tmp_path: Path, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(file_ops, "_IS_WINDOWS", True)
    tries: list[int] = []
    _deny_open(monkeypatch, tries)
    with pytest.raises(PermissionError):
        file_ops.read_bytes_stat(tmp_path / "a")
    assert len(tries) == dev_constants.FILE_READ_ATTEMPTS == 3 and sleeps == _linear(2)


def test_a_read_makes_one_attempt_off_windows(tmp_path: Path, sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_ops, "_IS_WINDOWS", False)
    tries: list[int] = []
    _deny_open(monkeypatch, tries)
    with pytest.raises(PermissionError):
        file_ops.read_bytes_stat(tmp_path / "a")
    assert len(tries) == 1 and sleeps == []


# --- replace_file ----------------------------------------------------------------------------------------------------


def test_replace_file_replaces_a_file_and_a_folder(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.write_bytes(b"new")
    dst.write_bytes(b"old")
    file_ops.replace_file(src, dst)
    assert dst.read_bytes() == b"new" and not src.exists()
    folder, target = tmp_path / "f", tmp_path / "t"
    folder.mkdir()
    (folder / "x").write_bytes(b"1")
    file_ops.replace_file(folder, target)
    assert (target / "x").read_bytes() == b"1" and not folder.exists()


def _patch_replace(monkeypatch: pytest.MonkeyPatch, fail: int) -> Flaky:
    real = os.replace
    flaky = Flaky(fail)

    def replace(src, dst):
        flaky(src, dst)
        real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    return flaky


def test_replace_file_rides_out_a_transient_sharing_violation(tmp_path: Path, sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.write_bytes(b"new")
    flaky = _patch_replace(monkeypatch, 7)
    file_ops.replace_file(src, dst)
    assert flaky.calls == 8 and dst.read_bytes() == b"new" and sleeps == _linear(7)


def test_replace_file_raises_when_the_failure_persists(tmp_path: Path, sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.write_bytes(b"new")
    dst.write_bytes(b"old")
    flaky = _patch_replace(monkeypatch, 10_000)
    with pytest.raises(PermissionError):
        file_ops.replace_file(src, dst)
    assert flaky.calls == 8 and dst.read_bytes() == b"old" and src.read_bytes() == b"new"


# --- delete_file ---------------------------------------------------------------------------------------------------


def test_delete_file_removes_a_file_and_a_missing_one_is_fine(tmp_path: Path) -> None:
    path = tmp_path / "a"
    path.write_bytes(b"x")
    file_ops.delete_file(path)
    assert not path.exists()
    file_ops.delete_file(path)
    file_ops.delete_file(str(tmp_path / "never"))


def _patch_unlink(monkeypatch: pytest.MonkeyPatch, fail: int) -> Flaky:
    real = pathlib.Path.unlink
    flaky = Flaky(fail)

    def unlink(self, missing_ok=False):
        flaky()
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    return flaky


def test_delete_file_rides_out_a_transient_sharing_violation(tmp_path: Path, sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a"
    path.write_bytes(b"x")
    flaky = _patch_unlink(monkeypatch, 2)
    file_ops.delete_file(path)
    assert flaky.calls == 3 and not path.exists() and sleeps == _linear(2)


def test_delete_file_raises_when_the_failure_persists(tmp_path: Path, sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a"
    path.write_bytes(b"x")
    flaky = _patch_unlink(monkeypatch, 10_000)
    with pytest.raises(PermissionError):
        file_ops.delete_file(path)
    assert flaky.calls == 8 and path.read_bytes() == b"x"


def test_delete_file_never_removes_a_folder(tmp_path: Path, sleeps: list[float]) -> None:
    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "x").write_bytes(b"1")
    with pytest.raises(OSError):  # IsADirectoryError on Linux; PermissionError (retried) on macOS and Windows
        file_ops.delete_file(folder)
    assert (folder / "x").read_bytes() == b"1"


# --- delete_tree ---------------------------------------------------------------------------------------------------


def test_delete_tree_removes_a_folder_with_content_and_a_missing_one_is_fine(tmp_path: Path) -> None:
    folder = tmp_path / "d"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "x").write_bytes(b"1")
    (folder / "y").write_bytes(b"2")
    file_ops.delete_tree(folder)
    assert not folder.exists()
    file_ops.delete_tree(folder)
    file_ops.delete_tree(str(tmp_path / "never"))


def _patch_rmtree(monkeypatch: pytest.MonkeyPatch, fail: int) -> Flaky:
    real = shutil.rmtree
    flaky = Flaky(fail)

    def rmtree(path, *args, **kwargs):
        flaky()
        return real(path, *args, **kwargs)

    monkeypatch.setattr(file_ops.shutil, "rmtree", rmtree)
    return flaky


def test_delete_tree_rides_out_a_transient_sharing_violation(tmp_path: Path, sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "x").write_bytes(b"1")
    flaky = _patch_rmtree(monkeypatch, 2)
    file_ops.delete_tree(folder)
    assert flaky.calls == 3 and not folder.exists() and sleeps == _linear(2)


def test_delete_tree_raises_when_the_failure_persists(tmp_path: Path, sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "x").write_bytes(b"1")
    flaky = _patch_rmtree(monkeypatch, 10_000)
    with pytest.raises(PermissionError):
        file_ops.delete_tree(folder)
    assert flaky.calls == 8 and (folder / "x").exists()


def test_delete_tree_never_removes_a_file(tmp_path: Path, sleeps: list[float]) -> None:
    path = tmp_path / "a"
    path.write_bytes(b"x")
    with pytest.raises(NotADirectoryError if os.name != "nt" else OSError):
        file_ops.delete_tree(path)
    assert path.read_bytes() == b"x"


def test_delete_tree_does_not_follow_a_symlink(tmp_path: Path, requires_symlinks: None, sleeps: list[float]) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep").write_bytes(b"1")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(NotADirectoryError, match="symbolic link"):
        file_ops.delete_tree(link)
    assert (target / "keep").read_bytes() == b"1" and link.is_symlink()


def test_delete_tree_does_not_report_a_dangling_symlink_as_deleted(tmp_path: Path, requires_symlinks: None, sleeps: list[float]) -> None:
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "nowhere")
    with pytest.raises(NotADirectoryError):
        file_ops.delete_tree(link)
    assert link.is_symlink()


# --- list_dir ------------------------------------------------------------------------------------------------------


def test_list_dir_gives_the_sorted_entry_names(tmp_path: Path) -> None:
    for name in ("b", "a", "c.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    assert file_ops.list_dir(tmp_path) == ["a", "b", "c.txt", "sub"]
    assert file_ops.list_dir(str(tmp_path)) == ["a", "b", "c.txt", "sub"]


def test_list_dir_of_a_missing_folder_is_empty_and_creates_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert file_ops.list_dir(missing) == []
    assert not missing.exists()


def test_list_dir_of_a_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "a"
    path.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        file_ops.list_dir(path)


# --- stat_path -----------------------------------------------------------------------------------------------------


def test_stat_path_of_a_missing_path_is_none(tmp_path: Path) -> None:
    assert file_ops.stat_path(tmp_path / "nope") is None


def test_stat_path_flags_a_file_and_a_folder(tmp_path: Path) -> None:
    path = tmp_path / "a"
    path.write_bytes(b"12345")
    st = file_ops.stat_path(path)
    assert st is not None
    assert (st.is_file, st.is_dir, st.size) == (True, False, 5)
    assert st.mtime_ns == os.stat(path).st_mtime_ns and st.inode == os.stat(path).st_ino
    folder = file_ops.stat_path(tmp_path)
    assert folder is not None and (folder.is_file, folder.is_dir) == (False, True)


def test_stat_path_key_equals_the_key_read_bytes_stat_returns(tmp_path: Path) -> None:
    path = tmp_path / "a"
    path.write_bytes(b"payload")
    st = file_ops.stat_path(path)
    assert st is not None
    _data, key_read = file_ops.read_bytes_stat(path)
    assert st.key == key_read == (st.inode, st.mtime_ns, st.size)
    path.write_bytes(b"payload, longer now")
    changed = file_ops.stat_path(path)
    assert changed is not None and changed.key != st.key


@pytest.mark.parametrize("error", [PermissionError(13, "denied"), OSError(5, "I/O error")])
def test_stat_path_raises_an_error_other_than_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    def broken(path, *args, **kwargs):
        raise error

    monkeypatch.setattr(os, "stat", broken)
    with pytest.raises(type(error)):
        file_ops.stat_path(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="the Windows error for a path under a file is not checked here")
def test_stat_path_under_a_file_raises_not_a_directory(tmp_path: Path) -> None:
    path = tmp_path / "a"
    path.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        file_ops.stat_path(path / "child")


def test_stat_path_under_a_file_follows_the_rule_on_every_system(tmp_path: Path) -> None:
    """The rule is the same everywhere: a missing path is ``None`` and any other ``OSError`` raises. What a path under a file is
    reported as differs by system (``NotADirectoryError`` on POSIX; not checked for Windows), so the expectation is what
    ``os.stat`` itself reports here. This test is not skipped on Windows: the Windows CI job is where it says something new."""
    path = tmp_path / "a"
    path.write_bytes(b"")
    child = path / "child"
    try:
        os.stat(child)
    except FileNotFoundError:
        assert file_ops.stat_path(child) is None
    except OSError as err:
        with pytest.raises(type(err)):
            file_ops.stat_path(child)
    else:
        pytest.fail("os.stat found a path under a file")
