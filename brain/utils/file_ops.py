"""The shared file accessors: one retry for a transient Windows ``PermissionError``, and the plain file operations that use it.

Windows refuses to replace, delete or open a file another handle holds for a moment (WinError 32). :func:`retry_transient`
is the one loop for that: a failed try is repeated on a short schedule and a failure that persists is raised, so a real
permission problem still fails loudly. Replace, delete and the whole-file byte read go through it. The schedule lives in
``brain/dev_constants.py`` (``FILE_RETRY_ATTEMPTS``, ``FILE_RETRY_SLEEP_S``, ``FILE_READ_ATTEMPTS``) and is read at the time of
the call.

Also here, because no shared function existed for them: :func:`list_dir` (the sorted names in a folder) and :func:`stat_path`
(one ``os.stat``, with the key a cache compares). Neither creates anything. JSON and JSONL are read and written through
``brain.health.json_io``, which reads its bytes with :func:`read_bytes_stat`.

This module sits below ``brain.utils.atomic_write`` and ``brain.health``: it imports only the standard library and
``brain.dev_constants``, so every module may call it without an import cycle.
"""

from __future__ import annotations

import os
import shutil
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from brain import dev_constants

_IS_WINDOWS = os.name == "nt"


def retry_transient[T](
    fn: Callable[..., T],
    *args: Any,
    attempts: int | None = None,
    sleep_s: float | None = None,
    windows_only: bool = False,
    **kwargs: Any,
) -> T:
    """Call ``fn(*args, **kwargs)``, retrying a ``PermissionError`` (the Windows sharing violation).

    Up to ``attempts`` tries in all (default ``FILE_RETRY_ATTEMPTS``), sleeping ``sleep_s`` (default ``FILE_RETRY_SLEEP_S``)
    times the number of the try that just failed; the last failure is raised. With ``windows_only`` one attempt is made on
    other systems, where a ``PermissionError`` is a real one. Any other exception is raised at once. ``fn`` must be safe to
    run again from the top. ``attempts`` and ``sleep_s`` are keyword-only and are not passed on to ``fn``.
    """
    if attempts is None:
        attempts = dev_constants.FILE_RETRY_ATTEMPTS
    if sleep_s is None:
        sleep_s = dev_constants.FILE_RETRY_SLEEP_S
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    tries = 1 if windows_only and not _IS_WINDOWS else attempts
    for n in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except PermissionError:
            if n == tries:
                raise
            time.sleep(sleep_s * n)
    raise AssertionError("unreachable")  # pragma: no cover


def stat_key(st: os.stat_result) -> tuple[int, int, int]:
    """The key a cache compares to see that a file has not changed: ``(inode, mtime_ns, size)``. The one definition."""
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _read_once(path: str | Path) -> tuple[bytes, tuple[int, int, int]]:
    with open(path, "rb") as fh:
        key = stat_key(os.fstat(fh.fileno()))
        return fh.read(), key


def read_bytes_stat(path: str | Path) -> tuple[bytes, tuple[int, int, int]]:
    """Read the whole file and return ``(bytes, (inode, mtime_ns, size))``, the key taken from the open handle.

    A Windows sharing violation is retried briefly (``FILE_READ_ATTEMPTS``); on other systems one attempt is made. A missing
    file raises ``FileNotFoundError``. The key equals ``stat_path(path).key`` for the same file, so a key taken before the
    read compares with the read's.
    """
    return retry_transient(_read_once, path, attempts=dev_constants.FILE_READ_ATTEMPTS, windows_only=True)


def read_bytes(path: str | Path) -> bytes:
    """Read the whole file (:func:`read_bytes_stat` without the key)."""
    return read_bytes_stat(path)[0]


def replace_file(src: str | Path, dst: str | Path) -> None:
    """``os.replace`` (a file, or a folder where the system allows it) with the transient-failure retry."""
    retry_transient(os.replace, src, dst)


def delete_file(path: str | Path) -> None:
    """Delete a file with the transient-failure retry. A missing file is not an error; a failure that persists is raised.

    A folder is not deleted (use :func:`delete_tree`): the delete raises an ``OSError``.
    """
    retry_transient(lambda: Path(path).unlink(missing_ok=True))


def _remove_tree(path: str | Path) -> None:
    if os.path.islink(path):
        raise NotADirectoryError(f"{path} is a symbolic link: delete_tree does not follow it")
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return  # already gone, or removed meanwhile by another process


def delete_tree(path: str | Path) -> None:
    """Delete a folder and everything in it, with the transient-failure retry.

    A missing folder is not an error; a failure that persists is raised. A file or a symbolic link is refused (an
    ``OSError``), never followed or removed, so only the folder named is deleted. A retry after a partial removal carries on
    with what is left.
    """
    retry_transient(_remove_tree, path)


def list_dir(folder: str | Path) -> list[str]:
    """The names of the entries in ``folder``, sorted. A missing folder is ``[]`` and is not created; any other error raises."""
    try:
        return sorted(os.listdir(folder))
    except FileNotFoundError:
        return []


class PathStat(NamedTuple):
    """One ``os.stat`` of a path: the three fields of :func:`stat_key` and whether it is a regular file or a folder."""

    inode: int
    mtime_ns: int
    size: int
    is_file: bool
    is_dir: bool

    @property
    def key(self) -> tuple[int, int, int]:
        """``(inode, mtime_ns, size)``: equal to what :func:`read_bytes_stat` returns for the same, unchanged file."""
        return self[:3]


def stat_path(path: str | Path) -> PathStat | None:
    """One ``os.stat`` of ``path`` (symbolic links followed). A missing path is ``None``; any other ``OSError`` raises.

    The accessor for a metadata check: exists (``is not None``), size, file or folder, and a cache's currency key. The caller
    decides what to log when it raises.
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return PathStat(*stat_key(st), stat.S_ISREG(st.st_mode), stat.S_ISDIR(st.st_mode))
