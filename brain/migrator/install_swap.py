"""Install a migrated persona over a live one under the persona's start lock (#754).

`nell migrate --install-as X --force` replaces the live persona folder. A running bridge holds the start lock
(`bridge.json.lock`) for its whole life, and every rebuild runs under it, so a migrate that swaps the folders while
the lock is held lets that bridge or rebuild keep writing by path, into the newly installed persona. The migrators
call `refuse_if_in_use` before any migration work and swap the folders with `install_over`, which takes the lock.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.daemon import LOCKFILE, acquire_lock, release_lock

_IS_WINDOWS = sys.platform.startswith("win")


class PersonaInUseError(RuntimeError):
    """A start, rebuild or running bridge holds the persona's start lock, so the install was refused."""

    def __init__(self, final_dir: Path, os_error: OSError | None = None) -> None:
        name = final_dir.name
        if os_error is None:
            why = f"Kindled `{name}` is running or starting: another process holds {final_dir / LOCKFILE}."
            fix = f"Stop it first (quit the app, or run `nell supervisor stop --persona {name}`)"
        else:
            # Windows refused to move the folder: a bridge that opened it, or Explorer, the search indexer or
            # antivirus holding a file in it.
            why = f"Kindled `{name}`'s folder {final_dir} could not be moved: it is running or starting, or a file in it is open ({os_error})."
            fix = f"Stop it (quit the app, or run `nell supervisor stop --persona {name}`) and close anything showing its folder"
        super().__init__(f"{why} {fix}, then run the migrate again. The installed persona was not changed.")


def refuse_if_in_use(final_dir: Path) -> None:
    """Raise `PersonaInUseError` when the persona's start lock is held. Takes the lock and lets it go at once.

    Nothing to check when the folder does not exist."""
    if not final_dir.is_dir():
        return
    fd = acquire_lock(final_dir)
    if fd is None:
        raise PersonaInUseError(final_dir)
    release_lock(final_dir, fd)


def install_over(final_dir: Path, work_dir: Path) -> Path | None:
    """Move the live `final_dir` to `<name>.backup-<UTC time>` and rename `work_dir` into its place, under the start lock.

    Returns the backup folder, or None when there was no live folder. Raises `PersonaInUseError`, with nothing moved,
    while the lock is held. The lock is taken before the first rename and let go after the second, so no start,
    rebuild or bridge can run in between; the locked file moves with the old folder into the backup, and
    `release_lock` only closes the fd, so no second lock file is ever created at the path.
    """
    if not final_dir.exists():
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(work_dir, final_dir)  # atomic-write: ok -- installs a persona directory (a directory rename), not a file write
        return None
    backup = final_dir.with_name(f"{final_dir.name}.backup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}")
    fd: int | None = acquire_lock(final_dir)
    if fd is None:
        raise PersonaInUseError(final_dir)
    if _IS_WINDOWS:
        # Windows won't rename a folder while a file inside it is open, our own lock handle included, so the lock is
        # let go first. A start that opens the folder in between holds a handle there, so the rename below fails.
        release_lock(final_dir, fd)
        fd = None
    try:
        try:
            os.rename(final_dir, backup)  # atomic-write: ok -- moves a persona directory aside to a backup (a directory rename), not a file write
        except PermissionError as exc:
            if not _IS_WINDOWS:
                raise
            raise PersonaInUseError(final_dir, exc) from exc
        try:
            os.rename(work_dir, final_dir)  # atomic-write: ok -- installs a persona directory (a directory rename), not a file write
        except BaseException:
            # A file held open in the build (antivirus on Windows) fails this rename: put the live folder back rather
            # than leave the persona path empty with the live folder in the backup. Still under the lock on POSIX.
            os.rename(backup, final_dir)  # atomic-write: ok -- puts a persona directory back from its backup (a directory rename), not a file write
            raise
    finally:
        if fd is not None:
            release_lock(backup, fd)
    return backup
