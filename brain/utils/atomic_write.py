"""The shared atomic writers: one home for every state-file write (#465 C2).

A write is a unique per-save temp in the target's directory, flushed and fsynced, then published with one atomic
``os.replace`` that retries a transient Windows ``PermissionError``; the directory is fsynced after, and the save's own
temp is removed if the save fails. This module holds the text writer with its optional ``.bak`` chain
(:func:`save_with_backup_text`, which ``brain.health.json_io`` writes through), the bytes variant
(:func:`save_bytes_atomic`), the one directory fsync (:func:`fsync_dir`) and the content-addressed blob writer shared by
images and file_store (:func:`save_blob_once`). The healing readers stay in ``brain.health.attempt_heal``.

Exclusive-create cases (the machine id, model blobs) and plain moves of finished files do not belong here; they carry an
``# atomic-write: ok -- <why>`` marker that ``tests/unit/test_no_hand_rolled_atomic_write.py`` requires.

This module imports only the standard library at module level and never ``brain.health``: it sits below it, so every
module may call it without an import cycle.
"""

from __future__ import annotations

import contextlib
import glob
import os
import re
import secrets
import shutil
import stat
import threading
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

_REPLACE_RETRIES = 8
_REPLACE_RETRY_SLEEP_S = 0.05


def _retry_perm(action: Callable[[], Any]) -> Any:
    """Run ``action`` with a short bounded retry on transient PermissionError.

    Windows refuses to rename a file another handle has open (WinError 32).
    The supervisor reads persona_config.json inside its ticks in the same
    process as the settings writer, so a save can collide with a read (#200).
    Mirrors brain.ingest.buffer._unlink_with_retry: a transient failure is
    retried, and a persistent one (on POSIX that means a real permission
    problem, which burns the whole ~1.4 s budget) is re-raised so it still
    fails loud. ``action`` must be safe to run again from the top.
    """
    for attempt in range(_REPLACE_RETRIES):
        try:
            return action()
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_RETRY_SLEEP_S * (attempt + 1))


def _replace_with_retry(src: Path, dst: Path) -> None:
    """os.replace with a short bounded retry on transient PermissionError (#200)."""
    _retry_perm(lambda: os.replace(src, dst))


def _copy_for_backup(src: Path, dst: Path) -> bool:
    """Copy ``src`` to ``dst`` (a temp name) for the backup chain; True if a copy was made.

    The live file is copied, never renamed, so the live path is never absent
    (#387). The copy keeps the source's mode (it may hold a token) from the
    moment it is created, and its mtime, as the old rename did; the data is
    fsynced so a backup is as durable as the file it replaces. Each retry
    attempt starts from a fresh temp, so a half-made copy cannot make the next
    attempt fail with something other than PermissionError. A source that has
    vanished means there is nothing to back up. Any other failure removes the
    partial temp and re-raises; the caller has not rotated anything yet.
    """

    def attempt() -> bool:
        dst.unlink(missing_ok=True)
        try:
            src_file = open(src, "rb")
        except FileNotFoundError:
            return False
        with src_file:
            mode = stat.S_IMODE(os.fstat(src_file.fileno()).st_mode)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
            with os.fdopen(os.open(dst, flags, mode), "wb") as out:
                shutil.copyfileobj(src_file, out)
                out.flush()
                os.fsync(out.fileno())
        shutil.copystat(src, dst)
        return True

    try:
        return _retry_perm(attempt)
    except Exception:
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# Per-writer temp files (#389). Two savers of one path used to share one
# ``<name>.new`` and one ``<name>.bak1.new``, so one could unlink or replace the
# other's in-flight temp and the loser raised FileNotFoundError. Every save now
# names its own temps ``<name>.t<pid>-<random>.new`` (and
# ``...bak1.new`` for the backup copy). Leftovers of a killed save are found by
# that pattern and removed without touching a temp another writer is still
# using: a temp registered in this process as in flight (keyed by its unique
# file name, so the spelling of the path does not matter) is never removed;
# one from this process that is not registered, or from a dead process, is
# stale; one owned by another live process is left.
_INFLIGHT: set[str] = set()
_INFLIGHT_LOCK = threading.Lock()


def _owner_alive(pid: int, digest: str | None = None) -> bool:
    # Lazy import: the bridge sits above this module (its state_file imports health, which imports this module), so
    # a module-level import would be a cycle. The bridge's pid check has the Windows-safe branch; a raw
    # os.kill(pid, 0) must not be used (on Windows signal 0 is a real Ctrl+C). A temp named with its writer's identity
    # digest (#472) is judged by that identity, so a reused pid does not keep a dead writer's temp; an older name,
    # by its pid only.
    try:
        if digest:
            from brain import process_identity as pi

            tok = pi.token_of(pid)
            if tok == pi.UNKNOWN:
                return True
            return tok != pi.GONE and pi.token_digest(tok) == digest
        from brain.bridge.state_file import pid_is_alive

        return pid_is_alive(pid)
    except Exception:  # housekeeping must never fail a save: unknown means alive, leave it
        return True


def _remove_stale_temps(path: Path) -> None:
    """Remove leftover save temps of ``path``; never another writer's live temp."""
    for legacy in (path.name + ".new", path.name + ".bak1.new"):  # pre-#389 fixed names
        with contextlib.suppress(OSError):
            path.with_name(legacy).unlink(missing_ok=True)
    pattern = re.compile(rf"^{re.escape(path.name)}\.t(\d+)-(?:k([0-9a-f]{{12}})-)?[0-9a-f]+\.(?:bak1\.)?new$")
    try:
        found = list(path.parent.glob(f"{glob.escape(path.name)}.t*-*.new"))
    except OSError:
        return
    for temp in found:
        m = pattern.match(temp.name)
        if m is None:
            continue
        pid = int(m.group(1))
        with _INFLIGHT_LOCK:
            if temp.name in _INFLIGHT:
                continue
        if pid == os.getpid() or not _owner_alive(pid, m.group(2)):
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)


@contextlib.contextmanager
def _temp_scope(path: Path) -> Iterator[tuple[Path, Path]]:
    """This save's own temp paths, registered as in flight; stale ones swept first.

    On an Exception, a KeyboardInterrupt or a SystemExit (everything a LIVE process can observe) the save's own temps
    are removed. A kill runs no code, so a BaseException that models one (``SimulatedCrash`` in the kill-simulation
    tests) is deliberately NOT caught: its leftovers stay and the next save sweeps them (#389). The registration is
    always dropped."""
    from brain import process_identity

    own = process_identity.own_token()
    # #472: the writer's identity digest in the name ("-k<digest>-"), so a reused pid cannot keep a dead writer's temp;
    # older versions' sweep regex does not match this shape and leaves it alone.
    ident = f"k{process_identity.token_digest(own)}-" if own else ""
    tag = f"t{os.getpid()}-{ident}{secrets.token_hex(6)}"
    new_path = path.with_name(f"{path.name}.{tag}.new")
    bak1_new = path.with_name(f"{path.name}.{tag}.bak1.new")
    with _INFLIGHT_LOCK:
        _INFLIGHT.update((new_path.name, bak1_new.name))
    try:
        _remove_stale_temps(path)
        yield new_path, bak1_new
    except (Exception, KeyboardInterrupt, SystemExit):
        for temp in (new_path, bak1_new):
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)
        raise
    finally:
        with _INFLIGHT_LOCK:
            _INFLIGHT.difference_update((new_path.name, bak1_new.name))


def _save_payload(
    path: Path,
    write_temp: Callable[[Path], None],
    *,
    fsync: bool,
    keep_backups: bool = False,
    backup_count: int = 3,
    backup_mode: int | None = None,
) -> None:
    """The one save engine under the text and bytes writers.

    ``write_temp(new_path)`` creates the save's own temp with the payload, flushed and (when ``fsync``) fsynced; it
    is the only part that differs between text and bytes (and it applies the file's own ``mode``). Everything else is
    shared: the per-save temp scope, the retried replace that publishes the temp and the directory fsync. The backup
    chain is optional: with ``keep_backups`` the live file is copied and ``.bak1`` .. ``.bak{backup_count}`` are
    rotated before the publish, and ``backup_mode`` (when given) is set on every rotated backup (a token file's
    backups must never be wider than the file). The bytes writer uses none of the chain.
    """
    bak1 = path.with_name(f"{path.name}.bak1")
    with _temp_scope(path) as (new_path, bak1_new):
        write_temp(new_path)

        # The slow, failure-prone copy happens before anything is rotated: if it
        # fails (disk full) the live file and the whole chain are untouched.
        backed_up = _copy_for_backup(path, bak1_new) if keep_backups else False

        # Rotate: drop oldest first, then walk down toward live. A step whose
        # source another saver just moved is skipped, not an error.
        if keep_backups:
            path.with_name(f"{path.name}.bak{backup_count}").unlink(missing_ok=True)
            for i in range(backup_count - 1, 0, -1):
                src = path.with_name(f"{path.name}.bak{i}")
                dst = path.with_name(f"{path.name}.bak{i + 1}")
                if src.exists():
                    try:
                        _replace_with_retry(src, dst)
                    except FileNotFoundError:
                        pass
        if backed_up:
            _replace_with_retry(bak1_new, bak1)

        _replace_with_retry(new_path, path)

        if backup_mode is not None and keep_backups:
            for i in range(1, max(backup_count, 1) + 1):
                try:
                    os.chmod(path.with_name(f"{path.name}.bak{i}"), backup_mode)
                except OSError:
                    pass

        # M-9: fsync the parent directory so the rename is durable, not just
        # atomic. POSIX guarantees os.replace's atomicity but not that the
        # directory entry has flushed to disk. On power loss between the
        # replace and the dir-block flush, the file may revert to the prior
        # name; the .bak rotation already preserves data, but durability
        # of the rename itself matters for the chain to behave correctly.
        if fsync:
            fsync_dir(path.parent)


def fsync_dir(directory: Path) -> None:
    """fsync a directory so a newly created or renamed entry is durable (M-9, C11).

    POSIX only: Windows has no ``O_DIRECTORY`` and raises, so it is skipped there. Best effort: a directory that
    cannot be opened or synced is ignored, because durability of the rename is a bonus on top of its atomicity and
    must never fail a save. The one directory fsync in the app; ``ingest.buffer`` and the writers below call it.
    """
    if os.name != "posix":
        return
    try:
        dir_fd = os.open(str(directory), os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _write_temp_file(
    new_path: Path,
    payload: str | bytes,
    *,
    mode: int | None,
    fsync: bool,
    encoding: str | None = None,
    newline: str | None = None,
) -> None:
    """Create the save's own temp with ``payload``, flushed and (when ``fsync``) fsynced, so the data is on disk before
    the replace that publishes it. The one temp-open under the text and the bytes writer.

    A ``str`` payload is written in text mode with ``encoding`` and ``newline`` (the same open as ``Path.write_text``,
    so the bytes written per platform are unchanged); ``bytes`` are written as they are. With ``mode`` the temp is
    created exclusively with that mode at ``os.open`` time (a secret is never wider than ``mode`` even for an instant,
    subject to the umask).
    """
    open_mode = "w" if isinstance(payload, str) else "wb"
    text_options = {"encoding": encoding, "newline": newline} if isinstance(payload, str) else {}
    if mode is None:
        new_file = new_path.open(open_mode, **text_options)
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        new_file = os.fdopen(os.open(new_path, flags, mode), open_mode, **text_options)
    with new_file as f:
        f.write(payload)
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def save_with_backup_text(
    path: Path,
    text: str,
    backup_count: int = 3,
    *,
    keep_backups: bool = True,
    mode: int | None = None,
    fsync: bool = True,
    newline: str | None = None,
    encoding: str = "utf-8",
) -> None:
    """Atomic raw-text save with .bak rotation.

    The live path always holds a complete file (#387): the new text is written
    and fsynced to a per-save temp (<path>.t<id>.new), the live file is COPIED
    (not renamed) to a second per-save temp (<path>.t<id>.bak1.new), the chain
    is rotated <path>.bak1 → ... → <path>.bak{N} (oldest dropped), the backup
    temp becomes <path>.bak1, and only then does one atomic replace put the new
    temp at <path>. A kill at any point leaves the old or the new file complete
    at <path>. Temps are named per save so concurrent savers of one path never
    share one (#389); leftovers of a killed save are removed by pattern at the
    start of the next save, never another writer's in-flight temp.

    The text variant — for callers whose payload is already a string and
    should not be JSON-encoded (markdown bodies, voice.md, work files).
    For JSON dicts, use save_with_backup which delegates here after encoding.

    Keyword-only additions for brain.health.json_io (every default reproduces
    the behaviour above byte for byte): ``keep_backups=False`` makes no backup
    copy and leaves any existing chain alone (``backup_count=0`` keeps its old
    meaning of one backup); ``mode`` creates the new file with that mode and
    sets it on every rotated backup (a token file must never be wider, even for
    an instant); ``fsync=False`` skips the new file's fsync and the directory
    fsync (the backup copy is still fsynced); ``newline`` is passed to the
    temp file's open (``"\n"`` gives the same bytes on every platform);
    ``encoding`` is the text encoding (default utf-8; the Windows task XML is UTF-16).
    """
    write_temp = partial(_write_temp_file, payload=text, mode=mode, fsync=fsync, encoding=encoding, newline=newline)
    _save_payload(
        path, write_temp, fsync=fsync, keep_backups=keep_backups, backup_count=backup_count, backup_mode=mode
    )


def save_bytes_atomic(path: Path, data: bytes, *, mode: int | None = None, fsync: bool = True) -> None:
    """Atomic raw-bytes save: the bytes variant of :func:`save_with_backup_text` (no backup chain).

    A unique per-save temp in ``path``'s directory is written (created with ``mode`` when given, so a secret is never
    wider than ``mode`` even for an instant, subject to the umask), flushed and, when ``fsync``, fsynced; one retried
    atomic replace publishes it and the directory is fsynced. A failure leaves ``path`` as it was (absent stays absent)
    and removes the save's own temp. For binary payloads: the identity key, uploaded blobs, a copied legacy file, a gzip
    archive.
    """
    write_temp = partial(_write_temp_file, payload=data, mode=mode, fsync=fsync)
    _save_payload(path, write_temp, fsync=fsync)


def save_blob_once(target: Path, data: bytes) -> None:
    """Write a content-addressed blob once: the one writer shared by ``brain.images`` and ``brain.file_store``.

    The name is the content's hash, so a target that already exists holds these bytes and nothing is written (dedup by
    name). Otherwise the bytes are saved atomically; if that fails but the target now exists, a concurrent writer of
    the same content won the race (identical concurrent uploads end with one complete file and no error, b557882d) and
    this is success; any other failure is re-raised with no temp left behind.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        save_bytes_atomic(target, data)
    except OSError:
        if target.exists():
            return
        raise
