"""Core heal + save helpers — atomic .bak rotation + corruption recovery."""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
import re
import secrets
import shutil
import stat
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.health.anomaly import BrainAnomaly
from brain.utils.time import iso_utc

logger = logging.getLogger(__name__)


def _filename_safe_timestamp(now: datetime) -> str:
    """Build an ISO-like timestamp safe for filenames on every platform.

    `iso_utc(now)` produces strings with `:` (e.g. `2026-04-25T18:30:00.123456Z`),
    which Windows rejects in filenames. We swap colons for hyphens so the
    quarantine filename round-trips on POSIX + Windows.
    """
    return iso_utc(now).replace(":", "-")


def _classify_cause(path: Path) -> str:
    """Heuristic: hand-edit vs disk vs unknown.

    user_edit: mtime within 60s, size < 100KB, content starts with { or [.
    Otherwise unknown. (No specific disk-error heuristic in v1.)
    """
    try:
        st = path.stat()
        if time.time() - st.st_mtime < 60 and st.st_size < 100_000:
            head = path.read_bytes()[:1]
            if head and head[:1] in (b"{", b"["):
                return "user_edit"
    except OSError:
        pass
    return "unknown"


def attempt_heal_text(
    path: Path,
    default_factory: Callable[[], str],
) -> tuple[str, BrainAnomaly | None]:
    """Like the JSON heal (brain.health.json_io) but for plain-text files (no JSON parsing).

    Missing → (default_factory(), None) silently.
    Empty → quarantine, walk .bak1/.bak2/.bak3, restore freshest valid backup.
    All baks missing or empty → write default text + return anomaly.

    What counts as "corrupt" for plain text: empty file (len == 0 after strip).
    This is the practical failure mode after a disk hiccup on an identity file.
    """
    if not path.exists():
        default_text = default_factory()
        path.write_text(default_text, encoding="utf-8")
        return default_text, None

    content = path.read_text(encoding="utf-8")
    if content.strip():
        return content, None

    # Empty file — treat as corrupt; start the heal cycle.
    return _heal_text_from_baks(path, default_factory)


def _heal_text_from_baks(
    path: Path,
    default_factory: Callable[[], str],
) -> tuple[str, BrainAnomaly]:
    now = datetime.now(UTC)
    likely_cause = _classify_cause(path)
    quarantine = path.with_name(f"{path.name}.corrupt-{_filename_safe_timestamp(now)}")
    os.replace(path, quarantine)

    for bak_index in (1, 2, 3):
        bak = path.with_name(f"{path.name}.bak{bak_index}")
        if not bak.exists():
            continue
        bak_content = bak.read_text(encoding="utf-8")
        if not bak_content.strip():
            # This bak is also empty — quarantine it too.
            bak_quarantine = path.with_name(
                f"{path.name}.bak{bak_index}.corrupt-{_filename_safe_timestamp(now)}"
            )
            os.replace(bak, bak_quarantine)
            continue

        # Found a valid bak — restore.
        os.replace(bak, path)
        return (
            bak_content,
            BrainAnomaly(
                timestamp=now,
                file=path.name,
                kind="json_parse_error",  # closest kind; text "empty" maps here
                action=f"restored_from_bak{bak_index}",  # type: ignore[arg-type]
                quarantine_path=quarantine.name,
                likely_cause=likely_cause,
                detail="empty text file",
            ),
        )

    # All baks corrupt or missing — reset to default.
    default_text = default_factory()
    path.write_text(default_text, encoding="utf-8")
    return (
        default_text,
        BrainAnomaly(
            timestamp=now,
            file=path.name,
            kind="json_parse_error",
            action="reset_to_default",
            quarantine_path=quarantine.name,
            likely_cause=likely_cause,
            detail="empty text file, all baks missing or empty",
        ),
    )


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
    # Lazy import: bridge sits above health and imports this module. The bridge's
    # pid check has the Windows-safe branch; a raw os.kill(pid, 0) must not be
    # used (on Windows signal 0 is a real Ctrl+C). A temp named with its writer's identity digest (#472) is judged by
    # that identity, so a reused pid does not keep a dead writer's temp; an older name, by its pid only.
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

    On an Exception (not a kill, which runs no code) the save's own temps are
    removed. The registration is always dropped."""
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
    except Exception:
        for temp in (new_path, bak1_new):
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)
        raise
    finally:
        with _INFLIGHT_LOCK:
            _INFLIGHT.difference_update((new_path.name, bak1_new.name))


def save_with_backup_text(
    path: Path,
    text: str,
    backup_count: int = 3,
    *,
    keep_backups: bool = True,
    mode: int | None = None,
    fsync: bool = True,
    newline: str | None = None,
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
    temp file's open (``"\n"`` gives the same bytes on every platform).
    """
    bak1 = path.with_name(f"{path.name}.bak1")
    with _temp_scope(path) as (new_path, bak1_new):
        # Same open as Path.write_text (text mode; no newline= unless asked) so
        # the bytes written per platform are unchanged; flushed and fsynced so
        # the data is on disk before the replace below can publish it.
        if mode is None:
            new_file = new_path.open("w", encoding="utf-8", newline=newline)
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            new_file = os.fdopen(
                os.open(new_path, flags, mode), "w", encoding="utf-8", newline=newline
            )
        with new_file as f:
            f.write(text)
            f.flush()
            if fsync:
                os.fsync(f.fileno())

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

        if mode is not None and keep_backups:
            for i in range(1, max(backup_count, 1) + 1):
                try:
                    os.chmod(path.with_name(f"{path.name}.bak{i}"), mode)
                except OSError:
                    pass

        # M-9: fsync the parent directory so the rename is durable, not just
        # atomic. POSIX guarantees os.replace's atomicity but not that the
        # directory entry has flushed to disk. On power loss between the
        # replace and the dir-block flush, the file may revert to the prior
        # name — the .bak rotation already preserves data, but durability
        # of the rename itself matters for the chain to behave correctly.
        if fsync and os.name == "posix":
            try:
                dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
            except OSError:
                return
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)


def save_with_backup(path: Path, data: Any, backup_count: int = 3) -> None:
    """Atomic JSON save with .bak rotation. Delegates to save_with_backup_text."""
    save_with_backup_text(path, json.dumps(data, indent=2) + "\n", backup_count=backup_count)
