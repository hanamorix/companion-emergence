"""brain.files.commit — perform an approved write (guard re-run) or decline.
Wires a file_write memory + feed event either way."""
from __future__ import annotations

import contextlib
import errno
import logging
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePath

from brain.files import pending, write_guard
from brain.files.audit import audit
from brain.files.write_guard import (
    _DIR_LIST,
    _DIR_OPEN,
    _MAX_RESULT_FILE_BYTES,
    _denied,
    check_size,
    check_write_target,
    dir_fd_denied,
)
from brain.utils.time import lenient_utc

logger = logging.getLogger(__name__)


class _TargetChangedError(OSError):
    """The file commit opened is not the one the guard checked (#694)."""


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)  # POSIX; Windows has no symlink-following open to refuse
_BINARY = getattr(os, "O_BINARY", 0)  # Windows: a raw fd is CRT text mode otherwise, translating "\n" a second time


def _open_final(name, op: str, dir_fd: int | None) -> int:
    """Open the target's last component without following a symlink; a create can only make a new file."""
    try:
        if op == "create":
            return os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _BINARY, 0o666, dir_fd=dir_fd)
        return os.open(name, os.O_RDWR | os.O_APPEND | _NOFOLLOW | _BINARY, dir_fd=dir_fd)  # append: ok -- the file tool's own guarded accessor
    except FileExistsError as exc:  # something appeared at a create target after it was checked
        raise _TargetChangedError(f"refused: {name} appeared after it was checked") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:  # the last component became a symlink after it was checked
            raise _TargetChangedError(f"refused: {name} became a symlink after it was checked") from exc
        raise


def _walk_without_links(folder: Path) -> tuple[int, list[str]]:
    """Open `folder` one component at a time from its root, refusing a link at every
    step (#699). The guard's path is fully resolved, so it holds no links when it is
    checked: a link found now was swapped in afterwards. Returns the deepest existing
    folder's descriptor and the names below it that don't exist yet."""
    parts = folder.parts
    dfd = os.open(parts[0], _DIR_OPEN)
    try:
        for i, name in enumerate(parts[1:], start=1):
            try:
                nfd = os.open(name, _DIR_OPEN | _NOFOLLOW, dir_fd=dfd)
            except FileNotFoundError:
                return dfd, list(parts[i:])
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):  # a link, or a file where a folder was
                    raise _TargetChangedError(f"refused: {name} on the path to {folder} changed after it was checked") from exc
                raise
            os.close(dfd)
            dfd = nfd
    except BaseException:
        os.close(dfd)
        raise
    return dfd, []


# A made folder's descriptor is held until the call ends, so a refused create can find the folder wherever a
# racer moved it (#703). Held for this many made folders at most: a create needing hundreds of new folders must not
# run out of descriptors where it used to succeed; the deeper ones are found by name only.
_HELD_MAX = 64


@dataclass(frozen=True)
class _Made:
    """A folder this call made (#703): its level in the missing list, its name, and its identity, read right
    after its ``mkdir``. Identity, not the name, decides whether a folder found later is ours."""

    level: int
    name: str
    dev: int
    ino: int


def _is_ours(st: os.stat_result, m: _Made) -> bool:
    return stat.S_ISDIR(st.st_mode) and (st.st_dev, st.st_ino) == (m.dev, m.ino)


def _rmdir_moved(own_fd: int, m: _Made) -> None:
    """Remove the made folder `own_fd` is open on from wherever it is now: its current parent through ``..``
    (never a link), the entry there with its identity. ``scandir`` needs a readable descriptor, not ``_DIR_OPEN``."""
    pfd = os.open("..", _DIR_LIST | _NOFOLLOW, dir_fd=own_fd)
    try:
        with os.scandir(pfd) as entries:
            for entry in entries:
                if _is_ours(entry.stat(follow_symlinks=False), m):
                    os.rmdir(entry.name, dir_fd=pfd)
                    return
    finally:
        os.close(pfd)


def _remove_made_posix(made: list[_Made], names: list[str], fds: dict[int, int]) -> None:
    """A refused or failed create removes the folders this call made (#703), deepest first, best effort.

    Only an entry with the recorded identity is removed, and only while empty (``rmdir`` refuses a non-empty
    folder and a symlink), so nothing this call did not make is touched. Every ``rmdir`` passes a ``dir_fd``:
    no path is followed. `fds` maps a level to its held descriptor (level -1 is the deepest folder that
    existed). First by name in the parent the folder was made in; if that entry is not ours, through the
    folder's own descriptor, wherever it was moved.
    """
    for m in reversed(made):
        found = False
        with contextlib.suppress(OSError):  # its parent can't be reached by name (moved, gone): search for it instead
            found = _rmdir_by_name(m, names, fds)
        own = fds.get(m.level)
        if not found and own is not None:
            with contextlib.suppress(OSError):
                _rmdir_moved(own, m)


def _rmdir_by_name(m: _Made, names: list[str], fds: dict[int, int]) -> bool:
    """True when the entry at the made folder's name, in the parent it was made in, is ours; it is then removed
    while empty (a non-empty one stays, and is not searched for elsewhere: it is right here)."""
    pfd, opened = _parent_descriptor(m.level, names, fds)
    try:
        try:
            st = os.stat(m.name, dir_fd=pfd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not _is_ours(st, m):
            return False
        with contextlib.suppress(OSError):
            os.rmdir(m.name, dir_fd=pfd)
        return True
    finally:
        for extra in opened:
            os.close(extra)


def _parent_descriptor(level: int, names: list[str], fds: dict[int, int]) -> tuple[int, list[int]]:
    """The descriptor of the folder at `level - 1`: held, or (past the descriptor cap) walked to by name from the
    deepest held one below it, without following a link. Also returns the descriptors opened for it, to close."""
    if level - 1 in fds:
        return fds[level - 1], []
    base = max(k for k in fds if k < level - 1)
    opened: list[int] = []
    dfd = fds[base]
    try:
        for name in names[base + 1 : level]:
            dfd = os.open(name, _DIR_OPEN | _NOFOLLOW, dir_fd=dfd)
            opened.append(dfd)
    except BaseException:
        for extra in opened:
            os.close(extra)
        raise
    return dfd, opened


def _open_pinned(path: Path, op: str, persona_dir: Path) -> int:
    """POSIX (#699): open the target inside the folder that was checked, not one looked up again.

    The folder is opened component by component, refusing a link at every step,
    so it is the approved folder and not one a swapped link points to. Missing
    folders and the file are made relative to it, again without following a link.
    After the open the folder is judged on its real ancestry (walked through ``..``
    from the descriptor), which catches it being moved into a protected folder
    meanwhile; a create it catches is removed again, and so are the folders this
    call made (#703, ``_remove_made_posix``).
    """
    dfd, missing = _walk_without_links(path.parent)
    fds: dict[int, int] = {-1: dfd}  # held descriptors by level; -1 is the deepest folder that existed
    made: list[_Made] = []
    try:
        for level, name in enumerate(missing):
            try:
                os.mkdir(name, 0o777, dir_fd=dfd)
            except FileExistsError:
                pass  # not ours: another writer made it, and it is never removed
            else:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                made.append(_Made(level, name, st.st_dev, st.st_ino))
            try:
                nfd = os.open(name, _DIR_OPEN | _NOFOLLOW, dir_fd=dfd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):  # a link (or a file) where a folder was made
                    raise _TargetChangedError(f"refused: {name} on the path to {path} changed after it was checked") from exc
                raise
            if dfd not in fds.values():
                os.close(dfd)
            dfd = nfd
            if made and made[-1].level == level and len(fds) <= _HELD_MAX:
                fds[level] = dfd
        fd = _open_final(path.name, op, dfd)
        try:
            denied = dir_fd_denied(dfd, persona_dir)
        except OSError:  # the folder became unreadable mid-commit: don't keep what can't be checked
            denied = True
        if denied:
            os.close(fd)
            if op == "create":
                with contextlib.suppress(OSError):  # best effort; the refusal stands either way
                    os.unlink(path.name, dir_fd=dfd)
            raise _TargetChangedError(f"refused: {path} moved into a protected location while it was opened")
        return fd
    except BaseException:
        _remove_made_posix(made, missing, fds)
        raise
    finally:
        if dfd not in fds.values():
            os.close(dfd)
        for held in fds.values():
            os.close(held)


def _missing_folders(folder: PurePath, exists: Callable[[PurePath], bool]) -> list[PurePath]:
    """The folders from `folder` up that don't exist yet, top-down (#703, Windows makes each one itself).

    Stops at the root, whose parent is itself: under a missing drive or share the walk ends there, and the
    first ``mkdir`` then fails as ``Path.mkdir(parents=True)`` did."""
    missing: list[PurePath] = []
    p = folder
    while not exists(p) and p != p.parent:
        missing.append(p)
        p = p.parent
    return missing[::-1]


def _handle_path(fd: int) -> str | None:
    """Windows: where an open file really is (GetFinalPathNameByHandleW), or None if that can't be read."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    get = ctypes.windll.kernel32.GetFinalPathNameByHandleW
    get.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get.restype = wintypes.DWORD
    buf = ctypes.create_unicode_buffer(32768)
    n = get(wintypes.HANDLE(msvcrt.get_osfhandle(fd)), buf, len(buf), 0)
    if not n or n >= len(buf):
        return None
    p = buf.value
    if p.startswith("\\\\?\\UNC\\"):
        return "\\\\" + p[8:]
    return p[4:] if p.startswith("\\\\?\\") else p


@dataclass(frozen=True)
class _MadeAt:
    """A folder this call made on Windows (#703): the path it was made at, how many levels above the file it
    is (1 = the file's folder), and its identity (the volume and file index), read right after its ``mkdir``."""

    path: Path
    depth: int
    dev: int
    ino: int


def _remove_made_windows(made: list[_MadeAt], real: str | None) -> None:
    """A refused or failed create removes the folders this call made (#703), deepest first, best effort.

    A folder is looked for at the path it was made at and, when the handle's real location is known, at the
    matching ancestor of that location (where the folders really are after a junction swap). Only the one with
    the recorded identity is removed, and only while empty. Identity is the junction guard: a junction stats as a
    directory but has its own file index. A file index of 0 is no identity: that folder is not removed."""
    for m in reversed(made):
        if m.ino == 0:
            continue
        candidates = [m.path]
        if real:
            ancestors = Path(real).parents
            if m.depth <= len(ancestors) - 1:  # a real location at a drive or share root is shallower: no candidate
                candidates.append(ancestors[m.depth - 1])
        for candidate in candidates:
            with contextlib.suppress(OSError):
                st = os.stat(candidate, follow_symlinks=False)
                if (st.st_dev, st.st_ino) == (m.dev, m.ino):
                    os.rmdir(candidate)
                    break


def _open_by_path(path: Path, op: str, persona_dir: Path) -> int:
    """Windows (#699): no dir_fd or O_NOFOLLOW there, so open by path, then require the
    opened handle's real location (a junction swapped in is resolved by the handle) to
    be the approved path, and outside the protected locations. A create that fails it
    is removed again. Missing folders are made by path first, one at a time, and a create
    that does not land removes the ones this call made (#703, ``_remove_made_windows``)."""
    made: list[_MadeAt] = []
    real: str | None = None
    try:
        for folder in _missing_folders(path.parent, os.path.exists):
            try:
                os.mkdir(folder)
            except FileExistsError:
                continue  # not ours: another writer made it, and it is never removed
            st = os.stat(folder, follow_symlinks=False)
            made.append(_MadeAt(Path(folder), len(path.relative_to(folder).parts), st.st_dev, st.st_ino))
        fd = _open_final(str(path), op, None)
        try:
            real = _handle_path(fd)
        except OSError:
            real = None
        if real is None or os.path.normcase(real) != os.path.normcase(str(path)) or _denied(Path(real), persona_dir):
            os.close(fd)
            if op == "create":
                with contextlib.suppress(OSError):  # best effort; the refusal stands either way
                    os.unlink(real or str(path))
            raise _TargetChangedError(f"refused: {path} leads to {real or 'an unreadable location'} now")
        return fd
    except BaseException:
        _remove_made_windows(made, real)
        raise


def _open_target(path: Path, op: str, expect: os.stat_result | None, persona_dir: Path):
    """Open the approved target and check the OPENED file, not the path (#694, #699).

    The guard checks a path, then commit opens it, so anything swapped in between
    (a hard link to ~/.ssh/authorized_keys, a symlink, a folder on the path
    replaced by a link) would be written to. The file is opened inside the folder
    that was checked (POSIX) or its handle's real location is judged (Windows);
    the last component never follows a symlink; a create can only make a new file;
    an append must reopen the file the guard saw. The opened file must be a
    regular file with exactly one link. Returns a text file (utf-8, newline
    translation as before) and whether an append needs a separating newline.
    """
    fd = _open_by_path(path, op, persona_dir) if write_guard._is_windows() else _open_pinned(path, op, persona_dir)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise _TargetChangedError(f"refused: {path} changed after it was checked (not a single-link file)")
        if expect is not None and not os.path.samestat(st, expect):
            raise _TargetChangedError(f"refused: {path} was replaced after it was checked")
        needs_separator = False
        if op == "append" and st.st_size > 0:  # #100: don't run the new block into the last line
            os.lseek(fd, -1, os.SEEK_END)
            needs_separator = os.read(fd, 1) != b"\n"
        return os.fdopen(fd, "a" if op == "append" else "w", encoding="utf-8"), needs_separator  # append: ok -- the file tool's own guarded accessor
    except BaseException:
        os.close(fd)
        raise


def _wire_memory(store, *, path: str, outcome: str) -> None:
    from brain.memory.store import Memory

    # No file content is ever stored, only the path and what happened to the write.
    if outcome == "committed":
        content = f"you let me write to {path}"
    elif outcome == "abandoned":
        # approved, but a crash meant the block never reached the file (found by reconcile, #345)
        content = f"you approved my write to {path}, but I can't find it there. It may not have landed."
    elif outcome == "late":
        content = f"the write to {path} that I thought hadn't landed did land after all"
    elif outcome == "unverified":
        content = f"you approved my write to {path}, but I couldn't confirm whether it landed"
    else:
        content = f"you declined my write to {path}"
    try:
        mem = Memory.create_new(
            content=content,
            memory_type="file_write",
            domain="interior",
            tags=["file_write", outcome],
        )
        # Automatic generated write → route through the consolidation gate (gated by memory_type).
        from brain.memory.pending import route_write

        route_write(store, mem, source="file_write")
    except Exception:
        logger.exception("file_write wire-back memory failed")


def _mark_final(persona_dir: Path, rid: str) -> bool:
    """Record 'committed' after the write has landed. True if it took.

    One retry, on a False return or an OSError: the write is already on disk, so a
    transient mark failure here is what strands a record in 'committing' (#344).
    """
    for _attempt in range(2):
        try:
            if pending.mark(persona_dir, rid, status="committed"):
                return True
        except OSError:
            logger.warning("final 'committed' mark raised for %s", rid, exc_info=True)
    return False


def commit_write(persona_dir: Path, rid: str, *, store) -> dict:
    rec = pending.get(persona_dir, rid)
    if rec is None or rec.get("status") != "pending":
        return {"ok": False, "error": "not a pending write"}
    if pending.is_expired(rec, datetime.now(UTC)):
        # The 24h sweep may not have run yet; an approval must not outlive the card's TTL (#346).
        pending.mark(persona_dir, rid, status="expired", expect="pending")
        return {"ok": False, "error": "this write expired (older than 24h) — ask again"}
    op, content = rec["op"], rec["content"]
    # TOCTOU: re-run the guard on the resolved path RIGHT NOW.
    g = check_write_target(rec["resolved_path"], op=op, persona_dir=persona_dir)
    if g.ok and g.resolved != Path(rec["resolved_path"]):
        # the card's path resolves somewhere else now (a folder on it was swapped for a link while the card
        # waited): write to the place the user approved or nowhere (#702)
        g = type(g)(ok=False, error=f"refused: {rec['resolved_path']} now leads to {g.resolved}")
    # the file the guard judged (one link); the opened file must be that one. Not a second stat:
    # a link swapped in between would become the "expected" file (#694)
    expect = g.stat if op == "append" else None
    if not g.ok or not check_size(content, op=op, resolved=g.resolved).ok:
        err = g.error or "size check failed"
        pending.mark(persona_dir, rid, status="refused")
        audit(
            persona_dir,
            event="commit_refused",
            id=rid,
            op=op,
            path=rec["resolved_path"],
            error=err,
        )
        return {"ok": False, "error": err}
    # #101: claim the record BEFORE writing. The write used to land first and
    # the status was set after — but pending.mark is silently fallible, so a
    # failed mark left the record 'pending' with the content already on disk,
    # and the status guard above stopped blocking a retry. Claiming first means
    # a retry can only re-enter if nothing was written.
    # #344: claimed_at is the age clock for recovering a record stranded here (proposed_at
    # is the wrong clock: a card approved at hour 23 is legitimately 'committing' at hour 24).
    # expect="pending" makes the claim a compare-and-set against a concurrent claimer/sweep.
    if not pending.mark(persona_dir, rid, status="committing", expect="pending",
                        claimed_at=datetime.now(UTC).isoformat()):
        audit(persona_dir, event="claim_failed", id=rid, op=op, path=rec["resolved_path"],
              error="could not record the committing status")
        return {"ok": False, "error": "could not claim the write for commit"}

    try:
        # #100: an append must not run together with the block before it.
        # write_guard already requires the target to exist for append, so the
        # only run-together case is a non-empty file with no trailing newline.
        # Missing parent folders are made inside _open_target, in the checked folder (#699).
        f, needs_separator = _open_target(g.resolved, op, expect, persona_dir)
        with f:
            if needs_separator:
                f.write("\n")
            f.write(content)
    except _TargetChangedError as exc:
        pending.mark(persona_dir, rid, status="refused")
        audit(persona_dir, event="commit_refused", id=rid, op=op, path=str(g.resolved), error=str(exc))
        return {"ok": False, "error": str(exc)}
    except OSError as exc:
        pending.mark(persona_dir, rid, status="error")
        audit(persona_dir, event="error", id=rid, op=op, path=str(g.resolved), error=str(exc))
        return {"ok": False, "error": str(exc)}
    # A hung commit can outlive reconcile's verdict: reconcile (rightly) saw no block and recorded
    # "may not have landed", then our write landed. The file is authoritative, so the record flips
    # to committed — but Nell holds a memory that invites re-proposing it, so correct that too.
    prior = pending.get(persona_dir, rid) or {}
    reconciled_lost = prior.get("status") == "error" and prior.get("resolved_by") == "reconcile"
    marked = _mark_final(persona_dir, rid)
    audit(
        persona_dir,
        event="commit",
        id=rid,
        op=op,
        path=str(g.resolved),
        content_sha=rec["content_sha"],
        outcome="committed",
        error=None if marked else "final status mark failed; record left committing for reconcile",
    )
    if marked:
        # Otherwise reconcile_stale_commits wires the memory — exactly once.
        _wire_memory(store, path=str(g.resolved),
                     outcome="late" if reconciled_lost else "committed")
    return {"ok": True, "path": str(g.resolved)}


def _inspect_target(rec: dict) -> str:
    """READ-ONLY look at the target: 'landed' | 'absent' | 'unreadable'.

    'absent' means the write never happened (FileNotFoundError with the parent present).
    A missing PARENT is 'unreadable', not 'absent': it may be an unmounted volume, and a
    landed write must not be recorded as lost. Never opens the target for writing.
    """
    from brain.tools.impls.propose_write import _block_present

    target = Path(rec["resolved_path"])
    try:
        st = target.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > _MAX_RESULT_FILE_BYTES:
            return "unreadable"  # a directory/FIFO/huge file: don't read it on a supervisor thread
        data = target.read_bytes()
    except FileNotFoundError:
        return "absent" if target.parent.is_dir() else "unreadable"
    except OSError:
        return "unreadable"
    # commit_write wrote `content` in text mode, so on Windows "\n" became os.linesep.
    # Mirror that exactly (create AND append); a text-mode read-back would collapse CR / CRLF.
    written = rec["content"].replace("\n", os.linesep)
    if rec["op"] == "create":
        return "landed" if written.encode("utf-8") in data else "absent"
    text = data.decode("utf-8", errors="replace")
    return "landed" if _block_present(text, written) else "absent"


_COMMIT_STALE = timedelta(minutes=10)
_COMMIT_UNVERIFIABLE = timedelta(hours=pending._TTL_HOURS)  # one clock for the whole card


def _claim_age(rec: dict, now: datetime) -> timedelta | None:
    """Age of the claim; None when no usable timestamp exists (treated as stale).

    Records claimed before #344 have no claimed_at: fall back to proposed_at (owner-ratified).
    A naive timestamp is read as UTC.
    """
    ts = lenient_utc(rec.get("claimed_at") or rec.get("proposed_at"))
    return None if ts is None else now - ts


def _reconcile_one(persona_dir: Path, rec: dict, *, now: datetime, stale_after: timedelta,
                   open_store) -> bool:
    """Resolve one stale 'committing' record. True if it moved to a terminal status."""
    # Read every field we will audit BEFORE touching state: a malformed record must raise
    # here (contained by the caller), never end up terminal without its audit row.
    rid, op, path = rec["id"], rec["op"], rec["resolved_path"]
    age = _claim_age(rec, now)
    if age is not None and age <= stale_after:
        return False  # a real write takes milliseconds; younger than this may be in flight
    verdict = _inspect_target(rec)
    if verdict == "unreadable":
        if age is not None and age <= _COMMIT_UNVERIFIABLE:
            logger.warning("pending write %s: cannot read %s yet; will retry", rid, path)
            return False
        status, event, why = "error", "commit_abandoned", "could not read target to verify"
        wire = "unverified"
    elif verdict == "landed":
        status, event, wire = "committed", "commit_reconciled", "committed"
        why = "block found in target; could not confirm this write placed it"
    else:
        status, event, wire = "error", "commit_abandoned", "abandoned"
        why = "target missing or does not contain the full block (a partial write may exist)"
    # Open the store BEFORE the terminal mark: once the record is 'committed' nothing would ever
    # wire its memory again, so a failure here must raise while the record is still retryable.
    store = open_store()
    # expect="committing": if the slow in-flight commit finished while we inspected, this refuses.
    if not pending.mark(persona_dir, rid, status=status, expect="committing",
                        resolved_by="reconcile"):
        return False
    if not audit(persona_dir, event=event, id=rid, op=op, path=path,
                 content_sha=rec.get("content_sha", ""), outcome=status, error=why):
        # The mark already happened, so the record is terminal; nothing else would say what we did.
        logger.error("pending write %s resolved %s by reconcile but its %s audit row was lost",
                     rid, status, event)
    if status == "error":
        logger.warning("pending write %s abandoned (%s): %s", rid, why, path)
    _wire_memory(store, path=path, outcome=wire)
    return True


def reconcile_stale_commits(persona_dir: Path, *, now: datetime,
                            stale_after: timedelta = _COMMIT_STALE) -> int:
    """Move records stranded in 'committing' to a terminal status (#344).

    commit_write claims a record ('committing') BEFORE writing (#101). A crash between
    the claim and the final 'committed' mark strands it. We never re-write (that would
    reopen #101's double-append) — we look at the target and record what is there.
    Returns the number of records moved to a terminal status.
    """
    from brain.memory.store import MemoryStore

    holder: list = []

    def _open_store():
        if not holder:  # lazy: most passes find nothing to wire
            holder.append(MemoryStore(persona_dir / "memories.db", integrity_check=False))
        return holder[0]

    n = 0
    try:
        for rec in pending.list_by_status(persona_dir, "committing"):
            try:  # one bad record must not strand the rest
                n += _reconcile_one(persona_dir, rec, now=now, stale_after=stale_after,
                                    open_store=_open_store)
            except Exception:
                logger.warning("reconcile of pending write %s failed", rec.get("id"),
                               exc_info=True)
    finally:
        if holder:
            holder[0].close()
    return n


def decline_write(persona_dir: Path, rid: str, *, store) -> dict:
    rec = pending.get(persona_dir, rid)
    if rec is None or rec.get("status") != "pending":
        return {"ok": False, "error": "not a pending write"}
    if not pending.mark(persona_dir, rid, status="declined", expect="pending"):
        return {"ok": False, "error": "not a pending write"}  # an approval claimed it first (#346)
    audit(
        persona_dir,
        event="decline",
        id=rid,
        op=rec["op"],
        path=rec["resolved_path"],
        outcome="declined",
    )
    _wire_memory(store, path=rec["resolved_path"], outcome="declined")
    return {"ok": True}
