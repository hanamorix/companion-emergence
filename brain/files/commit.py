"""brain.files.commit — perform an approved write (guard re-run) or decline.
Wires a file_write memory + feed event either way."""
from __future__ import annotations

import errno
import logging
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.files import pending
from brain.files.audit import audit
from brain.files.path_identity import stat_or_none as _stat_or_none
from brain.files.write_guard import _MAX_RESULT_FILE_BYTES, check_size, check_write_target
from brain.utils.time import lenient_utc

logger = logging.getLogger(__name__)


class _TargetChangedError(OSError):
    """The file commit opened is not the one the guard checked (#694)."""


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)  # POSIX; Windows has no symlink-following open to refuse
_BINARY = getattr(os, "O_BINARY", 0)  # Windows: a raw fd is CRT text mode otherwise, translating "\n" a second time


def _open_target(path: Path, op: str, expect: os.stat_result | None):
    """Open the approved target and check the OPENED file, not the path (#694).

    The guard checks a path, then this opens it again, so anything swapped in
    between (a hard link to ~/.ssh/authorized_keys, a symlink) would be written
    to. The last component is opened without following a symlink; a create uses
    O_EXCL, so it can only make a new file; an append must reopen the same file
    the guard saw. Either way the opened file must be a regular file with exactly
    one link. Returns a text file (utf-8, newline translation as before) and
    whether an append needs a separating newline.
    """
    try:
        if op == "create":
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _BINARY, 0o666)
        else:
            fd = os.open(path, os.O_RDWR | os.O_APPEND | _NOFOLLOW | _BINARY)
    except FileExistsError as exc:  # something appeared at a create target after it was checked
        raise _TargetChangedError(f"refused: {path} appeared after it was checked") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:  # the last component became a symlink after it was checked
            raise _TargetChangedError(f"refused: {path} became a symlink after it was checked") from exc
        raise
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
        return os.fdopen(fd, "a" if op == "append" else "w", encoding="utf-8"), needs_separator
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
    # the identity the guard just approved; the opened file must still be it (#694)
    expect = _stat_or_none(g.resolved) if g.ok and op == "append" else None
    if g.ok and op == "append" and expect is None:
        g = type(g)(ok=False, error=f"append target vanished after it was checked: {g.resolved}")
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
        g.resolved.parent.mkdir(parents=True, exist_ok=True)
        # #100: an append must not run together with the block before it.
        # write_guard already requires the target to exist for append, so the
        # only run-together case is a non-empty file with no trailing newline.
        f, needs_separator = _open_target(g.resolved, op, expect)
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
