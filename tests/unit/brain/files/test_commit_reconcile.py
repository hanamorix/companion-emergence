# tests/unit/brain/files/test_commit_reconcile.py
"""#344 — a record stranded in 'committing' (crash between the #101 claim and the
final 'committed' mark) is reconciled against the target file, never retried.

Criteria are docs/guarded-change/pending-committing-recovery/1.5-criteria.md.
The through-the-supervisor tests (C1, C2, C11, C15) live in
tests/bridge/test_supervisor_pending_sweep.py.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.files import commit as commit_mod
from brain.files import pending
from brain.files.commit import commit_write
from brain.memory.pending import PendingQueue
from brain.memory.store import MemoryStore


@pytest.fixture
def env(tmp_path, monkeypatch):
    """(persona_dir, writable out dir under a patched $HOME)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    out = tmp_path / "home" / "out"
    out.mkdir(parents=True)
    return tmp_path / "persona", out


def _now() -> datetime:
    return datetime.now(UTC)


def _strand(persona, op, target, content, monkeypatch) -> str:
    """Run the REAL commit_write with every 'committed' mark failing: the file is
    written, the claim is stamped, the record is left 'committing' — the issue's scenario."""
    rid = pending.create(persona, op=op, resolved_path=str(target.resolve()),
                         content=content, now=_now())
    real = pending.mark

    def _fail_committed(pd, r, *, status, **kw):
        return False if status == "committed" else real(pd, r, status=status, **kw)

    with monkeypatch.context() as m:
        m.setattr(pending, "mark", _fail_committed)
        store = MemoryStore(persona / "memories.db")
        try:
            res = commit_write(persona, rid, store=store)
        finally:
            store.close()
    assert res["ok"] is True
    assert pending.get(persona, rid)["status"] == "committing"
    return rid


def _claim_only(persona, op, target, content, *, age: timedelta) -> str:
    """A record claimed `age` ago whose write never happened (crash before the write)."""
    rid = pending.create(persona, op=op, resolved_path=str(target.resolve()),
                         content=content, now=_now() - age)
    assert pending.mark(persona, rid, status="committing",
                        claimed_at=(_now() - age).isoformat())
    return rid


_DELETE = object()


def _edit(persona, rid, **fields):
    p = persona / "pending_writes" / f"{rid}.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    for k, v in fields.items():
        if v is _DELETE:
            rec.pop(k, None)
        else:
            rec[k] = v
    p.write_text(json.dumps(rec), encoding="utf-8")


def _backdate(persona, rid, minutes):
    _edit(persona, rid, claimed_at=(_now() - timedelta(minutes=minutes)).isoformat())


def _audit(persona) -> list[dict]:
    p = persona / "write_audit.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def _events(persona, name) -> list[dict]:
    return [e for e in _audit(persona) if e["event"] == name]


def _wired(persona, target) -> int:
    """file_write is a GATED memory type: it lands in the consolidation queue, not memories.db."""
    # Counts the CLAIM "you let me write to X" only. An abandoned write is wired back too (#345)
    # as a different, true memory ("...it didn't land"); the invariant these tests pin is that a
    # write that did not land never produces the false claim.
    needle = str(target.resolve())
    return sum(1 for m in PendingQueue(persona).read_recent("file_write", limit=50)
               if needle in m.content and m.content.startswith("you let me write"))


def _commit_with_flaky_final_mark(persona, out, monkeypatch, *, fail_times, exc=None):
    """fail_times=-1 -> every 'committed' mark fails; N -> the first N fail, then it works."""
    t = out / "n.md"
    rid = pending.create(persona, op="create", resolved_path=str(t.resolve()),
                         content="body", now=_now())
    real, left = pending.mark, {"n": fail_times}

    def _flaky(pd, r, *, status, **kw):
        if status == "committed" and left["n"] != 0:
            left["n"] -= 1
            if exc:
                raise exc
            return False
        return real(pd, r, status=status, **kw)

    monkeypatch.setattr(pending, "mark", _flaky)
    store = MemoryStore(persona / "memories.db")
    try:
        return t, rid, commit_write(persona, rid, store=store)
    finally:
        store.close()


def _reconcile(persona, **kw):
    from brain.files.commit import reconcile_stale_commits

    return reconcile_stale_commits(persona, now=_now(), **kw)


# ---- C1 (unit half) / C9: landed write is reconciled, labelled, wired once ---


def test_reconcile_landed_append(env, monkeypatch):
    persona, out = env
    t = out / "n.md"
    t.write_text("seed\n", encoding="utf-8")
    rid = _strand(persona, "append", t, "BLOCK", monkeypatch)
    assert _wired(persona, t) == 0, "commit_write must leave the wire-back to reconcile"
    _backdate(persona, rid, 11)

    assert _reconcile(persona) == 1

    rec = pending.get(persona, rid)
    assert rec["status"] == "committed" and rec["resolved_by"] == "reconcile"
    (row,) = _events(persona, "commit_reconciled")
    assert row["id"] == rid
    assert "could not confirm this write placed it" in row["error"]  # C9 (owner ruling #1)
    assert _wired(persona, t) == 1
    assert t.read_text(encoding="utf-8") == "seed\nBLOCK"  # reconcile wrote nothing


def test_reconcile_landed_create_tolerates_later_edits_around_the_block(env, monkeypatch):
    persona, out = env
    t = out / "new.md"
    rid = _strand(persona, "create", t, "A\nB", monkeypatch)
    t.write_text("A\nB and then the user kept typing", encoding="utf-8")
    _backdate(persona, rid, 11)
    assert _reconcile(persona) == 1
    assert pending.get(persona, rid)["status"] == "committed"
    assert _wired(persona, t) == 1


# ---- C2: write did not land -> error, audited, never retried ---------------


def test_reconcile_missing_target_is_abandoned_at_ten_minutes(env):
    persona, out = env
    t = out / "never.md"  # crash before the write: parent exists, file does not
    rid = _claim_only(persona, "create", t, "body", age=timedelta(minutes=11))
    assert _reconcile(persona) == 1
    rec = pending.get(persona, rid)
    assert rec["status"] == "error" and rec["resolved_by"] == "reconcile"
    (row,) = _events(persona, "commit_abandoned")
    assert "target missing" in row["error"]
    assert not t.exists()
    assert _wired(persona, t) == 0


def test_abandoned_append_leaves_file_alone_and_is_never_retried(env):
    persona, out = env
    t = out / "n.md"
    t.write_bytes(b"seed\n")
    rid = _claim_only(persona, "append", t, "BLOCK", age=timedelta(minutes=11))
    assert _reconcile(persona) == 1
    assert t.read_bytes() == b"seed\n"
    assert pending.get(persona, rid)["status"] == "error"
    store = MemoryStore(persona / "memories.db")
    try:
        assert commit_write(persona, rid, store=store) == {
            "ok": False, "error": "not a pending write"}
    finally:
        store.close()
    assert pending.list_pending(persona, now=_now()) == []
    assert t.read_bytes() == b"seed\n"


def test_a_half_written_create_is_abandoned_not_reconciled(env):
    """The target exists but does not hold the full block (crash mid-write). Calling this
    'committed' and wiring 'you let me write to X' is exactly the say-vs-do lie of #78/#101."""
    persona, out = env
    t = out / "n.md"
    t.write_bytes(b"hel")
    rid = _claim_only(persona, "create", t, "hello", age=timedelta(minutes=11))
    assert _reconcile(persona) == 1
    assert pending.get(persona, rid)["status"] == "error"
    (row,) = _events(persona, "commit_abandoned")
    assert "partial write" in row["error"]
    assert _events(persona, "commit_reconciled") == [] and _wired(persona, t) == 0
    assert t.read_bytes() == b"hel"


@pytest.mark.parametrize("hay,block", [
    ("well done, truly\n", "done"),            # inside a longer line
    ("one\nx\ntwo\n", "one\ntwo"),             # both lines present but not adjacent
])
def test_a_short_append_inside_unrelated_text_is_not_mistaken_for_landed(env, hay, block):
    """Same guard as propose_write (#105): 'done' sits inside 'well done, truly' but was never
    appended; a block's lines scattered through the file are not the block either. 'committed'
    here would wire a memory of a write that didn't happen."""
    persona, out = env
    t = out / "n.md"
    t.write_text(hay, encoding="utf-8")
    rid = _claim_only(persona, "append", t, block, age=timedelta(minutes=11))
    assert _reconcile(persona) == 1
    assert pending.get(persona, rid)["status"] == "error"
    assert _wired(persona, t) == 0


# ---- C3: an in-flight commit is not reaped -----------------------------------


def test_fresh_claim_is_untouched_even_when_proposed_long_ago(env):
    persona, out = env
    rid = _claim_only(persona, "create", out / "n.md", "x", age=timedelta(minutes=5))
    _edit(persona, rid, proposed_at=(_now() - timedelta(days=30)).isoformat())
    assert _reconcile(persona) == 0
    assert pending.get(persona, rid)["status"] == "committing"


def test_the_age_gate_is_strictly_older_than_ten_minutes(env):
    from brain.files.commit import reconcile_stale_commits

    persona, out = env
    t0 = _now()
    rid = pending.create(persona, op="create", resolved_path=str((out / "n.md").resolve()),
                         content="x", now=t0)
    assert pending.mark(persona, rid, status="committing", claimed_at=t0.isoformat())
    assert reconcile_stale_commits(persona, now=t0 + timedelta(minutes=10)) == 0
    assert pending.get(persona, rid)["status"] == "committing"
    assert reconcile_stale_commits(persona, now=t0 + timedelta(minutes=10, seconds=1)) == 1


def test_the_unreadable_grace_is_strictly_longer_than_24_hours(env):
    from brain.files.commit import reconcile_stale_commits

    persona, out = env
    t0 = _now()
    t = out / "no-such-dir" / "n.md"  # missing parent => unreadable
    rid = pending.create(persona, op="create", resolved_path=str(t.resolve()), content="x", now=t0)
    assert pending.mark(persona, rid, status="committing", claimed_at=t0.isoformat())
    assert reconcile_stale_commits(persona, now=t0 + timedelta(hours=24)) == 0
    assert pending.get(persona, rid)["status"] == "committing"
    assert reconcile_stale_commits(persona, now=t0 + timedelta(hours=24, seconds=1)) == 1


def test_legacy_record_without_claimed_at_ages_from_proposed_at(env):
    persona, out = env
    stale = _claim_only(persona, "create", out / "a.md", "x", age=timedelta(minutes=30))
    fresh = _claim_only(persona, "create", out / "b.md", "x", age=timedelta(minutes=2))
    for rid in (stale, fresh):
        _edit(persona, rid, claimed_at=_DELETE)
    assert _reconcile(persona) == 1
    assert pending.get(persona, stale)["status"] == "error"
    assert pending.get(persona, fresh)["status"] == "committing"


# ---- C7: no lost update (inject the competing mark INSIDE the inspection) ----


@pytest.mark.parametrize("landed", [True, False])
def test_reconcile_does_not_overwrite_a_commit_that_finishes_mid_inspection(
        env, monkeypatch, landed):
    persona, out = env
    t = out / "n.md"
    if landed:
        rid = _strand(persona, "create", t, "hello", monkeypatch)
        _backdate(persona, rid, 11)
    else:
        rid = _claim_only(persona, "create", t, "hello", age=timedelta(minutes=11))
    real = commit_mod._inspect_target

    def _inspect_while_the_slow_commit_finishes(rec):
        assert pending.mark(persona, rec["id"], status="committed")
        return real(rec)

    monkeypatch.setattr(commit_mod, "_inspect_target", _inspect_while_the_slow_commit_finishes)
    _reconcile(persona)
    rec = pending.get(persona, rid)
    assert rec["status"] == "committed" and "resolved_by" not in rec
    assert _events(persona, "commit_reconciled") == []
    assert _events(persona, "commit_abandoned") == []
    assert _wired(persona, t) == 0


# ---- C13: a refused mark must not be audited or wired ------------------------


def test_refused_mark_inside_reconcile_leaves_no_audit_or_memory(env, monkeypatch):
    persona, out = env
    t = out / "n.md"
    rid = _strand(persona, "create", t, "hello", monkeypatch)
    _backdate(persona, rid, 11)
    real = pending.mark
    with monkeypatch.context() as m:
        m.setattr(pending, "mark",
                  lambda pd, r, *, status, **kw: False if kw.get("expect") == "committing"
                  else real(pd, r, status=status, **kw))
        assert _reconcile(persona) == 0
    assert pending.get(persona, rid)["status"] == "committing"
    assert _events(persona, "commit_reconciled") == [] and _wired(persona, t) == 0
    assert _reconcile(persona) == 1   # next pass, mark works: exactly once
    assert len(_events(persona, "commit_reconciled")) == 1 and _wired(persona, t) == 1


def test_unopenable_memory_store_leaves_the_record_for_the_next_pass(env, monkeypatch):
    """The store must be open BEFORE the terminal mark: once the record is 'committed' nothing
    would ever wire its memory again."""
    persona, out = env
    t = out / "n.md"
    rid = _strand(persona, "create", t, "hello", monkeypatch)
    _backdate(persona, rid, 11)

    def _boom(*a, **k):
        raise RuntimeError("memories.db locked")

    with monkeypatch.context() as m:
        m.setattr("brain.memory.store.MemoryStore", _boom)
        assert _reconcile(persona) == 0
    assert pending.get(persona, rid)["status"] == "committing"
    assert _events(persona, "commit_reconciled") == []
    assert _reconcile(persona) == 1          # store healthy again: resolved, wired exactly once
    assert _wired(persona, t) == 1


def test_a_lost_audit_row_is_logged_loudly_not_silently(env, monkeypatch, caplog):
    """#346: the mark already happened, so the record is terminal; if the audit append then
    fails nothing else would ever say what reconcile did."""
    import logging

    persona, out = env
    rid = _claim_only(persona, "create", out / "never.md", "x", age=timedelta(minutes=11))
    monkeypatch.setattr(commit_mod, "audit", lambda *a, **k: False)
    with caplog.at_level(logging.ERROR, logger="brain.files.commit"):
        assert _reconcile(persona) == 1
    assert pending.get(persona, rid)["status"] == "error"
    assert any(rid in r.getMessage() and "audit" in r.getMessage() for r in caplog.records)


def test_unopenable_memory_store_also_defers_an_abandoned_write(env, monkeypatch):
    """Same rule on the abandoned leg (#345): no terminal mark if Nell couldn't be told."""
    persona, out = env
    t = out / "never.md"
    rid = _claim_only(persona, "create", t, "x", age=timedelta(minutes=11))

    def _boom(*a, **k):
        raise RuntimeError("memories.db locked")

    with monkeypatch.context() as m:
        m.setattr("brain.memory.store.MemoryStore", _boom)
        assert _reconcile(persona) == 0
    assert pending.get(persona, rid)["status"] == "committing"
    assert _events(persona, "commit_abandoned") == []
    assert _reconcile(persona) == 1
    assert len(_queue_texts(persona, t)) == 1


# ---- C5: idempotent ----------------------------------------------------------


def test_second_pass_is_a_noop(env, monkeypatch):
    persona, out = env
    t = out / "n.md"
    rid = _strand(persona, "create", t, "hello", monkeypatch)
    _backdate(persona, rid, 11)
    assert _reconcile(persona) == 1
    opened = []
    real_store = MemoryStore
    monkeypatch.setattr("brain.memory.store.MemoryStore",
                        lambda *a, **k: opened.append(a) or real_store(*a, **k))
    assert _reconcile(persona) == 0
    assert opened == [], "a settled record must cost nothing: no store open, no target read"
    assert len(_events(persona, "commit_reconciled")) == 1
    assert _wired(persona, t) == 1


# ---- C10: byte-exact verification ---------------------------------------------


@pytest.mark.parametrize("content", ["a\nb", "a\r\nb\n", "a\rb", "é\n"])
def test_create_content_with_any_newline_style_reconciles(env, monkeypatch, content):
    persona, out = env
    rid = _strand(persona, "create", out / "n.md", content, monkeypatch)
    _backdate(persona, rid, 11)
    assert _reconcile(persona) == 1
    assert pending.get(persona, rid)["status"] == "committed"


def test_windows_style_crlf_on_disk_matches_lf_content(env, monkeypatch):
    persona, out = env
    t = out / "n.md"
    t.write_bytes(b"a\r\nb")  # what a Windows text-mode write of "a\nb" leaves
    rid = _claim_only(persona, "create", t, "a\nb", age=timedelta(minutes=11))
    monkeypatch.setattr(commit_mod.os, "linesep", "\r\n")
    assert _reconcile(persona) == 1
    monkeypatch.undo()
    assert pending.get(persona, rid)["status"] == "committed"


def test_non_utf8_append_target_is_still_verified(env, monkeypatch):
    persona, out = env
    t = out / "latin.md"
    t.write_bytes(b"\xff\xfe seed\n")
    rid = _strand(persona, "append", t, "BLOCK", monkeypatch)
    _backdate(persona, rid, 11)
    assert _reconcile(persona) == 1
    assert pending.get(persona, rid)["status"] == "committed"


# ---- C18: recovery never writes the user's file -------------------------------


def test_reconcile_only_ever_opens_the_target_for_reading(env, monkeypatch):
    import builtins

    persona, out = env
    landed, absent = out / "landed.md", out / "absent.md"
    rid1 = _strand(persona, "create", landed, "hello", monkeypatch)
    _backdate(persona, rid1, 11)
    _claim_only(persona, "create", absent, "x", age=timedelta(minutes=11))
    watched = {str(landed.resolve()), str(absent.resolve()), str(landed), str(absent)}
    real_path_open, real_open = Path.open, builtins.open

    def _guard(mode):
        assert mode.startswith("r") and "+" not in mode, f"target opened with mode {mode!r}"

    def _path_open(self, mode="r", *a, **k):
        if str(self) in watched:
            _guard(mode)
        return real_path_open(self, mode, *a, **k)

    def _open(file, mode="r", *a, **k):
        if str(file) in watched:
            _guard(mode)
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(Path, "open", _path_open)
    monkeypatch.setattr(builtins, "open", _open)
    assert _reconcile(persona) == 2


def test_windows_style_crlf_in_appended_content_is_found(env, monkeypatch):
    """Stage-6 F2: a Windows text-mode append of content containing CRLF leaves CR CR LF on disk;
    the needle must be translated the same way or a LANDED append reads as abandoned."""
    persona, out = env
    t = out / "n.md"
    t.write_bytes(b"seed\nA\r\r\nB")
    rid = _claim_only(persona, "append", t, "A\r\nB", age=timedelta(minutes=11))
    monkeypatch.setattr(commit_mod.os, "linesep", "\r\n")
    assert _reconcile(persona) == 1
    monkeypatch.undo()
    assert pending.get(persona, rid)["status"] == "committed"


# ---- C8 / C19b: unreadable is not missing ------------------------------------


def _unreadable_case(kind, out, monkeypatch):
    t = out / "n.md"
    if kind == "directory":
        t.mkdir()
    elif kind == "oversize":
        t.write_bytes(b"x" * 64)
        monkeypatch.setattr(commit_mod, "_MAX_RESULT_FILE_BYTES", 8)
    elif kind == "permission":
        t.write_bytes(b"x")
        real = Path.read_bytes

        def _deny(self):
            if self == t:
                raise PermissionError("denied")
            return real(self)

        monkeypatch.setattr(Path, "read_bytes", _deny)
    elif kind == "missing_parent":
        t = out / "no-such-dir" / "n.md"  # may be an unmounted volume: never 'absent'
    return t


@pytest.mark.parametrize("kind", ["directory", "oversize", "permission", "missing_parent"])
def test_unreadable_target_waits_then_errors_after_24h(env, monkeypatch, kind):
    persona, out = env
    t = _unreadable_case(kind, out, monkeypatch)
    rid = _claim_only(persona, "create", t, "body", age=timedelta(hours=1))
    assert _reconcile(persona) == 0
    assert pending.get(persona, rid)["status"] == "committing"
    _backdate(persona, rid, 25 * 60)
    assert _reconcile(persona) == 1
    assert pending.get(persona, rid)["status"] == "error"
    (row,) = _events(persona, "commit_abandoned")
    assert "could not read target" in row["error"]


def test_a_non_regular_target_is_never_read(env, monkeypatch):
    """A FIFO would block the supervisor thread forever in read_bytes: decide from stat() alone."""
    persona, out = env
    t = out / "n.md"
    t.mkdir()
    rid = _claim_only(persona, "create", t, "body", age=timedelta(hours=1))
    reads = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: reads.append(self) or real(self))
    assert _reconcile(persona) == 0
    assert reads == []
    assert pending.get(persona, rid)["status"] == "committing"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_a_fifo_target_never_blocks_the_supervisor_thread(env, monkeypatch):
    """The real threat behind S_ISREG: read_bytes on a FIFO with no writer blocks forever.
    The spy REFUSES instead of delegating so a regression fails fast rather than hanging."""
    persona, out = env
    t = out / "n.md"
    os.mkfifo(t)

    def _refuse(self):
        raise AssertionError(f"read_bytes called on non-regular {self}")

    monkeypatch.setattr(Path, "read_bytes", _refuse)
    rid = _claim_only(persona, "create", t, "body", age=timedelta(hours=1))
    assert _reconcile(persona) == 0
    assert pending.get(persona, rid)["status"] == "committing"


# ---- C12: one bad record never strands the rest ------------------------------


def test_one_bad_record_does_not_abort_the_pass(env, monkeypatch):
    persona, out = env
    good = _strand(persona, "create", out / "good.md", "hello", monkeypatch)
    _backdate(persona, good, 11)
    naive = _claim_only(persona, "create", out / "naive.md", "x", age=timedelta(minutes=11))
    _edit(persona, naive, claimed_at=(datetime.now() - timedelta(hours=3)).replace(tzinfo=None).isoformat())
    junk = _claim_only(persona, "create", out / "junk.md", "x", age=timedelta(minutes=11))
    _edit(persona, junk, claimed_at=12345)
    broken = _claim_only(persona, "create", out / "broken.md", "x", age=timedelta(minutes=11))
    _edit(persona, broken, op=_DELETE)

    _reconcile(persona)

    assert pending.get(persona, good)["status"] == "committed"
    assert pending.get(persona, naive)["status"] == "error"   # naive read as UTC -> stale
    assert pending.get(persona, junk)["status"] == "error"    # non-string -> stale
    assert pending.get(persona, broken)["status"] == "committing"  # contained, not fatal


# ---- #346: an expired-but-unswept card cannot be approved -----------------------


def test_a_23_hour_old_card_is_still_approvable(env):
    persona, out = env
    t = out / "n.md"
    rid = pending.create(persona, op="create", resolved_path=str(t.resolve()), content="x",
                         now=_now() - timedelta(hours=23))
    store = MemoryStore(persona / "memories.db")
    try:
        assert commit_write(persona, rid, store=store)["ok"] is True
    finally:
        store.close()
    assert t.read_text(encoding="utf-8") == "x"


def test_commit_refuses_a_card_past_its_ttl_even_if_unswept(env):
    persona, out = env
    t = out / "n.md"
    rid = pending.create(persona, op="create", resolved_path=str(t.resolve()), content="x",
                         now=_now() - timedelta(hours=25))
    store = MemoryStore(persona / "memories.db")
    try:
        res = commit_write(persona, rid, store=store)
    finally:
        store.close()
    assert res["ok"] is False and "expired" in res["error"]
    assert not t.exists()
    assert pending.get(persona, rid)["status"] == "expired"


def test_decline_cannot_overwrite_a_card_that_was_just_claimed_by_an_approval(env, monkeypatch):
    """#346 review: decline pre-checks 'pending' then marks unconditionally; racing an approval
    it could leave a written file under a record that reads 'declined'."""
    from brain.files.commit import decline_write

    persona, out = env
    rid = pending.create(persona, op="create", resolved_path=str((out / "n.md").resolve()),
                         content="x", now=_now())
    stale_view = pending.get(persona, rid)
    assert pending.mark(persona, rid, status="committing")  # the approval claimed it first
    real_get, first = pending.get, [True]

    def _stale_first_read(*a, **k):
        if first[0]:
            first[0] = False
            return stale_view
        return real_get(*a, **k)

    monkeypatch.setattr(pending, "get", _stale_first_read)
    store = MemoryStore(persona / "memories.db")
    try:
        res = decline_write(persona, rid, store=store)
    finally:
        store.close()
    assert res["ok"] is False
    assert pending.get(persona, rid)["status"] == "committing"
    assert _wired(persona, out / "n.md") == 0 and _queue_texts(persona, out / "n.md") == []


# ---- claim is a compare-and-set (ratified Q-B) ---------------------------------


def test_claim_refuses_a_record_someone_else_already_claimed(env, monkeypatch):
    """commit_write read 'pending', but another claimer got in before its claim: the CAS must
    refuse BEFORE the write, or both would append (#101)."""
    persona, out = env
    t = out / "n.md"
    t.write_text("seed\n", encoding="utf-8")
    rid = pending.create(persona, op="append", resolved_path=str(t.resolve()),
                         content="BLOCK", now=_now())
    stale_view = pending.get(persona, rid)  # what our commit_write saw
    assert pending.mark(persona, rid, status="committing")  # the rival claim lands first
    real_get, first = pending.get, [True]

    def _stale_first_read(*a, **k):  # only commit_write's own pre-check; mark's re-read is real
        if first[0]:
            first[0] = False
            return stale_view
        return real_get(*a, **k)

    monkeypatch.setattr(pending, "get", _stale_first_read)
    store = MemoryStore(persona / "memories.db")
    try:
        res = commit_write(persona, rid, store=store)
    finally:
        store.close()
    assert res["ok"] is False
    assert t.read_text(encoding="utf-8") == "seed\n", "the loser of the claim must not write"
    assert len(_events(persona, "claim_failed")) == 1


# ---- #345: an abandoned write reaches Nell (and the feed) --------------------


def _queue_texts(persona, target):
    needle = str(target.resolve())
    return [m.content for m in PendingQueue(persona).read_recent("file_write", limit=50)
            if needle in m.content]


def test_an_abandoned_write_is_wired_back_without_its_content(env):
    persona, out = env
    t = out / "never.md"
    _claim_only(persona, "create", t, "SECRET-BODY", age=timedelta(minutes=11))
    assert _reconcile(persona) == 1
    (text,) = _queue_texts(persona, t)
    assert "may not have landed" in text and "approved" in text
    assert "SECRET-BODY" not in text


def test_an_unverifiable_write_is_not_reported_as_not_landed(env):
    """Unreadable target after the 24h grace: we do NOT know it failed, so we must not say so."""
    persona, out = env
    t = out / "no-such-dir" / "n.md"
    _claim_only(persona, "create", t, "body", age=timedelta(hours=25))
    assert _reconcile(persona) == 1
    (text,) = _queue_texts(persona, t)
    assert "couldn't confirm" in text and "not have landed" not in text


def test_an_abandoned_write_surfaces_in_the_feed(env):
    """The reader leg (Organ DoD): reconcile -> consolidation drain -> the existing file_write feed."""
    from brain.bridge.feed import build_file_write_entries
    from brain.engines.consolidation import Decision, run_consolidation

    persona, out = env
    t = out / "never.md"
    _claim_only(persona, "create", t, "body", age=timedelta(minutes=11))
    assert _reconcile(persona) == 1
    store = MemoryStore(persona / "memories.db")
    try:
        run_consolidation(store, persona_dir=persona, classifier=lambda _c, _ctx: Decision("new"))
    finally:
        store.close()
    entries = build_file_write_entries(persona, limit=10)
    (entry,) = [e for e in entries if str(t.resolve()) in e.body]
    assert "may not have landed" in entry.body
    # The feed renders "<opener> <body>": "I wrote to a file — ...it didn't land" contradicts itself.
    assert "wrote" not in entry.opener


def test_a_hung_commit_that_lands_after_reconcile_corrects_the_record_and_nell(env, monkeypatch):
    """#345 review: reconcile (rightly) said "may not have landed" while the commit was hung; the
    write then lands. The record flips to committed (the file is authoritative) AND Nell must be
    told, or she holds a memory inviting a re-proposal of something that is already there."""
    persona, out = env
    t = out / "n.md"
    rid = pending.create(persona, op="create", resolved_path=str(t.resolve()), content="body",
                         now=_now())
    real_mkdir, fired = Path.mkdir, []

    def _mkdir_while_the_commit_is_hung(self, *a, **k):
        if self == t.parent and not fired:
            fired.append(1)  # we are between the claim and the write: reconcile runs now
            _backdate(persona, rid, 11)
            assert _reconcile(persona) == 1
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", _mkdir_while_the_commit_is_hung)
    store = MemoryStore(persona / "memories.db")
    try:
        assert commit_write(persona, rid, store=store)["ok"] is True
    finally:
        store.close()
    assert pending.get(persona, rid)["status"] == "committed"
    texts = _queue_texts(persona, t)
    assert any("may not have landed" in x for x in texts)
    assert any("did land after all" in x for x in texts)
    assert not any(x.startswith("you let me write") for x in texts)


# ---- C14: the final mark at the source ---------------------------------------


def test_persistent_final_mark_failure_defers_the_wireback_to_reconcile(env, monkeypatch):
    persona, out = env
    t, rid, res = _commit_with_flaky_final_mark(persona, out, monkeypatch, fail_times=-1)
    assert res["ok"] is True and t.read_text(encoding="utf-8") == "body"
    assert pending.get(persona, rid)["status"] == "committing"
    (row,) = _events(persona, "commit")
    assert "left committing for reconcile" in row["error"]
    assert _wired(persona, t) == 0


@pytest.mark.parametrize("exc", [None, OSError("disk full")])
def test_final_mark_is_retried_once(env, monkeypatch, exc):
    persona, out = env
    t, rid, res = _commit_with_flaky_final_mark(persona, out, monkeypatch, fail_times=1, exc=exc)
    assert res["ok"] is True
    assert pending.get(persona, rid)["status"] == "committed"
    assert len(_events(persona, "commit")) == 1 and _wired(persona, t) == 1


# ---- C6: the claim is stamped ---------------------------------------------


def test_claim_stamps_claimed_at(env, monkeypatch):
    persona, out = env
    rid = _strand(persona, "create", out / "n.md", "body", monkeypatch)
    rec = pending.get(persona, rid)
    assert datetime.fromisoformat(rec["claimed_at"]).tzinfo is not None
