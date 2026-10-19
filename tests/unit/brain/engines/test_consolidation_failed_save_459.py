"""#459 (+ #473's consolidation part): one failed item must not lose the rest of an already-claimed batch.

`_run_locked` drained the queue (read, then truncate) and then worked through the batch with nothing guarding the
per-item work, so any exception ended the run with the unprocessed rest neither promoted nor back in the queue; a kill
did the same. Now the batch is CLAIMED (kept in a processing file until each item is settled) and each item's fate is
decided by what its failure depends on (`failure_scope`):

- item scope (will fail the same way again): quarantined in `consolidation_quarantine.jsonl` with the error, logged once;
- environment scope (store or filesystem): requeued, and the pass ENDS (the rest stay in the processing file);
- already saved (a post-commit step failed): nothing to do;
- a deferral (`ThrottleDeferred`, e.g. an expired login, #473) is "not now": the item and the rest stay for the next run.

Synthetic names only. The persona is a tmp dir with a real MemoryStore and PendingQueue.
"""

from __future__ import annotations

import errno
import logging
import sqlite3
import threading

import pytest

import brain.engines.consolidation as cons
from brain.bridge.provider_auth import ProviderAuthDeferred
from brain.engines.consolidation import (
    _ARCHIVE_FILENAME,
    Decision,
    Reappraisal,
    run_consolidation,
)
from brain.health.json_io import read_jsonl, write_jsonl
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore

_PROCESSING = "pending_candidates.processing.jsonl"
_SETTLED = "pending_candidates.settled.jsonl"
_QUARANTINE = "consolidation_quarantine.jsonl"

A, B, C = "alpha memory", "bravo memory", "charlie memory"


@pytest.fixture
def persona(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    heb = HebbianMatrix(tmp_path / "hebbian.db")
    yield tmp_path, store, heb, PendingQueue(tmp_path)
    store.close()
    heb.close()


def _mem(content: str, mtype: str = "dream") -> Memory:
    return Memory.create_new(content=content, memory_type=mtype, domain="us", emotions={})


class Clf:
    """A classifier that says `new` and records the contents it was asked about."""

    def __init__(self, behaviour=None):
        self.calls: list[str] = []
        self._behaviour = behaviour

    def __call__(self, cand, _context):
        self.calls.append(cand.content)
        if self._behaviour is not None:
            return self._behaviour(cand)
        return Decision("new")


def _run(tmp, store, heb, clf, **kw):
    return run_consolidation(store, persona_dir=tmp, classifier=clf, hebbian=heb, **kw)


def _enqueue(queue, *contents) -> dict[str, Memory]:
    out = {}
    for c in contents:
        m = _mem(c)
        queue.enqueue(m, source="dream")
        out[c] = m
    return out


def _contents(store) -> list[str]:
    return sorted(r[0] for r in store._conn.execute("SELECT content FROM memories"))


def _queue_rows(queue) -> list[dict]:
    return read_jsonl(queue.path) if queue.path.exists() else []


def _quarantine(tmp) -> list[dict]:
    p = tmp / _QUARANTINE
    return read_jsonl(p) if p.exists() else []


def _unsettled(tmp) -> list[dict]:
    p, s = tmp / _PROCESSING, tmp / _SETTLED
    if not p.exists():
        return []
    done = {r["id"] for r in read_jsonl(s)} if s.exists() else set()
    return [e for e in read_jsonl(p) if e["_claim_id"] not in done]


def _label(entry: dict) -> str:
    return entry.get("content") or entry.get("memory_id")


def _fail(monkeypatch, obj, name, match, exc):
    """Make `obj.name(first_arg, ...)` raise `exc` when `match(first_arg)`, else delegate."""
    real = getattr(obj, name)

    def wrapper(first, *a, **k):
        if match(first):
            raise exc
        return real(first, *a, **k)

    monkeypatch.setattr(obj, name, wrapper)


def _fail_create(monkeypatch, store, content, exc):
    _fail(monkeypatch, store, "create", lambda m: m.content == content, exc)


def _errors_naming(caplog, memory: Memory) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR and memory.id in r.getMessage()]


_ENVIRONMENT_FAULTS = [
    sqlite3.OperationalError("database is locked"),
    OSError(errno.ENOSPC, "No space left on device"),
    OSError(errno.EIO, "Input/output error"),
    PermissionError("denied"),
    TimeoutError("slow"),
    sqlite3.DatabaseError("database disk image is malformed"),
    sqlite3.OperationalError("no such table: memories"),
]
_ITEM_FAULTS = [
    sqlite3.ProgrammingError("Error binding parameter 1"),
    sqlite3.DataError("string or blob too big"),
    TypeError("Object of type object is not JSON serializable"),
    ValueError("bad value"),
    RuntimeError("boom"),
]


# ---------------------------------------------------------------------------
# environment failure: requeue, the pass ends, nothing is lost
# ---------------------------------------------------------------------------


def test_an_environment_failure_requeues_the_item_ends_the_pass_and_loses_nothing(persona, monkeypatch, caplog):
    # C459-a
    tmp, store, heb, queue = persona
    mems = _enqueue(queue, A, B, C)
    _fail_create(monkeypatch, store, B, sqlite3.OperationalError("database is locked"))
    clf = Clf()

    with caplog.at_level(logging.INFO):
        result = _run(tmp, store, heb, clf)

    assert _contents(store) == [A]
    rows = _queue_rows(queue)
    assert [r["content"] for r in rows] == [B] and rows[0]["_source"] == "dream"
    assert [_label(e) for e in _unsettled(tmp)] == [C]  # C waits, not lost
    assert _quarantine(tmp) == []
    assert result.failed == 1 and result.quarantined == 0 and result.promoted == 1
    assert any(r.levelno == logging.WARNING and mems[B].id in r.getMessage() and "OperationalError" in r.getMessage()
               for r in caplog.records)

    monkeypatch.undo()
    _run(tmp, store, heb, clf)

    assert _contents(store) == sorted([A, B, C])
    assert _queue_rows(queue) == [] and not (tmp / _PROCESSING).exists() and not (tmp / _SETTLED).exists()
    assert sorted(clf.calls) == sorted([A, B, B, C]) and clf.calls.count(A) == 1  # exactly 4, none for A twice


@pytest.mark.parametrize("exc", _ENVIRONMENT_FAULTS, ids=lambda e: f"{type(e).__name__}:{str(e)[:18]}")
def test_every_environment_fault_type_is_requeued_not_quarantined(persona, monkeypatch, exc):
    # C459-b
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)
    _fail_create(monkeypatch, store, B, exc)

    _run(tmp, store, heb, Clf())

    assert [r["content"] for r in _queue_rows(queue)] == [B] and _quarantine(tmp) == []
    monkeypatch.undo()
    _run(tmp, store, heb, Clf())
    assert _contents(store) == sorted([A, B, C])


# ---------------------------------------------------------------------------
# item failure: quarantine, the pass continues
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc", _ITEM_FAULTS, ids=lambda e: type(e).__name__)
def test_an_item_failure_is_quarantined_with_evidence_and_the_rest_are_promoted(persona, monkeypatch, caplog, exc):
    # C459-c (injected faults)
    tmp, store, heb, queue = persona
    mems = _enqueue(queue, A, B, C)
    _fail_create(monkeypatch, store, B, exc)

    with caplog.at_level(logging.INFO):
        result = _run(tmp, store, heb, Clf())

    assert _contents(store) == sorted([A, C])
    assert _queue_rows(queue) == [] and _unsettled(tmp) == []
    recs = _quarantine(tmp)
    assert len(recs) == 1
    assert recs[0]["entry"]["content"] == B and recs[0]["entry"]["_source"] == "dream"
    assert recs[0]["error"].startswith(type(exc).__name__) and recs[0]["stage"] == "pass2"
    assert len(_errors_naming(caplog, mems[B])) == 1
    assert "bravo" not in caplog.text  # no memory content in any log line
    assert result.quarantined == 1 and result.failed == 1


def test_a_real_duplicate_id_with_different_content_is_quarantined_with_both_contents(persona, caplog):
    # C459-c (REAL IntegrityError, no injected fault): a row with B's id and other content already exists.
    tmp, store, heb, queue = persona
    existing = _mem("existing memory")
    store.create(existing)
    b = _mem(B)
    b.id = existing.id
    queue.enqueue(b, source="dream")
    _enqueue(queue, C)

    with caplog.at_level(logging.INFO):
        _run(tmp, store, heb, Clf())

    recs = _quarantine(tmp)
    assert len(recs) == 1 and recs[0]["entry"]["content"] == B
    assert recs[0]["existing_content"] == "existing memory"
    assert "IntegrityError" in recs[0]["error"]
    assert _contents(store) == sorted(["existing memory", C])


@pytest.mark.parametrize("variant", ["equal", "faded", "merged"])
def test_a_candidate_whose_row_already_exists_in_any_of_three_shapes_is_saved_not_quarantined(
    persona, monkeypatch, caplog, variant
):
    # C459-d: the failing `create` is the REAL duplicate-id IntegrityError; the probe says "saved" when the row's content
    # equals the candidate's, equals the row's content_snapshot (faded), or contains it (merged).
    tmp, store, heb, queue = persona
    b = _mem(B)
    row_content = {"equal": B, "faded": "a short summary", "merged": f"{B}\nplus a later fold"}[variant]
    pre = _mem(row_content)
    pre.id = b.id
    store.create(pre)
    if variant == "faded":
        store._conn.execute("UPDATE memories SET content_snapshot = ? WHERE id = ?", (B, b.id))
        store._conn.commit()
    queue.enqueue(b, source="dream")
    monkeypatch.setattr(cons, "_has_exact_existing", lambda *a, **k: False)  # let the "equal" variant reach `create`

    with caplog.at_level(logging.INFO):
        result = _run(tmp, store, heb, Clf())

    assert _quarantine(tmp) == [] and _queue_rows(queue) == [] and _unsettled(tmp) == []
    assert store.count(active_only=False) == 1 and result.quarantined == 0
    assert any(b.id in r.getMessage() for r in caplog.records)  # one log line names it


def test_a_quarantined_item_is_final_and_logged_once(persona, monkeypatch, caplog):
    # C459-e
    tmp, store, heb, queue = persona
    mems = _enqueue(queue, A, B, C)
    _fail_create(monkeypatch, store, B, ValueError("bad value"))
    clf = Clf()
    with caplog.at_level(logging.INFO):
        _run(tmp, store, heb, clf)
        _run(tmp, store, heb, clf)

    assert clf.calls.count(B) == 1
    assert len(_quarantine(tmp)) == 1
    assert len(_errors_naming(caplog, mems[B])) == 1


# ---------------------------------------------------------------------------
# store health: a closed or wrong-thread store hits EVERY item
# ---------------------------------------------------------------------------


def test_a_closed_store_is_environment_for_every_item_and_nothing_is_quarantined(persona):
    # C459-d2 (REAL fault)
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B)
    store.close()
    clf = Clf()

    _run(tmp, store, heb, clf)

    assert _quarantine(tmp) == []
    fresh = MemoryStore(tmp / "memories.db")
    try:
        _run(tmp, fresh, heb, clf)
        assert _contents(fresh) == sorted([A, B])
    finally:
        fresh.close()


def test_a_wrong_thread_store_is_environment_for_every_item_and_nothing_is_quarantined(persona):
    # C459-d2 (REAL fault): the gate runs on a thread that does not own the store's connection.
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B)
    clf = Clf()
    t = threading.Thread(target=lambda: _run(tmp, store, heb, clf))
    t.start()
    t.join()

    assert _quarantine(tmp) == []
    _run(tmp, store, heb, clf)
    assert _contents(store) == sorted([A, B])


# ---------------------------------------------------------------------------
# Pass 1 probe and the merge branch
# ---------------------------------------------------------------------------


def test_a_pass1_probe_environment_failure_requeues_and_nothing_is_lost(persona, monkeypatch):
    # C459-f
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)
    real = cons._has_exact_existing

    def probe(store_, content, norm):
        if content == B:
            raise sqlite3.OperationalError("database is locked")
        return real(store_, content, norm)

    monkeypatch.setattr(cons, "_has_exact_existing", probe)
    _run(tmp, store, heb, Clf())
    assert [r["content"] for r in _queue_rows(queue)] == [B]
    assert _quarantine(tmp) == []

    monkeypatch.undo()
    _run(tmp, store, heb, Clf())
    assert _contents(store) == sorted([A, B, C])


def test_a_pass1_probe_item_failure_is_quarantined_and_the_others_continue(persona, monkeypatch):
    # C459-f
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)
    real = cons._has_exact_existing

    def probe(store_, content, norm):
        if content == B:
            raise sqlite3.IntegrityError("constraint")
        return real(store_, content, norm)

    monkeypatch.setattr(cons, "_has_exact_existing", probe)
    _run(tmp, store, heb, Clf())

    assert _contents(store) == sorted([A, C]) and len(_quarantine(tmp)) == 1
    assert _quarantine(tmp)[0]["stage"] == "pass1"


@pytest.mark.parametrize("exc, expect", [(sqlite3.OperationalError("database is locked"), "requeue"),
                                         (sqlite3.IntegrityError("constraint"), "quarantine")])
def test_a_merge_branch_update_failure_follows_the_same_rules(persona, monkeypatch, exc, expect):
    # C459-g
    tmp, store, heb, queue = persona
    target = _mem("target memory")
    store.create(target)
    _enqueue(queue, A, B, C)

    def behaviour(cand):
        return Decision("merge", target_id=target.id) if cand.content == B else Decision("new")

    _fail(monkeypatch, store, "update", lambda mid: mid == target.id, exc)
    clf = Clf(behaviour)
    _run(tmp, store, heb, clf)

    if expect == "requeue":
        assert [r["content"] for r in _queue_rows(queue)] == [B] and _quarantine(tmp) == []
        monkeypatch.undo()
        _run(tmp, store, heb, clf)
        # documented residue: the re-run merge archives the pre-image a second time
        assert len(read_jsonl(tmp / _ARCHIVE_FILENAME)) == 2
        assert B in store.get(target.id, bump=False).content
    else:
        assert len(_quarantine(tmp)) == 1 and _contents(store) == sorted([A, C, "target memory"])


# ---------------------------------------------------------------------------
# reappraise items
# ---------------------------------------------------------------------------


def _reappraiser(m):
    return Reappraisal(importance=9.0)


def test_a_reappraise_item_environment_failure_is_requeued_as_a_reappraise_item(persona, monkeypatch):
    # C459-h
    tmp, store, heb, queue = persona
    r1, r2 = _mem("old one"), _mem("old two")
    store.create(r1)
    store.create(r2)
    queue.enqueue_reappraisal(r2.id, source="recall")  # healthy first
    queue.enqueue_reappraisal(r1.id, source="recall")
    _enqueue(queue, A, B)
    _fail(monkeypatch, store, "get", lambda mid: mid == r1.id, sqlite3.OperationalError("database is locked"))

    result = _run(tmp, store, heb, Clf(), reappraiser=_reappraiser)

    assert result.reappraised == 1 and store.get(r2.id, bump=False).importance == 9.0
    rows = _queue_rows(queue)
    assert len(rows) == 1 and rows[0]["_route"] == "reappraise_importance" and rows[0]["memory_id"] == r1.id
    assert _contents(store) == sorted(["old one", "old two"])  # A and B wait, unlost
    assert sorted(_label(e) for e in _unsettled(tmp)) == sorted([A, B])

    monkeypatch.undo()
    _run(tmp, store, heb, Clf(), reappraiser=_reappraiser)
    assert _contents(store) == sorted(["old one", "old two", A, B])
    assert store.get(r1.id, bump=False).importance == 9.0


def test_a_reappraise_item_item_failure_is_quarantined_and_the_rest_continue(persona, monkeypatch):
    # C459-h
    tmp, store, heb, queue = persona
    r1, r2 = _mem("old one"), _mem("old two")
    store.create(r1)
    store.create(r2)
    queue.enqueue_reappraisal(r1.id, source="recall")
    queue.enqueue_reappraisal(r2.id, source="recall")
    _enqueue(queue, A, B)
    _fail(monkeypatch, store, "get", lambda mid: mid == r1.id, sqlite3.IntegrityError("constraint"))

    result = _run(tmp, store, heb, Clf(), reappraiser=_reappraiser)

    recs = _quarantine(tmp)
    assert len(recs) == 1 and recs[0]["entry"]["memory_id"] == r1.id and recs[0]["stage"] == "reappraise"
    assert result.reappraised == 1 and _queue_rows(queue) == []
    assert A in _contents(store) and B in _contents(store)


# ---------------------------------------------------------------------------
# post-commit failure, keeping failure, settle is total
# ---------------------------------------------------------------------------


def test_a_post_commit_failure_is_neither_requeued_nor_quarantined(persona, monkeypatch, caplog):
    # C459-i
    tmp, store, heb, queue = persona
    target = _mem("target memory")
    store.create(target)
    mems = _enqueue(queue, A, B, C)

    def behaviour(cand):
        return Decision("correction", target_id=target.id) if cand.content == B else Decision("new")

    monkeypatch.setattr(heb, "set_edge_weight", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("edge")))
    clf = Clf(behaviour)
    with caplog.at_level(logging.INFO):
        _run(tmp, store, heb, clf)

    assert _contents(store) == sorted([A, B, C, "target memory"])
    assert _queue_rows(queue) == [] and _quarantine(tmp) == [] and _unsettled(tmp) == []
    named = [r.getMessage() for r in caplog.records if mems[B].id in r.getMessage()]
    assert named and any(target.id in m for m in named)  # the lost link can be rebuilt from the log
    monkeypatch.undo()
    _run(tmp, store, heb, clf)
    assert _contents(store).count(B) == 1


def test_a_failed_requeue_leaves_the_item_unsettled_and_the_next_run_handles_it(persona, monkeypatch, caplog):
    # C459-j (environment failure, the requeue itself raises)
    tmp, store, heb, queue = persona
    mems = _enqueue(queue, A, B, C)
    _fail_create(monkeypatch, store, B, sqlite3.OperationalError("database is locked"))
    real_enqueue = PendingQueue.enqueue

    def enqueue(self, mem, *, source):
        if mem.content == B:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_enqueue(self, mem, source=source)

    monkeypatch.setattr(PendingQueue, "enqueue", enqueue)
    with caplog.at_level(logging.INFO):
        _run(tmp, store, heb, Clf())

    assert B in [_label(e) for e in _unsettled(tmp)]
    assert len(_errors_naming(caplog, mems[B])) == 1 and "bravo" not in caplog.text
    monkeypatch.undo()
    _run(tmp, store, heb, Clf())
    assert _contents(store) == sorted([A, B, C])


def test_a_failed_quarantine_write_leaves_the_item_unsettled(persona, monkeypatch, caplog):
    # C459-j (item failure, the quarantine append raises)
    tmp, store, heb, queue = persona
    mems = _enqueue(queue, A, B, C)
    _fail_create(monkeypatch, store, B, ValueError("bad value"))
    real = cons.append_jsonl

    def append(path, *a, **k):
        if str(path).endswith(_QUARANTINE):
            raise OSError(errno.EIO, "I/O error")
        return real(path, *a, **k)

    monkeypatch.setattr(cons, "append_jsonl", append)
    with caplog.at_level(logging.INFO):
        _run(tmp, store, heb, Clf())

    assert B in [_label(e) for e in _unsettled(tmp)] and _quarantine(tmp) == []
    assert len(_errors_naming(caplog, mems[B])) >= 1 and "bravo" not in caplog.text


def test_the_saved_already_probe_failing_counts_as_not_saved(persona, monkeypatch):
    # C459-k(1)
    tmp, store, heb, queue = persona
    b = _enqueue(queue, A, B, C)[B]
    _fail_create(monkeypatch, store, B, ValueError("bad value"))
    _fail(monkeypatch, store, "get", lambda mid: mid == b.id, sqlite3.OperationalError("database is locked"))

    _run(tmp, store, heb, Clf())

    assert len(_quarantine(tmp)) == 1  # not saved: the item-scope failure is quarantined
    assert _contents(store).count(A) == 1


def test_settle_itself_failing_never_escapes_and_leaves_the_item_for_the_next_run(persona, monkeypatch, caplog):
    # C459-k(2)
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)
    _fail_create(monkeypatch, store, B, ValueError("bad value"))
    monkeypatch.setattr(cons, "settle_scope", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("internal")))

    with caplog.at_level(logging.INFO):
        _run(tmp, store, heb, Clf())  # must not raise

    assert B in [_label(e) for e in _unsettled(tmp)]
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    monkeypatch.undo()
    _run(tmp, store, heb, Clf())
    assert _contents(store) == sorted([A, B, C])


# ---------------------------------------------------------------------------
# the Decision boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_id", [["x"], {"a": 1}], ids=["list", "dict"])
def test_a_merge_with_an_unusable_target_id_is_deferred_like_a_merge_with_no_target(persona, target_id):
    # C459-l
    tmp, store, heb, queue = persona
    _enqueue(queue, B)
    clf = Clf(lambda cand: Decision("merge", target_id=target_id))

    result = _run(tmp, store, heb, clf)

    assert result.deferred == 1 and _quarantine(tmp) == []
    rows = _queue_rows(queue)
    assert [r["content"] for r in rows] == [B] and rows[0]["_source"] == "gate-deferred"
    assert clf.calls == [B]


@pytest.mark.parametrize("merged", [5, ["y"]], ids=["int", "list"])
def test_a_merge_with_non_string_merged_content_is_folded(persona, merged):
    # C459-l
    tmp, store, heb, queue = persona
    target = _mem("target memory")
    store.create(target)
    _enqueue(queue, B)

    result = _run(tmp, store, heb, Clf(lambda cand: Decision("merge", target_id=target.id, merged_content=merged)))

    assert result.merged == 1 and _quarantine(tmp) == []
    assert store.get(target.id, bump=False).content == f"target memory\n{B}"


def test_a_valid_merge_target_and_string_merged_content_still_merge(persona):
    # C459-l PIN
    tmp, store, heb, queue = persona
    target = _mem("target memory")
    store.create(target)
    _enqueue(queue, B)

    _run(tmp, store, heb, Clf(lambda cand: Decision("merge", target_id=target.id, merged_content="the merged text")))

    assert store.get(target.id, bump=False).content == "the merged text"


# ---------------------------------------------------------------------------
# counters, pins, in-batch duplicates
# ---------------------------------------------------------------------------


def test_the_result_log_carries_exactly_two_new_keys(persona):
    # C459-m
    tmp, store, heb, queue = persona
    keys = set(_run(tmp, store, heb, Clf()).as_log())

    assert keys == {"gate_batch", "exact_dropped", "salience_dropped", "duplicates", "promoted", "merged",
                    "corrections", "continuations", "deferred", "reappraised", "skipped", "failed", "quarantined"}


def test_a_failed_classifier_call_leaves_the_candidate_for_a_later_run(persona, caplog):
    # O1 / C459-x (amended): a classifier call that fails for ANY reason (not only a deferral) must not promote the candidate
    # unclassified. It is kept (requeued) and the pass ends, so the cost is one call per tick; the next run classifies it.
    tmp, store, heb, queue = persona
    mems = _enqueue(queue, A, B, C)

    def boom_on_b(cand):
        if cand.content == B:
            raise RuntimeError("claude exited with status 1")
        return Decision("new")

    clf = Clf(boom_on_b)
    with caplog.at_level(logging.INFO):
        result = _run(tmp, store, heb, clf)

    assert _contents(store) == [A]  # B and C are NOT promoted unclassified
    assert [r["content"] for r in _queue_rows(queue)] == [B] and _quarantine(tmp) == []
    assert [_label(e) for e in _unsettled(tmp)] == [C] and result.failed == 1
    assert any(mems[B].id in r.getMessage() and "RuntimeError" in r.getMessage() for r in caplog.records)

    healthy = Clf()
    _run(tmp, store, heb, healthy)
    assert _contents(store) == sorted([A, B, C])
    assert sorted(healthy.calls) == sorted([B, C])


class _FailingProvider:
    """generate() fails for prompts mentioning `refuse` with an ordinary error (the one call let through during an expired login)."""

    def __init__(self, refuse: str):
        self.refuse = refuse
        self.calls: list[str] = []

    def generate(self, prompt, *, system=None):
        self.calls.append(prompt)
        if self.refuse in prompt:
            raise RuntimeError("ClaudeCliProvider failed (exit 1)")
        return '{"verdict": "new"}'


def test_a_failing_real_haiku_classifier_call_does_not_promote_unclassified(persona):
    # O1: the real `_make_haiku_classifier`, whose `except Exception` used to return `Decision("new")` for a failed CALL.
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)
    provider = _FailingProvider(refuse="bravo")

    run_consolidation(store, persona_dir=tmp, provider=provider, hebbian=heb, reappraiser=_reappraiser)

    assert _contents(store) == [A]
    assert [r["content"] for r in _queue_rows(queue)] == [B]
    provider.refuse = "nothing-matches-this"
    run_consolidation(store, persona_dir=tmp, provider=provider, hebbian=heb, reappraiser=_reappraiser)
    assert _contents(store) == sorted([A, B, C])


def test_an_in_batch_duplicate_of_a_quarantined_item_survives_only_in_the_quarantine(persona, monkeypatch):
    # C459-o
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B)
    queue.enqueue(_mem("Bravo   memory"), source="dream")  # same normalized content as B
    _fail_create(monkeypatch, store, B, ValueError("bad value"))

    result = _run(tmp, store, heb, Clf())

    assert result.exact_dropped == 1
    assert _contents(store) == [A] and [r["entry"]["content"] for r in _quarantine(tmp)] == [B]


# ---------------------------------------------------------------------------
# crash safety
# ---------------------------------------------------------------------------


def test_a_kill_mid_run_loses_nothing_and_the_next_run_finishes_the_batch(persona):
    # C459-p: a BaseException is not caught anywhere; the processing file keeps the unsettled items.
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)

    def kill_on_b(cand):
        if cand.content == B:
            raise KeyboardInterrupt
        return Decision("new")

    with pytest.raises(KeyboardInterrupt):
        _run(tmp, store, heb, Clf(kill_on_b))

    assert _contents(store) == [A] and _queue_rows(queue) == []
    assert sorted(_label(e) for e in _unsettled(tmp)) == sorted([B, C])
    _run(tmp, store, heb, Clf())
    assert _contents(store) == sorted([A, B, C])
    assert not (tmp / _PROCESSING).exists() and not (tmp / _SETTLED).exists()


def test_a_kill_between_the_effect_and_its_settle_record_is_replayed_without_duplicating(persona, monkeypatch):
    # C459-q(3)
    from brain.memory.pending import ClaimedBatch

    tmp, store, heb, queue = persona
    _enqueue(queue, A, B)
    real = ClaimedBatch.settle

    def settle(self, entry):
        if entry.get("content") == A:
            raise KeyboardInterrupt
        return real(self, entry)

    monkeypatch.setattr(ClaimedBatch, "settle", settle)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp, store, heb, Clf())
    monkeypatch.undo()
    assert _contents(store) == [A]

    _run(tmp, store, heb, Clf())

    assert _contents(store) == sorted([A, B])  # A replayed and dropped as an exact duplicate: one row


def test_processing_file_present_and_queue_file_absent_is_recovered(persona):
    # C459-q(4)
    tmp, store, heb, queue = persona
    entry = {**_mem(A).to_dict(), "_source": "dream", "_enqueued_at": "t", "_claim_id": "old1"}
    write_jsonl(tmp / _PROCESSING, [entry])
    assert not queue.path.exists()

    _run(tmp, store, heb, Clf())

    assert _contents(store) == [A]


def test_processing_and_queue_holding_the_same_entries_end_as_one_row_each(persona):
    # C459-q(2): a crash between "write processing" and "empty queue" leaves the entries in both files.
    tmp, store, heb, queue = persona
    mems = _enqueue(queue, A, B)
    rows = read_jsonl(queue.path)
    write_jsonl(tmp / _PROCESSING, [{**r, "_claim_id": f"c{i}"} for i, r in enumerate(rows)])

    _run(tmp, store, heb, Clf())

    assert _contents(store) == sorted([A, B]) and mems


def test_a_kill_between_the_quarantine_write_and_the_settle_gives_a_second_record_on_replay(persona, monkeypatch, caplog):
    # C459-q(5): the stated residue: the replay fails again and quarantines the item a second time.
    from brain.memory.pending import ClaimedBatch

    tmp, store, heb, queue = persona
    mems = _enqueue(queue, A, B, C)
    _fail_create(monkeypatch, store, B, ValueError("bad value"))
    real = ClaimedBatch.settle

    def settle(self, entry):
        if entry.get("content") == B:
            raise KeyboardInterrupt
        return real(self, entry)

    with monkeypatch.context() as m:
        m.setattr(ClaimedBatch, "settle", settle)
        with pytest.raises(KeyboardInterrupt):
            _run(tmp, store, heb, Clf())
    assert len(_quarantine(tmp)) == 1

    with caplog.at_level(logging.INFO):
        _run(tmp, store, heb, Clf())

    assert len(_quarantine(tmp)) == 2
    assert len(_errors_naming(caplog, mems[B])) == 2  # the first run's ERROR was logged before caplog started: replay adds one
    assert _contents(store) == sorted([A, C])


# ---------------------------------------------------------------------------
# #473: a deferred refusal means "not now"
# ---------------------------------------------------------------------------


class _DeferringProvider:
    """A provider whose `generate` is refused (expired login) for prompts that mention `refuse`."""

    def __init__(self, refuse: str | None = None, reply: str = '{"verdict": "new"}'):
        self.refuse = refuse
        self.reply = reply
        self.calls: list[str] = []

    def generate(self, prompt, *, system=None):
        self.calls.append(prompt)
        if self.refuse is None or self.refuse in prompt:
            raise ProviderAuthDeferred("provider auth expired - deferred")
        return self.reply


def test_a_deferred_classifier_leaves_the_item_and_the_rest_for_the_next_run(persona, caplog):
    # C459-v (stub classifier)
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)

    def refuse_b(cand):
        if cand.content == B:
            raise ProviderAuthDeferred("provider auth expired - deferred")
        return Decision("new")

    clf = Clf(refuse_b)
    with caplog.at_level(logging.INFO):
        result = _run(tmp, store, heb, clf)

    assert _contents(store) == [A]  # nothing promoted unclassified
    assert sorted(_label(e) for e in _unsettled(tmp)) == sorted([B, C])
    assert _queue_rows(queue) == [] and _quarantine(tmp) == []
    assert result.failed == 0 and result.quarantined == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("deferred" in r.getMessage() for r in caplog.records)

    healthy = Clf()
    _run(tmp, store, heb, healthy)
    assert _contents(store) == sorted([A, B, C])
    assert len(clf.calls) + len(healthy.calls) == 4  # A, B; then B, C


def test_a_deferred_real_haiku_classifier_does_not_promote_unclassified(persona, caplog):
    # C459-v (the real `_make_haiku_classifier`, whose `except Exception` used to return `Decision("new")`)
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)
    provider = _DeferringProvider(refuse="bravo")

    with caplog.at_level(logging.INFO):
        result = run_consolidation(store, persona_dir=tmp, provider=provider, hebbian=heb, reappraiser=_reappraiser)

    assert _contents(store) == [A]
    assert sorted(_label(e) for e in _unsettled(tmp)) == sorted([B, C])
    assert result.failed == 0
    provider.refuse = "nothing-matches-this"
    run_consolidation(store, persona_dir=tmp, provider=provider, hebbian=heb, reappraiser=_reappraiser)
    assert _contents(store) == sorted([A, B, C])
    assert len(provider.calls) == 4


def test_a_deferred_reappraiser_does_not_consume_the_item(persona):
    # C459-w (stub reappraiser)
    tmp, store, heb, queue = persona
    r1 = _mem("old one")
    store.create(r1)
    queue.enqueue_reappraisal(r1.id, source="recall")
    _enqueue(queue, A, B)

    def deferred(m):
        raise ProviderAuthDeferred("provider auth expired - deferred")

    _run(tmp, store, heb, Clf(), reappraiser=deferred)

    assert _contents(store) == ["old one"]  # A and B wait
    assert any(e.get("memory_id") == r1.id for e in _unsettled(tmp))
    _run(tmp, store, heb, Clf(), reappraiser=_reappraiser)
    assert store.get(r1.id, bump=False).importance == 9.0
    assert _contents(store) == sorted(["old one", A, B])


def test_a_deferred_real_haiku_reappraiser_does_not_consume_the_item(persona):
    # C459-w (the real `_make_haiku_reappraiser`, whose `except Exception` used to return the importance unchanged)
    tmp, store, heb, queue = persona
    r1 = _mem("old one")
    store.create(r1)
    before = store.get(r1.id, bump=False).importance
    queue.enqueue_reappraisal(r1.id, source="recall")
    provider = _DeferringProvider()

    run_consolidation(store, persona_dir=tmp, provider=provider, hebbian=heb)

    assert any(e.get("memory_id") == r1.id for e in _unsettled(tmp))
    assert store.get(r1.id, bump=False).importance == before


def test_a_generic_reappraiser_fault_still_skips_the_item(persona):
    # C459-x PIN
    tmp, store, heb, queue = persona
    r1 = _mem("old one")
    store.create(r1)
    queue.enqueue_reappraisal(r1.id, source="recall")

    def boom(m):
        raise RuntimeError("boom")

    result = _run(tmp, store, heb, Clf(), reappraiser=boom)

    assert result.reappraised == 0 and _unsettled(tmp) == [] and _queue_rows(queue) == []


class _ReplyProvider:
    """A provider that ANSWERS: `replies` maps a marker in the prompt to a reply; `default` otherwise."""

    def __init__(self, default: str = '{"verdict": "new"}', replies: dict[str, str] | None = None):
        self.default = default
        self.replies = replies or {}
        self.calls: list[str] = []

    def generate(self, prompt, *, system=None):
        self.calls.append(prompt)
        for marker, reply in self.replies.items():
            if marker in prompt:
                return reply
        return self.default


_UNUSABLE_REPLIES = [
    ("", "empty reply"),
    ("   \n", "empty reply"),
    ("I cannot classify this.", "no JSON object"),
    ("[1, 2]", "no JSON object"),
    ('{"verdict": "duplicate", "names": ["Wren", "Pre', "no JSON object"),  # the object never closes
    ('{"verdict": "new", "names": ["Wren", }', "not valid JSON"),
    ("{bad json}", "not valid JSON"),
    ('{"verdict": "banana"}', "verdict"),
    ('{"verdict": 5}', "verdict"),
    ('{"names": ["Wren"]}', "verdict"),
    ('Here you go: {"verdict": "maybe"} done', "verdict"),
]


@pytest.mark.parametrize("reply, reason", _UNUSABLE_REPLIES, ids=lambda v: str(v)[:24])
def test_a_reply_with_no_usable_verdict_quarantines_the_candidate_and_never_promotes(persona, caplog, reply, reason):
    # A reply that ARRIVES but gives no usable verdict depends on this candidate's exchange, not on the store: an ITEM failure.
    # The candidate is quarantined with the raw reply as evidence, logged once with its id and the reason; the pass goes on.
    tmp, store, heb, queue = persona
    mems = _enqueue(queue, A, B, C)
    provider = _ReplyProvider(default=reply)

    with caplog.at_level(logging.INFO):
        result = run_consolidation(store, persona_dir=tmp, provider=provider, hebbian=heb, reappraiser=_reappraiser)

    assert _contents(store) == []  # nothing promoted unclassified
    recs = _quarantine(tmp)
    assert sorted(r["entry"]["content"] for r in recs) == sorted([A, B, C])  # the pass was NOT ended
    assert all(r["reply"] == reply and r["stage"] == "pass2" for r in recs)
    assert all("UnusableClassifierReply" in r["error"] and reason in r["error"] for r in recs)
    assert result.quarantined == 3 and result.promoted == 0 and _unsettled(tmp) == []
    for m in mems.values():
        assert len(_errors_naming(caplog, m)) == 1  # the candidate id and the reason, once
    if reply.strip():
        assert reply not in caplog.text  # the raw reply is evidence in the quarantine file, not in the log


def test_one_unusable_reply_does_not_stop_the_others(persona):
    tmp, store, heb, queue = persona
    _enqueue(queue, A, B, C)
    provider = _ReplyProvider(replies={"bravo": "no idea"})

    run_consolidation(store, persona_dir=tmp, provider=provider, hebbian=heb, reappraiser=_reappraiser)

    assert _contents(store) == sorted([A, C])
    assert [r["entry"]["content"] for r in _quarantine(tmp)] == [B]
    provider.replies = {}
    run_consolidation(store, persona_dir=tmp, provider=provider, hebbian=heb, reappraiser=_reappraiser)
    assert len([c for c in provider.calls if "bravo" in c]) == 1  # a quarantined candidate is final: not classified again


@pytest.mark.parametrize("reply, field", [
    ('{"verdict": "new"}', "promoted"),
    ('prefix {"verdict": "distinct", "names": []} suffix', "promoted"),
    ('{"verdict": "duplicate"}', "duplicates"),
])
def test_a_reply_with_a_usable_verdict_is_dispatched_as_before(persona, reply, field):
    # PIN
    tmp, store, heb, queue = persona
    _enqueue(queue, A)

    result = run_consolidation(store, persona_dir=tmp, provider=_ReplyProvider(default=reply), hebbian=heb,
                               reappraiser=_reappraiser)

    assert getattr(result, field) == 1 and _quarantine(tmp) == []
