"""Tests for P3 retention rework, Change 4 — one-time duplicate cleanup.

Covers C4.1 from changes/p3-retention/1.5-criteria.md.

Naming: synthetic user = Bob, persona = Canary, model = Claude. No Phoebe.
"""

from __future__ import annotations

import json

import pytest

from brain.engines.consolidation import _ARCHIVE_FILENAME
from brain.engines.dedup_sweep import run_dedup_sweep
from brain.memory.store import Memory, MemoryStore


def _mem(content: str, *, importance: float = 3.0) -> Memory:
    return Memory.create_new(
        content=content, memory_type="conversation", domain="us", importance=importance
    )


# --------------------------------------------------------------------------- C4.1
def test_c4_1_sweep_reduces_by_duplicate_surplus_lossless_and_reported(tmp_path):
    """A seeded corpus with a known exact-duplicate group (3 rows, same
    normalized content) plus distinct items: the sweep reduces the active
    count by exactly the duplicate surplus (2), every distinct content
    string remains present+active, each removed pre-image is archived, and
    a report is emitted. Fail-test: a no-op sweep (no reduction) or any
    missing distinct string flags; an unarchived merge flags."""
    store = MemoryStore(tmp_path / "memories.db")

    dup_a = _mem("Bob's favorite color is blue.", importance=2.0)
    dup_b = _mem("bob's   favorite color is blue.", importance=6.0)  # normalizes equal to dup_a
    dup_c = _mem("BOB'S FAVORITE COLOR IS BLUE.", importance=1.0)
    distinct_1 = _mem("Bob went hiking on Saturday.", importance=4.0)
    distinct_2 = _mem("Bob's dog is named Rex.", importance=5.0)

    for m in (dup_a, dup_b, dup_c, distinct_1, distinct_2):
        store.create(m)

    before_count = store.count(active_only=True)
    assert before_count == 5

    report = run_dedup_sweep(store, tmp_path)

    after_count = store.count(active_only=True)
    assert after_count == before_count - 2  # exactly the duplicate surplus (3 -> 1)
    assert report.groups_merged == 1
    assert report.rows_removed == 2

    survivors = {m.content for m in store.list_active()}
    assert "Bob went hiking on Saturday." in survivors
    assert "Bob's dog is named Rex." in survivors
    # Exactly one of the three duplicate spellings remains (whichever was
    # kept as canonical).
    dup_spellings = {dup_a.content, dup_b.content, dup_c.content}
    assert len(dup_spellings & survivors) == 1

    # Canonical carries the max importance across the merged group (6.0).
    remaining_dup_id = next(
        m.id for m in store.list_active() if m.content in dup_spellings
    )
    remaining = store.get(remaining_dup_id, bump=False)
    assert remaining.importance == pytest.approx(6.0)

    # Loss-preserving: every removed row's full pre-image is archived BEFORE
    # removal, reason "dedup_sweep".
    archive_path = tmp_path / _ARCHIVE_FILENAME
    assert archive_path.exists()
    archived_records = [json.loads(line) for line in archive_path.read_text().splitlines()]
    assert len(archived_records) == 2
    for rec in archived_records:
        assert rec["reason"] == "dedup_sweep"
    archived_contents = {rec["target"]["content"] for rec in archived_records}
    assert archived_contents == (dup_spellings - survivors)

    # Report file emitted.
    report_path = tmp_path / "dedup_sweep_report.jsonl"
    assert report_path.exists()
    report_records = [json.loads(line) for line in report_path.read_text().splitlines()]
    assert report_records[-1]["rows_removed"] == 2
    assert report_records[-1]["groups_merged"] == 1
    assert len(report_records[-1]["merges"]) == 2

    store.close()


def test_c4_1_sweep_is_idempotent(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    store.create(_mem("Repeated content.", importance=2.0))
    store.create(_mem("repeated   content.", importance=5.0))
    store.create(_mem("A distinct memory.", importance=1.0))

    first = run_dedup_sweep(store, tmp_path)
    assert first.rows_removed == 1
    second = run_dedup_sweep(store, tmp_path)
    assert second.rows_removed == 0
    assert second.groups_merged == 0
    store.close()


def test_c4_1_no_duplicates_is_a_true_no_op(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    store.create(_mem("First distinct memory."))
    store.create(_mem("Second distinct memory."))
    before = store.count(active_only=True)
    report = run_dedup_sweep(store, tmp_path)
    assert store.count(active_only=True) == before
    assert report.rows_removed == 0
    assert report.groups_merged == 0
    store.close()


def test_c4_1_optional_near_dup_judge_layer(tmp_path):
    """The optional injectable near-dup judge (default OFF) can merge a pair
    the exact-normalize layer does not catch."""
    store = MemoryStore(tmp_path / "memories.db")
    older = _mem("Bob likes tea.", importance=3.0)
    newer = _mem("Bob likes tea with honey.", importance=7.0)
    store.create(older)
    store.create(newer)

    def _judge(candidate: Memory, canonical: Memory) -> str:
        if "tea" in candidate.content and "tea" in canonical.content:
            return "duplicate"
        return "distinct"

    report = run_dedup_sweep(store, tmp_path, judge=_judge)
    assert report.rows_removed == 1
    survivors = store.list_active()
    assert len(survivors) == 1
    # Canonical (older) absorbed the higher importance from the near-dup.
    assert survivors[0].importance == pytest.approx(7.0)
    store.close()


def test_c4_1_no_judge_leaves_near_dups_untouched(tmp_path):
    """Default (no judge): only the deterministic exact-normalize layer
    runs — near-duplicates with different wording are left alone."""
    store = MemoryStore(tmp_path / "memories.db")
    store.create(_mem("Bob likes tea.", importance=3.0))
    store.create(_mem("Bob likes tea with honey.", importance=7.0))
    report = run_dedup_sweep(store, tmp_path)
    assert report.rows_removed == 0
    assert store.count(active_only=True) == 2
    store.close()


# --------------------------------------------------------------------------- #431
def _crystallise(persona_dir, memory_id: str) -> str:
    """Accept a soul candidate for ``memory_id`` through the real review code (the forgetting tests' helper)."""
    from tests.forgetting.test_orchestrator import _accept_candidate_for_real

    return _accept_candidate_for_real(persona_dir, memory_id)


def _active_ids(store: MemoryStore) -> set[str]:
    return {m.id for m in store.list_active()}


def test_431_exact_pass_keeps_the_crystallised_duplicate(tmp_path):
    """#431: a soul-crystallised memory must never leave memories.db through the sweep. Its exact
    duplicate scores higher, so on importance alone the crystallised row would be the one deleted; the
    sweep keeps it as canonical instead and removes the plain duplicate."""
    store = MemoryStore(tmp_path / "memories.db")
    soul = _mem("Bob held my hand at the harbour.", importance=2.0)
    plain = _mem("bob held my hand at the harbour.", importance=8.0)
    for m in (soul, plain):
        store.create(m)
    _crystallise(tmp_path, soul.id)

    report = run_dedup_sweep(store, tmp_path)

    assert _active_ids(store) == {soul.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(soul.id, plain.id)]
    assert store.get(soul.id, bump=False).importance == 8.0  # the group max still carries over
    store.close()


def test_431_exact_pass_deletes_neither_when_both_duplicates_are_soul_linked(tmp_path):
    """#431: one crystallised, one under soul review: both stay. Guards the group count too: a group
    the sweep could not merge must not be reported as merged, or every re-run reports a merge that
    never happened instead of a no-op."""
    from brain.ingest.soul_queue import queue_soul_candidate
    from brain.ingest.types import ExtractedItem

    store = MemoryStore(tmp_path / "memories.db")
    a = _mem("Bob read me the letter.", importance=2.0)
    b = _mem("bob read me the letter.", importance=5.0)
    for m in (a, b):
        store.create(m)
    _crystallise(tmp_path, a.id)
    queue_soul_candidate(
        tmp_path,
        memory_id=b.id,
        item=ExtractedItem(text="Bob read me the letter.", label="observation", importance=8),
        session_id="monologue",
    )

    report = run_dedup_sweep(store, tmp_path)

    assert _active_ids(store) == {a.id, b.id}
    assert (report.groups_merged, report.rows_removed, report.merges) == (0, 0, [])
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()
    store.close()


def test_431_near_dup_pass_keeps_the_crystallised_member_of_a_pair(tmp_path):
    """#431, near-dup layer: the pass removes the LATER of a duplicate pair. When the later one is
    crystallised it stays and the earlier, plain one goes instead."""
    from datetime import UTC, datetime, timedelta

    store = MemoryStore(tmp_path / "memories.db")
    plain = _mem("Bob and I watched the boats.", importance=6.0)
    soul = _mem("We watched the boats with Bob.", importance=2.0)
    object.__setattr__(plain, "created_at", datetime.now(UTC) - timedelta(days=2))
    object.__setattr__(soul, "created_at", datetime.now(UTC) - timedelta(days=1))
    for m in (plain, soul):
        store.create(m)
    _crystallise(tmp_path, soul.id)

    report = run_dedup_sweep(store, tmp_path, judge=lambda _a, _b: "duplicate")

    assert _active_ids(store) == {soul.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(soul.id, plain.id)]
    assert store.get(soul.id, bump=False).importance == 6.0
    store.close()


def test_431_near_dup_pass_leaves_a_pair_of_soul_linked_rows_alone(tmp_path):
    """#431, near-dup layer: one crystallised, one under soul review -> neither is deleted."""
    from brain.ingest.soul_queue import queue_soul_candidate
    from brain.ingest.types import ExtractedItem

    store = MemoryStore(tmp_path / "memories.db")
    a = _mem("Bob sang in the kitchen.")
    b = _mem("Bob was singing in the kitchen.")
    for m in (a, b):
        store.create(m)
    _crystallise(tmp_path, a.id)
    queue_soul_candidate(
        tmp_path,
        memory_id=b.id,
        item=ExtractedItem(text="Bob was singing in the kitchen.", label="observation", importance=8),
        session_id="monologue",
    )

    report = run_dedup_sweep(store, tmp_path, judge=lambda _a, _b: "duplicate")

    assert _active_ids(store) == {a.id, b.id}
    assert report.rows_removed == 0
    store.close()


def test_431_sweep_fails_closed_when_the_soul_exemptions_cannot_be_read(tmp_path):
    """#431 with #392 semantics: a real read failure (a directory where soul_candidates.jsonl should be)
    raises before any row is touched; nothing is archived, deleted or reported."""
    from brain.forgetting import SoulExemptionsUnavailableError

    store = MemoryStore(tmp_path / "memories.db")
    a = _mem("Bob fixed the gate.")
    b = _mem("bob fixed the gate.")
    for m in (a, b):
        store.create(m)
    (tmp_path / "soul_candidates.jsonl").mkdir()

    with pytest.raises(SoulExemptionsUnavailableError):
        run_dedup_sweep(store, tmp_path)

    assert _active_ids(store) == {a.id, b.id}
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()
    assert not (tmp_path / "dedup_sweep_report.jsonl").exists()
    store.close()


def _aged(content: str, days_ago: int, *, importance: float = 3.0) -> Memory:
    from datetime import UTC, datetime, timedelta

    m = _mem(content, importance=importance)
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=days_ago))
    return m


def test_431_near_dup_pass_stops_comparing_a_row_it_just_dropped(tmp_path):
    """#431: A plain (oldest), B soul-linked, C plain, all judged duplicates. A is dropped in
    favour of B; the pass must stop using A, or it merges C into the deleted A and C is lost too."""
    store = MemoryStore(tmp_path / "memories.db")
    a = _aged("Bob and the red kite.", 3)
    b = _aged("The red kite Bob flew.", 2)
    c = _aged("Bob flew a red kite.", 1)
    for m in (a, b, c):
        store.create(m)
    _crystallise(tmp_path, b.id)

    report = run_dedup_sweep(store, tmp_path, judge=lambda _x, _y: "duplicate")

    assert _active_ids(store) == {b.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(b.id, a.id), (b.id, c.id)]
    store.close()


def test_near_dup_pass_never_lowers_a_canonical_it_already_raised(tmp_path):
    """#431: A (1, oldest) absorbs C (9) and is raised to 9 in the store. Guards against comparing D (3)
    with A's stale in-memory importance of 1, which writes 3 back and loses the max. The canonical must
    end at the max of everything merged into it."""
    store = MemoryStore(tmp_path / "memories.db")
    a = _aged("Bob baked bread.", 3, importance=1.0)
    c = _aged("Bob was baking bread.", 2, importance=9.0)
    d = _aged("Bread, baked by Bob.", 1, importance=3.0)
    for m in (a, c, d):
        store.create(m)

    run_dedup_sweep(store, tmp_path, judge=lambda _x, _y: "duplicate")

    assert _active_ids(store) == {a.id}
    assert store.get(a.id, bump=False).importance == 9.0
    store.close()


def test_431_near_dup_swap_carries_the_max_and_retargets_earlier_merges(tmp_path):
    """#431: A (1, oldest) absorbs C (9), then A is dropped in favour of the soul-linked B (2).
    B must end at 9, A's archived pre-image must show the 9 it held, and no merge record may name the
    deleted A as its canonical: C's merge now points at B, where its content survives."""
    store = MemoryStore(tmp_path / "memories.db")
    a = _aged("Bob fed the swans.", 3, importance=1.0)
    c = _aged("Bob was feeding the swans.", 2, importance=9.0)
    b = _aged("The swans Bob fed.", 1, importance=2.0)
    for m in (a, c, b):
        store.create(m)
    _crystallise(tmp_path, b.id)

    report = run_dedup_sweep(store, tmp_path, judge=lambda _x, _y: "duplicate")

    assert _active_ids(store) == {b.id}
    assert store.get(b.id, bump=False).importance == 9.0
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(b.id, c.id), (b.id, a.id)]
    archived = [
        json.loads(line)["target"]
        for line in (tmp_path / _ARCHIVE_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert [(t["id"], t["importance"]) for t in archived] == [(c.id, 9.0), (a.id, 9.0)]
    store.close()


def test_431_failed_exemption_read_during_a_sweep_is_logged_as_the_sweep(tmp_path, caplog):
    """The shared loader's warnings name the pass that hit them, so a sweep failure is not read as a
    forgetting failure."""
    import logging

    from brain.forgetting import SoulExemptionsUnavailableError

    store = MemoryStore(tmp_path / "memories.db")
    (tmp_path / "soul_candidates.jsonl").mkdir()
    with caplog.at_level(logging.WARNING), pytest.raises(SoulExemptionsUnavailableError):
        run_dedup_sweep(store, tmp_path)
    store.close()

    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert messages and all(m.startswith("dedup_sweep: ") for m in messages), messages
