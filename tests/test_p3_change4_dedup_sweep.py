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


def test_431_near_dup_pass_never_lowers_a_canonical_it_already_raised(tmp_path):
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
    """#431: the shared loader's warnings name the pass that hit them. Guards against a failed read in
    a sweep being logged as "forgetting:", which sends the reader to the wrong pass."""
    import logging

    from brain.forgetting import SoulExemptionsUnavailableError

    store = MemoryStore(tmp_path / "memories.db")
    (tmp_path / "soul_candidates.jsonl").mkdir()
    with caplog.at_level(logging.WARNING), pytest.raises(SoulExemptionsUnavailableError):
        run_dedup_sweep(store, tmp_path)
    store.close()

    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert messages and all(m.startswith("dedup_sweep: ") for m in messages), messages


def test_431_near_dup_pass_skips_a_pair_whose_row_vanished_mid_pass(tmp_path, caplog):
    """#431: the row a pair would drop (here the later C) is deleted by another writer after the pass
    listed it. Guards against using the stale in-memory copy: that archives C's stale pre-image and
    then crashes the whole sweep with KeyError from hard_delete. The pair is skipped with a warning
    instead, and nothing is archived."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    a = _aged("Bob painted the fence.", 2)
    c = _aged("Bob was painting the fence.", 1)
    for m in (a, c):
        store.create(m)

    def _judge_while_c_vanishes(candidate: Memory, _canonical: Memory) -> str:
        store.hard_delete(candidate.id)  # a concurrent writer removes C between list and re-read
        return "duplicate"

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path, judge=_judge_while_c_vanishes)

    assert _active_ids(store) == {a.id}
    assert (report.rows_removed, report.merges) == (0, [])
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(a.id in m and c.id in m for m in warnings), warnings
    store.close()


def test_431_near_dup_pass_skips_a_pair_whose_survivor_vanished_mid_pass(tmp_path, caplog):
    """#431: the row a pair would keep (here the earlier ``cand``) is deleted by another writer after
    the pass listed it. Guards against checking only the dropped row: the pass would then archive and
    delete C and record a merge into the vanished A, so both rows leave the store and C survives only in
    the archive. C must stay, with no merge record, no archive row and a warning naming both ids."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    a = _aged("Bob mended the net.", 2, importance=5.0)
    c = _aged("Bob was mending the net.", 1, importance=1.0)
    for m in (a, c):
        store.create(m)

    def _judge_while_a_vanishes(_candidate: Memory, canonical: Memory) -> str:
        store.hard_delete(canonical.id)  # a concurrent writer removes A, the row that would be kept
        return "duplicate"

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path, judge=_judge_while_a_vanishes)

    assert _active_ids(store) == {c.id}
    # The sweep's re-read of C must not count as a recall: an unbumped row keeps the stored values.
    survivor = store.get(c.id, bump=False)
    assert (survivor.recall_count, survivor.last_accessed_at) == (0.0, None)
    assert (report.rows_removed, report.merges) == (0, [])
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(a.id in m and c.id in m for m in warnings), warnings
    store.close()


# How another writer takes a row out of play mid-pass: deleted (#431) or deactivated (#505).
_VANISH = {"delete": MemoryStore.hard_delete, "deactivate": MemoryStore.deactivate}


@pytest.mark.parametrize("how", sorted(_VANISH))
def test_431_near_dup_pass_stops_judging_a_cand_that_vanished(tmp_path, caplog, how):
    """#431/#505: once the pass finds that ``cand`` (the earlier row of a pair) is deleted or
    deactivated, it must stop comparing it. Guards against judging the dead row against every remaining
    row, which costs a judge call and logs another skip warning per pair (and, for a deactivated row,
    could merge later rows into it)."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    a = _aged("Bob swept the porch.", 3)
    c = _aged("Bob was sweeping the porch.", 2)
    d = _aged("The porch, swept by Bob.", 1)
    for m in (a, c, d):
        store.create(m)
    calls: list[tuple[str, str]] = []

    def _judge(candidate: Memory, canonical: Memory) -> str:
        calls.append((candidate.id, canonical.id))
        if len(calls) == 1:
            _VANISH[how](store, a.id)  # another writer removes A while its first pair is judged
        return "duplicate"

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path, judge=_judge)

    assert calls == [(c.id, a.id), (d.id, c.id)]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, warnings
    assert _active_ids(store) == {c.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(c.id, d.id)]
    store.close()


def _logging_judge(
    store: MemoryStore,
    *,
    vanish: str | None = None,
    how: str = "delete",
    distinct: set[str] = frozenset(),
):
    """A judge that records every (candidate, canonical) call, takes ``vanish`` out of play during the
    first call (``how``: deleted or deactivated, as a concurrent writer would), and calls a pair
    distinct when either id is in ``distinct``."""
    calls: list[tuple[str, str]] = []

    def _judge(candidate: Memory, canonical: Memory) -> str:
        calls.append((candidate.id, canonical.id))
        if vanish is not None and len(calls) == 1:
            _VANISH[how](store, vanish)
        return "distinct" if {candidate.id, canonical.id} & distinct else "duplicate"

    return _judge, calls


def test_431_near_dup_pass_keeps_judging_cand_after_a_later_row_vanished(tmp_path, caplog):
    """#431: the later row C vanishes while the A/C pair is judged. Guards two failures: stopping A's
    comparisons as if A had gone (D would never be merged into A), and judging the dead C again as a
    ``cand`` later (one judge call and one warning per remaining row)."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    a, c, d, e = (_aged(f"Bob counted the {w}.", n) for w, n in (("gulls", 4), ("geese", 3), ("terns", 2), ("crows", 1)))
    for m in (a, c, d, e):
        store.create(m)
    judge, calls = _logging_judge(store, vanish=c.id, distinct={e.id})

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path, judge=judge)

    assert calls == [(c.id, a.id), (d.id, a.id), (e.id, a.id)]
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    assert _active_ids(store) == {a.id, e.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(a.id, d.id)]
    store.close()


@pytest.mark.parametrize("how", sorted(_VANISH))
def test_431_near_dup_swap_keeps_judging_cand_after_the_soul_linked_row_vanished(tmp_path, caplog, how):
    """#431/#505 (deleted or deactivated), swapped orientation (the later row B is soul-linked, so B would be kept and the earlier A
    dropped): B vanishes while A/B is judged. A is still alive, so it must go on to be judged against D
    and merge it. Guards against treating the vanished kept row as if ``cand`` had gone."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    a, b, d = (_aged(f"Bob fed the {w}.", n) for w, n in (("ducks", 3), ("drakes", 2), ("ducklings", 1)))
    for m in (a, b, d):
        store.create(m)
    _crystallise(tmp_path, b.id)
    judge, calls = _logging_judge(store, vanish=b.id, how=how)

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path, judge=judge)

    assert calls == [(b.id, a.id), (d.id, a.id)]
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    assert _active_ids(store) == {a.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(a.id, d.id)]
    store.close()


@pytest.mark.parametrize("how", sorted(_VANISH))
def test_431_near_dup_swap_stops_judging_cand_once_it_vanished(tmp_path, caplog, how):
    """#431/#505 (deleted or deactivated), swapped orientation: B (later) is soul-linked, so ``cand`` A is the row to drop, and A
    vanishes while A/B is judged. Guards against going on to judge the dead A against D: the pass must
    move straight to B, which then keeps D's pair."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    a, b, d = (_aged(f"Bob fed the {w}.", n) for w, n in (("ducks", 3), ("drakes", 2), ("ducklings", 1)))
    for m in (a, b, d):
        store.create(m)
    _crystallise(tmp_path, b.id)
    judge, calls = _logging_judge(store, vanish=a.id, how=how)

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path, judge=judge)

    assert calls == [(b.id, a.id), (d.id, b.id)]
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    assert _active_ids(store) == {b.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(b.id, d.id)]
    store.close()


def _pair_judge(duplicate_groups: list[set[str]]):
    """A judge that calls a pair a duplicate only when both ids sit in one of ``duplicate_groups``, and
    records every (candidate, canonical) call."""
    calls: list[tuple[str, str]] = []

    def _judge(candidate: Memory, canonical: Memory) -> str:
        calls.append((candidate.id, canonical.id))
        pair = {candidate.id, canonical.id}
        return "duplicate" if any(pair <= g for g in duplicate_groups) else "distinct"

    return _judge, calls


def test_431_near_dup_pass_never_judges_a_row_it_already_merged_away(tmp_path, caplog):
    """#431: D is merged into A while A is ``cand``; when C (alive, distinct from A) becomes ``cand``,
    D is already gone. Guards against judging C against the deleted D: a wasted judge call and, on a
    "duplicate" verdict, a spurious vanished-row warning."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    a, c, d = (_aged(f"Bob {w} the boat.", n) for w, n in (("rowed", 3), ("sold", 2), ("was rowing", 1)))
    for m in (a, c, d):
        store.create(m)
    judge, calls = _pair_judge([{a.id, d.id}, {c.id, d.id}])

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path, judge=judge)

    assert calls == [(c.id, a.id), (d.id, a.id)]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert _active_ids(store) == {a.id, c.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(a.id, d.id)]
    store.close()


def test_431_near_dup_swap_retargets_only_the_merges_into_the_dropped_row(tmp_path):
    """#431: Y is merged into X, then (separately) A is dropped in favour of the soul-linked B. Guards
    against retargeting every earlier merge record to B: only records whose canonical was A may move;
    the X <- Y record must still name X, which is alive and holds Y's content."""
    store = MemoryStore(tmp_path / "memories.db")
    x, y, a, b = (_aged(f"Bob {w}.", n) for w, n in (("drew a map", 4), ("was drawing a map", 3), ("lit a fire", 2), ("was lighting a fire", 1)))
    for m in (x, y, a, b):
        store.create(m)
    _crystallise(tmp_path, b.id)
    judge, _calls = _pair_judge([{x.id, y.id}, {a.id, b.id}])

    report = run_dedup_sweep(store, tmp_path, judge=judge)

    assert _active_ids(store) == {x.id, b.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(x.id, y.id), (b.id, a.id)]
    store.close()


def test_near_dup_pass_skips_a_pair_whose_judge_raised(tmp_path):
    """A judge fault on one pair must skip that pair, never lose a row or end the sweep. Guards against
    falling through to the merge with the previous pair's verdict: C is a real duplicate of A and is
    merged, then the D/A call raises, and D must not be merged on C's "duplicate" verdict."""
    store = MemoryStore(tmp_path / "memories.db")
    a, c, d = (_aged(f"Bob {w} the hedge.", n) for w, n in (("cut", 3), ("trimmed", 2), ("planted", 1)))
    for m in (a, c, d):
        store.create(m)

    def _judge(candidate: Memory, canonical: Memory) -> str:
        if candidate.id == d.id:
            raise RuntimeError("judge fault")
        return "duplicate"

    report = run_dedup_sweep(store, tmp_path, judge=_judge)

    assert _active_ids(store) == {a.id, d.id}
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(a.id, c.id)]
    store.close()


def test_431_near_dup_merge_does_not_count_as_a_recall_and_previews_the_dropped_row(tmp_path):
    """#431: the near-dup pass re-reads both rows of a pair before merging. Guards two failures: a
    re-read that bumps recall (``store.get`` without ``bump=False``) raises the survivor's
    ``recall_count`` and ``last_accessed_at``, so a maintenance sweep skews forgetting salience; and a
    merge record whose preview shows the kept row's content instead of the removed one."""
    store = MemoryStore(tmp_path / "memories.db")
    a = _aged("Bob tuned the old guitar.", 2)
    c = _aged("Bob was tuning the old guitar.", 1)
    for m in (a, c):
        store.create(m)

    report = run_dedup_sweep(store, tmp_path, judge=lambda _x, _y: "duplicate")

    survivor = store.get(a.id, bump=False)
    assert _active_ids(store) == {a.id}
    assert (survivor.recall_count, survivor.last_accessed_at) == (0.0, None)
    assert [(m.canonical_id, m.removed_id, m.content_preview) for m in report.merges] == [
        (a.id, c.id, c.content[:120])
    ]
    store.close()


def test_505_near_dup_pass_treats_a_row_deactivated_mid_pass_as_vanished(tmp_path, caplog):
    """#505: the sweep lists only active rows, but ``store.get`` returns a row whatever its ``active``
    flag. Guards against archiving and hard-deleting C after another writer deactivated it mid-pass (a
    deactivated row is out of play, not a duplicate to merge): the pair is skipped with a warning, C
    is never judged again, and nothing is archived."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    a, c, d = (_aged(f"Bob {w} the lamp.", n) for w, n in (("lit", 3), ("was lighting", 2), ("fixed", 1)))
    for m in (a, c, d):
        store.create(m)
    calls: list[tuple[str, str]] = []

    def _judge(candidate: Memory, canonical: Memory) -> str:
        calls.append((candidate.id, canonical.id))
        if len(calls) == 1:
            store.deactivate(c.id)  # another writer deactivates C while A/C is judged
        return "distinct" if d.id in (candidate.id, canonical.id) else "duplicate"

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path, judge=_judge)

    assert calls == [(c.id, a.id), (d.id, a.id)]
    assert (report.rows_removed, report.merges) == (0, [])
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()
    assert store.get(c.id, bump=False) is not None, "a deactivated row must not be hard-deleted"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and a.id in warnings[0] and c.id in warnings[0], warnings
    store.close()


def _exact_group(store: MemoryStore) -> tuple[Memory, Memory, Memory]:
    """Three exact duplicates: K (oldest, most important: the canonical), then D2, then D1. The sweep
    lists newest first, so it removes D1 before D2."""
    k = _aged("Bob locked the shed.", 3, importance=9.0)
    d2 = _aged("bob locked the shed.", 2, importance=1.0)
    d1 = _aged("BOB LOCKED THE SHED.", 1, importance=5.0)
    for m in (k, d2, d1):
        store.create(m)
    return k, d2, d1


def _on_first_archive(monkeypatch, action) -> None:
    """Run ``action`` (another writer) inside the sweep's first archive call, then archive as usual."""
    import brain.engines.dedup_sweep as sweep

    real = sweep._archive_preimage
    fired: list[bool] = []

    def _hooked(persona_dir, target):
        if not fired:
            fired.append(True)
            action()
        real(persona_dir, target)

    monkeypatch.setattr(sweep, "_archive_preimage", _hooked)


def _archived_ids(persona_dir) -> list[str]:
    path = persona_dir / _ARCHIVE_FILENAME
    return [json.loads(line)["target"]["id"] for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("how", sorted(_VANISH))
def test_505_exact_pass_skips_a_duplicate_that_vanished_mid_group(tmp_path, monkeypatch, caplog, how):
    """#505: the exact pass removes duplicates from its opening ``list_active`` snapshot. Guards against
    archiving and deleting D2 from that stale copy after another writer deleted it (``hard_delete``
    KeyError ends the whole sweep) or deactivated it (a row out of play is removed anyway). D2 is
    skipped with a warning; D1 still merges into K."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    k, d2, d1 = _exact_group(store)
    _on_first_archive(monkeypatch, lambda: _VANISH[how](store, d2.id))

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path)

    assert (report.groups_merged, report.rows_removed) == (1, 1)
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(k.id, d1.id)]
    assert _archived_ids(tmp_path) == [d1.id]
    assert _active_ids(store) == {k.id}
    assert (store.get(d2.id, bump=False) is None) == (how == "delete"), "a deactivated D2 must not be deleted"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and d2.id in warnings[0], warnings
    store.close()


@pytest.mark.parametrize("how", sorted(_VANISH))
def test_505_exact_pass_stops_a_group_whose_canonical_vanished(tmp_path, monkeypatch, caplog, how):
    """#505: K is soul-linked, so it is the canonical though D1 outranks it. Another writer deletes or
    deactivates K while D1 is being merged into it. Guards against going on to delete D2 in favour of a
    row that is gone, and against carrying D1's importance onto it (``store.update`` KeyError for a
    deleted K, a silent write for a deactivated one). D2 stays; a warning names K."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    k, d2, d1 = _exact_group(store)
    store.update(k.id, importance=2.0)
    _crystallise(tmp_path, k.id)
    _on_first_archive(monkeypatch, lambda: _VANISH[how](store, k.id))

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path)

    assert (report.groups_merged, report.rows_removed) == (1, 1)
    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(k.id, d1.id)]
    assert _archived_ids(tmp_path) == [d1.id]
    assert _active_ids(store) == {d2.id}
    if how == "deactivate":
        assert store.get(k.id, bump=False).importance == 2.0, "nothing may be carried onto a gone canonical"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and k.id in warnings[0], warnings
    store.close()


def test_505_exact_pass_archives_the_row_as_it_was_when_removed(tmp_path, monkeypatch):
    """#505: another writer raises D2's importance while D1 is being archived. Guards against archiving
    D2 from the opening ``list_active`` snapshot: the archive is the only copy left of a removed row, so
    it must hold the row as it was deleted, and the canonical must carry that current importance."""
    store = MemoryStore(tmp_path / "memories.db")
    k, d2, d1 = _exact_group(store)
    _on_first_archive(monkeypatch, lambda: store.update(d2.id, importance=10.0))

    run_dedup_sweep(store, tmp_path)

    lines = (tmp_path / _ARCHIVE_FILENAME).read_text(encoding="utf-8").splitlines()
    archived = {t["id"]: t["importance"] for t in (json.loads(line)["target"] for line in lines)}
    assert archived == {d1.id: 5.0, d2.id: 10.0}
    assert store.get(k.id, bump=False).importance == 10.0
    store.close()


def test_505_exact_group_whose_duplicates_all_vanished_is_not_counted_as_merged(tmp_path, monkeypatch, caplog):
    """#505: D vanishes after the group is formed and before anything is removed. Guards against
    counting the group as merged when it removed no row (a report claiming a merge that never
    happened)."""
    import logging

    import brain.engines.dedup_sweep as sweep

    store = MemoryStore(tmp_path / "memories.db")
    k = _aged("Bob shut the gate.", 2, importance=9.0)
    d = _aged("bob shut the gate.", 1, importance=1.0)
    for m in (k, d):
        store.create(m)
    real_pick = sweep._pick_canonical

    def _pick_then_vanish(members, exempt):
        picked = real_pick(members, exempt)
        store.hard_delete(d.id)  # another writer, between grouping and removal
        return picked

    monkeypatch.setattr(sweep, "_pick_canonical", _pick_then_vanish)
    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path)

    assert (report.groups_merged, report.rows_removed, report.merges) == (0, 0, [])
    assert not (tmp_path / _ARCHIVE_FILENAME).exists()
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    store.close()


def test_505_near_dup_judge_sees_the_canonical_after_its_exact_merge(tmp_path):
    """#505: the exact pass hands its canonicals to the near-dup judge. Guards against handing over the
    opening snapshot: after D1 (5) merges into K, the judge must see K as it now is (importance 5), not
    as it was listed (1)."""
    store = MemoryStore(tmp_path / "memories.db")
    k = _aged("Bob wound the clock.", 3, importance=1.0)
    d1 = _aged("bob wound the clock.", 2, importance=5.0)
    other = _aged("Bob oiled the hinges.", 1)
    for m in (k, d1, other):
        store.create(m)
    _crystallise(tmp_path, k.id)  # K stays canonical although D1 outranks it
    seen: dict[str, float] = {}

    def _judge(candidate: Memory, canonical: Memory) -> str:
        seen[canonical.id] = canonical.importance
        return "distinct"

    run_dedup_sweep(store, tmp_path, judge=_judge)

    assert seen == {k.id: 5.0}
    store.close()


def test_505_exact_pass_skips_a_duplicate_edited_away_mid_group(tmp_path, monkeypatch, caplog):
    """#505: another writer rewrites D2 while D1 is being merged, so D2 is no longer a duplicate of K.
    Guards against deleting it on the strength of the opening snapshot's grouping: D2's new content
    would survive only in the archive. D2 is skipped with a warning and keeps its new content."""
    import logging

    store = MemoryStore(tmp_path / "memories.db")
    k, d2, d1 = _exact_group(store)
    _on_first_archive(monkeypatch, lambda: store.update(d2.id, content="Bob found the shed key."))

    with caplog.at_level(logging.WARNING, logger="brain.engines.dedup_sweep"):
        report = run_dedup_sweep(store, tmp_path)

    assert [(m.canonical_id, m.removed_id) for m in report.merges] == [(k.id, d1.id)]
    assert _archived_ids(tmp_path) == [d1.id]
    assert _active_ids(store) == {k.id, d2.id}
    assert store.get(d2.id, bump=False).content == "Bob found the shed key."
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and d2.id in warnings[0], warnings
    store.close()
