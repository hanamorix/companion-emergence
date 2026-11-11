"""Same-type merges in the temp consolidation gate (#619).

A candidate merges only into an existing memory of its own merge type (the three monologue types are one merge type); an
observation candidate may also merge into a fact, a fact never into an observation. The same rule holds for the judge's
context, the merge verdict, the pass-1 exact-duplicate drop and the new `search_text` type filter. Synthetic persona only:
persona "Canary", user "Bob".

The helpers are imported inside the tests so a missing name fails the test that needs it and not the whole module's
collection.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from brain.engines import consolidation
from brain.engines.consolidation import _ARCHIVE_FILENAME, Decision, run_consolidation
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore
from brain.tools.impls.search_memories import search_memories

TRIO = ("monologue", "monologue_trace", "monologue_soul_candidate")


@pytest.fixture
def persona(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(tmp_path / "hebbian.db")
    queue = PendingQueue(tmp_path)
    yield tmp_path, store, hebbian, queue
    store.close()
    hebbian.close()


def _mem(content, mtype, *, emotions=None, importance=None):
    return Memory.create_new(
        content=content, memory_type=mtype, domain="us", emotions=emotions or {}, importance=importance
    )


def _merge_into(target_id, merged=None):
    def _c(cand, _context):
        return Decision("merge", target_id=target_id, merged_content=merged or f"{cand.content} (merged)")

    return _c


def _run(persona, classifier):
    tmp, store, hebbian, _queue = persona
    return run_consolidation(store, persona_dir=tmp, hebbian=hebbian, classifier=classifier)


def _archive(tmp):
    path = tmp / _ARCHIVE_FILENAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _contents(store):
    return sorted(m.content for m in store.list_active())


# ------------------------------------------------------------------ helpers and constant
def test_merge_type_set_is_exactly_the_three_monologue_types():
    from brain import dev_constants

    assert dev_constants.MONOLOGUE_MERGE_TYPES == frozenset(TRIO)
    assert "monologue_emotion" not in dev_constants.MONOLOGUE_MERGE_TYPES
    assert "monologue_emotion" in dev_constants.MONOLOGUE_FAMILY_TYPES


def test_merge_type_and_matchable_types():
    from brain.engines.consolidation import matchable_types, merge_type

    assert len({merge_type(t) for t in TRIO}) == 1
    assert merge_type("monologue_emotion") not in {merge_type(t) for t in TRIO}
    assert merge_type("fact") == "fact" and merge_type("dream") == "dream"
    for t in TRIO:
        assert matchable_types(t) == frozenset(TRIO)
    assert matchable_types("observation") == frozenset({"observation", "fact"})
    assert matchable_types("fact") == frozenset({"fact"})
    assert matchable_types("dream") == frozenset({"dream"})


# ------------------------------------------------------------------ search_text memory_types
def test_search_text_memory_types_filters_in_sql_before_the_limit(persona):
    _tmp, store, _heb, _queue = persona
    wanted = _mem("harbour note wanted", "fact")
    store.create(wanted)
    for i in range(5):  # newer rows of another type would use up a LIMIT 2 if the filter ran after it
        store.create(_mem(f"harbour note other {i}", "dream"))
    hits = store.search_text("harbour", limit=2, bump=False, memory_types={"fact"})
    assert [m.id for m in hits] == [wanted.id]


def test_search_text_without_memory_types_is_unchanged(persona):
    _tmp, store, _heb, _queue = persona
    store.create(_mem("harbour a", "fact"))
    store.create(_mem("harbour b", "dream"))
    assert len(store.search_text("harbour", bump=False)) == 2
    assert len(store.search_text("harbour", bump=False, memory_types=None)) == 2
    assert len(store.search_text("harbour", bump=False, memory_types={"fact", "dream"})) == 2
    assert store.search_text("harbour", bump=False, memory_types=set()) == []


def test_search_text_with_empty_memory_types_does_not_bump(persona):
    _tmp, store, _heb, _queue = persona
    m = _mem("harbour a", "fact")
    store.create(m)
    assert store.search_text("harbour", memory_types=frozenset()) == []
    assert store.get(m.id, bump=False).recall_count == 0


# ------------------------------------------------------------------ the merge verdict
@pytest.mark.parametrize(
    "cand_type, target_type",
    [
        ("dream", "fact"),
        ("monologue_trace", "self_model_resolved"),
        ("fact", "monologue_trace"),
        ("fact", "observation"),
        ("journal_entry", "heartbeat"),
        ("observation", "dream"),
    ],
)
def test_a_merge_verdict_naming_a_row_of_another_type_stores_a_new_row(persona, cand_type, target_type):
    tmp, store, _heb, queue = persona
    target = _mem("Bob likes tea", target_type, emotions={"joy": 4.0})
    store.create(target, fold=False)
    cand = _mem("Bob takes honey in it", cand_type, emotions={"joy": 8.0, "calm": 3.0}, importance=8.0)
    queue.enqueue(cand, source="x")

    res = _run(persona, _merge_into(target.id, merged="Bob likes tea. Bob takes honey in it."))

    assert (res.merged, res.promoted) == (0, 1)
    assert _archive(tmp) == []  # no archive line for a merge that did not happen
    untouched = store.get(target.id, bump=False)
    assert untouched.content == "Bob likes tea" and untouched.emotions == {"joy": 4.0}
    stored = store.get(cand.id, bump=False)
    assert stored is not None and stored.content == "Bob takes honey in it"  # its OWN text, not the judge's merge
    assert stored.memory_type == cand_type and stored.importance == 8.0


@pytest.mark.parametrize("cand_type, target_type", [("observation", "fact"), ("monologue", "monologue_trace"),
                                                    ("monologue_soul_candidate", "monologue"),
                                                    ("monologue_trace", "monologue_soul_candidate")])
@pytest.mark.no_base_failure(reason="same-type and directed-exception merges behave as before; only the cross-type cases are new")
def test_an_allowed_pair_still_merges_and_the_row_keeps_its_type(persona, cand_type, target_type):
    tmp, store, _heb, queue = persona
    target = _mem("Bob likes tea", target_type)
    store.create(target)
    cand = _mem("Bob takes honey in it", cand_type)
    queue.enqueue(cand, source="x")

    res = _run(persona, _merge_into(target.id, merged="Bob likes tea with honey"))

    assert (res.merged, res.promoted) == (1, 0)
    row = store.get(target.id, bump=False)
    assert row.content == "Bob likes tea with honey" and row.memory_type == target_type
    assert store.get(cand.id, bump=False) is None
    assert len(_archive(tmp)) == 1


@pytest.mark.no_base_failure(reason="pins the spec's order: the fading deferral comes before the type check, so a fading target defers as it does on base")
def test_a_wrong_type_target_that_is_fading_still_defers(persona):
    """The spec puts the fading deferral first: a fading target of another type is re-judged on the next tick."""
    tmp, store, _heb, queue = persona
    target = _mem("Bob likes tea", "dream")
    store.create(target)
    store._conn.execute("UPDATE memories SET state = 'fading' WHERE id = ?", (target.id,))
    store._conn.commit()
    queue.enqueue(_mem("Bob takes honey in it", "fact"), source="x")
    res = _run(persona, _merge_into(target.id))
    assert (res.deferred, res.promoted, res.merged) == (1, 0, 0)


def test_a_wrong_type_merge_does_not_change_the_targets_feelings(persona, emotion_engine_stub):
    tmp, store, _heb, queue = persona
    emotion_engine_stub.live = True  # a live engine is when a merge writes feelings
    target = _mem("Bob likes tea", "self_model_resolved", emotions={"joy": 2.0})
    store.create(target, fold=False)
    cand = _mem("Bob dreamt of tea", "monologue_trace", emotions={"joy": 10.0, "fear": 9.0})
    queue.enqueue(cand, source="x")
    emotion_engine_stub.recorded_calls.clear()
    _run(persona, _merge_into(target.id))
    assert store.get(target.id, bump=False).emotions == {"joy": 2.0}
    # The override stores the candidate as a row: its one commit event comes from `create`, none from the merge path.
    assert [c[0] for c in emotion_engine_stub.recorded_calls].count("on_memory_committed") == 1
    assert store.get(cand.id, bump=False).emotions == {"joy": 10.0, "fear": 9.0}


def test_the_archive_line_records_the_candidates_type(persona):
    tmp, store, _heb, queue = persona
    target = _mem("Bob likes tea", "fact")
    store.create(target)
    queue.enqueue(_mem("Bob takes honey in it", "observation"), source="x")
    _run(persona, _merge_into(target.id))
    (line,) = _archive(tmp)
    assert line["merged_from_candidate_type"] == "observation"
    assert line["target"]["memory_type"] == "fact" and line["reason"] == "consolidation_merge"


def test_archive_preimage_requires_the_candidate_type(tmp_path):
    target = _mem("Bob likes tea", "fact")
    with pytest.raises(TypeError):
        consolidation._archive_preimage(tmp_path, target, "cand-id")  # type: ignore[call-arg]
    consolidation._archive_preimage(tmp_path, target, "cand-id", "dream")
    assert _archive(tmp_path)[0]["merged_from_candidate_type"] == "dream"


@pytest.mark.no_base_failure(reason="the only reader already ignores keys it does not know; this pins that a line without the new key (every old line) stays readable")
def test_a_line_without_the_candidate_type_is_read_without_error(tmp_path):
    from brain.soul.candidates import merge_target_map

    old = {"reason": "consolidation_merge", "target": {"id": "t1"}, "merged_from_candidate": "c1"}
    new = {**old, "target": {"id": "t2"}, "merged_from_candidate": "c2", "merged_from_candidate_type": "fact"}
    (tmp_path / _ARCHIVE_FILENAME).write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n")
    assert merge_target_map(tmp_path) == {"c1": {"t1"}, "c2": {"t2"}}


@pytest.mark.no_base_failure(reason="pins that dedup_sweep's own archive line already carries the removed row's memory_type (S15); no sweep code changes here")
def test_a_dedup_sweep_archive_line_carries_the_removed_rows_type(tmp_path):
    from brain.engines import dedup_sweep

    row = _mem("Bob likes tea", "observation")
    dedup_sweep._archive_preimage(tmp_path, row)
    (line,) = _archive(tmp_path)
    assert line["target"]["memory_type"] == "observation"


# ------------------------------------------------------------------ the judge's context
def _capture_context(seen):
    def _c(cand, context):
        seen[cand.content] = [m.memory_type for m in context]
        return Decision("new")

    return _c


def _pool_patch(monkeypatch, store, cand_vec, vec_for):
    monkeypatch.setattr(consolidation, "_embed_candidate", lambda _content: (cand_vec, "fake-model"))
    monkeypatch.setattr(consolidation, "build_embedding_matrix", lambda _path: object())
    monkeypatch.setattr(
        consolidation,
        "build_semantic_candidate_pool",
        lambda s, _matrix: {m.id: (m, vec_for(m)) for m in s.list_active()},
    )


def test_cosine_context_holds_only_types_the_candidate_may_match(persona, monkeypatch):
    tmp, store, _heb, queue = persona
    for i in range(9):  # nine closer rows of other types would fill the top 8 before a post-hoc filter
        store.create(_mem(f"Bob closer note {i}", "dream"))
    store.create(_mem("Bob far fact", "fact"))
    store.create(_mem("Bob far observation", "observation"))
    cand_vec = np.array([1.0, 0.0, 0.0])
    _pool_patch(monkeypatch, store, cand_vec, lambda m: np.array([1.0, 0.0, 0.0] if m.memory_type == "dream" else [1.0, 1.0, 0.0]))
    queue.enqueue(_mem("Bob says a new fact", "fact"), source="x")
    queue.enqueue(_mem("Bob dreamt a new thing", "dream"), source="x")
    queue.enqueue(_mem("Bob observed a thing", "observation"), source="x")
    seen: dict = {}
    _run(persona, _capture_context(seen))
    assert seen["Bob says a new fact"] == ["fact"]  # not the far observation, not a dream
    assert set(seen["Bob dreamt a new thing"]) == {"dream"} and len(seen["Bob dreamt a new thing"]) == 8
    # An observation also sees facts (the far one and the fact stored earlier in this run) and observations, and no dream.
    assert set(seen["Bob observed a thing"]) == {"fact", "observation"}


def test_lexical_context_holds_only_types_the_candidate_may_match(persona, monkeypatch):
    tmp, store, _heb, queue = persona
    monkeypatch.setattr(consolidation, "_embed_candidate", lambda _content: None)  # the lexical fallback
    for i in range(4):
        store.create(_mem(f"harbour tide table {i}", "dream"))
    store.create(_mem("harbour tide fact", "fact"))
    store.create(_mem("harbour tide watch", "observation"))
    queue.enqueue(_mem("harbour tide says something new", "fact"), source="x")
    queue.enqueue(_mem("harbour tide seems notable", "observation"), source="x")
    seen: dict = {}
    _run(persona, _capture_context(seen))
    assert seen["harbour tide says something new"] == ["fact"]
    # The fact stored earlier in this run is a fact row too.
    assert set(seen["harbour tide seems notable"]) == {"fact", "observation"}


def test_when_the_filter_empties_the_context_the_candidate_is_stored_as_new(persona, monkeypatch):
    tmp, store, _heb, queue = persona
    monkeypatch.setattr(consolidation, "_embed_candidate", lambda _content: None)
    store.create(_mem("harbour tide table", "dream"))
    queue.enqueue(_mem("harbour tide fact", "fact"), source="x")
    seen: dict = {}
    res = _run(persona, _capture_context(seen))
    assert seen["harbour tide fact"] == [] and res.promoted == 1


# ------------------------------------------------------------------ pass-1 exact-duplicate drop
def test_an_exact_duplicate_of_another_type_in_the_batch_is_not_dropped(persona):
    tmp, store, _heb, queue = persona
    queue.enqueue(_mem("Bob likes tea", "fact"), source="x")
    queue.enqueue(_mem("Bob likes tea", "dream"), source="x")
    res = _run(persona, lambda c, ctx: Decision("new"))
    assert res.exact_dropped == 0 and sorted(m.memory_type for m in store.list_active()) == ["dream", "fact"]


def test_an_exact_duplicate_of_another_type_already_stored_is_not_dropped(persona):
    tmp, store, _heb, queue = persona
    store.create(_mem("Bob likes tea", "conversation"))
    queue.enqueue(_mem("Bob likes tea", "dream"), source="x")
    res = _run(persona, lambda c, ctx: Decision("new"))
    assert res.exact_dropped == 0 and store.count(active_only=False) == 2


@pytest.mark.no_base_failure(reason="same-type exact duplicates are dropped as before; only the cross-type cases are new")
def test_an_exact_duplicate_of_the_same_type_is_still_dropped(persona):
    tmp, store, _heb, queue = persona
    store.create(_mem("Bob likes tea", "dream"))
    queue.enqueue(_mem("Bob likes tea", "dream"), source="x")
    queue.enqueue(_mem("Bob likes coffee", "fact"), source="x")
    queue.enqueue(_mem("bob  likes COFFEE", "fact"), source="x")
    res = _run(persona, lambda c, ctx: Decision("new"))
    assert res.exact_dropped == 2 and store.count(active_only=False) == 2


@pytest.mark.no_base_failure(reason="the type-blind drop on base already drops these; this pins that the three monologue types stay one type for the drop")
def test_the_three_monologue_types_are_one_type_for_the_exact_drop(persona):
    tmp, store, _heb, queue = persona
    store.create(_mem("Bob wonders about tea", "monologue_trace"))
    queue.enqueue(_mem("Bob wonders about tea", "monologue_soul_candidate"), source="x")
    queue.enqueue(_mem("Bob wonders about rain", "monologue"), source="x")
    queue.enqueue(_mem("Bob wonders about rain", "monologue_trace"), source="x")
    res = _run(persona, lambda c, ctx: Decision("new"))
    assert res.exact_dropped == 2


def test_exact_drop_observation_against_fact_but_not_the_reverse_in_the_batch(persona):
    tmp, store, _heb, queue = persona
    queue.enqueue(_mem("Bob likes tea", "fact"), source="x")
    queue.enqueue(_mem("Bob likes tea", "observation"), source="x")  # dropped: the same text was seen as a fact
    queue.enqueue(_mem("Bob likes rain", "observation"), source="x")
    queue.enqueue(_mem("Bob likes rain", "fact"), source="x")  # kept: a fact is not dropped for an earlier observation
    res = _run(persona, lambda c, ctx: Decision("new"))
    assert res.exact_dropped == 1
    assert sorted((m.content, m.memory_type) for m in store.list_active()) == [
        ("Bob likes rain", "fact"),
        ("Bob likes rain", "observation"),
        ("Bob likes tea", "fact"),
    ]


def test_exact_drop_observation_against_a_stored_fact_but_not_the_reverse(persona):
    tmp, store, _heb, queue = persona
    store.create(_mem("Bob likes tea", "fact"))
    store.create(_mem("Bob likes rain", "observation"))
    queue.enqueue(_mem("Bob likes tea", "observation"), source="x")  # dropped: a fact twin is stored
    queue.enqueue(_mem("Bob likes rain", "fact"), source="x")  # kept: only an observation twin is stored
    res = _run(persona, lambda c, ctx: Decision("new"))
    assert res.exact_dropped == 1
    assert sorted((m.content, m.memory_type) for m in store.list_active()) == [
        ("Bob likes rain", "fact"),
        ("Bob likes rain", "observation"),
        ("Bob likes tea", "fact"),
    ]


# ------------------------------------------------------------------ the issue's acceptance test
def test_a_genuine_fact_is_not_merged_into_a_monologue_trace_row_and_search_finds_it(persona):
    from brain.memory.relevance import CANDIDATE_POOL

    tmp, store, hebbian, queue = persona
    trace = _mem("Bob wondered about the quokka tide ledger", "monologue_trace", importance=3.0)
    store.create(trace)
    for i in range(CANDIDATE_POOL + 10):  # more genuine rows match the query's words than the keyword pool holds
        store.create(_mem(f"quokka entry {i} on the harbour wall", "conversation", importance=5.0))
    fact = _mem("Bob was given the ishihara quokka colour test", "fact", importance=8.0)
    queue.enqueue(fact, source="x")

    res = _run(persona, _merge_into(trace.id, merged=f"{trace.content}\n{fact.content}"))

    assert (res.merged, res.promoted) == (0, 1)
    assert store.get(trace.id, bump=False).content == trace.content
    out = search_memories("ishihara quokka", limit=5, mode="lexical", store=store, hebbian=hebbian, persona_dir=tmp)
    assert fact.id in [m["id"] for m in out["memories"]]
