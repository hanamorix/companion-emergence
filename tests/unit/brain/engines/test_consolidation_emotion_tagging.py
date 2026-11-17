"""#747: the consolidation gate tags a memory it writes without feelings (the emotion backfill moved into the temp gate).

Three writes pass through the gate: a promoted candidate, a merge target and a recalled row (the write-back). Each is tagged
when it has no stored emotions. The tagger is injected (`emotion_tagger=`), so no model is called. Synthetic content only
(synthetic user Bob, persona Canary).
"""

from __future__ import annotations

import itertools
import logging

import pytest

from brain.bridge.cli_throttle import ThrottleDeferred
from brain.engines.consolidation import Decision, Reappraisal, run_consolidation
from brain.health.json_io import read_jsonl
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore
from brain.utils import log_throttle

_N = itertools.count()
_PROCESSING = "pending_candidates.processing.jsonl"
_SETTLED = "pending_candidates.settled.jsonl"


@pytest.fixture
def persona(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(tmp_path / "hebbian.db")
    log_throttle._last.clear()  # the failure log warns once per interval: start each test with a fresh key
    yield tmp_path, store, hebbian, PendingQueue(tmp_path)
    store.close()
    hebbian.close()


class _Tagger:
    """A scripted tagger: records the memories it is asked about and returns ``tags`` or raises ``error``."""

    def __init__(self, tags=None, error=None, before=None):
        self.tags = {"joy": 6.0} if tags is None else tags
        self.error = error
        self.before = before  # called with the memory first (a probe of the state at the moment of the call)
        self.seen: list[Memory] = []

    def __call__(self, memory):
        self.seen.append(memory)
        if self.before is not None:
            self.before(memory)
        if self.error is not None:
            raise self.error
        return dict(self.tags)


def _mem(content, *, emotions=None, mtype="conversation", importance=None):
    return Memory.create_new(
        content=f"{content} {next(_N)}", memory_type=mtype, domain="us", emotions=emotions or {}, importance=importance
    )


def _gate(persona, cand, decision, tagger, **kw):
    tmp, store, hebbian, queue = persona
    queue.enqueue(cand, source="x")
    return run_consolidation(
        store,
        persona_dir=tmp,
        hebbian=hebbian,
        classifier=lambda c, _ctx: decision,
        emotion_tagger=tagger,
        **kw,
    )


def _recall(persona, memory, tagger, *, reappraiser=None):
    tmp, store, _heb, queue = persona
    queue.enqueue_reappraisal(memory.id, source="recall")
    return run_consolidation(
        store,
        persona_dir=tmp,
        reappraiser=reappraiser or (lambda _m: Reappraisal(7.0)),
        emotion_tagger=tagger,
    )


def _emotions(store, memory_id):
    return store.get(memory_id, bump=False)._stored_emotions


def _unsettled(tmp):
    processing = tmp / _PROCESSING
    if not processing.exists():
        return []
    settled = tmp / _SETTLED
    done = {r["id"] for r in read_jsonl(settled)} if settled.exists() else set()
    return [e["content"] for e in read_jsonl(processing) if e["_claim_id"] not in done]


# --- promote ------------------------------------------------------------------------------------------------------------


def test_a_promoted_row_without_feelings_is_tagged_after_its_embedding_and_its_edge(persona, emotion_engine_stub, monkeypatch):
    _tmp, store, _heb, _queue = persona
    existing = _mem("Bob's first cup of tea", emotions={"joy": 1.0})
    store.create(existing, fold=False)
    order: list[str] = []
    real_embed = MemoryStore.write_embedding
    monkeypatch.setattr(MemoryStore, "write_embedding", lambda self, *a, **k: (order.append("embedding"), real_embed(self, *a, **k))[1])
    real_edge = HebbianMatrix.set_edge_weight
    monkeypatch.setattr(HebbianMatrix, "set_edge_weight", lambda self, *a, **k: (order.append("edge"), real_edge(self, *a, **k))[1])
    cand = _mem("Bob's second cup of tea")
    tagger = _Tagger(before=lambda _m: order.append("tag"))

    res = _gate(persona, cand, Decision("continuation", target_id=existing.id), tagger)

    assert res.continuations == 1 and res.failed == 0
    assert order == ["embedding", "edge", "tag"]
    assert [m.id for m in tagger.seen] == [cand.id]
    assert _emotions(store, cand.id) == {"joy": 6.0}
    assert [c[0] for c in emotion_engine_stub.recorded_calls if c[0] == "on_emotions_assigned"] == ["on_emotions_assigned"]


@pytest.mark.parametrize("mtype", ["conversation", "dream", "meta"])
def test_any_memory_type_is_tagged(persona, mtype):
    _tmp, store, _heb, _queue = persona
    cand = _mem("Canary woke with a quiet idea", mtype=mtype)
    _gate(persona, cand, Decision("new"), _Tagger())
    assert _emotions(store, cand.id) == {"joy": 6.0}


# --- merge --------------------------------------------------------------------------------------------------------------


def _merge(persona, *, target_emotions=None, cand_emotions=None, tagger):
    _tmp, store, _heb, _queue = persona
    target = _mem("Bob likes tea", emotions=target_emotions)
    store.create(target, fold=False)
    cand = _mem("Bob likes tea with honey", emotions=cand_emotions)
    res = _gate(persona, cand, Decision("merge", target_id=target.id, merged_content="Bob likes tea with honey"), tagger)
    return target, res


def test_a_merged_target_without_feelings_is_tagged_after_the_merge_and_read_without_a_recall_bump(persona):
    _tmp, store, _heb, _queue = persona
    tagger = _Tagger()
    target, res = _merge(persona, tagger=tagger)  # engine not live: the merge writes the content only
    assert res.merged == 1 and res.failed == 0
    assert [m.content for m in tagger.seen] == ["Bob likes tea with honey"]  # the target as the merge left it
    got = store.get(target.id, bump=False)
    assert got._stored_emotions == {"joy": 6.0}
    assert got.recall_count == 0.0 and got.last_accessed_at is None


def test_a_live_merge_that_leaves_the_target_with_feelings_makes_no_tagger_call(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    tagger = _Tagger()
    target, res = _merge(persona, cand_emotions={"trust": 5.0}, tagger=tagger)
    assert res.merged == 1
    assert tagger.seen == []
    assert _emotions(store, target.id) == {"trust": 5.0}


def test_a_target_gone_after_the_merge_write_is_left_alone(persona, monkeypatch):
    _tmp, store, _heb, _queue = persona
    real_merge = MemoryStore.merge_into

    def merge_then_vanish(self, target, *a, **k):
        real_merge(self, target, *a, **k)
        self._conn.execute("DELETE FROM memories WHERE id = ?", (target.id,))
        self._conn.commit()

    monkeypatch.setattr(MemoryStore, "merge_into", merge_then_vanish)
    tagger = _Tagger()
    _target, res = _merge(persona, tagger=tagger)
    assert res.merged == 1 and res.failed == 0
    assert tagger.seen == []


# --- recalled row (the write-back) --------------------------------------------------------------------------------------


def test_a_recalled_row_without_feelings_is_tagged_before_the_write_back_re_reads_it(persona, emotion_engine_stub, monkeypatch):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    row = _mem("Bob's address", importance=3.0)
    store.create(row, fold=False)
    refreshed: list[dict] = []
    real_refresh = MemoryStore.refresh_faded_emotions
    monkeypatch.setattr(
        MemoryStore, "refresh_faded_emotions", lambda self, mid, faded, at: (refreshed.append(dict(faded)), real_refresh(self, mid, faded, at))[1]
    )
    tagger = _Tagger({"joy": 6.0, "trust": 3.0})

    res = _recall(persona, row, tagger)

    assert res.reappraised == 1 and res.failed == 0
    assert [m.id for m in tagger.seen] == [row.id]
    assert refreshed == [{"joy": 6.0, "trust": 3.0}]  # the refresh carries the tags
    assert _emotions(store, row.id) == {"joy": 6.0, "trust": 3.0}


def test_a_recalled_row_of_a_persona_whose_engine_is_not_live_is_tagged(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = False
    row = _mem("Bob's address", importance=3.0)
    store.create(row, fold=False)
    created = store.get(row.id, bump=False).created_at

    res = _recall(persona, row, _Tagger())

    assert res.reappraised == 1
    got = store.get(row.id, bump=False)
    assert got._stored_emotions == {"joy": 6.0}
    assert got.created_at == created  # not live: no write-back, only the tag


def test_the_recall_write_back_still_runs_when_the_tagger_fails(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    row = _mem("Bob's address", emotions={"joy": 8.0}, importance=3.0)
    store.create(row, fold=False)
    bare = _mem("Canary's window", importance=3.0)
    store.create(bare, fold=False)
    before = store.get(row.id, bump=False).created_at

    _recall(persona, row, _Tagger(error=RuntimeError("boom")))
    res = _recall(persona, bare, _Tagger(error=RuntimeError("boom")))

    assert res.reappraised == 1 and res.failed == 0
    assert store.get(row.id, bump=False).created_at > before  # the row with feelings was written back as usual
    assert _emotions(store, bare.id) == {}


# --- no tagger call, no write -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["promote", "merge", "recall"])
def test_a_row_with_feelings_is_not_sent_to_the_tagger(persona, emotion_engine_stub, path):
    tmp, store, _heb, _queue = persona
    tagger = _Tagger()
    if path == "promote":
        _gate(persona, _mem("Bob's tea", emotions={"awe": 4.0}), Decision("new"), tagger)
    elif path == "merge":
        _merge(persona, target_emotions={"awe": 4.0}, tagger=tagger)
    else:
        row = _mem("Bob's address", emotions={"awe": 4.0}, importance=3.0)
        store.create(row, fold=False)
        _recall(persona, row, tagger)
    assert tagger.seen == []


@pytest.mark.parametrize("path", ["promote", "merge", "recall"])
def test_a_tagger_that_returns_nothing_leaves_the_row_as_it_is(persona, monkeypatch, path):
    tmp, store, _heb, _queue = persona
    written: list[dict] = []
    real_update = MemoryStore.update
    monkeypatch.setattr(
        MemoryStore, "update", lambda self, mid, **f: (written.append(f) if "emotions" in f else None, real_update(self, mid, **f))[1]
    )
    tagger = _Tagger({})
    if path == "promote":
        cand = _mem("Bob's tea")
        _gate(persona, cand, Decision("new"), tagger)
        target_id = cand.id
    elif path == "merge":
        target, _res = _merge(persona, tagger=tagger)
        target_id = target.id
    else:
        row = _mem("Bob's address", importance=3.0)
        store.create(row, fold=False)
        _recall(persona, row, tagger)
        target_id = row.id
    assert len(tagger.seen) == 1
    assert written == []
    assert _emotions(store, target_id) == {}


def test_without_a_provider_or_a_tagger_no_model_call_is_made(persona):
    tmp, store, hebbian, queue = persona
    cand = _mem("Bob's tea")
    queue.enqueue(cand, source="x")
    res = run_consolidation(store, persona_dir=tmp, hebbian=hebbian, classifier=lambda c, _ctx: Decision("new"))
    assert res.promoted == 1
    assert _emotions(store, cand.id) == {}


def test_a_provider_builds_the_tagger_and_its_reply_tags_the_row(persona):
    tmp, store, hebbian, queue = persona

    class Provider:
        def __init__(self):
            self.prompts: list[str] = []

        def generate(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return '```json\n{"joy": 5}\n```'

    provider = Provider()
    cand = _mem("Bob's tea")
    queue.enqueue(cand, source="x")
    run_consolidation(store, persona_dir=tmp, hebbian=hebbian, classifier=lambda c, _ctx: Decision("new"), provider=provider)
    assert len(provider.prompts) == 1 and cand.content in provider.prompts[0]
    assert _emotions(store, cand.id) == {"joy": 5.0}


# --- failures -----------------------------------------------------------------------------------------------------------


_FAILURES = [
    pytest.param(lambda: _Tagger(error=ThrottleDeferred("not now")), id="deferred-call"),
    pytest.param(lambda: _Tagger(error=RuntimeError("boom")), id="tagger-error"),
]


@pytest.mark.parametrize("make_tagger", _FAILURES)
@pytest.mark.parametrize("path", ["promote", "merge", "recall"])
def test_a_tagging_failure_leaves_the_row_untagged_logs_and_settles_the_item_once(persona, caplog, make_tagger, path):
    tmp, store, _heb, _queue = persona
    tagger = make_tagger()
    with caplog.at_level(logging.DEBUG, logger="brain.engines.consolidation"):
        if path == "promote":
            cand = _mem("Bob's tea")
            res = _gate(persona, cand, Decision("new"), tagger)
            target_id, counted = cand.id, res.promoted
        elif path == "merge":
            target, res = _merge(persona, tagger=tagger)
            target_id, counted = target.id, res.merged
        else:
            row = _mem("Bob's address", importance=3.0)
            store.create(row, fold=False)
            res = _recall(persona, row, tagger)
            target_id, counted = row.id, res.reappraised
    assert counted == 1 and res.failed == 0
    assert _emotions(store, target_id) == {}
    assert _unsettled(tmp) == []  # the item is settled, not kept for a re-judge, a re-merge or a re-fade
    assert len(tagger.seen) == 1
    logged = [r for r in caplog.records if "emotion tagging failed" in r.getMessage()]
    assert len(logged) == 1 and not logged[0].exc_info  # no traceback: `warn_throttled(..., exc_info=False)`
    assert type(tagger.error).__name__ in logged[0].getMessage()


def test_a_store_update_failure_while_writing_the_tag_is_logged_and_the_item_settled(persona, caplog, monkeypatch):
    tmp, store, _heb, _queue = persona
    real_update = MemoryStore.update

    def failing_update(self, mid, **fields):
        if "emotions" in fields:
            raise RuntimeError("database is locked")
        return real_update(self, mid, **fields)

    monkeypatch.setattr(MemoryStore, "update", failing_update)
    cand = _mem("Bob's tea")
    with caplog.at_level(logging.DEBUG, logger="brain.engines.consolidation"):
        res = _gate(persona, cand, Decision("new"), _Tagger())
    assert res.promoted == 1 and res.failed == 0
    assert store.get(cand.id, bump=False) is not None
    assert _emotions(store, cand.id) == {}
    assert _unsettled(tmp) == []
    assert any("database is locked" in r.getMessage() for r in caplog.records)


def test_a_deferred_tagger_call_does_not_stop_the_pass_but_the_judges_next_deferred_call_does(persona):
    tmp, store, hebbian, queue = persona
    first, second = _mem("alpha item"), _mem("bravo item")
    queue.enqueue(first, source="x")
    queue.enqueue(second, source="x")

    def classifier(cand, _ctx):
        if cand.id == second.id:
            raise ThrottleDeferred("not now")
        return Decision("new")

    tagger = _Tagger(error=ThrottleDeferred("not now"))
    run_consolidation(store, persona_dir=tmp, hebbian=hebbian, classifier=classifier, emotion_tagger=tagger)

    assert store.get(first.id, bump=False) is not None  # promoted, settled once, untagged
    assert _emotions(store, first.id) == {}
    assert store.get(second.id, bump=False) is None  # never promoted unclassified
    assert _unsettled(tmp) == [second.content]  # kept for the next run


# --- feelings that land during a slow call are kept ---------------------------------------------------------------------


def test_feelings_written_during_the_tagger_call_are_kept_on_a_promoted_row(persona):
    _tmp, store, _heb, _queue = persona

    def other_writer(memory):
        store.update(memory.id, emotions={"awe": 2.0}, fold=False)

    cand = _mem("Bob's tea")
    _gate(persona, cand, Decision("new"), _Tagger({"joy": 9.0}, before=other_writer))
    assert _emotions(store, cand.id) == {"awe": 2.0}


def test_feelings_written_during_the_reappraiser_call_are_kept_when_a_tagger_is_present(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    row = _mem("Bob's address", importance=3.0)  # loaded with no emotions
    store.create(row, fold=False)

    def reappraiser(m):
        assert m._stored_emotions == {}  # the gate holds the stale row
        store.update(m.id, emotions={"awe": 6.0}, fold=False)  # another writer lands meanwhile
        return Reappraisal(7.0)

    tagger = _Tagger({"joy": 9.0})
    res = _recall(persona, row, tagger, reappraiser=reappraiser)
    assert res.reappraised == 1
    assert _emotions(store, row.id) == {"awe": 6.0}  # neither replaced by the tag nor erased by a fade of the stale row
