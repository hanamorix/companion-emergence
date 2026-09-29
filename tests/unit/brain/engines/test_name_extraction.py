"""N2 (name-recall fix): the gate judge and the on-recall re-appraiser also return names.

Spec section 5, "Names are added three ways", items 1 and 2 (S45, S51, S70). Criteria
C6 (gate judge and re-appraiser halves), plan P-17 and P-18. Fake providers only: no
Haiku call, no model load. The tool (item 3) is tested in
``tests/unit/brain/tools/test_add_known_name.py``.

Naming: synthetic user = Bob, persona = Canary. Every name below is invented.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from brain.engines import consolidation as cons
from brain.engines.consolidation import (
    Decision,
    Reappraisal,
    _make_haiku_classifier,
    _make_haiku_reappraiser,
    _noop_reappraiser,
    _parse_names,
    _parse_reappraisal,
    run_consolidation,
)
from brain.memory import known_names as kn
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore

_NEW_VERDICTS = ["merge", "distinct", "correction", "continuation", "new"]


class FakeProvider:
    """A scripted generation provider that records every call."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt, *, system=None, **_kw):
        self.calls.append((prompt, system))
        reply = self.reply(prompt) if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def persona(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(tmp_path / "hebbian.db")
    queue = PendingQueue(tmp_path)
    yield tmp_path, store, hebbian, queue
    store.close()
    hebbian.close()


def _mem(content, mtype="dream", *, importance=None):
    return Memory.create_new(content=content, memory_type=mtype, domain="us", importance=importance)


def _rows(persona_dir: Path) -> list[tuple[str, str, str]]:
    path = kn.known_names_path(persona_dir)
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(
            "SELECT name_lower, display, source FROM known_names ORDER BY name_lower"
        ).fetchall()
    finally:
        conn.close()


def _long_text(marker: str) -> str:
    """A 1,600-character text whose marker word sits far beyond the old 400 cut."""
    return ("filler words " * 110) + marker + (" more filler" * 10)


# ===========================================================================
# Gate judge (item 1): names on every verdict except "duplicate"
# ===========================================================================


def _gate(tmp, store, hebbian, provider):
    return run_consolidation(store, persona_dir=tmp, hebbian=hebbian, provider=provider)


@pytest.mark.parametrize("verdict", _NEW_VERDICTS)
def test_judge_names_written_for_every_verdict_but_duplicate(persona, verdict):
    """C6: names are added (source `gate`, display form intact) for merge, distinct,
    correction, continuation and new; the display keeps its case."""
    tmp, store, hebbian, queue = persona
    target = store.create(_mem("Bob's older sister is away", "conversation"))
    queue.enqueue(_mem("Bob's dog Pretzel sat with Wren", "dream"), source="x")
    reply = json.dumps({"verdict": verdict, "target_id": target, "names": ["Pretzel", "Wren"]})
    _gate(tmp, store, hebbian, FakeProvider(reply))
    assert _rows(tmp) == [("pretzel", "Pretzel", "gate"), ("wren", "Wren", "gate")]


def test_judge_duplicate_adds_no_names(persona):
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("Bob's dog Pretzel", "dream"), source="x")
    reply = json.dumps({"verdict": "duplicate", "names": ["Pretzel"]})
    res = _gate(tmp, store, hebbian, FakeProvider(reply))
    assert res.duplicates == 1
    assert _rows(tmp) == []
    assert not kn.known_names_path(tmp).exists()  # nothing written, file never created


def test_injected_decision_names_follow_the_same_rule(persona):
    """The rule lives in `_dispatch`, so an injected classifier gets it too: a
    duplicate Decision that carries names writes none; a new one writes them."""
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("dup with Zed", "dream"), source="x")
    queue.enqueue(_mem("fresh with Wren", "dream"), source="x")

    def _classifier(cand, _ctx):
        if "dup" in cand.content:
            return Decision("duplicate", names=("Zed",))
        return Decision("new", names=("Wren",))

    run_consolidation(store, persona_dir=tmp, hebbian=hebbian, classifier=_classifier)
    assert [r[0] for r in _rows(tmp)] == ["wren"]


def test_deferred_merge_writes_names_and_a_repeat_is_harmless(persona):
    """P-17: a merge `_dispatch` defers (target missing) still adds the candidate's
    names; the re-judged tick adds them again as a no-op (one row, first source kept)."""
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("orphan merge about Wren", "dream"), source="x")
    reply = json.dumps({"verdict": "merge", "target_id": "nonexistent", "names": ["Wren"]})
    first = _gate(tmp, store, hebbian, FakeProvider(reply))
    assert first.deferred == 1
    assert _rows(tmp) == [("wren", "Wren", "gate")]
    second = _gate(tmp, store, hebbian, FakeProvider(reply))  # the deferred candidate again
    assert second.deferred == 1
    assert _rows(tmp) == [("wren", "Wren", "gate")]


def test_judge_prompt_carries_the_whole_candidate(persona):
    """C6: a 1,000+ character candidate reaches the judge untruncated (the first-400
    cut is gone). Able to fail: a `[:400]` cut drops the marker."""
    tmp, store, hebbian, queue = persona
    text = _long_text("ENDMARKER")
    assert len(text) > 1000 and text.index("ENDMARKER") > 1000
    queue.enqueue(_mem(text, "dream"), source="x")
    provider = FakeProvider(json.dumps({"verdict": "new", "names": []}))
    _gate(tmp, store, hebbian, provider)
    assert len(provider.calls) == 1  # one call per candidate, verdict and names together
    prompt, system = provider.calls[0]
    assert text in prompt
    assert "ENDMARKER" in prompt
    assert "names" in system  # the revised prompt asks for them


@pytest.mark.parametrize(
    "names_value, expected",
    [
        ("Wren", []),  # a bare string is not a list: ignored, never iterated into letters
        ({"a": "Wren"}, []),
        (7, []),
        (None, []),
        ([], []),
        ([1, None, "", "   ", "Wren"], ["wren"]),
        (["Wren", "Wren", "wren"], ["wren"]),
    ],
)
def test_judge_malformed_names_never_break_the_verdict(persona, names_value, expected):
    """Fail-soft: whatever `names` holds, the candidate is still promoted."""
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("Bob and Wren walked", "dream"), source="x")
    reply = json.dumps({"verdict": "new", "names": names_value})
    res = _gate(tmp, store, hebbian, FakeProvider(reply))
    assert res.promoted == 1
    assert store.count(active_only=False) == 1
    assert [r[0] for r in _rows(tmp)] == expected


def test_judge_absent_names_key_is_a_plain_verdict(persona):
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("Bob walked", "dream"), source="x")
    res = _gate(tmp, store, hebbian, FakeProvider(json.dumps({"verdict": "new"})))
    assert res.promoted == 1
    assert _rows(tmp) == []


def test_judge_unparseable_reply_still_promotes_with_no_names(persona):
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("Bob walked", "dream"), source="x")
    res = _gate(tmp, store, hebbian, FakeProvider("no json here at all"))
    assert res.promoted == 1
    assert _rows(tmp) == []


def test_judge_unknown_verdict_falls_to_new_and_still_writes_names(persona):
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("Bob and Wren", "dream"), source="x")
    reply = json.dumps({"verdict": "sideways", "names": ["Wren"]})
    res = _gate(tmp, store, hebbian, FakeProvider(reply))
    assert res.promoted == 1
    assert [r[0] for r in _rows(tmp)] == ["wren"]


def test_gate_names_go_through_the_admission_function(persona):
    """S70: a recall-stopword entry is rejected with nothing written; the rest are
    admitted as extracted."""
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("Will and Grace and the Hague", "dream"), source="x")
    reply = json.dumps({"verdict": "new", "names": ["will", "Grace", "the", "the Hague"]})
    _gate(tmp, store, hebbian, FakeProvider(reply))
    assert [r[0] for r in _rows(tmp)] == ["grace", "the hague"]


def test_gate_writer_calls_admit_names_with_source_gate(persona, monkeypatch):
    """Guard against a writer that bypasses `admit_names`: the only write entry point
    is called, with source `gate`."""
    tmp, store, hebbian, queue = persona
    seen = []

    def _spy(persona_dir, names, source, **kw):
        seen.append((Path(persona_dir), list(names), source))
        return []

    monkeypatch.setattr(cons, "admit_names", _spy)
    queue.enqueue(_mem("Bob and Wren", "dream"), source="x")
    _gate(tmp, store, hebbian, FakeProvider(json.dumps({"verdict": "new", "names": ["Wren"]})))
    assert seen == [(tmp, ["Wren"], "gate")]


@pytest.mark.parametrize("exc", [OSError("disk gone"), RuntimeError("boom")])
def test_gate_names_write_failure_never_loses_the_candidate(persona, monkeypatch, exc, caplog):
    """P-17: a names-write failure is logged and the candidate is still promoted."""
    tmp, store, hebbian, queue = persona

    def _boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(cons, "admit_names", _boom)
    queue.enqueue(_mem("Bob and Wren", "dream"), source="x")
    with caplog.at_level(logging.WARNING, logger="brain.engines.consolidation"):
        res = _gate(
            tmp, store, hebbian, FakeProvider(json.dumps({"verdict": "new", "names": ["Wren"]}))
        )
    assert res.promoted == 1 and store.count(active_only=False) == 1
    assert any("known-names write failed" in r.getMessage() for r in caplog.records)


def test_classifier_reads_whole_candidate_directly(persona):
    """The Haiku classifier itself (no queue): the user prompt holds all of the candidate."""
    tmp, store, hebbian, queue = persona
    provider = FakeProvider(json.dumps({"verdict": "new", "names": ["Wren"]}))
    text = _long_text("TAILWORD")
    decision = _make_haiku_classifier(provider)(_mem(text), [])
    assert "TAILWORD" in provider.calls[0][0]
    assert decision == Decision("new", names=("Wren",))


# ===========================================================================
# Re-appraiser (item 2): number + names, whole memory
# ===========================================================================


def _reappraise_run(store, tmp, mem, provider=None, reappraiser=None):
    PendingQueue(tmp).enqueue_reappraisal(mem.id, source="recall")
    return run_consolidation(
        store,
        persona_dir=tmp,
        provider=provider,
        reappraiser=reappraiser,
        classifier=lambda c, x: Decision("new"),
    )


def test_reappraiser_updates_importance_and_adds_names(persona):
    """C6: the re-appraiser returns number + names; importance is updated AND names are
    added (source `reappraiser`, display intact)."""
    tmp, store, _h, _q = persona
    mem = _mem("Bob's dog Pretzel is thirty pounds", "conversation", importance=3.0)
    store.create(mem)
    provider = FakeProvider(json.dumps({"importance": 8, "names": ["Pretzel"]}))
    res = _reappraise_run(store, tmp, mem, provider=provider)
    assert res.reappraised == 1
    assert store.get(mem.id, bump=False).importance == pytest.approx(8.0)
    assert _rows(tmp) == [("pretzel", "Pretzel", "reappraiser")]


def test_reappraiser_prompt_carries_the_whole_memory(persona):
    """C6: the whole memory is read (today first 400 chars). Able to fail: a `[:400]`
    cut drops the marker."""
    tmp, store, _h, _q = persona
    text = _long_text("ENDMARKER")
    mem = _mem(text, "conversation", importance=3.0)
    store.create(mem)
    provider = FakeProvider(json.dumps({"importance": 5, "names": []}))
    _reappraise_run(store, tmp, mem, provider=provider)
    prompt, system = provider.calls[0]
    assert prompt == text
    assert "names" in system


def test_reappraiser_old_number_only_reply_still_scores(persona):
    """P-18: a reply with no JSON falls back to today's first-number regex, no names."""
    tmp, store, _h, _q = persona
    mem = _mem("a stable fact", "conversation", importance=3.0)
    store.create(mem)
    res = _reappraise_run(store, tmp, mem, provider=FakeProvider("7"))
    assert res.reappraised == 1
    assert store.get(mem.id, bump=False).importance == pytest.approx(7.0)
    assert _rows(tmp) == []


@pytest.mark.parametrize(
    "raw, importance, names",
    [
        ('{"importance": 6.5, "names": "Wren"}', 6.5, ()),  # bare string names: ignored
        ('{"importance": 6, "names": [3, null, "Zed"]}', 6.0, ("Zed",)),
        ('{"importance": 6, "names": {"a": 1}}', 6.0, ()),
        ('{"importance": 6}', 6.0, ()),
        ('{"importance": "7.5", "names": ["Zed"]}', 7.5, ("Zed",)),  # numeric string
        ('{"names": ["Zed"]}', 4.0, ("Zed",)),  # no importance: unchanged, names kept
        ('{"importance": "high", "names": ["Zed"]}', 4.0, ("Zed",)),
        ('{"importance": true, "names": ["Zed"]}', 4.0, ("Zed",)),  # a bool is not a score
        ('{"importance": "nan", "names": ["Zed"]}', 4.0, ("Zed",)),
        ('{"importance": 1e999}', 4.0, ()),  # non-finite: unchanged
        ('sure: {"importance": 9, "names": ["Zed"]} done', 9.0, ("Zed",)),  # prose around it
        ('{"importance": 7, "names": ["Wr', 7.0, ()),  # truncated JSON: number regex, no names
        ("Room 12, importance 3", 12.0, ()),  # regex fallback is today's first number
        ("no number", 4.0, ()),
        ("", 4.0, ()),
    ],
)
def test_parse_reappraisal_is_fail_soft(raw, importance, names):
    got = _parse_reappraisal(raw, 4.0)
    assert got == Reappraisal(importance, names)


def test_parse_reappraisal_json_importance_never_read_from_a_name():
    """A JSON object whose importance is missing must not fall back to the number regex,
    or a digit inside a name would become the score."""
    got = _parse_reappraisal('{"names": ["Room 12"]}', 4.0)
    assert got.importance == 4.0 and got.names == ("Room 12",)


def test_reappraiser_score_survives_malformed_names_end_to_end(persona):
    tmp, store, _h, _q = persona
    mem = _mem("Bob and Zed", "conversation", importance=3.0)
    store.create(mem)
    res = _reappraise_run(store, tmp, mem, provider=FakeProvider('{"importance": 9, "names": 5}'))
    assert res.reappraised == 1
    assert store.get(mem.id, bump=False).importance == pytest.approx(9.0)
    assert _rows(tmp) == []


def test_reappraiser_importance_is_still_clamped(persona):
    tmp, store, _h, _q = persona
    mem = _mem("Bob and Zed", "conversation", importance=3.0)
    store.create(mem)
    _reappraise_run(store, tmp, mem, provider=FakeProvider('{"importance": 42, "names": []}'))
    assert store.get(mem.id, bump=False).importance == pytest.approx(10.0)


def test_reappraiser_provider_failure_is_a_noop(persona):
    tmp, store, _h, _q = persona
    mem = _mem("Bob and Zed", "conversation", importance=3.5)
    store.create(mem)
    res = _reappraise_run(store, tmp, mem, provider=FakeProvider(RuntimeError("down")))
    assert res.reappraised == 1
    assert store.get(mem.id, bump=False).importance == pytest.approx(3.5)
    assert _rows(tmp) == []


def test_reappraiser_names_go_through_admission_and_carry_source(persona):
    """S70 for the second writer: a stopword entry is rejected, others admitted."""
    tmp, store, _h, _q = persona
    mem = _mem("Will visited Grace", "conversation", importance=3.0)
    store.create(mem)
    fake = FakeProvider(json.dumps({"importance": 5, "names": ["will", "Grace"]}))
    _reappraise_run(store, tmp, mem, provider=fake)
    assert _rows(tmp) == [("grace", "Grace", "reappraiser")]


def test_reappraiser_names_written_only_after_the_importance_update(persona):
    """P-18: a memory hard-deleted between the read and the write is skipped, and its
    names are NOT written. Able to fail: writing the names before the update."""
    tmp, store, _h, _q = persona
    mem = _mem("a memory that will be deleted mid-appraisal", "conversation", importance=2.0)
    store.create(mem)

    def _deleting(m):
        store.hard_delete(m.id)
        return Reappraisal(9.0, ("Wren",))

    res = _reappraise_run(store, tmp, mem, reappraiser=_deleting)
    assert res.reappraised == 0
    assert _rows(tmp) == []


@pytest.mark.parametrize("exc", [OSError("disk gone"), RuntimeError("boom")])
def test_reappraiser_names_write_failure_never_blocks_the_update(persona, monkeypatch, exc, caplog):
    tmp, store, _h, _q = persona
    mem = _mem("Bob and Wren", "conversation", importance=3.0)
    store.create(mem)

    def _boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(cons, "admit_names", _boom)
    with caplog.at_level(logging.WARNING, logger="brain.engines.consolidation"):
        res = _reappraise_run(store, tmp, mem, reappraiser=lambda m: Reappraisal(8.0, ("Wren",)))
    assert res.reappraised == 1
    assert store.get(mem.id, bump=False).importance == pytest.approx(8.0)
    assert any("known-names write failed" in r.getMessage() for r in caplog.records)


def test_reappraiser_writer_calls_admit_names_with_source_reappraiser(persona, monkeypatch):
    tmp, store, _h, _q = persona
    seen = []

    def _spy(persona_dir, names, source, **kw):
        seen.append((Path(persona_dir), list(names), source))
        return []

    monkeypatch.setattr(cons, "admit_names", _spy)
    mem = _mem("Bob and Wren", "conversation", importance=3.0)
    store.create(mem)
    _reappraise_run(store, tmp, mem, reappraiser=lambda m: Reappraisal(6.0, ("Wren",)))
    assert seen == [(tmp, ["Wren"], "reappraiser")]


def test_reappraiser_interface_is_number_plus_names_not_a_bare_number(persona, caplog):
    """The interface changed: a reappraiser returning a bare number is a contract
    violation, logged and skipped (importance untouched, not counted)."""
    tmp, store, _h, _q = persona
    mem = _mem("Bob and Zed", "conversation", importance=3.0)
    store.create(mem)
    res = _reappraise_run(store, tmp, mem, reappraiser=lambda m: 9.0)
    assert res.reappraised == 0
    assert store.get(mem.id, bump=False).importance == pytest.approx(3.0)


def test_noop_reappraiser_returns_current_importance_and_no_names():
    m = _mem("x", importance=4.5)
    assert _noop_reappraiser(m) == Reappraisal(4.5, ())


def test_haiku_reappraiser_direct_call_returns_a_reappraisal():
    provider = FakeProvider('{"importance": 2, "names": ["Wren"]}')
    got = _make_haiku_reappraiser(provider)(_mem("Bob and Wren", importance=6.0))
    assert got == Reappraisal(2.0, ("Wren",))


# ===========================================================================
# Pure parsing helper
# ===========================================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ()),
        (["Wren", " Pretzel "], ("Wren", "Pretzel")),
        (("Wren",), ("Wren",)),
        ("Wren", ()),
        (5, ()),
        ({"a": 1}, ()),
        ([None, 3, "", "  "], ()),
        (["José", "O'Brien"], ("José", "O'Brien")),  # display form kept intact (S87)
    ],
)
def test_parse_names(value, expected):
    assert _parse_names(value, source="test") == expected
