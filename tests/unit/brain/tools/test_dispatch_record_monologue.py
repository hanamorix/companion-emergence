"""record_monologue dispatch calls capture_monologue and returns monologue_text."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# ---- B1 PR 6: once the engine is live, the trace is the note (digest + the engine's strongest feelings) ----


def _live_persona(tmp_path: Path, **seed: float):
    """A live persona: seed first, then record the switch-over in the state the seed wrote."""
    from brain.emotion import engine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from tests.unit.brain.emotion._engine_support import make_live

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    store.create(
        Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=seed or {"joy": 5.0})
    )
    make_live(persona_dir, datetime.now(UTC))
    return persona_dir, store, hebbian, engine.engine_for(persona_dir)


def _settled(eng):
    eng.note_active(datetime.now(UTC))
    return eng.read()


def _anger_short_joy_long(eng):
    """Strongest short = anger, strongest long = joy (a back-dated change separates short from long)."""
    eng.fold_change({"joy": 4.0}, datetime.now(UTC) - timedelta(days=3), "monologue_emotion")
    eng.fold_change({"anger": 2.0}, None, "monologue_emotion")
    return _settled(eng)


def _record(persona_dir, store, hebbian, monologue, digest):
    from brain.tools.dispatch import dispatch

    return dispatch(
        "record_monologue",
        {"monologue": monologue, "feed_digest": digest},
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
    )


def _traces(persona_dir, limit=10):
    from brain.memory.pending import PendingQueue
    from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

    return PendingQueue(persona_dir).read_recent(MONOLOGUE_TRACE_TYPE, limit=limit)


def test_live_capture_stores_digest_two_feelings_and_keeps_raw_for_pass2(tmp_path: Path):
    from brain.monologue.trace import _trace_importance

    persona_dir, store, hebbian, eng = _live_persona(tmp_path, joy=5.0, anger=5.0)
    try:
        reading = _anger_short_joy_long(eng)
        top_short = max(reading, key=lambda name: reading[name].short)
        top_long = max(reading, key=lambda name: reading[name].long)
        assert (top_short, top_long) == ("anger", "joy")
        result = _record(persona_dir, store, hebbian, "raw drift about the rain", "she listened to the rain")
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert result["monologue_text"] == "raw drift about the rain"
    assert trace.content == "she listened to the rain"
    assert trace.emotions == pytest.approx({"anger": reading["anger"].short, "joy": reading["joy"].long})
    assert trace.importance == pytest.approx(_trace_importance(trace.emotions))


def test_live_capture_same_strongest_feeling_stores_one_key_with_the_long_value(tmp_path: Path):
    persona_dir, store, hebbian, eng = _live_persona(tmp_path, joy=5.0)
    try:
        eng.fold_change({"joy": 4.0}, datetime.now(UTC) - timedelta(days=3), "monologue_emotion")
        reading = _settled(eng)
        assert set(reading) == {"joy"} and reading["joy"].long > reading["joy"].short
        _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert trace.emotions == pytest.approx({"joy": reading["joy"].long})


def test_live_capture_with_an_empty_read_stores_no_feelings(tmp_path: Path):
    """A switched persona whose state is gone, then a memory with feelings: the engine applies nothing, so read() is {} while
    today's aggregate is not."""
    from brain.emotion import engine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from tests.unit.brain.emotion._engine_support import lose_the_state, make_live

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        make_live(persona_dir, datetime.now(UTC))
        assert engine.engine_for(persona_dir).is_live()  # the running engine has seen the switch-over; the state is gone afterwards
        lose_the_state(persona_dir)
        store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions={"joy": 8.0}))
        assert _settled(engine.engine_for(persona_dir)) == {}
        _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert trace.content == "she drifted"
    assert trace.emotions == {}
    assert trace.importance == pytest.approx(0.2)


def _digest_lines(persona_dir):
    path = persona_dir / "monologue_digest.jsonl"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_live_duplicate_guard_compares_digests(tmp_path: Path):
    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)
    try:
        first = _record(persona_dir, store, hebbian, "thought one", "she wondered")
        same_digest = _record(persona_dir, store, hebbian, "thought two", "she wondered")
        assert (first["ok"], same_digest) == (True, {"ok": True, "deduped": True})
        assert [t.content for t in _traces(persona_dir)] == ["she wondered"]
        assert len(_digest_lines(persona_dir)) == 1

        new_digest = _record(persona_dir, store, hebbian, "thought two", "she wondered again")
        assert new_digest == {"ok": True, "monologue_text": "thought two"}
        rerun = _record(persona_dir, store, hebbian, "thought two", "she wondered again")  # #93
        assert rerun == {"ok": True, "deduped": True}
        assert len(_traces(persona_dir)) == 2
        assert len(_digest_lines(persona_dir)) == 2
    finally:
        store.close()
        hebbian.close()


@pytest.mark.no_base_failure(reason="base guard always compares the raw thought; bites when the live guard drops the raw match")
def test_rerun_across_the_switch_over_is_still_deduped(tmp_path: Path):
    """#93 at the switch-over: the first call of a recruit-on-reach pair is captured off (raw trace), the switch appears,
    and the rerun is live. It must still be recognised as the same thought."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from tests.unit.brain.emotion._engine_support import make_live

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions={"joy": 5.0}))
        first = _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        make_live(persona_dir, datetime.now(UTC))
        rerun = _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        traces = _traces(persona_dir)
    finally:
        store.close()
        hebbian.close()
    assert first == {"ok": True, "monologue_text": "raw drift"}
    assert rerun == {"ok": True, "deduped": True}
    assert [t.content for t in traces] == ["raw drift"]
    assert len(_digest_lines(persona_dir)) == 1


def test_one_live_answer_per_capture(tmp_path: Path, monkeypatch):
    """The guard and the trace write share one answer: a switch that flips between the two asks can't split them."""
    from brain.emotion.engine import EmotionEngine

    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)
    answers = iter([True, False, False])
    monkeypatch.setattr(EmotionEngine, "is_live", lambda self: next(answers))
    try:
        _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert trace.content == "she drifted"


def test_engine_read_that_raises_stores_no_feelings_and_keeps_the_trace(tmp_path: Path, monkeypatch):
    from brain.emotion.engine import EmotionEngine

    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)

    def broken(self):
        raise RuntimeError("read failed")

    monkeypatch.setattr(EmotionEngine, "read", broken)
    try:
        _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert (trace.content, trace.emotions) == ("she drifted", {})


@pytest.mark.no_base_failure(reason="base never reads the switch; bites when _is_live stops treating a raise as off")
def test_switch_read_that_raises_counts_as_off_and_keeps_the_thought(tmp_path: Path, monkeypatch):
    from brain.emotion.engine import EmotionEngine

    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)

    def broken(self):
        raise OSError("switch unreadable")

    monkeypatch.setattr(EmotionEngine, "is_live", broken)
    try:
        result = _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert result == {"ok": True, "monologue_text": "raw drift"}
    assert trace.content == "raw drift"


@pytest.mark.no_base_failure(reason="on the base this persona is not live either (nothing there reads the state for liveness), so it passes; bites if an unreadable state ever counts as live")
def test_unreadable_state_counts_as_off_through_dispatch(tmp_path: Path):
    """A real unknown: the state file is there but no copy can be read. Owner 2026-11-05: "If the system can't tell if its
    switched, assume its not." Unlike patching is_live, this goes through the engine's own handling of the unknown."""
    from tests.unit.brain.emotion._engine_support import ruin_the_state

    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)
    ruin_the_state(persona_dir)
    try:
        _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert trace.content == "raw drift"


@pytest.mark.no_base_failure(reason="pins today's switch-off trace, which this PR must leave unchanged")
def test_switch_off_capture_stores_todays_trace(tmp_path: Path):
    from brain.emotion.aggregate import aggregate_state
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.monologue.trace import _trace_importance

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        store.create(
            Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions={"joy": 8.0, "grief": 3.0})
        )
        expected = dict(aggregate_state(store.list_active()).emotions)
        result = _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert result == {"ok": True, "monologue_text": "raw drift"}
    assert (trace.content, trace.emotions) == ("raw drift", expected)
    assert trace.importance == pytest.approx(_trace_importance(expected))


def test_write_trace_memory_without_a_persona_dir_is_off(tmp_path: Path):
    from brain.monologue.trace import write_trace_memory

    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)
    try:
        write_trace_memory(store, "raw drift", feed_digest="she drifted", persona_dir=None)
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert trace.content == "raw drift"


def test_live_answer_without_a_persona_dir_is_still_off(tmp_path: Path):
    """No persona means no engine to read, so the whole trace is today's, not digest content with old feelings."""
    from brain.monologue.trace import write_trace_memory

    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)
    try:
        from brain.emotion.aggregate import aggregate_state

        expected = dict(aggregate_state(store.list_active()).emotions)
        write_trace_memory(store, "raw drift", feed_digest="she drifted", persona_dir=None, live=True)
        trace = _traces(persona_dir)[0]
    finally:
        store.close()
        hebbian.close()
    assert (trace.content, trace.emotions) == ("raw drift", expected)


def test_live_interior_block_shows_the_digest_not_the_raw_thought(tmp_path: Path):
    from brain.monologue.ambient import build_interior_continuity_block

    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)
    try:
        _record(persona_dir, store, hebbian, "raw drift about lighthouses", "she thought about the sea")
        block = build_interior_continuity_block(store)
    finally:
        store.close()
        hebbian.close()
    assert "she thought about the sea" in block
    assert "lighthouses" not in block


def test_live_recall_matches_digest_words_not_raw_only_words(tmp_path: Path):
    from brain.monologue.recall import recall_monologue

    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)
    try:
        _record(persona_dir, store, hebbian, "raw drift about lighthouses", "she thought about the sea")
        by_digest = recall_monologue("sea", store=store, hebbian=hebbian, persona_dir=persona_dir)
        by_raw = recall_monologue("lighthouses", store=store, hebbian=hebbian, persona_dir=persona_dir)
    finally:
        store.close()
        hebbian.close()
    assert "she thought about the sea" in str(by_digest["monologues"])
    assert by_raw["monologues"] == []


def _promote_all(persona_dir, store, hebbian):
    """The real consolidation pass, with a classifier that calls every candidate new."""
    from brain.engines.consolidation import Decision, run_consolidation

    return run_consolidation(store, persona_dir=persona_dir, hebbian=hebbian, classifier=lambda _c, _ctx: Decision("new"))


def test_live_interest_sweep_sample_shows_the_promoted_digest(tmp_path: Path):
    from brain.engines.interest_sweep import _lived_sample

    persona_dir, store, hebbian, _eng = _live_persona(tmp_path)
    try:
        _record(persona_dir, store, hebbian, "raw drift about lighthouses", "she thought about the sea")
        _promote_all(persona_dir, store, hebbian)
        sample = _lived_sample(store)
    finally:
        store.close()
        hebbian.close()
    assert "(monologue_trace) she thought about the sea" in sample
    assert "lighthouses" not in sample


def test_engine_does_not_count_a_promoted_trace(tmp_path: Path):
    """Promotion goes through the real store.create; the trace's feelings differ from the other average, so a counted
    trace would move the engine. The merge path into counted rows is #619's to fix (a prerequisite of Engine 6)."""
    from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

    persona_dir, store, hebbian, eng = _live_persona(tmp_path, joy=5.0, anger=5.0)
    try:
        before = _anger_short_joy_long(eng)
        _record(persona_dir, store, hebbian, "raw drift", "she drifted")
        _promote_all(persona_dir, store, hebbian)
        promoted = store.list_by_type(MONOLOGUE_TRACE_TYPE, active_only=True, limit=5)
        after = _settled(eng)
    finally:
        store.close()
        hebbian.close()
    assert [m.content for m in promoted] == ["she drifted"]
    assert promoted[0].emotions["joy"] - before["joy"].short > 0.5  # stored (long) differs from the other average
    assert after == before


class _ReadingsEngine:
    """Stands in for the engine with fixed readings, to pin how the strongest feelings are picked."""

    def __init__(self, **readings):
        from brain.emotion.engine import Reading

        self._readings = {name: Reading(short, long, long - short, 0) for name, (short, long) in readings.items()}

    def read(self):
        return dict(self._readings)


def test_strongest_feelings_same_feeling_keeps_the_larger_value():
    from brain.monologue.trace import _strongest_feelings

    assert _strongest_feelings(_ReadingsEngine(joy=(4.0, 7.5), calm=(2.0, 3.0))) == {"joy": 7.5}
    assert _strongest_feelings(_ReadingsEngine(joy=(7.5, 4.0), calm=(2.0, 3.0))) == {"joy": 7.5}


def test_strongest_feelings_ties_go_to_the_alphabetically_first_name():
    from brain.monologue.trace import _strongest_feelings

    tied = _ReadingsEngine(joy=(5.0, 6.0), anger=(5.0, 2.0), calm=(1.0, 6.0))
    assert _strongest_feelings(tied) == {"anger": 5.0, "calm": 6.0}


def test_strongest_feelings_drops_zeros():
    from brain.monologue.trace import _strongest_feelings

    assert _strongest_feelings(_ReadingsEngine(joy=(0.0, 0.0), calm=(0.0, 0.0))) == {}
    assert _strongest_feelings(_ReadingsEngine(joy=(0.0, 3.0), calm=(0.0, 1.0))) == {"joy": 3.0}


def test_dispatch_record_monologue_returns_ok_with_monologue_text(tmp_path: Path):
    """dispatch record_monologue now returns monologue_text on the success path."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools.dispatch import dispatch

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        result = dispatch(
            "record_monologue",
            {"monologue": "thinking about Loopy", "feed_digest": "she searched and felt fond"},
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        assert result["ok"] is True
        assert result["monologue_text"] == "thinking about Loopy"
    finally:
        store.close()
        hebbian.close()


def test_dispatch_record_monologue_writes_digest(tmp_path: Path):
    """capture_monologue writes monologue_digest.jsonl synchronously inside dispatch."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools.dispatch import dispatch

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        result = dispatch(
            "record_monologue",
            {"monologue": "thought", "feed_digest": "digest"},
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        assert result["ok"] is True
        assert result["monologue_text"] == "thought"
        assert (persona_dir / "monologue_digest.jsonl").exists()
    finally:
        store.close()
        hebbian.close()


def test_dispatch_record_monologue_rejected_returns_error_dict(tmp_path: Path):
    """Whitespace monologue -> CaptureRejected -> error dict, no exception raised."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools.dispatch import dispatch

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        result = dispatch(
            "record_monologue",
            {"monologue": "   ", "feed_digest": "digest"},
            store=store,
            hebbian=hebbian,
            persona_dir=persona_dir,
        )
        assert "error" in result
        assert "ok" not in result
        assert not (persona_dir / "monologue_digest.jsonl").exists()
    finally:
        store.close()
        hebbian.close()


def test_dispatch_record_monologue_marks_duplicate_without_monologue_text(tmp_path: Path):
    """#175: the recruit-on-reach rerun dispatches record_monologue twice; the
    second is deduped by capture_monologue and must NOT come back carrying
    ``monologue_text`` (that key is what the audit row + pass-2 trigger key on)."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools.dispatch import dispatch

    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    args = {"monologue": "thought", "feed_digest": "digest"}
    try:
        first = dispatch("record_monologue", args, store=store, hebbian=hebbian, persona_dir=persona_dir)
        second = dispatch("record_monologue", args, store=store, hebbian=hebbian, persona_dir=persona_dir)
        assert first == {"ok": True, "monologue_text": "thought"}
        assert second == {"ok": True, "deduped": True}
    finally:
        store.close()
        hebbian.close()
