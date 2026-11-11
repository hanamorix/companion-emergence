import json

import pytest

from brain.maker.making_runner import make_and_wire
from brain.memory.store import MemoryStore
from brain.works.store import WorksStore


def _promote_pending(persona_dir):
    """Drain the consolidation gate with a promote-all classifier.

    `making` is a GATED type (Root-2 stopgap): make_and_wire enqueues the act-memory; it
    reaches memories.db only after a consolidation drain."""
    from brain.engines.consolidation import Decision, run_consolidation

    s = MemoryStore(persona_dir / "memories.db")
    try:
        run_consolidation(
            s, persona_dir=persona_dir, classifier=lambda _c, _ctx: Decision("new")
        )
    finally:
        s.close()


class _FakeProvider:
    def complete(self, prompt):  # match maker.make()'s call
        return json.dumps({"type": "vignette", "title": "Dusk", "content": "Light fell slow.",
                           "disposition": "eventual_share"})


def test_make_and_wire_persists_a_making(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memories.db")
    # charge sources file / state present
    from brain.maker.charge import MakerCharge, save_charge
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_FakeProvider(),
                  now=None)
    works = WorksStore(tmp_path / "works.db")
    # at least one making persisted
    assert works._conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 1
    works.close()
    store.close()


def test_make_and_wire_writes_act_memory_with_emotion_for_nondiscard(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    from brain.maker.charge import MakerCharge, save_charge
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_FakeProvider(), now=None)
    # making is GATED: the act-memory enqueues, promoted on drain.
    assert not store.list_by_type("making", limit=10)
    store.close()
    _promote_pending(tmp_path)
    store2 = MemoryStore(tmp_path / "memories.db")
    mems = store2.list_by_type("making", limit=10)
    assert len(mems) == 1
    # the emotion delta was applied W8-style: seeded onto the act-memory
    assert mems[0].emotions
    store2.close()


class _DiscardProvider:
    def complete(self, prompt):
        return json.dumps({"type": "vignette", "title": "Trash", "content": "x",
                           "disposition": "discard"})


def test_make_and_wire_discard_moves_no_emotion(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    from brain.maker.charge import MakerCharge, save_charge
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_DiscardProvider(), now=None)
    # making is GATED: the act-memory enqueues, promoted on drain.
    assert not store.list_by_type("making", limit=10)
    store.close()
    _promote_pending(tmp_path)
    store2 = MemoryStore(tmp_path / "memories.db")
    mems = store2.list_by_type("making", limit=10)
    assert len(mems) == 1
    # discard moves nothing — empty-emotion act-memory only
    assert not mems[0].emotions
    store2.close()


# ---- emotion rework B1 PR 3: the making nudge is on the 0-10 scale once the engine is live ----


def _go_live(persona_dir, store):
    """A live persona with no state file applies nothing, so seed a memory first, then write the switch."""
    from datetime import UTC, datetime

    from brain.emotion import engine_files
    from brain.memory.store import Memory

    store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions={"joy": 5.0}))
    assert engine_files.write_switch(persona_dir, datetime.now(UTC))


def _made_emotions(tmp_path, *, live):
    from brain.maker.charge import MakerCharge, save_charge

    store = MemoryStore(tmp_path / "memories.db")
    if live:
        _go_live(tmp_path, store)
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_FakeProvider(), now=None)
    store.close()
    _promote_pending(tmp_path)
    store2 = MemoryStore(tmp_path / "memories.db")
    try:
        return store2.list_by_type("making", limit=10)[0].emotions
    finally:
        store2.close()


def test_make_and_wire_stores_the_nudge_ten_times_when_live(tmp_path):
    # satisfaction is not a registered name, so the vocabulary filter drops it and tenderness is what reaches her.
    assert _made_emotions(tmp_path, live=True) == pytest.approx({"tenderness": 0.8})



@pytest.mark.no_base_failure(reason="pins today's small making nudge with the switch off (B1 PR 3); main stores it too")
def test_make_and_wire_keeps_todays_nudge_when_off(tmp_path):
    assert _made_emotions(tmp_path, live=False) == pytest.approx({"tenderness": 0.08})
