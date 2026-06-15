import json

from brain.maker.making_runner import make_and_wire
from brain.memory.store import MemoryStore
from brain.works.store import WorksStore


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
    mems = store.list_by_type("making", limit=10)
    assert len(mems) == 1
    # the emotion delta was applied W8-style: seeded onto the act-memory
    assert mems[0].emotions
    store.close()


class _DiscardProvider:
    def complete(self, prompt):
        return json.dumps({"type": "vignette", "title": "Trash", "content": "x",
                           "disposition": "discard"})


def test_make_and_wire_discard_moves_no_emotion(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    from brain.maker.charge import MakerCharge, save_charge
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_DiscardProvider(), now=None)
    mems = store.list_by_type("making", limit=10)
    assert len(mems) == 1
    # discard moves nothing — empty-emotion act-memory only
    assert not mems[0].emotions
    store.close()
