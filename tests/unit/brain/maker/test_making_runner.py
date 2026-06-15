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
