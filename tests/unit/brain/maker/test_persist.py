from brain.maker.maker import Making
from brain.maker.persist import persist_making
from brain.works.store import WorksStore


def test_private_making_keeps_content(tmp_path):
    store = WorksStore(tmp_path / "works.db")
    m = Making("elegy", "For the dog", "Soft paws, gone.", "private", "raw")
    wid = persist_making(tmp_path, store, m, charge_sources=["grief"])
    got = store.get(wid)
    assert got.disposition == "private" and got.origin == "maker"
    store.close()


def test_discard_drops_content_keeps_thin_row(tmp_path):
    store = WorksStore(tmp_path / "works.db")
    m = Making("note", "throwaway", "delete me", "discard", None)
    wid = persist_making(tmp_path, store, m, charge_sources=["dream"])
    got = store.get(wid)
    assert got.disposition == "discard"
    # content markdown must NOT exist on disk for a discard
    from brain.works.storage import _work_path
    assert not _work_path(tmp_path, wid).exists()
    store.close()
