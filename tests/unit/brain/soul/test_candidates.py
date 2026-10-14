"""brain.soul.candidates.list_crystallised_memory_ids — which accepted candidates
protect their memory from forgetting (#362)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from brain.soul.candidates import list_crystallised_memory_ids
from brain.soul.crystallization import Crystallization
from brain.soul.store import SoulStore


def _write_candidates(persona_dir: Path, records: list[dict]) -> None:
    (persona_dir / "soul_candidates.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


def _link_directly(persona_dir: Path, memory_id: str, *, revoke: bool = False) -> None:
    """A crystallization made the way the crystallize_soul tool makes one: no candidate record."""
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        c = Crystallization(
            id=f"direct-{memory_id}",
            moment="m",
            love_type="craft",
            why_it_matters="w",
            crystallized_at=datetime.now(UTC),
        )
        store.create(c)
        store.link_source_memory(c.id, memory_id)
        if revoke:
            store.mark_revoked(c.id, "test")
    finally:
        store.close()


def test_direct_link_is_returned_even_with_no_candidates_file(tmp_path):
    """#371 C10: a direct crystallisation has no candidate record, so the reader must not stop at
    'no soul_candidates.jsonl' before it consults the store."""
    _link_directly(tmp_path, "m-direct")
    assert not (tmp_path / "soul_candidates.jsonl").exists()
    assert list_crystallised_memory_ids(tmp_path) == ["m-direct"]


def test_direct_link_is_returned_when_the_candidates_file_has_no_accepted_record(tmp_path):
    """#371 C10 / stage-6 F5: the OTHER early return. A soul_candidates.jsonl that exists but holds only
    pending / rejected records must not stop the reader before it consults the store."""
    _write_candidates(
        tmp_path,
        [
            {"memory_id": "m-pending", "status": "pending"},
            {"memory_id": "m-rejected", "status": "rejected"},
        ],
    )
    _link_directly(tmp_path, "m-direct")
    assert list_crystallised_memory_ids(tmp_path) == ["m-direct"]


def test_the_reader_opens_the_soul_store_once_per_call(tmp_path, monkeypatch):
    """#371 C10 / stage-6 F6: each SoulStore open runs an integrity_check, so reading the revoked ids and
    the direct links must share ONE open, with both sources present."""
    _write_candidates(tmp_path, [{"memory_id": "m-cand", "status": "accepted", "crystallization_id": "gone"}])
    _link_directly(tmp_path, "m-direct")

    opens: list[int] = []
    real_init = SoulStore.__init__

    def counting_init(self, *args, **kwargs):
        opens.append(1)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(SoulStore, "__init__", counting_init)

    assert list_crystallised_memory_ids(tmp_path) == ["m-cand", "m-direct"]
    assert len(opens) == 1


def test_candidate_ids_first_then_direct_links_sorted_deduplicated_revoked_dropped(tmp_path):
    """#371 C10: union of both sources. Candidate ids lead; direct links follow sorted by memory id;
    a memory named by both appears once; a revoked direct crystallisation no longer protects."""
    _write_candidates(tmp_path, [{"memory_id": "m-cand", "status": "accepted"}])
    _link_directly(tmp_path, "m-z")
    _link_directly(tmp_path, "m-a")
    _link_directly(tmp_path, "m-cand")  # also named by the candidate path
    _link_directly(tmp_path, "m-revoked", revoke=True)
    assert list_crystallised_memory_ids(tmp_path) == ["m-cand", "m-a", "m-z"]


def test_an_unreadable_candidates_file_does_not_drop_the_direct_links(tmp_path, caplog):
    """Each source fails on its own: a real read failure on soul_candidates.jsonl (a directory where the
    file should be) must cost only the candidate source, loudly, and leave the direct links protecting."""
    import logging

    (tmp_path / "soul_candidates.jsonl").mkdir()
    _link_directly(tmp_path, "m-direct")

    with caplog.at_level(logging.WARNING):
        assert list_crystallised_memory_ids(tmp_path) == ["m-direct"]

    assert any("crystallised" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_a_broken_link_table_does_not_drop_the_accepted_candidates(tmp_path, caplog):
    """The other direction: the new crystallization_sources table is the only thing this reader gained, so a
    malformed one must not take the accepted-candidate protection (which worked before #371) down with it."""
    import logging

    _write_candidates(tmp_path, [{"memory_id": "m-cand", "status": "accepted"}])
    store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        store._conn.execute("CREATE TABLE crystallization_sources (not_the_real_schema TEXT)")
        store._conn.commit()
    finally:
        store.close()

    with caplog.at_level(logging.WARNING):
        assert list_crystallised_memory_ids(tmp_path) == ["m-cand"]

    assert any("crystallised" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_unknown_link_fails_toward_protection(tmp_path):
    """Losing a memory is the worse error: only an affirmative revoke drops the
    protection. A legacy record (no crystallization_id), a dangling id, and no
    crystallizations.db at all all stay protected — and the read must not
    create the database as a side effect."""
    _write_candidates(
        tmp_path,
        [
            {"memory_id": "m-legacy", "status": "accepted"},
            {"memory_id": "m-dangling", "status": "accepted", "crystallization_id": "gone"},
        ],
    )
    assert list_crystallised_memory_ids(tmp_path) == ["m-legacy", "m-dangling"]
    assert not (tmp_path / "crystallizations.db").exists()


def test_only_accepted_candidates_protect(tmp_path):
    _write_candidates(
        tmp_path,
        [
            {"memory_id": "m-accepted", "status": "accepted"},
            {"memory_id": "m-pending", "status": "pending"},
            {"memory_id": "m-auto", "status": "auto_pending"},
            {"memory_id": "m-rejected", "status": "rejected"},
            {"memory_id": "m-expired", "status": "expired"},
            {"memory_id": "m-no-status"},
        ],
    )
    assert list_crystallised_memory_ids(tmp_path) == ["m-accepted"]
