"""brain.soul.candidates.list_crystallised_memory_ids — which accepted candidates
protect their memory from forgetting (#362)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

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


def test_an_unreadable_candidates_file_raises_rather_than_returning_the_direct_links_alone(tmp_path):
    """#392: a real read failure on soul_candidates.jsonl (a directory where the file should be) must reach
    the caller. Returning just the direct links would look like a complete answer, and the caller
    (brain.forgetting) would run a pass with the accepted candidates un-exempt."""
    (tmp_path / "soul_candidates.jsonl").mkdir()
    _link_directly(tmp_path, "m-direct")

    with pytest.raises(OSError):
        list_crystallised_memory_ids(tmp_path)


def test_a_broken_link_table_raises_rather_than_returning_the_accepted_candidates_alone(tmp_path):
    """The other direction: a malformed crystallization_sources table must reach the caller, not be
    papered over with the accepted candidates alone."""
    _write_candidates(tmp_path, [{"memory_id": "m-cand", "status": "accepted"}])
    store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        store._conn.execute("CREATE TABLE crystallization_sources (not_the_real_schema TEXT)")
        store._conn.commit()
    finally:
        store.close()

    with pytest.raises(sqlite3.Error):
        list_crystallised_memory_ids(tmp_path)


def test_a_failing_revoked_read_raises_rather_than_dropping_every_accepted_candidate(tmp_path, monkeypatch):
    """#392: if the revoked list cannot be read, 'nothing is revoked' is a guess and 'everything is revoked'
    is a worse one. The reader raises and lets the caller decide."""
    _write_candidates(tmp_path, [{"memory_id": "m-cand", "status": "accepted", "crystallization_id": "c-1"}])
    SoulStore(str(tmp_path / "crystallizations.db")).close()

    def _boom(self):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(SoulStore, "list_revoked", _boom)

    with pytest.raises(sqlite3.OperationalError):
        list_crystallised_memory_ids(tmp_path)


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


# --------------------------------------------------------------------------- #430 merge targets
def _merge_row(candidate_id: str, target_id: str, content: str = "x") -> dict:
    """The shape ``consolidation._archive_preimage`` writes."""
    return {
        "archived_at": "2026-10-01T00:00:00+00:00",
        "reason": "consolidation_merge",
        "target": {"id": target_id, "content": content},
        "merged_from_candidate": candidate_id,
    }


def _archive_line(rec: dict) -> bytes:
    return (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")


def test_430_c4b_torn_archive_lines_are_skipped_with_a_warning(tmp_path, caplog):
    """#430 C4b: a cut JSON line, and a line cut mid multi-byte character with a later append fused onto
    it, are skipped with a warning; the other lines still map. The fused row's redirect is lost
    (expected: spec "Torn tail")."""
    import logging

    from brain.engines.consolidation import _ARCHIVE_FILENAME
    from brain.soul.candidates import merge_target_map

    cut_json = _archive_line(_merge_row("cand-a", "target-a"))[:40] + b"\n"
    cut_char = _archive_line(_merge_row("cand-b", "target-b", "café au lait"))
    cut_char = cut_char[: cut_char.index("é".encode()) + 1]  # half of the two-byte é, no newline
    fused = _archive_line(_merge_row("cand-c", "target-c"))  # appended onto the torn line
    good = _archive_line(_merge_row("cand-d", "target-d", "naïve"))
    (tmp_path / _ARCHIVE_FILENAME).write_bytes(cut_json + cut_char + fused + good)

    with caplog.at_level(logging.WARNING):
        got = merge_target_map(tmp_path)

    assert got == {"cand-d": {"target-d"}}
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len([m for m in warnings if "skipping malformed jsonl line" in m]) == 2, warnings


def test_430_c4c_only_merge_rows_of_soul_linked_ids_redirect(tmp_path):
    """#430 C4c: a dedup_sweep row and a merge row for an id nothing soul-links add nothing; one
    candidate with two merge rows (the first naming a row since deleted, the second a live row) maps to
    both, so the live target is exempt whichever came first."""
    from brain.engines.consolidation import _ARCHIVE_FILENAME
    from brain.forgetting import _load_soul_linked_ids
    from brain.soul.candidates import merge_target_map

    rows = [
        {"archived_at": "2026-10-01T00:00:00+00:00", "reason": "dedup_sweep", "target": {"id": "swept"}},
        _merge_row("unlinked-cand", "unlinked-target"),
        _merge_row("cand", "deleted-target"),
        _merge_row("cand", "live-target"),
    ]
    (tmp_path / _ARCHIVE_FILENAME).write_bytes(b"".join(_archive_line(r) for r in rows))
    _write_candidates(tmp_path, [{"memory_id": "cand", "status": "auto_pending"}])

    assert merge_target_map(tmp_path) == {
        "unlinked-cand": {"unlinked-target"},
        "cand": {"deleted-target", "live-target"},
    }
    crystallised, under_review = _load_soul_linked_ids(tmp_path)
    assert crystallised == set()
    assert under_review == {"cand", "deleted-target", "live-target"}


def test_430_missing_archive_maps_nothing_and_creates_no_lock_sidecar(tmp_path):
    from brain.soul.candidates import merge_target_map

    assert merge_target_map(tmp_path) == {}
    assert list(tmp_path.iterdir()) == []
