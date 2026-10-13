"""brain.soul.candidates.list_crystallised_memory_ids — which accepted candidates
protect their memory from forgetting (#362)."""

from __future__ import annotations

import json
from pathlib import Path

from brain.soul.candidates import list_crystallised_memory_ids


def _write_candidates(persona_dir: Path, records: list[dict]) -> None:
    (persona_dir / "soul_candidates.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


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
