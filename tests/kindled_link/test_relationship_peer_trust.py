"""peer_trust: the peer modifier's read of how much a peer is trusted (emotion rework B1, PR 1)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from brain.kindled_link.relationship import (
    PeerRelationshipState,
    peer_trust,
    persist_relationship_state,
)
from brain.kindled_link.store import KindledLinkStore, kindled_db_path


def _save_relationship(persona_dir, *, stage="stranger", trust_score=0.9):
    path = kindled_db_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = KindledLinkStore(path)
    try:
        state = PeerRelationshipState(peer_id="kid_a", stage=stage, trust_score=trust_score)
        persist_relationship_state(store, state, datetime.now(UTC))
    finally:
        store.close()


def test_no_database_answers_zero_and_creates_nothing(tmp_path):
    assert peer_trust(tmp_path, "kid_a") == 0.0
    assert not kindled_db_path(tmp_path).exists()
    assert not kindled_db_path(tmp_path).parent.exists()


@pytest.mark.no_base_failure(reason="pins the existing guard that reading never creates the database; the base code already holds it")
def test_a_missing_database_in_an_existing_folder_is_not_created_by_a_read(tmp_path):
    # With the folder present, a connect would succeed and create an empty database, so only the existence guard keeps it absent.
    folder = kindled_db_path(tmp_path).parent
    folder.mkdir(parents=True)
    assert peer_trust(tmp_path, "kid_a") == 0.0
    assert not kindled_db_path(tmp_path).exists()
    assert list(folder.iterdir()) == []


def test_a_path_that_cannot_be_inspected_answers_zero_and_warns(tmp_path, monkeypatch, caplog):
    # Python 3.12 and 3.13 raise PermissionError from a stat of a path under an unsearchable folder. A real database
    # holds trust 0.9, so an unguarded read would answer 0.9, not 0.0. raising=False: the base module has no stat_path.
    _save_relationship(tmp_path, trust_score=0.9)

    def denied(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr("brain.kindled_link.relationship.stat_path", denied, raising=False)
    assert peer_trust(tmp_path, "kid_a") == 0.0
    assert "peer_trust" in caplog.text


def test_answers_stored_trust_with_no_stage_cap(tmp_path):
    _save_relationship(tmp_path, stage="stranger", trust_score=0.9)
    assert peer_trust(tmp_path, "kid_a") == pytest.approx(0.9)


def test_missing_row_answers_zero(tmp_path):
    _save_relationship(tmp_path, trust_score=0.9)
    assert peer_trust(tmp_path, "someone_else") == 0.0


def test_corrupt_database_answers_zero_and_warns(tmp_path, caplog):
    path = kindled_db_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"garbage" * 200)
    assert peer_trust(tmp_path, "kid_a") == 0.0
    assert "peer_trust" in caplog.text


def test_unreadable_row_answers_zero(tmp_path):
    _save_relationship(tmp_path, trust_score=0.9)
    conn = sqlite3.connect(kindled_db_path(tmp_path))
    conn.execute("update relationship_state set evidence_json = 'not json'")
    conn.commit()
    conn.close()
    assert peer_trust(tmp_path, "kid_a") == 0.0
