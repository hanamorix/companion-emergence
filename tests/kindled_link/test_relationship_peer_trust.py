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
