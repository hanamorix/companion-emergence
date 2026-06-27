"""Tests for POST /kindled-link/identity/rotate bridge endpoint."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge.server import build_app
from brain.kindled_link.store import KindledLinkStore, kindled_db_path

_TOK = "secret-token"
_AUTH = {"Authorization": f"Bearer {_TOK}"}


def _app(tmp_path: Path):
    persona = tmp_path / "persona"
    persona.mkdir(parents=True, exist_ok=True)
    return build_app(persona_dir=persona, client_origin="tests", auth_token=_TOK), persona


def _seed_peer(persona: Path) -> None:
    from datetime import UTC, datetime

    from brain.kindled_link.identity import KindledIdentity

    KindledIdentity.load_or_create(persona)
    db = kindled_db_path(persona)
    db.parent.mkdir(parents=True, exist_ok=True)
    store = KindledLinkStore(db)
    store.upsert_peer(
        peer_id="kid_peer",
        identity_pub_hex="cc" * 32,
        fingerprint="kid_peer",
        consent_state="paired",
        relay_url="http://relay",
        now=datetime(2026, 6, 27, tzinfo=UTC),
    )
    store.close()


def test_rotate_returns_new_fingerprint(tmp_path: Path) -> None:
    from brain.kindled_link.identity import KindledIdentity

    app, persona = _app(tmp_path)
    old = KindledIdentity.load_or_create(persona)
    resp = TestClient(app).post("/kindled-link/identity/rotate", headers=_AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "new_key_id" in data
    assert "fingerprint_phrase" in data
    assert data["new_key_id"] != old.key_id


def test_rotate_changes_key_on_disk(tmp_path: Path) -> None:
    from brain.kindled_link.identity import KindledIdentity

    app, persona = _app(tmp_path)
    old = KindledIdentity.load_or_create(persona)
    TestClient(app).post("/kindled-link/identity/rotate", headers=_AUTH)
    reloaded = KindledIdentity.load_or_create(persona)
    assert reloaded.key_id != old.key_id


def test_rotate_queues_notice_for_paired_peer(tmp_path: Path) -> None:
    import json

    app, persona = _app(tmp_path)
    _seed_peer(persona)
    TestClient(app).post("/kindled-link/identity/rotate", headers=_AUTH)
    store = KindledLinkStore(kindled_db_path(persona))
    try:
        notices = store.pop_pending_rotation_notices()
    finally:
        store.close()
    assert len(notices) == 1
    assert notices[0]["peer_id"] == "kid_peer"
    env = json.loads(notices[0]["envelope_json"])
    assert env["message_type"] == "key_rotation_notice"


def test_rotate_no_peers_still_succeeds(tmp_path: Path) -> None:
    from brain.kindled_link.identity import KindledIdentity

    app, persona = _app(tmp_path)
    KindledIdentity.load_or_create(persona)
    resp = TestClient(app).post("/kindled-link/identity/rotate", headers=_AUTH)
    assert resp.status_code == 200
